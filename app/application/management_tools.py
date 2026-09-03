"""Shared implementations of the platform management tools.

The platform's CRUD/query operations over workflows, agent definitions and data
sources are exposed on two surfaces: as chat tools of the internal agent
(``app.infrastructure.orchestration.default_workflow``) and as MCP tools at
``/mcp/management`` (``app.api.mcp.management_server``).  Both call the
functions below, so there is exactly one implementation per operation and the
two surfaces cannot drift apart.

Every function returns the human/LLM-readable string the chat agent has always
returned — the return values are part of the tool contract and are asserted by
tests.  ``ask_user`` deliberately does *not* live here: it pauses the calling
graph via ``interrupt()`` and only makes sense inside the chat agent.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.domain.models.graph_run import GraphRun
from app.infrastructure.auth.authorization import Permission, missing_permission
from app.infrastructure.orchestration.yaml_graph import stream_graph_to_pause

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.infrastructure.config.graph_loader import YamlGraphRegistry
    from app.infrastructure.persistence.agent_backend import AgentDefinitionBackend
    from app.infrastructure.persistence.approval_backend import ApprovalCaseBackend
    from app.infrastructure.persistence.data_artifact_backend import DataArtifactBackend
    from app.infrastructure.persistence.data_source_backend import DataSourceDefinitionBackend
    from app.infrastructure.persistence.event_backend import EventDefinitionBackend
    from app.infrastructure.persistence.mongo import MongoGraphRunRepository
    from app.infrastructure.persistence.script_backend import ScriptDefinitionBackend
    from app.infrastructure.persistence.workflow_backend import WorkflowDefinitionBackend

logger = logging.getLogger(__name__)


# asyncio keeps only a weak reference to a running task, so a fire-and-forget
# continuation can be garbage-collected mid-flight and its exception swallowed.
# Hold a strong reference until it finishes and log any failure.  Both surfaces
# (the chat agent and /mcp/management) share this set — it is process-global on
# purpose, and nothing keys off its identity.
_background_tasks: set[asyncio.Task] = set()


def _refusal(required: Permission) -> str:
    return (
        f"Not permitted: this operation requires the '{required.value}' permission "
        "and the calling identity does not hold a role that grants it."
    )


def requires(permission: Permission):
    """Gate a shared core on *permission*.

    The REST API derives its requirement from the HTTP method, which cannot work
    here: every tool below is reached by one ``POST`` — to ``/mcp/management``, or
    to the chat endpoint that hands the same operations to an LLM. Without a gate
    of their own, holding WRITE (enough to POST) would be enough to delete, and
    holding only READ would still be enough for everything, because the
    management-MCP wrapper checks the ACCESS tier alone.

    The decorator is deliberately on the shared core rather than on either
    surface's wrapper: that is the whole reason this module exists, and a gate
    attached to one surface would simply be bypassed through the other.

    Refusals are returned as text, not raised. These are tool bodies whose return
    value is what the model sees, and every other failure here (unknown id,
    invalid JSON) is reported the same way.
    """
    def decorator(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_gate(*args, **kwargs):
                if missing_permission(permission):
                    logger.warning(
                        "tool %s refused: caller lacks %s", fn.__name__, permission.value
                    )
                    return _refusal(permission)
                return await fn(*args, **kwargs)
            async_gate.required_permission = permission
            return async_gate

        @functools.wraps(fn)
        def gate(*args, **kwargs):
            if missing_permission(permission):
                logger.warning(
                    "tool %s refused: caller lacks %s", fn.__name__, permission.value
                )
                return _refusal(permission)
            return fn(*args, **kwargs)
        gate.required_permission = permission
        return gate

    return decorator


def _spawn_background(coro, label: str) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _on_done(finished: asyncio.Task) -> None:
        _background_tasks.discard(finished)
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is not None:
            logger.error("%s failed: %s", label, exc, exc_info=exc)

    task.add_done_callback(_on_done)


@dataclass(frozen=True)
class ManagementDeps:
    """Everything the management tool cores need, and nothing more."""

    registry: "YamlGraphRegistry"
    run_repository: "MongoGraphRunRepository"
    workflow_backend: "WorkflowDefinitionBackend | None" = None
    agent_backend: "AgentDefinitionBackend | None" = None
    data_source_backend: "DataSourceDefinitionBackend | None" = None
    event_backend: "EventDefinitionBackend | None" = None
    script_backend: "ScriptDefinitionBackend | None" = None
    # Run download manifests written by `data` steps. None means the tools that
    # read them say so rather than pretending a run exported nothing.
    data_artifact_backend: "DataArtifactBackend | None" = None
    # Open/decided approval cases, for the read side of the approval queue.
    # None means the tools that read it say so rather than reporting an empty
    # queue, which would read as "nothing is waiting".
    approval_backend: "ApprovalCaseBackend | None" = None
    refresh_runner: "Callable[[str], Awaitable[None]] | None" = None
    refresh_datasources: "Callable[[], Awaitable[None]] | None" = None
    # PubSubSubscriberManager when Pub/Sub triggers are enabled, else None.
    pubsub_subscriber: Any = None


def deps_from_container(
    container: Any,
    refresh_datasources: "Callable[[], Awaitable[None]] | None" = None,
) -> ManagementDeps:
    """Build ``ManagementDeps`` from an ``ApplicationContainer``.

    The data source MCP refresher is not a container attribute (it lives in
    ``app.api.app._make_datasources_refresher``), so it is passed in.
    """
    return ManagementDeps(
        registry=container.yaml_graph_registry,
        run_repository=container.run_repository,
        workflow_backend=getattr(container, "workflow_backend", None),
        agent_backend=getattr(container, "agent_backend", None),
        data_source_backend=getattr(container, "data_source_backend", None),
        event_backend=getattr(container, "event_backend", None),
        script_backend=getattr(container, "script_backend", None),
        data_artifact_backend=getattr(container, "data_artifact_backend", None),
        approval_backend=getattr(container, "approval_backend", None),
        refresh_runner=getattr(container, "refresh_runner", None),
        refresh_datasources=refresh_datasources,
        pubsub_subscriber=getattr(container, "pubsub_subscriber", None),
    )


# ---------------------------------------------------------------------------
# Resolvers (private helpers over deps)
# ---------------------------------------------------------------------------

async def _unregistered_workflow_ids(deps: ManagementDeps) -> list[str]:
    """Ids that are stored but absent from the registry.

    A workflow whose graph fails to build is logged and skipped by the loader,
    so it never enters the registry — while its definition stays in the backing
    store, untouched. Best-effort: a store that cannot be listed simply yields
    nothing.
    """
    if deps.workflow_backend is None:
        return []
    try:
        stored = await deps.workflow_backend.list()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("could not list stored workflows: %s", exc)
        return []
    registered = {d["id"] for d in deps.registry.list_definitions()}
    return sorted(d.id for d in stored if d.id not in registered)


async def _resolve_workflow_id(deps: ManagementDeps, query: str):
    """Returns (resolved_id, None) or (None, error_str).

    Resolution prefers the registry, then falls back to the backing store. The
    fallback is what makes a broken workflow repairable: one whose graph fails
    to build is skipped by the loader, so registry-only resolution rendered it
    simultaneously invisible to `get_workflow` and unreachable by
    `update_workflow` — the two tools needed to diagnose and fix it. The
    definition was never gone, only unresolvable.
    """
    defs = deps.registry.list_definitions()
    for d in defs:
        if d["id"] == query:
            return d["id"], None
    for d in defs:
        if d.get("name", "").lower() == query.lower():
            return d["id"], None
    matches = [d for d in defs if query.lower() in d["id"].lower() or query.lower() in d.get("name", "").lower()]
    if len(matches) == 1:
        return matches[0]["id"], None
    if matches:
        cands = ", ".join(f"{d['id']} ({d.get('name', d['id'])})" for d in matches)
        return None, f"Ambiguous — multiple matches: {cands}"

    # Registry miss: the store may still hold it (build failure, or a
    # definition saved since the last refresh). Exact id only — fuzzy matching
    # against unbuildable workflows would be guessing.
    if deps.workflow_backend is not None:
        try:
            stored = await deps.workflow_backend.get(query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("workflow store lookup for %r failed: %s", query, exc)
            stored = None
        if stored is not None:
            logger.info(
                "workflow '%s' resolved from the store, not the registry — "
                "its graph most likely failed to build",
                query,
            )
            return stored.id, None

    available = ", ".join(f"{d['id']} ({d.get('name', d['id'])})" for d in defs) or "none"
    message = f"Workflow '{query}' not found. Available: {available}"
    broken = await _unregistered_workflow_ids(deps)
    if broken:
        message += (
            ". Stored but not registered (their graphs failed to build): "
            + ", ".join(broken)
        )
    return None, message


async def _resolve_agent_id(deps: ManagementDeps, query: str):
    """Returns (resolved_id, None) or (None, error_str)."""
    if deps.agent_backend is None:
        return None, "agent_backend not configured"
    agents = await deps.agent_backend.list()
    # exact id match
    for a in agents:
        if a.id == query:
            return a.id, None
    # exact name match (case-insensitive)
    for a in agents:
        if a.name.lower() == query.lower():
            return a.id, None
    # substring match
    matches = [a for a in agents if query.lower() in a.id.lower() or query.lower() in a.name.lower()]
    if len(matches) == 1:
        return matches[0].id, None
    if matches:
        cands = ", ".join(f"{a.id} ({a.name})" for a in matches)
        return None, f"Ambiguous — multiple matches: {cands}"
    return None, f"No agent found matching '{query}'. Available: {', '.join(f'{a.id} ({a.name})' for a in agents)}"


async def _resolve_datasource_id(deps: ManagementDeps, query: str):
    """Returns (resolved_id, None) or (None, error_str)."""
    if deps.data_source_backend is None:
        return None, "data_source_backend not configured"
    sources = await deps.data_source_backend.list()
    for s in sources:
        if s.id == query:
            return s.id, None
    for s in sources:
        if (s.name or "").lower() == query.lower():
            return s.id, None
    matches = [
        s for s in sources
        if query.lower() in s.id.lower() or query.lower() in (s.name or "").lower()
    ]
    if len(matches) == 1:
        return matches[0].id, None
    if matches:
        cands = ", ".join(f"{s.id} ({s.name})" for s in matches)
        return None, f"Ambiguous — multiple matches: {cands}"
    available = ", ".join(f"{s.id} ({s.name})" for s in sources) or "none"
    return None, f"Data source '{query}' not found. Available: {available}"


async def _resolve_event_id(deps: ManagementDeps, query: str):
    """Returns (resolved_id, None) or (None, error_str)."""
    if deps.event_backend is None:
        return None, "event_backend not configured"
    events = await deps.event_backend.list()
    for e in events:
        if e.id == query:
            return e.id, None
    for e in events:
        if (e.name or "").lower() == query.lower():
            return e.id, None
    matches = [
        e for e in events
        if query.lower() in e.id.lower() or query.lower() in (e.name or "").lower()
    ]
    if len(matches) == 1:
        return matches[0].id, None
    if matches:
        cands = ", ".join(f"{e.id} ({e.name})" for e in matches)
        return None, f"Ambiguous — multiple matches: {cands}"
    available = ", ".join(f"{e.id} ({e.name})" for e in events) or "none"
    return None, f"Event '{query}' not found. Available: {available}"


async def _resolve_script_id(deps: ManagementDeps, query: str):
    """Accept a script id or its display name, like the other resolvers."""
    if deps.script_backend is None:
        return None, "Script library not configured."
    found = await deps.script_backend.get(query)
    if found is not None:
        return found.id, None
    found = await deps.script_backend.get_by_name(query)
    if found is not None:
        return found.id, None
    return None, f"Script '{query}' not found."


async def _publish_datasources(deps: ManagementDeps) -> None:
    if deps.refresh_datasources is not None:
        try:
            await deps.refresh_datasources()
        except Exception:
            logger.exception("chat_agent: datasource tool refresh failed")


# ---------------------------------------------------------------------------
# Workflow / run tools
# ---------------------------------------------------------------------------

@requires(Permission.READ)
def list_workflows(deps: ManagementDeps) -> str:
    defs = deps.registry.list_definitions()
    if not defs:
        return "No workflows are currently configured."
    # DISABLED is called out inline so a workflow that refuses to start is
    # identifiable from the listing, without fetching each definition.
    lines = [
        f"- **{d['id']}** ({d.get('name', d['id'])})"
        + ("" if d.get("enabled", True) else " [DISABLED]")
        + f": {(d.get('description') or '').strip()}"
        for d in defs
    ]
    return "\n".join(lines)


@requires(Permission.WRITE)
async def run_workflow(
    deps: ManagementDeps,
    workflow_id: str,
    request: str,
    stream_fn: Any | None = None,
) -> str:
    """Start a workflow run and stream it to its first pause in the background.

    ``stream_fn`` exists only as a seam: the chat agent passes its own
    module-level ``stream_graph_to_pause`` binding so that tests patching
    ``default_workflow.stream_graph_to_pause`` still intercept the call.
    """
    stream = stream_fn or stream_graph_to_pause

    resolved, err = await _resolve_workflow_id(deps, workflow_id)
    if err:
        return err
    workflow_id = resolved
    runner = deps.registry.get(workflow_id)
    if runner is None:
        return f"Workflow '{workflow_id}' not found."

    from app.application.run_control import WorkflowDisabledError, ensure_workflow_enabled
    try:
        ensure_workflow_enabled(runner)
    except WorkflowDisabledError as exc:
        return exc.detail

    run_id = str(uuid4())
    child_run = GraphRun(
        id=run_id,
        graph_id=workflow_id,
        user_request=request,
        status="running",
        step_statuses={s["id"]: "pending" for s in runner.steps},
    )
    await deps.run_repository.create(child_run)
    _spawn_background(
        stream(runner, child_run, deps.run_repository, {"request": request}),
        f"run {run_id} of workflow {workflow_id}",
    )
    logger.info("chat_agent: spawned '%s' as run %s", workflow_id, run_id)
    return json.dumps({
        "__event__": "workflow_started",
        "run_id": run_id,
        "workflow_id": workflow_id,
        "workflow_name": runner.name,
    })


@requires(Permission.READ)
async def list_runs(
    deps: ManagementDeps, workflow_id: str | None = None, limit: int = 10
) -> str:
    runs = await deps.run_repository.list_recent(
        limit=min(limit, 20),
        workflow_id=workflow_id,
    )

    if not runs:
        return "No runs found."
    lines = [
        f"- **{r.id}** ({r.graph_id}) — status: {r.status}"
        + (f", started: {r.created_at}" if getattr(r, "created_at", None) else "")
        for r in runs
    ]
    return "\n".join(lines)


# Caps for get_run's rendering: enough to see what a run produced, bounded so a
# run carrying a large fetch cannot flood the caller's context.
_GET_RUN_MAX_KEYS = 12
_GET_RUN_MAX_CHARS = 600


def _compact(value: Any) -> str:
    """JSON-encode *value* on one line, truncated, with its size noted."""
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001 — rendering must never break the tool
        text = str(value)
    if len(text) <= _GET_RUN_MAX_CHARS:
        return text
    return f"{text[:_GET_RUN_MAX_CHARS]}… [{len(text)} chars total]"


@requires(Permission.READ)
async def get_run(deps: ManagementDeps, run_id: str) -> str:
    run = await deps.run_repository.get(run_id)
    if run is None:
        return f"Run '{run_id}' not found."
    parts = [
        f"Run: {run.id}",
        f"Workflow: {run.graph_id}",
        f"Status: {run.status}",
    ]
    if run.step_statuses:
        parts.append("Steps:")
        for step_id, status in run.step_statuses.items():
            parts.append(f"  - {step_id}: {status}")
    if run.state:
        # Render every non-internal state value, compactly. The previous version
        # only surfaced `.error` and `.status` sub-keys, so a step whose output was
        # any other shape showed as a bare key name — which made this tool unable
        # to answer the question it exists for ("what did the run actually
        # produce?"). Values are JSON-encoded and truncated instead, so the shape
        # is always visible and the response still cannot run away.
        output_keys = [k for k in run.state if not k.startswith("_")]
        if output_keys:
            parts.append("State keys: " + ", ".join(output_keys))
            for k in output_keys[:_GET_RUN_MAX_KEYS]:
                parts.append(f"  {k}: {_compact(run.state[k])}")
            if len(output_keys) > _GET_RUN_MAX_KEYS:
                parts.append(
                    f"  … {len(output_keys) - _GET_RUN_MAX_KEYS} more key(s) not shown"
                )
    # A run carries no `error` attribute; failures are recorded in state by the
    # step that caught them (`error`) and by the fail sentinel
    # (`__failed_step__`). Reading run.error raised AttributeError here, which
    # made get_run fail for every run, successful ones included.
    state = run.state if isinstance(run.state, dict) else {}
    failed_step = state.get("__failed_step__")
    if failed_step:
        parts.append(f"Failed step: {failed_step}")
    err = state.get("error")
    if err:
        parts.append(f"Error: {str(err)[:500]}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Run data downloads
# ---------------------------------------------------------------------------
#
# Two origins, one mechanism. A `data` step's curated exports are listed by
# default; the data source results already in the run's state are described in
# exactly the same shape behind an opt-in, and both are fetched from the same
# URL. That is deliberate: the data node is how data leaves the system, so a
# data source result rides the same rails rather than getting a reader of its
# own.
#
# A datasource entry is resolved by looking inside the state of the run named
# in the call, never by opening a stream id the caller supplied -- a tool that
# opened any `ds_*` id on request would be a cross-run read primitive. Doing it
# through the run makes a result exactly as accessible as the run that produced
# it, which is the rule the whole surface already follows.

def _download_path(artifact: Any) -> str:
    """Where an agent fetches the artifact's bytes.

    The REST manifest returns this path relative to the API root, which is what
    the frontend joins against its own base URL.  A tool caller has no such
    base, so the configured API prefix is included here — the point of the tool
    is that the URL it hands back is one that works.
    """
    from app.core.config import get_settings

    prefix = (getattr(get_settings(), "api_prefix", "") or "").rstrip("/")
    return f"{prefix}/runs/{artifact.run_id}/data/{artifact.id}"


def _render_artifact(artifact: Any) -> str:
    """One manifest entry, on one line.

    ``truncated`` is spelled out rather than shown as a flag: the whole risk of
    this feature is somebody treating a prefix as the complete answer, and an
    LLM reading `truncated: true` in a list of fields is exactly the reader
    most likely to skip past it.
    """
    source = f"{artifact.source_id}.{artifact.operation}".strip(".")
    what = (
        f"{artifact.items} item(s)" if artifact.shape == "list" else "1 document"
    )
    warning = (
        " — INCOMPLETE: this is a truncated prefix of the data, not the whole "
        "answer" if artifact.truncated else ""
    )
    return (
        f"- **{artifact.id}** {artifact.name} [{artifact.origin}] "
        f"({artifact.format}, {what}, {artifact.bytes} bytes) from step "
        f"'{artifact.step_id}'"
        + (f" via {source}" if source else "")
        + f" — download: {_download_path(artifact)}"
        + f" — expires: {artifact.expires_at.isoformat()}"
        + warning
    )


def _datasource_ttl() -> float:
    """How long an unpinned data source result is good for.

    The ordinary spill window, because a datasource entry is *not* pinned. A
    manifest that quoted the artifact TTL for it would be promising a week of
    availability for bytes the sweep takes in six hours.
    """
    from app.core.config import get_settings

    return float(getattr(get_settings(), "stream_ttl_seconds", 0.0) or 0.0)


@requires(Permission.READ)
async def list_run_data(
    deps: ManagementDeps, run_id: str, include_datasource: bool = False
) -> str:
    from app.application.data_artifacts import list_run_artifacts

    run = await deps.run_repository.get(run_id)
    if run is None:
        return f"Run '{run_id}' not found."
    rows = await list_run_artifacts(
        deps.data_artifact_backend,
        run,
        include_datasource=include_datasource,
        datasource_ttl_seconds=_datasource_ttl(),
    )
    if not rows:
        hint = (
            "" if include_datasource else
            " Pass include_datasource=true to list the raw data source results "
            "this run fetched as well."
        )
        if deps.data_artifact_backend is None and not include_datasource:
            return (
                "Curated exports are unavailable: this backend has no data "
                "artifact store configured." + hint
            )
        return (
            f"Run '{run_id}' has no downloadable data. Either it ran no `data` "
            f"step, or every selection that step declared resolved to nothing."
            + hint
        )
    return "\n".join(_render_artifact(a) for a in rows)


@requires(Permission.READ)
async def get_run_data_artifact(
    deps: ManagementDeps, run_id: str, artifact_id: str
) -> str:
    from app.application.data_artifacts import find_run_artifact

    run = await deps.run_repository.get(run_id)
    if run is None:
        return f"Run '{run_id}' not found."
    artifact = await find_run_artifact(
        deps.data_artifact_backend,
        run,
        artifact_id,
        datasource_ttl_seconds=_datasource_ttl(),
    )
    if artifact is None:
        return f"No data artifact '{artifact_id}' for run '{run_id}'."
    lines = [
        f"Artifact: {artifact.id}",
        f"Run: {artifact.run_id}",
        f"Step: {artifact.step_id}",
        f"Name: {artifact.name}",
        f"Origin: {artifact.origin}",
        f"Format: {artifact.format} (filename {artifact.filename})",
        f"Shape: {artifact.shape}",
        f"Items: {artifact.items}",
        f"Stored bytes: {artifact.bytes}",
        f"Truncated: {artifact.truncated}",
        f"Created: {artifact.created_at.isoformat()}",
        f"Expires: {artifact.expires_at.isoformat()}",
        f"Download: {_download_path(artifact)}",
    ]
    source = f"{artifact.source_id}.{artifact.operation}".strip(".")
    if source:
        lines.append(f"Source: {source}")
    if artifact.origin == "datasource":
        lines.append(
            "Note: this is a raw data source result, not a curated export. It "
            "is not pinned, so it is swept on the ordinary stream TTL at the "
            "expiry above and the download then returns 410."
        )
    if artifact.truncated:
        lines.append(
            "WARNING: this artifact is a truncated prefix of the data, not the "
            "whole answer. Do not present a download of it as complete."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Approval reads
# ---------------------------------------------------------------------------
#
# `approve_run` and `reject_run` have existed for a while; reading the case they
# decide has not. That was survivable while every case was "delete N rows", and
# it is not any more: a case now carries `change_kind`, a `details` map, and a
# cell-level `affected_sample` -- and for a generated (tier-2) write, `details`
# says whether a language model wrote the code that produced those values. An
# agent holding WRITE could approve exactly that, blind. These two tools are
# what make an informed decision possible, so they sit at READ alongside the
# GET routes in `app.api.routes.approvals` that answer the same questions.

# Cases one call will list. The queue is a human work list; past a couple of
# dozen the answer to "what is waiting" is "too much", not a longer page.
_APPROVAL_LIST_MAX = 50
# Characters of any one rendered value. A `params` map can carry a whole
# request body, and a tool response is somebody's context window.
_APPROVAL_VALUE_CHARS = 400
# Sample entries shown in full. The sample exists so a decider can recognise
# what is about to change, which the first several rows do.
_APPROVAL_SAMPLE_ITEMS = 10


def _approval_line(case: Any) -> str:
    verdict = ""
    if case.meta_llm is not None:
        confidence = (
            f", confidence {case.meta_llm.confidence:.2f}"
            if case.meta_llm.confidence is not None else ""
        )
        verdict = (
            f" — meta-LLM: {case.meta_llm.decision}{confidence}"
            + (" (autonomous)" if case.meta_llm.autonomous else "")
        )
    return (
        f"- **{case.id}** [{case.status}] {case.change_kind} via "
        f"{case.datasource_name or case.datasource_id}.{case.operation} "
        f"[{case.method}] — {case.affected_rows} row(s)"
        + (f", workflow '{case.workflow_name or case.workflow_id}'"
           if (case.workflow_name or case.workflow_id) else "")
        + (f", run {case.run_id}" if case.run_id else "")
        + f", opened {case.created_at.isoformat()}"
        + verdict
    )


@requires(Permission.READ)
async def list_pending_approvals(deps: ManagementDeps, limit: int = 20) -> str:
    if deps.approval_backend is None:
        return (
            "Approvals are unavailable: this backend has no approval store "
            "configured, so destructive operations run ungated."
        )
    cases = await deps.approval_backend.list(
        status="pending", limit=max(1, min(limit, _APPROVAL_LIST_MAX))
    )
    if not cases:
        return "No approvals are waiting."
    return "\n".join(
        [f"{len(cases)} approval(s) waiting:"]
        + [_approval_line(c) for c in cases]
        + ["Call get_approval with an id before approving or rejecting it."]
    )


@requires(Permission.READ)
async def get_approval(deps: ManagementDeps, case_id: str) -> str:
    if deps.approval_backend is None:
        return (
            "Approvals are unavailable: this backend has no approval store "
            "configured, so destructive operations run ungated."
        )
    case = await deps.approval_backend.get(case_id)
    if case is None:
        return f"Approval case '{case_id}' not found."

    parts = [
        f"Case: {case.id}",
        f"Status: {case.status}",
        f"Change kind: {case.change_kind}",
        f"Data source: {case.datasource_name or case.datasource_id}",
        f"Operation: {case.operation} [{case.method}]",
        f"Affected rows: {case.affected_rows}",
        f"Surface: {case.surface}",
        f"Opened: {case.created_at.isoformat()}",
    ]
    if case.workflow_id or case.workflow_name:
        parts.append(f"Workflow: {case.workflow_name or case.workflow_id}")
    if case.run_id:
        parts.append(f"Run: {case.run_id}" + (f" step '{case.step_id}'" if case.step_id else ""))
    if case.endpoint:
        parts.append(f"Endpoint: {case.endpoint}")

    if case.details:
        # The `details` map is where a tier-2 write says that generated code
        # produced these values. Rendered before the sample, because it changes
        # how much the sample is worth trusting.
        parts.append("Details:")
        for key, value in case.details.items():
            parts.append(f"  {key}: {_clip(str(value))}")
    if case.targets:
        parts.append("Targets: " + _clip(", ".join(str(t) for t in case.targets)))
    if case.params:
        parts.append("Params:")
        for key, value in case.params.items():
            parts.append(f"  {key}: {_clip(_compact(value))}")
    if case.affected_sample:
        shown = case.affected_sample[:_APPROVAL_SAMPLE_ITEMS]
        parts.append(f"Affected sample ({len(shown)} of {len(case.affected_sample)}):")
        for entry in shown:
            parts.append(f"  {_clip(entry if isinstance(entry, str) else _compact(entry))}")
    if case.meta_llm is not None:
        verdict = case.meta_llm
        parts.append(
            f"Meta-LLM verdict: {verdict.decision}"
            + (f" (confidence {verdict.confidence:.2f})"
               if verdict.confidence is not None else "")
            + (" — AUTONOMOUS: it decided on its own and the run is holding for "
               "the veto window" if verdict.autonomous else "")
        )
        if verdict.reason:
            parts.append(f"  reason: {_clip(verdict.reason)}")
        if verdict.model:
            parts.append(f"  model: {verdict.model}, history seen: {verdict.history_size}")
    if case.veto_deadline is not None:
        parts.append(f"Veto deadline: {case.veto_deadline.isoformat()}")
    if case.decided_at is not None:
        parts.append(
            f"Decided: {case.decided_at.isoformat()} by "
            f"{case.decided_by_name or case.decision_source or 'unknown'}"
            + (f" — {_clip(case.reason)}" if case.reason else "")
        )
    return "\n".join(parts)


def _clip(text: str) -> str:
    if len(text) <= _APPROVAL_VALUE_CHARS:
        return text
    return f"{text[:_APPROVAL_VALUE_CHARS]}… [{len(text)} chars total]"


def _sandbox_denial(steps: Any) -> str | None:
    """Refusal message when the calling principal may not disable the sandbox.

    Returns a string because these functions are MCP tool bodies: their return
    value is what the model sees, so a refusal has to be readable text rather than
    an exception. `None` means allowed.

    The permissions come from the ambient principal set by the authenticating ASGI
    wrapper — FastMCP hands tools no request object. An unauthenticated caller
    resolves to no permissions and is therefore refused, which is the safe default.
    """
    from app.infrastructure.auth.authorization import get_current_permissions
    from app.infrastructure.auth.sandbox_guard import (
        SandboxNotPermittedError,
        assert_sandbox_allowed,
    )

    try:
        assert_sandbox_allowed(steps, get_current_permissions())
    except SandboxNotPermittedError as exc:
        return str(exc)
    return None


def _implicit_edge_note(edges: list[tuple[str, str]]) -> str:
    """Name the edges that came from step order rather than from a `next`.

    A step that declares no destination falls through to the next one in the
    array, so a step list can mean more than it says. Writing those edges out is
    normalisation's job; saying so is this note's, because an author who did not
    intend one has no other way to notice — that silence is what let a workflow
    wire its own tail into a branch and post the same digest eleven times.
    """
    if not edges:
        return ""
    return (
        " Note: step order implied "
        + ", ".join(f"{src} → {dst}" for src, dst in edges)
        + ". These are now explicit `next` values; set `next: END` on a step that"
        " should terminate instead."
    )


def _captured_note(captured: list[str]) -> str:
    """Tell the model which python bodies became library scripts.

    Silent rewriting would make the agent re-send inline code it thinks is still
    inline; naming the ids lets it reference them directly next time.
    """
    if not captured:
        return ""
    return (
        " Inline python bodies were saved to the script library as "
        + ", ".join(f"'{sid}'" for sid in captured)
        + " and the steps now reference them by script_id."
    )


@requires(Permission.READ)
async def get_workflow(
    deps: ManagementDeps, workflow_id: str, include_steps: bool = True
) -> str:
    """Read one workflow in full: its flags and its steps.

    list_workflows returns only id, name and description, so until this existed
    the steps and the flags were unreadable from here -- while update_workflow
    replaces the *entire* step list. That combination is the dangerous one: an
    update had to be composed blind, and a caller who did not already know the
    current steps could only overwrite them. Read this first.

    It also surfaces `use_storage`, which is otherwise invisible: a `storage`
    step in a workflow whose flag is off fails at run time, and there was no way
    to check the flag beforehand.

    Args:
        workflow_id: The workflow id or its display name.
        include_steps: False for flags only, when the step list is long.
    """
    if deps.workflow_backend is None:
        return "Workflow backend not configured."
    resolved, err = await _resolve_workflow_id(deps, workflow_id)
    if err:
        return err
    defn = await deps.workflow_backend.get(resolved)
    if defn is None:
        return f"Workflow '{workflow_id}' not found."

    steps = defn.steps or []
    lines = [
        f"Workflow: {defn.id}",
        f"Name: {defn.name or '(unnamed)'}",
        f"Description: {defn.description or '(none)'}",
        f"Enabled: {defn.enabled}",
        f"Use storage: {defn.use_storage}",
        f"Use meta LLM: {defn.use_meta_llm}",
        f"Read-only: {defn.readonly}",
        f"Steps: {len(steps)}"
        + (" — " + ", ".join(f"{st.get('id')}({st.get('type')})" for st in steps
                             if isinstance(st, dict)) if steps else ""),
    ]
    if include_steps and steps:
        # The whole point is to be able to feed this straight back into
        # update_workflow, so the steps are returned complete rather than
        # truncated -- pass include_steps=False when only the flags are wanted.
        lines.append("steps_json:")
        lines.append(json.dumps(steps, ensure_ascii=False))
    return "\n".join(lines)


@requires(Permission.WRITE)
async def create_workflow(
    deps: ManagementDeps, workflow_id: str, name: str, description: str, steps_json: str,
    use_storage: bool = False, enabled: bool = True, use_meta_llm: bool = True,
) -> str:
    if deps.workflow_backend is None:
        return "Workflow creation unavailable: no persistent backend configured."
    try:
        steps = json.loads(steps_json)
    except json.JSONDecodeError as exc:
        return f"Invalid steps_json: {exc}"
    if not isinstance(steps, list):
        return "steps_json must be a JSON array."

    denied = _sandbox_denial(steps)
    if denied:
        return denied

    existing = await deps.workflow_backend.get(workflow_id)
    if existing is not None:
        return f"Workflow '{workflow_id}' already exists. Use update_workflow to modify it."

    from app.application.graph_layout import apply_layout
    from app.application.script_capture import capture_inline_scripts
    from app.application.step_normalization import implicit_edges, normalize_edges
    from app.domain.models.workflow_definition import WorkflowDefinition
    captured = await capture_inline_scripts(workflow_id, steps, deps.script_backend)
    implied = implicit_edges(steps)
    steps = normalize_edges(steps)
    # Nothing coming through this tool carries coordinates, and the canvas's
    # fallback for a step without one ignores the graph -- so the graph is what
    # places them here, once, at the point the definition is written.
    defn = WorkflowDefinition(
        id=workflow_id, name=name, description=description, steps=steps,
        use_storage=use_storage, enabled=enabled, use_meta_llm=use_meta_llm,
        ui=apply_layout(None, steps),
    )
    await deps.workflow_backend.create(defn)
    if deps.refresh_runner is not None:
        await deps.refresh_runner(workflow_id)
    # Say so when it is created disabled: a workflow that exists but will not
    # start is easy to forget about, and silence here reads as "it is live".
    state_note = "" if enabled else " It is disabled and will not start until enabled."
    storage_note = " Storage is enabled." if use_storage else ""
    return (
        f"Workflow '{workflow_id}' created with {len(steps)} step(s)."
        + state_note + storage_note + _captured_note(captured)
        + _implicit_edge_note(implied)
    )


@requires(Permission.WRITE)
async def update_workflow(
    deps: ManagementDeps,
    workflow_id: str,
    name: str | None = None,
    description: str | None = None,
    steps_json: str | None = None,
    enabled: bool | None = None,
    use_storage: bool | None = None,
    use_meta_llm: bool | None = None,
    relayout: bool = False,
) -> str:
    if deps.workflow_backend is None:
        return "Workflow updates unavailable: no persistent backend configured."

    resolved, err = await _resolve_workflow_id(deps, workflow_id)
    if err:
        return err
    workflow_id = resolved
    defn = await deps.workflow_backend.get(workflow_id)
    if defn is None:
        return f"Workflow '{workflow_id}' not found."
    if defn.readonly:
        return f"Workflow '{workflow_id}' is read-only and cannot be modified."

    captured: list[str] = []
    implied: list[tuple[str, str]] = []
    if name is not None:
        defn.name = name
    if description is not None:
        defn.description = description
    if enabled is not None:
        defn.enabled = enabled
    if use_storage is not None:
        defn.use_storage = use_storage
    if use_meta_llm is not None:
        defn.use_meta_llm = use_meta_llm
    if steps_json is not None:
        try:
            steps = json.loads(steps_json)
        except json.JSONDecodeError as exc:
            return f"Invalid steps_json: {exc}"
        if not isinstance(steps, list):
            return "steps_json must be a JSON array."
        denied = _sandbox_denial(steps)
        if denied:
            return denied
        from app.application.graph_layout import apply_layout
        from app.application.script_capture import capture_inline_scripts
        from app.application.step_normalization import implicit_edges, normalize_edges
        captured = await capture_inline_scripts(workflow_id, steps, deps.script_backend)
        implied = implicit_edges(steps)
        defn.steps = normalize_edges(steps)
        # A position already stored was put there by whoever arranged the canvas,
        # so only the new steps are placed unless a relayout was asked for.
        defn.ui = apply_layout(defn.ui, defn.steps, relayout=relayout)
    elif relayout:
        from app.application.graph_layout import apply_layout
        defn.ui = apply_layout(defn.ui, defn.steps, relayout=True)

    await deps.workflow_backend.update(workflow_id, defn)
    if deps.refresh_runner is not None:
        await deps.refresh_runner(workflow_id)
    state_note = "" if defn.enabled else " It is disabled and will not start."
    return (
        f"Workflow '{workflow_id}' updated." + state_note + _captured_note(captured)
        + _implicit_edge_note(implied)
    )


@requires(Permission.DELETE)
async def delete_workflow(deps: ManagementDeps, workflow_id: str) -> str:
    if deps.workflow_backend is None:
        return "Workflow deletion unavailable: no persistent backend configured."

    resolved, err = await _resolve_workflow_id(deps, workflow_id)
    if err:
        return err
    workflow_id = resolved
    defn = await deps.workflow_backend.get(workflow_id)
    if defn is None:
        return f"Workflow '{workflow_id}' not found."
    if defn.readonly:
        return f"Workflow '{workflow_id}' is read-only and cannot be deleted."

    await deps.workflow_backend.delete(workflow_id)
    deps.registry.remove(workflow_id)
    # Also drops the deleted workflow's cron jobs and Pub/Sub subscriptions —
    # refresh_runner finds no definition and only unregisters.
    if deps.refresh_runner is not None:
        await deps.refresh_runner(workflow_id)
    return f"Workflow '{workflow_id}' deleted."


# ---------------------------------------------------------------------------
# Agent definition tools
# ---------------------------------------------------------------------------

@requires(Permission.READ)
async def list_agents(deps: ManagementDeps) -> str:
    if deps.agent_backend is None:
        return "Agent backend not configured."
    agents = await deps.agent_backend.list()
    if not agents:
        return "No agents found."
    lines = [f"- **{a.id}** ({a.name}): {a.description or '(no description)'}" for a in agents]
    return "\n".join(lines)


@requires(Permission.READ)
async def get_agent(deps: ManagementDeps, agent_id: str) -> str:
    if deps.agent_backend is None:
        return "Agent backend not configured."
    resolved, err = await _resolve_agent_id(deps, agent_id)
    if err:
        return err
    agent = await deps.agent_backend.get(resolved)
    if agent is None:
        return f"Agent '{resolved}' not found."
    import json as _json
    return _json.dumps(agent.model_dump(mode="json"), indent=2, default=str)


@requires(Permission.WRITE)
async def create_agent(
    deps: ManagementDeps,
    agent_id: str,
    name: str,
    description: str = "",
    default_runtime: str = "local",
    agent_input_json: str = "{}",
) -> str:
    if deps.agent_backend is None:
        return "Agent backend not configured."
    import json as _json
    try:
        agent_input = _json.loads(agent_input_json)
        if not isinstance(agent_input, dict):
            return "agent_input_json must be a JSON object."
    except Exception as e:
        return f"Invalid agent_input_json: {e}"
    if default_runtime not in ("local", "docker", "k8s"):
        return f"Invalid default_runtime '{default_runtime}'. Must be one of: local, docker, k8s."
    existing = await deps.agent_backend.get(agent_id)
    if existing is not None:
        return f"Agent '{agent_id}' already exists. Use update_agent to modify it."
    from app.domain.models.agent_definition import AgentDefinition
    new_agent = AgentDefinition(
        id=agent_id,
        name=name,
        description=description,
        default_runtime=default_runtime,
        agent_input=agent_input,
    )
    await deps.agent_backend.create(new_agent)
    return f"Agent '{agent_id}' created."


@requires(Permission.WRITE)
async def update_agent(
    deps: ManagementDeps,
    agent_id: str,
    name: str = None,
    description: str = None,
    default_runtime: str = None,
    agent_input_json: str = None,
) -> str:
    if deps.agent_backend is None:
        return "Agent backend not configured."
    resolved, err = await _resolve_agent_id(deps, agent_id)
    if err:
        return err
    existing = await deps.agent_backend.get(resolved)
    if existing is None:
        return f"Agent '{resolved}' not found."
    import json as _json
    # Partial update — only mutate provided fields
    updated = existing.model_copy()
    if name is not None:
        updated.name = name
    if description is not None:
        updated.description = description
    if default_runtime is not None:
        if default_runtime not in ("local", "docker", "k8s"):
            return f"Invalid default_runtime '{default_runtime}'. Must be one of: local, docker, k8s."
        updated.default_runtime = default_runtime
    if agent_input_json is not None:
        try:
            agent_input = _json.loads(agent_input_json)
            if not isinstance(agent_input, dict):
                return "agent_input_json must be a JSON object."
            updated.agent_input = agent_input
        except Exception as e:
            return f"Invalid agent_input_json: {e}"
    await deps.agent_backend.update(resolved, updated)
    return f"Agent '{resolved}' updated."


@requires(Permission.DELETE)
async def delete_agent(deps: ManagementDeps, agent_id: str) -> str:
    if deps.agent_backend is None:
        return "Agent backend not configured."
    existing = await deps.agent_backend.get(agent_id)
    if existing is None:
        return f"Agent '{agent_id}' not found. Use list_agents to see available agents."
    await deps.agent_backend.delete(agent_id)
    return f"Agent '{agent_id}' deleted."


# ---------------------------------------------------------------------------
# Event tools
#
# An event is a Pub/Sub topic a workflow can be triggered by.  Events used to
# be data sources with kind="pubsub"; they are a resource of their own because
# nothing about base URL, auth or operations applies to them.
# ---------------------------------------------------------------------------

@requires(Permission.READ)
async def list_events(deps: ManagementDeps) -> str:
    """List the events (Pub/Sub topics) workflows can be triggered by."""
    if deps.event_backend is None:
        return "Event backend not configured."
    events = await deps.event_backend.list()
    if not events:
        return "No events found."
    return "\n".join(
        f"- **{e.id}** ({e.name or e.id}): {e.description or '(no description)'} — "
        f"topic: {e.topic or '(unset)'}, "
        f"subscription: {e.subscription or '(created on first use)'}"
        for e in events
    )


@requires(Permission.WRITE)
async def create_event(
    deps: ManagementDeps,
    event_id: str,
    name: str,
    topic: str,
    event_schema_json: str = "",
    subscription: str = "",
    project_id: str = "",
    description: str = "",
) -> str:
    """Register a Pub/Sub topic as a reusable event.

    An event exists so ``pubsub`` trigger steps can point at
    ``event: <event_id>`` instead of repeating topic, schema and subscription
    in every workflow.  Leave *subscription* empty to have one created (and
    saved back here) the first time a workflow subscribes.
    """
    if deps.event_backend is None:
        return "Event creation unavailable: no persistent backend configured."
    if not topic.strip():
        return "An event needs a topic."
    event_schema: dict | None = None
    if event_schema_json:
        try:
            event_schema = json.loads(event_schema_json)
        except json.JSONDecodeError as exc:
            return f"Invalid event_schema_json: {exc}"
        if not isinstance(event_schema, dict):
            return "event_schema_json must be a JSON object."

    existing = await deps.event_backend.get(event_id)
    if existing is not None:
        return f"Event '{event_id}' already exists. Use update_event to modify it."
    clash = await deps.event_backend.get_by_name(name) if name else None
    if clash is not None:
        return (
            f"An event named '{name}' already exists (id '{clash.id}'). "
            "Pick another name, or use update_event to change that one."
        )

    from app.domain.models.event_definition import EventDefinition
    try:
        defn = EventDefinition.model_validate({
            "id": event_id,
            "name": name,
            "description": description,
            "topic": topic.strip(),
            "subscription": subscription.strip(),
            "project_id": project_id.strip(),
            "event_schema": event_schema,
        })
    except Exception as exc:
        return f"Invalid event definition: {exc}"

    await deps.event_backend.create(defn)
    return (
        f"Event '{event_id}' created for topic '{topic}'"
        + (f" using subscription '{subscription}'." if subscription else " (subscription created on first use).")
    )


@requires(Permission.WRITE)
async def update_event(
    deps: ManagementDeps,
    event_id: str,
    name: str | None = None,
    description: str | None = None,
    topic: str | None = None,
    subscription: str | None = None,
    project_id: str | None = None,
    event_schema_json: str | None = None,
) -> str:
    """Change an existing event; omitted fields keep their stored value."""
    if deps.event_backend is None:
        return "Event updates unavailable: no persistent backend configured."
    resolved, err = await _resolve_event_id(deps, event_id)
    if err:
        return err
    existing = await deps.event_backend.get(resolved)
    if existing is None:
        return f"Event '{resolved}' not found."

    payload = existing.model_dump(mode="json")
    if name is not None:
        clash = await deps.event_backend.get_by_name(name)
        if clash is not None and clash.id != resolved:
            return f"An event named '{name}' already exists (id '{clash.id}')."
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    if topic is not None:
        payload["topic"] = topic.strip()
    if subscription is not None:
        payload["subscription"] = subscription.strip()
    if project_id is not None:
        payload["project_id"] = project_id.strip()
    if event_schema_json is not None:
        try:
            event_schema = json.loads(event_schema_json)
        except json.JSONDecodeError as exc:
            return f"Invalid event_schema_json: {exc}"
        if not isinstance(event_schema, dict):
            return "event_schema_json must be a JSON object."
        payload["event_schema"] = event_schema

    if not (payload.get("topic") or "").strip():
        return "An event needs a topic."

    from app.domain.models.event_definition import EventDefinition
    try:
        defn = EventDefinition.model_validate(payload)
    except Exception as exc:
        return f"Invalid event definition: {exc}"

    await deps.event_backend.update(resolved, defn)
    return f"Event '{resolved}' updated."


@requires(Permission.DELETE)
async def delete_event(deps: ManagementDeps, event_id: str) -> str:
    """Delete an event. Workflow steps still pointing at it stop resolving."""
    if deps.event_backend is None:
        return "Event deletion unavailable: no persistent backend configured."
    resolved, err = await _resolve_event_id(deps, event_id)
    if err:
        return err
    existing = await deps.event_backend.get(resolved)
    if existing is None:
        return f"Event '{resolved}' not found."
    await deps.event_backend.delete(resolved)
    return f"Event '{resolved}' deleted."


# ---------------------------------------------------------------------------
# Data source tools
# ---------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Script library
#
# `python` steps reference reusable scripts by `script_id`, and create_workflow
# captures inline bodies into the library automatically -- so the library fills
# up on its own but, until now, could only be read or emptied through the UI.
# Dead scripts from deleted workflows had no way out at all.
# --------------------------------------------------------------------------

@requires(Permission.READ)
async def list_scripts(deps: ManagementDeps) -> str:
    """List the Python scripts in the library that `python` steps can reference.

    Shows each script's size rather than its body: a library holding a few
    thousand-line scripts would otherwise flood the reply.
    """
    if deps.script_backend is None:
        return "Script library not configured."
    scripts = await deps.script_backend.list()
    if not scripts:
        return "No scripts found."
    return "\n".join(
        f"- **{sc.id}** ({sc.name or sc.id}): {sc.description or '(no description)'} — "
        f"{len(sc.code)} chars, {len(sc.code.splitlines())} lines"
        for sc in sorted(scripts, key=lambda x: x.id)
    )


@requires(Permission.READ)
async def get_script(deps: ManagementDeps, script_id: str, include_code: bool = True) -> str:
    """Read one script from the library.

    Args:
        script_id: The script id or its display name.
        include_code: Set False for just the metadata, when the body would be
            too long to be worth returning.
    """
    resolved, err = await _resolve_script_id(deps, script_id)
    if err:
        return err
    script = await deps.script_backend.get(resolved)
    if script is None:
        return f"Script '{script_id}' not found."
    head = (
        f"Script: {script.id}\nName: {script.name or '(unnamed)'}\n"
        f"Description: {script.description or '(none)'}\n"
        f"Size: {len(script.code)} chars, {len(script.code.splitlines())} lines"
    )
    if not include_code:
        return head
    return head + "\n--- code ---\n" + script.code


@requires(Permission.WRITE)
async def create_script(
    deps: ManagementDeps, script_id: str, name: str, code: str, description: str = ""
) -> str:
    """Add a Python script to the library so `python` steps can reference it.

    Args:
        script_id: Unique kebab-case identifier (e.g. "csm-deadline-compute").
        name: Human-readable display name, unique within the library.
        code: The script body. It runs with a `state` dict in scope and must set
            `output`; it cannot import project modules, so it has to be
            self-contained.
        description: What the script does — shown in the library and node picker.
    """
    if deps.script_backend is None:
        return "Script library not configured."
    existing = await deps.script_backend.get(script_id)
    if existing is not None:
        return (
            f"Script '{script_id}' already exists ({len(existing.code)} chars). "
            f"Use update_script to change it."
        )
    from app.domain.models.script_definition import ScriptDefinition
    defn = ScriptDefinition(
        id=script_id, name=name, description=description or None, code=code
    )
    await deps.script_backend.create(defn)
    return (
        f"Script '{script_id}' created ({len(code)} chars, "
        f"{len(code.splitlines())} lines)."
    )


@requires(Permission.WRITE)
async def update_script(
    deps: ManagementDeps,
    script_id: str,
    name: str | None = None,
    code: str | None = None,
    description: str | None = None,
) -> str:
    """Change a script in the library; omitted fields keep their stored value.

    Args:
        script_id: The script id or its display name.
        name: New display name (omit to keep current).
        code: New body (omit to keep current).
        description: New description (omit to keep current).
    """
    resolved, err = await _resolve_script_id(deps, script_id)
    if err:
        return err
    script = await deps.script_backend.get(resolved)
    if script is None:
        return f"Script '{script_id}' not found."
    before = len(script.code)
    if name is not None:
        script.name = name
    if description is not None:
        script.description = description
    if code is not None:
        script.code = code
    await deps.script_backend.update(resolved, script)
    # Report the size change: a script that silently became empty, or shrank by
    # half, is worth noticing at the point of the edit.
    delta = f" ({before} -> {len(script.code)} chars)" if code is not None else ""
    return f"Script '{resolved}' updated{delta}."


@requires(Permission.DELETE)
async def delete_script(deps: ManagementDeps, script_id: str) -> str:
    """Remove a script from the library.

    Refuses while a workflow still references it: deleting one out from under a
    `python` step turns that step into a runtime failure with a confusing
    message, and the reference is cheap to check first.

    Args:
        script_id: The script id or its display name.
    """
    resolved, err = await _resolve_script_id(deps, script_id)
    if err:
        return err
    if deps.workflow_backend is not None:
        users = []
        for wf in await deps.workflow_backend.list():
            for step in wf.steps or []:
                if isinstance(step, dict) and step.get("script_id") == resolved:
                    users.append(f"{wf.id}:{step.get('id')}")
        if users:
            return (
                f"Script '{resolved}' is still referenced by {', '.join(users)}. "
                f"Point those steps elsewhere first, or delete the workflow."
            )
    await deps.script_backend.delete(resolved)
    return f"Script '{resolved}' deleted."


@requires(Permission.READ)
async def get_event(deps: ManagementDeps, event_id: str) -> str:
    """Read one event in full, including its payload schema.

    list_events deliberately omits the schema; without this there was no way to
    see what an event actually validates against.

    Args:
        event_id: The event id or its display name.
    """
    resolved, err = await _resolve_event_id(deps, event_id)
    if err:
        return err
    event = await deps.event_backend.get(resolved)
    if event is None:
        return f"Event '{event_id}' not found."
    lines = [
        f"Event: {event.id}",
        f"Name: {event.name or '(unnamed)'}",
        f"Topic: {event.topic or '(unset)'}",
        f"Subscription: {event.subscription or '(created on first use)'}",
        f"Project: {event.project_id or '(backend default)'}",
        f"Description: {event.description or '(none)'}",
    ]
    if event.event_schema:
        lines.append("Schema: " + json.dumps(event.event_schema, ensure_ascii=False))
    else:
        lines.append("Schema: (none — every message is accepted)")
    return "\n".join(lines)


@requires(Permission.READ)
async def get_datasource(deps: ManagementDeps, source_id: str) -> str:
    """Read one data source in full: auth type, and every operation's shape.

    list_datasources gives only operation names, so the paths, params, templates
    and mappings were previously invisible from here -- which also made
    update_datasource dangerous, since it replaces the whole operation list and
    there was no way to read the current one back first.

    Secrets are never returned: the auth block is reported by type only.

    Args:
        source_id: The data source id or its display name.
    """
    resolved, err = await _resolve_datasource_id(deps, source_id)
    if err:
        return err
    source = await deps.data_source_backend.get(resolved)
    if source is None:
        return f"Data source '{source_id}' not found."
    lines = [
        f"Data source: {source.id}",
        f"Name: {source.name or '(unnamed)'}",
        f"Kind: {source.kind}",
        f"Base URL: {source.base_url or '(unset)'}",
        f"Auth: {getattr(source.auth, 'type', 'none')} (secret values not shown)",
        f"Timeout: {source.timeout_seconds}s, retries: {source.retries.attempts}",
        f"Operations ({len(source.operations)}):",
    ]
    for op in source.operations:
        detail = {
            "name": op.name,
            "method": op.method,
            "description": op.description,
            "path": op.path,
            "query": (op.query[:200] + "…") if op.query and len(op.query) > 200 else op.query,
            "variables": op.variables,
            # Reported because update_datasource replaces the whole operation
            # list: an operation read back without its query_params would lose
            # the arguments its API takes in the query string of a write.
            "query_params": op.query_params,
            "params": [
                {"name": p.name, "type": p.type, "required": p.required} for p in op.params
            ],
            "destructive": op.destructive,
            "mapping": op.mapping,
            "paginate": op.paginate.model_dump() if op.paginate else None,
            "retries": op.retries.model_dump() if op.retries else None,
        }
        lines.append("  " + json.dumps({k: v for k, v in detail.items() if v is not None},
                                       ensure_ascii=False))
    return "\n".join(lines)


@requires(Permission.READ)
async def list_datasources(deps: ManagementDeps) -> str:
    if deps.data_source_backend is None:
        return "Data source backend not configured."
    sources = await deps.data_source_backend.list()
    if not sources:
        return "No data sources found."
    lines = []
    for s in sources:
        detail = "operations: " + (", ".join(op.name for op in s.operations) or "(no operations)")
        lines.append(
            f"- **{s.id}** ({s.name or s.id}, {s.kind}): "
            f"{s.description or '(no description)'} — {detail}"
        )
    return "\n".join(lines)


def _google_subject_error(auth: Any) -> str | None:
    """Why a ``google`` auth block is unusable, as a tool-reply string.

    Thin wrapper over
    ``app.infrastructure.auth.google_token_provider.check_impersonate_subject``
    so the MCP write paths enforce the same restriction as the REST ones: a
    ``google`` block may name only the service account
    ``GOOGLE_IMPERSONATE_SA`` points at, because the backend can impersonate
    every account it has been granted token-creator on and the auth block is
    caller-supplied.
    """
    from app.infrastructure.auth.google_token_provider import (
        check_impersonate_subject,
    )
    return check_impersonate_subject(auth)


@requires(Permission.READ)
async def resolve_google_file(deps: ManagementDeps, ref: str) -> str:
    """Resolve a Google Drive URL / file id and report whether we can reach it.

    Same check the editor's Verify button makes, so an agent can attach a
    spreadsheet by URL the way a person does: the document has to be shared
    with the backend's service account, and only the backend can find out
    whether it is.
    """
    from app.core.config import get_settings
    from app.infrastructure.datasources.google_sheets import (
        resolve_google_file as _resolve,
    )
    result = await _resolve(ref, get_settings())
    if result.get("status") != "ok":
        return f"{result.get('status')}: {result.get('error')}"
    return "\n".join([
        f"File id: {result['file_id']}",
        f"Name: {result['name'] or '(untitled)'}",
        f"Type: {result['mime_type']}",
        f"Writable: {'yes' if result['can_edit'] else 'no (shared read-only)'}",
    ])


@requires(Permission.WRITE)
async def create_google_sheets_datasource(
    deps: ManagementDeps,
    source_id: str = "google-sheets",
    name: str = "Google Sheets",
    description: str = "",
) -> str:
    """Create the Google Sheets data source with its Sheets v4 operations.

    The operation templates come from code
    (``app.infrastructure.datasources.google_sheets``) rather than being
    retyped, so this and the editor's "Google Sheets" preset produce the same
    source.  Writes are gated behind approval and default to ``RAW`` values.
    """
    if deps.data_source_backend is None:
        return "Data source creation unavailable: no persistent backend configured."

    from app.core.config import get_settings
    from app.domain.models.data_source_definition import (
        DataSourceDefinition,
        validate_operations,
    )
    from app.infrastructure.datasources.google_sheets import google_sheets_template

    settings = get_settings()
    template = google_sheets_template(settings)
    if not template["service_account"]:
        return (
            "Google auth is not configured on this backend — set "
            "GOOGLE_IMPERSONATE_SA to the service account spreadsheets are "
            "shared with, then create the source."
        )

    existing = await deps.data_source_backend.get(source_id)
    if existing is not None:
        return f"Data source '{source_id}' already exists. Use update_datasource to modify it."

    try:
        defn = DataSourceDefinition.model_validate({
            "id": source_id,
            "name": name or template["name"],
            "description": description or template["description"],
            "kind": template["kind"],
            "base_url": template["base_url"],
            "auth": template["auth"],
            "operations": template["operations"],
        })
        validate_operations(defn)
    except Exception as exc:
        return f"Invalid data source definition: {exc}"

    await deps.data_source_backend.create(defn)
    await _publish_datasources(deps)
    return (
        f"Data source '{source_id}' created with {len(defn.operations)} "
        f"operation(s): {', '.join(op.name for op in defn.operations)}. "
        f"Share each spreadsheet with {template['service_account']} "
        "(Editor, for writes) before calling it."
    )


# ---------------------------------------------------------------------------
# Sheet bindings
#
# Same operations as the REST routes in app/api/routes/datasources.py, and the
# same two validation layers: the pydantic shape, then validate_bindings (an
# unknown column, a mode missing its required field).  An agent authoring a
# binding must not have an easier time than a person authoring one in the form
# -- a binding is what makes a write land on the right cells, and it is checked
# the same way whichever surface wrote it.
# ---------------------------------------------------------------------------

async def _binding_source(deps: ManagementDeps, source_id: str):
    """``(source, None)`` or ``(None, error_str)`` for a binding tool."""
    if deps.data_source_backend is None:
        return None, "Data source backend not configured."
    resolved, err = await _resolve_datasource_id(deps, source_id)
    if err:
        return None, err
    source = await deps.data_source_backend.get(resolved)
    if source is None:
        return None, f"Data source '{resolved}' not found."
    # A stored `google` auth block naming a foreign principal is refused here
    # too: this is the surface that would mint the token from it.
    subject_error = _google_subject_error(
        source.auth.model_dump(mode="json") if source.auth else None
    )
    if subject_error:
        return None, subject_error
    return source, None


def _render_binding(binding: Any) -> str:
    payload = binding.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False)


def _tier_line(binding: Any) -> str:
    """One line saying whether a binding holds generated code, and its state.

    Tier is an outcome, never a setting: it is answered by looking at whether
    the binding carries a transform. An agent reading a binding back has to be
    told this without asking -- "deterministic form" and "code a model wrote"
    are very different things to be about to run.
    """
    from app.application.sheet_compute_service import compute_status

    status = compute_status(binding)
    if not status["generated"]:
        return "tier 1: mapped declaratively, deterministic, no code."

    golden = status.get("golden") or {}
    age = golden.get("verified_days_ago")
    verified = (
        f"last verified {age:.1f}d ago" if isinstance(age, (int, float))
        else "never verified"
    )
    flags = []
    if not status["activated"]:
        flags.append("NOT ACTIVATED (compiled but switched off)")
    if status["stale"]:
        flags.append(f"STALE ({status['stale_reason']})")
    if status["edited_by_human"]:
        flags.append("edited by hand — will not be regenerated")
    tail = ("; " + "; ".join(flags)) if flags else ""
    return (
        f"tier 2: GENERATED CODE ({status['script_id']}, model "
        f"{status['model_id'] or 'unknown'}), {verified}{tail}. "
        f"Instruction: {status['instruction'] or '(none)'}"
    )


# ---------------------------------------------------------------------------
# Tier 2: generated transforms
# ---------------------------------------------------------------------------
# Every rule these enforce lives in app.application.sheet_compute_service, so
# this surface, the REST API and the chat agent cannot differ on any of them --
# which is the failure worth designing against, because a caller who finds one
# lenient surface stops using the strict one.
#
# Permission tiers follow the rest of the datasource tooling: reading is READ,
# mutating is WRITE. On top of that, the three tools that *store executable
# code* (compile, edit, activate) additionally require ADMIN, raised from
# inside the service by auth.sandbox_guard.assert_generated_code_allowed. WRITE
# gets you to the tool; ADMIN is what lets it save code nobody has read yet.


def _compute_deps(deps: ManagementDeps) -> dict[str, Any]:
    """The persistence and publication hooks the tier-2 service takes."""
    return {
        "backend": deps.data_source_backend,
        "script_backend": deps.script_backend,
        "publish": lambda: _publish_datasources(deps),
    }


def _render_compute_result(result: dict[str, Any], *, code: bool = True) -> str:
    """A tier-2 service result as text an agent can act on."""
    status = result.get("status")
    name = result.get("binding") or "?"

    if status == "needs":
        lines = [
            f"Binding '{name}': the instruction is ambiguous about the data, so "
            "nothing was generated. Answer these and call compile again with "
            "the answers (the exact question text is the key):",
        ]
        for need in result.get("needs") or []:
            options = ", ".join(need.get("options") or []) or "(free text)"
            lines.append(f"- {need['question']}\n  options: {options}")
        return "\n".join(lines)

    if status == "stale":
        return (
            f"Binding '{name}' is now STALE and switched off: "
            f"{result.get('error')}"
        )

    compute = result.get("compute") or {}
    lines = [f"Binding '{name}': {status}."]
    if compute:
        lines.append(
            f"script {compute.get('script_id')} (hash {compute.get('content_hash')}), "
            f"activated={compute.get('activated')}, stale={compute.get('stale')}, "
            f"edited_by_human={compute.get('edited_by_human')}"
        )
    if result.get("rationale"):
        lines.append(f"Rationale: {result['rationale']}")
    if "output" in result:
        lines.append(
            "Output on the verification rows: "
            + json.dumps(result["output"], ensure_ascii=False, default=str)[:1500]
        )
    if code and result.get("code"):
        lines.append("Code:\n" + result["code"])
    if status == "ok" and not (compute.get("activated") if compute else True):
        lines.append(
            "It is NOT running yet. Review the code and the output above, then "
            "call activate_sheet_binding_code to switch it on."
        )
    return "\n".join(lines)


async def _compute_call(fn: Any, **kwargs: Any) -> str:
    """Run one tier-2 service call, turning every refusal into tool text."""
    from app.application.sheet_compute_service import ComputeServiceError
    from app.infrastructure.auth.sandbox_guard import GeneratedCodeNotPermittedError

    try:
        result = await fn(**kwargs)
    except GeneratedCodeNotPermittedError as exc:
        return f"Not permitted: {exc}"
    except ComputeServiceError as exc:
        return f"Refused: {exc}"
    if result.get("status") == "error":
        return (
            f"Binding '{result.get('binding')}': compilation failed after "
            f"{result.get('attempts')} attempt(s). The last checker rejection "
            f"was:\n{result.get('error')}"
        )
    return _render_compute_result(result)


@requires(Permission.WRITE)
async def compile_sheet_binding_code(
    deps: ManagementDeps,
    name: str,
    instruction: str = "",
    answers_json: str = "",
    force: bool = False,
    source_id: str = "google-sheets",
) -> str:
    """Generate the computation of a binding a declarative form cannot express.

    Tier 1 (save_sheet_binding) covers reading rows, reading a row by key and
    setting named columns. It cannot express computation across rows -- grouping,
    aggregation, arithmetic. This generates just that part: a small Python
    function that turns the sheet's records into values, while the binding keeps
    declaring the document, the tab, the key column and the columns that may be
    written. Generated code produces values, never addresses, and cannot touch a
    column the binding does not list.

    Save the binding first (save_sheet_binding), including the column map a
    write is allowed to set; this fills in its computation.

    Three possible outcomes: clarifying questions (nothing is stored -- answer
    them and call again with answers_json), a successful compile (the code is
    stored switched OFF, and activate_sheet_binding_code turns it on after you
    have read it), or a failure after several attempts with the checker's own
    rejection.

    Requires admin permission: it stores code that this backend later executes.

    Args:
        name: Binding to generate the computation for.
        instruction: What to compute, in plain language. Omit to re-use the
            instruction already stored on the binding, which is what a plain
            recompile means.
        answers_json: JSON object of {"<question>": "<answer>"} answering a
            previous call's questions. Folded into the binding, so later
            recompiles reproduce rather than re-guess.
        force: Re-run the model even when nothing about the request changed.
        source_id: Data source holding the binding (default "google-sheets").
    """
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    answers: dict[str, str] = {}
    if answers_json:
        try:
            parsed = json.loads(answers_json)
        except json.JSONDecodeError as exc:
            return f"Invalid answers_json: {exc}"
        if not isinstance(parsed, dict):
            return "answers_json must be a JSON object of question -> answer."
        answers = {str(k): str(v) for k, v in parsed.items()}

    from app.core.config import get_settings
    from app.application.sheet_compute_service import compile_compute

    executor = _binding_executor(deps)
    if executor is None:
        return "Data source executor not configured."
    return await _compute_call(
        compile_compute,
        source=source,
        name=name,
        instruction=instruction or None,
        answers=answers,
        settings=get_settings(),
        executor=executor,
        force=force,
        **_compute_deps(deps),
    )


@requires(Permission.READ)
async def get_sheet_binding_code(
    deps: ManagementDeps, name: str, source_id: str = "google-sheets"
) -> str:
    """Read the generated code of a binding, with its verification state.

    Use this before activating a binding, and whenever one behaves oddly: it
    reports the code itself, the model and instruction it came from, whether it
    is activated or stale, whether a person has edited it, and when its golden
    fixture last reproduced.

    Args:
        name: Binding to read the code of.
        source_id: Data source holding the binding (default "google-sheets").
    """
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    binding = source.get_binding(name)
    if binding is None:
        return f"Binding '{name}' not found on '{source.id}'."
    if binding.compute is None:
        return (
            f"Binding '{name}' is a tier-1 binding: mapped declaratively, "
            "deterministic, no code. There is nothing to review."
        )
    from app.infrastructure.datasources.sheet_compute import TRANSFORM_SIGNATURE

    golden = binding.resolution.golden
    fixture = (
        f"{len(golden.input_rows)} row(s), output hash {golden.output_hash}"
        if golden else "none"
    )
    return "\n".join([
        _tier_line(binding),
        f"Signature: {TRANSFORM_SIGNATURE}",
        f"Golden fixture: {fixture}",
        f"Output shape: {binding.compute.output_shape}",
        "Code:",
        binding.compute.code,
    ])


@requires(Permission.WRITE)
async def edit_sheet_binding_code(
    deps: ManagementDeps,
    name: str,
    code: str,
    source_id: str = "google-sheets",
) -> str:
    """Replace a binding's generated code with a hand-written version.

    Held to exactly the checks the generated version was: the same allow-list on
    what the code may contain, the same sandbox, the same run-it-twice
    determinism check, and the same rule that a write may only produce values
    for the columns the binding lists. Being hand-written changes who is
    accountable, not what the code may do.

    This permanently stops regeneration for this binding: a later compile
    refuses rather than overwriting the edit. The binding is also switched off
    again, because this is new code and the previous approval was of the
    previous code.

    Requires admin permission: it stores code that this backend later executes.

    Args:
        name: Binding whose code to replace.
        code: The full replacement source, defining transform(records, params).
        source_id: Data source holding the binding (default "google-sheets").
    """
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    from app.core.config import get_settings
    from app.application.sheet_compute_service import edit_compute_code

    return await _compute_call(
        edit_compute_code,
        source=source,
        name=name,
        code=code,
        settings=get_settings(),
        **_compute_deps(deps),
    )


@requires(Permission.WRITE)
async def activate_sheet_binding_code(
    deps: ManagementDeps, name: str, source_id: str = "google-sheets"
) -> str:
    """Switch a compiled transform on, after re-proving it against its fixture.

    Compiling and activating are two separate events on purpose: the first says
    the code passed its checks, the second says somebody read the code and its
    output and accepted it. Read get_sheet_binding_code first -- activating
    without doing so is exactly the step this design exists to make deliberate.

    Requires admin permission: it puts generated code into service.

    Args:
        name: Binding to activate.
        source_id: Data source holding the binding (default "google-sheets").
    """
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    from app.core.config import get_settings
    from app.application.sheet_compute_service import activate_compute

    return await _compute_call(
        activate_compute,
        source=source,
        name=name,
        settings=get_settings(),
        **_compute_deps(deps),
    )


@requires(Permission.WRITE)
async def retest_sheet_binding_code(
    deps: ManagementDeps, name: str, source_id: str = "google-sheets"
) -> str:
    """Re-run a binding's frozen test and re-check the sheet's header row.

    Two independent questions: does the code still compute the answer it was
    approved for, and does the sheet still have the header row it was written
    against. Either one failing marks the binding stale and switches it off,
    which is a change to it -- hence a write, not a read.

    Args:
        name: Binding to re-test.
        source_id: Data source holding the binding (default "google-sheets").
    """
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    from app.core.config import get_settings
    from app.application.sheet_compute_service import retest_compute

    executor = _binding_executor(deps)
    if executor is None:
        return "Data source executor not configured."
    return await _compute_call(
        retest_compute,
        source=source,
        name=name,
        settings=get_settings(),
        executor=executor,
        **_compute_deps(deps),
    )


@requires(Permission.WRITE)
async def mark_sheet_binding_stale(
    deps: ManagementDeps,
    name: str,
    reason: str = "",
    source_id: str = "google-sheets",
) -> str:
    """Switch a binding's generated code off until somebody re-confirms it.

    Use this the moment a generated binding looks wrong. It stops running
    immediately; re-testing it and activating it again are what bring it back.
    No admin permission is needed -- stopping something suspicious should never
    be the privileged direction.

    Args:
        name: Binding to mark stale.
        reason: Why, recorded on the binding and shown in the editor.
        source_id: Data source holding the binding (default "google-sheets").
    """
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    from app.application.sheet_compute_service import mark_compute_stale

    return await _compute_call(
        mark_compute_stale,
        source=source,
        name=name,
        reason=reason,
        **_compute_deps(deps),
    )


@requires(Permission.READ)
async def probe_google_sheet(
    deps: ManagementDeps,
    file_id: str,
    sheet: str = "",
    header_row: int = 1,
    source_id: str = "google-sheets",
) -> str:
    """Read a spreadsheet's tabs, header row and a few real rows.

    The one call a binding is authored from: every column name a binding may
    use has to come from here, and the fingerprint it returns is what detects a
    later change to the header row.
    """
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    executor = _binding_executor(deps)
    if executor is None:
        return "Data source executor not configured."
    from app.infrastructure.datasources.sheet_binding_runtime import probe_sheet
    try:
        result = await probe_sheet(source, executor, file_id, sheet or None, header_row)
    except Exception as exc:  # noqa: BLE001 — reported like the REST probe does
        return f"Could not probe that spreadsheet: {exc}"
    lines = [
        f"Spreadsheet: {file_id}",
        f"Tab: {result['sheet']} (sheet_id {result['sheet_id']})",
        "Tabs: " + ", ".join(f"{t['title']} ({t['sheet_id']})" for t in result["tabs"]),
        "Named ranges: " + (", ".join(n["name"] for n in result["named_ranges"]) or "(none)"),
        f"Header row: {result['header_row']}",
        "Headers: " + ", ".join(result["headers"]),
        f"Fingerprint: {result['fingerprint']}",
        "Sample rows:",
    ]
    for row in result["sample_rows"]:
        lines.append("  " + json.dumps(row, ensure_ascii=False))
    return "\n".join(lines)


def _binding_executor(deps: ManagementDeps) -> Any:
    """The shared executor, or a fresh one.

    ManagementDeps deliberately carries only backends, so the executor is not
    one of its fields; a plain instance is equivalent here because everything a
    binding needs comes from the source definition it is handed.
    """
    from app.infrastructure.datasources.executor import DataSourceExecutor
    return DataSourceExecutor()


@requires(Permission.READ)
async def list_sheet_bindings(deps: ManagementDeps, source_id: str = "google-sheets") -> str:
    """List the sheet bindings on a data source and what each one compiles to."""
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    if not source.bindings:
        return f"Data source '{source.id}' has no bindings."
    lines = [f"Bindings on '{source.id}' ({len(source.bindings)}):"]
    for binding in source.bindings:
        target = f"{binding.document.name or binding.document.file_id} / {binding.document.sheet}"
        detail = (
            f"mode {binding.read.mode}" if binding.operation == "read" and binding.read
            else f"mode {binding.write.mode}" if binding.write else "?"
        )
        lines.append(
            f"- **{binding.name}** ({binding.operation}, {detail}) on {target} "
            f"-> operation '{binding.name}'"
        )
        lines.append("  " + _tier_line(binding))
    return "\n".join(lines)


@requires(Permission.READ)
async def get_sheet_binding(
    deps: ManagementDeps, name: str, source_id: str = "google-sheets"
) -> str:
    """Read one binding back in full, as the JSON save_sheet_binding accepts."""
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    binding = source.get_binding(name)
    if binding is None:
        return f"Binding '{name}' not found on '{source.id}'."
    return _render_binding(binding)


@requires(Permission.WRITE)
async def save_sheet_binding(
    deps: ManagementDeps,
    binding_json: str,
    source_id: str = "google-sheets",
) -> str:
    """Create or replace a sheet binding, compiling it into an operation.

    Idempotent by name: a binding whose name already exists is replaced, so
    re-sending a corrected binding is the way to fix one.
    """
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    try:
        payload = json.loads(binding_json)
    except json.JSONDecodeError as exc:
        return f"Invalid binding_json: {exc}"
    if not isinstance(payload, dict):
        return "binding_json must be a JSON object describing one binding."

    from app.domain.models.data_source_definition import validate_operations
    from app.domain.models.sheet_binding import (
        SheetBinding,
        header_fingerprint,
        validate_bindings,
    )
    from app.infrastructure.datasources.sheet_binding_compile import (
        refresh_binding_operations,
        stamp_compiled,
    )
    from app.infrastructure.datasources.sheet_binding_library import (
        ensure_binding_scripts,
    )

    try:
        binding = SheetBinding.model_validate(payload)
    except Exception as exc:
        return f"Invalid binding: {exc}"
    # Provenance and generated code are not caller-supplied; see the REST path
    # (`_parse_binding`) for the reasoning. Carried over from what is stored, so
    # editing the form of a tier-2 binding keeps its code, and never read from
    # the payload, so this tool cannot fabricate LLM provenance or smuggle in a
    # transform.
    stored = source.get_binding(binding.name)
    if stored is not None and stored.compute is not None:
        binding.compute = stored.compute.model_copy(deep=True)
        binding.resolution = stored.resolution.model_copy(deep=True)
    else:
        binding.compute = None
        binding.resolution.tier = "binding"
        binding.resolution.authored_by = "human"
        binding.resolution.instruction = None
        binding.resolution.model_id = None
        binding.resolution.answers = {}
        binding.resolution.golden = None
        binding.resolution.script_id = None
        binding.resolution.edited_by_human = False
    if not binding.sheet_schema.fingerprint:
        binding.sheet_schema.fingerprint = header_fingerprint(binding.sheet_schema.headers)

    replacing = source.get_binding(binding.name) is not None
    if not replacing and source.get_operation(binding.name) is not None:
        return (
            f"'{binding.name}' is already an operation of '{source.id}' — a "
            "binding compiles to an operation, so the names cannot collide."
        )
    bindings = [b for b in source.bindings if b.name != binding.name] + [binding]
    try:
        validate_bindings(bindings)
    except ValueError as exc:
        return f"Invalid binding: {exc}"
    updated = source.model_copy(update={
        "bindings": [stamp_compiled(b) for b in bindings],
        "operations": refresh_binding_operations(source, bindings),
    })
    try:
        validate_operations(updated)
    except ValueError as exc:
        return f"Invalid binding: {exc}"

    await deps.data_source_backend.update(source.id, updated)
    await ensure_binding_scripts(deps.script_backend)
    await _publish_datasources(deps)
    verb = "replaced" if replacing else "created"
    params = ", ".join(p.name for p in updated.get_operation(binding.name).params) or "none"
    return (
        f"Binding '{binding.name}' {verb} on '{source.id}' and compiled into "
        f"operation '{binding.name}' (params: {params}). "
        + (
            "It is a write, so calls go through the approval gate."
            if binding.operation == "write" else ""
        )
    ).strip()


@requires(Permission.DELETE)
async def delete_sheet_binding(
    deps: ManagementDeps, name: str, source_id: str = "google-sheets"
) -> str:
    """Delete a binding and the operation it compiled to."""
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    if source.get_binding(name) is None:
        return f"Binding '{name}' not found on '{source.id}'."
    from app.infrastructure.datasources.sheet_binding_compile import (
        refresh_binding_operations,
    )
    bindings = [b for b in source.bindings if b.name != name]
    updated = source.model_copy(update={
        "bindings": bindings,
        "operations": refresh_binding_operations(source, bindings),
    })
    await deps.data_source_backend.update(source.id, updated)
    await _publish_datasources(deps)
    return f"Binding '{name}' and its operation deleted from '{source.id}'."


@requires(Permission.READ)
async def preview_sheet_binding(
    deps: ManagementDeps,
    name: str,
    state_json: str = "",
    source_id: str = "google-sheets",
) -> str:
    """Resolve a binding against sample state and report what it would do.

    A write is planned, never sent: the sheet is read, the row resolved, the
    header fingerprint checked and the write composed, and what comes back is
    the target cells with their before and after values. A read is executed and
    its result summarised.
    """
    source, err = await _binding_source(deps, source_id)
    if err:
        return err
    binding = source.get_binding(name)
    if binding is None:
        return f"Binding '{name}' not found on '{source.id}'."
    state: dict = {}
    if state_json:
        try:
            state = json.loads(state_json)
        except json.JSONDecodeError as exc:
            return f"Invalid state_json: {exc}"
        if not isinstance(state, dict):
            return "state_json must be a JSON object."

    executor = _binding_executor(deps)
    from app.infrastructure.datasources.sheet_binding_runtime import (
        params_from_state,
        plan_write_binding,
        render_cell_changes,
        run_read_binding,
    )
    params = params_from_state(binding, state)
    try:
        if binding.operation == "read":
            result = await run_read_binding(source, executor, binding, params)
            count = len(result) if isinstance(result, list) else (0 if result is None else 1)
            return (
                f"Binding '{name}' would return {count} row(s):\n"
                + json.dumps(result, ensure_ascii=False, default=str)[:4000]
            )
        plan = await plan_write_binding(source, executor, binding, params)
        if plan["status"] == "skipped":
            return f"Binding '{name}' would write nothing: {plan['reason']}"
        lines = [
            f"Binding '{name}' would write to "
            f"{binding.document.name or binding.document.file_id} / "
            f"{binding.document.sheet}"
            + (f" row {plan['row_number']}" if plan.get("row_number") else " (new row)"),
            f"valueInputOption {plan['value_input_option']}, "
            f"blank_policy {plan['blank_policy']}",
        ]
        lines += ["  " + line for line in render_cell_changes(plan["cells"])]
        lines.append("Columns not listed above are not touched.")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 — a preview reports, never raises
        return f"Binding '{name}' could not be previewed: {exc}"


@requires(Permission.WRITE)
async def create_pubsub_datasource(
    deps: ManagementDeps,
    source_id: str,
    name: str,
    topic: str,
    event_schema_json: str = "",
    subscription: str = "",
    project_id: str = "",
    description: str = "",
) -> str:
    """Deprecated spelling of ``create_event`` — Pub/Sub topics are events now."""
    return await create_event(
        deps, source_id, name, topic, event_schema_json, subscription, project_id, description,
    )


@requires(Permission.READ)
def list_pubsub_subscriptions(deps: ManagementDeps) -> str:
    """Report which workflow steps are currently subscribed, and to what."""
    if deps.pubsub_subscriber is None:
        return "Pub/Sub triggers are disabled (PUBSUB_ENABLED is false)."
    registrations = deps.pubsub_subscriber.registrations()
    if not registrations:
        return "No Pub/Sub trigger is currently subscribed."
    return "\n".join(
        f"- **{key}** → {subscription}" for key, subscription in sorted(registrations.items())
    )


@requires(Permission.WRITE)
async def create_datasource(
    deps: ManagementDeps,
    source_id: str,
    name: str,
    base_url: str,
    operations_json: str,
    description: str = "",
    kind: str = "http",
    auth_json: str = "",
) -> str:
    if deps.data_source_backend is None:
        return "Data source creation unavailable: no persistent backend configured."
    try:
        operations = json.loads(operations_json)
    except json.JSONDecodeError as exc:
        return f"Invalid operations_json: {exc}"
    if not isinstance(operations, list):
        return "operations_json must be a JSON array."
    auth: dict = {"type": "none"}
    if auth_json:
        try:
            auth = json.loads(auth_json)
        except json.JSONDecodeError as exc:
            return f"Invalid auth_json: {exc}"

    # `google` auth may name only the configured principal — the MCP surface
    # must not be a way around the check the REST create path makes.
    subject_error = _google_subject_error(auth)
    if subject_error:
        return subject_error

    existing = await deps.data_source_backend.get(source_id)
    if existing is not None:
        return f"Data source '{source_id}' already exists. Use update_datasource to modify it."

    from app.domain.models.data_source_definition import (
        DataSourceDefinition,
        validate_operations,
    )
    try:
        defn = DataSourceDefinition.model_validate({
            "id": source_id,
            "name": name,
            "description": description,
            "kind": kind,
            "base_url": base_url,
            "auth": auth,
            "operations": operations,
        })
        validate_operations(defn)
    except Exception as exc:
        return f"Invalid data source definition: {exc}"

    await deps.data_source_backend.create(defn)
    await _publish_datasources(deps)
    return f"Data source '{source_id}' created with {len(defn.operations)} operation(s)."


@requires(Permission.WRITE)
async def update_datasource(
    deps: ManagementDeps,
    source_id: str,
    name: str | None = None,
    description: str | None = None,
    base_url: str | None = None,
    operations_json: str | None = None,
    auth_json: str | None = None,
    pubsub_json: str | None = None,
) -> str:
    if deps.data_source_backend is None:
        return "Data source updates unavailable: no persistent backend configured."
    resolved, err = await _resolve_datasource_id(deps, source_id)
    if err:
        return err
    existing = await deps.data_source_backend.get(resolved)
    if existing is None:
        return f"Data source '{resolved}' not found."

    payload = existing.model_dump(mode="json")
    if name is not None:
        payload["name"] = name
    if description is not None:
        payload["description"] = description
    if base_url is not None:
        payload["base_url"] = base_url
    if operations_json is not None:
        try:
            operations = json.loads(operations_json)
        except json.JSONDecodeError as exc:
            return f"Invalid operations_json: {exc}"
        if not isinstance(operations, list):
            return "operations_json must be a JSON array."
        payload["operations"] = operations
    if auth_json is not None:
        try:
            payload["auth"] = json.loads(auth_json)
        except json.JSONDecodeError as exc:
            return f"Invalid auth_json: {exc}"
        subject_error = _google_subject_error(payload["auth"])
        if subject_error:
            return subject_error
    if pubsub_json is not None:
        # {topic, subscription, project_id, event_schema} — kind == "pubsub".
        try:
            pubsub_block = json.loads(pubsub_json)
        except json.JSONDecodeError as exc:
            return f"Invalid pubsub_json: {exc}"
        if not isinstance(pubsub_block, dict):
            return "pubsub_json must be a JSON object."
        payload["pubsub"] = pubsub_block

    from app.domain.models.data_source_definition import (
        DataSourceDefinition,
        validate_operations,
    )
    try:
        defn = DataSourceDefinition.model_validate(payload)
        validate_operations(defn)
    except Exception as exc:
        return f"Invalid data source definition: {exc}"

    await deps.data_source_backend.update(resolved, defn)
    await _publish_datasources(deps)
    return f"Data source '{resolved}' updated."


# --- Data source schema import ---
#
# Specifications are parsed by code, never re-typed by the model: the tools
# below hand back operation *names*, and the create/extend tools copy the
# parsed operation objects verbatim. That keeps a 500-endpoint OpenAPI
# document out of the context window and out of the failure modes of
# transcribing JSON.

def _parse_auth_block(auth_json: str):
    """(auth_model, None) or (None, error_str) from a JSON auth block."""
    if not auth_json:
        return None, None
    from pydantic import TypeAdapter

    from app.domain.models.data_source_definition import AnyDataSourceAuth
    try:
        return TypeAdapter(AnyDataSourceAuth).validate_python(json.loads(auth_json)), None
    except Exception as exc:
        return None, f"Invalid auth_json: {exc}"


async def _load_spec(schema_url: str, kind: str, auth_json: str):
    """(spec_dict, None) or (None, error_str)."""
    from app.infrastructure.datasources.discovery import (
        SpecFetchError,
        SpecParseError,
        fetch_and_parse_spec,
    )
    auth_model, err = _parse_auth_block(auth_json)
    if err:
        return None, err
    try:
        spec = await fetch_and_parse_spec(
            schema_url,
            kind="graphql" if kind == "graphql" else "http",
            auth=auth_model,
        )
    except (SpecFetchError, SpecParseError) as exc:
        return None, f"Could not import the schema: {exc}"
    return spec, None


def _select_operations(spec: dict, names_csv: str):
    """(operations, None) or (None, error_str) — the named subset of a spec."""
    wanted = [n.strip() for n in names_csv.split(",") if n.strip()]
    available = {op["name"]: op for op in spec["operations"]}
    if not wanted:
        return None, (
            "operation_names is required — call import_datasource_schema first "
            "and pass a comma-separated subset of the names it lists."
        )
    missing = [n for n in wanted if n not in available]
    if missing:
        return None, (
            f"Unknown operation(s): {', '.join(missing)}. "
            f"Available: {', '.join(list(available)[:40])}"
            + (" …" if len(available) > 40 else "")
        )
    return [available[n] for n in wanted], None


@requires(Permission.WRITE)
async def import_datasource_schema(
    deps: ManagementDeps, schema_url: str, kind: str = "http", auth_json: str = ""
) -> str:
    spec, err = await _load_spec(schema_url, kind, auth_json)
    if err:
        return err
    lines = [
        f"Schema: {spec['source']} ({spec['kind']})",
        f"Declared base URL: {spec['base_url'] or '(none declared)'}",
        f"Operations: {len(spec['operations'])}",
    ]
    for op in spec["operations"]:
        target = op.get("path") or (op.get("query") or "")[:90]
        params = ", ".join(
            f"{p['name']}{'' if p['required'] else '?'}" for p in op.get("params") or []
        )
        summary = (op.get("summary") or "").strip()
        lines.append(
            f"- {op['name']} [{op['method']}] {target}"
            + (f" params: {params}" if params else "")
            + (f" — {summary}" if summary else "")
        )
    return "\n".join(lines)


@requires(Permission.WRITE)
async def create_datasource_from_schema(
    deps: ManagementDeps,
    source_id: str,
    name: str,
    schema_url: str,
    operation_names: str,
    base_url: str = "",
    description: str = "",
    kind: str = "http",
    auth_json: str = "",
) -> str:
    if deps.data_source_backend is None:
        return "Data source creation unavailable: no persistent backend configured."
    existing = await deps.data_source_backend.get(source_id)
    if existing is not None:
        return (
            f"Data source '{source_id}' already exists. Use "
            "add_datasource_operations_from_schema to extend it."
        )
    spec, err = await _load_spec(schema_url, kind, auth_json)
    if err:
        return err
    operations, err = _select_operations(spec, operation_names)
    if err:
        return err

    resolved_base = base_url.strip() or spec["base_url"] or ""
    if not resolved_base:
        return (
            "No base URL: the specification declares none, so pass base_url "
            "explicitly."
        )
    auth: dict = {"type": "none"}
    if auth_json:
        try:
            auth = json.loads(auth_json)
        except json.JSONDecodeError as exc:
            return f"Invalid auth_json: {exc}"
    subject_error = _google_subject_error(auth)
    if subject_error:
        return subject_error

    from app.domain.models.data_source_definition import (
        DataSourceDefinition,
        validate_operations,
    )
    try:
        defn = DataSourceDefinition.model_validate({
            "id": source_id,
            "name": name,
            "description": description,
            "kind": spec["kind"] if spec["kind"] == "graphql" else "http",
            "base_url": resolved_base,
            "auth": auth,
            "operations": operations,
        })
        validate_operations(defn)
    except Exception as exc:
        return f"Invalid data source definition: {exc}"

    await deps.data_source_backend.create(defn)
    await _publish_datasources(deps)
    return (
        f"Data source '{source_id}' created from {spec['source']} with "
        f"{len(defn.operations)} operation(s): "
        f"{', '.join(op.name for op in defn.operations)}"
    )


@requires(Permission.WRITE)
async def add_datasource_operations_from_schema(
    deps: ManagementDeps,
    source_id: str,
    schema_url: str,
    operation_names: str,
    kind: str = "http",
    auth_json: str = "",
) -> str:
    if deps.data_source_backend is None:
        return "Data source updates unavailable: no persistent backend configured."
    resolved, err = await _resolve_datasource_id(deps, source_id)
    if err:
        return err
    existing = await deps.data_source_backend.get(resolved)
    if existing is None:
        return f"Data source '{resolved}' not found."
    spec, err = await _load_spec(schema_url, kind, auth_json)
    if err:
        return err
    operations, err = _select_operations(spec, operation_names)
    if err:
        return err

    payload = existing.model_dump(mode="json")
    present = {op["name"] for op in payload["operations"]}
    added = [op for op in operations if op["name"] not in present]
    skipped = [op["name"] for op in operations if op["name"] in present]
    if not added:
        return f"Nothing to add — already present: {', '.join(skipped)}"
    payload["operations"] = [*payload["operations"], *added]

    from app.domain.models.data_source_definition import (
        DataSourceDefinition,
        validate_operations,
    )
    try:
        defn = DataSourceDefinition.model_validate(payload)
        validate_operations(defn)
    except Exception as exc:
        return f"Invalid data source definition: {exc}"

    await deps.data_source_backend.update(resolved, defn)
    await _publish_datasources(deps)
    return (
        f"Added {len(added)} operation(s) to '{resolved}': "
        f"{', '.join(op['name'] for op in added)}"
        + (f" (skipped, already present: {', '.join(skipped)})" if skipped else "")
    )


@requires(Permission.DELETE)
async def delete_datasource(deps: ManagementDeps, source_id: str) -> str:
    if deps.data_source_backend is None:
        return "Data source deletion unavailable: no persistent backend configured."
    resolved, err = await _resolve_datasource_id(deps, source_id)
    if err:
        return err
    existing = await deps.data_source_backend.get(resolved)
    if existing is None:
        return f"Data source '{resolved}' not found."
    await deps.data_source_backend.delete(resolved)
    await _publish_datasources(deps)
    return f"Data source '{resolved}' deleted."


# ---------------------------------------------------------------------------
# Messaging tools
#
# The same provider abstraction the `slack` workflow step uses
# (app.infrastructure.messaging), so a message posted by an operator over MCP
# and one posted by a workflow step travel the identical code path.  The
# credential is never a parameter: providers read it from settings, which is why
# none of these take a token and none of them can be talked into using one.
# ---------------------------------------------------------------------------

_MESSAGING_PREVIEW_CHARS = 300


def _messaging(provider: str = ""):
    from app.infrastructure.messaging import get_provider

    return get_provider(provider or None)


def _render_messages(messages: list[Any]) -> str:
    if not messages:
        return "No messages."
    lines = []
    for message in messages:
        text = (message.text or "").replace("\n", " ⏎ ")
        if len(text) > _MESSAGING_PREVIEW_CHARS:
            text = text[:_MESSAGING_PREVIEW_CHARS] + "…"
        thread = f" [thread {message.thread_id}]" if message.thread_id else ""
        lines.append(f"- {message.id} {message.author or 'unknown'}{thread}: {text}")
    return "\n".join(lines)


@requires(Permission.WRITE)
async def post_message(
    deps: ManagementDeps,
    channel: str,
    text: str,
    thread_id: str = "",
    provider: str = "slack",
) -> str:
    try:
        posted = await _messaging(provider).post_message(
            channel, text, thread_id=thread_id or None
        )
    except Exception as exc:  # noqa: BLE001 — a tool reports, it does not raise
        return f"Could not post the message: {exc}"
    where = f" in thread {thread_id}" if thread_id else ""
    return f"Posted to {posted.channel}{where} as message {posted.id}."


@requires(Permission.READ)
async def read_messages(
    deps: ManagementDeps,
    channel: str,
    limit: int = 20,
    oldest: str = "",
    provider: str = "slack",
) -> str:
    try:
        messages = await _messaging(provider).read_history(
            channel, oldest=oldest or None, limit=limit
        )
    except Exception as exc:  # noqa: BLE001
        return f"Could not read the channel: {exc}"
    return _render_messages(messages)


@requires(Permission.READ)
async def read_thread(
    deps: ManagementDeps,
    channel: str,
    thread_id: str,
    provider: str = "slack",
) -> str:
    try:
        messages = await _messaging(provider).read_thread(channel, thread_id)
    except Exception as exc:  # noqa: BLE001
        return f"Could not read the thread: {exc}"
    return _render_messages(messages)


@requires(Permission.WRITE)
async def send_direct_message(
    deps: ManagementDeps,
    user_id: str,
    text: str,
    provider: str = "slack",
) -> str:
    try:
        messaging = _messaging(provider)
        channel = await messaging.open_dm(user_id)
        posted = await messaging.post_message(channel, text)
    except Exception as exc:  # noqa: BLE001
        return f"Could not send the direct message: {exc}"
    return f"Direct message sent to {user_id} (channel {channel}, message {posted.id})."


@requires(Permission.DELETE)
async def delete_message(
    deps: ManagementDeps,
    channel: str,
    message_id: str,
    provider: str = "slack",
) -> str:
    try:
        await _messaging(provider).delete_message(channel, message_id)
    except Exception as exc:  # noqa: BLE001
        return f"Could not delete the message: {exc}"
    return f"Message {message_id} deleted from {channel}."
