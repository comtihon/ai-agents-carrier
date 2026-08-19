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
    from app.infrastructure.persistence.data_source_backend import DataSourceDefinitionBackend
    from app.infrastructure.persistence.mongo import MongoGraphRunRepository
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
        refresh_runner=getattr(container, "refresh_runner", None),
        refresh_datasources=refresh_datasources,
        pubsub_subscriber=getattr(container, "pubsub_subscriber", None),
    )


# ---------------------------------------------------------------------------
# Resolvers (private helpers over deps)
# ---------------------------------------------------------------------------

async def _resolve_workflow_id(deps: ManagementDeps, query: str):
    """Returns (resolved_id, None) or (None, error_str)."""
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
    available = ", ".join(f"{d['id']} ({d.get('name', d['id'])})" for d in defs) or "none"
    return None, f"Workflow '{query}' not found. Available: {available}"


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
    lines = [
        f"- **{d['id']}** ({d.get('name', d['id'])}): {(d.get('description') or '').strip()}"
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
        # Include output values for failed/finished steps — skip internal keys
        output_keys = [k for k in run.state if not k.startswith("_")]
        if output_keys:
            parts.append("State keys: " + ", ".join(output_keys))
            for k in output_keys[:8]:  # cap to avoid huge responses
                v = run.state[k]
                if isinstance(v, dict) and "error" in v:
                    parts.append(f"  {k}.error: {v['error'][:300]}")
                elif isinstance(v, dict) and "status" in v:
                    parts.append(f"  {k}.status: {v.get('status')} {str(v.get('body',''))[:200]}")
    if run.error:
        parts.append(f"Error: {run.error[:500]}")
    return "\n".join(parts)


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


@requires(Permission.WRITE)
async def create_workflow(
    deps: ManagementDeps, workflow_id: str, name: str, description: str, steps_json: str
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

    from app.domain.models.workflow_definition import WorkflowDefinition
    defn = WorkflowDefinition(id=workflow_id, name=name, description=description, steps=steps)
    await deps.workflow_backend.create(defn)
    if deps.refresh_runner is not None:
        await deps.refresh_runner(workflow_id)
    return f"Workflow '{workflow_id}' created with {len(steps)} step(s)."


@requires(Permission.WRITE)
async def update_workflow(
    deps: ManagementDeps,
    workflow_id: str,
    name: str | None = None,
    description: str | None = None,
    steps_json: str | None = None,
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

    if name is not None:
        defn.name = name
    if description is not None:
        defn.description = description
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
        defn.steps = steps

    await deps.workflow_backend.update(workflow_id, defn)
    if deps.refresh_runner is not None:
        await deps.refresh_runner(workflow_id)
    return f"Workflow '{workflow_id}' updated."


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
# Data source tools
# ---------------------------------------------------------------------------

@requires(Permission.READ)
async def list_datasources(deps: ManagementDeps) -> str:
    if deps.data_source_backend is None:
        return "Data source backend not configured."
    sources = await deps.data_source_backend.list()
    if not sources:
        return "No data sources found."
    lines = []
    for s in sources:
        if s.kind == "pubsub" and s.pubsub is not None:
            detail = (
                f"topic: {s.pubsub.topic or '(unset)'}, "
                f"subscription: {s.pubsub.subscription or '(created on first use)'}"
            )
        else:
            detail = "operations: " + (", ".join(op.name for op in s.operations) or "(no operations)")
        lines.append(
            f"- **{s.id}** ({s.name or s.id}, {s.kind}): "
            f"{s.description or '(no description)'} — {detail}"
        )
    return "\n".join(lines)


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
    """Register a Pub/Sub topic as a reusable data source.

    Such a source has no operations — it exists so ``pubsub`` trigger steps can
    point at ``datasource: <source_id>`` instead of repeating topic, schema and
    subscription in every workflow.  Leave *subscription* empty to have one
    created (and saved back here) the first time a workflow subscribes.
    """
    if deps.data_source_backend is None:
        return "Data source creation unavailable: no persistent backend configured."
    if not topic.strip():
        return "A Pub/Sub data source needs a topic."
    event_schema: dict | None = None
    if event_schema_json:
        try:
            event_schema = json.loads(event_schema_json)
        except json.JSONDecodeError as exc:
            return f"Invalid event_schema_json: {exc}"
        if not isinstance(event_schema, dict):
            return "event_schema_json must be a JSON object."

    existing = await deps.data_source_backend.get(source_id)
    if existing is not None:
        return f"Data source '{source_id}' already exists. Use update_datasource to modify it."

    from app.domain.models.data_source_definition import DataSourceDefinition
    try:
        defn = DataSourceDefinition.model_validate({
            "id": source_id,
            "name": name,
            "description": description,
            "kind": "pubsub",
            "pubsub": {
                "topic": topic.strip(),
                "subscription": subscription.strip(),
                "project_id": project_id.strip(),
                "event_schema": event_schema,
            },
        })
    except Exception as exc:
        return f"Invalid data source definition: {exc}"

    await deps.data_source_backend.create(defn)
    await _publish_datasources(deps)
    return (
        f"Pub/Sub data source '{source_id}' created for topic '{topic}'"
        + (f" using subscription '{subscription}'." if subscription else " (subscription created on first use).")
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
