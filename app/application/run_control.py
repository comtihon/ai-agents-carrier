"""Shared implementations of run control: terminate / approve / reject / retry /
restart-from-step.

These were lifted from ``app.api.routes.workflows`` so the REST routes, the
internal chat agent and the ``/mcp/management`` MCP surface all drive a run
through exactly the same state transitions.

Each operation is split in two halves:

* a *pre-schedule* function (the ones below) that performs every check and
  every persisted state change, and
* a *continuation* (``_resume_approved`` / ``_resume_rejected`` /
  ``_retry_graph``) that streams the graph.

The split keeps REST semantics byte-identical: FastAPI schedules the
continuation as a ``BackgroundTask`` (i.e. after the response is sent), while
the MCP/agent surfaces schedule it with ``asyncio.create_task``.

Failures are raised as ``RunControlError`` carrying an HTTP status code: REST
translates it to ``HTTPException``, the tool surfaces to an error string.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langgraph.types import Command

from app.infrastructure.auth.authorization import Permission, missing_permission

if TYPE_CHECKING:
    from app.core.container import ApplicationContainer
    from app.domain.models.graph_run import GraphRun
    from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner

logger = logging.getLogger(__name__)


class RunControlError(Exception):
    """A run control operation was refused. Carries the REST status code."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _require_write() -> None:
    """Refuse a run-control operation the calling principal may not perform.

    Controlling a live run — terminating it, resuming it past an approval gate,
    replaying steps — is a mutation, so it needs WRITE. The REST routes already
    get that from their HTTP method; this check is what gives the same guarantee
    to the surfaces that reach these cores through a single ``POST``: the
    ``/mcp/management`` tools, whose transport wrapper checks only the ACCESS
    tier, and the chat agent's run-control tools.

    Raised rather than returned: every other refusal here is a
    ``RunControlError``, which REST turns into an HTTPException and the tool
    surfaces into an error string, so a 403 needs no new failure mode.

    An unbound principal (no authenticating wrapper ran — a Slack approval
    callback, a webhook, an in-process caller) is allowed; see
    ``missing_permission``.
    """
    if missing_permission(Permission.WRITE):
        raise RunControlError(
            403,
            "Missing 'write' permission: controlling a run requires a role that "
            "grants write access.",
        )


# ---------------------------------------------------------------------------
# The disabled-workflow guard
# ---------------------------------------------------------------------------
# One check, one message, every entry point. A workflow carrying enabled=False
# starts no runs at all: not from POST /workflows/runs, not from
# POST /webhooks/{workflow_id}, not from a cron or Pub/Sub trigger, not from the
# `run_workflow` MCP tool, not from a `workflow` step in another workflow, and
# not by replaying an existing run through retry / restart-from-step. Hiding the
# UI button is not enforcement; this is.
#
# 409 rather than 403: the caller is allowed to start this workflow and holds
# every permission the request needs — it is the workflow's own state that
# conflicts with it. Flipping the flag back makes the identical call from the
# identical principal succeed, which is a conflict, not an authorization
# failure.
WORKFLOW_DISABLED_STATUS = 409


def workflow_disabled_detail(workflow_id: str, name: str = "") -> str:
    """The single wording every surface reports for a disabled workflow."""
    label = workflow_id if not name or name == workflow_id else f"{workflow_id} ({name})"
    return (
        f"Workflow '{label}' is disabled and cannot be started. "
        "Enable it before starting a run."
    )


class WorkflowDisabledError(RunControlError):
    """A run was refused because its workflow is disabled.

    A RunControlError subclass so every surface that already translates one —
    REST into an HTTPException, the MCP tools into an error string — reports this
    refusal with no new failure mode.
    """

    def __init__(self, workflow_id: str, name: str = "") -> None:
        super().__init__(WORKFLOW_DISABLED_STATUS, workflow_disabled_detail(workflow_id, name))
        self.workflow_id = workflow_id


def ensure_workflow_enabled(workflow: Any) -> None:
    """Raise WorkflowDisabledError when *workflow* is disabled.

    Takes anything carrying ``id``/``name``/``enabled`` — a WorkflowDefinition
    from the backend or a YamlGraphRunner from the registry — so the
    backend-driven and registry-driven entry points share one guard. Missing
    attributes read as enabled, which keeps legacy definitions running.
    """
    if not getattr(workflow, "enabled", True):
        raise WorkflowDisabledError(
            str(getattr(workflow, "id", "?")), str(getattr(workflow, "name", "") or "")
        )


async def ensure_workflow_id_enabled(
    container: "ApplicationContainer", workflow_id: str
) -> None:
    """Guard by id, reading the *current* definition rather than a snapshot.

    Used where only the id is at hand (replaying an existing run). The stored
    definition wins over the registry copy because the registry is refreshed
    asynchronously and could still hold a pre-disable runner.
    """
    backend = getattr(container, "workflow_backend", None)
    if backend is not None:
        defn = await backend.get(workflow_id)
        if defn is not None:
            ensure_workflow_enabled(defn)
            return
    runner = container.yaml_graph_registry.get(workflow_id)
    if runner is not None:
        ensure_workflow_enabled(runner)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# TODO: layering wart — ``_stream_graph`` (and ``_get_runner_for_run``) still
# live in app.api.routes.workflows because _stream_graph is deeply entangled
# with the REST response helpers there.  They are imported lazily, inside the
# functions, so module import order stays one-way (workflows -> run_control)
# and there is no import cycle.  Move them into this layer separately.
def _runner_for_run(
    run: "GraphRun", container: "ApplicationContainer"
) -> "YamlGraphRunner | None":
    from app.api.routes.workflows import _get_runner_for_run
    return _get_runner_for_run(run, container)


def _require_runner(
    run: "GraphRun", container: "ApplicationContainer"
) -> "YamlGraphRunner":
    runner = _runner_for_run(run, container)
    if runner is None:
        raise RunControlError(
            404,
            f"Runner for workflow '{run.graph_id}' not found. "
            "The run may have been started before a server restart.",
        )
    return runner


# ---------------------------------------------------------------------------
# terminate
# ---------------------------------------------------------------------------

async def terminate_run(
    container: "ApplicationContainer", run_id: str
) -> "GraphRun":
    """Terminate a running agent and mark the run as failed."""
    _require_write()
    run = await container.run_repository.get(run_id)
    if run is None:
        raise RunControlError(404, "Run not found")
    if run.status not in ("running", "waiting_agent"):
        raise RunControlError(409, f"Run is not active (status: {run.status})")
    from app.services.agent_cleanup import cleanup_run_agents
    await cleanup_run_agents(run_id, container.settings)
    run.status = "failed"
    run.state = {**(run.state or {}), "error": "Terminated by user"}
    run.touch()
    await container.run_repository.update(run)
    container.live_runners.pop(run_id, None)
    return run


# ---------------------------------------------------------------------------
# approve / reject
# ---------------------------------------------------------------------------

async def _claim_for_resume(
    container: "ApplicationContainer", run_id: str
) -> "GraphRun":
    # Atomic claim. Two concurrent /approve requests must not both schedule
    # a resume task on the same runner — see claim_for_resume's docstring
    # for the failure mode that motivated this.
    run = await container.run_repository.claim_for_resume(run_id)
    if run is None:
        existing = await container.run_repository.get(run_id)
        if existing is None:
            raise RunControlError(404, "Run not found")
        raise RunControlError(
            409, f"Run is not awaiting approval (status: {existing.status})"
        )
    return run


async def approve_run(
    container: "ApplicationContainer", run_id: str
) -> "tuple[GraphRun, YamlGraphRunner]":
    """Claim a waiting run for approval; the caller schedules ``_resume_approved``."""
    _require_write()
    run = await _claim_for_resume(container, run_id)
    runner = _require_runner(run, container)

    # Flip the approval step to finished synchronously so polling clients see
    # the transition immediately, not after the resume task drains. The
    # subsequent step's status will be set by the chunk handler in
    # stream_graph_to_pause when the resumed graph reaches it.
    if run.current_step:
        run.step_statuses[run.current_step] = "finished"
        run.touch()
        await container.run_repository.update(run)
    return run, runner


async def reject_run(
    container: "ApplicationContainer", run_id: str
) -> "tuple[GraphRun, YamlGraphRunner]":
    """Claim a waiting run for rejection; the caller schedules ``_resume_rejected``."""
    _require_write()
    run = await _claim_for_resume(container, run_id)
    runner = _require_runner(run, container)

    if run.current_step:
        run.step_statuses[run.current_step] = "finished"
        run.touch()
        await container.run_repository.update(run)
    return run, runner


async def _resume_approved(
    runner: "YamlGraphRunner",
    run: "GraphRun",
    container: "ApplicationContainer",
    corrections: dict | None,
    approver_name: str | None = None,
    approver_id: str | None = None,
    approver_source: str = "ui",
) -> None:
    from app.api.routes.workflows import _stream_graph
    await _stream_graph(
        runner, run, container,
        Command(resume={
            "approved": True,
            "corrections": corrections,
            "approver_name": approver_name,
            "approver_id": approver_id,
            "approver_source": approver_source,
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }),
        base_url=container.settings.base_url,
    )
    if run.status in ("completed", "failed", "cancelled", "rejected"):
        container.live_runners.pop(run.id, None)


async def _resume_rejected(
    runner: "YamlGraphRunner",
    run: "GraphRun",
    container: "ApplicationContainer",
    reason: str | None,
    approver_name: str | None = None,
    approver_id: str | None = None,
    approver_source: str = "ui",
) -> None:
    from app.api.routes.workflows import _stream_graph
    await _stream_graph(
        runner, run, container,
        Command(resume={
            "approved": False,
            "reason": reason,
            "approver_name": approver_name,
            "approver_id": approver_id,
            "approver_source": approver_source,
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }),
    )
    if run.status == "completed":
        run.status = "rejected"
        run.touch()
        await container.run_repository.update(run)
    if run.status in ("cancelled", "rejected"):
        from app.services.agent_cleanup import cleanup_run_agents
        await cleanup_run_agents(run.id, container.settings)
    if run.status in ("completed", "failed", "cancelled", "rejected"):
        container.live_runners.pop(run.id, None)


# ---------------------------------------------------------------------------
# retry / restart-from-step
# ---------------------------------------------------------------------------

async def retry_run(
    container: "ApplicationContainer", run_id: str
) -> "tuple[GraphRun, YamlGraphRunner, Any]":
    """Retry a failed run from the last completed step, skipping already-finished steps.

    The caller schedules ``_retry_graph(runner, run, container, resume_input)``.
    """
    _require_write()
    run = await container.run_repository.get(run_id)
    if run is None:
        raise RunControlError(404, "Run not found")
    if run.status != "failed":
        raise RunControlError(409, f"Run is not in failed state (status={run.status})")
    # Retrying re-executes graph steps, so it is a start like any other and the
    # guard applies. Approve/reject deliberately do NOT check: a run already
    # paused at a human gate must stay closable, or disabling a workflow would
    # strand every in-flight approval with no way to finish or reject it.
    await ensure_workflow_id_enabled(container, run.graph_id)

    runner = container._build_runner_for_recovery(run)
    if runner is None:
        raise RunControlError(409, "Workflow definition not available for retry")

    # Reconstruct accumulated state from already-completed steps (in step order)
    accumulated: dict[str, Any] = {"request": run.user_request}
    last_done: str | None = None
    for step in runner.steps:
        sid = step["id"]
        if run.step_statuses.get(sid) in ("finished", "skipped"):
            last_done = sid
            output = run.step_outputs.get(sid)
            if output and isinstance(output, dict):
                accumulated.update(output)

    # Seed mid-execution state keys persisted before step completion (but NOT _visit_counts
    # so that retried runs start with a fresh loop counter).
    if run.state and isinstance(run.state, dict):
        for k, v in run.state.items():
            if k.startswith("_") and k != "_visit_counts" and v is not None:
                accumulated.setdefault(k, v)

    # Reset failed step and all subsequent steps back to "pending"
    found_failed = False
    for step in runner.steps:
        sid = step["id"]
        if not found_failed and run.step_statuses.get(sid) == "failed":
            found_failed = True
        if found_failed:
            run.step_statuses[sid] = "pending"

    # Seed the LangGraph checkpoint at the last completed step
    config = _config(run.id)
    if last_done is not None:
        await runner.graph.aupdate_state(config, accumulated, as_node=last_done)
        resume_input: Any = None  # resume from checkpoint
    else:
        resume_input = accumulated  # no completed steps — start fresh

    run.status = "running"
    run.current_step = None
    run.state = accumulated
    run.touch()
    await container.run_repository.update(run)

    container.live_runners[run.id] = runner
    return run, runner, resume_input


async def restart_from_step(
    container: "ApplicationContainer", run_id: str, step_id: str
) -> "tuple[GraphRun, YamlGraphRunner, Any]":
    """Restart a finished/failed run from *step_id*, discarding that step onwards.

    The caller schedules ``_retry_graph(runner, run, container, resume_input)``.
    """
    _require_write()
    run = await container.run_repository.get(run_id)
    if run is None:
        raise RunControlError(404, "Run not found")
    if run.status in ("cancelled", "rejected"):
        raise RunControlError(409, f"Cannot restart a {run.status} run")
    if run.status == "running":
        raise RunControlError(409, "Cannot restart a currently running workflow")
    # Same reasoning as retry_run: replaying steps starts the workflow again.
    await ensure_workflow_id_enabled(container, run.graph_id)

    runner = container._build_runner_for_recovery(run)
    if runner is None:
        raise RunControlError(409, "Workflow definition unavailable")

    step_ids = [s["id"] for s in runner.steps]
    if step_id not in step_ids:
        raise RunControlError(409, f"Unknown step_id: {step_id}")

    restart_idx = step_ids.index(step_id)

    # Terminate any live agent container via the runtime (best-effort).
    try:
        from app.runtime.docker import DockerRuntime
        await DockerRuntime(
            registry_username=container.settings.docker_registry_username,
            registry_password=container.settings.docker_registry_password,
        ).terminate_by_run_id(None, run_id)
    except Exception:
        logger.debug("run %s: docker cleanup on restart-from-step failed", run_id, exc_info=True)
    container.live_runners.pop(run_id, None)

    # Build accumulated state from steps BEFORE restart_idx
    accumulated: dict = {"request": run.user_request}
    # Seed underscore-prefixed internal state keys from run.state (e.g. conv IDs),
    # excluding _visit_counts so the restarted run gets a fresh loop counter.
    for k, v in (run.state or {}).items():
        if k.startswith("_") and k != "_visit_counts":
            accumulated.setdefault(k, v)
    last_done: str | None = None
    for step in runner.steps[:restart_idx]:
        sid = step["id"]
        if run.step_statuses.get(sid) in ("finished", "skipped"):
            output = run.step_outputs.get(sid) or {}
            for k, v in output.items():
                if k == "_visit_counts":
                    continue
                accumulated.setdefault(k, v)
            last_done = sid

    # Reset step_statuses, step_inputs, step_outputs for restart_idx and beyond
    for step in runner.steps[restart_idx:]:
        sid = step["id"]
        run.step_statuses[sid] = "pending"
        run.step_inputs.pop(sid, None)
        run.step_outputs.pop(sid, None)

    # Clear per-attempt transient keys so they don't suppress new notifications
    for _k in ("_slack_ask_context_ts", "_slack_ask_context_channel", "_pending_question"):
        accumulated.pop(_k, None)

    run.current_step = None
    run.agent_url = None
    run.status = "running"
    run.state = accumulated

    run.touch()
    await container.run_repository.update(run)

    # Re-seed LangGraph checkpoint
    config = {"configurable": {"thread_id": run_id}}
    if last_done is not None:
        await runner.graph.aupdate_state(config, accumulated, as_node=last_done)
        resume_input: Any = None
    else:
        resume_input = accumulated

    container.live_runners[run_id] = runner
    return run, runner, resume_input


async def _retry_graph(
    runner: "YamlGraphRunner",
    run: "GraphRun",
    container: "ApplicationContainer",
    resume_input: Any,
) -> None:
    from app.api.routes.workflows import _stream_graph
    await _stream_graph(runner, run, container, resume_input, base_url=container.settings.base_url)
    if run.status in ("completed", "failed", "cancelled", "rejected"):
        container.live_runners.pop(run.id, None)
