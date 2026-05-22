from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.config.graph_loader import build_runner_from_definition
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.tracing.callback_handler import RunTraceAccumulator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflows"])

_STEP_TYPE_MAP: dict[str, str] = {
    "llm_structured": "llm",  # DEPRECATED: use langgraph-agent or claude-agent instead
    "llm": "llm",
    "mcp": "fetch",
    "human_approval": "approval",
    "execute": "execute",
    "workflow": "workflow",
    "cron": "cron",
    "http": "http",
    "langgraph-agent": "agent",
    "claude-agent": "agent",
}


# ─── Request / response models ────────────────────────────────────────────────

class RunRequest(BaseModel):
    workflow_id: str
    user_request: str
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict | None = None


class ApproveRequest(BaseModel):
    corrections: dict[str, Any] | None = None


class RejectRequest(BaseModel):
    reason: str | None = None


class WorkflowDefinitionRequest(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    steps: list[dict[str, Any]] = []
    ui: dict[str, Any] = Field(default_factory=dict)


class WorkflowDefinitionUpdateRequest(BaseModel):
    name: str = ""
    description: str = ""
    steps: list[dict[str, Any]] = []
    ui: dict[str, Any] = Field(default_factory=dict)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _langgraph_status(snap) -> str:
    return "waiting_approval" if snap.next else "completed"


def _require_backend(container: ApplicationContainer) -> None:
    if container.workflow_backend is None:
        raise HTTPException(status_code=501, detail="Workflow backend not configured")


def _steps_from_definition(
    run: GraphRun,
    runner: YamlGraphRunner | None,
) -> tuple[str, list[dict]]:
    """Return (workflow_name, steps_list) for a run response.

    Priority:
    1. Live runner (still in memory with step definitions).
    2. Definition snapshot stored in the run at start time.
    3. Fallback: empty steps with graph_id as name.
    """
    if runner is not None:
        name = runner.name
        steps = [
            {
                "id": s["id"],
                "type": _STEP_TYPE_MAP.get(s.get("type", "llm"), s.get("type", "llm")),
                "name": s.get("name", s["id"]),
                "status": run.step_statuses.get(s["id"], "pending"),
                "input": run.step_inputs.get(s["id"]),
                "output": run.step_outputs.get(s["id"]),
            }
            for s in runner.steps
        ]
        return name, steps

    if run.workflow_definition:
        raw = run.workflow_definition
        raw_name: str = raw.get("name") or ""
        name = raw_name or raw["id"].replace("-", " ").replace("_", " ").title()
        steps = [
            {
                "id": s["id"],
                "type": _STEP_TYPE_MAP.get(s.get("type", "llm"), s.get("type", "llm")),
                "name": s.get("name", s["id"]),
                "status": run.step_statuses.get(s["id"], "pending"),
                "input": run.step_inputs.get(s["id"]),
                "output": run.step_outputs.get(s["id"]),
            }
            for s in raw.get("steps", [])
        ]
        return name, steps

    return run.graph_id, []


def _get_runner_for_run(run: GraphRun, container: ApplicationContainer) -> YamlGraphRunner | None:
    """Return the runner for an existing run.

    Checks live_runners first (in-flight run with its original definition),
    then falls back to the registry only when no definition snapshot exists.
    When a snapshot is stored we skip the registry fallback so that a newer
    definition (updated after the run started) does not leak into the response.
    """
    runner = container.live_runners.get(run.id)
    if runner is not None:
        return runner
    if run.workflow_definition is None:
        return container.yaml_graph_registry.get(run.graph_id)
    return None


async def _get_interrupt_payload(runner: YamlGraphRunner | None, run: GraphRun) -> dict:
    """Extract the rendered interrupt payload from the paused LangGraph snapshot."""
    if run.status != "waiting_approval":
        return {}
    # Primary: use the __interrupt__ chunk persisted to step_outputs during streaming
    raw = (run.step_outputs or {}).get("__interrupt__")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("value"), dict):
                return item["value"]
    # Fallback: query the MongoDB checkpoint (covers cases where step_outputs is absent)
    if runner is not None:
        try:
            snap = await runner.graph.aget_state(_config(run.id))
            for task in snap.tasks:
                for intr in task.interrupts:
                    if isinstance(intr.value, dict):
                        return intr.value
            for intr in getattr(snap, "interrupts", ()):
                if isinstance(intr.value, dict):
                    return intr.value
        except Exception:
            logger.exception("run %s: failed to read interrupt from checkpoint", run.id)
    return {}


async def _run_response(run: GraphRun, runner: YamlGraphRunner | None = None) -> dict:
    workflow_name, steps = _steps_from_definition(run, runner)
    interrupt_payload = await _get_interrupt_payload(runner, run)
    return {
        "id": run.id,
        "workflow_id": run.graph_id,
        "workflow_name": workflow_name,
        "user_request": run.user_request,
        "status": run.status,
        "current_step": run.current_step,
        "steps": steps,
        "approval_status": "pending" if run.status == "waiting_approval" else "not_required",
        "approval_gates": [],
        "plan": None,
        "tool_call_results": [],
        "llm_agent_results": [],
        "action_results": [],
        "execution_results": [],
        "intermediate_outputs": run.state,
        "interrupt_payload": interrupt_payload,
        "waiting_transition": (
            run.waiting_transition.model_dump(mode="json")
            if run.waiting_transition is not None
            else None
        ),
        "error": run.state.get("error") if run.status == "failed" else None,
        "metadata": {},
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }


def _init_step_statuses(runner: YamlGraphRunner) -> dict[str, str]:
    return {s["id"]: "pending" for s in runner.steps}


def _step_status_for_output(node_name: str, output: dict) -> str:
    from app.infrastructure.orchestration.yaml_graph import step_status_from_output
    return step_status_from_output(node_name, output)


async def _stream_graph(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
    input_value: Any,
    base_url: str | None = None,
) -> None:
    """Stream graph execution, updating step statuses in DB after each node."""
    step_ids = [s["id"] for s in runner.steps]

    # Bind this run to the runner so node-body helpers
    # (`_wrap_with_status_running`, `_save_conv_id`) write through to the
    # *current* run. Without this binding, those helpers reused whatever
    # run reference was last set on the runner — typically a stale one
    # left behind by `stream_graph_to_pause` during the post-restart
    # recovery path — and their saves would clobber the live run with
    # the stale state (e.g. flipping status back to waiting_approval
    # mid-stream after the user had already approved).
    runner._current_run = run
    runner._current_run_repository = container.run_repository

    if isinstance(input_value, dict):
        current_state: dict = dict(input_value)
    else:
        try:
            snap = await runner.graph.aget_state(_config(run.id))
            current_state = dict(snap.values) if snap.values else {}
        except Exception:
            current_state = {}

    # Set up local trace accumulator (works regardless of LangSmith configuration)
    trace_accumulator = RunTraceAccumulator()
    stream_started_at = datetime.now(timezone.utc)

    # Build a streaming config that includes our trace callback
    stream_config = {
        **_config(run.id),
        "callbacks": [trace_accumulator],
    }

    def _persist_trace(extra_errors: list[str] | None = None) -> None:
        elapsed = (datetime.now(timezone.utc) - stream_started_at).total_seconds() * 1000
        td = trace_accumulator.to_trace_data(latency_ms=elapsed)
        if extra_errors:
            td["errors"].extend(extra_errors)
        run.trace_data = td

    # Track the last node fully handled in this call to identify the failing step precisely.
    last_processed: str | None = None
    try:
        async for chunk in runner.graph.astream(
            input_value, stream_config, stream_mode="updates",
        ):
            for node_name, output in chunk.items():
                if node_name in ("__start__", "__end__"):
                    continue
                status = _step_status_for_output(node_name, output)
                run.step_inputs[node_name] = dict(current_state)
                run.step_statuses[node_name] = status
                run.current_step = node_name
                if output:
                    run.step_outputs[node_name] = output
                    if isinstance(output, dict):
                        current_state.update(output)
                logger.info("run %s: step '%s' → %s", run.id, node_name, status)
                last_processed = node_name
                # Persist partial trace data after each step so polling clients
                # see live data during execution.
                _persist_trace()
                run.touch()
                await container.run_repository.update(run)
    except Exception as exc:
        logger.exception("run %s: graph execution failed", run.id)
        # Attribute the failure only to a step we can identify with confidence:
        # one currently "running" (its wrapper marked it so before raising) or
        # one tagged via the __failed_step__ sentinel. Anything else (e.g. a
        # framework-level recursion limit) leaves step_statuses untouched —
        # promoting the next forward step to "failed" misleads the UI when
        # the failure was inside a retry loop, not on a yet-to-run node.
        running_sid = next(
            (sid for sid, st in run.step_statuses.items() if st == "running"),
            None,
        )
        if running_sid is not None:
            run.step_inputs[running_sid] = dict(current_state)
            run.step_statuses[running_sid] = "failed"
        else:
            failed_sid = current_state.get("__failed_step__") if isinstance(current_state, dict) else None
            if isinstance(failed_sid, str) and failed_sid in run.step_statuses:
                run.step_inputs[failed_sid] = dict(current_state)
                run.step_statuses[failed_sid] = "failed"
        run.status = "failed"
        # Preserve accumulated step outputs so retry can recover OpenHands conv IDs
        # and other intermediate state without needing to re-run completed steps.
        run.state = {**current_state, "error": str(exc)}
        run.current_step = None
        _persist_trace(extra_errors=[str(exc)])
        run.touch()
        await container.run_repository.update(run)
        return

    snap = await runner.graph.aget_state(_config(run.id))
    run.status = _langgraph_status(snap)
    run.current_step = snap.next[0] if snap.next else None
    run.state = snap.values
    _persist_trace()
    run.touch()
    await container.run_repository.update(run)

    if run.status == "waiting_approval" and base_url and run.current_step:
        step = next((s for s in runner.steps if s["id"] == run.current_step), None)
        if step and step.get("type") == "ask_context":
            from app.core.config import get_settings
            from app.infrastructure.notifications.webhook_notifier import (
                post_slack_ask_context, post_slack_thread_questions,
            )
            settings = get_settings()
            if settings.slack_bot_token and settings.slack_approvals_channel:
                questions: list[str] = []
                for task in snap.tasks:
                    for intr in task.interrupts:
                        if isinstance(intr.value, dict) and intr.value.get("type") == "ask_context":
                            questions = intr.value.get("questions", [])
                if not questions:
                    for intr in getattr(snap, "interrupts", ()):
                        if isinstance(intr.value, dict) and intr.value.get("type") == "ask_context":
                            questions = intr.value.get("questions", [])
                existing_ts = snap.values.get("_slack_ask_context_ts")
                existing_channel = snap.values.get("_slack_ask_context_channel")
                if questions:
                    if existing_ts and existing_channel:
                        # Loop-back: post new questions as a reply in the existing thread
                        await post_slack_thread_questions(
                            settings.slack_bot_token, existing_channel, existing_ts, questions,
                        )
                    else:
                        notif_resp = await post_slack_ask_context(
                            settings.slack_bot_token, settings.slack_approvals_channel,
                            questions, run.id, snap.values,
                        )
                        if notif_resp and notif_resp.get("ok"):
                            ts = notif_resp.get("ts")
                            channel = notif_resp.get("channel")
                            if ts and channel:
                                await runner.graph.aupdate_state(_config(run.id), {
                                    "_slack_ask_context_ts": ts,
                                    "_slack_ask_context_channel": channel,
                                })
                                run.state = {**run.state, "_slack_ask_context_ts": ts, "_slack_ask_context_channel": channel}
                                run.touch()
                                await container.run_repository.update(run)
        elif step and step.get("notify"):
            from app.infrastructure.notifications.webhook_notifier import send_approval_notification
            notif_resp = await send_approval_notification(step["notify"], run.id, snap.values, base_url)
            if notif_resp and notif_resp.get("ok"):
                ts = notif_resp.get("ts")
                channel = notif_resp.get("channel")
                if ts and channel:
                    await runner.graph.aupdate_state(_config(run.id), {"_slack_thread_ts": ts, "_slack_channel": channel})
                    run.state = {**run.state, "_slack_thread_ts": ts, "_slack_channel": channel}
                    run.touch()
                    await container.run_repository.update(run)


async def _execute_graph(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
) -> None:
    """Run the graph to its first interrupt (or completion) and persist the result."""
    run.step_statuses = _init_step_statuses(runner)
    await _stream_graph(runner, run, container, {"request": run.user_request}, base_url=container.settings.base_url)
    # Remove from live_runners when run reaches a terminal or waiting state.
    # waiting_approval is kept — resume needs the runner with its MemorySaver state.
    if run.status in ("completed", "failed", "cancelled"):
        container.live_runners.pop(run.id, None)


async def _get_runner_or_404(
    run: GraphRun, container: ApplicationContainer
) -> YamlGraphRunner:
    runner = _get_runner_for_run(run, container)
    if runner is None:
        raise HTTPException(
            status_code=404,
            detail=f"Runner for workflow '{run.graph_id}' not found. "
                   "The run may have been started before a server restart.",
        )
    return runner


# ─── Workflow definition list + create (no path param — must come before run routes) ─────

@router.get("")
async def list_workflows(container: ApplicationContainer = Depends(get_container)):
    """List all workflows (summary view from the registry)."""
    return container.yaml_graph_registry.list_definitions()


@router.post("", status_code=201)
async def create_workflow(
    body: WorkflowDefinitionRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Create a new workflow definition and store it in the backend."""
    _require_backend(container)
    assert container.workflow_backend is not None

    existing = await container.workflow_backend.get(body.id)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Workflow '{body.id}' already exists")

    defn = WorkflowDefinition(
        id=body.id,
        name=body.name,
        description=body.description,
        steps=body.steps,
        ui=body.ui,
    )
    saved = await container.workflow_backend.create(defn)
    await container.refresh_runner(body.id)
    return saved.model_dump(mode="json")


# ─── Run endpoints ────────────────────────────────────────────────────────────
# IMPORTANT: these routes MUST be registered before /{workflow_id} routes so
# that Starlette does not match "runs" as a {workflow_id} path parameter.

@router.get("/runs")
async def list_runs(
    limit: int = 50,
    workflow_id: str | None = None,
    status: str | None = None,
    search: str | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    """List recent workflow runs, newest first.

    Optional filters: workflow_id, status (running|waiting_approval|completed|failed|cancelled),
    search (case-insensitive substring match on user_request).
    """
    runs = await container.run_repository.list_recent(
        limit=limit, workflow_id=workflow_id, status=status, search=search
    )
    responses = []
    for run in runs:
        responses.append(await _run_response(run, _get_runner_for_run(run, container)))
    return responses


@router.post("/runs")
async def start_run(
    body: RunRequest,
    background_tasks: BackgroundTasks,
    container: ApplicationContainer = Depends(get_container),
):
    """Start a new workflow run.

    Fetches the latest workflow definition from the backend (if configured),
    builds a fresh runner, and stores it in live_runners for the duration of
    the run.  The definition is also snapshotted into GraphRun so that
    approval-resume always uses the same version.
    """
    if container.workflow_backend is not None:
        defn = await container.workflow_backend.get(body.workflow_id)
        if defn is None:
            raise HTTPException(
                status_code=404, detail=f"Workflow '{body.workflow_id}' not found"
            )
        runner = build_runner_from_definition(
            defn,
            llm=container.llm,
            llm_factory=container.llm_factory,
            mcp_tools_provider=container.mcp_tools_provider,
            registry=container.yaml_graph_registry,
            run_repository=container.run_repository,
            openhands=container.openhands,
            checkpointer=container.checkpointer,
        )
        if container.agent_backend is not None:
            runner._agent_backend = container.agent_backend
        definition_snapshot: dict | None = defn.to_raw_dict()
    else:
        # Legacy path: no backend configured, use registry directly (tests).
        runner = container.yaml_graph_registry.get(body.workflow_id)
        if runner is None:
            raise HTTPException(
                status_code=404, detail=f"Workflow '{body.workflow_id}' not found"
            )
        definition_snapshot = None

    thread_id = str(uuid4())
    container.live_runners[thread_id] = runner

    run = GraphRun(
        id=thread_id,
        graph_id=body.workflow_id,
        user_request=body.user_request,
        status="running",
        workflow_definition=definition_snapshot,
    )
    await container.run_repository.create(run)
    background_tasks.add_task(_execute_graph, runner, run, container)

    return await _run_response(run, runner)


@router.get("/runs/{run_id}/trace")
async def get_run_trace(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Return the trace data for a run (LLM calls, tool calls, token usage, latency, errors).

    Supports live polling — partial data is returned while the run is still in progress.
    If LANGCHAIN_TRACING_V2 is enabled and a LangSmith run ID is recorded,
    a link to the LangSmith UI is included in the response.
    """
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    langsmith_url: str | None = None
    if run.langsmith_run_id:
        project = os.environ.get("LANGCHAIN_PROJECT", "default")
        langsmith_url = (
            f"https://smith.langchain.com/projects/{project}/runs/{run.langsmith_run_id}"
        )

    return {
        "run_id": run_id,
        "status": run.status,
        "trace_data": run.trace_data or {},
        "langsmith_run_id": run.langsmith_run_id,
        "langsmith_url": langsmith_url,
    }


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner_for_run(run, container)
    return await _run_response(run, runner)


@router.post("/runs/{run_id}/approve")
async def approve_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    body: ApproveRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    # Atomic claim. Two concurrent /approve requests must not both schedule
    # a resume task on the same runner — see claim_for_resume's docstring
    # for the failure mode that motivated this.
    run = await container.run_repository.claim_for_resume(run_id)
    if run is None:
        existing = await container.run_repository.get(run_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Run not found")
        raise HTTPException(
            status_code=409,
            detail=f"Run is not awaiting approval (status: {existing.status})",
        )
    runner = await _get_runner_or_404(run, container)

    # Flip the approval step to finished synchronously so polling clients see
    # the transition immediately, not after the resume task drains. The
    # subsequent step's status will be set by the chunk handler in
    # stream_graph_to_pause when the resumed graph reaches it.
    if run.current_step:
        run.step_statuses[run.current_step] = "finished"
        run.touch()
        await container.run_repository.update(run)

    corrections = body.corrections if body else None
    background_tasks.add_task(_resume_approved, runner, run, container, corrections)

    return await _run_response(run, runner)


async def _resume_approved(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
    corrections: dict | None,
) -> None:
    await _stream_graph(
        runner, run, container,
        Command(resume={"approved": True, "corrections": corrections}),
        base_url=container.settings.base_url,
    )
    if run.status in ("completed", "failed", "cancelled"):
        container.live_runners.pop(run.id, None)


@router.post("/runs/{run_id}/reject")
async def reject_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    body: RejectRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.claim_for_resume(run_id)
    if run is None:
        existing = await container.run_repository.get(run_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Run not found")
        raise HTTPException(
            status_code=409,
            detail=f"Run is not awaiting approval (status: {existing.status})",
        )
    runner = await _get_runner_or_404(run, container)

    if run.current_step:
        run.step_statuses[run.current_step] = "finished"
        run.touch()
        await container.run_repository.update(run)

    background_tasks.add_task(_resume_rejected, runner, run, container, body.reason if body else None)

    return await _run_response(run, runner)


async def _resume_rejected(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
    reason: str | None,
) -> None:
    await _stream_graph(
        runner, run, container,
        Command(resume={"approved": False, "reason": reason}),
    )
    if run.status == "completed":
        run.status = "cancelled"
        run.touch()
        await container.run_repository.update(run)
    if run.status in ("completed", "failed", "cancelled"):
        container.live_runners.pop(run.id, None)


@router.post("/runs/{run_id}/retry")
async def retry_run(
    run_id: str,
    background_tasks: BackgroundTasks,
    container: ApplicationContainer = Depends(get_container),
):
    """Retry a failed run from the last completed step, skipping already-finished steps."""
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "failed":
        raise HTTPException(
            status_code=409, detail=f"Run is not in failed state (status={run.status})"
        )

    runner = container._build_runner_for_recovery(run)
    if runner is None:
        raise HTTPException(
            status_code=409, detail="Workflow definition not available for retry"
        )

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
    background_tasks.add_task(_retry_graph, runner, run, container, resume_input)

    return await _run_response(run, runner)


async def _retry_graph(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
    resume_input: Any,
) -> None:
    await _stream_graph(runner, run, container, resume_input, base_url=container.settings.base_url)
    if run.status in ("completed", "failed", "cancelled"):
        container.live_runners.pop(run.id, None)


# ─── Workflow definition detail / update / delete ─────────────────────────────
# These routes use {workflow_id} path parameters and MUST be registered AFTER
# the /runs/... routes above so Starlette does not match "runs" as a {workflow_id}.

@router.get("/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Get the full definition of a workflow."""
    _require_backend(container)
    assert container.workflow_backend is not None

    defn = await container.workflow_backend.get(workflow_id)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return defn.model_dump(mode="json")


@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    body: WorkflowDefinitionUpdateRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Update an existing workflow definition.

    In-flight runs are not affected — they continue with the definition
    snapshot captured at run-start time.  New runs will use this version.
    """
    _require_backend(container)
    assert container.workflow_backend is not None

    existing = await container.workflow_backend.get(workflow_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    if existing.readonly:
        raise HTTPException(status_code=403, detail=f"Workflow '{workflow_id}' is read-only")

    defn = WorkflowDefinition(
        id=workflow_id,
        name=body.name,
        description=body.description,
        steps=body.steps,
        ui=body.ui,
        created_at=existing.created_at,
    )
    saved = await container.workflow_backend.update(workflow_id, defn)
    await container.refresh_runner(workflow_id)
    return saved.model_dump(mode="json")


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Delete a workflow definition."""
    _require_backend(container)
    assert container.workflow_backend is not None

    existing = await container.workflow_backend.get(workflow_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    if existing.readonly:
        raise HTTPException(status_code=403, detail=f"Workflow '{workflow_id}' is read-only")

    await container.workflow_backend.delete(workflow_id)
    await container.refresh_runner(workflow_id)
