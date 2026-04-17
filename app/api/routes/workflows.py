from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.config.graph_loader import build_runner_from_definition
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflows"])

_STEP_TYPE_MAP: dict[str, str] = {
    "llm_structured": "llm",
    "llm": "llm",
    "mcp": "fetch",
    "human_approval": "approval",
    "execute": "execute",
    "workflow": "workflow",
    "cron": "cron",
    "http": "http",
}


# ─── Request / response models ────────────────────────────────────────────────

class RunRequest(BaseModel):
    workflow_id: str
    user_request: str
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict | None = None


class ApproveRequest(BaseModel):
    feedback: str | None = None


class RejectRequest(BaseModel):
    reason: str | None = None


class WorkflowDefinitionRequest(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    steps: list[dict[str, Any]] = []


class WorkflowDefinitionUpdateRequest(BaseModel):
    name: str = ""
    description: str = ""
    steps: list[dict[str, Any]] = []


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
    then falls back to the registry (post-completion or legacy test setup).
    """
    runner = container.live_runners.get(run.id)
    if runner is None:
        runner = container.yaml_graph_registry.get(run.graph_id)
    return runner


def _run_response(run: GraphRun, runner: YamlGraphRunner | None = None) -> dict:
    workflow_name, steps = _steps_from_definition(run, runner)
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
        "error": run.state.get("error") if run.status == "failed" else None,
        "metadata": {},
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }


def _init_step_statuses(runner: YamlGraphRunner) -> dict[str, str]:
    return {s["id"]: "pending" for s in runner.steps}


def _step_status_for_output(output: dict) -> str:
    return "skipped" if output == {} else "finished"


async def _stream_graph(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
    input_value: Any,
    base_url: str | None = None,
) -> None:
    """Stream graph execution, updating step statuses in DB after each node."""
    step_ids = [s["id"] for s in runner.steps]

    if isinstance(input_value, dict):
        current_state: dict = dict(input_value)
    else:
        try:
            snap = runner.graph.get_state(_config(run.id))
            current_state = dict(snap.values) if snap.values else {}
        except Exception:
            current_state = {}

    try:
        async for chunk in runner.graph.astream(
            input_value, _config(run.id), stream_mode="updates",
        ):
            for node_name, output in chunk.items():
                if node_name in ("__start__", "__end__"):
                    continue
                status = _step_status_for_output(output)
                run.step_inputs[node_name] = dict(current_state)
                run.step_statuses[node_name] = status
                run.current_step = node_name
                if output:
                    run.step_outputs[node_name] = output
                    if isinstance(output, dict):
                        current_state.update(output)
                logger.info("run %s: step '%s' → %s", run.id, node_name, status)
                run.touch()
                await container.run_repository.update(run)
    except Exception as exc:
        logger.exception("run %s: graph execution failed", run.id)
        for sid in step_ids:
            if run.step_statuses.get(sid) == "pending":
                run.step_inputs[sid] = dict(current_state)
                run.step_statuses[sid] = "failed"
                break
        run.status = "failed"
        run.state = {"error": str(exc)}
        run.current_step = None
        run.touch()
        await container.run_repository.update(run)
        return

    snap = runner.graph.get_state(_config(run.id))
    run.status = _langgraph_status(snap)
    run.current_step = snap.next[0] if snap.next else None
    run.state = snap.values
    run.touch()
    await container.run_repository.update(run)

    if run.status == "waiting_approval" and base_url and run.current_step:
        from app.infrastructure.notifications.webhook_notifier import send_approval_notification
        step = next((s for s in runner.steps if s["id"] == run.current_step), None)
        if step and step.get("notify"):
            await send_approval_notification(step["notify"], run.id, snap.values, base_url)


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
    )
    saved = await container.workflow_backend.create(defn)
    await container.refresh_runner(body.id)
    return saved.model_dump(mode="json")


# ─── Run endpoints ────────────────────────────────────────────────────────────
# IMPORTANT: these routes MUST be registered before /{workflow_id} routes so
# that Starlette does not match "runs" as a {workflow_id} path parameter.

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
            mcp_tools_provider=container.mcp_tools_provider,
            registry=container.yaml_graph_registry,
            run_repository=container.run_repository,
            openhands=container.openhands,
        )
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

    return _run_response(run, runner)


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner_for_run(run, container)
    return _run_response(run, runner)


@router.post("/runs/{run_id}/approve")
async def approve_run(
    run_id: str,
    body: ApproveRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = await _get_runner_or_404(run, container)

    run.status = "running"
    run.touch()
    await container.run_repository.update(run)

    await _stream_graph(runner, run, container, Command(resume={"approved": True}))

    if run.status in ("completed", "failed", "cancelled"):
        container.live_runners.pop(run_id, None)

    return _run_response(run, runner)


@router.post("/runs/{run_id}/reject")
async def reject_run(
    run_id: str,
    body: RejectRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = await _get_runner_or_404(run, container)

    run.status = "running"
    run.touch()
    await container.run_repository.update(run)

    await _stream_graph(
        runner, run, container,
        Command(resume={"approved": False, "reason": body.reason if body else None}),
    )

    # Override: if no more gates remain after rejection, mark as cancelled
    if run.status == "completed":
        run.status = "cancelled"
        run.touch()
        await container.run_repository.update(run)

    if run.status in ("completed", "failed", "cancelled"):
        container.live_runners.pop(run_id, None)

    return _run_response(run, runner)


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
