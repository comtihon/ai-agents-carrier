from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner

router = APIRouter(prefix="/workflows", tags=["workflows"])

# ── Request / response models ─────────────────────────────────────────────────

class WorkflowRunRequest(BaseModel):
    workflow_id: str
    user_request: str
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict | None = None


class ApproveRequest(BaseModel):
    feedback: str | None = None


class RejectRequest(BaseModel):
    reason: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

# Map YAML step types → frontend-friendly step types
_STEP_TYPE_MAP: dict[str, str] = {
    "llm_structured": "llm",
    "llm": "llm",
    "mcp": "fetch",
    "human_approval": "approval",
    "execute": "execute",
}


def _runner_to_definition(runner: YamlGraphRunner) -> dict:
    steps = [
        {
            "id": step["id"],
            "type": _STEP_TYPE_MAP.get(step["type"], step["type"]),
            "name": step.get("id"),
        }
        for step in runner._steps
    ]
    return {
        "id": runner.id,
        "name": runner.name,
        "description": runner.description,
        "steps": steps,
    }


def _get_runner(container: ApplicationContainer, graph_id: str) -> YamlGraphRunner:
    runner = container.yaml_graph_registry.get(graph_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{graph_id}' not found")
    return runner


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _snap_status(snap) -> str:
    return "waiting_approval" if snap.next else "completed"


def _run_to_response(run: GraphRun, workflow_name: str = "") -> dict:
    state = run.state or {}
    user_request = state.get("request", "")
    current_step: str | None = state.get("_current_step")

    # Determine approval_status
    if run.status == "waiting_approval":
        approval_status = "pending"
    elif run.status == "completed":
        approved = state.get("approved")
        if approved is True:
            approval_status = "approved"
        elif approved is False:
            approval_status = "rejected"
        else:
            approval_status = "not_required"
    else:
        approval_status = "not_required"

    # Build synthetic approval gate so the UI renders the approval panel
    approval_gates: list[dict] = []
    if run.status == "waiting_approval":
        gate_id = current_step or "approval"
        approval_gates = [
            {
                "id": gate_id,
                "step_id": gate_id,
                "status": "pending",
                "created_at": run.created_at.isoformat(),
                "metadata": {
                    "interrupt_payload": state.get("_interrupt_payload"),
                },
            }
        ]

    return {
        "id": run.id,
        "workflow_id": run.graph_id,
        "workflow_name": workflow_name,
        "user_request": user_request,
        "status": run.status,
        "current_step": current_step,
        "approval_status": approval_status,
        "approval_gates": approval_gates,
        "plan": None,
        "tool_call_results": [],
        "llm_agent_results": [],
        "action_results": [],
        "execution_results": [],
        "intermediate_outputs": {
            k: v for k, v in state.items() if not k.startswith("_")
        },
        "error": state.get("error"),
        "metadata": {},
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }


async def _invoke_and_persist(
    runner: YamlGraphRunner,
    run: GraphRun,
    input_: dict | Command,
    container: ApplicationContainer,
) -> GraphRun:
    """Run ainvoke, capture state/status, persist, and return updated run."""
    config = _config(run.id)
    try:
        await runner.graph.ainvoke(input_, config)
    except Exception as exc:
        run.status = "failed"
        run.state = {**run.state, "error": str(exc)}
        await container.run_repository.update(run)
        raise

    snap = runner.graph.get_state(config)
    run.status = _snap_status(snap)

    # Merge graph values; stash current interrupted step and any interrupt payload
    current_step = snap.next[0] if snap.next else None
    interrupt_payload = None
    if snap.tasks:
        # LangGraph stores the interrupt value in the first pending task's interrupts
        task_interrupts = getattr(snap.tasks[0], "interrupts", None)
        if task_interrupts:
            interrupt_payload = getattr(task_interrupts[0], "value", None)

    run.state = {
        **snap.values,
        "_current_step": current_step,
        "_interrupt_payload": interrupt_payload,
    }
    await container.run_repository.update(run)
    return run


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("")
async def list_workflows(container: ApplicationContainer = Depends(get_container)):
    """List all available YAML-defined workflows with full metadata."""
    runners = [
        container.yaml_graph_registry.get(gid)
        for gid in container.yaml_graph_registry.list_ids()
    ]
    return [_runner_to_definition(r) for r in runners if r is not None]


@router.post("/runs")
async def submit_run(
    body: WorkflowRunRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Start a new workflow run."""
    runner = _get_runner(container, body.workflow_id)
    thread_id = str(uuid4())

    run = GraphRun(
        id=thread_id,
        graph_id=body.workflow_id,
        status="running",
        # Pre-populate request so it's available even on early failure
        state={"request": body.user_request},
    )
    await container.run_repository.create(run)

    run = await _invoke_and_persist(
        runner, run, {"request": body.user_request}, container
    )

    return _run_to_response(run, workflow_name=runner.name)


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Fetch the current state of a workflow run."""
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    runner = container.yaml_graph_registry.get(run.graph_id)
    workflow_name = runner.name if runner else run.graph_id
    return _run_to_response(run, workflow_name=workflow_name)


@router.post("/runs/{run_id}/approve")
async def approve_run(
    run_id: str,
    body: ApproveRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    """Approve a run that is waiting for human approval."""
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner(container, run.graph_id)

    run = await _invoke_and_persist(
        runner, run, Command(resume={"approved": True}), container
    )
    return _run_to_response(run, workflow_name=runner.name)


@router.post("/runs/{run_id}/reject")
async def reject_run(
    run_id: str,
    body: RejectRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    """Reject a run that is waiting for human approval."""
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner(container, run.graph_id)

    reason = body.reason if body else None
    run = await _invoke_and_persist(
        runner, run, Command(resume={"approved": False, "reason": reason}), container
    )
    return _run_to_response(run, workflow_name=runner.name)


@router.post("/runs/{run_id}/gates/{gate_id}/approve")
async def approve_gate(
    run_id: str,
    gate_id: str,
    body: ApproveRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    """Approve a specific approval gate (delegates to run-level approve)."""
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner(container, run.graph_id)

    run = await _invoke_and_persist(
        runner, run, Command(resume={"approved": True}), container
    )
    return _run_to_response(run, workflow_name=runner.name)


@router.post("/runs/{run_id}/gates/{gate_id}/reject")
async def reject_gate(
    run_id: str,
    gate_id: str,
    body: RejectRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    """Reject a specific approval gate (delegates to run-level reject)."""
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner(container, run.graph_id)

    reason = body.reason if body else None
    run = await _invoke_and_persist(
        runner, run, Command(resume={"approved": False, "reason": reason}), container
    )
    return _run_to_response(run, workflow_name=runner.name)
