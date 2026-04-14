from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun

router = APIRouter(prefix="/workflows", tags=["workflows"])


# ─── Request models ───────────────────────────────────────────────────────────

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _langgraph_status(snap) -> str:
    return "waiting_approval" if snap.next else "completed"


def _get_runner(container: ApplicationContainer, graph_id: str):
    runner = container.yaml_graph_registry.get(graph_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{graph_id}' not found")
    return runner


def _run_response(run: GraphRun, workflow_name: str = "") -> dict:
    return {
        "id": run.id,
        "workflow_id": run.graph_id,
        "workflow_name": workflow_name or run.graph_id,
        "user_request": run.user_request,
        "status": run.status,
        "current_step": None,
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


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("")
async def list_workflows(container: ApplicationContainer = Depends(get_container)):
    return container.yaml_graph_registry.list_definitions()


@router.post("/runs")
async def start_run(
    body: RunRequest,
    container: ApplicationContainer = Depends(get_container),
):
    runner = _get_runner(container, body.workflow_id)
    thread_id = str(uuid4())

    run = GraphRun(
        id=thread_id,
        graph_id=body.workflow_id,
        user_request=body.user_request,
        status="running",
    )
    await container.run_repository.create(run)

    try:
        await runner.graph.ainvoke({"request": body.user_request}, _config(thread_id))
    except Exception:
        run.status = "failed"
        await container.run_repository.update(run)
        raise

    snap = runner.graph.get_state(_config(thread_id))
    run.status = _langgraph_status(snap)
    run.state = snap.values
    await container.run_repository.update(run)

    return _run_response(run, runner.name)


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = container.yaml_graph_registry.get(run.graph_id)
    return _run_response(run, runner.name if runner else "")


@router.post("/runs/{run_id}/approve")
async def approve_run(
    run_id: str,
    body: ApproveRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner(container, run.graph_id)

    await runner.graph.ainvoke(Command(resume={"approved": True}), _config(run_id))

    snap = runner.graph.get_state(_config(run_id))
    run.status = _langgraph_status(snap)
    run.state = snap.values
    await container.run_repository.update(run)

    return _run_response(run, runner.name)


@router.post("/runs/{run_id}/reject")
async def reject_run(
    run_id: str,
    body: RejectRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner(container, run.graph_id)

    await runner.graph.ainvoke(
        Command(resume={"approved": False, "reason": body.reason if body else None}),
        _config(run_id),
    )

    snap = runner.graph.get_state(_config(run_id))
    run.status = _langgraph_status(snap)
    run.state = snap.values
    await container.run_repository.update(run)

    return _run_response(run, runner.name)
