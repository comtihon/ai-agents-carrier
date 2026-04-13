from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun

router = APIRouter(prefix="/graphs", tags=["graphs"])


class RunRequest(BaseModel):
    request: str
    thread_id: str | None = None


class ApproveRequest(BaseModel):
    reason: str | None = None


class RejectRequest(BaseModel):
    reason: str | None = None


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _get_runner(container: ApplicationContainer, graph_id: str):
    runner = container.yaml_graph_registry.get(graph_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")
    return runner


def _langgraph_status(snap) -> str:
    if snap.next:
        return "waiting_approval"
    return "completed"


@router.get("")
async def list_graphs(container: ApplicationContainer = Depends(get_container)):
    return {"graphs": container.yaml_graph_registry.list_ids()}


@router.post("/{graph_id}/runs")
async def start_run(
    graph_id: str,
    body: RunRequest,
    container: ApplicationContainer = Depends(get_container),
):
    runner = _get_runner(container, graph_id)
    thread_id = body.thread_id or str(uuid4())

    run = GraphRun(id=thread_id, graph_id=graph_id, status="running")
    await container.run_repository.create(run)

    try:
        await runner.graph.ainvoke({"request": body.request}, _config(thread_id))
    except Exception:
        run.status = "failed"
        await container.run_repository.update(run)
        raise

    snap = runner.graph.get_state(_config(thread_id))
    run.status = _langgraph_status(snap)
    run.state = snap.values
    await container.run_repository.update(run)

    return {"graph_id": graph_id, "thread_id": thread_id, "status": run.status, "state": run.state}


@router.get("/{graph_id}/runs/{thread_id}")
async def get_run(
    graph_id: str,
    thread_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    _get_runner(container, graph_id)
    run = await container.run_repository.get(thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"graph_id": run.graph_id, "thread_id": run.id, "status": run.status, "state": run.state}


@router.post("/{graph_id}/runs/{thread_id}/approve")
async def approve_run(
    graph_id: str,
    thread_id: str,
    body: ApproveRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    runner = _get_runner(container, graph_id)
    run = await container.run_repository.get(thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    await runner.graph.ainvoke(Command(resume={"approved": True}), _config(thread_id))

    snap = runner.graph.get_state(_config(thread_id))
    run.status = _langgraph_status(snap)
    run.state = snap.values
    await container.run_repository.update(run)

    return {"graph_id": graph_id, "thread_id": thread_id, "status": run.status, "state": run.state}


@router.post("/{graph_id}/runs/{thread_id}/reject")
async def reject_run(
    graph_id: str,
    thread_id: str,
    body: RejectRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    runner = _get_runner(container, graph_id)
    run = await container.run_repository.get(thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    await runner.graph.ainvoke(
        Command(resume={"approved": False, "reason": body.reason if body else None}),
        _config(thread_id),
    )

    snap = runner.graph.get_state(_config(thread_id))
    run.status = _langgraph_status(snap)
    run.state = snap.values
    await container.run_repository.update(run)

    return {"graph_id": graph_id, "thread_id": thread_id, "status": run.status, "state": run.state}
