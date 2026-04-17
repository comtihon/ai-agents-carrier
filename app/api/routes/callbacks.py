from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.infrastructure.orchestration.yaml_graph import stream_graph_to_pause

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/callbacks", tags=["callbacks"])


class RejectCallbackBody(BaseModel):
    reason: str | None = None


@router.post("/{run_id}/approve")
async def callback_approve(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Approve a paused run. The run_id in the path acts as the auth token."""
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "waiting_approval":
        raise HTTPException(status_code=409, detail=f"Run is not awaiting approval (status: {run.status})")

    runner = container.live_runners.get(run_id) or container.yaml_graph_registry.get(run.graph_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Runner for workflow '{run.graph_id}' not found")

    run.status = "running"
    run.touch()
    await container.run_repository.update(run)

    await stream_graph_to_pause(runner, run, container.run_repository, Command(resume={"approved": True}))

    if run.status in ("completed", "failed", "cancelled"):
        container.live_runners.pop(run_id, None)

    logger.info("run %s: approved via callback", run_id)
    return {"run_id": run_id, "status": run.status}


@router.post("/{run_id}/reject")
async def callback_reject(
    run_id: str,
    body: RejectCallbackBody | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    """Reject a paused run. The run_id in the path acts as the auth token."""
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "waiting_approval":
        raise HTTPException(status_code=409, detail=f"Run is not awaiting approval (status: {run.status})")

    runner = container.live_runners.get(run_id) or container.yaml_graph_registry.get(run.graph_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Runner for workflow '{run.graph_id}' not found")

    run.status = "running"
    run.touch()
    await container.run_repository.update(run)

    reason = body.reason if body else None
    await stream_graph_to_pause(
        runner, run, container.run_repository,
        Command(resume={"approved": False, "reason": reason}),
    )

    if run.status == "completed":
        run.status = "cancelled"
        run.touch()
        await container.run_repository.update(run)

    if run.status in ("completed", "failed", "cancelled"):
        container.live_runners.pop(run_id, None)

    logger.info("run %s: rejected via callback (reason=%s)", run_id, reason)
    return {"run_id": run_id, "status": run.status}
