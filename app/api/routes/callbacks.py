from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from langgraph.types import Command
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.infrastructure.orchestration.yaml_graph import stream_graph_to_pause

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/callbacks", tags=["callbacks"])


class RejectCallbackBody(BaseModel):
    reason: str | None = None


def _html(title: str, emoji: str, body: str) -> HTMLResponse:
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh; margin: 0; background: #0f172a; color: #e2e8f0;
    }}
    .card {{
      text-align: center; padding: 2.5rem 3rem;
      background: #1e293b; border-radius: 1rem;
      border: 1px solid #334155; max-width: 400px;
    }}
    .emoji {{ font-size: 3rem; margin-bottom: 0.75rem; }}
    h1 {{ margin: 0 0 0.5rem; font-size: 1.4rem; color: #f1f5f9; }}
    p  {{ margin: 0; color: #94a3b8; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="emoji">{emoji}</div>
    <h1>{title}</h1>
    <p>{body}</p>
  </div>
</body>
</html>"""
    return HTMLResponse(content=content)


async def _do_approve(run_id: str, container: ApplicationContainer) -> str:
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
    return run.status


async def _do_reject(run_id: str, reason: str | None, container: ApplicationContainer) -> str:
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
    return run.status


# ── POST endpoints (called by machines / API clients) ─────────────────────────

@router.post("/{run_id}/approve")
async def callback_approve(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Approve a paused run. The run_id in the path acts as the auth token."""
    status = await _do_approve(run_id, container)
    return {"run_id": run_id, "status": status}


@router.post("/{run_id}/reject")
async def callback_reject(
    run_id: str,
    body: RejectCallbackBody | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    """Reject a paused run. The run_id in the path acts as the auth token."""
    reason = body.reason if body else None
    status = await _do_reject(run_id, reason, container)
    return {"run_id": run_id, "status": status}


# ── GET endpoints (opened in browser via Slack link buttons) ──────────────────

@router.get("/{run_id}/approve", response_class=HTMLResponse)
async def callback_approve_get(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Approve via browser link (e.g. Slack button URL). Returns a confirmation page."""
    try:
        await _do_approve(run_id, container)
    except HTTPException as exc:
        if exc.status_code == 409:
            return _html(
                "Already actioned", "ℹ️",
                "This run has already been approved or rejected.",
            )
        raise
    return _html("Approved", "✅", "The workflow run has been approved and will continue.")


@router.get("/{run_id}/reject", response_class=HTMLResponse)
async def callback_reject_get(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Reject via browser link (e.g. Slack button URL). Returns a confirmation page."""
    try:
        await _do_reject(run_id, reason=None, container=container)
    except HTTPException as exc:
        if exc.status_code == 409:
            return _html(
                "Already actioned", "ℹ️",
                "This run has already been approved or rejected.",
            )
        raise
    return _html("Rejected", "🚫", "The workflow run has been rejected and will not continue.")
