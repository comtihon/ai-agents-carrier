"""Live view of concurrency-limited agent runtimes (app.application.concurrency).

    GET /agent-runtimes/concurrency[?key=<runtime key>]
        → [{key, kind, runner_id, acp_agent, agent_id, limit, holders: [...], queue: [...]}]
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.application.concurrency import describe_for_caller
from app.core.container import ApplicationContainer

router = APIRouter(prefix="/agent-runtimes", tags=["agents"])


@router.get("/concurrency")
async def list_concurrency(key: str | None = None, container: ApplicationContainer = Depends(get_container)):
    """Who holds and who waits for each limited agent runtime the caller can see."""
    return await describe_for_caller(
        agent_backend=container.agent_backend,
        runner_backend=getattr(container, "runner_backend", None),
        workflow_backend=container.workflow_backend,
        key=key,
    )
