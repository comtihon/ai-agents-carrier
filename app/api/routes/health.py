from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer

router = APIRouter(tags=["health"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness(container: ApplicationContainer = Depends(get_container)) -> dict[str, str]:
    graphs_loaded = bool(container.yaml_graph_registry.list_ids())
    return {"status": "ready" if graphs_loaded else "not_ready"}
