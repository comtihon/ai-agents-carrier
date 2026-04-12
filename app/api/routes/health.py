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
    mongo_ok = await container.mongo_provider.ping()
    workflows_ok = bool(container.workflow_registry.list_definitions())
    return {"status": "ready" if mongo_ok and workflows_ok else "not_ready"}
