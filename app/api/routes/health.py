from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness(container: ApplicationContainer = Depends(get_container)) -> dict[str, str]:
    graphs_loaded = bool(container.yaml_graph_registry.list_ids())
    return {"status": "ready" if graphs_loaded else "not_ready"}


@router.get("/health/brain")
async def brain_health(request: Request) -> dict[str, str | None]:
    """State of the shared company memory (gbrain) integration.

    ``{"status": "not_configured" | "unavailable" | "ok", "detail": str | null}``.
    Always 200 and never raises: it reports on an optional dependency, so it
    must not take the liveness of this backend with it. The detail never
    carries the gbrain URL or key.
    """
    container = getattr(request.app.state, "container", None)
    client = getattr(container, "brain_client", None)
    if client is None:
        return {"status": "not_configured", "detail": None}
    try:
        return (await client.status()).as_dict()
    except Exception as exc:  # noqa: BLE001 — status() should not raise; be sure
        logger.warning("brain health check failed: %s", type(exc).__name__)
        return {"status": "unavailable", "detail": type(exc).__name__}
