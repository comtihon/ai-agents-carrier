"""REST API for DataSourceDefinition CRUD.

Route ordering note
-------------------
Literal-segment routes are registered BEFORE parameterised routes
(``/{source_id}``) so that Starlette does not swallow the literal as a path
param.

Every successful mutation triggers a rebuild of the ``/mcp/datasources`` tool
list plus a refresh of the ``datasources`` MCP integration, so newly created
operations become callable by agents without a restart.  Refresh failures are
logged, never raised — the definition is already persisted.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    validate_operations,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasources", tags=["datasources"])


# ─── Request models ───────────────────────────────────────────────────────────

class CreateDataSourceRequest(BaseModel):
    id: str
    name: str = ""
    description: str | None = None
    kind: str = "http"
    base_url: str = ""
    auth: dict | None = None
    default_headers: dict[str, str] = Field(default_factory=dict)
    operations: list[dict] = Field(default_factory=list)
    cache: dict | None = None
    timeout_seconds: float = 30
    retries: dict | None = None


class UpdateDataSourceRequest(BaseModel):
    # None = omitted by the caller -> preserve the existing value.
    name: str | None = None
    description: str | None = None
    kind: str | None = None
    base_url: str | None = None
    auth: dict | None = None
    default_headers: dict[str, str] | None = None
    operations: list[dict] | None = None
    cache: dict | None = None
    timeout_seconds: float | None = None
    retries: dict | None = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_backend(container: ApplicationContainer) -> None:
    if container.data_source_backend is None:
        raise HTTPException(status_code=501, detail="Data source backend not configured")


def _build_definition(data: dict[str, Any]) -> DataSourceDefinition:
    """Validate the payload into a definition, mapping errors onto HTTP 422."""
    try:
        defn = DataSourceDefinition.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
    try:
        validate_operations(defn)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return defn


async def _refresh_datasource_tools(container: ApplicationContainer) -> None:
    """Rebuild the MCP tool list, then reconnect the ``datasources`` server."""
    try:
        from app.api.mcp.datasources_server import (
            get_datasources_mcp,
            rebuild_datasource_tools,
        )
        await rebuild_datasource_tools(
            get_datasources_mcp(), container.data_source_backend, lambda: container
        )
    except Exception:
        logger.exception("failed to rebuild datasources MCP tools")
        return
    try:
        await container.mcp_tools_provider.refresh_server("datasources")
    except Exception:
        logger.warning("datasources MCP refresh failed", exc_info=True)


# ─── Collection routes ────────────────────────────────────────────────────────

@router.get("")
async def list_datasources(
    container: ApplicationContainer = Depends(get_container),
):
    """List all registered data source definitions."""
    _require_backend(container)
    assert container.data_source_backend is not None
    sources = await container.data_source_backend.list()
    return [s.model_dump(mode="json") for s in sources]


@router.post("", status_code=201)
async def create_datasource(
    body: CreateDataSourceRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Register a new data source definition."""
    _require_backend(container)
    assert container.data_source_backend is not None

    existing = await container.data_source_backend.get(body.id)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Data source '{body.id}' already exists")

    payload = body.model_dump(exclude_none=True)
    defn = _build_definition(payload)
    saved = await container.data_source_backend.create(defn)
    await _refresh_datasource_tools(container)
    return saved.model_dump(mode="json")


# ─── Item routes (parameterised — must come AFTER literal-segment routes) ─────

@router.get("/{source_id}")
async def get_datasource(
    source_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Get a specific data source definition by ID."""
    _require_backend(container)
    assert container.data_source_backend is not None

    defn = await container.data_source_backend.get(source_id)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Data source '{source_id}' not found")
    return defn.model_dump(mode="json")


@router.put("/{source_id}")
async def update_datasource(
    source_id: str,
    body: UpdateDataSourceRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Update an existing data source definition (omitted fields are preserved)."""
    _require_backend(container)
    assert container.data_source_backend is not None

    existing = await container.data_source_backend.get(source_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Data source '{source_id}' not found")

    payload = existing.model_dump(mode="json")
    payload.update(body.model_dump(exclude_none=True))
    payload["id"] = source_id
    payload["created_at"] = existing.created_at
    defn = _build_definition(payload)
    saved = await container.data_source_backend.update(source_id, defn)
    await _refresh_datasource_tools(container)
    return saved.model_dump(mode="json")


@router.delete("/{source_id}", status_code=204)
async def delete_datasource(
    source_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Delete a data source definition."""
    _require_backend(container)
    assert container.data_source_backend is not None

    existing = await container.data_source_backend.get(source_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Data source '{source_id}' not found")

    await container.data_source_backend.delete(source_id)
    await _refresh_datasource_tools(container)
