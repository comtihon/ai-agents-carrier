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

Secret handling
---------------
Auth blocks store secret values (token / password / header value) in the
definition itself.  Responses never echo them back: every secret field is
replaced with ``REDACTED_SECRET``.  On update, an incoming secret equal to
that placeholder (or an omitted auth block) preserves the stored value.

Instead of the secret itself, an auth block may carry ``from_config`` naming a
key of the backend's forwardable config; the value is resolved here at write
time and stored like any other secret, so the caller never handles it.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from app.api.dependencies import get_container
from app.core.config import Settings, get_settings
from app.core.container import ApplicationContainer
from app.domain.models.data_source_definition import (
    AnyDataSourceAuth,
    DataSourceDefinition,
    validate_operations,
)
from app.infrastructure.datasources.discovery import probe_and_discover
from app.infrastructure.datasources.try_run import try_operation

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


class ProbeDataSourceRequest(BaseModel):
    base_url: str
    kind: str = "http"
    # Auth block in the stored (secret-bearing) shape, e.g.
    # {"type": "bearer", "token": "..."}, or with the secret named as
    # {"type": "bearer", "from_config": "AFP_SERVICE_TOKEN"}; omit or
    # {"type": "none"} for none.
    auth: dict | None = None


class TryOperationRequest(BaseModel):
    base_url: str
    kind: str = "http"
    # Auth in the stored shape, or with the secret named via ``from_config``.
    # Secret fields may carry the redaction placeholder (or the block may be
    # omitted) when ``source_id`` points at a stored definition — the stored
    # secrets are used then.
    auth: dict | None = None
    source_id: str | None = None
    # Every operation of the draft definition (templates may reference
    # sibling operations), plus the name of the one to execute.
    operations: list[dict] = Field(default_factory=list)
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 30


# ─── Secret redaction ─────────────────────────────────────────────────────────

REDACTED_SECRET = "********"

# Secret field(s) per auth type; anything else in the block is not secret.
_SECRET_FIELDS: dict[str, tuple[str, ...]] = {
    "bearer": ("token",),
    "basic": ("password",),
    "header": ("value",),
}

_AUTH_ADAPTER: TypeAdapter = TypeAdapter(AnyDataSourceAuth)


def _redact_secrets(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace secret auth values in a dumped definition with the placeholder."""
    auth = payload.get("auth")
    if isinstance(auth, dict):
        for field in _SECRET_FIELDS.get(auth.get("type", ""), ()):
            if field in auth:
                auth[field] = REDACTED_SECRET
    return payload


def _merge_auth_secrets(incoming: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    """Fill redacted placeholders in an incoming auth block from the stored one.

    Only applies when the auth type is unchanged — switching types requires
    real secret values.
    """
    existing_auth = existing.get("auth")
    if not isinstance(existing_auth, dict) or incoming.get("type") != existing_auth.get("type"):
        return incoming
    for field in _SECRET_FIELDS.get(incoming.get("type", ""), ()):
        if incoming.get(field) == REDACTED_SECRET and field in existing_auth:
            incoming[field] = existing_auth[field]
    return incoming


FROM_CONFIG_FIELD = "from_config"


def _resolve_auth_from_config(auth: Any, settings: Settings) -> Any:
    """Turn a ``from_config`` reference into the named backend config value.

    Callers that must not handle a secret themselves may send, in place of the
    secret field, ``{"type": "bearer", "from_config": "AFP_SERVICE_TOKEN"}``.
    The key names an entry of the backend's forwardable config (the same set
    ``GET /llm/config/keys`` exposes by name); its value replaces the auth
    type's single secret field and is stored like any pasted secret.

    An unknown or blank key is a 422: storing an empty secret instead would
    resurface later as an opaque 401 from the target API.
    """
    if not isinstance(auth, dict) or FROM_CONFIG_FIELD not in auth:
        return auth
    resolved = dict(auth)
    key = (resolved.pop(FROM_CONFIG_FIELD) or "").strip()
    auth_type = resolved.get("type", "")
    fields = _SECRET_FIELDS.get(auth_type, ())
    if not fields:
        raise HTTPException(
            status_code=422,
            detail=f"auth type '{auth_type}' has no secret that can come from config",
        )
    if not key:
        raise HTTPException(status_code=422, detail="auth.from_config must name a config key")
    available = settings.get_forwardable_config()
    if key not in available:
        raise HTTPException(status_code=422, detail=f"config key '{key}' is not set on this backend")
    resolved[fields[0]] = available[key]
    return resolved


def _reject_placeholder_secrets(auth: Any) -> None:
    """422 when a secret field carries the redaction placeholder (create/type switch)."""
    if not isinstance(auth, dict):
        return
    for field in _SECRET_FIELDS.get(auth.get("type", ""), ()):
        if auth.get(field) == REDACTED_SECRET:
            raise HTTPException(
                status_code=422,
                detail=f"auth.{field} must be a real secret value, not '{REDACTED_SECRET}'",
            )


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
    """List all registered data source definitions (auth secrets redacted)."""
    _require_backend(container)
    assert container.data_source_backend is not None
    sources = await container.data_source_backend.list()
    return [_redact_secrets(s.model_dump(mode="json")) for s in sources]


@router.post("", status_code=201)
async def create_datasource(
    body: CreateDataSourceRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Register a new data source definition.

    Auth secrets must be real values (the redaction placeholder is rejected);
    the response echoes the definition with secrets redacted.
    """
    _require_backend(container)
    assert container.data_source_backend is not None

    existing = await container.data_source_backend.get(body.id)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Data source '{body.id}' already exists")

    payload = body.model_dump(exclude_none=True)
    if payload.get("auth") is not None:
        payload["auth"] = _resolve_auth_from_config(payload["auth"], container.settings)
    _reject_placeholder_secrets(payload.get("auth"))
    defn = _build_definition(payload)
    saved = await container.data_source_backend.create(defn)
    await _refresh_datasource_tools(container)
    return _redact_secrets(saved.model_dump(mode="json"))


@router.post("/probe")
async def probe_datasource(
    body: ProbeDataSourceRequest,
    settings: Settings = Depends(get_settings),
):
    """Probe a base URL and attempt schema discovery (OpenAPI / GraphQL).

    The auth block uses the stored shape (secret values inline) or names a
    config key via ``from_config``.  Target-server failures never surface as a
    5xx here — they are encoded in the response: ``url_status``
    ok|unauthorized|unreachable, ``auth_status`` ok|failed|skipped, ``error``
    human-readable detail or null, ``discovered`` schema or null.
    """
    auth_model = None
    if body.auth is not None:
        auth = _resolve_auth_from_config(body.auth, settings)
        try:
            auth_model = _AUTH_ADAPTER.validate_python(auth)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
    kind = "graphql" if body.kind == "graphql" else "http"
    return await probe_and_discover(body.base_url, kind=kind, auth=auth_model)


@router.post("/try-operation")
async def try_datasource_operation(
    body: TryOperationRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Execute one operation of a draft definition and sample its output.

    The target operation runs without its mapping / response_schema /
    pagination so the raw response shape is visible; the result carries a
    size-capped ``api_output`` sample plus a meta-LLM ``suggested_mapping``
    (JMESPath), or ``status: error`` with detail when the call fails.
    """
    auth = _resolve_auth_from_config(body.auth, container.settings)
    if body.source_id and container.data_source_backend is not None:
        stored = await container.data_source_backend.get(body.source_id)
        if stored is not None:
            stored_payload = stored.model_dump(mode="json")
            if auth is None:
                auth = stored_payload.get("auth")
            elif isinstance(auth, dict):
                auth = _merge_auth_secrets(auth, stored_payload)
    _reject_placeholder_secrets(auth)

    payload: dict[str, Any] = {
        "id": body.source_id or "__try__",
        "kind": "graphql" if body.kind == "graphql" else "http",
        "base_url": body.base_url,
        "operations": body.operations,
        "timeout_seconds": body.timeout_seconds,
    }
    if auth is not None:
        payload["auth"] = auth
    defn = _build_definition(payload)
    if defn.get_operation(body.operation) is None:
        raise HTTPException(status_code=422, detail=f"Unknown operation '{body.operation}'")

    return await try_operation(
        defn, body.operation, body.params, container.settings, container.data_source_executor
    )


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
    return _redact_secrets(defn.model_dump(mode="json"))


@router.put("/{source_id}")
async def update_datasource(
    source_id: str,
    body: UpdateDataSourceRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Update an existing data source definition (omitted fields are preserved).

    Auth secret fields equal to the redaction placeholder keep the stored
    value (same auth type only); an omitted auth block keeps the stored auth
    entirely.  The response redacts secrets.
    """
    _require_backend(container)
    assert container.data_source_backend is not None

    existing = await container.data_source_backend.get(source_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Data source '{source_id}' not found")

    existing_payload = existing.model_dump(mode="json")
    incoming = body.model_dump(exclude_none=True)
    if isinstance(incoming.get("auth"), dict):
        incoming["auth"] = _resolve_auth_from_config(incoming["auth"], container.settings)
        incoming["auth"] = _merge_auth_secrets(incoming["auth"], existing_payload)
        _reject_placeholder_secrets(incoming["auth"])
    payload = existing_payload
    payload.update(incoming)
    payload["id"] = source_id
    payload["created_at"] = existing.created_at
    defn = _build_definition(payload)
    saved = await container.data_source_backend.update(source_id, defn)
    await _refresh_datasource_tools(container)
    return _redact_secrets(saved.model_dump(mode="json"))


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
