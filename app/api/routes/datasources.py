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

Two auth types carry no secret at all and so have nothing to redact:
``service_identity`` (a token minted from the backend's own OAuth2 key) and
``google`` (a token minted by impersonating the configured Google service
account).  ``google``'s ``impersonate_subject`` is not a secret but it is not
caller-controlled either — see ``_reject_foreign_google_subject``.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from app.api.dependencies import get_container
from app.core.config import Settings, get_settings
from app.core.container import ApplicationContainer
from app.domain.models.data_source_definition import (
    AnyDataSourceAuth,
    DataSourceDefinition,
    validate_operations,
)
from app.domain.models.sheet_binding import (
    BindingValidationError,
    SheetBinding,
    header_fingerprint,
    validate_bindings,
)
from app.infrastructure.datasources.discovery import (
    MAX_IMPORTED_OPERATIONS,
    MAX_SPEC_BYTES,
    SpecFetchError,
    SpecParseError,
    fetch_and_parse_spec,
    parse_spec,
    probe_and_discover,
)
from app.infrastructure.auth.google_token_provider import check_impersonate_subject
from app.infrastructure.datasources.destructive import is_destructive
from app.infrastructure.datasources.google_sheets import (
    google_sheets_template,
    resolve_google_file,
)
from app.infrastructure.datasources.sheet_binding_compile import (
    refresh_binding_operations,
    stamp_compiled,
)
from app.infrastructure.datasources.sheet_binding_library import ensure_binding_scripts
from app.infrastructure.datasources.sheet_binding_resolver import SheetBindingError
from app.infrastructure.datasources.sheet_compute import TRANSFORM_SIGNATURE
from app.infrastructure.datasources.sheet_binding_runtime import (
    params_from_state,
    plan_write_binding,
    probe_sheet,
    render_cell_changes,
    run_read_binding,
)
from app.infrastructure.datasources.try_run import shrink_sample, try_operation

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
    # kind == "pubsub" only: {topic, subscription, project_id, event_schema}.
    pubsub: dict | None = None
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
    # Omitted keeps the stored bindings; sent replaces them wholesale, and the
    # compiled operations are rebuilt from them (see update_datasource).
    bindings: list[dict] | None = None
    pubsub: dict | None = None
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


class FetchSchemaRequest(BaseModel):
    """Import a specification the caller points at explicitly."""

    schema_url: str
    kind: str = "http"
    # Auth in the stored (secret-bearing) shape, or with the secret named via
    # ``from_config``; omit for an unauthenticated fetch.
    auth: dict | None = None


class ResolveGoogleFileRequest(BaseModel):
    """A pasted Drive URL, or a bare file id."""

    # Named `ref` rather than `url` because the Picker hands back an id, not a
    # URL, and both go through the same parse.
    ref: str


class ProbeSheetRequest(BaseModel):
    """Everything the binding editor's form is built from, in one request."""

    file_id: str
    # Omit for the first tab.  The tab is addressed by title because that is
    # what the Sheets values API takes; the numeric id comes back in the
    # response and is what a binding stores as the authoritative identity.
    sheet: str | None = None
    header_row: int = 1
    # Which stored source's Google credential and operations to probe through.
    # Not the auth block itself: minting a Google token needs the impersonated
    # credential, which only the backend holds.
    source_id: str = "google-sheets"


class SaveBindingRequest(BaseModel):
    """A binding as the editor authors it — the model does the validating."""

    binding: dict[str, Any]


class PreviewBindingRequest(BaseModel):
    """Sample inputs to resolve a binding against, without writing anything.

    Either shape works: ``state`` is a snapshot of workflow state (what the
    editor sends, since that is what the author is reasoning about) and
    ``params`` names the compiled operation's params directly (what a caller
    already speaking the operation's language sends).
    """

    state: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


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
    # Set by the client after the user read the row count and clicked through
    # the confirmation. Without it a destructive operation is previewed and
    # refused — see ``try_datasource_operation``.
    confirm_destructive: bool = False


# ─── Secret redaction ─────────────────────────────────────────────────────────

REDACTED_SECRET = "********"

# Secret field(s) per auth type; anything else in the block is not secret.
# `none`, `service_identity` and `google` are absent on purpose: they store no
# secret, so there is nothing to redact, preserve across an update, or resolve
# from config (a `from_config` on one of them is a 422 below).
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


def _reject_foreign_google_subject(auth: Any, settings: Settings) -> None:
    """422 when a ``google`` auth block names a principal we will not impersonate.

    The auth block is caller-supplied and the backend can impersonate whatever
    it has been granted ``serviceAccountTokenCreator`` on, so an unchecked
    ``impersonate_subject`` would let anyone able to create a data source
    borrow another service account's authority.  The executor ignores a
    foreign value anyway (see
    ``app.infrastructure.auth.google_token_provider.resolve_impersonate_subject``);
    this exists so the caller is told instead of finding out later that its
    value was silently dropped.
    """
    error = check_impersonate_subject(auth, settings)
    if error:
        raise HTTPException(status_code=422, detail=error)


def _reject_stored_foreign_google_subject(
    source: DataSourceDefinition, settings: Settings
) -> None:
    """422 when a *stored* source's ``google`` auth names a foreign principal.

    The create/update paths check the incoming auth block, but a document
    written before that check existed (or by a different route) could still
    name another service account. Anything that mints a Google token from a
    stored definition — the sheet probe, a binding preview — checks it again
    before doing so, because that is the moment the authority would actually be
    borrowed.
    """
    _reject_foreign_google_subject(
        source.auth.model_dump(mode="json") if source.auth else None, settings
    )


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
    # Pub/Sub topics are their own resource now — a data source is a callable
    # API.  Existing kind="pubsub" documents still deserialise (the migration
    # script moves them), but no new one can be written here.
    if data.get("kind") == "pubsub":
        raise HTTPException(
            status_code=422,
            detail="Pub/Sub topics are events now — use /events instead of a pubsub data source",
        )
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
    view: Literal["full", "summary"] = "full",
    container: ApplicationContainer = Depends(get_container),
):
    """List all registered data source definitions (auth secrets redacted).

    ``view=summary`` keeps each operation's name and method -- the list view
    needs the methods to aggregate a risk badge -- and drops the rest of the
    operation: path, query, params, response_schema, pagination.  Those schemas
    are the bulk of a source (one imported OpenAPI spec can carry dozens of
    operations), and none of them is read until a source is opened.  ``auth`` is
    dropped too rather than redacted, so a summary carries no secret shape at
    all.  Default stays ``full``.
    """
    _require_backend(container)
    assert container.data_source_backend is not None
    sources = await container.data_source_backend.list()
    if view == "summary":
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "kind": s.kind,
                "base_url": s.base_url,
                "operations": [
                    {"name": op.name, "method": op.method} for op in s.operations
                ],
            }
            for s in sources
        ]
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
    _reject_foreign_google_subject(payload.get("auth"), container.settings)
    defn = _build_definition(payload)
    saved = await container.data_source_backend.create(defn)
    await _refresh_datasource_tools(container)
    return _redact_secrets(saved.model_dump(mode="json"))


@router.post("/probe")
async def probe_datasource(
    body: ProbeDataSourceRequest,
    settings: Settings = Depends(get_settings),
):
    """Probe a base URL for reachability and credential acceptance.

    Schema discovery happens here only for ``kind == "graphql"``, where the
    endpoint URL is itself the way to fetch the schema (introspection).  HTTP
    sources are never guessed at: point ``POST /datasources/schema/fetch`` at
    the specification URL, or upload the file to
    ``POST /datasources/schema/upload``.

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


@router.post("/google/resolve")
async def resolve_google_datasource_file(
    body: ResolveGoogleFileRequest,
    settings: Settings = Depends(get_settings),
):
    """Resolve a pasted Google Drive URL (or file id) and check our access.

    Follows the ``/probe`` pattern: a target-server failure is never a 5xx, it
    is encoded in the response.  That matters more here than for a probe,
    because the *expected* first answer is "the service account cannot see this
    document" — nobody has shared it yet — and the caller has to be able to
    show the address to share it with rather than an error page.

    ``{status: "ok", file_id, name, mime_type, can_edit, service_account}`` on
    success; otherwise ``status`` is one of ``invalid`` (not a Drive
    link/id), ``wrong_type`` (a Doc or a Slides deck, not a Sheet),
    ``no_access`` (403/404 from Drive — share it with ``service_account`` as
    Editor and try again), ``not_configured`` (GOOGLE_IMPERSONATE_SA unset) or
    ``error``, each with ``error`` carrying the detail.
    """
    return await resolve_google_file(body.ref, settings)


@router.get("/google/sheets-template")
async def get_google_sheets_template(settings: Settings = Depends(get_settings)):
    """The ``google-sheets`` data source, ready to be POSTed to ``/datasources``.

    The operation templates are code, not something a person retypes: the
    editor prefills its form from this and the normal create path stores it,
    so validation, redaction and the MCP tool refresh all still apply.  Also
    carries ``service_account`` (the address documents must be shared with) and
    ``default_value_input_option``, both of which the editor displays.
    """
    return google_sheets_template(settings)


@router.post("/google/probe-sheet")
async def probe_google_sheet(
    body: ProbeSheetRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Read a spreadsheet's tabs, named ranges, header row and a few real rows.

    This one endpoint drives the whole binding editor: the tab dropdown, the
    detected-schema table, every column dropdown in the read and write forms,
    and the fingerprint that gets stored with the binding.  Which is the point
    — no screen in that editor accepts a typed column name, so a binding cannot
    name a column the sheet does not have.

    Follows the ``/probe`` contract for failures: a target-server problem is
    encoded in the response rather than raised, because "the service account
    cannot see that document" is an expected answer the editor has to be able
    to show (see ``POST /datasources/google/resolve``).  ``status`` is ``ok``
    or ``error`` with ``error`` carrying the detail.
    """
    _require_backend(container)
    assert container.data_source_backend is not None
    source = await container.data_source_backend.get(body.source_id)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Data source '{body.source_id}' not found — create the Google "
                "Sheets source before probing a spreadsheet"
            ),
        )
    if container.data_source_executor is None:
        raise HTTPException(status_code=501, detail="Data source executor not configured")
    _reject_stored_foreign_google_subject(source, container.settings)
    try:
        result = await probe_sheet(
            source,
            container.data_source_executor,
            body.file_id,
            body.sheet,
            body.header_row,
        )
    except SheetBindingError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — reported like /probe does
        logger.info("probe-sheet '%s' failed: %s", body.file_id, exc)
        return {"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}
    return {"status": "ok", "error": None, **result}


@router.post("/schema/fetch")
async def fetch_datasource_schema(
    body: FetchSchemaRequest,
    settings: Settings = Depends(get_settings),
):
    """Fetch an API specification from an explicit URL and map it to operations.

    ``schema_url`` may point at an OpenAPI/Swagger document (JSON or YAML), a
    GraphQL introspection result, GraphQL SDL, or — when ``kind`` is
    ``graphql`` — at the GraphQL endpoint itself, which is introspected.

    Returns ``{kind, source, base_url, operations}``.  The operations are a
    pick-list: nothing is stored until the caller saves the data source with
    the subset it wants.  A URL that cannot be read or a body that is not a
    specification is a 422 with the detail, never a 5xx.
    """
    auth_model = None
    if body.auth is not None:
        auth = _resolve_auth_from_config(body.auth, settings)
        try:
            auth_model = _AUTH_ADAPTER.validate_python(auth)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc
    kind = "graphql" if body.kind == "graphql" else "http"
    try:
        return await fetch_and_parse_spec(
            body.schema_url, kind=kind, auth=auth_model,
            max_operations=MAX_IMPORTED_OPERATIONS,
        )
    except (SpecFetchError, SpecParseError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/schema/upload")
async def upload_datasource_schema(
    file: UploadFile = File(...),
    kind: str = Form("http"),
):
    """Map an uploaded API specification file to operations.

    Same result shape as ``POST /datasources/schema/fetch``; the document is
    read from the upload instead of a URL.  ``kind`` is accepted for symmetry
    but the document type is detected from the content, so an OpenAPI file
    uploaded against ``kind=graphql`` still parses as OpenAPI.
    """
    raw = await file.read()
    if len(raw) > MAX_SPEC_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Specification is larger than {MAX_SPEC_BYTES // (1024 * 1024)} MB",
        )
    try:
        return parse_spec(
            raw,
            source=file.filename or "uploaded specification",
            max_operations=MAX_IMPORTED_OPERATIONS,
        )
    except SpecParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/try-operation")
async def try_datasource_operation(
    body: TryOperationRequest,
    request: Request,
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
    _reject_foreign_google_subject(auth, container.settings)

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

    gate = await _gate_try_run(request, body, defn, container)
    if gate is not None:
        return gate

    return await try_operation(
        defn, body.operation, body.params, container.settings, container.data_source_executor
    )


async def _gate_try_run(
    request: Request,
    body: TryOperationRequest,
    defn: DataSourceDefinition,
    container: ApplicationContainer,
) -> dict[str, Any] | None:
    """Stop a try run of a destructive operation until it is confirmed.

    Try run is the one surface where the approver is already present: a person
    in the editor, one click from deleting whatever the operation points at.
    Blocking on Slack would make a delete endpoint untestable, so the gate is
    a two-step instead — the first call previews and refuses, returning the row
    count and the targets for the UI to put in front of them; the second call
    carries ``confirm_destructive`` and runs.

    That is a self-approval, and it is recorded as one: the case is written
    with ``surface="try_run"``, which keeps it in the audit trail and out of
    the history that grants the meta-LLM autonomy.

    Returns the response to send instead of running, or ``None`` to proceed.
    """
    if not getattr(container.settings, "approvals_enabled", True):
        return None
    executor = container.data_source_executor
    service = getattr(container, "approval_service", None)
    if executor is None or service is None:
        return None

    op = defn.get_operation(body.operation)
    if op is None or not is_destructive(op, defn):
        return None

    try:
        plan = await executor.preview(defn, body.operation, body.params)
    except Exception as exc:
        # The preview is real traffic against the target API, and a try run
        # reports failures rather than raising them.
        logger.info("try-operation '%s' preview failed: %s", body.operation, exc)
        return {
            "status": "error",
            "error": f"Could not work out what this would delete: {exc}",
            "api_output": None,
            "suggested_mapping": None,
        }

    if plan.affected_rows < 1:
        return None

    if not body.confirm_destructive:
        return {
            "status": "confirmation_required",
            "error": None,
            "api_output": None,
            "suggested_mapping": None,
            "destructive": {
                "operation": body.operation,
                "method": (op.method or "").upper(),
                "affected_rows": plan.affected_rows,
                "targets": plan.targets,
                "affected_sample": [str(item) for item in plan.sample],
            },
        }

    claims = getattr(request.state, "jwt_claims", None) or {}
    await service.record_confirmed(
        source=defn,
        operation=body.operation,
        method=(op.method or "").upper(),
        params=body.params,
        affected_rows=plan.affected_rows,
        targets=plan.targets,
        sample=plan.sample,
        decided_by_name=str(
            claims.get("name") or claims.get("preferred_username")
            or claims.get("email") or claims.get("sub") or ""
        ),
        decided_by_id=str(claims.get("sub") or ""),
    )
    return None


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
        _reject_foreign_google_subject(incoming["auth"], container.settings)
    payload = existing_payload
    payload.update(incoming)
    payload["id"] = source_id
    payload["created_at"] = existing.created_at
    defn = _build_definition(payload)
    # A PUT that replaces ``operations`` wholesale would drop the operations the
    # source's bindings compiled to, leaving bindings the runtime can no longer
    # reach. Recompiling here keeps the two halves of a binding together no
    # matter which field the caller happened to send.
    if defn.bindings:
        defn = defn.model_copy(update={
            "operations": refresh_binding_operations(defn, defn.bindings)
        })
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


# ─── Sheet bindings (nested under a source — after the item routes) ───────────

def _binding_payload(binding: SheetBinding) -> dict[str, Any]:
    """A binding as the editor reads it back, aliases and all."""
    return binding.model_dump(mode="json")


def _parse_binding(
    payload: dict[str, Any], existing: SheetBinding | None = None
) -> SheetBinding:
    """Validate one binding, mapping every rejection onto HTTP 422.

    Two layers, and both matter: pydantic rejects a malformed shape, and
    :func:`validate_bindings` rejects a *well-formed* binding that cannot work —
    an unknown column, a mode missing its required field.  The second is the one
    an author actually hits, so its messages name the column and the field.

    **Where the tier-1/tier-2 line is drawn.**  Provenance and generated code
    are never read from the request body — they are carried over from what is
    already stored (*existing*), or left at their tier-1 defaults when nothing
    is.  So this endpoint cannot be used to fabricate LLM provenance or to
    smuggle in a transform: posting ``tier: "script"`` and a ``compute`` block
    here has no effect at all.  Only the compile and edit endpoints below write
    those fields, and both go through the ADMIN gate.  Editing the *form* of a
    tier-2 binding through this endpoint therefore keeps its code and its
    provenance intact, which is what an author expects when they change its
    output key.
    """
    try:
        binding = SheetBinding.model_validate(payload)
    except ValidationError as exc:
        # include_context=False because a custom validator's `ctx` carries the
        # ValueError object itself, which JSONResponse cannot serialise; the
        # message is already in `msg`.
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_url=False, include_context=False),
        ) from exc

    if existing is not None and existing.compute is not None:
        # A stored tier-2 binding keeps the half this endpoint does not author:
        # its generated code and the provenance describing where that came from.
        binding.compute = existing.compute.model_copy(deep=True)
        binding.resolution = existing.resolution.model_copy(deep=True)
    else:
        # Provenance is set here, not accepted from the caller: a binding
        # authored through this endpoint was authored by a person in a form, and
        # a caller cannot claim otherwise by posting it.
        binding.compute = None
        binding.resolution.tier = "binding"
        binding.resolution.authored_by = "human"
        binding.resolution.instruction = None
        binding.resolution.model_id = None
        binding.resolution.answers = {}
        binding.resolution.golden = None
        binding.resolution.script_id = None
        binding.resolution.edited_by_human = False

    # A binding saved with headers but no fingerprint would have no drift
    # protection at all, which is the one thing this feature must not allow to
    # happen quietly.
    if not binding.sheet_schema.fingerprint:
        binding.sheet_schema.fingerprint = header_fingerprint(binding.sheet_schema.headers)
    return binding


async def _persist_bindings(
    container: ApplicationContainer,
    existing: DataSourceDefinition,
    bindings: list[SheetBinding],
) -> DataSourceDefinition:
    """Validate, compile and store a new binding list for *existing*.

    The compile step is what makes a binding callable, so it happens on the
    same write: the operation list is refreshed from the bindings, the whole
    definition is re-validated as one (a binding whose name collides with a raw
    operation has to fail here, not at run time), and the MCP tool list is
    rebuilt so the new operation is immediately reachable by an agent.
    """
    try:
        validate_bindings(bindings)
    except BindingValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    operations = refresh_binding_operations(existing, bindings)
    updated = existing.model_copy(update={
        "bindings": [stamp_compiled(b) for b in bindings],
        "operations": operations,
    })
    try:
        validate_operations(updated)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    assert container.data_source_backend is not None
    saved = await container.data_source_backend.update(updated.id, updated)
    # Best-effort, both of these: the binding is already stored and usable.
    await ensure_binding_scripts(getattr(container, "script_backend", None))
    await _refresh_datasource_tools(container)
    return saved


async def _load_source(
    container: ApplicationContainer, source_id: str
) -> DataSourceDefinition:
    _require_backend(container)
    assert container.data_source_backend is not None
    source = await container.data_source_backend.get(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Data source '{source_id}' not found")
    return source


@router.get("/{source_id}/bindings")
async def list_datasource_bindings(
    source_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Every binding on a source, plus the operation each one compiled to."""
    from app.application.sheet_compute_service import compute_status

    source = await _load_source(container, source_id)
    return [
        {
            **_binding_payload(binding),
            "compiled_operation": binding.name,
            # The tier the editor badges the row with. Derived, never a stored
            # claim: it is "does this binding carry code", answered by looking.
            "compute_status": compute_status(binding),
        }
        for binding in source.bindings
    ]


@router.post("/{source_id}/bindings", status_code=201)
async def create_datasource_binding(
    source_id: str,
    body: SaveBindingRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Add a binding and compile it into an operation of the same name.

    409 when the name is taken — by another binding, or by one of the source's
    raw operations.  The two share a namespace because a binding *is* an
    operation once compiled, and silently shadowing ``get_values`` would be a
    very confusing way to lose it.
    """
    source = await _load_source(container, source_id)
    binding = _parse_binding(body.binding)
    if source.get_binding(binding.name) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Binding '{binding.name}' already exists on '{source_id}'",
        )
    if source.get_operation(binding.name) is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"'{binding.name}' is already an operation of '{source_id}' — a "
                "binding compiles to an operation, so the names cannot collide"
            ),
        )
    saved = await _persist_bindings(container, source, [*source.bindings, binding])
    return _binding_payload(saved.get_binding(binding.name))  # type: ignore[arg-type]


@router.put("/{source_id}/bindings/{name}")
async def update_datasource_binding(
    source_id: str,
    name: str,
    body: SaveBindingRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Replace a binding in place, recompiling its operation.

    A rename is allowed (the body's ``name`` wins) and takes the operation with
    it: the old one is dropped, since the only thing that made it exist was the
    binding under its old name.
    """
    source = await _load_source(container, source_id)
    if source.get_binding(name) is None:
        raise HTTPException(
            status_code=404, detail=f"Binding '{name}' not found on '{source_id}'"
        )
    binding = _parse_binding(body.binding, existing=source.get_binding(name))
    if binding.name != name and source.get_binding(binding.name) is not None:
        raise HTTPException(
            status_code=409, detail=f"Binding '{binding.name}' already exists"
        )
    bindings = [binding if b.name == name else b for b in source.bindings]
    saved = await _persist_bindings(container, source, bindings)
    return _binding_payload(saved.get_binding(binding.name))  # type: ignore[arg-type]


@router.delete("/{source_id}/bindings/{name}", status_code=204)
async def delete_datasource_binding(
    source_id: str,
    name: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Delete a binding and the operation it compiled to."""
    source = await _load_source(container, source_id)
    if source.get_binding(name) is None:
        raise HTTPException(
            status_code=404, detail=f"Binding '{name}' not found on '{source_id}'"
        )
    await _persist_bindings(
        container, source, [b for b in source.bindings if b.name != name]
    )


@router.post("/{source_id}/bindings/{name}/preview")
async def preview_datasource_binding(
    source_id: str,
    name: str,
    body: PreviewBindingRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Resolve a binding against sample state and show what it would do.

    A write is planned but never sent: the reads run (they have to — the row
    number and the "before" values come from the live sheet), the header
    fingerprint is checked, the write is composed, and the response carries the
    target range plus the before/after of every cell.  ``changes`` is the same
    list the approval gate renders, so what an author sees here is what an
    approver will see later.

    A read is executed — a read binding has nothing to preview *but* its result
    — and the output is size-capped with ``try_run.shrink_sample`` so a
    thousand-row sheet does not come back whole.  No mapping is suggested and no
    model is called: a binding is deterministic, and a preview that consulted an
    LLM would not be a preview of it.
    """
    source = await _load_source(container, source_id)
    binding = source.get_binding(name)
    if binding is None:
        raise HTTPException(
            status_code=404, detail=f"Binding '{name}' not found on '{source_id}'"
        )
    executor = container.data_source_executor
    if executor is None:
        raise HTTPException(status_code=501, detail="Data source executor not configured")
    _reject_stored_foreign_google_subject(source, container.settings)

    params = dict(body.params) or params_from_state(binding, body.state)
    try:
        if binding.operation == "read":
            result = await run_read_binding(source, executor, binding, params)
            return {
                "status": "ok",
                "error": None,
                "binding": name,
                "operation": "read",
                "params": params,
                "range": None,
                "rows": len(result) if isinstance(result, list) else (0 if result is None else 1),
                "output": shrink_sample(result),
                "changes": [],
                "cells": [],
            }
        plan = await plan_write_binding(source, executor, binding, params)
        return {
            "status": plan["status"],
            "error": None,
            "binding": name,
            "operation": "write",
            "params": params,
            "reason": plan.get("reason"),
            "mode": plan.get("mode"),
            "row_number": plan.get("row_number"),
            # The A1 range(s) the write would target — what the author asked
            # for when they pressed Preview.
            "range": ", ".join(
                entry["range"] for entry in ((plan.get("call") or {}).get("params", {}).get("data") or [])
            ) or None,
            "value_input_option": plan.get("value_input_option"),
            "blank_policy": plan.get("blank_policy"),
            "cells": plan.get("cells") or [],
            "changes": render_cell_changes(plan.get("cells") or []),
            "cells_total": plan.get("cells_total"),
        }
    except SheetBindingError as exc:
        # Fingerprint drift lands here, and it is the message the author has to
        # read, so it is returned rather than raised as a 500.
        return {"status": "error", "error": str(exc), "binding": name, "changes": [], "cells": []}
    except Exception as exc:  # noqa: BLE001 — a preview reports, never 500s
        logger.info("binding preview '%s.%s' failed: %s", source_id, name, exc)
        return {
            "status": "error",
            "error": f"{exc.__class__.__name__}: {exc}",
            "binding": name,
            "changes": [],
            "cells": [],
        }


# ─── Tier 2: generated transforms (nested under a binding) ────────────────────
#
# The endpoints the escalation flow drives, all of them thin: every rule lives
# in app.application.sheet_compute_service, so the MCP tools and the chat agent
# enforce exactly the same ones. Storing code needs ADMIN (see
# auth.sandbox_guard.assert_generated_code_allowed, raised from the service);
# reading it, re-testing it and marking it stale do not.


class CompileComputeRequest(BaseModel):
    """An instruction, and any answers to questions a previous compile asked.

    SECURITY: `instruction` is untrusted text. It is stored on the binding and
    fed back into a compile prompt on every recompile, so it is handled as data
    throughout — interpolated only into a delimited user-role block, never into
    a system prompt, and never into anything that decides which gates run. See
    `sheet_compute_generate`.
    """

    # Omitted re-uses the instruction already stored on the binding, which is
    # what a plain "recompile" means.
    instruction: str | None = None
    # question -> answer. Folded into the stored answers, so a recompile is
    # reproducible rather than a fresh guess.
    answers: dict[str, str] = {}
    # Re-run the model even when nothing about the request changed.
    force: bool = False
    # Sample `params` to verify the transform against, when the binding takes
    # any. Not persisted.
    params: dict[str, Any] = {}


class EditComputeRequest(BaseModel):
    """Hand-written replacement for a generated transform."""

    code: str
    params: dict[str, Any] = {}


class MarkStaleRequest(BaseModel):
    reason: str = ""


def _compute_service_error(exc: Exception) -> HTTPException:
    """A service refusal as a 422 the editor can render verbatim.

    422 rather than 500 because every one of them is a message written for the
    person who pressed the button: a failing gate, a hand-edited binding that
    will not be regenerated, a flag that is off.
    """
    return HTTPException(status_code=422, detail=str(exc))


async def _compute_ctx(container: ApplicationContainer, source_id: str):
    """The source, its executor and the persistence hooks the service needs."""
    source = await _load_source(container, source_id)
    executor = container.data_source_executor
    if executor is None:
        raise HTTPException(status_code=501, detail="Data source executor not configured")
    _reject_stored_foreign_google_subject(source, container.settings)
    return source, executor


@router.post("/{source_id}/bindings/{name}/compile")
async def compile_datasource_binding(
    source_id: str,
    name: str,
    body: CompileComputeRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Generate the computation of a binding a tier-1 form cannot express.

    Returns one of three things, and the editor renders each differently:

    ``{"status": "needs", "needs": [{"question", "options"}]}``
        The request is ambiguous about the data. Nothing is stored; the answers
        come back to this endpoint and are folded in.
    ``{"status": "ok", "code", "rationale", "output", "compute": {...}}``
        Every gate passed. The code is stored **inert** — the response carries
        the code, the model's one-line rationale and its output on the author's
        own sample rows, and activation is a separate, explicit call.
    ``{"status": "error", "error"}`` (as HTTP 422)
        Out of attempts, with the last gate's own message.

    Needs ADMIN: storing code a model wrote, which this backend then executes,
    is a privileged act even though the code runs sandboxed.
    """
    from app.application.sheet_compute_service import (
        ComputeServiceError,
        compile_compute,
    )
    from app.infrastructure.auth.sandbox_guard import GeneratedCodeNotPermittedError

    source, executor = await _compute_ctx(container, source_id)
    try:
        result = await compile_compute(
            source=source,
            name=name,
            instruction=body.instruction,
            answers=body.answers,
            settings=container.settings,
            executor=executor,
            backend=container.data_source_backend,
            script_backend=getattr(container, "script_backend", None),
            publish=lambda: _refresh_datasource_tools(container),
            params=body.params,
            force=body.force,
        )
    except GeneratedCodeNotPermittedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ComputeServiceError as exc:
        raise _compute_service_error(exc) from exc
    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result)
    return result


@router.get("/{source_id}/bindings/{name}/code")
async def get_datasource_binding_code(
    source_id: str,
    name: str,
    container: ApplicationContainer = Depends(get_container),
):
    """The generated transform of a binding, with its verification state.

    A read: no ADMIN, because reviewing what is already stored is how somebody
    decides whether to trust it, and making that privileged would leave the
    code unreviewable by the people it affects.
    """
    from app.application.sheet_compute_service import compute_status

    source = await _load_source(container, source_id)
    binding = source.get_binding(name)
    if binding is None:
        raise HTTPException(
            status_code=404, detail=f"Binding '{name}' not found on '{source_id}'"
        )
    if binding.compute is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Binding '{name}' is a tier-1 binding — a form, with no code. "
                "That is the normal case and needs no review."
            ),
        )
    return {
        "binding": name,
        "code": binding.compute.code,
        "signature": TRANSFORM_SIGNATURE,
        "compute": compute_status(binding),
        "golden": (
            binding.resolution.golden.model_dump(mode="json")
            if binding.resolution.golden else None
        ),
    }


@router.put("/{source_id}/bindings/{name}/code")
async def edit_datasource_binding_code(
    source_id: str,
    name: str,
    body: EditComputeRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Replace a generated transform with a hand-written one.

    Held to exactly the gates the generated version was: the same AST
    allow-list, the same sandbox, the same determinism double-run, the same
    shape and whitelist checks. Being written by a person changes who is
    accountable, not what the code is allowed to do.

    Sets ``edited_by_human``, which permanently stops regeneration — a later
    recompile refuses rather than overwriting the edit. Activation is cleared:
    this is new code, and the previous approval was of the previous code.
    """
    from app.application.sheet_compute_service import (
        ComputeServiceError,
        edit_compute_code,
    )
    from app.infrastructure.auth.sandbox_guard import GeneratedCodeNotPermittedError

    source = await _load_source(container, source_id)
    try:
        return await edit_compute_code(
            source=source,
            name=name,
            code=body.code,
            settings=container.settings,
            backend=container.data_source_backend,
            script_backend=getattr(container, "script_backend", None),
            publish=lambda: _refresh_datasource_tools(container),
            params=body.params,
        )
    except GeneratedCodeNotPermittedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ComputeServiceError as exc:
        raise _compute_service_error(exc) from exc


@router.post("/{source_id}/bindings/{name}/activate")
async def activate_datasource_binding_code(
    source_id: str,
    name: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Turn a compiled transform on, re-proving it against its fixture first.

    Compiling and activating are deliberately two events: the first says the
    code passes its checks, the second says a person looked at the code and at
    its output on their own rows and accepted it. Generated code never starts
    running just because it compiled.
    """
    from app.application.sheet_compute_service import (
        ComputeServiceError,
        activate_compute,
    )
    from app.infrastructure.auth.sandbox_guard import GeneratedCodeNotPermittedError

    source = await _load_source(container, source_id)
    try:
        return await activate_compute(
            source=source,
            name=name,
            settings=container.settings,
            backend=container.data_source_backend,
            script_backend=getattr(container, "script_backend", None),
            publish=lambda: _refresh_datasource_tools(container),
        )
    except GeneratedCodeNotPermittedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ComputeServiceError as exc:
        raise _compute_service_error(exc) from exc


@router.post("/{source_id}/bindings/{name}/retest")
async def retest_datasource_binding_code(
    source_id: str,
    name: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Re-run the golden fixture and re-check the sheet's header row.

    Two independent questions: does the code still compute the frozen answer,
    and is the schema it was authored against still the sheet's schema. Either
    one failing marks the binding stale and switches it off, which is why this
    reports rather than raising — the state it leaves behind is the answer.
    """
    from app.application.sheet_compute_service import ComputeServiceError, retest_compute

    source, executor = await _compute_ctx(container, source_id)
    try:
        return await retest_compute(
            source=source,
            name=name,
            settings=container.settings,
            executor=executor,
            backend=container.data_source_backend,
            script_backend=getattr(container, "script_backend", None),
            publish=lambda: _refresh_datasource_tools(container),
        )
    except ComputeServiceError as exc:
        raise _compute_service_error(exc) from exc


@router.post("/{source_id}/bindings/{name}/stale")
async def mark_datasource_binding_stale(
    source_id: str,
    name: str,
    body: MarkStaleRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Switch a generated transform off until somebody re-confirms it.

    Not gated on ADMIN: stopping something that looks wrong is a safety action,
    and needing a privileged role to do it would be the wrong trade.
    """
    from app.application.sheet_compute_service import (
        ComputeServiceError,
        mark_compute_stale,
    )

    source = await _load_source(container, source_id)
    try:
        return await mark_compute_stale(
            source=source,
            name=name,
            reason=body.reason,
            backend=container.data_source_backend,
            script_backend=getattr(container, "script_backend", None),
            publish=lambda: _refresh_datasource_tools(container),
        )
    except ComputeServiceError as exc:
        raise _compute_service_error(exc) from exc
