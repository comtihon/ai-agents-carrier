"""FastMCP server exposing every data source operation as an MCP tool.

The server is mounted by ``app.api.app`` and is the single access path for
data sources: agents reach it through the ``datasources`` MCP integration, and
workflows reach the same executor through the ``data_source`` step type.

One tool is registered per ``<data source> x <operation>``, named
``ds_<source_id>_<operation>``.  The tool's input schema is built from the
operation's declared ``params`` only — upstream operations of the DAG stay
invisible to the caller.  Handlers resolve the executor from the container at
*call* time so CRUD changes and container startup order never matter.

Scoping
-------
The registered tool set is process-global: it mirrors the whole registry, one
tool per operation of every source.  What a *caller* may reach is narrower.  A
spawned agent presents a signed capability grant (see
``app.infrastructure.auth.datasource_grant``) which the ASGI gate in front of
the mount verifies and publishes in a context variable for the duration of that
request; :class:`ScopedDatasourcesMCP` then filters ``list_tools`` **and**
refuses ``call_tool`` for anything outside it.  Both, because listing is not
authorization — an agent can guess a tool name it was never shown, and the tool
it would reach executes under this backend's own credentials.

No grant in context means the caller is this backend itself, holding the static
``MCP_DATASOURCES_API_KEY`` over loopback (``app.infrastructure.tools.mcp_client``),
and sees everything.  Nothing else can reach the mount without a grant: the gate
401s anything that is neither.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings

from app.domain.models.data_source_definition import (
    OperationDefinition,
    ParamSpec,
    extract_refs,
)
from app.infrastructure.auth.datasource_grant import DatasourceGrant

logger = logging.getLogger(__name__)

# Mounted at "/mcp" by create_app, so the full endpoint is /mcp/datasources.
STREAMABLE_HTTP_PATH = "/datasources"

_PY_TYPES: dict[str, type] = {
    "string": str,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")

# HTTP methods that cannot change upstream state.  Everything else is a write,
# and the tool description says so, so the model treats it as consequential.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# The grant in force for the current request, or None for an unscoped caller.
# Set by app.api.app._DatasourcesAuthWrapper once per HTTP request; safe as a
# context variable because the mount is stateless_http (one request, one task,
# and child tasks inherit a copy of the context that started them).
_GRANT: ContextVar[DatasourceGrant | None] = ContextVar(
    "datasource_grant", default=None
)

_mcp: FastMCP | None = None
_rebuild_lock = asyncio.Lock()


def set_current_grant(grant: DatasourceGrant | None) -> Token:
    """Publish *grant* for this request; pass the token to :func:`reset_current_grant`.

    Callers must set it explicitly to ``None`` for an unscoped caller rather
    than leaving it unset, so a value can never leak from one request into
    another that reuses the same context.
    """
    return _GRANT.set(grant)


def reset_current_grant(token: Token) -> None:
    _GRANT.reset(token)


def current_grant() -> DatasourceGrant | None:
    return _GRANT.get()


def grant_tool_names(grant: DatasourceGrant) -> frozenset[str]:
    """The tool names *grant* authorizes.

    Built with the same :func:`tool_name_for` used at registration, so the
    sanitisation and collision handling there can never drift out of step with
    what is checked here.  A source with an empty operation list contributes
    nothing, which is what makes an empty ``allowed_operations`` a deny.
    """
    return frozenset(
        tool_name_for(source_id, operation)
        for source_id, operations in grant.grants.items()
        for operation in operations
    )


class ScopedDatasourcesMCP(FastMCP):
    """FastMCP that answers only for the operations the caller was granted.

    ``FastMCP._setup_handlers`` registers ``self.list_tools`` / ``self.call_tool``
    as bound methods, so overriding them here is enough to gate every protocol
    request without touching the tool manager or re-registering per caller.
    """

    async def list_tools(self):  # type: ignore[override]
        tools = await super().list_tools()
        grant = _GRANT.get()
        if grant is None:
            return tools
        allowed = grant_tool_names(grant)
        return [tool for tool in tools if tool.name in allowed]

    async def call_tool(self, name: str, arguments: dict[str, Any]):  # type: ignore[override]
        grant = _GRANT.get()
        if grant is not None and name not in grant_tool_names(grant):
            # Refuse before the handler runs, so nothing resolves a data source
            # or mints an upstream token for an operation that was never
            # granted.  The message names the tool but not the alternatives:
            # a caller that guessed learns only that the guess failed.
            logger.warning(
                "run '%s' (agent '%s') called ungranted data source tool '%s' — refused",
                grant.run_id, grant.agent_id, name,
            )
            raise ToolError(f"Operation not granted: {name}")
        return await super().call_tool(name, arguments)


# Host header values FastMCP's DNS-rebinding guard accepts without being told.
# Mirrors app.api.mcp.management_server.DEFAULT_ALLOWED_HOSTS: the backend's own
# loopback MCP client, in both bare and with-port forms.  FastMCP's own default
# accepts only the with-port forms, which is why a spawned agent dialling this
# mount under a real hostname needs the deployment's address added — see
# Settings.datasources_mcp_allowed_hosts.
DEFAULT_ALLOWED_HOSTS = [
    "localhost",
    "localhost:*",
    "127.0.0.1",
    "127.0.0.1:*",
    "[::1]",
    "[::1]:*",
]


def _transport_security(
    allowed_hosts: list[str] | None = None,
) -> TransportSecuritySettings:
    """DNS-rebinding settings for the mount, loopback plus *allowed_hosts*.

    The loopback defaults are always kept: they are how this backend reaches its
    own mounted endpoint, and losing them would break that regardless of what a
    deployment declares.
    """
    hosts = list(
        dict.fromkeys(
            [h for h in (allowed_hosts or []) if h] + DEFAULT_ALLOWED_HOSTS
        )
    )
    origins = [f"{scheme}://{host}" for host in hosts for scheme in ("http", "https")]
    return TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins)


def build_datasources_mcp(
    allowed_hosts: list[str] | None = None,
) -> ScopedDatasourcesMCP:
    """Create the (stateless) FastMCP server instance for data sources."""
    return ScopedDatasourcesMCP(
        "datasources",
        instructions=(
            "Query configured data sources. Each tool wraps one operation of "
            "one data source; upstream dependencies are resolved automatically."
        ),
        stateless_http=True,
        streamable_http_path=STREAMABLE_HTTP_PATH,
        transport_security=_transport_security(allowed_hosts),
    )


def get_datasources_mcp() -> ScopedDatasourcesMCP:
    """Return the process-wide data sources MCP server, building it on demand."""
    global _mcp
    if _mcp is None:
        from app.core.config import get_settings

        _mcp = build_datasources_mcp(get_settings().datasources_mcp_allowed_hosts())
    return _mcp


def tool_name_for(source_id: str, operation: str) -> str:
    """Return the MCP tool name for one source/operation pair."""
    return (
        f"ds_{_NAME_SANITIZE_RE.sub('_', source_id)}"
        f"_{_NAME_SANITIZE_RE.sub('_', operation)}"
    )


def graphql_query_templates_params(op: OperationDefinition) -> bool:
    """Whether a GraphQL operation splices caller input into its query document.

    ``{params.x}`` belongs in ``variables``, where the value travels as data.
    In ``query`` it is substituted into the document text itself
    (``DataSourceExecutor._request_once``), so a caller granted the operation
    can rewrite the query — select other fields, or call a mutation the
    operation never named.  There is no escaping that makes an arbitrary value
    safe in an arbitrary position of a GraphQL document, so such an operation is
    not exposed as an agent-facing tool at all.
    """
    return any(head == "params" for head, _ in extract_refs(op.query or ""))


def _tool_description(defn: Any, op: OperationDefinition) -> str:
    """Description shown to the model, leading with what the operation can do.

    The method and a READ/WRITE marker come first because they are the part
    that changes how the operation should be treated: everything outside
    ``_SAFE_METHODS`` can change or destroy upstream state.  GraphQL sources
    have no per-operation method and are exposed read-only, so they are labelled
    as such rather than inheriting the ``"GET"`` default.
    """
    if getattr(defn, "kind", "http") == "graphql":
        method, marker = "GRAPHQL", "READ"
    else:
        method = (op.method or "GET").upper()
        marker = "READ" if method in _SAFE_METHODS else "WRITE"
    return (
        f"{defn.name or defn.id} — {op.name} [{method} · {marker}]. "
        f"{(defn.description or '').strip()}"
    ).strip()


async def _await_approval(
    container: Any, source: Any, operation: str, params: dict[str, Any]
) -> dict[str, Any] | None:
    """Hold a destructive tool call until a person approves it.

    The agent surface cannot suspend the way a workflow step can: an MCP tool
    call is one HTTP request with an agent blocked on the other end of it, and
    there is no checkpoint to interrupt into. So this blocks — opens the same
    approval case the ``data_source`` step opens, announces it the same way,
    and polls until somebody answers or ``APPROVAL_WAIT_TIMEOUT_SECONDS``
    elapses. To the agent it simply looks like a slow tool.

    Returns ``None`` when the call may proceed, or the error dict the agent
    should see when it may not.
    """
    service = getattr(container, "approval_service", None)
    executor = getattr(container, "data_source_executor", None)
    if service is None or executor is None:
        return None
    settings = getattr(container, "settings", None)
    if settings is not None and not getattr(settings, "approvals_enabled", True):
        return None

    op = source.get_operation(operation)
    if op is None:
        return None
    from app.infrastructure.datasources.destructive import is_destructive
    if not is_destructive(op, source):
        return None

    grant = current_grant()
    run_id = grant.run_id if grant is not None else ""
    agent_id = grant.agent_id if grant is not None else ""
    # The grant names the run, and the run names the workflow — which is what
    # scopes the decision history and decides whether the meta-LLM is allowed
    # to weigh in at all.
    workflow_id = ""
    repository = getattr(container, "run_repository", None)
    if run_id and repository is not None:
        try:
            run = await repository.get(run_id)
            workflow_id = getattr(run, "graph_id", "") if run is not None else ""
        except Exception:
            logger.debug("approval gate: run lookup failed for %s", run_id, exc_info=True)

    plan = await executor.preview(source, operation, params)
    if plan.affected_rows < 1:
        return None

    case = await service.open_case(
        source=source,
        operation=operation,
        method=(op.method or "").upper(),
        params=params,
        affected_rows=plan.affected_rows,
        targets=plan.targets,
        sample=plan.sample,
        workflow_id=workflow_id,
        run_id=run_id,
        agent_id=agent_id,
        surface="mcp",
    )
    logger.info(
        "data source '%s' operation '%s': %d row(s) held for approval (case %s)",
        source.id, operation, plan.affected_rows, case.id,
    )

    if case.status == "pending":
        case = await service.wait_for_decision(case.id) or case
    elif case.veto_deadline is not None:
        case = await service.wait_out_veto(case)

    if case.status == "approved":
        return None
    return {
        "error": (
            f"Refused: deleting {case.affected_rows} row(s) via "
            f"'{source.id}.{operation}' was not approved ({case.status})."
        ),
        "approval_case_id": case.id,
        "approval_status": case.status,
        "reason": case.reason,
    }


def _make_handler(
    source_id: str,
    operation: str,
    params: list[ParamSpec],
    container_getter: Callable[[], Any],
) -> Callable[..., Any]:
    """Build an async handler whose signature mirrors *params*."""

    async def handler(**kwargs: Any) -> Any:
        container = container_getter()
        backend = getattr(container, "data_source_backend", None)
        executor = getattr(container, "data_source_executor", None)
        if backend is None or executor is None:
            return {"error": "Data sources are not configured on this backend"}
        source = await backend.get(source_id)
        if source is None:
            return {"error": f"Data source '{source_id}' not found"}
        try:
            refusal = await _await_approval(container, source, operation, kwargs)
            if refusal is not None:
                return refusal
            return await executor.execute(source, operation, kwargs)
        except Exception as exc:  # noqa: BLE001 — surfaced to the MCP caller
            logger.exception(
                "data source '%s' operation '%s' failed", source_id, operation
            )
            return {"error": str(exc)}

    signature_params = [
        inspect.Parameter(
            spec.name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=_PY_TYPES.get(spec.type, str),
            default=inspect.Parameter.empty if spec.required else None,
        )
        for spec in params
    ]
    handler.__signature__ = inspect.Signature(signature_params)  # type: ignore[attr-defined]
    handler.__annotations__ = {
        spec.name: _PY_TYPES.get(spec.type, str) for spec in params
    }
    handler.__name__ = tool_name_for(source_id, operation)
    return handler


async def rebuild_datasource_tools(
    mcp: FastMCP,
    backend: Any,
    container_getter: Callable[[], Any],
) -> None:
    """Replace all registered tools with the current data source definitions."""
    if backend is None:
        return
    async with _rebuild_lock:
        # Read the definitions inside the lock too, so a concurrent rebuild
        # triggered by another CRUD change can't race this one and clear a
        # tool set built from stale/mixed definitions.
        definitions = await backend.list()
        mcp._tool_manager._tools.clear()
        count = 0
        seen_names: dict[str, str] = {}  # tool name -> "source_id.operation"
        for defn in definitions:
            for op in defn.operations:
                name = tool_name_for(defn.id, op.name)
                origin = f"{defn.id}.{op.name}"
                if name in seen_names:
                    logger.error(
                        "datasources MCP: tool name collision '%s' — '%s' and "
                        "'%s' both sanitize to this name; keeping '%s' and "
                        "skipping '%s'. Rename one of the source/operation "
                        "ids to avoid this.",
                        name, seen_names[name], origin, seen_names[name], origin,
                    )
                    continue
                if defn.kind == "graphql" and graphql_query_templates_params(op):
                    logger.warning(
                        "datasources MCP: not exposing '%s' — its GraphQL query "
                        "templates caller input ({params.*}) into the query "
                        "document, so a caller could rewrite the query. Move "
                        "those placeholders into the operation's `variables` to "
                        "make it grantable.",
                        origin,
                    )
                    continue
                description = _tool_description(defn, op)
                try:
                    mcp.add_tool(
                        _make_handler(defn.id, op.name, op.params, container_getter),
                        name=name,
                        description=description,
                    )
                    seen_names[name] = origin
                    count += 1
                except Exception:
                    logger.exception(
                        "failed to register MCP tool for data source '%s' "
                        "operation '%s'",
                        defn.id, op.name,
                    )
    logger.info("datasources MCP: registered %d tool(s)", count)
