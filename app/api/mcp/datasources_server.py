"""FastMCP server exposing every data source operation as an MCP tool.

The server is mounted by ``app.api.app`` and is the single access path for
data sources: agents reach it through the ``datasources`` MCP integration, and
workflows reach the same executor through the ``data_source`` step type.

One tool is registered per ``<data source> x <operation>``, named
``ds_<source_id>_<operation>``.  The tool's input schema is built from the
operation's declared ``params`` only — upstream operations of the DAG stay
invisible to the caller.  Handlers resolve the executor from the container at
*call* time so CRUD changes and container startup order never matter.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.domain.models.data_source_definition import ParamSpec

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

_mcp: FastMCP | None = None
_rebuild_lock = asyncio.Lock()


def build_datasources_mcp() -> FastMCP:
    """Create the (stateless) FastMCP server instance for data sources."""
    return FastMCP(
        "datasources",
        instructions=(
            "Query configured data sources. Each tool wraps one operation of "
            "one data source; upstream dependencies are resolved automatically."
        ),
        stateless_http=True,
        streamable_http_path=STREAMABLE_HTTP_PATH,
    )


def get_datasources_mcp() -> FastMCP:
    """Return the process-wide data sources MCP server, building it on demand."""
    global _mcp
    if _mcp is None:
        _mcp = build_datasources_mcp()
    return _mcp


def tool_name_for(source_id: str, operation: str) -> str:
    """Return the MCP tool name for one source/operation pair."""
    return (
        f"ds_{_NAME_SANITIZE_RE.sub('_', source_id)}"
        f"_{_NAME_SANITIZE_RE.sub('_', operation)}"
    )


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
                description = (
                    f"{defn.name or defn.id} — {op.name}. "
                    f"{(defn.description or '').strip()}"
                ).strip()
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
