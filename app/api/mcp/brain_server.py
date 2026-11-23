"""FastMCP server exposing a curated slice of gbrain to agents at ``/mcp/brain``.

gbrain (the shared company memory) has well over a hundred tools; an agent gets
seven.  Each is a thin wrapper that calls the matching upstream tool through
:class:`app.infrastructure.brain.gbrain_client.GbrainClient` with the carrier's
own credential, which therefore never leaves this process.

Read tools (``read`` and ``write`` grants)::

    brain_search      -> search
    brain_query       -> query
    brain_get_page    -> get_page
    brain_recall      -> recall
    brain_list_pages  -> list_pages

Write tools (``write`` grants only)::

    brain_put_page    -> put_page   (slug forced under GBRAIN_WRITE_NAMESPACE/)
    brain_remember    -> remember   (provenance forced to the carrier run)

Scoping mirrors ``app.api.mcp.datasources_server``: the ASGI gate
(``app.api.app._BrainAuthWrapper``) verifies the caller's signed
:class:`~app.infrastructure.auth.brain_grant.BrainGrant` and publishes it in a
context variable; :class:`ScopedBrainMCP` filters ``list_tools`` **and** refuses
``call_tool`` by it.  Unlike the data sources bridge there is no unscoped
caller — nothing inside the backend uses this endpoint — so *no grant* means
*no tools*, here as well as at the gate.
"""
# No `from __future__ import annotations`: the tool signatures below embed the
# configured namespace in their Field descriptions, which FastMCP must be able
# to evaluate — stringified annotations would reference a closure variable.
import logging
import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from app.api.mcp.datasources_server import _transport_security
from app.infrastructure.auth.brain_grant import BrainGrant

logger = logging.getLogger(__name__)

# Mounted at "/mcp" by create_app, so the full endpoint is /mcp/brain.
STREAMABLE_HTTP_PATH = "/brain"

READ_TOOLS = frozenset({
    "brain_search", "brain_query", "brain_get_page", "brain_recall", "brain_list_pages",
})
WRITE_TOOLS = frozenset({"brain_put_page", "brain_remember"})

_GRANT: ContextVar[BrainGrant | None] = ContextVar("brain_grant", default=None)


def set_current_brain_grant(grant: BrainGrant | None) -> Token:
    return _GRANT.set(grant)


def reset_current_brain_grant(token: Token) -> None:
    _GRANT.reset(token)


def current_brain_grant() -> BrainGrant | None:
    return _GRANT.get()


def allowed_tool_names(grant: BrainGrant | None) -> frozenset[str]:
    """Tools *grant* authorizes; nothing at all without a grant."""
    if grant is None:
        return frozenset()
    return READ_TOOLS | WRITE_TOOLS if grant.can_write else READ_TOOLS


class ScopedBrainMCP(FastMCP):
    """FastMCP that answers only for the tools the caller's grant allows."""

    async def list_tools(self):  # type: ignore[override]
        allowed = allowed_tool_names(_GRANT.get())
        return [t for t in await super().list_tools() if t.name in allowed]

    async def call_tool(self, name: str, arguments: dict[str, Any]):  # type: ignore[override]
        grant = _GRANT.get()
        if name not in allowed_tool_names(grant):
            logger.warning(
                "brain tool '%s' refused for run '%s' (agent '%s', access %s)",
                name,
                grant.run_id if grant else "-",
                grant.agent_id if grant else "-",
                grant.access if grant else "none",
            )
            raise ToolError(f"Tool not granted: {name}")
        return await super().call_tool(name, arguments)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(limit: int | None, default: int, cap: int) -> int:
    try:
        value = int(limit) if limit is not None else default
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, cap))


def namespaced_slug(slug: str, namespace: str) -> str:
    """Return *slug* under ``<namespace>/``, or raise ``ToolError``.

    Agents write only into the carrier's own corner of the shared memory. A
    slug that already starts with the namespace is kept; any other is prefixed.
    Traversal (``..``), absolute slugs, backslashes and empty segments are
    rejected rather than "fixed", because silently rewriting them would put the
    page somewhere the agent did not name.
    """
    ns = (namespace or "carrier").strip().strip("/") or "carrier"
    raw = (slug or "").strip()
    if not raw:
        raise ToolError("slug must not be empty")
    if raw.startswith("/"):
        raise ToolError("slug must be relative (no leading '/')")
    if "\\" in raw:
        raise ToolError("slug must not contain '\\'")
    parts = raw.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ToolError("slug must not contain '..', '.' or empty segments")
    if parts[0] != ns:
        raw = f"{ns}/{raw}"
    return raw


def provenance(grant: BrainGrant) -> str:
    return (
        f"carrier run={grant.run_id or '-'} workflow={grant.workflow_id or '-'} "
        f"agent={grant.agent_id or '-'}"
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def build_brain_mcp(
    container_getter: Callable[[], Any],
    *,
    write_namespace: str = "carrier",
    allowed_hosts: list[str] | None = None,
) -> ScopedBrainMCP:
    """Create the stateless brain MCP server with its seven curated tools."""
    mcp = ScopedBrainMCP(
        "brain",
        instructions=(
            "Shared company memory (gbrain). Search and read what the company "
            "knows; when granted write access, record new knowledge under the "
            f"'{write_namespace}/' namespace."
        ),
        stateless_http=True,
        streamable_http_path=STREAMABLE_HTTP_PATH,
        transport_security=_transport_security(allowed_hosts),
    )

    def _client():
        client = getattr(container_getter(), "brain_client", None)
        if client is None:
            raise ToolError("Shared memory is not configured on this backend")
        return client

    def _grant() -> BrainGrant:
        # ScopedBrainMCP.call_tool already refused a missing grant; this is
        # the belt to that pair of braces.
        grant = _GRANT.get()
        if grant is None:
            raise ToolError("No shared memory access granted")
        return grant

    async def _call(name: str, args: dict[str, Any]) -> Any:
        from app.infrastructure.brain.gbrain_client import GbrainError

        try:
            return await _client().call_tool(name, args)
        except GbrainError as exc:
            raise ToolError(str(exc)) from None

    # --- read ---------------------------------------------------------------

    @mcp.tool(name="brain_search")
    async def brain_search(
        query: Annotated[str, Field(description="Keywords or exact names to look up")],
        limit: int = 10,
    ):
        """Cheap hybrid / keyword search of the shared company memory. Use it
        for names, identifiers and exact phrases. Returns matching pages."""
        return await _call("search", {"query": query, "limit": _clamp(limit, 10, 50)})

    @mcp.tool(name="brain_query")
    async def brain_query(
        query: Annotated[str, Field(description="A question or topic in plain language")],
        limit: int = 10,
    ):
        """Semantic question answering over the shared company memory. Use it
        for concept or topic questions ("how do we handle X?")."""
        return await _call("query", {"query": query, "limit": _clamp(limit, 10, 50)})

    @mcp.tool(name="brain_get_page")
    async def brain_get_page(
        slug: Annotated[str, Field(description="Page slug, e.g. 'people/jane-doe'")],
    ):
        """Read one page of the shared memory in full, including its revision
        (pass that revision as expected_revision when updating the page)."""
        return await _call("get_page", {"slug": slug})

    @mcp.tool(name="brain_recall")
    async def brain_recall(
        entity: Annotated[str | None, Field(description="Entity the facts are about")] = None,
        query: Annotated[str | None, Field(description="Free-text filter")] = None,
        question: Annotated[str | None, Field(description="A question to answer from facts")] = None,
        limit: int = 20,
    ):
        """Recall short remembered facts, by entity, text or question."""
        return await _call("recall", {
            "entity": entity, "query": query, "question": question,
            "limit": _clamp(limit, 20, 100),
        })

    @mcp.tool(name="brain_list_pages")
    async def brain_list_pages(
        slug_prefix: Annotated[str | None, Field(description="Only slugs starting with this")] = None,
        type: Annotated[str | None, Field(description="Only pages of this type")] = None,  # noqa: A002
        limit: int = 50,
    ):
        """List pages of the shared memory, optionally by slug prefix or type
        (at most 100)."""
        return await _call("list_pages", {
            "slug_prefix": slug_prefix, "type": type, "limit": _clamp(limit, 50, 100),
        })

    # --- write --------------------------------------------------------------

    @mcp.tool(name="brain_put_page")
    async def brain_put_page(
        slug: Annotated[str, Field(description=(
            f"Page slug; it is stored under '{write_namespace}/' (prefixed when missing)"
        ))],
        content: Annotated[str, Field(description="Full page content (markdown)")],
        reason: Annotated[str, Field(description="Why this page is written or changed")],
        expected_revision: Annotated[int | str | None, Field(description=(
            "Required when the page already exists: the revision you read with "
            "brain_get_page. Omit for a new page."
        ))] = None,
    ):
        """Create or replace a page in the shared memory under the carrier
        namespace. Never put secrets, credentials or customer personal data in
        it."""
        grant = _grant()
        final_slug = namespaced_slug(slug, write_namespace)
        logger.info(
            "brain write put_page slug='%s' run='%s' workflow='%s' agent='%s'",
            final_slug, grant.run_id, grant.workflow_id, grant.agent_id,
        )
        return await _call("put_page", {
            "slug": final_slug,
            "content": content,
            "content_origin": "inferred",
            "reason": f"{provenance(grant)}: {(reason or '').strip()}",
            "request_id": str(uuid.uuid4()),
            "expected_revision": expected_revision,
        })

    @mcp.tool(name="brain_remember")
    async def brain_remember(
        fact: Annotated[str, Field(description="One short, self-contained fact")],
        entity: Annotated[str, Field(description="Who or what the fact is about")],
        kind: str = "fact",
    ):
        """Remember one fact about an entity in the shared memory. Never store
        secrets, credentials or customer personal data."""
        grant = _grant()
        logger.info(
            "brain write remember entity='%s' run='%s' workflow='%s' agent='%s'",
            entity, grant.run_id, grant.workflow_id, grant.agent_id,
        )
        return await _call("remember", {
            "fact": fact,
            "entity": entity,
            "kind": kind or "fact",
            "provenance": provenance(grant),
            "content_origin": "inferred",
        })

    return mcp
