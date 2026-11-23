"""Thin async adapter over gbrain's MCP streamable-http endpoint.

gbrain *is* an MCP server, so this is an MCP client: one short session per
call, authenticated with ``Authorization: Bearer GBRAIN_API_KEY``.  Agents
never talk to gbrain directly — they talk to ``/mcp/brain`` with a scoped grant,
and the curated tools there call through this client with the carrier's own
credential (see ``app.api.mcp.brain_server``).

Two promises:

* :meth:`GbrainClient.call_tool` returns the tool's parsed content or raises a
  :class:`GbrainError` with a clear message — on a tool error, a timeout, or a
  transport failure alike.
* :meth:`GbrainClient.status` never raises and never reveals the URL or the key.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# How long a status probe result is reused. /health/brain is unauthenticated
# and may be polled; this keeps that from turning into load on gbrain.
STATUS_CACHE_SECONDS = 30.0


class GbrainError(RuntimeError):
    """A gbrain call failed. The message is safe to show to an agent."""


class GbrainTimeout(GbrainError):
    """A gbrain call did not finish within ``GBRAIN_TIMEOUT_SECONDS``."""


@dataclass(frozen=True)
class BrainStatus:
    status: str  # "not_configured" | "unavailable" | "ok"
    detail: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {"status": self.status, "detail": self.detail}


def _leaf(exc: BaseException) -> BaseException:
    """The first non-group exception inside an (anyio) exception group."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _safe_reason(exc: BaseException) -> str:
    """A short failure reason that cannot carry the URL or the bearer.

    httpx and the MCP SDK put the request URL into their messages, so only
    the exception's *kind* (plus an HTTP status code) is ever reported.
    """
    import httpx

    exc = _leaf(exc)
    if isinstance(exc, GbrainError):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return "timed out"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "timed out"
    if isinstance(exc, httpx.HTTPError):
        return "connection failed"
    return type(exc).__name__


def _parse_result(result: Any) -> Any:
    """Turn a ``CallToolResult`` into plain data: JSON if the text is JSON."""
    texts = [
        getattr(item, "text", "")
        for item in (getattr(result, "content", None) or [])
        if getattr(item, "type", None) == "text"
    ]
    text = "\n".join(t for t in texts if t)
    if not text:
        structured = getattr(result, "structuredContent", None)
        return structured if structured is not None else ""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


class _closing:
    """Async context wrapping *inner* that also closes *client* on exit."""

    def __init__(self, inner, client) -> None:
        self._inner = inner
        self._client = client

    async def __aenter__(self):
        try:
            return await self._inner.__aenter__()
        except BaseException:
            await self._client.aclose()
            raise

    async def __aexit__(self, *exc):
        try:
            return await self._inner.__aexit__(*exc)
        finally:
            await self._client.aclose()


class GbrainClient:
    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        tool_prefix: str = "",
        timeout_seconds: float = 15.0,
    ) -> None:
        self._url = url.strip()
        self._api_key = (api_key or "").strip() or None
        self._prefix = tool_prefix or ""
        self._timeout = float(timeout_seconds)
        self._status_cache: tuple[float, BrainStatus] | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> "GbrainClient | None":
        """A client for the configured gbrain, or ``None`` when it is off."""
        if not settings.gbrain_configured():
            return None
        return cls(
            settings.gbrain_url,
            settings.gbrain_api_key,
            tool_prefix=settings.gbrain_tool_prefix,
            timeout_seconds=settings.gbrain_timeout_seconds,
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def upstream_name(self, name: str) -> str:
        return f"{self._prefix}{name}"

    def _transport(self):
        """The SDK's streamable-http client context, newest API first."""
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError:  # mcp < 1.24
            from mcp.client.streamable_http import streamablehttp_client

            return streamablehttp_client(
                self._url, headers=self._headers(), timeout=self._timeout
            )
        import httpx

        # The SDK leaves an injected client open; it is closed with the
        # session because terminate/close happen inside the same context.
        http_client = httpx.AsyncClient(
            headers=self._headers(),
            timeout=httpx.Timeout(self._timeout),
            follow_redirects=True,
        )
        return _closing(streamable_http_client(self._url, http_client=http_client), http_client)

    async def _with_session(self, fn):
        """Open one MCP session, run ``fn(session)``, close it, under the timeout."""
        import anyio
        from mcp import ClientSession

        try:
            with anyio.fail_after(self._timeout):
                async with self._transport() as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        return await fn(session)
        except GbrainError:
            raise
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            # The caller's own cancellation must propagate unchanged.
            raise
        except BaseException as exc:  # noqa: BLE001 — includes anyio exception groups
            leaf = _leaf(exc)
            if isinstance(leaf, GbrainError):
                raise leaf from None
            if isinstance(leaf, TimeoutError):
                raise GbrainTimeout(
                    f"gbrain did not answer within {self._timeout:g}s"
                ) from None
            raise GbrainError(f"gbrain is unreachable ({_safe_reason(exc)})") from None

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call gbrain tool *name* (unprefixed) and return its parsed content."""
        upstream = self.upstream_name(name)
        args = {k: v for k, v in (arguments or {}).items() if v is not None}

        async def _call(session):
            result = await session.call_tool(upstream, args)
            if getattr(result, "isError", False):
                message = _parse_result(result)
                if not isinstance(message, str):
                    message = json.dumps(message, default=str)
                raise GbrainError(
                    f"gbrain tool '{name}' failed: {message[:2000] or 'no detail'}"
                )
            return _parse_result(result)

        return await self._with_session(_call)

    async def list_tool_names(self) -> list[str]:
        async def _list(session):
            listed = await session.list_tools()
            return [t.name for t in listed.tools]

        return await self._with_session(_list)

    async def status(self) -> BrainStatus:
        """Probe gbrain, cached for :data:`STATUS_CACHE_SECONDS`. Never raises."""
        now = time.monotonic()
        cached = self._status_cache
        if cached is not None and now - cached[0] < STATUS_CACHE_SECONDS:
            return cached[1]
        result = await self._probe()
        self._status_cache = (time.monotonic(), result)
        return result

    async def _probe(self) -> BrainStatus:
        try:
            await self.call_tool("get_brain_health", {})
            return BrainStatus("ok")
        except Exception as exc:  # noqa: BLE001
            first = _safe_reason(exc)
        # The health tool may be absent or not granted to this key; a session
        # that can at least list tools still means gbrain is up.
        try:
            await self.list_tool_names()
            return BrainStatus("ok", "get_brain_health unavailable; tool listing succeeded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("gbrain status probe failed: %s / %s", first, _safe_reason(exc))
            return BrainStatus("unavailable", _safe_reason(exc)[:200])
