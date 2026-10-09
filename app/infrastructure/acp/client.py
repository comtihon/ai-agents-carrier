"""ACP client over WebSocket, for agents served by acp-web-proxy.

One `AcpConnection` is one logical proxy connection (one agent process). It
correlates JSON-RPC requests and responses, hands agent→client requests and
notifications to callbacks, and survives socket drops: the proxy numbers every
frame it sends, so the client counts what it received and reconnects with
``Acp-Connection-Id`` + ``Acp-Last-Seq``; the proxy replays the rest. Requests
in flight stay pending across a reconnect — their responses arrive with the
replay.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

logger = logging.getLogger(__name__)

NotificationHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
RequestHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]

CLOSE_REPLAY_GAP = 4409
CLOSE_CONNECTION_ENDED = 4001


class AcpError(RuntimeError):
    def __init__(self, message: str, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class AcpConnectionLost(AcpError):
    """The connection cannot be resumed (agent gone, replay gap, retries spent)."""


class AcpMethodError(AcpError):
    """The agent (or the proxy) answered a request with a JSON-RPC error."""


def to_ws_url(agent_url: str, path: str = "/acp") -> str:
    url = agent_url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    elif url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    return url + path


class AcpConnection:
    def __init__(
        self,
        url: str,
        token: str,
        *,
        on_notification: NotificationHandler | None = None,
        on_request: RequestHandler | None = None,
        connection_id: str | None = None,
        max_reconnects: int = 20,
        reconnect_delay: float = 1.0,
        max_message_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self.url = url
        self.token = token
        self.connection_id = connection_id
        self.received = 0
        self._on_notification = on_notification
        self._on_request = on_request
        self._ids = itertools.count(1)
        self._pending: dict[Any, asyncio.Future] = {}
        self._ws: Any = None
        self._reader: asyncio.Task | None = None
        self._connected = asyncio.Event()
        self._closed = False
        self._lost: AcpError | None = None
        self._max_reconnects = max_reconnects
        self._reconnect_delay = reconnect_delay
        self._max_size = max_message_bytes
        self._handler_tasks: set[asyncio.Task] = set()

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def connect(self, last_seq: int = 0) -> None:
        """Open (or resume, when ``connection_id`` is set) the connection."""
        self.received = last_seq
        await self._open()
        self._reader = asyncio.create_task(self._read_loop(), name=f"acp-reader-{self.connection_id}")

    async def _open(self) -> None:
        headers = {"Authorization": f"Bearer {self.token}"}
        if self.connection_id:
            headers["Acp-Connection-Id"] = self.connection_id
            headers["Acp-Last-Seq"] = str(self.received)
        try:
            ws = await websockets.connect(self.url, additional_headers=headers, max_size=self._max_size, ping_interval=20, ping_timeout=60)
        except InvalidStatus as exc:
            status = exc.response.status_code
            if status in (401, 403, 404):
                raise AcpConnectionLost(f"proxy refused the connection (HTTP {status})") from exc
            raise
        self._ws = ws
        cid = ws.response.headers.get("Acp-Connection-Id") if ws.response is not None else None
        if cid:
            self.connection_id = cid
        self._connected.set()

    async def close(self) -> None:
        self._closed = True
        if self._reader is not None:
            self._reader.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._fail_pending(AcpConnectionLost("connection closed"))

    # ── reading ─────────────────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        attempts = 0
        while not self._closed:
            try:
                async for message in self._ws:
                    attempts = 0
                    self.received += 1
                    self._dispatch(message)
                code = getattr(self._ws, "close_code", None)
            except ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
            except asyncio.CancelledError:
                return
            except Exception as exc:  # network error
                logger.warning("acp %s: socket error: %s", self.connection_id, exc)
                code = None
            if self._closed:
                return
            self._connected.clear()
            if code in (CLOSE_REPLAY_GAP, CLOSE_CONNECTION_ENDED):
                self._lose(AcpConnectionLost(f"proxy ended the connection (close code {code})"))
                return
            attempts += 1
            if attempts > self._max_reconnects:
                self._lose(AcpConnectionLost(f"could not reconnect after {self._max_reconnects} attempts"))
                return
            await asyncio.sleep(min(self._reconnect_delay * attempts, 15))
            try:
                await self._open()
                logger.info("acp %s: resumed at frame %d", self.connection_id, self.received)
            except AcpConnectionLost as exc:
                self._lose(exc)
                return
            except Exception as exc:
                logger.warning("acp %s: reconnect failed: %s", self.connection_id, exc)

    def _lose(self, err: AcpError) -> None:
        self._lost = err
        self._fail_pending(err)

    def _fail_pending(self, err: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(err)
        self._pending.clear()

    def _dispatch(self, message: str | bytes) -> None:
        try:
            msg = json.loads(message)
        except (TypeError, ValueError):
            logger.warning("acp %s: non-JSON frame dropped", self.connection_id)
            return
        if not isinstance(msg, dict):
            return
        method = msg.get("method")
        if method is None:
            fut = self._pending.pop(msg.get("id"), None)
            if fut is None or fut.done():
                return  # a replayed response we already handled
            if "error" in msg and msg["error"] is not None:
                err = msg["error"] or {}
                fut.set_exception(AcpMethodError(str(err.get("message", "error")), err.get("code"), err.get("data")))
            else:
                fut.set_result(msg.get("result"))
            return
        params = msg.get("params") or {}
        if "id" in msg and msg["id"] is not None:
            self._spawn(self._answer(msg["id"], method, params))
        elif self._on_notification is not None:
            self._spawn(self._on_notification(method, params))

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)

    async def _answer(self, req_id: Any, method: str, params: dict[str, Any]) -> None:
        if self._on_request is None:
            await self._send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"client does not handle {method}"}})
            return
        try:
            result = await self._on_request(method, params)
            await self._send({"jsonrpc": "2.0", "id": req_id, "result": result})
        except AcpMethodError as exc:
            await self._send({"jsonrpc": "2.0", "id": req_id, "error": {"code": exc.code or -32603, "message": str(exc)}})
        except Exception as exc:
            logger.exception("acp %s: handler for %s failed", self.connection_id, method)
            await self._send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(exc)}})

    # ── writing ─────────────────────────────────────────────────────────────

    async def _send(self, obj: dict[str, Any]) -> None:
        """Send once a socket is up; a frame that hits a dying socket is resent
        on the next one (the proxy has not seen it, so nothing is duplicated)."""
        data = json.dumps(obj)
        for _ in range(self._max_reconnects + 1):
            if self._lost is not None:
                raise self._lost
            await asyncio.wait_for(self._connected.wait(), timeout=120)
            ws = self._ws
            try:
                await ws.send(data)
                return
            except ConnectionClosed:
                if self._ws is ws:
                    self._connected.clear()
                await asyncio.sleep(0.05)
        raise AcpConnectionLost("could not deliver a frame: connection kept dropping")

    async def request(self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None, req_id: Any = None) -> Any:
        if self._lost is not None:
            raise self._lost
        rid = req_id if req_id is not None else next(self._ids)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            return await asyncio.wait_for(fut, timeout) if timeout else await fut
        finally:
            self._pending.pop(rid, None)

    def expect(self, req_id: Any) -> asyncio.Future:
        """Wait for the response to a request sent by an earlier connection (resume)."""
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        return fut

    def next_id(self) -> int:
        return next(self._ids)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})
