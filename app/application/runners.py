"""Runners: enrollment by number matching, credentials, and live sockets.

A runner is acp-web-proxy in dial mode on a machine carrier cannot reach. It
keeps one **control** socket to carrier; for every ACP connection carrier
needs, carrier sends ``_proxy/open`` and the runner dials a **data** socket
back, which carries ACP exactly like a pod's ``/acp`` socket.

Enrollment (docs: acp-web-proxy docs/runners.md):

1. the machine POSTs an enrollment and gets a 6-digit ``code`` plus a
   ``poll_token``; it shows the code to its user;
2. a signed-in user types the code into carrier's UI, sees the machine's
   details and approves; the approver owns the runner;
3. the machine polls with its ``poll_token`` and receives the runner
   credential (``<runner_id>.<secret>``) once.

The code only selects the enrollment; the credential goes only to whoever holds
the poll token, which never leaves the machine. Codes and secrets are stored
hashed.

Live sockets are held in this process, so carrier serves runners from a single
replica.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any

from websockets.exceptions import ConnectionClosed

from app.domain.models.runner import AgentLimit, Runner, RunnerEnrollment

logger = logging.getLogger(__name__)

CODE_DIGITS = 6
CLOSE_SUPERSEDED = 4000
CLOSE_REVOKED = 4403
AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_code(code: str) -> str | None:
    digits = re.sub(r"[\s-]", "", code or "")
    return digits if re.fullmatch(rf"\d{{{CODE_DIGITS}}}", digits) else None


def clean_agents(agents: Any) -> list[str]:
    """Agent names as a runner may report them (they key Mongo paths)."""
    out: list[str] = []
    for a in agents or []:
        if isinstance(a, str) and AGENT_NAME_RE.match(a) and a not in out:
            out.append(a)
    return out[:32]


class SlidingWindowLimiter:
    """At most ``limit`` events per key in ``window_s`` seconds (in-process)."""

    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def _trim(self, key: str, now: float) -> deque[float]:
        q = self._events[key]
        while q and q[0] <= now - self.window_s:
            q.popleft()
        return q

    def blocked(self, key: str) -> bool:
        return len(self._trim(key, time.monotonic())) >= self.limit

    def hit(self, key: str) -> None:
        now = time.monotonic()
        self._trim(key, now).append(now)


class EnrollmentError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ─── enrollment ──────────────────────────────────────────────────────────────


class EnrollmentService:
    def __init__(self, backend: Any, *, ttl_s: int = 600) -> None:
        self.backend = backend
        self.ttl_s = ttl_s
        self.creates = SlidingWindowLimiter(10, 600)
        self.wrong_codes = SlidingWindowLimiter(5, 600)

    async def create(self, body: dict[str, Any], ip: str | None) -> tuple[RunnerEnrollment, str, str]:
        key = ip or "?"
        if self.creates.blocked(key):
            raise EnrollmentError(429, "too many enrollment requests from this address; try again later")
        self.creates.hit(key)
        now = utcnow()
        await self.backend.delete_expired_enrollments(now - timedelta(days=1))
        for _ in range(20):
            code = f"{secrets.randbelow(10 ** CODE_DIGITS):0{CODE_DIGITS}d}"
            if await self.backend.find_pending_by_code(sha256(code), now) is None:
                break
        else:  # pragma: no cover - a million codes are not all pending
            raise EnrollmentError(503, "no free approval number; try again")
        poll_token = secrets.token_urlsafe(32)
        agents = clean_agents(body.get("agents"))
        default_agent = body.get("default_agent")
        enrollment = RunnerEnrollment(
            id=uuid.uuid4().hex,
            code_hash=sha256(code),
            poll_token_hash=sha256(poll_token),
            hostname=str(body.get("hostname") or "")[:128],
            os=str(body.get("os") or "")[:32],
            arch=str(body.get("arch") or "")[:32],
            tz=(str(body["tz"])[:64] if body.get("tz") else None),
            version=str(body.get("version") or "")[:32],
            agents=agents,
            default_agent=default_agent if default_agent in agents else (agents[0] if agents else None),
            request_ip=ip,
            created_at=now,
            expires_at=now + timedelta(seconds=self.ttl_s),
        )
        await self.backend.create_enrollment(enrollment)
        return enrollment, code, poll_token

    async def lookup(self, code: str, user_key: str) -> RunnerEnrollment:
        if self.wrong_codes.blocked(user_key):
            raise EnrollmentError(429, "too many wrong numbers; wait a few minutes and try again")
        normalized = normalize_code(code)
        found = await self.backend.find_pending_by_code(sha256(normalized), utcnow()) if normalized else None
        if found is None:
            self.wrong_codes.hit(user_key)
            raise EnrollmentError(404, "no machine is waiting with that number (it may have expired)")
        return found

    async def _pending(self, enrollment_id: str, code: str, user_key: str) -> RunnerEnrollment:
        e = await self.backend.get_enrollment(enrollment_id)
        normalized = normalize_code(code)
        if (
            e is None
            or e.status != "pending"
            or e.expires_at <= utcnow()
            or normalized is None
            or not hmac.compare_digest(e.code_hash, sha256(normalized))
        ):
            self.wrong_codes.hit(user_key)
            raise EnrollmentError(404, "no machine is waiting with that number (it may have expired)")
        return e

    async def approve(
        self,
        enrollment_id: str,
        code: str,
        *,
        user_key: str,
        subject: Any,
        name: str | None,
        visibility: str,
        team_id: str | None,
    ) -> Runner:
        e = await self._pending(enrollment_id, code, user_key)
        runner = Runner(
            id=f"rnr-{uuid.uuid4().hex[:10]}",
            name=(name or e.hostname or "runner").strip()[:80],
            hostname=e.hostname,
            os=e.os,
            arch=e.arch,
            tz=e.tz,
            version=e.version,
            agents=list(e.agents),
            default_agent=e.default_agent,
            visibility=visibility,  # type: ignore[arg-type]
            owner_id=getattr(subject, "id", None),
            owner_name=getattr(subject, "name", None) or getattr(subject, "email", None),
            team_id=team_id if visibility == "team" else None,
            approved_by=getattr(subject, "id", None),
            approved_by_name=getattr(subject, "name", None) or getattr(subject, "email", None),
            created_at=utcnow(),
        )
        if not await self.backend.transition_enrollment(
            e.id, "pending",
            {"status": "approved", "runner_id": runner.id, "decided_by": runner.approved_by, "decided_by_name": runner.approved_by_name},
        ):
            raise EnrollmentError(409, "this request was already decided")
        await self.backend.save_runner(runner)
        return runner

    async def deny(self, enrollment_id: str, code: str, *, user_key: str, subject: Any) -> None:
        e = await self._pending(enrollment_id, code, user_key)
        await self.backend.transition_enrollment(
            e.id, "pending", {"status": "denied", "decided_by": getattr(subject, "id", None)}
        )

    async def poll(self, enrollment_id: str, poll_token: str | None) -> dict[str, Any]:
        e = await self.backend.get_enrollment(enrollment_id)
        if e is None or not poll_token or not hmac.compare_digest(e.poll_token_hash, sha256(poll_token)):
            raise EnrollmentError(404, "unknown enrollment")
        if e.status == "consumed":
            raise EnrollmentError(410, "the credential was already handed out")
        if e.status == "denied":
            return {"status": "denied"}
        if e.status == "pending":
            return {"status": "expired"} if e.expires_at <= utcnow() else {"status": "pending"}
        # approved: mint the credential, exactly once.
        if not await self.backend.transition_enrollment(e.id, "approved", {"status": "consumed"}):
            raise EnrollmentError(410, "the credential was already handed out")
        runner = await self.backend.get_runner(e.runner_id or "")
        if runner is None:
            return {"status": "denied"}
        secret = secrets.token_urlsafe(32)
        await self.backend.update_runner(runner.id, {"secret_hash": sha256(secret)})
        return {"status": "approved", "runner_id": runner.id, "runner_name": runner.name, "token": f"{runner.id}.{secret}"}


async def authenticate_runner(backend: Any, authorization: str | None) -> Runner | None:
    """The runner a ``Bearer <runner_id>.<secret>`` header belongs to."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    runner_id, _, secret = token.partition(".")
    if not runner_id or not secret:
        return None
    runner = await backend.get_runner(runner_id)
    if runner is None or not runner.secret_hash:
        return None
    return runner if hmac.compare_digest(runner.secret_hash, sha256(secret)) else None


# ─── live sockets ────────────────────────────────────────────────────────────


class RunnerOffline(RuntimeError):
    pass


class InboundSocket:
    """A data socket the runner dialled, shaped like a ``websockets`` client
    connection for ``AcpConnection``: async iteration over text frames,
    ``send``, ``close``, ``close_code``."""

    def __init__(self, ws: Any, connection_id: str) -> None:
        self.ws = ws
        self.connection_id = connection_id
        self.close_code: int | None = None
        self.done = asyncio.Event()

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    self.close_code = msg.get("code")
                    return
                text = msg.get("text")
                if text is None and msg.get("bytes") is not None:
                    text = msg["bytes"].decode()
                if text is not None:
                    yield text
        except Exception as exc:  # the socket died under us
            logger.debug("runner data socket %s ended: %s", self.connection_id, exc)
        finally:
            self.done.set()

    async def send(self, data: str) -> None:
        try:
            await self.ws.send_text(data)
        except Exception as exc:
            raise ConnectionClosed(None, None) from exc

    async def close(self) -> None:
        try:
            await self.ws.close()
        except Exception:
            pass
        self.done.set()


class RunnerSession:
    """One runner's control socket."""

    def __init__(self, runner_id: str, ws: Any) -> None:
        self.runner_id = runner_id
        self.ws = ws
        self._pending: dict[str, asyncio.Future] = {}
        self._next = 1
        self._lock = asyncio.Lock()
        self.closed = False

    async def send(self, obj: dict[str, Any]) -> None:
        async with self._lock:
            await self.ws.send_text(json.dumps(obj))

    async def request(self, method: str, params: dict[str, Any], timeout: float = 30) -> Any:
        if self.closed:
            raise RunnerOffline(f"runner {self.runner_id} is offline")
        rid = f"c{self._next}"
        self._next += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)

    def resolve(self, msg: dict[str, Any]) -> None:
        fut = self._pending.get(str(msg.get("id")))
        if fut is None or fut.done():
            return
        if msg.get("error"):
            fut.set_exception(RuntimeError(str((msg["error"] or {}).get("message", "error"))))
        else:
            fut.set_result(msg.get("result"))

    def fail_all(self) -> None:
        self.closed = True
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RunnerOffline(f"runner {self.runner_id} disconnected"))
        self._pending.clear()


class RunnerHub:
    """Control sockets of connected runners, and data sockets being opened."""

    def __init__(self) -> None:
        self.sessions: dict[str, RunnerSession] = {}
        self._online: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._data: dict[tuple[str, str], asyncio.Future] = {}
        # Set by the container: where runners and their limits are stored.
        self.backend: Any = None

    def is_online(self, runner_id: str) -> bool:
        s = self.sessions.get(runner_id)
        return s is not None and not s.closed

    async def attach(self, session: RunnerSession) -> None:
        old = self.sessions.get(session.runner_id)
        self.sessions[session.runner_id] = session
        if old is not None and old is not session:
            old.fail_all()
            try:
                await old.ws.close(code=CLOSE_SUPERSEDED)
            except Exception:
                pass
        self._online[session.runner_id].set()

    def detach(self, session: RunnerSession) -> None:
        session.fail_all()
        if self.sessions.get(session.runner_id) is session:
            del self.sessions[session.runner_id]
            self._online[session.runner_id].clear()

    async def drop(self, runner_id: str) -> None:
        """Revoke: close the runner's control socket for good."""
        session = self.sessions.pop(runner_id, None)
        self._online[runner_id].clear()
        if session is not None:
            session.fail_all()
            try:
                await session.ws.close(code=CLOSE_REVOKED)
            except Exception:
                pass

    async def wait_online(self, runner_id: str, timeout: float) -> bool:
        if self.is_online(runner_id):
            return True
        try:
            await asyncio.wait_for(self._online[runner_id].wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return self.is_online(runner_id)

    async def open_socket(
        self, runner_id: str, connection_id: str | None, last_seq: int, *, wait_s: float, resume: bool
    ) -> InboundSocket:
        """Have the runner dial a data socket for *connection_id* (new when None)."""
        if not await self.wait_online(runner_id, wait_s):
            raise RunnerOffline(f"runner {runner_id} did not come online within {int(wait_s)}s")
        cid = connection_id or uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._data[(runner_id, cid)] = fut
        try:
            session = self.sessions[runner_id]
            try:
                await session.request("_proxy/open", {"connection_id": cid, "last_seq": last_seq, "resume": resume})
            except RuntimeError as exc:
                if "unknown connection" in str(exc) or "has ended" in str(exc):
                    from app.infrastructure.acp.client import AcpConnectionLost
                    raise AcpConnectionLost(f"runner {runner_id}: {exc}") from exc
                raise
            return await asyncio.wait_for(fut, 60)
        finally:
            self._data.pop((runner_id, cid), None)

    def offer_data_socket(self, runner_id: str, connection_id: str, sock: InboundSocket) -> bool:
        fut = self._data.get((runner_id, connection_id))
        if fut is None or fut.done():
            return False
        fut.set_result(sock)
        return True

    async def close_connection(self, runner_id: str, connection_id: str) -> None:
        session = self.sessions.get(runner_id)
        if session is None:
            return
        try:
            await session.request("_proxy/close", {"connection_id": connection_id}, timeout=10)
        except Exception:
            pass

    def opener(self, runner_id: str, *, wait_s: float):
        """A ``SocketOpener`` for ``AcpConnection``."""

        async def _open(connection_id: str | None, received: int) -> InboundSocket:
            return await self.open_socket(
                runner_id, connection_id, received, wait_s=wait_s, resume=connection_id is not None
            )

        return _open

    # ── limits ──────────────────────────────────────────────────────────────

    async def set_limit(self, runner_id: str, agent: str, limit: AgentLimit | None) -> None:
        if self.backend is None or not AGENT_NAME_RE.match(agent):
            return
        await self.backend.set_limit(runner_id, agent, limit)

    async def active_limit(self, runner_id: str, agent: str) -> AgentLimit | None:
        if self.backend is None:
            return None
        runner = await self.backend.get_runner(runner_id)
        limit = (runner.limits if runner else {}).get(agent)
        if limit is None:
            return None
        now = utcnow()
        if limit.until is not None:
            return limit if limit.until > now else None
        # Reset time unknown: assume it holds for a while, then let a step try.
        return limit if limit.updated_at and limit.updated_at > now - timedelta(minutes=30) else None


_hub = RunnerHub()


def get_runner_hub() -> RunnerHub:
    return _hub
