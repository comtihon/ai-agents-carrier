"""Login script: deployment-supplied JavaScript run when a user logs in.

A deployment may configure a script (``AUTH_LOGIN_SCRIPT`` inline, or
``AUTH_LOGIN_SCRIPT_PATH`` to a file such as a mounted ConfigMap) that defines::

    function onLogin(ctx) {
      // ctx.claims  the validated token's claims
      // ctx.user    the stored user ({sub, name, email, team}) or null
      // ctx.teams   every existing team ([{id, name}])
      return { roles: [...], name: "...", email: "...", team: {...} | null };
    }

Every field of the returned object is optional; see :func:`parse_decision` for
the exact contract. It lets one deployment map whatever its identity provider
puts in a token (groups, custom claims, email domains) onto carrier's roles and
teams without code changes.

The script runs in an embedded V8 isolate (``mini-racer``) with no network,
filesystem, process or module access: the only input is ``ctx`` and the only
output is the returned value, serialized through JSON. Each run gets a fresh
isolate, so nothing one login computes can leak into another, and is bounded by
a wall-clock timeout and a heap limit.

This module only *evaluates* the script and validates what it returns. When it
runs and what its answer changes is app.application.login.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEAM_ROLES = ("member", "admin")

# Heap limit for one run. A role mapping needs a few kilobytes; this only stops
# a runaway script from taking the backend's memory with it.
DEFAULT_MAX_MEMORY_BYTES = 64 * 1024 * 1024

_MAX_ROLES = 256
_MAX_STRING = 512
_RESULT_KEYS = frozenset({"roles", "name", "email", "team"})
_TEAM_KEYS = frozenset({"id", "name", "role"})

# Calls onLogin with ctx parsed from a JSON string literal (never spliced in as
# code) and hands the result back as JSON text, which is the only shape that
# crosses the boundary. A promise is refused rather than serialized as "{}".
_INVOKE = """(function () {{
  if (typeof onLogin !== "function") throw new Error("onLogin is not defined");
  const result = onLogin(JSON.parse({payload}));
  if (result !== null && typeof result === "object" && typeof result.then === "function") {{
    throw new Error("onLogin must return a value, not a promise");
  }}
  return result === undefined ? "null" : JSON.stringify(result);
}})()"""


class LoginScriptError(Exception):
    """The script failed, timed out, or returned something invalid."""


class LoginScriptConfigError(ValueError):
    """The configured script cannot be used at all; startup must fail."""


class _Keep:
    """Marker for "leave the stored team membership alone"."""

    _instance: "_Keep | None" = None

    def __new__(cls) -> "_Keep":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "KEEP"


KEEP = _Keep()


@dataclass(frozen=True)
class TeamAssignment:
    """Put the user in one team, named by id or by name, with *role*."""

    role: str = "member"
    id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class LoginDecision:
    """What a script decided. ``None`` / ``KEEP`` fields change nothing."""

    # Replaces the roles read from the token, for permission mapping only.
    roles: frozenset[str] | None = None
    name: str | None = None
    email: str | None = None
    # KEEP: leave membership as stored; None: in no team; else assign.
    team: TeamAssignment | None | _Keep = field(default=KEEP)


# ─── Output validation ────────────────────────────────────────────────────────

def _string(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise LoginScriptError(f"{what} must be a string, got {type(value).__name__}")
    value = value.strip()
    if not value:
        raise LoginScriptError(f"{what} must not be empty")
    if len(value) > _MAX_STRING:
        raise LoginScriptError(f"{what} is longer than {_MAX_STRING} characters")
    return value


def _roles(value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        raise LoginScriptError(f"roles must be an array of strings, got {type(value).__name__}")
    if len(value) > _MAX_ROLES:
        raise LoginScriptError(f"roles holds more than {_MAX_ROLES} entries")
    return frozenset(_string(role, "every role") for role in value)


def _team(value: Any) -> TeamAssignment | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise LoginScriptError(f"team must be an object or null, got {type(value).__name__}")
    unknown = set(value) - _TEAM_KEYS
    if unknown:
        raise LoginScriptError(f"team has unknown field(s): {', '.join(sorted(unknown))}")
    has_id = value.get("id") is not None
    has_name = value.get("name") is not None
    if has_id == has_name:
        raise LoginScriptError("team must name exactly one of 'id' or 'name'")
    role = value.get("role")
    role = "member" if role is None else role
    if role not in TEAM_ROLES:
        raise LoginScriptError(f"team.role must be one of {', '.join(TEAM_ROLES)}")
    if has_id:
        return TeamAssignment(role=role, id=_string(value["id"], "team.id"))
    return TeamAssignment(role=role, name=_string(value["name"], "team.name"))


def parse_decision(raw: Any) -> LoginDecision:
    """Validate what ``onLogin`` returned (already JSON-decoded).

    The contract, strictly: ``null``/``undefined`` changes nothing; otherwise an
    object whose only keys are

    * ``roles`` — array of non-empty strings, replacing the token's roles;
    * ``name``, ``email`` — non-empty strings, replacing the stored profile;
    * ``team`` — ``null`` (in no team) or ``{id | name, role?}`` with ``role``
      ``"member"`` (default) or ``"admin"``.

    ``null`` for ``roles``/``name``/``email`` means the same as leaving it out;
    leaving ``team`` out keeps the stored membership. Any deviation — an unknown
    key included, so a typo cannot silently do nothing — rejects the whole
    result rather than applying part of it.
    """
    if raw is None:
        return LoginDecision()
    if not isinstance(raw, dict):
        raise LoginScriptError(f"onLogin must return an object, got {type(raw).__name__}")
    unknown = set(raw) - _RESULT_KEYS
    if unknown:
        raise LoginScriptError(f"unknown field(s) in result: {', '.join(sorted(unknown))}")
    return LoginDecision(
        roles=_roles(raw["roles"]) if raw.get("roles") is not None else None,
        name=_string(raw["name"], "name") if raw.get("name") is not None else None,
        email=_string(raw["email"], "email") if raw.get("email") is not None else None,
        team=_team(raw["team"]) if "team" in raw else KEEP,
    )


# ─── Engine ───────────────────────────────────────────────────────────────────

class LoginScript:
    """One configured script. :meth:`run` evaluates it in a fresh isolate."""

    def __init__(
        self,
        source: str,
        *,
        timeout_ms: int = 1000,
        max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES,
        origin: str = "login script",
    ) -> None:
        if not source or not source.strip():
            raise LoginScriptConfigError(f"{origin} is empty")
        self.source = source
        self.origin = origin
        self.timeout_ms = timeout_ms
        self.max_memory_bytes = max_memory_bytes
        self._check()

    def _check(self) -> None:
        """Fail fast on a script that does not even define ``onLogin``.

        Evaluated on a worker thread: mini-racer refuses a timeout on a thread
        that runs an event loop, and the app may be built inside one.
        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(self._check_blocking).result()

    def _check_blocking(self) -> None:
        try:
            with _isolate() as ctx:
                ctx.eval(self.source, timeout_sec=self.timeout_ms / 1000,
                         max_memory=self.max_memory_bytes)
                kind = ctx.eval("typeof onLogin", timeout_sec=self.timeout_ms / 1000)
        except Exception as exc:
            raise LoginScriptConfigError(f"{self.origin} failed to load: {exc}") from exc
        if kind != "function":
            raise LoginScriptConfigError(f"{self.origin} does not define a function onLogin(ctx)")

    def run(self, ctx: dict[str, Any]) -> LoginDecision:
        """Evaluate ``onLogin(ctx)``. Raises :class:`LoginScriptError`.

        Blocking, and must not be called on an event-loop thread (mini-racer
        refuses timeouts there); async callers use :meth:`arun`.
        """
        try:
            payload = json.dumps(json.dumps(ctx, default=str))
        except (TypeError, ValueError) as exc:
            raise LoginScriptError(f"login context is not serializable: {exc}") from exc

        deadline = time.monotonic() + self.timeout_ms / 1000
        try:
            with _isolate() as isolate:
                isolate.eval(self.source, timeout_sec=self._remaining(deadline),
                             max_memory=self.max_memory_bytes)
                text = isolate.eval(_INVOKE.format(payload=payload),
                                    timeout_sec=self._remaining(deadline),
                                    max_memory=self.max_memory_bytes)
        except LoginScriptError:
            raise
        except TimeoutError as exc:
            raise LoginScriptError(f"timed out after {self.timeout_ms} ms") from exc
        except Exception as exc:
            raise LoginScriptError(f"{type(exc).__name__}: {_first_line(exc)}") from exc

        if not isinstance(text, str):
            raise LoginScriptError("onLogin returned a value that is not JSON-serializable")
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise LoginScriptError(f"onLogin returned invalid JSON: {exc}") from exc
        return parse_decision(raw)

    async def arun(self, ctx: dict[str, Any]) -> LoginDecision:
        """:meth:`run` off the event loop."""
        return await asyncio.to_thread(self.run, ctx)

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise LoginScriptError("timed out")
        return remaining


# mini-racer drives each isolate from an asyncio loop. Left to itself it starts
# (and stops) a loop thread per isolate, and a V8 callback that arrives after a
# timed-out isolate's loop has stopped is reported as an error. One long-lived
# loop thread, shared by every isolate, avoids both.
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _shared_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()
            threading.Thread(
                target=loop.run_forever, name="login-script-v8", daemon=True
            ).start()
            _loop = loop
        return _loop


def _isolate():
    """A fresh V8 isolate, closed when the ``with`` block ends."""
    from py_mini_racer import mini_racer

    return mini_racer(event_loop=_shared_loop())


def _first_line(exc: Exception) -> str:
    text = str(exc).strip()
    # V8 messages start with "<anonymous>:LINE: Error: ..." followed by the
    # source line and a caret; the first line is the useful part.
    return text.splitlines()[0] if text else type(exc).__name__


def load_login_script(settings: Any) -> LoginScript | None:
    """The configured script, or ``None`` when none is configured.

    Raises :class:`LoginScriptConfigError` when both sources are set, the file
    cannot be read, or the script does not load: a deployment that asked for a
    login script must not silently start without one.
    """
    inline = getattr(settings, "auth_login_script", None)
    path = getattr(settings, "auth_login_script_path", None)
    inline = inline if inline and inline.strip() else None
    path = path.strip() if path and path.strip() else None
    if inline and path:
        raise LoginScriptConfigError(
            "Set AUTH_LOGIN_SCRIPT or AUTH_LOGIN_SCRIPT_PATH, not both"
        )
    if not (inline or path):
        return None
    timeout_ms = int(getattr(settings, "auth_login_script_timeout_ms", 1000) or 1000)
    if path:
        try:
            source = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise LoginScriptConfigError(f"Cannot read AUTH_LOGIN_SCRIPT_PATH {path}: {exc}") from exc
        script = LoginScript(source, timeout_ms=timeout_ms, origin=f"login script {path}")
    else:
        script = LoginScript(inline, timeout_ms=timeout_ms, origin="AUTH_LOGIN_SCRIPT")
    logger.info("Login script loaded from %s (timeout %d ms)", script.origin, timeout_ms)
    return script
