"""Managing the login script at runtime: read it, replace it, dry-run it.

Super admins only. Where the active script comes from:

* ``config`` — the deployment set ``AUTH_LOGIN_SCRIPT`` or
  ``AUTH_LOGIN_SCRIPT_PATH``. It wins, and cannot be changed here (409): the
  deployment's own configuration must not be overridden from the API.
* ``db`` — a super admin stored one (app.infrastructure.persistence.
  login_script_store). Saving validates it first (it must load and define
  ``onLogin``) and invalidates this replica's cached login decisions at once;
  other replicas notice the new version within ``LoginService.refresh_seconds``.
* ``none`` — neither: roles come from the token.

A dry run evaluates a script (the given one, or the active one) against
claims the caller supplies, with the real team list and optionally a real
stored user as ``ctx``, and reports what it would decide. It writes nothing:
no profile, no membership, no team, no cached decision.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from app.application.login import effective_roles, login_context, read_directory
from app.application.teams import require_super_admin
from app.application.visibility import VisibilityError
from app.infrastructure.auth.authorization import (
    AuthorizationPolicy,
    Permission,
    get_current_subject,
)
from app.infrastructure.auth.login_script import (
    KEEP,
    LoginDecision,
    LoginScript,
    LoginScriptConfigError,
    LoginScriptError,
)

# Generous for a role mapping; stops a pasted bundle from landing in MongoDB.
MAX_SCRIPT_BYTES = 256 * 1024


class LoginScriptConflict(VisibilityError):
    status_code = 409


class LoginScriptInvalid(VisibilityError):
    status_code = 400


class LoginScriptUnavailable(VisibilityError):
    status_code = 503


def _configured_source(settings: Any) -> tuple[bool, str | None]:
    """(whether the deployment configured a script, its source if readable)."""
    inline = getattr(settings, "auth_login_script", None)
    path = getattr(settings, "auth_login_script_path", None)
    if inline and inline.strip():
        return True, inline
    if path and path.strip():
        try:
            return True, Path(path.strip()).read_text(encoding="utf-8")
        except OSError:
            return True, None
    return False, None


def _timeout_ms(settings: Any) -> int:
    return int(getattr(settings, "auth_login_script_timeout_ms", 1000) or 1000)


async def _compile(source: str, settings: Any) -> LoginScript:
    """Load *source* the way logins will. Raises :class:`LoginScriptInvalid`."""
    if len(source.encode("utf-8")) > MAX_SCRIPT_BYTES:
        raise LoginScriptInvalid(f"The script is larger than {MAX_SCRIPT_BYTES // 1024} KiB")
    try:
        return await asyncio.to_thread(
            LoginScript, source, timeout_ms=_timeout_ms(settings), origin="login script"
        )
    except LoginScriptConfigError as exc:
        raise LoginScriptInvalid(_short(str(exc))) from exc


def _short(message: str) -> str:
    # V8 appends the offending source line and a caret; the first line says it.
    return message.strip().splitlines()[0] if message.strip() else "invalid script"


def _stamp(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


async def describe(settings: Any, store: Any) -> dict[str, Any]:
    """Where the active script comes from, and the script itself."""
    require_super_admin("read the login script")
    configured, text = _configured_source(settings)
    info: dict[str, Any] = {
        "source": "none",
        "editable": store is not None and not configured,
        "script": None,
        "timeout_ms": _timeout_ms(settings),
        "updated_at": None,
        "updated_by": None,
        # Addition: whether scripts run at all (only with OAuth on).
        "active": bool(getattr(settings, "oauth_enabled", False)),
        "version": None,
    }
    if configured:
        info.update(source="config", script=text)
        return info
    stored = await store.get() if store is not None else None
    if stored is not None:
        info.update(
            updated_at=_stamp(stored.updated_at),
            updated_by=stored.updated_by,
            version=stored.version,
        )
        if stored.script:
            info.update(source="db", script=stored.script)
    return info


async def save(
    settings: Any, store: Any, service: Any, script: str | None
) -> dict[str, Any]:
    """Store *script* (``None`` or blank: disable the stored script)."""
    require_super_admin("change the login script")
    configured, _ = _configured_source(settings)
    if configured:
        raise LoginScriptConflict(
            "The login script is set by the deployment (AUTH_LOGIN_SCRIPT or "
            "AUTH_LOGIN_SCRIPT_PATH) and cannot be changed here"
        )
    if store is None:
        raise LoginScriptUnavailable("No store for the login script is configured")
    text = script if script is not None and script.strip() else None
    if text is not None:
        await _compile(text, settings)
    await store.save(text, _caller())
    if service is not None:
        service.invalidate()
    return await describe(settings, store)


def _caller() -> str | None:
    subject = get_current_subject()
    if subject is None:
        return None
    return subject.email or subject.name or subject.id


async def dry_run(
    settings: Any,
    store: Any,
    directory: Any,
    policy: AuthorizationPolicy,
    *,
    script: str | None,
    claims: dict[str, Any],
    sub: str | None,
) -> dict[str, Any]:
    """Evaluate a script against *claims*; writes nothing."""
    require_super_admin("test the login script")
    started = time.perf_counter()

    def outcome(ok: bool, result: Any = None, error: str | None = None) -> dict[str, Any]:
        return {
            "ok": ok,
            "result": result,
            "error": error,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    source = script if script is not None and script.strip() else None
    if source is None:
        source = await _active_source(settings, store)
        if source is None:
            return outcome(False, error="No login script is active; pass one to test")
    try:
        compiled = await _compile(source, settings)
    except LoginScriptInvalid as exc:
        return outcome(False, error=exc.detail)

    user, teams = await read_directory(directory, sub)
    try:
        decision = await compiled.arun(login_context(claims, user, teams))
    except LoginScriptError as exc:
        return outcome(False, error=str(exc))
    return outcome(True, result=decision_dict(decision, claims, teams, policy))


async def _active_source(settings: Any, store: Any) -> str | None:
    configured, text = _configured_source(settings)
    if configured:
        return text
    stored = await store.get() if store is not None else None
    return stored.script if stored is not None and stored.script else None


def decision_dict(
    decision: LoginDecision, claims: dict[str, Any], teams: list[Any], policy: AuthorizationPolicy
) -> dict[str, Any]:
    """*decision*, normalized, plus what it would mean for the caller.

    ``team_action``: ``keep`` (membership untouched), ``clear`` (in no team),
    ``assign`` (an existing team), ``create`` (a team named but not existing
    yet: a real login would create it) or ``unknown_team`` (an id that does
    not exist: a real login leaves membership unchanged).
    """
    team: dict[str, Any] | None = None
    if decision.team is KEEP:
        action = "keep"
    elif decision.team is None:
        action = "clear"
    else:
        assignment = decision.team
        match = next(
            (
                t for t in teams
                if (assignment.id is not None and t.id == assignment.id)
                or (assignment.id is None and t.name == assignment.name)
            ),
            None,
        )
        if match is not None:
            action = "assign"
            team = {"id": match.id, "name": match.name, "role": assignment.role}
        else:
            action = "unknown_team" if assignment.id is not None else "create"
            team = {"id": assignment.id, "name": assignment.name, "role": assignment.role}
    roles = effective_roles(policy, claims, decision)
    strict = policy.permissions_for_roles_strict(roles)
    return {
        "roles": sorted(decision.roles) if decision.roles is not None else None,
        "name": decision.name,
        "email": decision.email,
        "team": team,
        "team_action": action,
        # What the caller would be authorized with.
        "effective_roles": sorted(roles),
        "permissions": sorted(p.value for p in strict),
        "is_super_admin": Permission.SUPER_ADMIN in strict,
        "admitted": Permission.ACCESS in policy.permissions_for_roles(roles),
    }
