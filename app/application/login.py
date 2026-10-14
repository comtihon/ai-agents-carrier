"""Running the login script and applying what it decides.

There is no login endpoint: a bearer token simply arrives with each request. A
"login" is therefore the first time this process sees a given token. That is
when the configured script (app.infrastructure.auth.login_script) runs and its
side effects — the stored profile and team membership — are written. The
decision is then cached under a hash of the token until the token expires, so
every later request with the same token reuses the script's roles without
running it again. A new token (a refresh, a new session) is a new login.

Failure policy. A script that throws, times out or returns something invalid
must neither lock everybody out nor grant anything: the caller is served
exactly as if no script were configured — the token's own roles, the stored
profile and membership untouched — and the error is logged. That failure is
cached only briefly, so a transient problem heals on the next request after it.

Side effects are written only for a caller who will be served: one whose
resulting roles grant no access is refused and leaves nothing behind.

Only users reach this. The management MCP static API key carries no token
claims to run a script on, and keeps its fixed, capped permissions.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Callable

from app.application.teams import remember_teams
from app.infrastructure.auth.authorization import AuthorizationPolicy, Permission, Subject
from app.infrastructure.auth.login_script import (
    KEEP,
    LoginDecision,
    LoginScript,
    LoginScriptError,
    TeamAssignment,
)

logger = logging.getLogger(__name__)


class LoginService:
    """Runs the login script once per token and remembers the answer."""

    def __init__(
        self,
        script: LoginScript,
        *,
        policy: AuthorizationPolicy | None = None,
        max_entries: int = 10_000,
        default_ttl_seconds: float = 300.0,
        max_ttl_seconds: float = 24 * 3600.0,
        failure_ttl_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.script = script
        # Decides whether the caller will be served at all; side effects are
        # written only for callers that are.
        self.policy = policy or AuthorizationPolicy()
        self._max_entries = max_entries
        # For tokens with no ``exp`` claim (opaque tokens validated via userinfo).
        self._default_ttl = default_ttl_seconds
        self._max_ttl = max_ttl_seconds
        self._failure_ttl = failure_ttl_seconds
        self._clock = clock
        self._cache: "OrderedDict[str, tuple[LoginDecision | None, float]]" = OrderedDict()
        self._inflight: dict[str, asyncio.Task] = {}

    async def decision_for(
        self, token: str, claims: dict[str, Any], directory: Any
    ) -> LoginDecision | None:
        """The script's decision for *token*; ``None`` means "as if no script".

        Concurrent first requests with one token share a single run.
        """
        key = hashlib.sha256(token.encode()).hexdigest()
        cached = self._cached(key)
        if cached is not None:
            return cached[0]
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._login(key, claims, directory))
            self._inflight[key] = task
            task.add_done_callback(lambda _t, k=key: self._inflight.pop(k, None))
        return await asyncio.shield(task)

    # ── cache ──

    def _cached(self, key: str) -> tuple[LoginDecision | None] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        decision, expires_at = entry
        if self._clock() >= expires_at:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return (decision,)

    def _remember(self, key: str, decision: LoginDecision | None, ttl: float) -> None:
        self._cache[key] = (decision, self._clock() + ttl)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def _token_ttl(self, claims: dict[str, Any]) -> float:
        exp = claims.get("exp")
        if isinstance(exp, (int, float)) and not isinstance(exp, bool):
            return min(max(float(exp) - self._clock(), 1.0), self._max_ttl)
        return self._default_ttl

    # ── one login ──

    async def _login(self, key: str, claims: dict[str, Any], directory: Any) -> LoginDecision | None:
        sub = str(claims["sub"]) if claims.get("sub") else None
        user, teams = await self._read_directory(directory, sub)
        try:
            decision = await self.script.arun(_context(claims, user, teams))
        except LoginScriptError as exc:
            logger.error(
                "login script failed for %s, using the token's roles: %s", sub or "<no sub>", exc
            )
            self._remember(key, None, self._failure_ttl)
            return None
        except Exception:
            logger.exception("login script crashed for %s, using the token's roles", sub)
            self._remember(key, None, self._failure_ttl)
            return None

        if not self._admitted(claims, decision):
            # Refused at the door (no role granting access): record nothing,
            # so an identity outside this API's tenancy leaves no trace.
            self._remember(key, decision, self._token_ttl(claims))
            return decision
        applied = await self._apply(directory, sub, user, teams, decision)
        # A side effect that did not land is retried soon rather than at the
        # token's expiry; the roles stand either way.
        self._remember(key, decision, self._token_ttl(claims) if applied else self._failure_ttl)
        return decision

    def _admitted(self, claims: dict[str, Any], decision: LoginDecision) -> bool:
        roles = effective_roles(self.policy, claims, decision)
        return Permission.ACCESS in self.policy.permissions_for_roles(roles)

    async def _read_directory(self, directory: Any, sub: str | None) -> tuple[Any, list[Any]]:
        if directory is None:
            return None, []
        try:
            user = await directory.get_user(sub) if sub else None
            teams = await directory.list_teams()
        except Exception:
            logger.exception("login: cannot read the user directory; running the script without it")
            return None, []
        return user, teams

    async def _apply(
        self, directory: Any, sub: str | None, user: Any, teams: list[Any], decision: LoginDecision
    ) -> bool:
        """Write the profile and membership the script decided. True when done."""
        if directory is None or not sub:
            return True
        try:
            if decision.name or decision.email:
                await directory.record_profile(sub, decision.name, decision.email)
            if decision.team is not KEEP:
                await self._apply_team(directory, sub, user, teams, decision.team)
        except Exception:
            logger.exception("login: could not apply the login script's result for %s", sub)
            return False
        return True

    async def _apply_team(
        self,
        directory: Any,
        sub: str,
        user: Any,
        teams: list[Any],
        assignment: TeamAssignment | None,
    ) -> None:
        current_team = getattr(user, "team_id", None)
        current_role = getattr(user, "team_role", None)
        if assignment is None:
            if current_team:
                await directory.set_membership(sub, None, None)
                logger.info("login script removed %s from team %s", sub, current_team)
            return
        team = await _resolve_team(directory, teams, assignment)
        if team is None:
            logger.error(
                "login script assigned %s to unknown team id %r; membership unchanged",
                sub, assignment.id,
            )
            return
        remember_teams([team])
        if current_team != team.id or current_role != assignment.role:
            await directory.set_membership(sub, team.id, assignment.role)
            logger.info(
                "login script put %s in team %s (%s) as %s", sub, team.name, team.id, assignment.role
            )


async def _resolve_team(directory: Any, teams: list[Any], assignment: TeamAssignment) -> Any:
    """The team *assignment* names. A team named but missing is created."""
    from app.infrastructure.persistence.team_directory import TeamNameTaken

    if assignment.id is not None:
        return await directory.get_team(assignment.id)
    for team in teams:
        if team.name == assignment.name:
            return team
    try:
        team = await directory.create_team(assignment.name)
        logger.info("login script created team %s (%s)", team.name, team.id)
        return team
    except TeamNameTaken:
        # Created concurrently (another replica, another login): use that one.
        for team in await directory.list_teams():
            if team.name == assignment.name:
                return team
        raise


def _context(claims: dict[str, Any], user: Any, teams: list[Any]) -> dict[str, Any]:
    """The ``ctx`` argument of ``onLogin``."""
    names = {t.id: t.name for t in teams}
    stored = None
    if user is not None:
        team = None
        if user.team_id:
            team = {
                "id": user.team_id,
                "name": names.get(user.team_id),
                "role": user.team_role or "member",
            }
        stored = {"sub": user.sub, "name": user.name, "email": user.email, "team": team}
    return {
        "claims": claims,
        "user": stored,
        "teams": [{"id": t.id, "name": t.name} for t in teams],
    }


# ─── Used by the authenticating entrypoints ───────────────────────────────────

async def login_decision(
    service: LoginService | None, token: str, claims: dict[str, Any], directory: Any
) -> LoginDecision | None:
    """The decision for this request, or ``None`` when no script is configured."""
    if service is None:
        return None
    return await service.decision_for(token, claims, directory)


def effective_roles(
    policy: AuthorizationPolicy, claims: dict[str, Any], decision: LoginDecision | None
) -> set[str]:
    """The script's roles when it returned some, else the token's."""
    if decision is not None and decision.roles is not None:
        return set(decision.roles)
    return policy.roles_of(claims)


def with_profile(subject: Subject, decision: LoginDecision | None) -> Subject:
    """*subject* with the name/email the script set, if any."""
    if decision is None:
        return subject
    return replace(
        subject,
        name=decision.name or subject.name,
        email=decision.email or subject.email,
    )
