"""The login script end to end: OAuthMiddleware -> LoginService -> directory.

A fake auth service maps a bearer token to claims; the script then decides the
caller's roles, profile and team. What is asserted is what the rest of the
backend sees afterwards: the bound permissions and subject, and the users and
teams stored in the (in-memory) directory.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.api.app import _ManagementAuthWrapper
from app.api.middleware.auth import OAuthMiddleware
from app.application import teams
from app.application.login import LoginService
from app.infrastructure.auth.auth_service import AuthError
from app.infrastructure.auth.authorization import (
    AuthorizationPolicy,
    get_current_permissions,
    get_current_subject,
)
from app.infrastructure.auth.login_script import LoginScript
from app.infrastructure.persistence.team_directory import InMemoryTeamDirectory

POLICY = AuthorizationPolicy(
    enforce=True,
    access_roles=frozenset({"staff"}),
    read_roles=frozenset({"staff"}),
    write_roles=frozenset({"staff"}),
    admin_roles=frozenset({"ops"}),
    super_admin_roles=frozenset({"platform"}),
)

# Generic mapping: `groups` -> roles, `department` -> team, `lead` -> team admin.
SCRIPT = """
function onLogin(ctx) {
  const c = ctx.claims;
  if (c.mode === "throw") throw new Error("broken on purpose");
  if (c.mode === "invalid") return { roles: "staff" };
  if (c.mode === "loop") { while (true) {} }
  const out = {};
  if (c.groups) out.roles = c.groups;
  if (c.display) out.name = c.display;
  if (c.department === "") out.team = null;
  else if (c.department) out.team = { name: c.department, role: c.lead ? "admin" : "member" };
  return out;
}
"""


class _CountingScript(LoginScript):
    def __init__(self, source: str = SCRIPT, **kw) -> None:
        super().__init__(source, **kw)
        self.runs = 0

    def run(self, ctx):
        self.runs += 1
        return super().run(ctx)


class _FakeAuth:
    def __init__(self, claims_by_token: dict[str, dict]) -> None:
        self.claims = claims_by_token

    async def validate_token(self, token: str) -> dict:
        if token not in self.claims:
            raise AuthError("Invalid token")
        return dict(self.claims[token])


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def env():
    teams._NAMES.clear()
    teams._NAMES_LOADED_AT = 0.0
    directory = InMemoryTeamDirectory()
    script = _CountingScript(timeout_ms=300)
    clock = _Clock()
    service = LoginService(script, policy=POLICY, clock=clock, failure_ttl_seconds=30)
    tokens: dict[str, dict] = {}

    app = FastAPI()
    app.state.container = SimpleNamespace(team_directory=directory)

    @app.get("/api/v1/probe")
    async def probe() -> dict:
        subject = get_current_subject()
        return {
            "permissions": sorted(p.value for p in get_current_permissions()),
            "super_admin": subject.is_super_admin,
            "name": subject.name,
            "team_id": subject.team_id,
            "team_role": subject.team_role,
        }

    app.add_middleware(
        OAuthMiddleware, auth_service=_FakeAuth(tokens), policy=POLICY, login_service=service
    )

    async def get(token: str):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await c.get("/api/v1/probe", headers={"Authorization": f"Bearer {token}"})

    return SimpleNamespace(
        directory=directory, script=script, clock=clock, tokens=tokens, get=get, service=service
    )


def _claims(sub: str, *, exp: float = 1_000_000.0 + 3600, **extra) -> dict:
    return {"sub": sub, "exp": exp, **extra}


# ── Roles ─────────────────────────────────────────────────────────────────────

async def test_script_roles_drive_permissions_and_super_admin(env) -> None:
    # The token itself carries no roles at all; the script supplies them.
    env.tokens["t"] = _claims("u1", groups=["staff", "platform"])
    body = (await env.get("t")).json()
    assert body["super_admin"] is True
    assert "super_admin" in body["permissions"] and "write" in body["permissions"]


async def test_script_roles_replace_the_tokens_roles(env) -> None:
    env.tokens["t"] = _claims("u1", roles=["staff", "platform", "ops"], groups=["staff"])
    body = (await env.get("t")).json()
    assert body["super_admin"] is False
    assert "admin" not in body["permissions"]
    assert "read" in body["permissions"]


async def test_no_roles_from_the_script_keeps_the_tokens(env) -> None:
    env.tokens["t"] = _claims("u1", roles=["staff", "platform"])
    body = (await env.get("t")).json()
    assert body["super_admin"] is True


async def test_script_roles_can_deny_access(env) -> None:
    env.tokens["t"] = _claims("u1", roles=["staff"], groups=["customer"], department="Sales")
    assert (await env.get("t")).status_code == 403
    # Refused at the door: nothing is recorded, no team is created.
    assert await env.directory.list_teams() == []
    assert await env.directory.get_user("u1") is None


# ── Profile and team ──────────────────────────────────────────────────────────

async def test_assignment_by_name_creates_the_team_once(env) -> None:
    env.tokens["a"] = _claims("alice", groups=["staff"], department="Data", lead=True)
    env.tokens["b"] = _claims("bob", groups=["staff"], department="Data")
    alice = (await env.get("a")).json()
    bob = (await env.get("b")).json()

    [team] = await env.directory.list_teams()
    assert team.name == "Data"
    assert (alice["team_id"], alice["team_role"]) == (team.id, "admin")
    assert (bob["team_id"], bob["team_role"]) == (team.id, "member")
    assert teams.team_name(team.id) == "Data"


async def test_assignment_by_id_of_an_unknown_team_changes_nothing(env) -> None:
    red = await env.directory.create_team("Red")
    await env.directory.set_membership("u1", red.id, "member")
    script = _CountingScript(
        "function onLogin(ctx) { return { roles: ['staff'], team: { id: 'missing' } }; }"
    )
    service = LoginService(script, policy=POLICY)
    decision = await service.decision_for("tok", _claims("u1"), env.directory)
    assert decision.roles == frozenset({"staff"})
    user = await env.directory.get_user("u1")
    assert (user.team_id, user.team_role) == (red.id, "member")


async def test_team_null_removes_membership(env) -> None:
    red = await env.directory.create_team("Red")
    await env.directory.set_membership("u1", red.id, "admin")
    env.tokens["t"] = _claims("u1", groups=["staff"], department="")
    body = (await env.get("t")).json()
    assert body["team_id"] is None
    assert (await env.directory.get_user("u1")).team_id is None


async def test_team_left_out_keeps_manual_membership(env) -> None:
    red = await env.directory.create_team("Red")
    await env.directory.set_membership("u1", red.id, "admin")
    env.tokens["t"] = _claims("u1", groups=["staff"])
    body = (await env.get("t")).json()
    assert (body["team_id"], body["team_role"]) == (red.id, "admin")


async def test_profile_is_stored_and_bound(env) -> None:
    env.tokens["t"] = _claims("u1", name="token name", groups=["staff"], display="Script Name")
    body = (await env.get("t")).json()
    assert body["name"] == "Script Name"
    assert (await env.directory.get_user("u1")).name == "Script Name"


async def test_script_sees_stored_user_and_teams(env) -> None:
    red = await env.directory.create_team("Red")
    await env.directory.set_membership("u1", red.id, "member")
    script = LoginScript(
        "function onLogin(ctx) { return { roles: ['staff'],"
        " name: ctx.user.team.name + '/' + ctx.user.team.role + '/' + ctx.teams.length }; }"
    )
    decision = await LoginService(script, policy=POLICY).decision_for(
        "tok", _claims("u1"), env.directory
    )
    assert decision.name == "Red/member/1"


# ── When it runs ──────────────────────────────────────────────────────────────

async def test_runs_once_per_token(env) -> None:
    env.tokens["t1"] = _claims("u1", groups=["staff"], department="Data")
    env.tokens["t2"] = _claims("u1", groups=["staff"], department="Data")
    for _ in range(3):
        assert (await env.get("t1")).status_code == 200
    assert env.script.runs == 1
    assert (await env.get("t2")).status_code == 200
    assert env.script.runs == 2


async def test_cached_decision_expires_with_the_token(env) -> None:
    env.tokens["t"] = _claims("u1", exp=env.clock.now + 60, groups=["staff"])
    await env.get("t")
    env.clock.now += 59
    await env.get("t")
    assert env.script.runs == 1
    env.clock.now += 2
    await env.get("t")
    assert env.script.runs == 2


async def test_concurrent_first_requests_share_one_run(env) -> None:
    import asyncio

    env.tokens["t"] = _claims("u1", groups=["staff"], department="Data")
    results = await asyncio.gather(*(env.get("t") for _ in range(5)))
    assert all(r.status_code == 200 for r in results)
    assert env.script.runs == 1
    assert len(await env.directory.list_teams()) == 1


async def test_cache_is_bounded() -> None:
    service = LoginService(_CountingScript(), policy=POLICY, max_entries=2)
    for i in range(5):
        await service.decision_for(f"tok{i}", _claims("u1", groups=["staff"]), None)
    assert len(service._cache) == 2


# ── Failure ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["throw", "invalid", "loop"])
async def test_failure_falls_back_to_the_tokens_roles(env, mode, caplog) -> None:
    red = await env.directory.create_team("Red")
    await env.directory.set_membership("u1", red.id, "member")
    env.tokens["t"] = _claims("u1", mode=mode, roles=["staff"], groups=["staff", "platform"])
    body = (await env.get("t")).json()
    # The token's own roles: access, but not the super admin the script's
    # groups would have meant. Membership untouched.
    assert body["super_admin"] is False
    assert "read" in body["permissions"]
    assert (body["team_id"], body["team_role"]) == (red.id, "member")
    assert "login script failed" in caplog.text


async def test_failure_without_token_roles_grants_nothing(env) -> None:
    env.tokens["t"] = _claims("u1", mode="throw", groups=["staff"])
    assert (await env.get("t")).status_code == 403


async def test_failure_is_retried_soon(env) -> None:
    env.tokens["t"] = _claims("u1", mode="throw", roles=["staff"])
    await env.get("t")
    await env.get("t")
    assert env.script.runs == 1
    env.clock.now += 31
    await env.get("t")
    assert env.script.runs == 2


# ── Management MCP ────────────────────────────────────────────────────────────

async def _whoami(request):
    subject = get_current_subject()
    return JSONResponse({
        "permissions": sorted(p.value for p in get_current_permissions()),
        "super_admin": bool(subject and subject.is_super_admin),
        "team_id": subject.team_id if subject else None,
    })


def _management(env, api_key: str | None = "key"):
    inner = Starlette(routes=[Route("/management", _whoami, methods=["POST"])])
    wrapper = _ManagementAuthWrapper(
        inner, api_key=api_key, oauth_enabled=True,
        auth_service=_FakeAuth(env.tokens), policy=POLICY, login_service=env.service,
    )

    async def app(scope, receive, send):
        scope["app"] = SimpleNamespace(state=SimpleNamespace(
            container=SimpleNamespace(team_directory=env.directory)))
        await wrapper(scope, receive, send)

    return app


async def test_management_mcp_user_token_runs_the_script(env) -> None:
    env.tokens["t"] = _claims("u1", groups=["staff", "platform"], department="Data")
    async with AsyncClient(transport=ASGITransport(app=_management(env)), base_url="http://t") as c:
        body = (await c.post("/management", headers={"Authorization": "Bearer t"})).json()
    assert body["super_admin"] is True
    assert body["team_id"] == (await env.directory.list_teams())[0].id
    assert env.script.runs == 1


async def test_management_api_key_never_runs_the_script(env) -> None:
    async with AsyncClient(transport=ASGITransport(app=_management(env)), base_url="http://t") as c:
        body = (await c.post("/management", headers={"Authorization": "Bearer key"})).json()
    assert env.script.runs == 0
    assert body["super_admin"] is False
    assert "super_admin" not in body["permissions"] and "admin" not in body["permissions"]
