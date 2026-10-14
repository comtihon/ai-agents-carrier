"""The login script stored in the database and edited at runtime.

* ``/auth/login-script`` — read, replace and dry-run it (super admins only);
  a deployment-configured script wins and cannot be changed (409).
* ``LoginService`` — runs the stored script, picks up a change made on another
  replica within ``refresh_seconds``, and reruns every token's login after a
  change.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.api.middleware.auth import OAuthMiddleware
from app.application import teams
from app.application.login import LoginService
from app.infrastructure.auth.authorization import get_current_subject
from app.infrastructure.persistence.login_script_store import InMemoryLoginScriptStore
from app.infrastructure.persistence.team_directory import InMemoryTeamDirectory
from tests.test_login_flow import POLICY, _Clock, _FakeAuth, _claims
from tests.test_resource_visibility import _container
from tests.test_teams import _shim

ROLES_FROM_GROUPS = "function onLogin(ctx) { return { roles: ctx.claims.groups || [] }; }"
PLATFORM_FOR_ALL = "function onLogin(ctx) { return { roles: ['staff', 'platform'] }; }"
TEAM_BY_DEPARTMENT = """
function onLogin(ctx) {
  return {
    roles: ctx.claims.groups,
    name: ctx.user ? ctx.user.name + "!" : null,
    team: ctx.claims.department ? { name: ctx.claims.department, role: "admin" } : undefined,
  };
}
"""


@pytest.fixture
async def api():
    teams._NAMES.clear()
    teams._NAMES_LOADED_AT = 0.0
    directory = InMemoryTeamDirectory()
    red = await directory.create_team("Red")
    await directory.set_membership("alice", red.id, "admin")
    await directory.record_profile("alice", "Alice", "alice@example.com")
    store = InMemoryLoginScriptStore()
    container = _container()
    container.team_directory = directory
    container.login_script_store = store
    app = create_app()
    app.state.container = container
    app.state.authz_policy = POLICY
    service = LoginService(policy=POLICY)
    app.state.login_service = service

    async with AsyncClient(
        transport=ASGITransport(app=_shim(app, directory)), base_url="http://test"
    ) as client:
        def call(user: str):
            headers = {"X-Test-User": user}
            return SimpleNamespace(
                get=lambda url, **kw: client.get(url, headers=headers, **kw),
                put=lambda url, **kw: client.put(url, headers=headers, **kw),
                post=lambda url, **kw: client.post(url, headers=headers, **kw),
            )
        yield SimpleNamespace(
            as_=call, app=app, store=store, directory=directory, red=red, service=service
        )


URL = "/api/v1/auth/login-script"


# ─── GET / PUT ───────────────────────────────────────────────────────────────

async def test_nothing_stored_reads_as_none(api):
    body = (await api.as_("admin").get(URL)).json()
    assert body["source"] == "none" and body["editable"] is True
    assert body["script"] is None and body["updated_at"] is None and body["updated_by"] is None
    assert body["timeout_ms"] == 1000


async def test_put_stores_and_get_reads_it_back(api):
    resp = await api.as_("admin").put(URL, json={"script": ROLES_FROM_GROUPS})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "db" and body["editable"] is True
    assert body["script"] == ROLES_FROM_GROUPS
    assert body["updated_by"] == "admin@example.com" and body["updated_at"]
    assert (await api.as_("admin").get(URL)).json() == body


@pytest.mark.parametrize("empty", [None, "", "   \n"])
async def test_null_or_empty_disables(api, empty):
    await api.as_("admin").put(URL, json={"script": ROLES_FROM_GROUPS})
    body = (await api.as_("admin").put(URL, json={"script": empty})).json()
    assert body["source"] == "none" and body["script"] is None
    assert body["updated_by"] == "admin@example.com"
    assert (await api.store.get()).script is None


@pytest.mark.parametrize(
    "script, fragment",
    [
        ("function onLogin(ctx) { return {;", "SyntaxError"),
        ("function other() {}", "onLogin"),
        ("throw new Error('boom')", "boom"),
    ],
)
async def test_put_rejects_a_script_that_does_not_load(api, script, fragment):
    await api.as_("admin").put(URL, json={"script": ROLES_FROM_GROUPS})
    resp = await api.as_("admin").put(URL, json={"script": script})
    assert resp.status_code == 400
    assert fragment in resp.json()["detail"]
    # The previous script stays.
    assert (await api.store.get()).script == ROLES_FROM_GROUPS


async def test_put_rejects_an_oversized_script(api):
    huge = ROLES_FROM_GROUPS + "//" + "x" * (300 * 1024)
    resp = await api.as_("admin").put(URL, json={"script": huge})
    assert resp.status_code == 400 and "KiB" in resp.json()["detail"]


async def test_only_super_admins(api):
    for user in ("alice", "dave"):
        assert (await api.as_(user).get(URL)).status_code == 403
        assert (await api.as_(user).put(URL, json={"script": ROLES_FROM_GROUPS})).status_code == 403
        assert (await api.as_(user).post(f"{URL}/test", json={"claims": {}})).status_code == 403
    assert await api.store.get() is None


async def test_a_configured_script_wins_and_is_read_only(api):
    api.app.state.settings = api.app.state.settings.model_copy(
        update={"auth_login_script": PLATFORM_FOR_ALL, "auth_login_script_timeout_ms": 500}
    )
    await api.store.save(ROLES_FROM_GROUPS, "someone")
    body = (await api.as_("admin").get(URL)).json()
    assert body == {
        **body,
        "source": "config", "editable": False, "script": PLATFORM_FOR_ALL,
        "timeout_ms": 500, "updated_at": None, "updated_by": None,
    }
    resp = await api.as_("admin").put(URL, json={"script": ROLES_FROM_GROUPS})
    assert resp.status_code == 409
    # The dry run of "the active script" is the configured one.
    result = (await api.as_("admin").post(f"{URL}/test", json={"claims": {}})).json()
    assert result["result"]["roles"] == ["platform", "staff"]


async def test_a_configured_script_file_is_reported(api, tmp_path):
    path = tmp_path / "login.js"
    path.write_text(PLATFORM_FOR_ALL)
    api.app.state.settings = api.app.state.settings.model_copy(
        update={"auth_login_script_path": str(path)}
    )
    body = (await api.as_("admin").get(URL)).json()
    assert (body["source"], body["editable"], body["script"]) == ("config", False, PLATFORM_FOR_ALL)


async def test_without_a_store_nothing_is_editable(api):
    api.app.state.container.login_script_store = None
    body = (await api.as_("admin").get(URL)).json()
    assert (body["source"], body["editable"]) == ("none", False)
    assert (await api.as_("admin").put(URL, json={"script": ROLES_FROM_GROUPS})).status_code == 503


# ─── Dry run ─────────────────────────────────────────────────────────────────

async def test_dry_run_of_a_given_script_writes_nothing(api):
    resp = await api.as_("admin").post(f"{URL}/test", json={
        "script": TEAM_BY_DEPARTMENT,
        "claims": {"sub": "zed", "groups": ["staff", "platform"], "department": "Green"},
    })
    body = resp.json()
    assert resp.status_code == 200 and body["ok"] is True and body["error"] is None
    assert isinstance(body["duration_ms"], (int, float))
    result = body["result"]
    assert result["roles"] == ["platform", "staff"]
    assert result["team"] == {"id": None, "name": "Green", "role": "admin"}
    assert result["team_action"] == "create"
    assert result["is_super_admin"] is True and result["admitted"] is True
    assert "super_admin" in result["permissions"]
    # Nothing written: no team, no user, nothing stored.
    assert [t.name for t in await api.directory.list_teams()] == ["Red"]
    assert await api.directory.get_user("zed") is None
    assert await api.store.get() is None


async def test_dry_run_with_sub_sees_the_stored_user_and_real_teams(api):
    script = (
        "function onLogin(ctx) { return { name: ctx.user.name + '/' + ctx.user.team.name"
        " + '/' + ctx.teams.map(t => t.name).join(','), team: { name: 'Red' } }; }"
    )
    body = (await api.as_("admin").post(f"{URL}/test", json={
        "script": script, "claims": {"roles": ["staff"]}, "sub": "alice",
    })).json()
    assert body["ok"] is True, body
    assert body["result"]["name"] == "Alice/Red/Red"
    assert body["result"]["team"] == {"id": api.red.id, "name": "Red", "role": "member"}
    assert body["result"]["team_action"] == "assign"
    # roles left out: the token's own.
    assert body["result"]["roles"] is None and body["result"]["effective_roles"] == ["staff"]


async def test_dry_run_without_sub_has_no_user(api):
    body = (await api.as_("admin").post(f"{URL}/test", json={
        "script": "function onLogin(ctx) { return { name: String(ctx.user) }; }", "claims": {},
    })).json()
    assert body["result"]["name"] == "null"
    assert body["result"]["team_action"] == "keep" and body["result"]["team"] is None


async def test_dry_run_reports_clear_and_unknown_team(api):
    clear = (await api.as_("admin").post(f"{URL}/test", json={
        "script": "function onLogin() { return { team: null }; }", "claims": {}})).json()
    assert clear["result"]["team_action"] == "clear"
    unknown = (await api.as_("admin").post(f"{URL}/test", json={
        "script": "function onLogin() { return { team: { id: 'nope' } }; }", "claims": {}})).json()
    assert unknown["result"]["team_action"] == "unknown_team"


async def test_dry_run_uses_the_active_script_when_none_is_given(api):
    no_script = (await api.as_("admin").post(f"{URL}/test", json={"claims": {}})).json()
    assert no_script["ok"] is False and "No login script" in no_script["error"]
    await api.as_("admin").put(URL, json={"script": ROLES_FROM_GROUPS})
    body = (await api.as_("admin").post(f"{URL}/test", json={"claims": {"groups": ["x"]}})).json()
    assert body["ok"] is True and body["result"]["roles"] == ["x"]
    assert body["result"]["admitted"] is False


@pytest.mark.parametrize(
    "script, fragment",
    [
        ("function onLogin() { return { roles: 'staff' }; }", "roles must be an array"),
        ("function onLogin() { throw new Error('nope'); }", "nope"),
        ("function onLogin() { while (true) {} }", "timed out"),
        ("function onLogin( {", "SyntaxError"),
    ],
)
async def test_dry_run_reports_errors(api, script, fragment):
    api.app.state.settings = api.app.state.settings.model_copy(
        update={"auth_login_script_timeout_ms": 200}
    )
    body = (await api.as_("admin").post(f"{URL}/test", json={"script": script, "claims": {}})).json()
    assert body["ok"] is False and body["result"] is None
    assert fragment in body["error"]


# ─── Runs at login, picked up across replicas ────────────────────────────────

async def test_put_invalidates_this_replicas_cached_decisions(api):
    store, service = api.store, api.service
    await api.as_("admin").put(URL, json={"script": ROLES_FROM_GROUPS})
    claims = _claims("u1", groups=["staff"])
    assert (await service.decision_for("tok", claims, None, store)).roles == {"staff"}
    await api.as_("admin").put(URL, json={"script": PLATFORM_FOR_ALL})
    # Same token, no waiting: the new script decides.
    assert (await service.decision_for("tok", claims, None, store)).roles == {"staff", "platform"}
    await api.as_("admin").put(URL, json={"script": None})
    assert await service.decision_for("tok", claims, None, store) is None


async def test_another_replica_picks_up_a_change_within_the_refresh_interval():
    store = InMemoryLoginScriptStore()
    clock = _Clock()
    replica = LoginService(policy=POLICY, clock=clock, refresh_seconds=3)
    claims = _claims("u1", groups=["staff"])

    assert await replica.decision_for("tok", claims, None, store) is None
    await store.save(ROLES_FROM_GROUPS, "admin")  # written by another replica
    assert await replica.decision_for("tok", claims, None, store) is None  # not yet
    clock.now += 3
    assert (await replica.decision_for("tok", claims, None, store)).roles == {"staff"}

    await store.save(PLATFORM_FOR_ALL, "admin")
    clock.now += 1
    assert (await replica.decision_for("tok", claims, None, store)).roles == {"staff"}
    clock.now += 2
    # The cached decision for this token is not reused under the new version.
    assert (await replica.decision_for("tok", claims, None, store)).roles == {"staff", "platform"}


async def test_version_is_checked_cheaply_and_the_script_compiled_once():
    store = InMemoryLoginScriptStore()
    await store.save(ROLES_FROM_GROUPS, "admin")
    gets = 0
    original = store.get

    async def counting_get():
        nonlocal gets
        gets += 1
        return await original()

    store.get = counting_get
    clock = _Clock()
    replica = LoginService(policy=POLICY, clock=clock, refresh_seconds=3)
    for i in range(5):
        clock.now += 4
        await replica.decision_for(f"tok{i}", _claims("u1", groups=["staff"]), None, store)
    assert gets == 1


async def test_an_unreadable_store_keeps_the_last_script(caplog):
    store = InMemoryLoginScriptStore()
    await store.save(ROLES_FROM_GROUPS, "admin")
    clock = _Clock()
    replica = LoginService(policy=POLICY, clock=clock, refresh_seconds=3)
    claims = _claims("u1", groups=["staff"])
    assert (await replica.decision_for("t1", claims, None, store)).roles == {"staff"}

    async def broken():
        raise RuntimeError("mongo down")

    store.version = broken
    clock.now += 10
    assert (await replica.decision_for("t2", claims, None, store)).roles == {"staff"}
    assert "cannot read the stored login script" in caplog.text


async def test_a_configured_script_ignores_the_store():
    from app.infrastructure.auth.login_script import LoginScript

    store = InMemoryLoginScriptStore()
    await store.save(ROLES_FROM_GROUPS, "admin")
    service = LoginService(LoginScript(PLATFORM_FOR_ALL), policy=POLICY)
    decision = await service.decision_for("t", _claims("u1", groups=["x"]), None, store)
    assert decision.roles == {"staff", "platform"}


async def test_a_stored_script_drives_login_end_to_end_and_records_roles():
    """Through OAuthMiddleware: the stored script's roles are what the request
    is authorized with and what is recorded on the user."""
    teams._NAMES.clear()
    teams._NAMES_LOADED_AT = 0.0
    directory = InMemoryTeamDirectory()
    store = InMemoryLoginScriptStore()
    await store.save(TEAM_BY_DEPARTMENT, "admin")
    tokens = {"t": _claims("u1", groups=["staff", "platform"], department="Ops")}

    app = FastAPI()
    app.state.container = SimpleNamespace(team_directory=directory, login_script_store=store)

    @app.get("/api/v1/probe")
    async def probe() -> dict:
        subject = get_current_subject()
        return {"super_admin": subject.is_super_admin, "team_role": subject.team_role}

    app.add_middleware(
        OAuthMiddleware, auth_service=_FakeAuth(tokens), policy=POLICY,
        login_service=LoginService(policy=POLICY),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        body = (await c.get("/api/v1/probe", headers={"Authorization": "Bearer t"})).json()
    assert body == {"super_admin": True, "team_role": "admin"}
    user = await directory.get_user("u1")
    assert user.roles == ["platform", "staff"] and user.is_super_admin is True
    assert [t.name for t in await directory.list_teams()] == ["Ops"]
