"""Teams: membership, team admins, super admins, and the "team" visibility.

Cast (in-memory directory, filled per test by ``_seed``):

* red team  — alice (admin), bob (member)
* blue team — carol (member)
* dave      — in no team
* admin     — super admin, in no team
* boss      — super admin, member of blue (which must never narrow them)

The caller is bound by an ASGI shim keyed on ``X-Test-User`` that goes through
:func:`resolve_subject`, exactly as OAuthMiddleware does, so the team comes
from the directory on every request.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.application import management_tools as core
from app.application import teams
from app.application.teams import resolve_subject
from app.infrastructure.auth.authorization import (
    Subject,
    reset_current_subject,
    set_current_subject,
)
from app.infrastructure.persistence.team_directory import InMemoryTeamDirectory
from tests.test_resource_visibility import _agent, _container, _workflow

_SUPER = {"admin", "boss"}


def _shim(app, directory):
    async def wrapped(scope, receive, send):
        user = None
        if scope["type"] == "http":
            user = dict(scope.get("headers") or ()).get(b"x-test-user")
        if not user:
            return await app(scope, receive, send)
        name = user.decode()
        subject = await resolve_subject(
            directory,
            Subject(id=name, name=name.title(), email=f"{name}@example.com", is_super_admin=name in _SUPER),
        )
        token = set_current_subject(subject)
        try:
            return await app(scope, receive, send)
        finally:
            reset_current_subject(token)
    return wrapped


@pytest.fixture
async def env():
    teams._NAMES.clear()
    teams._NAMES_LOADED_AT = 0.0
    directory = InMemoryTeamDirectory()
    red = await directory.create_team("Red")
    blue = await directory.create_team("Blue")
    teams.remember_teams([red, blue])
    await directory.set_membership("alice", red.id, "admin")
    await directory.set_membership("bob", red.id, "member")
    await directory.set_membership("carol", blue.id, "member")
    await directory.set_membership("boss", blue.id, "member")
    container = _container()
    container.team_directory = directory
    app = create_app()
    app.state.container = container
    async with AsyncClient(
        transport=ASGITransport(app=_shim(app, directory)), base_url="http://test"
    ) as client:
        def as_(user):
            headers = {"X-Test-User": user}

            class _Caller:
                async def get(self, url, **kw):
                    return await client.get(url, headers=headers, **kw)

                async def post(self, url, **kw):
                    return await client.post(url, headers=headers, **kw)

                async def put(self, url, **kw):
                    return await client.put(url, headers=headers, **kw)

                async def delete(self, url, **kw):
                    return await client.delete(url, headers=headers, **kw)
            return _Caller()
        yield as_, container, directory, red.id, blue.id


def _ids(resp) -> set[str]:
    assert resp.status_code == 200, resp.text
    return {r["id"] for r in resp.json()}


# ─── /me and the user directory ──────────────────────────────────────────────

async def test_me_reports_team_and_role(env):
    as_, _, _, red, _ = env
    me = (await as_("alice").get("/api/v1/me")).json()
    assert me["team"] == {"id": red, "name": "Red", "role": "admin"}
    assert me["is_super_admin"] is False and me["is_admin"] is False
    assert (await as_("bob").get("/api/v1/me")).json()["team"]["role"] == "member"
    assert (await as_("dave").get("/api/v1/me")).json()["team"] is None
    admin = (await as_("admin").get("/api/v1/me")).json()
    assert admin["is_super_admin"] is True


async def test_callers_are_recorded_in_the_user_directory(env):
    as_, _, directory, _, _ = env
    await as_("erin").get("/api/v1/me")
    erin = await directory.get_user("erin")
    assert erin.name == "Erin" and erin.email == "erin@example.com"
    assert erin.last_seen is not None and erin.team_id is None


# ─── Super admin manages teams ───────────────────────────────────────────────

async def test_super_admin_creates_renames_and_deletes_teams(env):
    as_, _, _, _, _ = env
    resp = await as_("admin").post("/api/v1/teams", json={"name": "Green"})
    assert resp.status_code == 201
    green = resp.json()["id"]
    assert (await as_("admin").post("/api/v1/teams", json={"name": "Green"})).status_code == 409
    renamed = await as_("admin").put(f"/api/v1/teams/{green}", json={"name": "Lime"})
    assert renamed.json()["name"] == "Lime"
    assert {t["name"] for t in (await as_("admin").get("/api/v1/teams")).json()} == {
        "Red", "Blue", "Lime"
    }
    assert (await as_("admin").delete(f"/api/v1/teams/{green}")).status_code == 204
    assert (await as_("admin").get(f"/api/v1/teams")).status_code == 200


async def test_only_super_admin_creates_renames_or_deletes_teams(env):
    as_, _, _, red, _ = env
    for user in ("alice", "bob", "dave"):
        assert (await as_(user).post("/api/v1/teams", json={"name": "X"})).status_code == 403
        assert (await as_(user).put(f"/api/v1/teams/{red}", json={"name": "X"})).status_code == 403
        assert (await as_(user).delete(f"/api/v1/teams/{red}")).status_code == 403


async def test_teams_list_is_own_team_for_non_super_admins(env):
    as_, _, _, red, _ = env
    mine = (await as_("bob").get("/api/v1/teams")).json()
    assert [t["id"] for t in mine] == [red]
    assert {(m["sub"], m["role"]) for m in mine[0]["members"]} == {
        ("alice", "admin"), ("bob", "member"),
    }
    assert (await as_("dave").get("/api/v1/teams")).json() == []


async def test_a_team_with_resources_cannot_be_deleted(env):
    as_, _, _, red, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("shared", visibility="team"))
    resp = await as_("admin").delete(f"/api/v1/teams/{red}")
    assert resp.status_code == 409
    assert "1 resource" in resp.json()["detail"]


async def test_super_admin_moves_users_between_teams(env):
    as_, _, directory, red, blue = env
    resp = await as_("admin").put(f"/api/v1/teams/{blue}/members/bob")
    assert resp.status_code == 200
    assert (await directory.get_user("bob")).team_id == blue


# ─── Team admins ──────────────────────────────────────────────────────────────

async def test_team_admin_adds_a_user_in_no_team(env):
    as_, _, directory, red, _ = env
    resp = await as_("alice").put(f"/api/v1/teams/{red}/members/dave")
    assert resp.status_code == 200
    assert ("dave", "member") in {(m["sub"], m["role"]) for m in resp.json()["members"]}
    assert (await directory.get_user("dave")).team_id == red


async def test_team_admin_cannot_take_a_user_from_another_team(env):
    as_, _, directory, red, blue = env
    resp = await as_("alice").put(f"/api/v1/teams/{red}/members/carol")
    assert resp.status_code == 409
    assert "only a super admin can move users between teams" in resp.json()["detail"]
    assert (await directory.get_user("carol")).team_id == blue


async def test_team_admin_cannot_touch_another_team(env):
    as_, _, _, _, blue = env
    assert (await as_("alice").put(f"/api/v1/teams/{blue}/members/dave")).status_code == 403
    assert (await as_("alice").delete(f"/api/v1/teams/{blue}/members/carol")).status_code == 403


async def test_a_plain_member_cannot_manage_members(env):
    as_, _, _, red, _ = env
    assert (await as_("bob").put(f"/api/v1/teams/{red}/members/dave")).status_code == 403
    assert (await as_("bob").delete(f"/api/v1/teams/{red}/members/alice")).status_code == 403


async def test_team_admin_promotes_demotes_and_removes(env):
    as_, _, directory, red, _ = env
    resp = await as_("alice").put(f"/api/v1/teams/{red}/members/bob", json={"role": "admin"})
    assert (await directory.get_user("bob")).team_role == "admin"
    resp = await as_("alice").put(f"/api/v1/teams/{red}/members/bob", json={"role": "member"})
    assert resp.status_code == 200
    assert (await directory.get_user("bob")).team_role == "member"
    resp = await as_("alice").delete(f"/api/v1/teams/{red}/members/bob")
    assert resp.status_code == 200
    assert (await directory.get_user("bob")).team_id is None


async def test_the_last_admin_of_a_team_with_members_stays(env):
    as_, _, _, red, _ = env
    resp = await as_("alice").put(f"/api/v1/teams/{red}/members/alice", json={"role": "member"})
    assert resp.status_code == 409
    assert (await as_("alice").delete(f"/api/v1/teams/{red}/members/alice")).status_code == 409


async def test_users_list_is_scoped(env):
    as_, _, _, _, _ = env
    everyone = {u["sub"] for u in (await as_("admin").get("/api/v1/users")).json()}
    assert {"alice", "bob", "carol", "boss"} <= everyone
    await as_("dave").get("/api/v1/me")  # recorded, in no team
    seen_by_alice = {u["sub"] for u in (await as_("alice").get("/api/v1/users")).json()}
    # Her own team plus everyone in no team (here dave, and the super admin).
    assert seen_by_alice == {"alice", "bob", "dave", "admin"}
    assert (await as_("bob").get("/api/v1/users")).status_code == 403
    assert (await as_("dave").get("/api/v1/users")).status_code == 403


# ─── Team visibility ─────────────────────────────────────────────────────────

async def test_team_resources_are_visible_to_the_team_and_super_admins_only(env):
    as_, _, _, red, _ = env
    resp = await as_("alice").post("/api/v1/agents", json=_agent("red-agent", visibility="team"))
    assert resp.status_code == 201
    body = resp.json()
    assert body["team_id"] == red and body["team_name"] == "Red"
    assert "red-agent" in _ids(await as_("bob").get("/api/v1/agents"))
    for outsider in ("carol", "dave"):
        assert "red-agent" not in _ids(await as_(outsider).get("/api/v1/agents"))
        assert (await as_(outsider).get("/api/v1/agents/red-agent")).status_code == 404
    # Super admins: one in no team, one in another team.
    for super_admin in ("admin", "boss"):
        assert "red-agent" in _ids(await as_(super_admin).get("/api/v1/agents"))


async def test_team_summaries_carry_team_fields(env):
    as_, container, _, red, _ = env
    await as_("alice").post("/api/v1/workflows", json=_workflow("red-wf", visibility="team"))
    # The list is served from the registry, which these tests do not build;
    # stand in for it with the stored definitions, in the registry's shape.
    stored = container.workflow_backend.inner.docs
    container.yaml_graph_registry.list_definitions = lambda: [
        {
            "id": d.id, "name": d.name, "visibility": d.visibility,
            "owner_id": d.owner_id, "owner_name": d.owner_name, "team_id": d.team_id,
        }
        for d in stored.values()
    ]
    listed = (await as_("bob").get("/api/v1/workflows")).json()
    entry = next(w for w in listed if w["id"] == "red-wf")
    assert entry["team_id"] == red and entry["team_name"] == "Red"
    assert entry["can_edit"] is True  # team members edit team workflows


async def test_members_edit_a_team_workflow_but_not_a_team_agent(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/workflows", json=_workflow("red-wf", visibility="team"))
    await as_("alice").post("/api/v1/agents", json=_agent("red-agent", visibility="team"))

    resp = await as_("bob").put(
        "/api/v1/workflows/red-wf", json={"name": "Renamed by Bob", "steps": []}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["owner_id"] == "alice"  # still Alice's

    agent = (await as_("bob").get("/api/v1/agents/red-agent")).json()
    assert agent["can_edit"] is False
    resp = await as_("bob").put("/api/v1/agents/red-agent", json={"name": "Nope"})
    assert resp.status_code == 403


async def test_only_the_owner_deletes_or_changes_visibility_of_a_team_workflow(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/workflows", json=_workflow("red-wf", visibility="team"))
    resp = await as_("bob").put(
        "/api/v1/workflows/red-wf", json={"name": "x", "steps": [], "visibility": "public"}
    )
    assert resp.status_code == 403
    assert (await as_("bob").delete("/api/v1/workflows/red-wf")).status_code == 403
    assert (await as_("alice").delete("/api/v1/workflows/red-wf")).status_code in (200, 204)


async def test_leaving_a_team_removes_access(env):
    as_, _, directory, _, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("red-agent", visibility="team"))
    await as_("bob").post("/api/v1/agents", json=_agent("bobs-red", visibility="team"))
    await directory.set_membership("bob", None, None)
    ids = _ids(await as_("bob").get("/api/v1/agents"))
    assert "red-agent" not in ids
    assert "bobs-red" in ids  # still the owner
    # ... and the team keeps it.
    assert "bobs-red" in _ids(await as_("alice").get("/api/v1/agents"))


# ─── Choosing the team ───────────────────────────────────────────────────────

async def test_a_member_shares_with_their_own_team_only(env):
    as_, _, _, red, blue = env
    resp = await as_("bob").post(
        "/api/v1/agents", json=_agent("a", visibility="team", team_id=blue)
    )
    assert resp.status_code == 403
    resp = await as_("bob").post("/api/v1/agents", json=_agent("b", visibility="team", team_id=red))
    assert resp.status_code == 201 and resp.json()["team_id"] == red


async def test_without_a_team_you_cannot_share_with_one(env):
    as_, _, _, _, _ = env
    resp = await as_("dave").post("/api/v1/agents", json=_agent("a", visibility="team"))
    assert resp.status_code == 422


async def test_super_admin_picks_any_team_and_needs_to_name_one(env):
    as_, _, _, red, _ = env
    resp = await as_("admin").post("/api/v1/agents", json=_agent("a", visibility="team"))
    assert resp.status_code == 422
    assert "team_id required" in resp.json()["detail"]
    resp = await as_("admin").post(
        "/api/v1/agents", json=_agent("a", visibility="team", team_id="nope")
    )
    assert resp.status_code == 422
    resp = await as_("admin").post(
        "/api/v1/agents", json=_agent("a", visibility="team", team_id=red)
    )
    assert resp.status_code == 201 and resp.json()["team_id"] == red
    # In blue, but sharing with red is fine for a super admin.
    resp = await as_("boss").post(
        "/api/v1/agents", json=_agent("b", visibility="team", team_id=red)
    )
    assert resp.status_code == 201 and resp.json()["team_id"] == red


async def test_super_admin_manages_any_teams_resources(env):
    as_, _, _, _, blue = env
    await as_("alice").post("/api/v1/workflows", json=_workflow("red-wf", visibility="team"))
    resp = await as_("boss").put(
        "/api/v1/workflows/red-wf",
        json={"name": "x", "steps": [], "visibility": "team", "team_id": blue},
    )
    assert resp.status_code == 200 and resp.json()["team_id"] == blue
    assert (await as_("admin").delete("/api/v1/workflows/red-wf")).status_code in (200, 204)


# ─── References between scopes ───────────────────────────────────────────────

async def test_a_public_workflow_cannot_use_a_team_agent(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("red-agent", visibility="team"))
    resp = await as_("alice").post(
        "/api/v1/workflows", json=_workflow("wf", "red-agent", visibility="public")
    )
    assert resp.status_code == 422
    assert "public workflow cannot use team agent" in resp.json()["detail"]


async def test_a_team_workflow_uses_its_own_teams_agent_and_public_ones(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("red-agent", visibility="team"))
    await as_("carol").post("/api/v1/agents", json=_agent("pub", visibility="public"))
    resp = await as_("bob").post(
        "/api/v1/workflows", json=_workflow("wf", "red-agent", visibility="team")
    )
    assert resp.status_code == 201, resp.text
    resp = await as_("bob").post("/api/v1/workflows", json=_workflow("wf2", "pub", visibility="team"))
    assert resp.status_code == 201


async def test_a_team_workflow_cannot_use_a_private_agent(env):
    as_, _, _, _, _ = env
    await as_("bob").post("/api/v1/agents", json=_agent("mine"))
    resp = await as_("bob").post("/api/v1/workflows", json=_workflow("wf", "mine", visibility="team"))
    assert resp.status_code == 422
    assert "team workflow cannot use private agent" in resp.json()["detail"]


async def test_another_teams_agent_is_not_even_visible(env):
    as_, _, _, _, _ = env
    await as_("carol").post("/api/v1/agents", json=_agent("blue-agent", visibility="team"))
    resp = await as_("bob").post(
        "/api/v1/workflows", json=_workflow("wf", "blue-agent", visibility="team")
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Agent 'blue-agent' not found"


async def test_rules_bind_super_admins_too(env):
    as_, _, _, red, _ = env
    await as_("carol").post("/api/v1/agents", json=_agent("blue-agent", visibility="team"))
    resp = await as_("admin").post(
        "/api/v1/workflows",
        json=_workflow("wf", "blue-agent", visibility="team", team_id=red),
    )
    assert resp.status_code == 422
    assert "another team" in resp.json()["detail"]


async def test_a_private_workflow_uses_its_owners_team_agent(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("red-agent", visibility="team"))
    resp = await as_("bob").post("/api/v1/workflows", json=_workflow("wf", "red-agent"))
    assert resp.status_code == 201, resp.text
    assert resp.json()["visibility"] == "private"


async def test_agent_data_source_addons_follow_the_same_rules(env):
    as_, _, _, _, _ = env
    await as_("alice").post(
        "/api/v1/datasources", json={"id": "crm", "base_url": "https://x", "visibility": "team"}
    )
    addon = [{"type": "datasource", "source_id": "crm", "allowed_operations": []}]
    resp = await as_("bob").post(
        "/api/v1/agents", json=_agent("ok", visibility="team", addons=addon)
    )
    assert resp.status_code == 201, resp.text
    resp = await as_("bob").post(
        "/api/v1/agents", json=_agent("bad", visibility="public", addons=addon)
    )
    assert resp.status_code == 422
    assert "public agent cannot use team data source" in resp.json()["detail"]


async def test_scripts_take_the_team_option_too(env):
    as_, _, _, red, _ = env
    resp = await as_("alice").post(
        "/api/v1/scripts", json={"name": "Shared", "code": "output = 1", "visibility": "team"}
    )
    assert resp.status_code in (200, 201), resp.text
    assert resp.json()["team_id"] == red
    names = {s["name"] for s in (await as_("bob").get("/api/v1/scripts")).json()}
    assert "Shared" in names
    names = {s["name"] for s in (await as_("carol").get("/api/v1/scripts")).json()}
    assert "Shared" not in names


# ─── Narrowing ───────────────────────────────────────────────────────────────

async def test_narrowing_is_refused_while_a_wider_or_foreign_referrer_uses_it(env):
    as_, _, _, red, blue = env
    await as_("alice").post("/api/v1/agents", json=_agent("shared", visibility="public"))
    await as_("bob").post("/api/v1/workflows", json=_workflow("red-wf", "shared", visibility="team"))

    resp = await as_("alice").put("/api/v1/agents/shared", json={"visibility": "private"})
    assert resp.status_code == 409
    assert "team workflow 'red-wf'" in resp.json()["detail"]

    # Narrowing to the referrer's own team keeps it working.
    resp = await as_("alice").put("/api/v1/agents/shared", json={"visibility": "team"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["team_id"] == red

    # Moving it to another team would not.
    resp = await as_("admin").put(
        "/api/v1/agents/shared", json={"visibility": "team", "team_id": blue}
    )
    assert resp.status_code == 409


async def test_narrowing_does_not_name_referrers_the_caller_cannot_see(env):
    as_, _, _, _, _ = env
    await as_("carol").post("/api/v1/agents", json=_agent("shared", visibility="public"))
    await as_("bob").post("/api/v1/workflows", json=_workflow("red-wf", "shared", visibility="team"))
    resp = await as_("carol").put("/api/v1/agents/shared", json={"visibility": "team"})
    assert resp.status_code == 409
    assert "red-wf" not in resp.json()["detail"]
    assert "a workflow you cannot see" in resp.json()["detail"]


# ─── Shared layer: management tools ──────────────────────────────────────────

async def test_management_tools_honour_team_visibility(env):
    as_, container, directory, _, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("red-agent", visibility="team"))
    deps = core.deps_from_container(container)
    for user, expected in (("bob", True), ("carol", False)):
        subject = await resolve_subject(directory, Subject(id=user, name=user))
        token = set_current_subject(subject)
        try:
            agents = await container.agent_backend.list()
        finally:
            reset_current_subject(token)
        assert ("red-agent" in {a.id for a in agents}) is expected
    assert deps is not None


# ─── Super admin is its own role, not ADMIN ──────────────────────────────────

@pytest.mark.parametrize(
    "super_admin_roles, sees_everything",
    [(frozenset(), True), (frozenset({"root"}), False)],
)
def test_an_admin_only_user_is_super_admin_only_without_a_distinct_role(
    super_admin_roles, sees_everything
):
    """AUTH_SUPER_ADMIN_ROLES unset: the ADMIN roles are super admins. Set: an
    ADMIN (unsandboxed steps) no longer sees other users' private resources."""
    from app.application.visibility import can_see
    from app.domain.models.agent_definition import AgentDefinition
    from app.infrastructure.auth.authorization import (
        AuthorizationPolicy,
        Permission,
        subject_from_claims,
    )

    policy = AuthorizationPolicy(
        enforce=True,
        access_roles=frozenset({"staff"}),
        admin_roles=frozenset({"ops"}),
        super_admin_roles=super_admin_roles,
    )
    claims = {"sub": "olga", "roles": ["staff", "ops"]}
    assert Permission.ADMIN in policy.permissions_for_claims(claims)
    subject = subject_from_claims(
        claims, Permission.SUPER_ADMIN in policy.evaluate_shadow(claims)
    )
    secret = AgentDefinition(id="secret", name="s", visibility="private", owner_id="alice")
    token = set_current_subject(subject)
    try:
        assert can_see(secret) is sees_everything
    finally:
        reset_current_subject(token)
