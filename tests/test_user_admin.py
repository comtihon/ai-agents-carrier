"""User management for super admins: the user list's shape, setting or
clearing a user's team, deleting a user, and the roles recorded per user.

Uses the cast and the ``X-Test-User`` shim of tests.test_teams.
"""
from __future__ import annotations

from datetime import datetime

from app.application.teams import resolve_subject
from app.infrastructure.auth.authorization import Subject
from app.infrastructure.persistence.team_directory import InMemoryTeamDirectory
from tests.test_teams import env  # noqa: F401 — fixture


def _by_sub(resp) -> dict[str, dict]:
    assert resp.status_code == 200, resp.text
    return {u["sub"]: u for u in resp.json()}


# ─── GET /users ──────────────────────────────────────────────────────────────

async def test_user_items_carry_team_roles_and_last_seen(env):
    as_, _, _, red, _ = env
    await as_("admin").get("/api/v1/me")
    users = _by_sub(await as_("admin").get("/api/v1/users"))
    alice = users["alice"]
    assert alice["team"] == {"id": red, "name": "Red", "role": "admin"}
    assert alice["is_super_admin"] is False and alice["roles"] == []
    # Never seen (seeded directly): no last_seen yet.
    assert users["bob"]["last_seen"] is None
    admin = users["admin"]
    assert admin["team"] is None and admin["is_super_admin"] is True
    datetime.fromisoformat(admin["last_seen"])
    assert set(alice) >= {"sub", "name", "email", "team", "is_super_admin", "roles", "last_seen"}
    # Flat fields stay for existing clients.
    assert (alice["team_id"], alice["team_name"], alice["team_role"]) == (red, "Red", "admin")


async def test_team_admins_keep_their_scoped_list(env):
    as_, _, _, _, _ = env
    subs = set(_by_sub(await as_("alice").get("/api/v1/users")))
    assert "carol" not in subs and {"alice", "bob"} <= subs
    assert (await as_("bob").get("/api/v1/users")).status_code == 403


# ─── PUT /users/{sub}/team ───────────────────────────────────────────────────

async def test_super_admin_sets_and_clears_a_users_team(env):
    as_, _, directory, red, blue = env
    resp = await as_("admin").put("/api/v1/users/carol/team", json={"team_id": red, "role": "admin"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["team"] == {"id": red, "name": "Red", "role": "admin"}
    assert (await directory.get_user("carol")).team_id == red

    resp = await as_("admin").put("/api/v1/users/carol/team", json={"team_id": None})
    assert resp.status_code == 200 and resp.json()["team"] is None
    assert (await directory.get_user("carol")).team_id is None

    # Default role is member.
    resp = await as_("admin").put("/api/v1/users/carol/team", json={"team_id": blue})
    assert resp.json()["team"]["role"] == "member"


async def test_super_admin_is_not_bound_by_the_last_admin_rule(env):
    as_, _, directory, red, _ = env
    resp = await as_("admin").put("/api/v1/users/alice/team", json={"team_id": None})
    assert resp.status_code == 200
    assert (await directory.get_user("alice")).team_id is None


async def test_set_user_team_rejects_unknowns(env):
    as_, _, _, red, _ = env
    assert (await as_("admin").put(
        "/api/v1/users/carol/team", json={"team_id": "nope"})).status_code == 404
    assert (await as_("admin").put(
        "/api/v1/users/nobody/team", json={"team_id": red})).status_code == 404
    assert (await as_("admin").put(
        "/api/v1/users/carol/team", json={"team_id": red, "role": "owner"})).status_code == 422


async def test_only_a_super_admin_sets_a_users_team(env):
    as_, _, directory, red, _ = env
    await as_("dave").get("/api/v1/me")
    # Even a team admin adding to their own team must use /teams/{id}/members.
    resp = await as_("alice").put("/api/v1/users/dave/team", json={"team_id": red})
    assert resp.status_code == 403
    assert (await directory.get_user("dave")).team_id is None


# ─── DELETE /users/{sub} ─────────────────────────────────────────────────────

async def test_super_admin_deletes_a_user(env):
    as_, _, directory, red, _ = env
    assert (await as_("admin").delete("/api/v1/users/bob")).status_code == 204
    assert await directory.get_user("bob") is None
    assert "bob" not in _by_sub(await as_("admin").get("/api/v1/users"))
    assert "bob" not in {m["sub"] for m in (await as_("admin").get("/api/v1/teams")).json()[1]["members"]}
    assert (await as_("admin").delete("/api/v1/users/bob")).status_code == 404


async def test_a_deleted_user_who_calls_again_is_recorded_again_in_no_team(env):
    as_, _, directory, _, _ = env
    await as_("bob").get("/api/v1/me")
    assert (await as_("admin").delete("/api/v1/users/bob")).status_code == 204
    # Recorded again at once, not after the touch interval.
    me = (await as_("bob").get("/api/v1/me")).json()
    assert me["team"] is None
    assert (await directory.get_user("bob")) is not None


async def test_only_a_super_admin_deletes_users(env):
    as_, _, directory, _, _ = env
    assert (await as_("alice").delete("/api/v1/users/bob")).status_code == 403
    assert await directory.get_user("bob") is not None


# ─── Roles recorded on the user ──────────────────────────────────────────────

async def test_touch_records_roles_and_rewrites_when_they_change():
    directory = InMemoryTeamDirectory()
    writes = []
    original = directory._upsert_seen

    async def counting(*args):
        writes.append(args)
        await original(*args)

    directory._upsert_seen = counting
    subject = Subject(id="u1", name="U", roles=frozenset({"staff"}), is_super_admin=False)
    await resolve_subject(directory, subject)
    await resolve_subject(directory, subject)
    assert len(writes) == 1  # within the touch interval
    user = await directory.get_user("u1")
    assert (user.roles, user.is_super_admin) == (["staff"], False)

    promoted = Subject(id="u1", name="U", roles=frozenset({"staff", "platform"}), is_super_admin=True)
    await resolve_subject(directory, promoted)
    assert len(writes) == 2  # changed roles are written at once
    user = await directory.get_user("u1")
    assert (user.roles, user.is_super_admin) == (["platform", "staff"], True)


async def test_a_subject_without_roles_leaves_stored_roles_alone():
    directory = InMemoryTeamDirectory()
    await directory.touch_user("u1", "U", None, ["staff"], True)
    directory.forget_touch("u1")
    await directory.touch_user("u1", "U", None)
    user = await directory.get_user("u1")
    assert (user.roles, user.is_super_admin) == (["staff"], True)
