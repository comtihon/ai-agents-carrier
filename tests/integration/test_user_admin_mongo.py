"""User records and the stored login script against a real MongoDB.

* the user directory records effective roles and the super-admin flag, and
  deletes a user;
* the login-script store bumps its version on every save (disabling too), so
  two replicas sharing it agree on what is current.
"""
from __future__ import annotations

from datetime import timezone

import pytest

from app.application.login import LoginService
from app.infrastructure.persistence.login_script_store import MongoLoginScriptStore
from app.infrastructure.persistence.team_directory import MongoTeamDirectory
from tests.test_login_flow import POLICY, _Clock, _claims

_DB = "test_langgraph_integration"
_URI = "mongodb://localhost:27017"


@pytest.fixture
async def directory():
    directory = MongoTeamDirectory(_URI, _DB)
    await directory._users.delete_many({})
    await directory._teams.delete_many({})
    await directory.database["auth_settings"].delete_many({})
    yield directory
    await directory.close()


async def test_roles_are_recorded_and_a_user_deleted(directory):
    red = await directory.create_team("Red")
    await directory.touch_user("u1", "U", "u@example.com", ["staff", "platform"], True)
    await directory.set_membership("u1", red.id, "admin")
    user = await directory.get_user("u1")
    assert user.roles == ["platform", "staff"] and user.is_super_admin is True
    assert user.last_seen.tzinfo is timezone.utc
    # A touch without roles (e.g. a profile recorded by a login) leaves them.
    await directory.record_profile("u1", "New Name", None)
    user = await directory.get_user("u1")
    assert (user.name, user.roles, user.team_id) == ("New Name", ["platform", "staff"], red.id)

    assert await directory.delete_user("u1") is True
    assert await directory.get_user("u1") is None
    assert await directory.members(red.id) == []
    assert await directory.delete_user("u1") is False


async def test_store_versions_every_save(directory):
    store = MongoLoginScriptStore(directory.database)
    assert await store.get() is None and await store.version() == 0
    first = await store.save("function onLogin() { return null; }", "admin@example.com")
    assert first.version == 1 and first.updated_by == "admin@example.com"
    assert first.updated_at.tzinfo is not None
    disabled = await store.save(None, "admin@example.com")
    assert disabled.version == 2 and disabled.script is None
    assert await store.version() == 2
    assert (await store.get()).script is None


async def test_two_replicas_agree_through_the_store(directory):
    store = MongoLoginScriptStore(directory.database)
    clock = _Clock()
    a = LoginService(policy=POLICY, clock=clock, refresh_seconds=3)
    b = LoginService(policy=POLICY, clock=clock, refresh_seconds=3)
    claims = _claims("u1", groups=["staff"])

    await store.save("function onLogin(ctx) { return { roles: ctx.claims.groups }; }", "x")
    a.invalidate()
    assert (await a.decision_for("t", claims, directory, store)).roles == {"staff"}
    assert (await b.decision_for("t", claims, directory, store)).roles == {"staff"}

    await store.save("function onLogin() { return { roles: ['staff', 'platform'] }; }", "x")
    a.invalidate()  # the replica that saved
    assert (await a.decision_for("t", claims, directory, store)).roles == {"staff", "platform"}
    clock.now += 3
    assert (await b.decision_for("t", claims, directory, store)).roles == {"staff", "platform"}
