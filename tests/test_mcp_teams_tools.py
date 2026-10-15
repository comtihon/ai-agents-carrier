"""Team, user and visibility management through the management MCP tools.

Every call goes through the registered FastMCP tool (the same wrappers an MCP
client reaches), with the caller bound the way _ManagementAuthWrapper binds
one: a Subject whose team comes from the directory via resolve_subject, on
every call.

Cast:

* red team  — alice (admin), bob (member)
* blue team — carol (admin), erin (member)
* dave      — in no team
* sam smith, sam jones — in no team (an ambiguous "sam")
* root      — super admin, in no team
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.api.mcp.management_server import (
    build_management_mcp,
    register_management_tools,
)
from app.application import teams
from app.application.teams import resolve_subject
from app.application.visibility import unwrap
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.auth.authorization import (
    Permission,
    Subject,
    reset_current_permissions,
    reset_current_subject,
    set_current_permissions,
    set_current_subject,
)
from app.infrastructure.persistence.team_directory import InMemoryTeamDirectory
from tests.test_resource_visibility import _container

_PEOPLE = {
    # sub: (name, email)
    "alice": ("Alice Adams", "alice@example.com"),
    "bob": ("Bob Brown", "bob@example.com"),
    "carol": ("Carol Clark", "carol@example.com"),
    "erin": ("Erin Evans", "erin@example.com"),
    "dave": ("Dave Davis", "dave@example.com"),
    "sam1": ("Sam Smith", "sam.smith@example.com"),
    "sam2": ("Sam Jones", "sam.jones@example.com"),
    "root": ("Root Admin", "root@example.com"),
}
_SUPER = {"root"}


def _text(result) -> str:
    # FastMCP.call_tool returns content blocks, or (content, structured).
    if isinstance(result, tuple):
        result = result[0]
    return "\n".join(getattr(block, "text", "") for block in result)


def _registry_over(container):
    """A registry stand-in listing what the workflow store holds, as the
    real registry does after refresh_runner."""
    registry = MagicMock()

    def list_definitions():
        return [
            {
                "id": w.id, "name": w.name, "visibility": w.visibility,
                "owner_id": w.owner_id, "team_id": w.team_id,
            }
            for w in unwrap(container.workflow_backend).docs.values()
        ]

    registry.list_definitions.side_effect = list_definitions
    return registry


@pytest.fixture
async def env():
    teams._NAMES.clear()
    teams._NAMES_LOADED_AT = 0.0
    directory = InMemoryTeamDirectory()
    for sub, (name, email) in _PEOPLE.items():
        await directory.touch_user(sub, name, email, [], sub in _SUPER)
    red = await directory.create_team("Red")
    blue = await directory.create_team("Blue")
    teams.remember_teams([red, blue])
    await directory.set_membership("alice", red.id, "admin")
    await directory.set_membership("bob", red.id, "member")
    await directory.set_membership("carol", blue.id, "admin")
    await directory.set_membership("erin", blue.id, "member")

    container = _container()
    container.team_directory = directory
    container.yaml_graph_registry = _registry_over(container)
    mcp = build_management_mcp()
    register_management_tools(mcp, lambda: container)

    async def call(caller: str | None, tool: str, /, permissions=None, **args) -> str:
        subject = None
        if caller is not None:
            name, email = _PEOPLE.get(caller, (caller, None))
            subject = await resolve_subject(
                directory,
                Subject(id=caller, name=name, email=email, is_super_admin=caller in _SUPER),
            )
        token = set_current_subject(subject)
        perm_token = set_current_permissions(permissions) if permissions is not None else None
        try:
            return _text(await mcp.call_tool(tool, args))
        finally:
            if perm_token is not None:
                reset_current_permissions(perm_token)
            reset_current_subject(token)

    async def as_owner(user: str, coro_fn):
        """Run a raw (scoped) backend call as *user*."""
        subject = await resolve_subject(
            directory, Subject(id=user, name=_PEOPLE[user][0], is_super_admin=user in _SUPER)
        )
        token = set_current_subject(subject)
        try:
            return await coro_fn()
        finally:
            reset_current_subject(token)

    yield call, as_owner, container, directory, red, blue


async def _membership(directory, sub):
    user = await directory.get_user(sub)
    return user.team_id, user.team_role


# ─── The three example requests ──────────────────────────────────────────────

async def test_make_user_an_admin_of_a_team(env):
    """"make user X an admin of team Y\""""
    call, _, _, directory, red, _ = env
    out = await call("root", "set_team_member", user="dave@example.com", team="Red", role="admin")
    assert "is now admin of team 'Red'" in out and "was: in no team" in out
    assert await _membership(directory, "dave") == (red.id, "admin")


async def test_add_user_to_a_team(env):
    """"add user X to team Y\""""
    call, _, _, directory, red, _ = env
    out = await call("alice", "set_team_member", user="Dave Davis", team="red")
    assert "is now member of team 'Red'" in out
    assert await _membership(directory, "dave") == (red.id, "member")


async def test_make_workflow_team_owned_with_team_visibility(env):
    """"make workflow A team owned by team B with team visibility\""""
    call, as_owner, container, _, _, blue = env
    await as_owner(
        "alice",
        lambda: container.workflow_backend.create(
            WorkflowDefinition(id="daily-report", name="Daily report", steps=[])
        ),
    )
    out = await call(
        "root", "set_resource_visibility",
        kind="workflow", resource="Daily report", visibility="team", team="Blue",
    )
    assert "is now team (team 'Blue')" in out and "was: private" in out
    stored = await unwrap(container.workflow_backend).get("daily-report")
    assert (stored.visibility, stored.team_id, stored.owner_id) == ("team", blue.id, "alice")
    container.refresh_runner.assert_awaited_with("daily-report")
    # Idempotent.
    again = await call(
        "root", "set_resource_visibility",
        kind="workflow", resource="daily-report", visibility="team", team=blue.id,
    )
    assert "already team (team 'Blue')" in again


# ─── Authorization matrix: set_team_member / remove_team_member ──────────────

@pytest.mark.parametrize(
    "caller, allowed",
    [
        ("root", True),    # super admin
        ("alice", True),   # admin of Red
        ("carol", False),  # admin of another team (Blue)
        ("bob", False),    # plain member of Red
        ("dave", False),   # in no team
    ],
)
async def test_add_member_matrix(env, caller, allowed):
    call, _, _, directory, red, _ = env
    # By id, so every caller reaches the authorization check itself (a name is
    # only resolved among the teams the caller can see).
    out = await call(caller, "set_team_member", user="sam1", team=red.id, role="member")
    if allowed:
        assert "is now member of team 'Red'" in out
        assert await _membership(directory, "sam1") == (red.id, "member")
    else:
        assert out.startswith("Refused: Only a super admin or an admin of this team")
        assert await _membership(directory, "sam1") == (None, None)


@pytest.mark.parametrize(
    "caller, allowed",
    # Team admin status is assigned by a super admin only — not by the team's
    # own admin either.
    [("root", True), ("alice", False), ("carol", False), ("bob", False), ("dave", False)],
)
async def test_promote_matrix(env, caller, allowed):
    call, _, _, directory, red, _ = env
    out = await call(caller, "set_team_member", user="bob", team=red.id, role="admin")
    assert ("is now admin of team 'Red'" in out) is allowed, out
    assert await _membership(directory, "bob") == (red.id, "admin" if allowed else "member")


@pytest.mark.parametrize(
    "caller, allowed",
    [("root", True), ("alice", True), ("carol", False), ("bob", False), ("dave", False)],
)
async def test_remove_member_matrix(env, caller, allowed):
    call, _, _, directory, red, _ = env
    out = await call(caller, "remove_team_member", user="bob", team=red.id)
    assert ("removed from team 'Red'" in out) is allowed, out
    assert await _membership(directory, "bob") == ((None, None) if allowed else (red.id, "member"))


async def test_other_teams_are_invisible_by_name_to_a_team_admin(env):
    call, _, _, _, _, _ = env
    out = await call("carol", "set_team_member", user="dave", team="Red")
    assert out.startswith("Team 'Red' not found") and "Blue" in out and "Red (id" not in out


async def test_team_admin_cannot_escalate(env):
    call, _, _, directory, red, blue = env
    # Into another team, as its admin.
    out = await call("alice", "set_team_member", user="alice", team=blue.id, role="admin")
    assert out.startswith("Refused:")
    assert await _membership(directory, "alice") == (red.id, "admin")
    # A member promoting themselves.
    out = await call("bob", "set_team_member", user="bob", team=red.id, role="admin")
    assert out.startswith("Refused:")
    # Pulling a user out of another team (only a super admin moves users).
    out = await call("alice", "set_team_member", user="erin", team=red.id)
    assert out.startswith("Refused: User is in another team")
    assert await _membership(directory, "erin") == (blue.id, "member")
    # ...and that user is not even discoverable by email.
    out = await call("alice", "set_team_member", user="erin@example.com", team=red.id)
    assert "not found among the users you can manage" in out
    # Team lifecycle is super admin only.
    for tool, args in [
        ("create_team", {"name": "Green"}),
        ("rename_team", {"team": "Red", "new_name": "Crimson"}),
        ("delete_team", {"team": "Red"}),
    ]:
        assert (await call("alice", tool, **args)).startswith("Refused: Only a super admin"), tool


async def test_team_admin_cannot_make_demote_or_remove_admins(env):
    call, _, _, directory, red, _ = env
    # Adding a user straight in as admin.
    out = await call("alice", "set_team_member", user="dave", team="Red", role="admin")
    assert out == "Refused: Only a super admin may make someone a team admin"
    assert await _membership(directory, "dave") == (None, None)
    # Stepping down / leaving.
    out = await call("alice", "set_team_member", user="alice", team="Red", role="member")
    assert out == "Refused: Only a super admin may change a team admin's role"
    out = await call("alice", "remove_team_member", user="alice", team="Red")
    assert out == "Refused: Only a super admin may remove a team admin"
    # Another admin of the same team, appointed by the super admin.
    await call("root", "set_team_member", user="bob", team="Red", role="admin")
    out = await call("alice", "set_team_member", user="bob", team="Red", role="member")
    assert out.startswith("Refused: Only a super admin")
    out = await call("alice", "remove_team_member", user="bob", team="Red")
    assert out.startswith("Refused: Only a super admin")
    assert await _membership(directory, "alice") == (red.id, "admin")
    assert await _membership(directory, "bob") == (red.id, "admin")
    # The super admin removes and demotes admins.
    assert "is now member" in await call("root", "set_team_member", user="bob", team="Red")
    assert "removed from team 'Red'" in await call("root", "remove_team_member", user="alice", team="Red")


@pytest.mark.parametrize(
    "caller, allowed",
    [("root", True), ("alice", False), ("carol", False), ("bob", False), ("dave", False)],
)
async def test_team_lifecycle_matrix(env, caller, allowed):
    call, _, _, directory, _, _ = env
    outs = [
        await call(caller, "create_team", name="Green"),
        await call(caller, "rename_team", team="Green" if allowed else "Red", new_name="Lime"),
        await call(caller, "delete_team", team="Lime" if allowed else "Blue"),
    ]
    for out in outs:
        assert out.startswith("Refused:") is not allowed or "not found" in out, (caller, out)
    names = {t.name for t in await directory.list_teams()}
    assert names == {"Red", "Blue"}  # created-renamed-deleted, or untouched


async def test_super_admin_moves_a_user_between_teams(env):
    call, _, _, directory, red, _ = env
    out = await call("root", "set_team_member", user="erin", team="Red")
    assert "is now member of team 'Red' (was: member of team 'Blue')" in out
    assert await _membership(directory, "erin") == (red.id, "member")


async def test_read_only_caller_cannot_write(env):
    call, _, _, directory, _red, _ = env
    out = await call(
        "root", "set_team_member", permissions=frozenset({Permission.ACCESS, Permission.READ}),
        user="dave", team="Red",
    )
    assert out.startswith("Not permitted: this operation requires the 'write' permission")
    assert await _membership(directory, "dave") == (None, None)


async def test_api_key_caller_manages_nothing(env):
    """The shared API key binds an empty Subject: no team, not super admin."""
    _, _, container, directory, red, _ = env
    mcp = build_management_mcp()
    register_management_tools(mcp, lambda: container)
    token = set_current_subject(Subject())
    try:
        out = _text(await mcp.call_tool("set_team_member", {"user": "dave", "team": red.id}))
        listed = _text(await mcp.call_tool("list_users", {}))
    finally:
        reset_current_subject(token)
    assert out.startswith("Refused:")
    assert listed.startswith("Refused:")
    assert await _membership(directory, "dave") == (None, None)


# ─── Resolution ──────────────────────────────────────────────────────────────

async def test_users_resolve_by_email_name_and_id(env):
    call, _, _, _directory, _red, _ = env
    for query in ("DAVE@example.com", "dave davis", "dave"):
        await call("root", "remove_team_member", user="dave", team="Red")
        out = await call("root", "set_team_member", user=query, team="Red")
        assert "Dave Davis <dave@example.com> (id: dave) is now member" in out, query


async def test_ambiguous_user_lists_candidates(env):
    call, _, _, directory, _, _ = env
    out = await call("root", "set_team_member", user="sam", team="Red")
    assert out.startswith("Ambiguous user 'sam'")
    assert "Sam Smith <sam.smith@example.com> (id: sam1)" in out
    assert "Sam Jones <sam.jones@example.com> (id: sam2)" in out
    assert await _membership(directory, "sam1") == (None, None)


async def test_unknown_user_and_team(env):
    call, _, _, _, _, _ = env
    out = await call("root", "set_team_member", user="nobody@example.com", team="Red")
    assert "not found" in out and "first sign-in" in out
    out = await call("root", "set_team_member", user="dave", team="Purple")
    assert out.startswith("Team 'Purple' not found") and "Red (id" in out and "Blue (id" in out


async def test_ambiguous_team(env):
    call, _, _, _directory, _, _ = env
    await call("root", "create_team", name="Ops East")
    await call("root", "create_team", name="Ops West")
    out = await call("root", "set_team_member", user="dave", team="ops")
    assert out.startswith("Ambiguous team 'ops'") and "Ops East" in out and "Ops West" in out


# ─── Idempotence and team lifecycle ──────────────────────────────────────────

async def test_writes_are_idempotent(env):
    call, _, _, directory, _, _ = env
    await call("root", "set_team_member", user="dave", team="Red", role="admin")
    out = await call("root", "set_team_member", user="dave", team="Red", role="admin")
    assert "already admin of team 'Red'; nothing changed" in out
    out = await call("root", "remove_team_member", user="dave", team="Blue")
    assert "is not a member of team 'Blue'; nothing changed" in out
    first = await call("root", "create_team", name="Green")
    again = await call("root", "create_team", name="Green")
    assert "created" in first and "already exists" in again
    assert len([t for t in await directory.list_teams() if t.name == "Green"]) == 1


async def test_rename_and_delete_team(env):
    call, _, _, directory, red, _ = env
    assert "renamed to 'Crimson'" in await call("root", "rename_team", team="Red", new_name="Crimson")
    assert "deleted" in await call("root", "delete_team", team="Crimson")
    assert await directory.get_team(red.id) is None
    assert await _membership(directory, "alice") == (None, None)


async def test_delete_team_refused_while_it_owns_resources(env):
    call, as_owner, container, _, _, _ = env
    await as_owner(
        "alice",
        lambda: container.agent_backend.create(
            AgentDefinition(id="helper", name="Helper", visibility="team")
        ),
    )
    out = await call("root", "delete_team", team="Red")
    assert out.startswith("Refused: Team") and "still has 1 resource" in out


# ─── Listing and identity ────────────────────────────────────────────────────

async def test_list_teams_and_users_follow_rest_scoping(env):
    call, _, _, _, _, _ = env
    all_teams = await call("root", "list_teams")
    assert "**Red**" in all_teams and "**Blue**" in all_teams
    own = await call("bob", "list_teams")
    assert "**Red**" in own and "Blue" not in own
    assert "Bob Brown <bob@example.com> (id: bob): member" in own

    users = await call("alice", "list_users")
    assert "dave@example.com" in users and "bob@example.com" in users
    assert "erin@example.com" not in users  # in another team
    assert (await call("bob", "list_users")).startswith(
        "Refused: Only a super admin or a team admin may list users"
    )
    assert "Alice Adams" in await call("root", "list_users", query="alice@example.com")


async def test_who_am_i(env):
    call, _, _, _, _, _ = env
    out = await call("alice", "who_am_i")
    assert "Super admin: no." in out and "Team: Red" in out and "role: admin" in out
    assert "Super admin: yes" in await call("root", "who_am_i")
    assert "Team: none." in await call("dave", "who_am_i")


# ─── set_resource_visibility authorization ───────────────────────────────────

async def _alice_workflow(as_owner, container, **extra):
    await as_owner(
        "alice",
        lambda: container.workflow_backend.create(
            WorkflowDefinition(id="wf", name="WF", steps=[], **extra)
        ),
    )


async def test_owner_shares_with_own_team_but_not_another(env):
    call, as_owner, container, _, red, _ = env
    await _alice_workflow(as_owner, container)
    out = await call(
        "alice", "set_resource_visibility",
        kind="workflow", resource="wf", visibility="team", team=red.id,
    )
    assert "now team (team 'Red')" in out
    # Another team by id: refused by the scoped backend.
    blue_id = next(t.id for t in await env[3].list_teams() if t.name == "Blue")
    out = await call(
        "alice", "set_resource_visibility",
        kind="workflow", resource="wf", visibility="team", team=blue_id,
    )
    assert out == "Refused: You can only share with your own team"
    # Omitted team: the caller's own.
    out = await call("alice", "set_resource_visibility", kind="workflow", resource="wf", visibility="public")
    out = await call("alice", "set_resource_visibility", kind="workflow", resource="wf", visibility="team")
    assert "now team (team 'Red')" in out


async def test_non_owner_cannot_change_visibility(env):
    call, as_owner, container, _, _, _ = env
    await _alice_workflow(as_owner, container, visibility="public")
    for caller in ("bob", "carol", "dave"):
        out = await call(
            caller, "set_resource_visibility", kind="workflow", resource="wf", visibility="private"
        )
        assert out.startswith("Refused: Only the owner of workflow 'wf'"), (caller, out)
    stored = await unwrap(container.workflow_backend).get("wf")
    assert stored.visibility == "public"


async def test_private_resource_is_not_found_for_others(env):
    call, as_owner, container, _, _, _ = env
    await _alice_workflow(as_owner, container)
    out = await call("bob", "set_resource_visibility", kind="workflow", resource="wf", visibility="public")
    assert "not found" in out
    assert (await unwrap(container.workflow_backend).get("wf")).visibility == "private"


async def test_visibility_for_agents_and_reference_checks(env):
    call, as_owner, container, _, _, _ = env
    await as_owner(
        "alice",
        lambda: container.agent_backend.create(AgentDefinition(id="helper", name="Helper", visibility="public")),
    )
    await as_owner(
        "alice",
        lambda: container.workflow_backend.create(
            WorkflowDefinition(
                id="wf", name="WF", visibility="public",
                steps=[{"id": "a", "type": "langgraph-agent", "agent_id": "helper"}],
            )
        ),
    )
    out = await call("alice", "set_resource_visibility", kind="agent", resource="Helper", visibility="private")
    assert out.startswith("Refused: Cannot make agent 'helper' private")
    out = await call("alice", "set_resource_visibility", kind="workflow", resource="wf", visibility="private")
    assert "now private" in out
    out = await call("alice", "set_resource_visibility", kind="agent", resource="Helper", visibility="private")
    assert "Agent 'helper' is now private" in out


async def test_visibility_argument_validation(env):
    call, as_owner, container, _, _, _ = env
    await _alice_workflow(as_owner, container)
    assert "Unknown kind" in await call(
        "alice", "set_resource_visibility", kind="event", resource="wf", visibility="team"
    )
    assert "Unknown visibility" in await call(
        "alice", "set_resource_visibility", kind="workflow", resource="wf", visibility="everyone"
    )
    assert "only be given with visibility 'team'" in await call(
        "alice", "set_resource_visibility", kind="workflow", resource="wf", visibility="public", team="Red"
    )


# ─── The chat agent's copy of the tools ──────────────────────────────────────

async def test_chat_agent_tools_reach_the_same_directory(env):
    from tests.test_management_mcp_tools import _agent_tools
    _, _, container, directory, red, _ = env
    tools = {t.name: t for t in await _agent_tools(container=container)}
    subject = await resolve_subject(
        directory, Subject(id="alice", name="Alice Adams", email="alice@example.com")
    )
    token = set_current_subject(subject)
    try:
        out = await tools["set_team_member"].ainvoke({"user": "dave@example.com", "team": "Red"})
        refused = await tools["create_team"].ainvoke({"name": "Green"})
    finally:
        reset_current_subject(token)
    assert "is now member of team 'Red'" in out
    assert refused.startswith("Refused: Only a super admin")
    assert await _membership(directory, "dave") == (red.id, "member")


# ─── Who may re-scope a resource (owner team / visibility) ────────────────────

async def _red_agent(as_owner, container, owner="bob"):
    """A Red team agent, owned by plain member *owner*."""
    await as_owner(
        owner,
        lambda: container.agent_backend.create(
            AgentDefinition(id="red-agent", name="Red agent", visibility="team")
        ),
    )


@pytest.mark.parametrize(
    "caller, allowed",
    [
        ("root", True),    # super admin: any resource
        ("alice", True),   # admin of Red: a resource shared with Red
        ("bob", True),     # its owner
        ("carol", False),  # admin of another team: cannot even see it
        ("erin", False),   # member of another team
        ("dave", False),   # in no team
    ],
)
async def test_rescope_team_resource_matrix(env, caller, allowed):
    call, as_owner, container, _, _, _ = env
    await _red_agent(as_owner, container)
    out = await call(caller, "set_resource_visibility", kind="agent", resource="red-agent", visibility="public")
    assert ("is now public" in out) is allowed, (caller, out)
    stored = await unwrap(container.agent_backend).get("red-agent")
    assert stored.visibility == ("public" if allowed else "team")
    assert stored.owner_id == "bob"  # re-scoping never changes the owner


async def test_plain_team_member_cannot_rescope_a_teammates_resource(env):
    call, as_owner, container, _, _, _ = env
    await _red_agent(as_owner, container, owner="alice")
    out = await call("bob", "set_resource_visibility", kind="agent", resource="red-agent", visibility="public")
    assert out.startswith("Refused: Only the owner of agent 'red-agent'")


async def test_team_admin_rescopes_only_within_their_team(env):
    call, as_owner, container, _directory, _red, blue = env
    await _red_agent(as_owner, container)
    # Not to another team...
    out = await call(
        "alice", "set_resource_visibility", kind="agent", resource="red-agent",
        visibility="team", team=blue.id,
    )
    assert out == "Refused: You can only share with your own team"
    # ...but may narrow it to its owner, after which it is out of her reach.
    out = await call("alice", "set_resource_visibility", kind="agent", resource="red-agent", visibility="private")
    assert "is now private" in out
    out = await call("alice", "set_resource_visibility", kind="agent", resource="red-agent", visibility="team")
    assert out.startswith("No agent found matching 'red-agent'")
    # A team admin's extra right is re-scoping only: no editing, no deleting.
    await call("bob", "set_resource_visibility", kind="agent", resource="red-agent", visibility="team")
    out = await call("alice", "update_agent", agent_id="red-agent", description="hijacked")
    assert out.startswith("Refused: Only the owner"), out
    out = await call("alice", "delete_agent", agent_id="red-agent")
    assert out.startswith("Refused: Only the owner"), out
    assert await unwrap(container.agent_backend).get("red-agent") is not None


async def test_team_admin_cannot_rescope_a_public_resource_of_a_member(env):
    """Only resources *shared with* their team; a member's public one is the
    member's (or a super admin's) to narrow."""
    call, as_owner, container, _, _, _ = env
    await as_owner(
        "bob",
        lambda: container.agent_backend.create(
            AgentDefinition(id="pub", name="Pub", visibility="public")
        ),
    )
    out = await call("alice", "set_resource_visibility", kind="agent", resource="pub", visibility="team")
    assert out.startswith("Refused: Only the owner of agent 'pub'")


# ─── The MCP login path ──────────────────────────────────────────────────────
#
# Through _ManagementAuthWrapper with an OAuth bearer, exactly as an MCP client
# (or the external-api-service gateway forwarding the user's token) reaches
# /mcp/management: roles come from the token, super admin is evaluated
# strictly from AUTH_SUPER_ADMIN_ROLES, and the team and team role come from
# carrier's directory — never from the token's own claims.

_TOKENS = {
    "tok-root": {"sub": "root", "email": "root@example.com", "roles": ["staff", "carrier-super"]},
    "tok-alice": {"sub": "alice", "email": "alice@example.com", "roles": ["staff"]},
    "tok-bob": {
        "sub": "bob", "email": "bob@example.com", "roles": ["staff"],
        # Claims a token might carry; none of them may grant anything.
        "team_role": "admin", "is_super_admin": True,
    },
    # Holds ADMIN (unsandboxed steps), which is not super admin.
    "tok-ops": {"sub": "dave", "email": "dave@example.com", "roles": ["staff", "ops"]},
    "tok-outsider": {"sub": "eve", "email": "eve@example.com", "roles": ["customer"]},
}


def _login_app(container, mcp):
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    from app.api.app import _ManagementAuthWrapper
    from app.infrastructure.auth.auth_service import AuthService
    from app.infrastructure.auth.authorization import AuthorizationPolicy

    async def tool(request: Request):
        body = await request.json()
        return PlainTextResponse(_text(await mcp.call_tool(body["tool"], body["args"])))

    auth = MagicMock(spec=AuthService)

    async def validate(token):
        return dict(_TOKENS[token])

    auth.validate_token = validate
    staff = frozenset({"staff"})
    policy = AuthorizationPolicy(
        enforce=True,
        access_roles=staff, read_roles=staff, write_roles=staff, delete_roles=staff,
        admin_roles=frozenset({"ops"}), super_admin_roles=frozenset({"carrier-super"}),
    )
    wrapper = _ManagementAuthWrapper(
        Starlette(routes=[Route("/management", tool, methods=["POST"])]),
        api_key="shared-key", oauth_enabled=True, auth_service=auth, policy=policy,
    )
    app_holder = MagicMock()
    app_holder.state.container = container

    async def with_app(scope, receive, send):
        # What the mounted wrapper sees in production: scope["app"] is carrier.
        scope["app"] = app_holder
        await wrapper(scope, receive, send)

    return with_app


@pytest.mark.parametrize(
    "token, add_member, make_admin, create_team",
    [
        ("tok-root", True, True, True),       # super admin
        ("tok-alice", True, False, False),    # team admin of Red
        ("tok-bob", False, False, False),     # plain member (spoofed claims ignored)
        ("tok-ops", False, False, False),     # ADMIN permission, no team, not super
        ("shared-key", False, False, False),  # shared API key: nobody
    ],
)
async def test_mcp_login_maps_to_the_role_rules(env, token, add_member, make_admin, create_team):
    from httpx import ASGITransport, AsyncClient

    _, _, container, _directory, red, _ = env
    mcp = build_management_mcp()
    register_management_tools(mcp, lambda: container)
    app = _login_app(container, mcp)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        async def call(tool, **args):
            resp = await client.post(
                "/management", json={"tool": tool, "args": args},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200, resp.text
            return resp.text

        out = await call("set_team_member", user="sam1", team=red.id)
        assert ("is now member" in out) is add_member, out
        out = await call("set_team_member", user="sam2", team=red.id, role="admin")
        assert ("is now admin" in out) is make_admin, out
        out = await call("create_team", name="Green")
        assert ("created" in out) is create_team, out
        if not (add_member or make_admin or create_team):
            assert out.startswith("Refused:")

        if token == "tok-bob":
            assert "Super admin: no." in await call("who_am_i")
            assert "role: member" in await call("who_am_i")


async def test_mcp_login_rejects_a_token_without_access(env):
    from httpx import ASGITransport, AsyncClient

    _, _, container, _, _red, _ = env
    mcp = build_management_mcp()
    register_management_tools(mcp, lambda: container)
    app = _login_app(container, mcp)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        resp = await client.post(
            "/management", json={"tool": "list_teams", "args": {}},
            headers={"Authorization": "Bearer tok-outsider"},
        )
    assert resp.status_code == 403
