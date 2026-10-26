"""Grant / revoke / list through the management MCP tools.

Cast from tests/test_mcp_teams_tools.py — red: alice (admin), bob; blue:
carol (admin), erin; dave in no team; root a super admin.
"""
from __future__ import annotations

from app.application.visibility import unwrap
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.data_source_definition import DataSourceDefinition
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.auth.authorization import Permission
from tests.test_mcp_teams_tools import env  # noqa: F401 — the fixture


async def _seed(as_owner, container, red):
    await as_owner("alice", lambda: container.data_source_backend.create(DataSourceDefinition(
        id="crm", name="CRM", base_url="https://crm", visibility="team", team_id=red.id,
        operations=[{"name": "list", "method": "GET", "path": "/l"}],
    )))
    await as_owner("alice", lambda: container.agent_backend.create(
        AgentDefinition(id="helper", name="Helper", visibility="team", team_id=red.id)))
    await as_owner("alice", lambda: container.workflow_backend.create(
        WorkflowDefinition(id="enrich", name="Enrich", visibility="team", team_id=red.id)))
    await as_owner("carol", lambda: container.workflow_backend.create(
        WorkflowDefinition(id="blue-report", name="Blue report")))


async def test_grant_a_datasource_to_a_user_by_email(env):
    call, as_owner, container, _, red, _ = env
    await _seed(as_owner, container, red)
    out = await call("alice", "grant_resource_access", kind="data source", resource="CRM",
                     grantee="dave@example.com")
    assert "Granted user Dave Davis read access to datasource 'crm'" in out
    stored = await unwrap(container.data_source_backend).get("crm")
    assert [(g.grantee_id, g.access) for g in stored.grants] == [("dave", "read")]

    again = await call("alice", "grant_resource_access", kind="datasource", resource="crm",
                       grantee="dave", access="read-write")
    assert "from read to read & write" in again
    same = await call("alice", "grant_resource_access", kind="datasource", resource="crm",
                      grantee="dave", access="rw")
    assert "nothing changed" in same

    listed = await call("alice", "list_resource_grants", kind="datasource", resource="crm")
    assert "user Dave Davis (id: dave): read & write" in listed


async def test_let_team_b_workflow_call_team_a_workflow(env):
    call, as_owner, container, _, red, _ = env
    await _seed(as_owner, container, red)
    out = await call("alice", "grant_resource_access", kind="workflow", resource="Enrich",
                     grantee="blue-report", grantee_type="workflow")
    assert "Granted workflow" in out and "workflow 'enrich'" in out
    stored = await unwrap(container.workflow_backend).get("enrich")
    assert [(g.grantee_type, g.grantee_id) for g in stored.grants] == [("workflow", "blue-report")]
    container.refresh_runner.assert_awaited_with("enrich")

    out = await call("alice", "revoke_resource_access", kind="workflow", resource="enrich",
                     grantee="blue-report", grantee_type="workflow")
    assert out.startswith("Revoked workflow")
    assert (await unwrap(container.workflow_backend).get("enrich")).grants == []


async def test_grant_an_agent_and_revoke_by_grant_id(env):
    call, as_owner, container, _, red, _ = env
    await _seed(as_owner, container, red)
    await call("alice", "grant_resource_access", kind="agent", resource="helper",
               grantee="Carol Clark", access="read_write")
    grant_id = (await unwrap(container.agent_backend).get("helper")).grants[0].id
    out = await call("alice", "revoke_resource_access", kind="agent", resource="helper", grantee=grant_id)
    assert "Revoked user" in out
    nothing = await call("alice", "revoke_resource_access", kind="agent", resource="helper", grantee="carol")
    assert "holds no grant" in nothing


async def test_refusals_come_back_as_text(env):
    call, as_owner, container, _, red, _ = env
    await _seed(as_owner, container, red)
    # Bob is in red but not its admin: he may use it, not share it.
    out = await call("bob", "grant_resource_access", kind="agent", resource="helper", grantee="dave")
    assert out.startswith("Refused:")
    # A team member needs no grant.
    out = await call("alice", "grant_resource_access", kind="agent", resource="helper", grantee="bob")
    assert out.startswith("Refused:") and "can already use it" in out
    # Carol cannot see the red agent at all.
    out = await call("carol", "list_resource_grants", kind="agent", resource="helper")
    assert "no agent found" in out.lower()
    out = await call("alice", "grant_resource_access", kind="script", resource="x", grantee="dave")
    assert "grants exist for datasource, agent and workflow" in out


async def test_permission_tiers(env):
    call, as_owner, container, _, red, _ = env
    await _seed(as_owner, container, red)
    read_only = frozenset({Permission.READ})
    out = await call("alice", "grant_resource_access", permissions=read_only,
                     kind="agent", resource="helper", grantee="dave")
    assert "WRITE" in out or "permission" in out.lower()
    listed = await call("alice", "list_resource_grants", permissions=read_only,
                        kind="agent", resource="helper")
    assert "not shared beyond its visibility" in listed
    write_only = frozenset({Permission.READ, Permission.WRITE})
    out = await call("alice", "revoke_resource_access", permissions=write_only,
                     kind="agent", resource="helper", grantee="dave")
    assert "DELETE" in out or "permission" in out.lower()
