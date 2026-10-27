"""Grants on agents and workflows (data sources: tests/test_datasource_grants.py).

Same cast as tests/test_teams.py — red: alice (admin), bob; blue: carol;
dave in no team; admin a super admin.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.application.visibility import granted_for_workflow, unwrap
from app.infrastructure.auth.authorization import Subject, reset_current_subject, set_current_subject
from tests.test_teams import env  # noqa: F401 — the fixture


def _agent_step_workflow(workflow_id: str, agent_id: str = "helper", **extra) -> dict:
    return {
        "id": workflow_id, "name": workflow_id,
        "steps": [{"id": "a", "type": "langgraph-agent", "agent_id": agent_id}],
        **extra,
    }


def _call_workflow(workflow_id: str, child: str, **extra) -> dict:
    return {
        "id": workflow_id, "name": workflow_id,
        "steps": [{"id": "c", "type": "workflow", "workflow_id": child}],
        **extra,
    }


async def _grant(as_, kind: str, resource: str, grantee_type: str, grantee_id: str,
                 access: str = "read", by: str = "alice"):
    return await as_(by).post(
        f"/api/v1/{kind}/{resource}/grants",
        json={"grantee_type": grantee_type, "grantee_id": grantee_id, "access": access},
    )


def _registry_over(container):
    registry = MagicMock()

    def list_definitions():
        return [
            {
                "id": w.id, "name": w.name, "visibility": w.visibility,
                "owner_id": w.owner_id, "team_id": w.team_id,
                "grants": [g.model_dump(mode="json") for g in w.grants],
            }
            for w in unwrap(container.workflow_backend).docs.values()
        ]

    registry.list_definitions.side_effect = list_definitions
    return registry


# ─── Agents ──────────────────────────────────────────────────────────────────

async def test_agent_user_grant_read_lets_them_use_it_but_not_edit(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/agents", json={"id": "helper", "name": "Helper", "visibility": "team"})
    await as_("carol").get("/api/v1/me")
    assert (await as_("carol").get("/api/v1/agents/helper")).status_code == 404

    assert (await _grant(as_, "agents", "helper", "user", "carol")).status_code == 201
    got = (await as_("carol").get("/api/v1/agents/helper")).json()
    assert got["granted_access"] == "read" and got["can_edit"] is False and "grants" not in got
    assert (await as_("carol").post("/api/v1/workflows", json=_agent_step_workflow("mine"))).status_code == 201
    assert (await as_("carol").put("/api/v1/agents/helper", json={"name": "X"})).status_code == 403


async def test_agent_read_write_grant_lets_the_user_edit_but_not_delete_or_reshare(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/agents", json={"id": "helper", "name": "Helper", "visibility": "private"})
    await as_("carol").get("/api/v1/me")
    await as_("dave").get("/api/v1/me")
    await _grant(as_, "agents", "helper", "user", "carol", "read_write")

    got = (await as_("carol").get("/api/v1/agents/helper")).json()
    assert got["can_edit"] is True
    resp = await as_("carol").put("/api/v1/agents/helper", json={"name": "Better", "visibility": "private"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["owner_id"] == "alice"  # still Alice's
    assert (await as_("carol").put("/api/v1/agents/helper", json={"name": "Mine", "visibility": "public"})).status_code == 403
    assert (await as_("carol").delete("/api/v1/agents/helper")).status_code == 403
    assert (await _grant(as_, "agents", "helper", "user", "dave", by="carol")).status_code == 403


async def test_agent_workflow_grant_works_in_that_workflow_only(env):
    as_, container, _, _, _ = env
    await as_("alice").post("/api/v1/agents", json={"id": "helper", "name": "Helper", "visibility": "team"})
    await as_("carol").post("/api/v1/workflows", json={"id": "report", "name": "Report", "steps": []})
    assert (await as_("carol").put("/api/v1/workflows/report", json=_agent_step_workflow("report"))).status_code == 422

    assert (await _grant(as_, "agents", "helper", "workflow", "report")).status_code == 201
    assert (await as_("carol").put("/api/v1/workflows/report", json=_agent_step_workflow("report"))).status_code == 200
    assert (await as_("carol").post("/api/v1/workflows", json=_agent_step_workflow("other"))).status_code == 422
    assert (await as_("carol").get("/api/v1/agents/helper")).status_code == 404

    listed = (await as_("carol").get("/api/v1/agents", params={"workflow_id": "report"})).json()
    assert {a["id"]: a for a in listed}["helper"]["granted_to_workflow"] == "read"
    summary = (await as_("carol").get("/api/v1/agents", params={"workflow_id": "report", "view": "summary"})).json()
    assert {a["id"]: a for a in summary}["helper"]["granted_to_workflow"] == "read"

    # At run time the workflow's agent lookups see it; another's do not.
    token = set_current_subject(Subject(id="carol", name="Carol"))
    try:
        assert await granted_for_workflow(container.agent_backend, "report").get("helper") is not None
        assert await granted_for_workflow(container.agent_backend, "other").get("helper") is None
    finally:
        reset_current_subject(token)


async def test_a_workflow_grant_on_an_agent_is_use_only(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/agents", json={"id": "helper", "name": "Helper", "visibility": "team"})
    await as_("carol").post("/api/v1/workflows", json={"id": "report", "name": "Report", "steps": []})
    assert (await _grant(as_, "agents", "helper", "workflow", "report", "read_write")).status_code == 422


# ─── Workflows ───────────────────────────────────────────────────────────────

async def test_team_a_workflow_calls_team_b_workflow_through_a_grant(env):
    as_, container, _, _, _ = env
    container.yaml_graph_registry = _registry_over(container)
    # Team red's workflow; team blue's caller.
    assert (await as_("alice").post("/api/v1/workflows", json={
        "id": "enrich", "name": "Enrich", "steps": [], "visibility": "team"})).status_code == 201
    assert (await as_("carol").post("/api/v1/workflows", json={
        "id": "blue-report", "name": "Blue report", "steps": [], "visibility": "team"})).status_code == 201
    assert (await as_("carol").put("/api/v1/workflows/blue-report",
                                   json=_call_workflow("blue-report", "enrich"))).status_code == 422

    resp = await _grant(as_, "workflows", "enrich", "workflow", "blue-report")
    assert resp.status_code == 201, resp.text
    assert (await as_("carol").put("/api/v1/workflows/blue-report",
                                   json=_call_workflow("blue-report", "enrich"))).status_code == 200
    # Not callable from Carol's other workflows, not listed for her elsewhere.
    assert (await as_("carol").post("/api/v1/workflows",
                                    json=_call_workflow("other", "enrich"))).status_code == 422
    plain = {w["id"] for w in (await as_("carol").get("/api/v1/workflows")).json()}
    assert "enrich" not in plain
    listed = (await as_("carol").get("/api/v1/workflows", params={"workflow_id": "blue-report"})).json()
    assert {w["id"]: w for w in listed}["enrich"]["granted_to_workflow"] == "read"
    # The grant list is Alice's to see.
    assert "grants" not in {w["id"]: w for w in listed}["enrich"]
    grants = (await as_("alice").get("/api/v1/workflows/enrich/grants")).json()
    assert [(g["grantee_type"], g["grantee_id"]) for g in grants] == [("workflow", "blue-report")]

    # Revoked: the next save that relies on it is refused.
    await as_("alice").delete(f"/api/v1/workflows/enrich/grants/{grants[0]['id']}")
    assert (await as_("carol").put("/api/v1/workflows/blue-report",
                                   json=_call_workflow("blue-report", "enrich"))).status_code == 422


async def test_workflow_user_grants_read_and_read_write(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/workflows", json={"id": "enrich", "name": "Enrich", "steps": [], "visibility": "team"})
    await as_("carol").get("/api/v1/me")
    await _grant(as_, "workflows", "enrich", "user", "carol")
    got = (await as_("carol").get("/api/v1/workflows/enrich")).json()
    assert got["granted_access"] == "read" and got["can_edit"] is False
    # Usable from her own private workflows only.
    assert (await as_("carol").post("/api/v1/workflows", json=_call_workflow("mine", "enrich"))).status_code == 201
    assert (await as_("carol").post("/api/v1/workflows",
                                    json=_call_workflow("ours", "enrich", visibility="team"))).status_code == 422
    assert (await as_("carol").put("/api/v1/workflows/enrich", json={"name": "X", "steps": []})).status_code == 403

    await _grant(as_, "workflows", "enrich", "user", "carol", "read_write")
    resp = await as_("carol").put("/api/v1/workflows/enrich", json={"name": "Enrich v2", "steps": []})
    assert resp.status_code == 200, resp.text
    assert resp.json()["visibility"] == "team" and resp.json()["owner_id"] == "alice"


async def test_a_workflow_cannot_be_granted_to_itself_or_in_scope_workflows(env):
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/workflows", json={"id": "enrich", "name": "Enrich", "steps": [], "visibility": "team"})
    await as_("bob").post("/api/v1/workflows", json={"id": "red-2", "name": "R", "steps": [], "visibility": "team"})
    assert (await _grant(as_, "workflows", "enrich", "workflow", "enrich")).status_code == 409
    assert (await _grant(as_, "workflows", "enrich", "workflow", "red-2")).status_code == 409


async def test_deleting_a_workflow_drops_grants_to_it_on_every_kind(env):
    as_, container, _, _, _ = env
    await as_("alice").post("/api/v1/agents", json={"id": "helper", "name": "Helper", "visibility": "team"})
    await as_("alice").post("/api/v1/workflows", json={"id": "enrich", "name": "Enrich", "steps": [], "visibility": "team"})
    await as_("carol").post("/api/v1/workflows", json={"id": "report", "name": "Report", "steps": []})
    await _grant(as_, "agents", "helper", "workflow", "report")
    await _grant(as_, "workflows", "enrich", "workflow", "report")
    assert (await as_("carol").delete("/api/v1/workflows/report")).status_code == 204
    assert (await as_("alice").get("/api/v1/agents/helper/grants")).json() == []
    assert (await as_("alice").get("/api/v1/workflows/enrich/grants")).json() == []


async def test_grantee_editor_keeps_references_they_cannot_see(env):
    """A read_write grantee re-saving an agent keeps what it already used."""
    as_, _, _, _, _ = env
    await as_("alice").post("/api/v1/datasources", json={
        "id": "crm", "name": "CRM", "base_url": "https://crm.example.com",
        "operations": [{"name": "list", "method": "GET", "path": "/items"}],
        "visibility": "private",
    })
    agent = {"id": "helper", "name": "Helper", "visibility": "private", "addons": [
        {"type": "datasource", "source_id": "crm", "allowed_operations": ["list"]},
    ]}
    assert (await as_("alice").post("/api/v1/agents", json=agent)).status_code == 201
    await as_("carol").get("/api/v1/me")
    await _grant(as_, "agents", "helper", "user", "carol", "read_write")
    resp = await as_("carol").put("/api/v1/agents/helper", json={**agent, "name": "Helper 2"})
    assert resp.status_code == 200, resp.text


async def test_runner_carries_grants_into_the_registry_listing():
    from app.domain.models.ownership import ResourceGrant
    from app.domain.models.workflow_definition import WorkflowDefinition
    from app.infrastructure.config.graph_loader import _copy_ownership

    runner = SimpleNamespace()
    defn = WorkflowDefinition(id="w", grants=[ResourceGrant(
        id="g", grantee_type="workflow", grantee_id="x", access="read")])
    _copy_ownership(runner, defn)
    assert runner.grants[0].grantee_id == "x"
