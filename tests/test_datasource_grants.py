"""Grants: opening a data source to a user or a workflow outside its scope.

Same cast as tests/test_teams.py — red: alice (admin), bob; blue: carol;
dave in no team; admin a super admin. Alice's team source ``crm`` has a read
operation (``list``) and a write (``create``).
"""
from __future__ import annotations

from types import SimpleNamespace

from app.application.visibility import (
    operation_denied,
    sources_for_workflow,
    workflow_grant_access,
)
from app.infrastructure.auth.authorization import Subject, reset_current_subject, set_current_subject
from tests.test_teams import env  # noqa: F401 — the fixture

_SOURCE = {
    "id": "crm",
    "name": "CRM",
    "base_url": "https://crm.example.com",
    "operations": [
        {"name": "list", "method": "GET", "path": "/items"},
        {"name": "create", "method": "POST", "path": "/items"},
    ],
}


def _step_workflow(workflow_id: str, operation: str = "list", **extra) -> dict:
    return {
        "id": workflow_id,
        "name": workflow_id,
        "steps": [{"id": "fetch", "type": "data_source", "source": "crm", "operation": operation}],
        **extra,
    }


async def _team_source(as_, visibility: str = "team") -> None:
    resp = await as_("alice").post("/api/v1/datasources", json={**_SOURCE, "visibility": visibility})
    assert resp.status_code == 201, resp.text


async def _grant(as_, grantee_type: str, grantee_id: str, access: str = "read", by: str = "alice"):
    return await as_(by).post(
        "/api/v1/datasources/crm/grants",
        json={"grantee_type": grantee_type, "grantee_id": grantee_id, "access": access},
    )


# ─── User grants ─────────────────────────────────────────────────────────────

async def test_user_grant_makes_the_source_visible_and_usable_in_private_workflows(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    await as_("carol").get("/api/v1/me")  # carol is known to the directory
    assert "crm" not in {s["id"] for s in (await as_("carol").get("/api/v1/datasources")).json()}

    resp = await _grant(as_, "user", "carol")
    assert resp.status_code == 201, resp.text
    assert resp.json()["access"] == "read" and resp.json()["grantee_name"] == "Carol"

    listed = {s["id"]: s for s in (await as_("carol").get("/api/v1/datasources")).json()}
    assert listed["crm"]["granted_access"] == "read"
    assert listed["crm"]["can_edit"] is False
    assert "grants" not in listed["crm"]  # who else holds one is the owner's business
    assert (await as_("carol").post("/api/v1/workflows", json=_step_workflow("mine"))).status_code == 201


async def test_read_grant_refuses_write_operations_read_write_allows_them(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    await as_("carol").get("/api/v1/me")
    await _grant(as_, "user", "carol", "read")
    resp = await as_("carol").post("/api/v1/workflows", json=_step_workflow("w", "create"))
    assert resp.status_code == 422
    assert "only read access" in resp.json()["detail"]

    # Granting again changes the level rather than adding a second grant.
    await _grant(as_, "user", "carol", "read_write")
    grants = (await as_("alice").get("/api/v1/datasources/crm/grants")).json()
    assert [g["access"] for g in grants] == ["read_write"]
    assert (await as_("carol").post("/api/v1/workflows", json=_step_workflow("w", "create"))).status_code == 201


async def test_user_grant_does_not_extend_to_the_grantees_team_workflows(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    await as_("carol").get("/api/v1/me")
    await _grant(as_, "user", "carol")
    resp = await as_("carol").post(
        "/api/v1/workflows", json=_step_workflow("shared", visibility="team")
    )
    assert resp.status_code == 422


async def test_grantee_cannot_edit_delete_or_reshare(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    await as_("carol").get("/api/v1/me")
    await _grant(as_, "user", "carol", "read_write")
    assert (await as_("carol").put("/api/v1/datasources/crm", json={"name": "Mine"})).status_code == 403
    assert (await as_("carol").delete("/api/v1/datasources/crm")).status_code == 403
    assert (await _grant(as_, "user", "dave", by="carol")).status_code == 403
    assert (await as_("carol").get("/api/v1/datasources/crm/grants")).status_code == 403


async def test_owner_edits_keep_grants(env):
    as_, container, _, _, _ = env
    await _team_source(as_)
    await as_("carol").get("/api/v1/me")
    await _grant(as_, "user", "carol")
    resp = await as_("alice").put("/api/v1/datasources/crm", json={"name": "Renamed", "grants": []})
    assert resp.status_code == 200
    assert [g["grantee_id"] for g in resp.json()["grants"]] == ["carol"]


async def test_who_may_be_granted(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    assert (await _grant(as_, "user", "alice")).status_code == 409   # the owner
    assert (await _grant(as_, "user", "bob")).status_code == 409     # in the team
    assert (await _grant(as_, "user", "nobody")).status_code == 422  # unknown
    assert (await _grant(as_, "workflow", "missing")).status_code == 422
    # Bob (a red member, not an admin) may use it but not share it.
    await as_("dave").get("/api/v1/me")
    assert (await _grant(as_, "user", "dave", by="bob")).status_code == 403
    # Super admin may.
    assert (await _grant(as_, "user", "dave", by="admin")).status_code == 201


async def test_public_sources_need_no_grants(env):
    as_, _, _, _, _ = env
    await _team_source(as_, "public")
    await as_("carol").get("/api/v1/me")
    assert (await _grant(as_, "user", "carol")).status_code == 409


async def test_revoking_hides_the_source_again(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    await as_("carol").get("/api/v1/me")
    grant = (await _grant(as_, "user", "carol")).json()
    assert (await as_("carol").get("/api/v1/datasources/crm")).status_code == 200
    assert (await as_("alice").delete(f"/api/v1/datasources/crm/grants/{grant['id']}")).status_code == 204
    assert (await as_("carol").get("/api/v1/datasources/crm")).status_code == 404
    assert (await as_("alice").delete(f"/api/v1/datasources/crm/grants/{grant['id']}")).status_code == 404
    assert (await as_("alice").get("/api/v1/datasources/crm/grants")).json() == []


# ─── Workflow grants ─────────────────────────────────────────────────────────

async def test_workflow_grant_is_usable_in_that_workflow_only(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    # Carol's private workflow, created before it uses the source.
    created = await as_("carol").post("/api/v1/workflows", json={"id": "report", "name": "Report", "steps": []})
    assert created.status_code == 201, created.text
    assert (await as_("carol").put("/api/v1/workflows/report", json=_step_workflow("report"))).status_code == 422

    resp = await _grant(as_, "workflow", "report")
    assert resp.status_code == 201, resp.text
    # Alice cannot see Carol's private workflow, so its name is not echoed.
    assert resp.json()["grantee_name"] is None

    assert (await as_("carol").put("/api/v1/workflows/report", json=_step_workflow("report"))).status_code == 200
    # Nowhere else: not listed, not by id, not in a new workflow.
    assert "crm" not in {s["id"] for s in (await as_("carol").get("/api/v1/datasources")).json()}
    assert (await as_("carol").get("/api/v1/datasources/crm")).status_code == 404
    assert (await as_("carol").post("/api/v1/workflows", json=_step_workflow("other"))).status_code == 422

    # In the editor of that workflow it is offered, marked as granted to it.
    listed = (await as_("carol").get("/api/v1/datasources", params={"workflow_id": "report"})).json()
    granted = {s["id"]: s for s in listed}["crm"]
    assert granted["granted_to_workflow"] == "read"
    assert "grants" not in granted and granted["auth"]["type"] == "none"
    one = await as_("carol").get("/api/v1/datasources/crm", params={"workflow_id": "report"})
    assert one.status_code == 200
    # ...but not to someone who cannot see the workflow.
    dave = (await as_("dave").get("/api/v1/datasources", params={"workflow_id": "report"})).json()
    assert "crm" not in {s["id"] for s in dave}


async def test_workflow_grant_respects_read_only(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    await as_("carol").post("/api/v1/workflows", json={"id": "report", "name": "Report", "steps": []})
    await _grant(as_, "workflow", "report", "read")
    resp = await as_("carol").put("/api/v1/workflows/report", json=_step_workflow("report", "create"))
    assert resp.status_code == 422


async def test_workflow_already_in_scope_needs_no_grant(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    await as_("bob").post("/api/v1/workflows", json={"id": "red-flow", "name": "R", "steps": [], "visibility": "team"})
    assert (await _grant(as_, "workflow", "red-flow")).status_code == 409


async def test_deleting_the_workflow_drops_its_grant(env):
    as_, container, _, _, _ = env
    await _team_source(as_)
    await as_("carol").post("/api/v1/workflows", json={"id": "report", "name": "Report", "steps": []})
    await _grant(as_, "workflow", "report")
    assert (await as_("carol").delete("/api/v1/workflows/report")).status_code == 204
    assert (await as_("alice").get("/api/v1/datasources/crm/grants")).json() == []


async def test_candidates_leave_out_owner_and_team(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    for user in ("carol", "dave"):
        await as_(user).get("/api/v1/me")
    resp = await as_("alice").get("/api/v1/datasources/crm/grants/candidates")
    assert resp.status_code == 200
    ids = {u["id"] for u in resp.json()["users"]}
    assert {"carol", "dave"} <= ids
    assert not ids & {"alice", "bob"}


# ─── Run time ────────────────────────────────────────────────────────────────

async def test_run_time_resolution_and_read_only_enforcement(env):
    as_, container, _, red, _ = env
    await _team_source(as_)
    await as_("carol").post("/api/v1/workflows", json={"id": "report", "name": "Report", "steps": []})
    await _grant(as_, "workflow", "report", "read")
    runner = SimpleNamespace(id="report", visibility="private", owner_id="carol", team_id=None)

    token = set_current_subject(Subject(id="carol", name="Carol"))
    try:
        source = await sources_for_workflow(container.data_source_backend, "report").get("crm")
        assert source is not None
        assert await sources_for_workflow(container.data_source_backend, "other").get("crm") is None
        access = workflow_grant_access(source, runner)
        assert access == "read"
        assert operation_denied(source, "list", access) is None
        assert "writes" in operation_denied(source, "create", access)
    finally:
        reset_current_subject(token)

    # The team's own workflows are never restricted by a grant.
    team_runner = SimpleNamespace(id="x", visibility="team", owner_id="bob", team_id=red)
    assert workflow_grant_access(source, team_runner) is None


async def test_read_grant_limits_agent_addons_too(env):
    as_, _, _, _, _ = env
    await _team_source(as_)
    await as_("carol").get("/api/v1/me")
    await _grant(as_, "user", "carol", "read")

    def agent(ops):
        return {"id": "helper", "name": "Helper", "addons": [
            {"type": "datasource", "source_id": "crm", "allowed_operations": ops},
        ]}
    assert (await as_("carol").post("/api/v1/agents", json=agent(["create"]))).status_code == 422
    assert (await as_("carol").post("/api/v1/agents", json=agent(["list"]))).status_code == 201
