"""
Integration test: human_approval as a real goalkeeper using the conditional
``routes`` + END shape (approve -> next step, reject -> END/rejected), plus the
append-only ``approval_history`` audit trail and approver-identity capture.

These complement ``test_approval_flow.py`` (which exercises the older
``when:``-guard shape). Here every gate carries an explicit two-entry routes
list so a rejection terminates the run instead of merely skipping the guarded
step.
"""
from __future__ import annotations

import pytest

from tests.integration.conftest import build_int_client, make_mock_llm

# ---------------------------------------------------------------------------
# Graph definitions under test
# ---------------------------------------------------------------------------

_GRAPH = {
    "id": "approval-routes-test",
    "steps": [
        {
            "id": "plan",
            "type": "llm",
            "output_key": "plan",
            "system_prompt": "Produce a short plan.",
            "user_template": "{request}",
        },
        {
            "id": "gate",
            "type": "human_approval",
            "output_key": "plan_approved",
            "interrupt_payload": {"plan": "{plan}"},
            "routes": [
                {"when": "plan_approved", "next": "implement"},
                {"next": "END"},
            ],
        },
        {
            "id": "implement",
            "type": "llm",
            "output_key": "implementation",
            "system_prompt": "Implement the plan.",
            "user_template": "Plan: {plan}",
        },
    ],
}
_GRAPH_ID = "approval-routes-test"

_MULTI_GRAPH = {
    "id": "approval-routes-multi",
    "steps": [
        {
            "id": "plan",
            "type": "llm",
            "output_key": "plan",
            "system_prompt": "Plan.",
            "user_template": "{request}",
        },
        {
            "id": "gate1",
            "type": "human_approval",
            "output_key": "gate1_approved",
            "interrupt_payload": {"plan": "{plan}"},
            "routes": [
                {"when": "gate1_approved", "next": "gate2"},
                {"next": "END"},
            ],
        },
        {
            "id": "gate2",
            "type": "human_approval",
            "output_key": "gate2_approved",
            "interrupt_payload": {"plan": "{plan}"},
            "routes": [
                {"when": "gate2_approved", "next": "done"},
                {"next": "END"},
            ],
        },
        {
            "id": "done",
            "type": "llm",
            "output_key": "final_output",
            "system_prompt": "Finish.",
            "user_template": "Plan: {plan}",
        },
    ],
}
_MULTI_GRAPH_ID = "approval-routes-multi"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ui_approve_routes_to_next_step() -> None:
    llm = make_mock_llm(text_responses=["the plan", "the implementation"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "ship X"},
        )
        assert start.status_code == 200, start.text
        run_id = start.json()["id"]

        paused = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert paused.json()["status"] == "waiting_approval"

        approve = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
        assert approve.status_code == 200, approve.text

        body = (await client.get(f"/api/v1/workflows/runs/{run_id}")).json()
        assert body["status"] == "completed"
        outputs = body["intermediate_outputs"]
        # Downstream step ran.
        assert outputs["implementation"] == "the implementation"
        assert outputs["plan_approved"] is True

        history = outputs["approval_history"]
        assert len(history) == 1
        rec = history[0]
        assert rec["step_id"] == "gate"
        assert rec["approved"] is True
        assert rec["approver_source"] == "ui"
        # Test client injects no real JWT claims — name/id are null-tolerant.
        assert rec["approver_name"] in (None, "")
        assert rec["approver_id"] in (None, "")
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_ui_reject_routes_to_end_rejected() -> None:
    llm = make_mock_llm(text_responses=["the plan"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "ship Y"},
        )
        assert start.status_code == 200
        run_id = start.json()["id"]

        reject = await client.post(
            f"/api/v1/workflows/runs/{run_id}/reject",
            json={"reason": "plan is wrong"},
        )
        assert reject.status_code == 200, reject.text

        body = (await client.get(f"/api/v1/workflows/runs/{run_id}")).json()
        assert body["status"] == "rejected"
        outputs = body["intermediate_outputs"]
        # Downstream step did NOT run.
        assert "implementation" not in outputs
        assert outputs["plan_approved"] is False

        history = outputs["approval_history"]
        assert len(history) == 1
        rec = history[0]
        assert rec["approved"] is False
        assert rec["reason"] == "plan is wrong"
        assert rec["approver_source"] == "ui"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_multi_gate_history_accumulates_in_order() -> None:
    llm = make_mock_llm(text_responses=["the plan", "the final"])
    client, mongo = await build_int_client(_MULTI_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _MULTI_GRAPH_ID, "user_request": "two gates"},
        )
        assert start.status_code == 200
        run_id = start.json()["id"]

        # Pauses at gate1.
        first = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert first.json()["status"] == "waiting_approval"
        assert first.json()["current_step"] == "gate1"

        a1 = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
        assert a1.status_code == 200, a1.text

        # Pauses at gate2.
        mid = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert mid.json()["status"] == "waiting_approval"
        assert mid.json()["current_step"] == "gate2"

        a2 = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
        assert a2.status_code == 200, a2.text

        body = (await client.get(f"/api/v1/workflows/runs/{run_id}")).json()
        assert body["status"] == "completed"
        history = body["intermediate_outputs"]["approval_history"]
        assert [r["step_id"] for r in history] == ["gate1", "gate2"]
        assert all(r["approved"] is True for r in history)
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_slack_identity_recorded_in_history() -> None:
    """A Slack-shaped resume (via callbacks._do_approve with identity args) must
    record approver_source == "slack" plus the given name/id in the audit trail."""
    from app.api.routes.callbacks import _do_approve

    llm = make_mock_llm(text_responses=["the plan", "the implementation"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "slack approve"},
        )
        assert start.status_code == 200
        run_id = start.json()["id"]

        paused = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert paused.json()["status"] == "waiting_approval"

        container = client._transport.app.state.container  # type: ignore[attr-defined]
        status = await _do_approve(
            run_id, container,
            approver_slack_id="U999",
            approver_name="Slack Sam",
            approver_id="U999",
            approver_source="slack",
        )
        assert status == "completed"

        body = (await client.get(f"/api/v1/workflows/runs/{run_id}")).json()
        assert body["status"] == "completed"
        rec = body["intermediate_outputs"]["approval_history"][0]
        assert rec["approver_source"] == "slack"
        assert rec["approver_name"] == "Slack Sam"
        assert rec["approver_id"] == "U999"
        assert rec["approved"] is True
    finally:
        await mongo.close()
