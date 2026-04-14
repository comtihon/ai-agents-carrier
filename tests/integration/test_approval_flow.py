"""
Integration test: full HTTP approval / rejection flow against real MongoDB.

Scenarios
─────────
1. Submit run → POST returns status=running immediately; background task runs
   the graph, pauses at human_approval, and persists status=waiting_approval
   so a subsequent GET reflects the correct state.
2. Approve → implement step (when: approved) runs → status=completed.
3. Reject  → implement step is skipped → status=cancelled, implementation absent.
"""
from __future__ import annotations

import pytest

from tests.integration.conftest import build_int_client, make_mock_llm

# ---------------------------------------------------------------------------
# Graph definition under test
# ---------------------------------------------------------------------------

_GRAPH = {
    "id": "approval-test",
    "steps": [
        {
            "id": "plan",
            "type": "llm",
            "output_key": "plan",
            "system_prompt": "Produce a short plan.",
            "user_template": "{request}",
        },
        {
            "id": "wait_for_approval",
            "type": "human_approval",
            "interrupt_payload": {"plan": "{plan}"},
        },
        {
            "id": "implement",
            "type": "llm",
            "when": "approved",
            "output_key": "implementation",
            "system_prompt": "Implement the plan.",
            "user_template": "Plan: {plan}",
        },
    ],
}

_GRAPH_ID = "approval-test"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_pauses_at_approval() -> None:
    """
    Submit a request → POST returns status=running immediately (non-blocking).
    Background task runs the graph, pauses at human_approval, and persists the
    result, so GET reflects status=waiting_approval with the plan in state.
    """
    llm = make_mock_llm(text_responses=["step-by-step plan stub"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        resp = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "implement feature X"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        run_id = body["id"]

        # POST returns immediately with "running"
        assert body["status"] == "running"

        # Background task completes within the ASGI call; GET reflects final state
        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert get_resp.status_code == 200
        persisted = get_resp.json()
        assert persisted["id"] == run_id
        assert persisted["status"] == "waiting_approval"
        assert persisted["intermediate_outputs"]["plan"] == "step-by-step plan stub"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_approve_runs_implement_step() -> None:
    """
    Submit → approve → implement step runs → status=completed,
    implementation key present in state.
    """
    llm = make_mock_llm(text_responses=["the plan", "the implementation"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "implement feature Y"},
        )
        assert start.status_code == 200
        run_id = start.json()["id"]

        approve = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
        assert approve.status_code == 200, approve.text
        body = approve.json()

        assert body["status"] == "completed"
        assert body["intermediate_outputs"]["plan"] == "the plan"
        assert body["intermediate_outputs"]["implementation"] == "the implementation"
        assert body["intermediate_outputs"]["approved"] is True

        # MongoDB reflects completed status
        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert get_resp.json()["status"] == "completed"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_reject_skips_implement_step() -> None:
    """
    Submit → reject → implement step is skipped (when: approved=False) →
    status=cancelled, implementation key absent.
    """
    llm = make_mock_llm(text_responses=["the plan"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "implement feature Z"},
        )
        run_id = start.json()["id"]

        reject = await client.post(
            f"/api/v1/workflows/runs/{run_id}/reject",
            json={"reason": "plan looks wrong"},
        )
        assert reject.status_code == 200, reject.text
        body = reject.json()

        assert body["status"] == "cancelled"
        assert body["intermediate_outputs"]["approved"] is False
        assert body["intermediate_outputs"]["reject_reason"] == "plan looks wrong"
        # implement step skipped — key not set
        assert "implementation" not in body["intermediate_outputs"]
    finally:
        await mongo.close()
