"""
Integration test: full HTTP approval / rejection flow against real MongoDB.

Scenarios
─────────
1. Submit run → graph pauses at human_approval → status=waiting_approval
   MongoDB row is persisted and readable via GET.
2. Approve → implement step (when: approved) runs → status=completed.
3. Reject  → implement step is skipped        → status=completed, implementation absent.
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
    Submit a request → the graph interrupts at human_approval →
    status=waiting_approval, MongoDB row persisted.
    """
    llm = make_mock_llm(text_responses=["step-by-step plan stub"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        resp = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "implement feature X"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        thread_id = body["thread_id"]

        assert body["status"] == "waiting_approval"
        assert body["state"]["plan"] == "step-by-step plan stub"

        # ------------------------------------------------------------------
        # MongoDB persisted the run
        # ------------------------------------------------------------------
        get_resp = await client.get(f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}")
        assert get_resp.status_code == 200
        persisted = get_resp.json()
        assert persisted["thread_id"] == thread_id
        assert persisted["status"] == "waiting_approval"
        assert persisted["state"]["plan"] == "step-by-step plan stub"
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
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "implement feature Y"},
        )
        assert start.status_code == 200
        thread_id = start.json()["thread_id"]

        approve = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}/approve"
        )
        assert approve.status_code == 200, approve.text
        body = approve.json()

        assert body["status"] == "completed"
        assert body["state"]["plan"] == "the plan"
        assert body["state"]["implementation"] == "the implementation"
        assert body["state"]["approved"] is True

        # MongoDB reflects completed status
        get_resp = await client.get(f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}")
        assert get_resp.json()["status"] == "completed"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_reject_skips_implement_step() -> None:
    """
    Submit → reject → implement step is skipped (when: approved=False) →
    status=completed, implementation key absent.
    """
    llm = make_mock_llm(text_responses=["the plan"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "implement feature Z"},
        )
        thread_id = start.json()["thread_id"]

        reject = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}/reject",
            json={"reason": "plan looks wrong"},
        )
        assert reject.status_code == 200, reject.text
        body = reject.json()

        assert body["status"] == "completed"
        assert body["state"]["approved"] is False
        assert body["state"]["reject_reason"] == "plan looks wrong"
        # implement step skipped — key not set
        assert "implementation" not in body["state"]
    finally:
        await mongo.close()
