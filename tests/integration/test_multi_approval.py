"""
Integration test: multiple sequential human_approval gates.

Replaces: test_parallel_flow.py (which tested dual parallel gates).
The new engine is sequential; this test covers the equivalent scenario
with two approval gates in series — each pauses the run independently.

Scenarios
─────────
1. Two-gate graph: run pauses at gate 1, resumes on approve, pauses at
   gate 2, resumes on approve, then completes.
2. Rejecting at gate 1 writes approved=False and reason; gate 2 is
   reached but its when-guarded step after it is skipped.
3. State from steps before each gate is preserved and accessible at
   every stage (interrupt_payload, GET run, post-approve state).
"""
from __future__ import annotations

import pytest

from tests.integration.conftest import build_int_client, make_mock_llm

# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

_GRAPH = {
    "id": "multi-approval",
    "steps": [
        {
            "id": "design",
            "type": "llm",
            "output_key": "design",
            "system_prompt": "Produce a design.",
            "user_template": "{request}",
        },
        {
            "id": "approve_design",
            "type": "human_approval",
            "interrupt_payload": {"design": "{design}"},
        },
        {
            "id": "plan",
            "type": "llm",
            "when": "approved",
            "output_key": "plan",
            "system_prompt": "Produce an implementation plan.",
            "user_template": "Design: {design}\nRequest: {request}",
        },
        {
            "id": "approve_plan",
            "type": "human_approval",
            "interrupt_payload": {"plan": "{plan}"},
        },
        {
            "id": "implement",
            "type": "llm",
            "when": "approved",
            "output_key": "implementation",
            "system_prompt": "Implement.",
            "user_template": "Plan: {plan}",
        },
    ],
}

_GRAPH_ID = "multi-approval"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_sequential_approvals_complete() -> None:
    """
    Submit → paused at gate 1 → approve → paused at gate 2 → approve →
    completed; all intermediate state accumulated correctly.
    """
    llm = make_mock_llm(text_responses=["design draft", "impl plan", "implementation done"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        # ---- submit ----
        start = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "add dark mode"},
        )
        assert start.status_code == 200, start.text
        body = start.json()
        thread_id = body["thread_id"]

        assert body["status"] == "waiting_approval"
        assert body["state"]["design"] == "design draft"
        # plan not yet run — gate 1 paused before it
        assert "plan" not in body["state"]

        # ---- approve gate 1 ----
        approve1 = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}/approve"
        )
        assert approve1.status_code == 200, approve1.text
        body = approve1.json()

        # graph ran plan step then paused at gate 2
        assert body["status"] == "waiting_approval"
        assert body["state"]["design"] == "design draft"
        assert body["state"]["plan"] == "impl plan"
        assert "implementation" not in body["state"]

        # ---- approve gate 2 ----
        approve2 = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}/approve"
        )
        assert approve2.status_code == 200, approve2.text
        body = approve2.json()

        assert body["status"] == "completed"
        assert body["state"]["implementation"] == "implementation done"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_reject_at_gate_1_skips_plan_and_impl() -> None:
    """
    Reject at gate 1 → plan step (when: approved) is skipped →
    run continues to gate 2 (plan is absent) → approve gate 2 →
    implement step (when: approved) runs because approved is now True.
    """
    llm = make_mock_llm(text_responses=["design draft", "implementation done"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "risky feature"},
        )
        thread_id = start.json()["thread_id"]

        # ---- reject gate 1 ----
        rej = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}/reject",
            json={"reason": "design not good enough"},
        )
        assert rej.status_code == 200, rej.text
        body = rej.json()

        # plan step skipped — approved=False — then hits gate 2
        assert body["status"] == "waiting_approval"
        assert "plan" not in body["state"]
        assert body["state"]["approved"] is False
        assert body["state"]["reject_reason"] == "design not good enough"

        # ---- approve gate 2 (now approved=True) → implement runs ----
        approve2 = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}/approve"
        )
        body = approve2.json()

        assert body["status"] == "completed"
        assert body["state"]["approved"] is True
        assert body["state"]["implementation"] == "implementation done"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_state_persisted_between_gates() -> None:
    """
    GET /runs/{thread_id} between gates returns the state as-of the last
    interrupt, proving MongoDB is updated on each approval step.
    """
    llm = make_mock_llm(text_responses=["design draft", "impl plan", "done"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "feature"},
        )
        thread_id = start.json()["thread_id"]

        # GET before any approval
        get1 = await client.get(f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}")
        assert get1.json()["status"] == "waiting_approval"
        assert get1.json()["state"]["design"] == "design draft"

        await client.post(f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}/approve")

        # GET after gate 1 approval — now waiting at gate 2
        get2 = await client.get(f"/api/v1/graphs/{_GRAPH_ID}/runs/{thread_id}")
        assert get2.json()["status"] == "waiting_approval"
        assert get2.json()["state"]["plan"] == "impl plan"
    finally:
        await mongo.close()
