"""
Integration test: multiple sequential human_approval gates.

Replaces: test_parallel_flow.py (which tested dual parallel gates).
The new engine is sequential; this test covers the equivalent scenario
with two approval gates in series — each pauses the run independently.

Scenarios
─────────
1. Two-gate graph: POST /runs returns "running" immediately; GET after POST
   shows waiting_approval at gate 1; approve → waiting_approval at gate 2;
   approve → completed.
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
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "add dark mode"},
        )
        assert start.status_code == 200, start.text
        run_id = start.json()["id"]

        # POST returns immediately with "running"; GET reflects post-background-task state
        get_start = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = get_start.json()
        assert body["status"] == "waiting_approval"
        assert body["intermediate_outputs"]["design"] == "design draft"
        # plan not yet run — gate 1 paused before it
        assert "plan" not in body["intermediate_outputs"]

        # ---- approve gate 1 ----
        approve1 = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
        assert approve1.status_code == 200, approve1.text

        # Background task completes; GET reflects state after gate 1 (paused at gate 2)
        get1 = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = get1.json()

        # graph ran plan step then paused at gate 2
        assert body["status"] == "waiting_approval"
        assert body["intermediate_outputs"]["design"] == "design draft"
        assert body["intermediate_outputs"]["plan"] == "impl plan"
        assert "implementation" not in body["intermediate_outputs"]

        # ---- approve gate 2 ----
        approve2 = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
        assert approve2.status_code == 200, approve2.text

        # Background task completes; GET reflects final state
        get2 = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = get2.json()

        assert body["status"] == "completed"
        assert body["intermediate_outputs"]["implementation"] == "implementation done"
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
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "risky feature"},
        )
        run_id = start.json()["id"]

        # ---- reject gate 1 ----
        rej = await client.post(
            f"/api/v1/workflows/runs/{run_id}/reject",
            json={"reason": "design not good enough"},
        )
        assert rej.status_code == 200, rej.text

        # Background task completes; GET reflects state after gate 1 rejection
        get_rej = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = get_rej.json()

        # plan step skipped — approved=False — then hits gate 2
        assert body["status"] == "waiting_approval"
        assert "plan" not in body["intermediate_outputs"]
        assert body["intermediate_outputs"]["approved"] is False
        assert body["intermediate_outputs"]["reject_reason"] == "design not good enough"

        # ---- approve gate 2 (now approved=True) → implement runs ----
        approve2 = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
        assert approve2.status_code == 200, approve2.text

        # Background task completes; GET reflects final state
        get2 = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = get2.json()

        assert body["status"] == "completed"
        assert body["intermediate_outputs"]["approved"] is True
        assert body["intermediate_outputs"]["implementation"] == "implementation done"
    finally:
        await mongo.close()


_GRAPH_GATED_APPROVAL = {
    "id": "gated-approval",
    "steps": [
        {
            "id": "deploy_uat_gate",
            "type": "human_approval",
            "output_key": "deploy_approved",
            "interrupt_payload": {"target": "uat"},
        },
        {
            "id": "deploy_uat",
            "type": "llm",
            "when": "deploy_approved",
            "output_key": "uat_result",
            "system_prompt": "Deploy.",
            "user_template": "{request}",
        },
        # The gate the user reported as firing-when-it-shouldn't:
        # `when: deploy_approved` must skip this approval prompt entirely
        # if the previous gate was rejected, not just skip the deploy step.
        {
            "id": "deploy_staging_gate",
            "type": "human_approval",
            "output_key": "release_approved",
            "when": "deploy_approved",
            "interrupt_payload": {"target": "staging"},
        },
        {
            "id": "deploy_staging",
            "type": "llm",
            "when": "release_approved",
            "output_key": "staging_result",
            "system_prompt": "Deploy staging.",
            "user_template": "{request}",
        },
    ],
}


@pytest.mark.asyncio
async def test_rejecting_first_approval_skips_when_gated_second_approval() -> None:
    """
    Reproduces the user-reported bug: rejecting the deploy-to-UAT gate
    skipped the deploy step (correct, `when: deploy_approved`) but the
    next human_approval (`when: deploy_approved`) STILL prompted. Cause:
    `_approval_node` did not honour `step.when`, unlike every other node
    type. After the fix the second approval must skip and the run must
    reach a terminal state without ever re-entering waiting_approval.
    """
    llm = make_mock_llm(text_responses=["unused"])
    client, mongo = await build_int_client(_GRAPH_GATED_APPROVAL, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "gated-approval", "user_request": "ship it"},
        )
        assert start.status_code == 200, start.text
        run_id = start.json()["id"]

        # Paused at deploy_uat_gate.
        first = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert first.json()["status"] == "waiting_approval"
        assert first.json()["current_step"] == "deploy_uat_gate"

        # Reject — deploy_approved becomes False.
        rej = await client.post(
            f"/api/v1/workflows/runs/{run_id}/reject",
            json={"reason": "skip uat"},
        )
        assert rej.status_code == 200, rej.text

        # The previously-broken behaviour: status would be waiting_approval
        # at deploy_staging_gate. With the fix the gate skips and the run
        # ends (cancelled, because the rejection cascades through the
        # remaining when-guarded steps).
        final = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = final.json()
        assert body["status"] in ("completed", "cancelled"), (
            f"second approval prompted again — current_step={body['current_step']}, "
            f"status={body['status']}"
        )
        assert body["intermediate_outputs"]["deploy_approved"] is False
        # The release_approved gate must not have prompted, so its key is absent.
        assert "release_approved" not in body["intermediate_outputs"]
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_state_persisted_between_gates() -> None:
    """
    GET /runs/{run_id} between gates returns the state as-of the last
    interrupt, proving MongoDB is updated on each approval step.
    """
    llm = make_mock_llm(text_responses=["design draft", "impl plan", "done"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "feature"},
        )
        run_id = start.json()["id"]

        # GET before any approval
        get1 = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert get1.json()["status"] == "waiting_approval"
        assert get1.json()["intermediate_outputs"]["design"] == "design draft"

        await client.post(f"/api/v1/workflows/runs/{run_id}/approve")

        # GET after gate 1 approval — now waiting at gate 2
        get2 = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert get2.json()["status"] == "waiting_approval"
        assert get2.json()["intermediate_outputs"]["plan"] == "impl plan"
    finally:
        await mongo.close()
