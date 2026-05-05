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

import asyncio

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

        # Background task completes; GET reflects final state
        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = get_resp.json()

        assert body["status"] == "completed"
        assert body["intermediate_outputs"]["plan"] == "the plan"
        assert body["intermediate_outputs"]["implementation"] == "the implementation"
        assert body["intermediate_outputs"]["approved"] is True
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_double_approve_returns_409() -> None:
    """
    Once a run has been approved, the second /approve POST must 409 instead of
    re-resuming the graph. Without this guard, multi-clicks in the UI would
    re-trigger the next node and create duplicate side effects (e.g. duplicate
    Jira tickets).
    """
    llm = make_mock_llm(text_responses=["the plan", "the implementation"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "implement feature dup"},
        )
        assert start.status_code == 200
        run_id = start.json()["id"]

        first = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
        assert first.status_code == 200, first.text

        # Run is now past the approval gate; a second approve must be refused.
        second = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
        assert second.status_code == 409, second.text
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_concurrent_approve_serialised_no_double_resume() -> None:
    """
    Two concurrent POST /approve requests on the same waiting_approval run
    must serialise: exactly one returns 200, the other 409. The run must
    end up in a terminal state — never wedged back at waiting_approval.

    Background
    ──────────
    Pre-fix the handler did read-then-write on run.status (TOCTOU). Two
    near-simultaneous clicks both passed `if status != waiting_approval`,
    both wrote status=running, both scheduled a background resume task on
    the same langgraph runner. The two resumes raced: one would land its
    end-of-stream `run.status = waiting_approval if snap.next else completed`
    write *after* the other had already moved past, leaving the run stuck
    at waiting_approval — the user-facing symptom was "I press approve but
    it returns back".

    Fix is `MongoGraphRunRepository.claim_for_resume`, an atomic
    `find_one_and_update({status: waiting_approval} → {status: running})`.
    Only one caller observes the swap; the other gets None → 409.
    """
    llm = make_mock_llm(text_responses=["the plan", "the implementation"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "concurrent-approve"},
        )
        assert start.status_code == 200
        run_id = start.json()["id"]

        # Sanity: the run is paused at the approval gate.
        paused = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert paused.json()["status"] == "waiting_approval"

        r1, r2 = await asyncio.gather(
            client.post(f"/api/v1/workflows/runs/{run_id}/approve"),
            client.post(f"/api/v1/workflows/runs/{run_id}/approve"),
        )
        codes = sorted([r1.status_code, r2.status_code])
        assert codes == [200, 409], (
            f"expected one 200 and one 409, got {codes} (r1={r1.text!r}, r2={r2.text!r})"
        )

        # Run progresses past the gate — never falls back to waiting_approval.
        final = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert final.json()["status"] == "completed", (
            f"run wedged at status={final.json()['status']}"
        )
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_claim_for_resume_atomic_at_repo_level() -> None:
    """
    The atomic guarantee lives in the repository, not the handler. Two
    concurrent claims against the same waiting_approval run must produce
    exactly one winner — verified directly on the repo to pin the
    contract independently of the HTTP layer.
    """
    from app.core.config import Settings
    from app.domain.models.graph_run import GraphRun
    from app.infrastructure.persistence.mongo import MongoClientProvider

    settings = Settings(
        mongodb_uri="mongodb://localhost:27017",
        mongodb_database="test_langgraph_integration",
        environment="test",
    )
    mongo = MongoClientProvider(settings)
    repo = mongo.get_repository()
    try:
        await repo._collection.delete_many({})
        run = GraphRun(
            id="claim-race",
            graph_id="approval-test",
            user_request="x",
            status="waiting_approval",
        )
        await repo.create(run)

        results = await asyncio.gather(
            repo.claim_for_resume("claim-race"),
            repo.claim_for_resume("claim-race"),
            repo.claim_for_resume("claim-race"),
        )
        winners = [r for r in results if r is not None]
        losers = [r for r in results if r is None]
        assert len(winners) == 1, f"expected exactly one winner, got {len(winners)}"
        assert len(losers) == 2
        assert winners[0].status == "running"

        # And after the swap, status is persisted as running — a fourth
        # claim cannot succeed.
        again = await repo.claim_for_resume("claim-race")
        assert again is None
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_callback_approve_unblocks_workflow() -> None:
    """
    POST /api/v1/callbacks/{run_id}/approve (no auth) should resume the run
    exactly like the authenticated /approve endpoint.
    """
    llm = make_mock_llm(text_responses=["the plan", "the implementation"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "implement feature CB"},
        )
        assert start.status_code == 200
        run_id = start.json()["id"]

        # Verify paused
        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert get_resp.json()["status"] == "waiting_approval"

        approve = await client.post(f"/api/v1/callbacks/{run_id}/approve")
        assert approve.status_code == 200, approve.text
        body = approve.json()

        assert body["status"] == "completed"
        assert body["run_id"] == run_id

        # MongoDB reflects completed status
        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        result = get_resp.json()
        assert result["status"] == "completed"
        assert result["intermediate_outputs"]["approved"] is True
        assert result["intermediate_outputs"]["implementation"] == "the implementation"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_callback_reject_unblocks_workflow() -> None:
    """
    POST /api/v1/callbacks/{run_id}/reject should cancel the run and skip
    conditional steps, matching the behaviour of the authenticated /reject endpoint.
    """
    llm = make_mock_llm(text_responses=["the plan"])
    client, mongo = await build_int_client(_GRAPH, llm)
    try:
        start = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "implement feature CB-reject"},
        )
        assert start.status_code == 200
        run_id = start.json()["id"]

        reject = await client.post(
            f"/api/v1/callbacks/{run_id}/reject",
            json={"reason": "not ready"},
        )
        assert reject.status_code == 200, reject.text
        body = reject.json()

        assert body["status"] == "cancelled"
        assert body["run_id"] == run_id

        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        result = get_resp.json()
        assert result["status"] == "cancelled"
        assert result["intermediate_outputs"]["approved"] is False
        assert result["intermediate_outputs"]["reject_reason"] == "not ready"
        assert "implementation" not in result["intermediate_outputs"]
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

        # Background task completes; GET reflects final state
        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = get_resp.json()

        assert body["status"] == "cancelled"
        assert body["intermediate_outputs"]["approved"] is False
        assert body["intermediate_outputs"]["reject_reason"] == "plan looks wrong"
        # implement step skipped — key not set
        assert "implementation" not in body["intermediate_outputs"]
    finally:
        await mongo.close()
