"""
Integration test: full HTTP approval flow against real MongoDB.

Flow under test
───────────────
POST /runs  →  plan LLM stub called  →  status: waiting_approval
GET  /runs/{id}  →  state persisted in MongoDB
POST /runs/{id}/approve  →  execute LLM stub called  →  status: completed
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.application.services.planning_service import PlanningService
from app.domain.models.runtime import OpenHandsExecutionResult, PlanResult, RepositoryTask
from app.infrastructure.integrations.openhands import OpenHandsAdapter

# ---------------------------------------------------------------------------
# Stubs returned by the two LLM calls
# ---------------------------------------------------------------------------

_PLAN_STUB = PlanResult(
    summary="plan stub",
    tasks=[
        RepositoryTask(
            repo="acme/app",
            instructions="do work X",
            order=1,
            step_id="execute_app",
        )
    ],
    execution_order=["acme/app"],
    outputs_required=["branch", "summary", "pr_url"],
)

_EXECUTION_STUB = OpenHandsExecutionResult(
    branch="feature/stub-branch",
    summary="execution stub result",
    pr_url="https://github.com/acme/app/pull/1",
    status="success",
)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_waits_for_approval_then_executes(client) -> None:
    """
    Scenario: a user submits "do work X", the system plans it (first LLM stub),
    pauses for human approval, then after approval executes it (second LLM stub).
    """
    plan_llm_stub = AsyncMock(return_value=_PLAN_STUB)
    execute_llm_stub = AsyncMock(return_value=_EXECUTION_STUB)

    with (
        patch.object(PlanningService, "create_plan", plan_llm_stub),
        patch.object(OpenHandsAdapter, "execute_task", execute_llm_stub),
    ):
        # ------------------------------------------------------------------
        # 1. Submit the workflow — plan stub is called, run pauses at approval
        # ------------------------------------------------------------------
        response = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "feature_flow", "user_request": "do work X"},
        )
        assert response.status_code == 201, response.text
        run = response.json()["run"]
        run_id = run["id"]

        assert run["status"] == "waiting_approval"
        assert run["approval_status"] == "pending"
        assert run["plan"]["summary"] == "plan stub"

        plan_llm_stub.assert_called_once()
        execute_llm_stub.assert_not_called()

        # ------------------------------------------------------------------
        # 2. Read back from MongoDB to verify state is durably persisted
        # ------------------------------------------------------------------
        response = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert response.status_code == 200
        persisted = response.json()["run"]
        assert persisted["status"] == "waiting_approval"
        assert persisted["plan"]["summary"] == "plan stub"

        # ------------------------------------------------------------------
        # 3. Approve — execute stub is called, run completes
        # ------------------------------------------------------------------
        response = await client.post(
            f"/api/v1/workflows/runs/{run_id}/approve",
            json={"feedback": "looks good"},
        )
        assert response.status_code == 200, response.text
        run = response.json()["run"]

        assert run["status"] == "completed"
        assert run["approval_status"] == "approved"
        assert run["metadata"].get("approval_feedback") == "looks good"

        execute_llm_stub.assert_called_once()
