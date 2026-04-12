"""
Integration test: MCP fetch → LLM plan → HTTP action → result.

Flow under test
───────────────
POST /runs  →  MCP tool stub called (fetch_context node)
            →  LLM stub called      (plan node)
            →  HTTP executor stub   (run_actions node)
            →  status: completed
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.application.services.planning_service import PlanningService
from app.domain.models.runtime import PlanResult
from app.infrastructure.actions.http_executor import HttpStepExecutor
from app.infrastructure.tools.mcp_client import McpToolsProvider

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_MCP_OUTPUT = {"issue_title": "Add dark mode", "issue_body": "Users want dark mode support."}

_PLAN_STUB = PlanResult(
    summary="mcp plan stub",
    tasks=[],           # no openhands tasks — execute node is a no-op
    execution_order=[],
    outputs_required=[],
)

_HTTP_OUTPUT = {"status": "notified", "webhook_id": "wh-42"}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_fetch_llm_plan_http_action(client) -> None:
    """
    Scenario: user submits a request, the system:
      1. fetches context from an MCP tool (stubbed),
      2. calls the LLM to produce a plan (stubbed),
      3. fires an HTTP action (stubbed),
      4. returns a completed run with all intermediate outputs present.
    """
    # MCP tool stub — get_tool() returns an object whose ainvoke() returns _MCP_OUTPUT
    fake_mcp_tool = MagicMock()
    fake_mcp_tool.ainvoke = AsyncMock(return_value=_MCP_OUTPUT)

    plan_stub = AsyncMock(return_value=_PLAN_STUB)
    http_stub = AsyncMock(return_value=_HTTP_OUTPUT)

    with (
        patch.object(McpToolsProvider, "get_tool", return_value=fake_mcp_tool),
        patch.object(PlanningService, "create_plan", plan_stub),
        patch.object(HttpStepExecutor, "execute", http_stub),
    ):
        # ------------------------------------------------------------------
        # Submit — the whole flow runs without an approval pause
        # ------------------------------------------------------------------
        response = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "mcp_llm_http_flow", "user_request": "add dark mode"},
        )
        assert response.status_code == 201, response.text
        run = response.json()["run"]

        # ------------------------------------------------------------------
        # Final state
        # ------------------------------------------------------------------
        assert run["status"] == "completed"
        assert run["approval_status"] == "not_required"

        # ------------------------------------------------------------------
        # MCP tool was called with the step's tool_input
        # ------------------------------------------------------------------
        fake_mcp_tool.ainvoke.assert_called_once_with({"issue_id": "123"})

        tool_call_results = run["tool_call_results"]
        assert len(tool_call_results) == 1
        assert tool_call_results[0]["step_id"] == "fetch_data"
        assert tool_call_results[0]["tool"] == "get_issue_details"
        assert tool_call_results[0]["status"] == "success"
        assert tool_call_results[0]["output"] == _MCP_OUTPUT

        # MCP output is stored under the step's output_key
        assert run["intermediate_outputs"]["issue_data"] == _MCP_OUTPUT

        # ------------------------------------------------------------------
        # LLM (plan) was called once
        # ------------------------------------------------------------------
        plan_stub.assert_called_once()
        assert run["plan"]["summary"] == "mcp plan stub"

        # ------------------------------------------------------------------
        # HTTP action was called once and its output is recorded
        # ------------------------------------------------------------------
        http_stub.assert_called_once()

        action_results = run["action_results"]
        assert len(action_results) == 1
        assert action_results[0]["step_id"] == "notify"
        assert action_results[0]["type"] == "http"
        assert action_results[0]["status"] == "success"
        assert action_results[0]["output"] == _HTTP_OUTPUT

        assert run["intermediate_outputs"]["notify"] == _HTTP_OUTPUT
