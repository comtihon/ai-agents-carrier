"""
Integration test: parallel fetch + parallel LLM + dual approval gates.

Flow under test
───────────────
POST /runs  →  fetch_figma  ┐  (parallel)
                fetch_github ┘
            →  design_analysis LLM  ┐  (parallel)
               plan_creation LLM    ┘
            →  two approval gates created: approval_design, approval_plan
            →  status: waiting_approval

POST /runs/{id}/gates/approval_design/approve
            →  gate approved; plan gate still pending
            →  status: still waiting_approval

POST /runs/{id}/gates/approval_plan/approve
            →  both gates approved → run resumes → status: completed
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.application.services.llm_agent_service import LlmAgentService
from app.application.services.planning_service import PlanningService
from app.domain.models.runtime import LlmStepResult, PlanResult
from app.infrastructure.tools.mcp_client import McpToolsProvider

# ---------------------------------------------------------------------------
# Stub data
# ---------------------------------------------------------------------------

_FIGMA_CONTEXT = {"components": ["Button", "Modal", "ThemeProvider"]}
_GITHUB_CONTEXT = {"files": ["src/App.tsx", "src/theme.ts"], "default_branch": "main"}

_DESIGN_RESPONSE = "Updated design: switch to CSS custom properties for all colour tokens."
_PLAN_RESPONSE = "Plan: 1. Add CSS vars  2. Update ThemeProvider  3. Add tests"

_EMPTY_PLAN = PlanResult(summary="no tasks", tasks=[], execution_order=[], outputs_required=[])


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_fetch_llm_dual_gate_approval(client) -> None:
    """
    Scenario:
      - Two MCP fetch steps run in parallel (Figma + GitHub).
      - Two LLM steps run in parallel (design analysis + plan creation).
      - Two approval gates are created; the run pauses.
      - Approving the first gate keeps the run waiting.
      - Approving the second gate resumes and completes the run.
      - The LLM is NOT called again on resume (skip-on-resume logic).
    """
    # --- MCP tool stubs ---
    figma_tool = MagicMock()
    figma_tool.name = "figma_get_file"
    figma_tool.ainvoke = AsyncMock(return_value=_FIGMA_CONTEXT)

    github_tool = MagicMock()
    github_tool.name = "github_get_repo"
    github_tool.ainvoke = AsyncMock(return_value=_GITHUB_CONTEXT)

    def _get_tool(name: str):
        return {"figma_get_file": figma_tool, "github_get_repo": github_tool}.get(name)

    # --- LLM stubs: two parallel steps, each returning a distinct response ---
    design_result = LlmStepResult(response=_DESIGN_RESPONSE, tool_calls_made=[])
    plan_result = LlmStepResult(response=_PLAN_RESPONSE, tool_calls_made=[])
    llm_run_mock = AsyncMock(side_effect=[design_result, plan_result])

    with (
        patch.object(McpToolsProvider, "get_tool", side_effect=_get_tool),
        patch.object(LlmAgentService, "run", llm_run_mock),
        patch.object(PlanningService, "create_plan", AsyncMock(return_value=_EMPTY_PLAN)),
    ):
        # ------------------------------------------------------------------
        # 1. Submit — both fetches and both LLM steps run, then pauses
        # ------------------------------------------------------------------
        response = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "parallel_flow", "user_request": "add dark mode"},
        )
        assert response.status_code == 201, response.text
        run = response.json()["run"]
        run_id = run["id"]

        assert run["status"] == "waiting_approval"

        # Two gates created, both pending
        gates = {g["gate_id"]: g for g in run["approval_gates"]}
        assert len(gates) == 2
        assert gates["approval_design"]["status"] == "pending"
        assert gates["approval_plan"]["status"] == "pending"

        # Both MCP tools were called (parallel fetch)
        figma_tool.ainvoke.assert_called_once()
        github_tool.ainvoke.assert_called_once()
        assert run["intermediate_outputs"]["design_context"] == _FIGMA_CONTEXT
        assert run["intermediate_outputs"]["code_context"] == _GITHUB_CONTEXT

        # Both LLM steps ran (parallel LLM)
        assert llm_run_mock.call_count == 2
        assert run["intermediate_outputs"]["design_update"]["response"] == _DESIGN_RESPONSE
        assert run["intermediate_outputs"]["solution_plan"]["response"] == _PLAN_RESPONSE

        # ------------------------------------------------------------------
        # 2. Approve first gate — still waiting for the second
        # ------------------------------------------------------------------
        response = await client.post(
            f"/api/v1/workflows/runs/{run_id}/gates/approval_design/approve",
            json={"feedback": "design looks great"},
        )
        assert response.status_code == 200, response.text
        run = response.json()["run"]

        assert run["status"] == "waiting_approval"

        gates = {g["gate_id"]: g for g in run["approval_gates"]}
        assert gates["approval_design"]["status"] == "approved"
        assert gates["approval_design"]["feedback"] == "design looks great"
        assert gates["approval_plan"]["status"] == "pending"

        # LLM still called only twice (no re-run yet)
        assert llm_run_mock.call_count == 2

        # ------------------------------------------------------------------
        # 3. Approve second gate — both approved → run resumes and completes
        # ------------------------------------------------------------------
        response = await client.post(
            f"/api/v1/workflows/runs/{run_id}/gates/approval_plan/approve",
            json={"feedback": "plan approved"},
        )
        assert response.status_code == 200, response.text
        run = response.json()["run"]

        assert run["status"] == "completed"

        gates = {g["gate_id"]: g for g in run["approval_gates"]}
        assert gates["approval_design"]["status"] == "approved"
        assert gates["approval_plan"]["status"] == "approved"
        assert gates["approval_plan"]["feedback"] == "plan approved"

        # LLM was NOT called again on resume — skip-on-resume logic works
        assert llm_run_mock.call_count == 2
