"""
Integration test: classifier node selects the right MCP fetcher.

Flow under test
───────────────
POST /runs  →  classifier node: LLM picks "fetch_github" (not Jira)
            →  fetch_context node: only github_get_repo is called
            →  llm_agent node: runs with repo_context available
            →  status: completed
            →  jira tool never invoked
            →  repo_context populated with GitHub data
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager

from app.api.app import create_app
from app.application.services.classifier_service import ClassifierService
from app.application.services.llm_agent_service import LlmAgentService
from app.core.config import Settings
from app.core.container import build_container
from app.domain.models.runtime import LlmStepResult
from app.infrastructure.tools.mcp_client import McpToolsProvider

# ---------------------------------------------------------------------------
# Test settings
# ---------------------------------------------------------------------------

_TEST_SETTINGS = Settings(
    mongodb_uri="mongodb://localhost:27017",
    mongodb_database="test_langgraph_integration",
    openhands_mock_mode=True,
    environment="test",
    workflow_definitions_path="workflows",
)

# ---------------------------------------------------------------------------
# Stub data
# ---------------------------------------------------------------------------

_GITHUB_OUTPUT = {
    "repo": "acme/my-service",
    "default_branch": "main",
    "open_issues": 3,
    "description": "Core service for the Acme platform.",
}

_LLM_ANSWER = "To implement feature X in acme/my-service, create a new module under src/features/x."


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def classifier_client():
    """
    Full FastAPI app with:
      - ClassifierService.classify stubbed to select only fetch_github
      - McpToolsProvider.get_tool stubbed for both jira and github tools
      - LlmAgentService.run stubbed to return a final answer
    """
    # --- MCP tool stubs ---
    github_tool = MagicMock()
    github_tool.description = "Fetches repository metadata from GitHub."
    github_tool.ainvoke = AsyncMock(return_value=_GITHUB_OUTPUT)

    jira_tool = MagicMock()
    jira_tool.description = "Fetches issue details from Jira."
    jira_tool.ainvoke = AsyncMock(return_value={"issue": "should not be called"})

    def _get_tool(name: str):
        if name == "github_get_repo":
            return github_tool
        if name == "jira_get_issue":
            return jira_tool
        return None

    # --- LLM stub ---
    fake_llm_run = AsyncMock(
        return_value=LlmStepResult(response=_LLM_ANSWER, tool_calls_made=[])
    )

    def _build(settings: Settings):
        container = build_container(settings)
        container.graph_runner._llm_agent_service._llm_with_tools = None
        return container

    with (
        patch("app.api.app.build_container", side_effect=lambda _: _build(_TEST_SETTINGS)),
        patch("app.api.app._register_langserve_routes"),
        patch.object(McpToolsProvider, "get_tool", side_effect=_get_tool),
        patch.object(ClassifierService, "classify", AsyncMock(return_value=["fetch_github"])),
        patch.object(LlmAgentService, "run", fake_llm_run),
    ):
        app = create_app()
        async with LifespanManager(app):
            await app.state.container.workflow_run_repository._collection.delete_many({})
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as http_client:
                yield SimpleNamespace(
                    client=http_client,
                    github_tool=github_tool,
                    jira_tool=jira_tool,
                )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classifier_selects_github_fetcher(classifier_client) -> None:
    """
    Scenario: user asks to implement a feature in a GitHub repo.
    The classifier picks only the GitHub fetcher; Jira is skipped.
    The LLM step receives the GitHub context and produces a final answer.
    """
    response = await classifier_client.client.post(
        "/api/v1/workflows/runs",
        json={
            "workflow_id": "classifier_flow",
            "user_request": "implement feature X in repo acme/my-service",
        },
    )
    assert response.status_code == 201, response.text
    run = response.json()["run"]

    # ------------------------------------------------------------------
    # Run completed end-to-end
    # ------------------------------------------------------------------
    assert run["status"] == "completed"

    # ------------------------------------------------------------------
    # Only the GitHub tool was called; Jira was skipped
    # ------------------------------------------------------------------
    classifier_client.github_tool.ainvoke.assert_called_once_with({"repo": "acme/my-service"})
    classifier_client.jira_tool.ainvoke.assert_not_called()

    # ------------------------------------------------------------------
    # Only one tool_call_result — the GitHub fetch
    # ------------------------------------------------------------------
    tool_call_results = run["tool_call_results"]
    assert len(tool_call_results) == 1
    assert tool_call_results[0]["step_id"] == "fetch_github"
    assert tool_call_results[0]["tool"] == "github_get_repo"
    assert tool_call_results[0]["status"] == "success"
    assert tool_call_results[0]["output"] == _GITHUB_OUTPUT

    # ------------------------------------------------------------------
    # GitHub context stored under the step's output_key
    # ------------------------------------------------------------------
    assert run["intermediate_outputs"]["repo_context"] == _GITHUB_OUTPUT

    # Jira output key is absent
    assert "jira_context" not in run["intermediate_outputs"]

    # ------------------------------------------------------------------
    # LLM step ran and its answer is stored
    # ------------------------------------------------------------------
    assert len(run["llm_agent_results"]) == 1
    assert run["llm_agent_results"][0]["step_id"] == "ask_llm"
    assert run["llm_agent_results"][0]["status"] == "success"
    assert run["intermediate_outputs"]["agent_result"]["response"] == _LLM_ANSWER
