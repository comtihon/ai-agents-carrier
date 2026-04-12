"""
Integration test: LLM tool-calling loop.

Flow under test
───────────────
POST /runs  →  LlmAgentService.run called:
               1. LLM called with user request
               2. LLM returns tool_call for "search_docs"
               3. "search_docs" local tool executed
               4. Tool result appended to messages, LLM called again
               5. LLM returns final text answer
            →  status: completed
            →  intermediate_outputs["agent_result"]["response"] = final answer
            →  intermediate_outputs["agent_result"]["tool_calls_made"] = [tool call record]
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.application.services.llm_agent_service import LlmAgentService
from app.core.config import Settings
from app.core.container import build_container

# ---------------------------------------------------------------------------
# Test settings — isolated DB, no real external calls
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

_TOOL_RESULT = {"results": ["Use CSS variables for theming.", "See MDN prefers-color-scheme guide."]}

_FINAL_ANSWER = "To add dark mode, use CSS custom properties and the prefers-color-scheme media query."


# ---------------------------------------------------------------------------
# Fixture — injects a fake LlmAgentService so the loop runs without real LLM
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def llm_client():
    """
    Spin up the full FastAPI app with a fake LlmAgentService injected into the
    graph runner.  The fake service uses:
      - a stub local tool ("search_docs") whose ainvoke returns _TOOL_RESULT
      - a stub LLM whose first call requests that tool, second returns _FINAL_ANSWER
    """
    # --- fake local tool ---
    fake_tool = MagicMock()
    fake_tool.name = "search_docs"
    fake_tool.ainvoke = AsyncMock(return_value=_TOOL_RESULT)

    # --- fake LLM: call 1 → tool_call, call 2 → final answer ---
    first_llm_response = AIMessage(
        content="",
        tool_calls=[{
            "id": "call_abc",
            "name": "search_docs",
            "args": {"query": "dark mode CSS"},
            "type": "tool_call",
        }],
    )
    second_llm_response = AIMessage(content=_FINAL_ANSWER)

    fake_bound_llm = MagicMock()
    fake_bound_llm.ainvoke = AsyncMock(side_effect=[first_llm_response, second_llm_response])

    fake_llm = MagicMock()
    fake_llm.bind_tools.return_value = fake_bound_llm

    # --- wire fake service into the container ---
    fake_service = LlmAgentService(llm=fake_llm, tools=[fake_tool])

    def _build(settings: Settings):
        container = build_container(settings)
        container.graph_runner._llm_agent_service = fake_service
        return container

    with (
        patch("app.api.app.build_container", side_effect=lambda _: _build(_TEST_SETTINGS)),
        patch("app.api.app._register_langserve_routes"),
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
                    llm=fake_bound_llm,
                    tool=fake_tool,
                )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_calls_tool_then_returns_final_answer(llm_client) -> None:
    """
    Scenario: the user asks a question, the LLM requests a local tool call,
    the tool runs, its result is fed back to the LLM, and the LLM returns
    a final answer stored in the run's intermediate outputs.
    """
    response = await llm_client.client.post(
        "/api/v1/workflows/runs",
        json={"workflow_id": "llm_tool_call_flow", "user_request": "How do I add dark mode?"},
    )
    assert response.status_code == 201, response.text
    run = response.json()["run"]

    # ------------------------------------------------------------------
    # Run completed end-to-end
    # ------------------------------------------------------------------
    assert run["status"] == "completed"

    # ------------------------------------------------------------------
    # LLM was called exactly twice: first returns tool_call, then final answer
    # ------------------------------------------------------------------
    assert llm_client.llm.ainvoke.call_count == 2

    # ------------------------------------------------------------------
    # Local tool was invoked once with the args the LLM requested
    # ------------------------------------------------------------------
    llm_client.tool.ainvoke.assert_called_once_with({"query": "dark mode CSS"})

    # ------------------------------------------------------------------
    # Final LLM answer is stored in intermediate_outputs under the step's output_key
    # ------------------------------------------------------------------
    agent_result = run["intermediate_outputs"]["agent_result"]
    assert agent_result["response"] == _FINAL_ANSWER

    # ------------------------------------------------------------------
    # The tool call record is present with name, args, and result
    # ------------------------------------------------------------------
    assert len(agent_result["tool_calls_made"]) == 1
    tool_call = agent_result["tool_calls_made"][0]
    assert tool_call["name"] == "search_docs"
    assert tool_call["args"] == {"query": "dark mode CSS"}
    assert tool_call["result"] == _TOOL_RESULT
