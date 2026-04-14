"""
Integration tests for the agentic tool-calling loop inside llm_structured steps.

Two scenarios
─────────────
1. bash_tool_loop
   Request arrives → LLM asks to run a bash-like tool → we execute it and
   feed the result back → LLM returns the final structured output.

2. multi_turn_mcp_loop
   Request arrives → LLM asks to call MCP tool A → we fetch and feed back →
   LLM asks to call MCP tool B with different arguments → we fetch and feed
   back → LLM returns the final structured output.

Both tests also assert that the tool results actually appeared in the messages
the LLM received on subsequent iterations.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from tests.integration.conftest import build_int_client

_SUBMIT = YamlGraphRunner._SUBMIT_TOOL

# ---------------------------------------------------------------------------
# Shared graph: a single llm_structured step that may loop over tool calls
# ---------------------------------------------------------------------------

_GRAPH_ID = "agentic-loop"

_GRAPH = {
    "id": _GRAPH_ID,
    "steps": [
        {
            "id": "compute",
            "type": "llm_structured",
            "system_prompt": "Use available tools to gather information, then submit the final answer.",
            "user_template": "{request}",
            "output": [
                {"name": "answer", "type": "str", "description": "The final answer"},
            ],
        }
    ],
}


# ---------------------------------------------------------------------------
# Mock LLM factory for multi-iteration loops
# ---------------------------------------------------------------------------

def _make_loop_llm(
    loop_responses: list[list[dict[str, Any]]],
    text_responses: list[str] | None = None,
) -> tuple[MagicMock, list[list[Any]]]:
    """
    Build a mock LLM whose bind_tools().ainvoke() returns a different set of
    tool_calls on each successive call, driven by *loop_responses*.

    Also returns *captured_messages*: a list that will be populated with the
    full message list passed to the LLM on each iteration, so tests can
    assert that tool results were properly threaded back.

    *text_responses* drives direct .ainvoke() calls (llm steps); unused here
    but required for compatibility with build_int_client.
    """
    r_iter = iter(loop_responses)
    t_iter = iter(text_responses or ["default"])
    captured_messages: list[list[Any]] = []

    llm = MagicMock()

    async def _ainvoke(messages, **kwargs):
        return AIMessage(content=next(t_iter, "default"))

    llm.ainvoke = AsyncMock(side_effect=_ainvoke)

    def _bind_tools(tools, **kwargs):
        chain = MagicMock()

        async def _chain_ainvoke(messages, **kwargs):
            captured_messages.append(list(messages))
            return AIMessage(content="", tool_calls=next(r_iter))

        chain.ainvoke = AsyncMock(side_effect=_chain_ainvoke)
        return chain

    llm.bind_tools = MagicMock(side_effect=_bind_tools)
    return llm, captured_messages


def _mock_tool(return_value: Any) -> MagicMock:
    t = MagicMock()
    t.ainvoke = AsyncMock(return_value=return_value)
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_tool_loop() -> None:
    """
    LLM asks to run a bash-like tool, receives the output, then submits the
    final structured answer.

    Verifies:
    - The tool was called with exactly the arguments the LLM requested.
    - On the second LLM call the message history contains a ToolMessage whose
      content matches the tool's return value (i.e. the result was fed back).
    - The run completes with the expected structured output in state.
    """
    bash_tool = _mock_tool("2")

    llm, captured = _make_loop_llm([
        # Iteration 1 — LLM decides it needs to run a command
        [{"name": "bash", "args": {"command": "echo $((1+1))"}, "id": "tc-bash-1"}],
        # Iteration 2 — LLM has the result, submits the final answer
        [{"name": _SUBMIT, "args": {"answer": "2"}, "id": "tc-submit-1"}],
    ])

    client, mongo = await build_int_client(
        _GRAPH, llm, mcp_tools={"bash": bash_tool}
    )
    try:
        resp = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "what is 1+1?"},
        )
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["id"]

        # Tool was invoked with the right arguments
        bash_tool.ainvoke.assert_called_once_with({"command": "echo $((1+1))"})

        # On the second LLM call the messages must include the tool result
        assert len(captured) == 2, f"expected 2 LLM iterations, got {len(captured)}"
        second_call_messages = captured[1]
        tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "2"

        # Final structured output is stored in run state
        state = (await client.get(f"/api/v1/workflows/runs/{run_id}")).json()["intermediate_outputs"]
        assert state["answer"] == "2"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_multi_turn_mcp_loop() -> None:
    """
    LLM calls MCP tool A, receives the result, then calls MCP tool B with
    different arguments, receives that result, and finally submits the
    structured output.

    Verifies:
    - Each MCP tool was called exactly once with the expected arguments.
    - On the third LLM call the message history contains ToolMessages for
      both tool A and tool B (results from both turns are in context).
    - The run completes with the expected structured output in state.
    """
    search_a_result = {"hits": ["alpha", "beta"]}
    search_b_result = {"hits": ["gamma"]}

    search_a = _mock_tool(search_a_result)
    search_b = _mock_tool(search_b_result)

    llm, captured = _make_loop_llm([
        # Iteration 1 — LLM requests the first MCP tool
        [{"name": "search_a", "args": {"query": "first query"}, "id": "tc-a-1"}],
        # Iteration 2 — LLM has search_a result, requests a second MCP tool
        [{"name": "search_b", "args": {"query": "second query"}, "id": "tc-b-1"}],
        # Iteration 3 — LLM has both results, submits final answer
        [{"name": _SUBMIT, "args": {"answer": "combined answer"}, "id": "tc-submit-1"}],
    ])

    client, mongo = await build_int_client(
        _GRAPH, llm, mcp_tools={"search_a": search_a, "search_b": search_b}
    )
    try:
        resp = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": _GRAPH_ID, "user_request": "gather info from both sources"},
        )
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["id"]

        # Both tools were called with the exact arguments the LLM requested
        search_a.ainvoke.assert_called_once_with({"query": "first query"})
        search_b.ainvoke.assert_called_once_with({"query": "second query"})

        # Three LLM iterations should have happened
        assert len(captured) == 3, f"expected 3 LLM iterations, got {len(captured)}"

        # Second call: search_a result is in context
        second_call_tool_msgs = [m for m in captured[1] if isinstance(m, ToolMessage)]
        assert len(second_call_tool_msgs) == 1
        assert str(search_a_result) in second_call_tool_msgs[0].content

        # Third call: both search_a and search_b results are in context
        third_call_tool_msgs = [m for m in captured[2] if isinstance(m, ToolMessage)]
        assert len(third_call_tool_msgs) == 2
        tool_contents = " ".join(m.content for m in third_call_tool_msgs)
        assert str(search_a_result) in tool_contents
        assert str(search_b_result) in tool_contents

        # Final structured output is stored in run state
        state = (await client.get(f"/api/v1/workflows/runs/{run_id}")).json()["intermediate_outputs"]
        assert state["answer"] == "combined answer"
    finally:
        await mongo.close()
