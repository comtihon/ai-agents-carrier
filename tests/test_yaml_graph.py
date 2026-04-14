from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock

from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.tools.mcp_client import McpToolsProvider


def _make_runner(steps: list[dict], extra_responses: list | None = None) -> YamlGraphRunner:
    responses = extra_responses or [AIMessage(content="test output")]
    llm = FakeMessagesListChatModel(responses=responses)
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    definition = {"id": "test-graph", "steps": steps}
    return YamlGraphRunner(definition, llm=llm, mcp_tools_provider=mcp)


@pytest.mark.asyncio
async def test_llm_node_writes_output_key():
    runner = _make_runner([
        {"id": "step1", "type": "llm", "output_key": "result"},
    ])
    config = {"configurable": {"thread_id": "t1"}}
    state = await runner.graph.ainvoke({"request": "hello"}, config)
    assert state["result"] == "test output"


@pytest.mark.asyncio
async def test_llm_node_when_skips_if_falsy():
    runner = _make_runner([
        {"id": "step1", "type": "llm", "output_key": "result", "when": "missing_key"},
    ])
    config = {"configurable": {"thread_id": "t2"}}
    state = await runner.graph.ainvoke({"request": "hello"}, config)
    # node skipped — output_key not set
    assert "result" not in state


@pytest.mark.asyncio
async def test_llm_node_when_runs_if_truthy():
    runner = _make_runner([
        {"id": "step1", "type": "llm", "output_key": "result", "when": "flag"},
    ])
    config = {"configurable": {"thread_id": "t3"}}
    # inject flag=True into initial state via a passthrough step
    runner2 = _make_runner([
        {"id": "set_flag", "type": "llm", "output_key": "flag"},
        {"id": "step1", "type": "llm", "output_key": "result", "when": "flag"},
    ], extra_responses=[AIMessage(content="true"), AIMessage(content="output")])
    state = await runner2.graph.ainvoke({"request": "hello"}, {"configurable": {"thread_id": "t3b"}})
    assert state["flag"] == "true"
    assert state["result"] == "output"


@pytest.mark.asyncio
async def test_mcp_node_missing_tool_returns_message():
    runner = _make_runner([
        {"id": "fetch", "type": "mcp", "tool": "nonexistent", "output_key": "data", "tool_input": {"query": "{request}"}},
    ])
    config = {"configurable": {"thread_id": "t4"}}
    state = await runner.graph.ainvoke({"request": "find bugs"}, config)
    assert "not available" in state["data"]


@pytest.mark.asyncio
async def test_mcp_node_calls_tool():
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="x")])
    mcp = MagicMock(spec=McpToolsProvider)
    fake_tool = AsyncMock()
    fake_tool.ainvoke = AsyncMock(return_value="jira results")
    mcp.get_tool = MagicMock(return_value=fake_tool)
    definition = {"id": "g", "steps": [
        {"id": "fetch", "type": "mcp", "tool": "jira_search", "output_key": "jira_data",
         "tool_input": {"query": "{request}"}},
    ]}
    runner = YamlGraphRunner(definition, llm=llm, mcp_tools_provider=mcp)
    state = await runner.graph.ainvoke({"request": "sprint issues"}, {"configurable": {"thread_id": "t5"}})
    assert state["jira_data"] == "jira results"
    fake_tool.ainvoke.assert_called_once_with({"query": "sprint issues"})


@pytest.mark.asyncio
async def test_human_approval_interrupts():
    runner = _make_runner([
        {"id": "plan", "type": "llm", "output_key": "plan"},
        {"id": "approve", "type": "human_approval"},
    ])
    config = {"configurable": {"thread_id": "t6"}}
    # First invocation should interrupt
    await runner.graph.ainvoke({"request": "do something"}, config)
    snap = runner.graph.get_state(config)
    assert snap.next  # interrupted, waiting for approval


@pytest.mark.asyncio
async def test_execute_node_without_openhands():
    runner = _make_runner([
        {"id": "impl", "type": "execute", "output_key": "result",
         "repo_template": "org/repo", "instructions_template": "do it"},
    ])
    config = {"configurable": {"thread_id": "t7"}}
    state = await runner.graph.ainvoke({"request": "implement"}, config)
    assert state["result"] == "OpenHands not configured"


@pytest.mark.asyncio
async def test_execute_node_calls_openhands():
    from app.infrastructure.integrations.openhands import OpenHandsAdapter
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="x")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    openhands = AsyncMock(spec=OpenHandsAdapter)
    openhands.execute = AsyncMock(return_value={"status": "success", "branch": "feature/x"})
    definition = {"id": "g2", "steps": [
        {"id": "impl", "type": "execute", "output_key": "result",
         "repo_template": "org/repo", "instructions_template": "{request}"},
    ]}
    runner = YamlGraphRunner(definition, llm=llm, mcp_tools_provider=mcp, openhands=openhands)
    state = await runner.graph.ainvoke({"request": "build feature"}, {"configurable": {"thread_id": "t8"}})
    assert state["result"]["status"] == "success"
    openhands.execute.assert_called_once_with(repo="org/repo", instructions="build feature")


@pytest.mark.asyncio
async def test_unknown_step_type_raises():
    with pytest.raises(ValueError, match="Unknown step type"):
        _make_runner([{"id": "bad", "type": "nonsense"}])


def test_render_template():
    runner = _make_runner([{"id": "s", "type": "llm", "output_key": "r"}])
    state = {"request": "hello", "plan": "do x"}
    assert runner._render("Request: {request}, Plan: {plan}", state) == "Request: hello, Plan: do x"


def test_render_missing_key_renders_empty():
    runner = _make_runner([{"id": "s", "type": "llm", "output_key": "r"}])
    result = runner._render("Value: {missing}", {})
    assert result == "Value: "
