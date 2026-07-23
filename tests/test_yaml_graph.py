from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command
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


# ── human_approval as a conditional goalkeeper (routes + END) ─────────────────

def _approval_gate_steps() -> list[dict]:
    """A minimal plan -> approval-gate -> downstream graph where the gate uses the
    new approve->next / reject->END conditional routes shape."""
    return [
        {"id": "plan", "type": "llm", "output_key": "plan"},
        {"id": "gate", "type": "human_approval", "output_key": "approved",
         "routes": [{"when": "approved", "next": "after"}, {"next": "END"}]},
        {"id": "after", "type": "llm", "output_key": "done"},
    ]


def test_human_approval_routes_compile():
    # Two routes on a human_approval step must compile (proves human_approval is
    # now in _MULTI_OUTPUT_TYPES).
    runner = _make_runner(_approval_gate_steps())
    assert runner.graph is not None


def test_plain_step_two_routes_still_raises():
    # Regression: a step type NOT in _MULTI_OUTPUT_TYPES still cannot carry >1 route.
    with pytest.raises(ValueError, match="cannot have more than"):
        _make_runner([
            {"id": "s", "type": "llm", "output_key": "r",
             "routes": [{"when": "r", "next": "a"}, {"next": "END"}]},
            {"id": "a", "type": "llm", "output_key": "x"},
        ])


@pytest.mark.asyncio
async def test_approval_approve_runs_downstream():
    runner = _make_runner(
        _approval_gate_steps(),
        extra_responses=[AIMessage(content="the plan"), AIMessage(content="the result")],
    )
    config = {"configurable": {"thread_id": "appr-approve"}}
    await runner.graph.ainvoke({"request": "do it"}, config)  # pauses at gate
    decision = {
        "approved": True,
        "reason": None,
        "approver_name": "Alice",
        "approver_id": "u-1",
        "approver_source": "ui",
        "decided_at": "2026-07-23T00:00:00+00:00",
    }
    state = await runner.graph.ainvoke(Command(resume=decision), config)
    # Downstream node executed.
    assert state.get("done") == "the result"
    # Audit record captured the decision identity.
    history = state["approval_history"]
    assert len(history) == 1
    rec = history[0]
    assert rec["step_id"] == "gate"
    assert rec["approved"] is True
    assert rec["approver_name"] == "Alice"
    assert rec["approver_id"] == "u-1"
    assert rec["approver_source"] == "ui"
    assert rec["decided_at"] == "2026-07-23T00:00:00+00:00"


@pytest.mark.asyncio
async def test_approval_reject_ends_graph():
    runner = _make_runner(
        _approval_gate_steps(),
        extra_responses=[AIMessage(content="the plan"), AIMessage(content="the result")],
    )
    config = {"configurable": {"thread_id": "appr-reject"}}
    await runner.graph.ainvoke({"request": "do it"}, config)  # pauses at gate
    decision = {
        "approved": False,
        "reason": "not good enough",
        "approver_name": "Bob",
        "approver_id": "u-2",
        "approver_source": "ui",
        "decided_at": "2026-07-23T00:00:00+00:00",
    }
    state = await runner.graph.ainvoke(Command(resume=decision), config)
    # Downstream node did NOT execute — graph ended at the gate.
    assert "done" not in state
    assert state["approved"] is False
    rec = state["approval_history"][0]
    assert rec["approved"] is False
    assert rec["reason"] == "not good enough"
    assert rec["approver_source"] == "ui"


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
    call_kwargs = openhands.execute.call_args.kwargs
    assert call_kwargs["repo"] == "org/repo"
    assert call_kwargs["instructions"] == "build feature"
    assert call_kwargs["existing_conv_id"] is None
    assert callable(call_kwargs["conv_id_callback"])


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
