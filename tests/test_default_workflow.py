"""
Unit tests for the default ReAct chat agent.

Covers:
- "develop a feature X" → LLM calls run_workflow tool → GraphRun created, background task spawned.
- "2+2" → LLM replies directly, no tool calls, no workflow spawned.
- "do the thing" (ambiguous) → LLM calls ask_user → graph pauses at interrupt,
  resumes with answers, LLM produces final reply.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.default_workflow import build_default_workflow


# ── helpers ───────────────────────────────────────────────────────────────────

def _tool_call_msg(tool_name: str, args: dict, call_id: str = "tc_1") -> AIMessage:
    """AIMessage carrying a single tool call — triggers the tools node."""
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _fake_registry(workflow_ids: list[str] | None = None) -> MagicMock:
    workflow_ids = workflow_ids or ["develop-a-ticket"]
    runner = MagicMock()
    runner.name = "Develop a Ticket"
    runner.steps = [{"id": "analyze", "type": "llm"}, {"id": "plan", "type": "llm"}]

    registry = MagicMock()
    registry.list_ids.return_value = workflow_ids
    registry.list_definitions.return_value = [
        {"id": wid, "name": wid.replace("-", " ").title(), "description": f"Workflow {wid}.", "steps": []}
        for wid in workflow_ids
    ]
    registry.get.side_effect = lambda wid: runner if wid in workflow_ids else None
    return registry


def _fake_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.list = AsyncMock(return_value=[])
    repo.get = AsyncMock(return_value=None)
    return repo


def _llm_with_responses(responses: list) -> MagicMock:
    """LLM whose bind_tools(…).ainvoke(…) returns *responses* in sequence."""
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    return llm


_BASE_STATE: dict = {"messages": [], "copilotkit": {"actions": [], "context": []}}
_STREAM_FN = "app.infrastructure.orchestration.default_workflow.stream_graph_to_pause"


# ── test: run_workflow tool spawns child workflow ─────────────────────────────

@pytest.mark.asyncio
async def test_run_workflow_tool_spawns_child_workflow():
    """
    Input  : "develop feature X"
    LLM 1  : calls run_workflow(workflow_id="develop-a-ticket", request="develop feature X")
    LLM 2  : confirmation reply after tool result
    Expect : GraphRun created with correct fields, stream_graph_to_pause called.
    """
    llm = _llm_with_responses([
        _tool_call_msg("run_workflow", {
            "workflow_id": "develop-a-ticket",
            "request": "develop feature X",
        }),
        AIMessage(content="Started the workflow."),
    ])
    registry = _fake_registry(["develop-a-ticket"])
    repo = _fake_repo()

    graph = build_default_workflow(llm, registry, repo)

    with patch(_STREAM_FN, new_callable=AsyncMock) as mock_stream:
        result = await graph.ainvoke(
            {**_BASE_STATE, "messages": [HumanMessage(content="develop feature X")]},
            {"configurable": {"thread_id": "test-spawn"}},
        )
        await asyncio.sleep(0)

    # child run persisted
    repo.create.assert_awaited_once()
    created_run: GraphRun = repo.create.call_args[0][0]
    assert created_run.graph_id == "develop-a-ticket"
    assert created_run.user_request == "develop feature X"
    assert created_run.status == "running"
    assert set(created_run.step_statuses.values()) == {"pending"}

    # background streaming scheduled
    mock_stream.assert_called_once()
    stream_args = mock_stream.call_args[0]
    assert stream_args[1] is created_run
    assert stream_args[2] is repo
    assert stream_args[3] == {"request": "develop feature X"}

    # final AIMessage present
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    assert ai_msgs, "Expected at least one final AIMessage"


# ── test: direct reply without tool calls ────────────────────────────────────

@pytest.mark.asyncio
async def test_arithmetic_question_returns_direct_reply():
    """
    Input  : "2+2"
    LLM    : AIMessage(content="4") — no tool calls
    Expect : "4" returned, no GraphRun created, no task scheduled.
    """
    llm = _llm_with_responses([AIMessage(content="4")])
    registry = _fake_registry(["develop-a-ticket"])
    repo = _fake_repo()

    graph = build_default_workflow(llm, registry, repo)

    with patch(_STREAM_FN, new_callable=AsyncMock) as mock_stream:
        result = await graph.ainvoke(
            {**_BASE_STATE, "messages": [HumanMessage(content="2+2")]},
            {"configurable": {"thread_id": "test-reply"}},
        )
        await asyncio.sleep(0)

    repo.create.assert_not_awaited()
    mock_stream.assert_not_called()

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs, "Expected at least one AIMessage"
    assert ai_msgs[-1].content == "4"


# ── test: ask_user tool pauses and resumes ────────────────────────────────────

@pytest.mark.asyncio
async def test_ask_context_pauses_and_resumes():
    """
    Input  : "do the thing" (ambiguous)
    LLM 1  : calls ask_user(questions=["Which thing?"])  → interrupt fires
    Resume : answers={"0": "deploy the app"}
    LLM 2  : AIMessage("Got it!")
    Expect : graph pauses with ask_context interrupt, resumes to produce reply.
    """
    from langgraph.types import Command

    llm = _llm_with_responses([
        _tool_call_msg("ask_user", {"questions": ["Which thing?"]}, call_id="tc_ask"),
        AIMessage(content="Got it!"),
    ])
    registry = _fake_registry(["develop-a-ticket"])
    repo = _fake_repo()

    graph = build_default_workflow(llm, registry, repo)
    config = {"configurable": {"thread_id": "test-ask"}}

    with patch(_STREAM_FN, new_callable=AsyncMock):
        await graph.ainvoke(
            {**_BASE_STATE, "messages": [HumanMessage(content="do the thing")]},
            config,
        )

    # graph should have paused at ask_context interrupt
    snap = graph.get_state(config)
    interrupt_vals = [
        intr.value
        for task in snap.tasks
        for intr in getattr(task, "interrupts", [])
    ]
    assert any(
        isinstance(v, dict) and v.get("type") == "ask_context"
        for v in interrupt_vals
    ), f"Expected ask_context interrupt, got: {interrupt_vals}"

    # resume with user answers
    with patch(_STREAM_FN, new_callable=AsyncMock):
        result2 = await graph.ainvoke(Command(resume={"0": "deploy the app"}), config)
        await asyncio.sleep(0)

    ai_msgs = [m for m in result2["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    assert ai_msgs, "Expected AIMessage after resume"
    assert ai_msgs[-1].content == "Got it!"
