"""
Unit tests for the default CopilotKit workflow.

Covers:
- "develop a feature X" → LLM decides run_workflow → child GraphRun created for
  the matching workflow with the original request, background task scheduled.
- "2+2" → LLM decides reply → direct answer "4" returned, no workflow spawned.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.default_workflow import RouterDecision, build_default_workflow


# ── shared helpers ────────────────────────────────────────────────────────────

def _fake_runner(workflow_id: str = "develop-a-ticket", name: str = "Develop a Ticket") -> MagicMock:
    runner = MagicMock()
    runner.name = name
    runner.steps = [{"id": "analyze", "type": "llm"}, {"id": "plan", "type": "llm"}]
    return runner


def _fake_registry(workflow_ids: list[str] | None = None) -> MagicMock:
    """Registry mock pre-loaded with fake workflow definitions."""
    workflow_ids = workflow_ids or ["develop-a-ticket"]
    registry = MagicMock()
    registry.list_ids.return_value = workflow_ids
    registry.list_definitions.return_value = [
        {
            "id": wid,
            "name": wid.replace("-", " ").title(),
            "description": f"Workflow {wid}.",
            "steps": [{"id": "step1", "type": "llm"}],
        }
        for wid in workflow_ids
    ]
    registry.get.side_effect = lambda wid: _fake_runner(wid) if wid in workflow_ids else None
    return registry


def _fake_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    return repo


async def _astream_chunks(content: str):
    """Async generator that yields a single AIMessage chunk."""
    yield AIMessage(content=content)


def _llm_with_decision(decision: RouterDecision) -> MagicMock:
    """
    LLM mock whose with_structured_output(...).ainvoke(...) returns *decision*.
    The astream fallback (reply node when reply_text is empty) yields a single chunk.
    """
    structured = AsyncMock()
    structured.ainvoke = AsyncMock(return_value=decision)

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    llm.astream = MagicMock(side_effect=lambda msgs: _astream_chunks(decision.reply_text or ""))
    return llm


# CopilotKit base state required by DefaultWorkflowState
_BASE_STATE: dict = {"messages": [], "copilotkit": {"actions": [], "context": []}}

# Module path used for patching stream_graph_to_pause
_STREAM_FN = "app.infrastructure.orchestration.default_workflow.stream_graph_to_pause"


# ── test: workflow is spawned ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_develop_feature_spawns_child_workflow():
    """
    Input  : "develop a feature X"
    LLM    : RouterDecision(action="run_workflow", workflow_id="develop-a-ticket",
                            workflow_request="develop a feature X")
    Expect : GraphRun created for "develop-a-ticket" with that request,
             stream_graph_to_pause called (background task), confirmation AIMessage.
    """
    decision = RouterDecision(
        action="run_workflow",
        workflow_id="develop-a-ticket",
        workflow_request="develop a feature X",
    )
    llm = _llm_with_decision(decision)
    registry = _fake_registry(["develop-a-ticket"])
    repo = _fake_repo()

    graph = build_default_workflow(llm, registry, repo)

    # Patch stream_graph_to_pause so the background task is harmless.
    # Calling the AsyncMock records the call immediately (before create_task runs it).
    with patch(_STREAM_FN, new_callable=AsyncMock) as mock_stream:
        result = await graph.ainvoke(
            {**_BASE_STATE, "messages": [HumanMessage(content="develop a feature X")]},
            {"configurable": {"thread_id": "test-spawn-thread"}},
        )
        # Let event loop drain so the background task completes cleanly
        await asyncio.sleep(0)

    # ── registry lookup ───────────────────────────────────────────────────────
    registry.get.assert_called_with("develop-a-ticket")

    # ── child run persisted ───────────────────────────────────────────────────
    repo.create.assert_awaited_once()
    created_run: GraphRun = repo.create.call_args[0][0]

    assert created_run.graph_id == "develop-a-ticket"
    assert created_run.user_request == "develop a feature X"
    assert created_run.status == "running"
    assert set(created_run.step_statuses.values()) == {"pending"}

    # ── background streaming task scheduled ──────────────────────────────────
    mock_stream.assert_called_once()
    stream_args = mock_stream.call_args[0]
    assert stream_args[1] is created_run          # run object passed through
    assert stream_args[2] is repo                 # repository passed through
    assert stream_args[3] == {"request": "develop a feature X"}

    # ── confirmation message returned ─────────────────────────────────────────
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs, "Expected at least one AIMessage in result"
    # Message must reference the run ID so the user can track it in the panel
    assert created_run.id in ai_msgs[-1].content


# ── test: direct reply, no workflow ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_arithmetic_question_returns_direct_reply():
    """
    Input  : "2+2"
    LLM    : RouterDecision(action="reply", reply_text="4")
    Expect : AIMessage("4") in output, no GraphRun created, no task scheduled.
    """
    decision = RouterDecision(action="reply", reply_text="4")
    llm = _llm_with_decision(decision)
    registry = _fake_registry(["develop-a-ticket"])
    repo = _fake_repo()

    graph = build_default_workflow(llm, registry, repo)

    with patch(_STREAM_FN, new_callable=AsyncMock) as mock_stream:
        result = await graph.ainvoke(
            {**_BASE_STATE, "messages": [HumanMessage(content="2+2")]},
            {"configurable": {"thread_id": "test-reply-thread"}},
        )
        await asyncio.sleep(0)

    # ── no workflow spawned ───────────────────────────────────────────────────
    repo.create.assert_not_awaited()
    mock_stream.assert_not_called()

    # ── direct reply returned ─────────────────────────────────────────────────
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs, "Expected at least one AIMessage in result"
    assert ai_msgs[-1].content == "4"
