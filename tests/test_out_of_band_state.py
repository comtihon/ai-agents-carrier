from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.yaml_graph import (
    YamlGraphRunner,
    merge_out_of_band_state,
    stream_graph_to_pause,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider


class _InMemoryRepo:
    """Minimal run repository backed by a dict — get() returns the same object
    that update()/create() last stored, so out-of-band writes survive."""

    def __init__(self):
        self.docs: dict[str, GraphRun] = {}

    async def create(self, run: GraphRun) -> None:
        self.docs[run.id] = run

    async def update(self, run: GraphRun) -> None:
        self.docs[run.id] = run

    async def get(self, run_id: str):
        return self.docs.get(run_id)


def _build_runner() -> YamlGraphRunner:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])
    return YamlGraphRunner(
        {"id": "simple", "steps": [{"id": "step1", "type": "llm", "output_key": "answer"}]},
        llm=llm,
        mcp_tools_provider=mcp,
    )


# ---------------------------------------------------------------------------
# merge_out_of_band_state unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_fresh_out_of_band_wins():
    repo = _InMemoryRepo()
    repo.docs["r1"] = GraphRun(
        id="r1", graph_id="g", status="running",
        state={"_agent_progress_a": ["msg1", "msg2"]},
    )
    merged = await merge_out_of_band_state(repo, "r1", {"answer": "x"})
    assert merged["_agent_progress_a"] == ["msg1", "msg2"]
    assert merged["answer"] == "x"


@pytest.mark.asyncio
async def test_merge_ignores_non_prefixed_and_falsy():
    repo = _InMemoryRepo()
    repo.docs["r1"] = GraphRun(
        id="r1", graph_id="g", status="running",
        state={"other_key": "should_not_leak", "_agent_progress_a": []},
    )
    merged = await merge_out_of_band_state(repo, "r1", {"answer": "x"})
    assert "other_key" not in merged
    # Falsy out-of-band value must not override / inject.
    assert "_agent_progress_a" not in merged
    assert merged == {"answer": "x"}


@pytest.mark.asyncio
async def test_merge_returns_unchanged_when_repo_raises():
    repo = MagicMock()

    async def _boom(_):
        raise RuntimeError("mongo down")

    repo.get = _boom
    merged = await merge_out_of_band_state(repo, "r1", {"answer": "x"})
    assert merged == {"answer": "x"}


@pytest.mark.asyncio
async def test_merge_returns_unchanged_when_repo_none():
    merged = await merge_out_of_band_state(None, "r1", {"answer": "x"})
    assert merged == {"answer": "x"}


# ---------------------------------------------------------------------------
# End-to-end: an out-of-band progress trail survives a run to completion.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_survives_stream_to_completion():
    runner = _build_runner()
    repo = _InMemoryRepo()
    run = GraphRun(id="r1", graph_id="simple", user_request="hi", status="running")
    # Simulate an out-of-band write (POST /agent/progress) that never flowed
    # through the LangGraph checkpoint.
    run.state = {"_agent_progress_step1": ["msg1", "msg2"]}
    await repo.create(run)
    run.step_statuses = {s["id"]: "pending" for s in runner.steps}

    await stream_graph_to_pause(runner, run, repo, {"request": "hi"})

    assert run.status == "completed"
    assert run.state.get("_agent_progress_step1") == ["msg1", "msg2"]
