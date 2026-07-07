"""Tests for tool-gating sentinels on the agent progress callback endpoint.

The `POST /runs/{id}/agent/progress` endpoint interprets `__tool_start__:` /
`__tool_end__:` control messages to maintain `run.state["_active_tools"]`,
clears both active lists on `__mcp_clear__:`, and never appends any unknown
`__…__`-prefixed sentinel to the progress list.
"""
from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


def _build_container(run: GraphRun) -> ApplicationContainer:
    settings = Settings()
    repo = AsyncMock(spec=MongoGraphRunRepository)
    repo.get = AsyncMock(return_value=run)
    repo.update = AsyncMock()
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    openhands = MagicMock(spec=OpenHandsAdapter)
    return ApplicationContainer(
        settings=settings,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=openhands,
    )


def _make_run(state: dict | None = None) -> GraphRun:
    return GraphRun(
        id="tid1",
        graph_id="simple",
        user_request="hello",
        status="running",
        current_step="step1",
        state=state or {},
    )


async def _post(run: GraphRun, message: str):
    container = _build_container(run)
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.post(
            "/api/v1/runs/tid1/agent/progress", json={"message": message}
        )


@pytest.mark.asyncio
async def test_tool_start_adds():
    run = _make_run()
    resp = await _post(run, "__tool_start__:" + json.dumps({"tool": "github"}))
    assert resp.status_code == 202
    assert resp.json()["status"] == "tool_started"
    assert run.state["_active_tools"] == ["github"]


@pytest.mark.asyncio
async def test_duplicate_tool_start_allowed():
    run = _make_run({"_active_tools": ["github"]})
    await _post(run, "__tool_start__:" + json.dumps({"tool": "github"}))
    assert run.state["_active_tools"] == ["github", "github"]


@pytest.mark.asyncio
async def test_tool_end_removes_one():
    run = _make_run({"_active_tools": ["github", "github", "jira"]})
    resp = await _post(run, "__tool_end__:" + json.dumps({"tool": "github"}))
    assert resp.json()["status"] == "tool_ended"
    assert run.state["_active_tools"] == ["github", "jira"]


@pytest.mark.asyncio
async def test_tool_end_absent_noop():
    run = _make_run({"_active_tools": ["jira"]})
    resp = await _post(run, "__tool_end__:" + json.dumps({"tool": "github"}))
    assert resp.json()["status"] == "tool_ended"
    assert run.state["_active_tools"] == ["jira"]


@pytest.mark.asyncio
async def test_mcp_clear_empties_both_active_lists():
    run = _make_run(
        {"_active_mcp_servers": ["atlassian"], "_active_tools": ["github"]}
    )
    resp = await _post(run, "__mcp_clear__:" + json.dumps({}))
    assert resp.json()["status"] == "mcp_cleared"
    assert run.state["_active_mcp_servers"] == []
    assert run.state["_active_tools"] == []


@pytest.mark.asyncio
async def test_unknown_sentinel_not_appended():
    run = _make_run()
    resp = await _post(run, "__bogus__:x")
    assert resp.json()["status"] == "sentinel_ignored"
    assert "_agent_progress_step1" not in run.state
    assert "_agent_progress__unscoped" not in run.state


@pytest.mark.asyncio
async def test_plain_message_still_appended():
    run = _make_run()
    resp = await _post(run, "hello world")
    assert resp.json()["status"] == "progress_stored"
    assert run.state["_agent_progress_step1"] == ["hello world"]
