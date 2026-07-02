"""Regression test: GET /runs/{id}/trace must scope agent_progress to the
run's current_step only — never leak progress recorded for other steps.

`_agent_progress_<step_id>` keys accumulate in `run.state` across the whole
run (each step's progress is written under its own key), so the trace
endpoint must select only the key matching `run.current_step`.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


def _build_registry() -> YamlGraphRegistry:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner(
        {"id": "simple", "steps": [{"id": "step1", "type": "llm", "output_key": "answer"}]},
        llm=llm,
        mcp_tools_provider=mcp,
    )
    return YamlGraphRegistry({"simple": runner})


def _build_container(registry: YamlGraphRegistry, run: GraphRun) -> ApplicationContainer:
    settings = Settings()
    repo = AsyncMock(spec=MongoGraphRunRepository)
    repo.get = AsyncMock(return_value=run)
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.start = AsyncMock()
    mcp.stop = AsyncMock()
    mcp.get_tool = MagicMock(return_value=None)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    openhands = MagicMock(spec=OpenHandsAdapter)
    return ApplicationContainer(
        settings=settings,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=registry,
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=openhands,
    )


@pytest.mark.asyncio
async def test_run_trace_scopes_agent_progress_to_current_step():
    run = GraphRun(
        id="tid1",
        graph_id="simple",
        user_request="hello",
        status="running",
        current_step="step_b",
        state={
            "_agent_progress_step_b": ["scoped"],
            "_agent_progress_step_a": ["other"],
        },
    )
    registry = _build_registry()
    container = _build_container(registry, run)
    app = create_app()
    app.state.container = container

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows/runs/tid1/trace")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["agent_progress"] == ["scoped"], (
        f"agent_progress must be scoped to current_step ('step_b') only, got {data['agent_progress']!r}"
    )
