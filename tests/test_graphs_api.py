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


def _build_container(registry: YamlGraphRegistry) -> ApplicationContainer:
    settings = Settings()
    repo = AsyncMock(spec=MongoGraphRunRepository)
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.get = AsyncMock(
        return_value=GraphRun(id="tid1", graph_id="simple", user_request="hello", status="running")
    )
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


@pytest.fixture
async def client():
    registry = _build_registry()
    container = _build_container(registry)
    app = create_app()

    # Bypass lifespan — inject container directly
    app.state.container = container

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, container


@pytest.mark.asyncio
async def test_list_workflows(client):
    c, _ = client
    resp = await c.get("/api/v1/workflows")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(w["id"] == "simple" for w in data)
    assert all("name" in w and "description" in w and "steps" in w for w in data)


@pytest.mark.asyncio
async def test_start_run(client):
    c, container = client
    resp = await c.post(
        "/api/v1/workflows/runs",
        json={"workflow_id": "simple", "user_request": "hello"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow_id"] == "simple"
    assert data["user_request"] == "hello"
    assert "id" in data
    assert data["status"] in ("running", "waiting_approval", "completed")
    container.run_repository.create.assert_called_once()
    container.run_repository.update.assert_called()


@pytest.mark.asyncio
async def test_start_run_step_statuses_in_response(client):
    c, container = client
    resp = await c.post(
        "/api/v1/workflows/runs",
        json={"workflow_id": "simple", "user_request": "hello"},
    )
    assert resp.status_code == 200
    data = resp.json()
    steps = data["steps"]
    assert len(steps) == 1
    assert steps[0]["id"] == "step1"
    assert "status" in steps[0]

    # Background task should have updated the run with step_statuses
    update_calls = container.run_repository.update.call_args_list
    assert len(update_calls) >= 1
    last_run: GraphRun = update_calls[-1].args[0]
    assert last_run.step_statuses.get("step1") == "finished"


@pytest.mark.asyncio
async def test_start_run_unknown_workflow(client):
    c, _ = client
    resp = await c.post(
        "/api/v1/workflows/runs",
        json={"workflow_id": "nonexistent", "user_request": "hi"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_run(client):
    c, container = client
    resp = await c.get("/api/v1/workflows/runs/tid1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "tid1"
    assert data["workflow_id"] == "simple"
    assert data["user_request"] == "hello"


@pytest.mark.asyncio
async def test_get_run_not_found(client):
    c, container = client
    container.run_repository.get = AsyncMock(return_value=None)
    resp = await c.get("/api/v1/workflows/runs/missing")
    assert resp.status_code == 404
