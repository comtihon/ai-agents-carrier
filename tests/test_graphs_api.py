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


# ─── Filter tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_runs_no_filter(client):
    c, container = client
    container.run_repository.list_recent = AsyncMock(return_value=[])
    resp = await c.get("/api/v1/workflows/runs")
    assert resp.status_code == 200
    assert resp.json() == []
    container.run_repository.list_recent.assert_called_once_with(
        limit=50, workflow_id=None, status=None, search=None
    )


@pytest.mark.asyncio
async def test_list_runs_status_filter(client):
    c, container = client
    run = GraphRun(id="r1", graph_id="simple", user_request="build feature", status="running")
    container.run_repository.list_recent = AsyncMock(return_value=[run])
    resp = await c.get("/api/v1/workflows/runs?status=running")
    assert resp.status_code == 200
    container.run_repository.list_recent.assert_called_once_with(
        limit=50, workflow_id=None, status="running", search=None
    )
    data = resp.json()
    assert len(data) == 1
    assert data[0]["status"] == "running"


@pytest.mark.asyncio
async def test_list_runs_search_filter(client):
    c, container = client
    container.run_repository.list_recent = AsyncMock(return_value=[])
    resp = await c.get("/api/v1/workflows/runs?search=dark+mode")
    assert resp.status_code == 200
    container.run_repository.list_recent.assert_called_once_with(
        limit=50, workflow_id=None, status=None, search="dark mode"
    )


@pytest.mark.asyncio
async def test_list_runs_combined_filters(client):
    c, container = client
    run = GraphRun(id="r2", graph_id="simple", user_request="build feature", status="completed")
    container.run_repository.list_recent = AsyncMock(return_value=[run])
    resp = await c.get(
        "/api/v1/workflows/runs?workflow_id=simple&status=completed&search=build&limit=10"
    )
    assert resp.status_code == 200
    container.run_repository.list_recent.assert_called_once_with(
        limit=10, workflow_id="simple", status="completed", search="build"
    )
    assert len(resp.json()) == 1


# ─── Approve handler claims atomically ────────────────────────────────────────
# These tests pin the contract that the /approve handler relies on
# `claim_for_resume` (an atomic find_one_and_update) instead of the previous
# read-then-write on `run.status`. The integration test
# `test_concurrent_approve_serialised_no_double_resume` exercises the actual
# race against MongoDB; these unit tests just verify the handler wiring.

@pytest.mark.asyncio
async def test_approve_uses_claim_for_resume_not_read_then_write(client):
    """Happy-path approve must go through the atomic claim, not get/update."""
    c, container = client
    container.live_runners["tid1"] = list(container.yaml_graph_registry._runners.values())[0]
    waiting = GraphRun(
        id="tid1", graph_id="simple", user_request="hello",
        status="waiting_approval", current_step="step1",
    )
    container.run_repository.claim_for_resume = AsyncMock(return_value=waiting)
    container.run_repository.get = AsyncMock()  # must NOT be called for the gate check

    resp = await c.post("/api/v1/workflows/runs/tid1/approve")

    assert resp.status_code == 200, resp.text
    container.run_repository.claim_for_resume.assert_awaited_once_with("tid1")


@pytest.mark.asyncio
async def test_approve_returns_409_when_claim_loses_race(client):
    """When claim_for_resume returns None (someone else claimed first or the
    run moved past), the second click sees 409 — never schedules a duplicate
    resume task."""
    c, container = client
    container.run_repository.claim_for_resume = AsyncMock(return_value=None)
    # The handler still queries `get` to differentiate 404 vs 409.
    container.run_repository.get = AsyncMock(return_value=GraphRun(
        id="tid1", graph_id="simple", user_request="hello", status="running",
    ))

    resp = await c.post("/api/v1/workflows/runs/tid1/approve")

    assert resp.status_code == 409, resp.text
    assert "running" in resp.json().get("detail", "")
    container.run_repository.claim_for_resume.assert_awaited_once_with("tid1")


@pytest.mark.asyncio
async def test_approve_returns_404_when_run_missing(client):
    """Claim returns None and there is no run at all — 404 not 409."""
    c, container = client
    container.run_repository.claim_for_resume = AsyncMock(return_value=None)
    container.run_repository.get = AsyncMock(return_value=None)

    resp = await c.post("/api/v1/workflows/runs/tid1/approve")

    assert resp.status_code == 404, resp.text
