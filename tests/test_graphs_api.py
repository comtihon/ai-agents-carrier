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


class _CapturingLiveRunners(dict):
    """dict subclass that remembers the last inserted runner even after it
    gets popped — needed because the graph in these tests runs to completion
    synchronously within the request/response cycle (background tasks run
    before Starlette returns the response), so `live_runners` is already
    empty by the time `await client.post(...)` resolves."""

    def __setitem__(self, key, value):
        self.last_inserted = value
        super().__setitem__(key, value)


def _build_container_with_backend(registry: YamlGraphRegistry) -> ApplicationContainer:
    """Container variant with workflow_backend + the 3 repos wired, so that
    start_run() takes the `workflow_backend is not None` branch and we can
    verify dependency injection onto the freshly built runner."""
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

    simple_defn = MagicMock()
    simple_defn.id = "simple"
    simple_defn.to_raw_dict = MagicMock(return_value={"id": "simple", "steps": [{"id": "step1", "type": "llm", "output_key": "answer"}]})
    workflow_backend = MagicMock()
    workflow_backend.get = AsyncMock(return_value=simple_defn)

    warm_pod_repository = MagicMock()
    pvc_lease_repository = MagicMock()
    agent_task_repository = MagicMock()

    return ApplicationContainer(
        settings=settings,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=registry,
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=openhands,
        workflow_backend=workflow_backend,
        warm_pod_repository=warm_pod_repository,
        pvc_lease_repository=pvc_lease_repository,
        agent_task_repository=agent_task_repository,
    )


@pytest.mark.asyncio
async def test_start_run_injects_runner_dependencies():
    """start_run() must inject pvc_lease/agent_task/warm_pod repositories onto
    the runner it builds — regression test for the missing
    `_inject_runner_dependencies` call in the POST /workflows/runs handler."""
    registry = _build_registry()
    container = _build_container_with_backend(registry)
    container.live_runners = _CapturingLiveRunners()
    app = create_app()
    app.state.container = container

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "simple", "user_request": "hello"},
        )
        assert resp.status_code == 200, resp.text

    runner = container.live_runners.last_inserted

    assert runner._warm_pod_repository is container.warm_pod_repository
    assert runner._pvc_lease_repository is container.pvc_lease_repository
    assert runner._agent_task_repository is container.agent_task_repository


@pytest.mark.asyncio
async def test_start_run_does_not_force_inject_none_dependencies(client):
    """When the container has no repos configured (legacy/test setups), the
    runner's dependency attrs must remain None rather than being overwritten
    with something truthy."""
    c, container = client
    container.live_runners = _CapturingLiveRunners()
    resp = await c.post(
        "/api/v1/workflows/runs",
        json={"workflow_id": "simple", "user_request": "hello"},
    )
    assert resp.status_code == 200, resp.text
    runner = container.live_runners.last_inserted

    assert runner._warm_pod_repository is None
    assert runner._pvc_lease_repository is None
    assert runner._agent_task_repository is None


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
    container.run_repository.count_recent = AsyncMock(return_value=0)
    resp = await c.get("/api/v1/workflows/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["runs"] == []
    assert data["total"] == 0
    container.run_repository.list_recent.assert_called_once_with(
        limit=50, offset=0, workflow_id=None, status=None, search=None, exclude_workflow_ids=None
    )


@pytest.mark.asyncio
async def test_list_runs_status_filter(client):
    c, container = client
    run = GraphRun(id="r1", graph_id="simple", user_request="build feature", status="running")
    container.run_repository.list_recent = AsyncMock(return_value=[run])
    container.run_repository.count_recent = AsyncMock(return_value=1)
    resp = await c.get("/api/v1/workflows/runs?status=running")
    assert resp.status_code == 200
    container.run_repository.list_recent.assert_called_once_with(
        limit=50, offset=0, workflow_id=None, status="running", search=None, exclude_workflow_ids=None
    )
    data = resp.json()
    assert data["total"] == 1
    assert len(data["runs"]) == 1
    assert data["runs"][0]["status"] == "running"


@pytest.mark.asyncio
async def test_list_runs_search_filter(client):
    c, container = client
    container.run_repository.list_recent = AsyncMock(return_value=[])
    container.run_repository.count_recent = AsyncMock(return_value=0)
    resp = await c.get("/api/v1/workflows/runs?search=dark+mode")
    assert resp.status_code == 200
    data = resp.json()
    assert data["runs"] == []
    assert data["total"] == 0
    container.run_repository.list_recent.assert_called_once_with(
        limit=50, offset=0, workflow_id=None, status=None, search="dark mode", exclude_workflow_ids=None
    )


@pytest.mark.asyncio
async def test_list_runs_combined_filters(client):
    c, container = client
    run = GraphRun(id="r2", graph_id="simple", user_request="build feature", status="completed")
    container.run_repository.list_recent = AsyncMock(return_value=[run])
    container.run_repository.count_recent = AsyncMock(return_value=1)
    resp = await c.get(
        "/api/v1/workflows/runs?workflow_id=simple&status=completed&search=build&limit=10"
    )
    assert resp.status_code == 200
    container.run_repository.list_recent.assert_called_once_with(
        limit=10, offset=0, workflow_id="simple", status="completed", search="build", exclude_workflow_ids=None
    )
    data = resp.json()
    assert data["total"] == 1
    assert len(data["runs"]) == 1


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


@pytest.mark.asyncio
async def test_stream_graph_rebinds_runner_to_current_run():
    """
    `workflows.py::_stream_graph` must rebind `runner._current_run` (and
    `_current_run_repository`) to the run it was invoked with, *before*
    streaming begins.

    Why: `_wrap_with_status_running` and `_save_conv_id` write step
    statuses / OpenHands conversation IDs through `runner._current_run`
    and persist via `runner._current_run_repository`. After the
    post-restart recovery path (`stream_graph_to_pause`) those refs are
    left pointing at the *recovery* run object. A subsequent /approve
    that re-uses the cached runner would then have its node-body helper
    saves write the stale recovery state on top of the live run —
    observably flipping `status` back to `waiting_approval` mid-stream
    and showing the user the same approval form again right after they
    submitted.

    This test pins the rebinding so a future refactor can't quietly
    re-introduce that drift.
    """
    from unittest.mock import AsyncMock as AM, MagicMock as MM

    from app.api.routes.workflows import _stream_graph
    from app.domain.models.graph_run import GraphRun

    async def empty_stream(*_a, **_k):
        if False:
            yield  # marks this as an async generator

    snap = MM()
    snap.next = ()
    snap.values = {}

    runner = MM()
    runner.graph = MM()
    runner.graph.astream = empty_stream
    runner.graph.aget_state = AM(return_value=snap)
    runner.steps = []

    # Simulate the stale binding left behind by a prior recovery.
    stale_run = GraphRun(id="stale", graph_id="g", user_request="x", status="completed")
    stale_repo = object()
    runner._current_run = stale_run
    runner._current_run_repository = stale_repo

    fresh_run = GraphRun(id="fresh", graph_id="g", user_request="y", status="running")
    container = MM()
    container.run_repository = AM()
    container.settings = MM(base_url=None)
    container.live_runners = {}

    await _stream_graph(runner, fresh_run, container, {"request": "y"})

    assert runner._current_run is fresh_run, (
        "stream_graph must rebind runner._current_run to the live run "
        f"(still pointing at {runner._current_run!r})"
    )
    assert runner._current_run_repository is container.run_repository
