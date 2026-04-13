"""
Unit tests for the /api/v1/workflows/* endpoints.

Covers:
- GET  /api/v1/workflows               → list workflow definitions
- POST /api/v1/workflows/runs          → start a run
- GET  /api/v1/workflows/runs/{id}     → fetch run by ID
- POST /api/v1/workflows/runs/{id}/approve
- POST /api/v1/workflows/runs/{id}/reject
- POST /api/v1/workflows/runs/{id}/gates/{gate_id}/approve
- POST /api/v1/workflows/runs/{id}/gates/{gate_id}/reject
- 404 responses for unknown workflow / run
- Step-type mapping (llm_structured→llm, mcp→fetch, human_approval→approval)
- YamlGraphRunner.name derivation (explicit > title-cased id)
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm(*responses: str) -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content=r) for r in responses])


def _make_runner(definition: dict, llm=None) -> YamlGraphRunner:
    if llm is None:
        llm = _make_llm("ok")
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    return YamlGraphRunner(definition, llm=llm, mcp_tools_provider=mcp)


def _make_container(registry: YamlGraphRegistry, stored_run: GraphRun | None = None) -> ApplicationContainer:
    settings = Settings()
    repo = AsyncMock(spec=MongoGraphRunRepository)
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.get = AsyncMock(return_value=stored_run)
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.start = AsyncMock()
    mcp.stop = AsyncMock()
    mcp.get_tool = MagicMock(return_value=None)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    return ApplicationContainer(
        settings=settings,
        llm=_make_llm("x"),
        mcp_tools_provider=mcp,
        yaml_graph_registry=registry,
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=MagicMock(spec=OpenHandsAdapter),
    )


def _simple_runner() -> YamlGraphRunner:
    return _make_runner({
        "id": "simple",
        "name": "Simple Workflow",
        "description": "A simple test workflow.",
        "steps": [{"id": "step1", "type": "llm", "output_key": "answer"}],
    })


def _approval_runner(llm=None) -> YamlGraphRunner:
    return _make_runner(
        {
            "id": "approvable",
            "name": "Approvable Workflow",
            "description": "Pauses for approval.",
            "steps": [
                {"id": "plan", "type": "llm", "output_key": "plan"},
                {"id": "approve", "type": "human_approval"},
                {
                    "id": "implement",
                    "type": "llm",
                    "when": "approved",
                    "output_key": "implementation",
                },
            ],
        },
        llm=llm,
    )


@pytest.fixture
def simple_client():
    runner = _simple_runner()
    registry = YamlGraphRegistry({"simple": runner})
    container = _make_container(registry)
    app = create_app()
    app.state.container = container
    return app, container


@pytest.fixture
def approval_client():
    runner = _approval_runner(llm=_make_llm("the plan", "the impl"))
    registry = YamlGraphRegistry({"approvable": runner})
    container = _make_container(registry)
    app = create_app()
    app.state.container = container
    return app, container


# ── GET /api/v1/workflows ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_workflows_returns_definitions(simple_client):
    app, _ = simple_client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    wf = data[0]
    assert wf["id"] == "simple"
    assert wf["name"] == "Simple Workflow"
    assert wf["description"] == "A simple test workflow."
    assert wf["steps"][0]["id"] == "step1"
    assert wf["steps"][0]["type"] == "llm"


@pytest.mark.asyncio
async def test_list_workflows_step_type_mapping():
    """llm_structured→llm, mcp→fetch, human_approval→approval, execute→execute."""
    runner = _make_runner({
        "id": "typed",
        "steps": [
            {"id": "s1", "type": "llm_structured", "output": [{"name": "flag", "type": "bool"}]},
            {"id": "s2", "type": "mcp", "tool": "search", "output_key": "ctx", "tool_input": {}},
            {"id": "s3", "type": "human_approval"},
            {"id": "s4", "type": "execute", "output_key": "res"},
            {"id": "s5", "type": "llm", "output_key": "ans"},
        ],
    })
    registry = YamlGraphRegistry({"typed": runner})
    container = _make_container(registry)
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows")
    assert resp.status_code == 200
    types = [s["type"] for s in resp.json()[0]["steps"]]
    assert types == ["llm", "fetch", "approval", "execute", "llm"]


@pytest.mark.asyncio
async def test_list_workflows_name_fallback():
    """When no name is in the YAML, it's title-cased from the id."""
    runner = _make_runner({
        "id": "my-cool-workflow",
        "steps": [{"id": "s", "type": "llm", "output_key": "r"}],
    })
    registry = YamlGraphRegistry({"my-cool-workflow": runner})
    container = _make_container(registry)
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows")
    assert resp.json()[0]["name"] == "My Cool Workflow"


@pytest.mark.asyncio
async def test_list_workflows_empty():
    registry = YamlGraphRegistry({})
    container = _make_container(registry)
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows")
    assert resp.status_code == 200
    assert resp.json() == []


# ── POST /api/v1/workflows/runs ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_submit_run_returns_workflow_run_shape(simple_client):
    app, container = simple_client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "simple", "user_request": "do something"},
        )
    assert resp.status_code == 200
    data = resp.json()
    # WorkflowRun shape
    assert "id" in data
    assert data["workflow_id"] == "simple"
    assert data["workflow_name"] == "Simple Workflow"
    assert data["user_request"] == "do something"
    assert data["status"] in ("running", "waiting_approval", "completed", "failed")
    assert "approval_gates" in data
    assert isinstance(data["approval_gates"], list)
    assert "intermediate_outputs" in data
    assert "created_at" in data
    assert "updated_at" in data
    container.run_repository.create.assert_called_once()
    container.run_repository.update.assert_called()


@pytest.mark.asyncio
async def test_submit_run_completes_simple_graph(simple_client):
    app, _ = simple_client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "simple", "user_request": "hello"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["approval_status"] == "not_required"
    assert data["approval_gates"] == []


@pytest.mark.asyncio
async def test_submit_run_unknown_workflow(simple_client):
    app, _ = simple_client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "does-not-exist", "user_request": "hi"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_submit_run_pauses_at_approval(approval_client):
    app, container = approval_client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "approvable", "user_request": "build it"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "waiting_approval"
    assert data["approval_status"] == "pending"
    # Synthetic approval gate created
    assert len(data["approval_gates"]) == 1
    gate = data["approval_gates"][0]
    assert gate["status"] == "pending"
    assert gate["step_id"] == "approve"


# ── GET /api/v1/workflows/runs/{run_id} ───────────────────────────────────────

@pytest.mark.asyncio
async def test_get_run_by_id(simple_client):
    app, container = simple_client
    stored = GraphRun(id="run-abc", graph_id="simple", status="completed")
    container.run_repository.get = AsyncMock(return_value=stored)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows/runs/run-abc")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "run-abc"
    assert data["workflow_id"] == "simple"
    assert data["workflow_name"] == "Simple Workflow"
    assert data["status"] == "completed"


@pytest.mark.asyncio
async def test_get_run_not_found(simple_client):
    app, container = simple_client
    container.run_repository.get = AsyncMock(return_value=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows/runs/missing-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_run_waiting_has_approval_gate(simple_client):
    app, container = simple_client
    stored = GraphRun(
        id="run-xyz",
        graph_id="simple",
        status="waiting_approval",
        state={"request": "do it", "_current_step": "approve"},
    )
    container.run_repository.get = AsyncMock(return_value=stored)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows/runs/run-xyz")
    data = resp.json()
    assert data["status"] == "waiting_approval"
    assert data["approval_status"] == "pending"
    assert len(data["approval_gates"]) == 1
    assert data["approval_gates"][0]["step_id"] == "approve"


@pytest.mark.asyncio
async def test_get_run_user_request_from_state(simple_client):
    app, container = simple_client
    stored = GraphRun(
        id="run-r",
        graph_id="simple",
        status="completed",
        state={"request": "my original request", "answer": "done"},
    )
    container.run_repository.get = AsyncMock(return_value=stored)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows/runs/run-r")
    assert resp.json()["user_request"] == "my original request"


@pytest.mark.asyncio
async def test_get_run_intermediate_outputs_exclude_private_keys(simple_client):
    app, container = simple_client
    stored = GraphRun(
        id="run-priv",
        graph_id="simple",
        status="completed",
        state={
            "request": "req",
            "answer": "42",
            "_current_step": None,
            "_interrupt_payload": None,
        },
    )
    container.run_repository.get = AsyncMock(return_value=stored)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows/runs/run-priv")
    outputs = resp.json()["intermediate_outputs"]
    assert "_current_step" not in outputs
    assert "_interrupt_payload" not in outputs
    assert outputs["answer"] == "42"


# ── POST /api/v1/workflows/runs/{id}/approve ─────────────────────────────────

@pytest.mark.asyncio
async def test_approve_run(approval_client):
    app, container = approval_client
    # First start a real run so LangGraph has checkpoint state
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        start = await c.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "approvable", "user_request": "build it"},
        )
        assert start.status_code == 200
        run_id = start.json()["id"]

        # Patch repo.get to return a run that mirrors the in-progress state
        run_obj = GraphRun(
            id=run_id, graph_id="approvable", status="waiting_approval",
            state={"request": "build it", "_current_step": "approve"},
        )
        container.run_repository.get = AsyncMock(return_value=run_obj)

        approve = await c.post(f"/api/v1/workflows/runs/{run_id}/approve")
    assert approve.status_code == 200
    data = approve.json()
    assert data["status"] == "completed"
    assert data["approval_status"] == "approved"
    assert data["approval_gates"] == []


@pytest.mark.asyncio
async def test_approve_run_not_found(simple_client):
    app, container = simple_client
    container.run_repository.get = AsyncMock(return_value=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/workflows/runs/no-such-run/approve")
    assert resp.status_code == 404


# ── POST /api/v1/workflows/runs/{id}/reject ───────────────────────────────────

@pytest.mark.asyncio
async def test_reject_run(approval_client):
    app, container = approval_client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        start = await c.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "approvable", "user_request": "build it"},
        )
        run_id = start.json()["id"]

        run_obj = GraphRun(
            id=run_id, graph_id="approvable", status="waiting_approval",
            state={"request": "build it", "_current_step": "approve"},
        )
        container.run_repository.get = AsyncMock(return_value=run_obj)

        reject = await c.post(
            f"/api/v1/workflows/runs/{run_id}/reject",
            json={"reason": "not convinced"},
        )
    assert reject.status_code == 200
    data = reject.json()
    assert data["status"] == "completed"
    assert data["approval_status"] == "rejected"


@pytest.mark.asyncio
async def test_reject_run_not_found(simple_client):
    app, container = simple_client
    container.run_repository.get = AsyncMock(return_value=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/workflows/runs/no-such-run/reject")
    assert resp.status_code == 404


# ── Gate-level approve / reject ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approve_gate_delegates_to_run_approve(approval_client):
    app, container = approval_client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        start = await c.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "approvable", "user_request": "build it"},
        )
        run_id = start.json()["id"]
        run_obj = GraphRun(
            id=run_id, graph_id="approvable", status="waiting_approval",
            state={"request": "build it", "_current_step": "approve"},
        )
        container.run_repository.get = AsyncMock(return_value=run_obj)

        resp = await c.post(
            f"/api/v1/workflows/runs/{run_id}/gates/approve/approve"
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_reject_gate_delegates_to_run_reject(approval_client):
    app, container = approval_client
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        start = await c.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "approvable", "user_request": "build it"},
        )
        run_id = start.json()["id"]
        run_obj = GraphRun(
            id=run_id, graph_id="approvable", status="waiting_approval",
            state={"request": "build it", "_current_step": "approve"},
        )
        container.run_repository.get = AsyncMock(return_value=run_obj)

        resp = await c.post(
            f"/api/v1/workflows/runs/{run_id}/gates/approve/reject",
            json={"reason": "nope"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_gate_approve_run_not_found(simple_client):
    app, container = simple_client
    container.run_repository.get = AsyncMock(return_value=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/api/v1/workflows/runs/ghost/gates/g1/approve")
    assert resp.status_code == 404


# ── approval_status derivation ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_approval_status_approved_when_completed_with_approved_true(simple_client):
    app, container = simple_client
    stored = GraphRun(
        id="r1", graph_id="simple", status="completed",
        state={"request": "r", "approved": True},
    )
    container.run_repository.get = AsyncMock(return_value=stored)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows/runs/r1")
    assert resp.json()["approval_status"] == "approved"


@pytest.mark.asyncio
async def test_approval_status_rejected_when_completed_with_approved_false(simple_client):
    app, container = simple_client
    stored = GraphRun(
        id="r2", graph_id="simple", status="completed",
        state={"request": "r", "approved": False},
    )
    container.run_repository.get = AsyncMock(return_value=stored)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows/runs/r2")
    assert resp.json()["approval_status"] == "rejected"


@pytest.mark.asyncio
async def test_approval_status_not_required_when_completed_no_approval(simple_client):
    app, container = simple_client
    stored = GraphRun(
        id="r3", graph_id="simple", status="completed",
        state={"request": "r"},
    )
    container.run_repository.get = AsyncMock(return_value=stored)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/api/v1/workflows/runs/r3")
    assert resp.json()["approval_status"] == "not_required"
