"""
Integration tests: runs survive backend restart.

Simulates a restart by replacing the registry runner with a fresh instance
(new MemorySaver, no checkpoint state) and clearing live_runners — the same
effect as a real process restart.  _recover_run must reconstruct the
LangGraph checkpoint from MongoDB-persisted GraphRun state.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import MagicMock

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.persistence.mongo import MongoClientProvider
from app.infrastructure.tools.mcp_client import McpToolsProvider

from tests.integration.conftest import make_mock_llm

_MONGO_URI = "mongodb://localhost:27017"
_TEST_DB = "test_langgraph_integration"

_APPROVAL_GRAPH = {
    "id": "restart-approval",
    "steps": [
        {
            "id": "plan",
            "type": "llm",
            "output_key": "plan",
            "system_prompt": "Produce a short plan.",
            "user_template": "{request}",
        },
        {
            "id": "wait_for_approval",
            "type": "human_approval",
            "interrupt_payload": {"plan": "{plan}"},
        },
        {
            "id": "implement",
            "type": "llm",
            "when": "approved",
            "output_key": "implementation",
            "system_prompt": "Implement the plan.",
            "user_template": "Plan: {plan}",
        },
    ],
}

_SIMPLE_GRAPH = {
    "id": "restart-simple",
    "steps": [
        {
            "id": "step1",
            "type": "llm",
            "output_key": "answer",
            "system_prompt": "Answer concisely.",
            "user_template": "{request}",
        },
    ],
}


def _make_container(graph_def: dict, llm, mcp, mongo_provider: MongoClientProvider) -> ApplicationContainer:
    repo = mongo_provider.get_repository()
    runner = YamlGraphRunner(graph_def, llm=llm, mcp_tools_provider=mcp)
    registry = YamlGraphRegistry({runner.id: runner})
    openhands = MagicMock(spec=OpenHandsAdapter)
    return ApplicationContainer(
        settings=Settings(
            mongodb_uri=_MONGO_URI,
            mongodb_database=_TEST_DB,
            openhands_mock_mode=True,
            environment="test",
        ),
        llm=llm,
        mcp_tools_provider=mcp,
        yaml_graph_registry=registry,
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=openhands,
    )


def _fresh_mcp():
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])
    return mcp


@pytest.mark.asyncio
async def test_waiting_approval_survives_restart() -> None:
    """
    A run in waiting_approval can be approved after a simulated backend restart.

    Flow:
      1. Start run → reaches waiting_approval (plan step + interrupt).
      2. Simulate restart: replace registry runner with a fresh instance
         (new empty MemorySaver) and clear live_runners.
      3. Call _recover_run to re-seed the interrupt from MongoDB state.
      4. Approve → implement step runs → completed.
    """
    llm = make_mock_llm(text_responses=["the plan", "the implementation"])
    mcp = _fresh_mcp()

    mongo_provider = MongoClientProvider(Settings(
        mongodb_uri=_MONGO_URI, mongodb_database=_TEST_DB,
        openhands_mock_mode=True, environment="test",
    ))
    repo = mongo_provider.get_repository()
    await repo._collection.delete_many({})

    container = _make_container(_APPROVAL_GRAPH, llm, mcp, mongo_provider)
    app = create_app()
    app.state.container = container

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # ── 1. Start run ──────────────────────────────────────────────────
            resp = await client.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "restart-approval", "user_request": "build feature X"},
            )
            assert resp.status_code == 200, resp.text
            run_id = resp.json()["id"]

            get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
            assert get_resp.json()["status"] == "waiting_approval"
            assert get_resp.json()["intermediate_outputs"]["plan"] == "the plan"

            # ── 2. Simulate restart ───────────────────────────────────────────
            fresh_runner = YamlGraphRunner(_APPROVAL_GRAPH, llm=llm, mcp_tools_provider=mcp)
            container.yaml_graph_registry = YamlGraphRegistry({"restart-approval": fresh_runner})
            container.live_runners.clear()

            # Sanity: live_runners is empty — approve would 404 without recovery
            assert "restart-approval" not in container.live_runners
            assert run_id not in container.live_runners

            # ── 3. Recover ────────────────────────────────────────────────────
            run = await repo.get(run_id)
            assert run is not None
            assert run.status == "waiting_approval"
            await container._recover_run(run)

            # Recovery must have re-armed the interrupt and stored in live_runners
            assert run_id in container.live_runners

            # ── 4. Approve ────────────────────────────────────────────────────
            approve_resp = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
            assert approve_resp.status_code == 200, approve_resp.text

            # Background task completes; GET reflects final state
            get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
            body = get_resp.json()

            assert body["status"] == "completed"
            assert body["intermediate_outputs"]["plan"] == "the plan"
            assert body["intermediate_outputs"]["implementation"] == "the implementation"
            assert body["intermediate_outputs"]["approved"] is True

            # Persisted in MongoDB
            final = await repo.get(run_id)
            assert final is not None
            assert final.status == "completed"
    finally:
        await mongo_provider.close()


@pytest.mark.asyncio
async def test_rejection_survives_restart() -> None:
    """Same as above but rejects the run after restart recovery."""
    llm = make_mock_llm(text_responses=["the plan"])
    mcp = _fresh_mcp()

    mongo_provider = MongoClientProvider(Settings(
        mongodb_uri=_MONGO_URI, mongodb_database=_TEST_DB,
        openhands_mock_mode=True, environment="test",
    ))
    repo = mongo_provider.get_repository()
    await repo._collection.delete_many({})

    container = _make_container(_APPROVAL_GRAPH, llm, mcp, mongo_provider)
    app = create_app()
    app.state.container = container

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "restart-approval", "user_request": "build feature Y"},
            )
            run_id = resp.json()["id"]

            get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
            assert get_resp.json()["status"] == "waiting_approval"

            # Simulate restart
            fresh_runner = YamlGraphRunner(_APPROVAL_GRAPH, llm=llm, mcp_tools_provider=mcp)
            container.yaml_graph_registry = YamlGraphRegistry({"restart-approval": fresh_runner})
            container.live_runners.clear()

            run = await repo.get(run_id)
            await container._recover_run(run)
            assert run_id in container.live_runners

            # Reject
            reject_resp = await client.post(
                f"/api/v1/workflows/runs/{run_id}/reject",
                json={"reason": "not ready"},
            )
            assert reject_resp.status_code == 200, reject_resp.text

            # Background task completes; GET reflects final state
            get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
            body = get_resp.json()

            assert body["status"] == "cancelled"
            assert body["intermediate_outputs"]["approved"] is False
            assert body["intermediate_outputs"]["reject_reason"] == "not ready"
            assert "implementation" not in body["intermediate_outputs"]
    finally:
        await mongo_provider.close()


@pytest.mark.asyncio
async def test_startup_recover_incomplete_runs() -> None:
    """
    _recover_incomplete_runs is called at startup and re-arms all waiting_approval
    runs found in MongoDB, so subsequent approvals succeed.
    """
    llm = make_mock_llm(text_responses=["the plan", "the implementation"])
    mcp = _fresh_mcp()

    mongo_provider = MongoClientProvider(Settings(
        mongodb_uri=_MONGO_URI, mongodb_database=_TEST_DB,
        openhands_mock_mode=True, environment="test",
    ))
    repo = mongo_provider.get_repository()
    await repo._collection.delete_many({})

    container = _make_container(_APPROVAL_GRAPH, llm, mcp, mongo_provider)
    app = create_app()
    app.state.container = container

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "restart-approval", "user_request": "startup recovery test"},
            )
            run_id = resp.json()["id"]

            get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
            assert get_resp.json()["status"] == "waiting_approval"

            # Simulate restart: fresh registry + clear live_runners
            fresh_runner = YamlGraphRunner(_APPROVAL_GRAPH, llm=llm, mcp_tools_provider=mcp)
            container.yaml_graph_registry = YamlGraphRegistry({"restart-approval": fresh_runner})
            container.live_runners.clear()

            # Trigger the startup recovery path
            await container._recover_incomplete_runs()

            assert run_id in container.live_runners

            approve_resp = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
            assert approve_resp.status_code == 200, approve_resp.text

            # Background task completes; GET reflects final state
            get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
            assert get_resp.json()["status"] == "completed"
    finally:
        await mongo_provider.close()
