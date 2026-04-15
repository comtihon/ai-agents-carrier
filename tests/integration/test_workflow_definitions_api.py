"""
Integration tests for the workflow definitions CRUD API and run versioning.

Scenarios
─────────
1. POST /api/v1/workflows creates a definition and it appears in GET listing.
2. GET /api/v1/workflows/{id} returns the full definition.
3. PUT /api/v1/workflows/{id} updates the definition; new runs use the new version.
4. DELETE /api/v1/workflows/{id} removes the workflow; 404 on subsequent requests.
5. POST /api/v1/workflows returns 409 when the id already exists.
6. Run versioning: a run started before an update continues with the old definition
   (approval-resume still works correctly with the old runner).
7. Run versioning: a run started after an update uses the new definition.
8. Backend seeded from local YAML files at startup.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoClientProvider
from app.infrastructure.persistence.workflow_backend import (
    LocalFilesWorkflowBackend,
    MongoWorkflowBackend,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider
from tests.integration.conftest import _MONGO_URI, _TEST_DB, make_mock_llm

_WORKFLOW_DB = "test_workflow_definitions"

# ---------------------------------------------------------------------------
# Test workflow YAML definitions
# ---------------------------------------------------------------------------

_GRAPH_V1 = {
    "id": "versioned-wf",
    "name": "Versioned Workflow V1",
    "description": "Version 1",
    "steps": [
        {
            "id": "step_v1",
            "type": "llm",
            "output_key": "answer",
            "user_template": "{request}",
        }
    ],
}

_GRAPH_V2 = {
    "id": "versioned-wf",
    "name": "Versioned Workflow V2",
    "description": "Version 2",
    "steps": [
        {
            "id": "step_v2",
            "type": "llm",
            "output_key": "answer",
            "user_template": "{request}",
        }
    ],
}

_GRAPH_WITH_APPROVAL = {
    "id": "approval-versioned",
    "name": "Approval Versioned",
    "description": "Workflow with approval gate",
    "steps": [
        {
            "id": "plan",
            "type": "llm",
            "output_key": "plan",
            "user_template": "{request}",
        },
        {
            "id": "approve",
            "type": "human_approval",
            "interrupt_payload": {"plan": "{plan}"},
        },
        {
            "id": "execute",
            "type": "llm",
            "output_key": "result",
            "when": "approved",
            "user_template": "Execute: {plan}",
        },
    ],
}


# ---------------------------------------------------------------------------
# Client builder using LocalFilesWorkflowBackend (in temp dir)
# ---------------------------------------------------------------------------

async def _build_client_with_localfiles(
    tmp_path,
    llm: Any,
    seed_definitions: list[dict] | None = None,
) -> tuple[AsyncClient, MongoClientProvider]:
    """Build a test client backed by a LocalFilesWorkflowBackend."""
    import yaml

    # Pre-seed YAML files into the temp graphs directory
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    for defn in (seed_definitions or []):
        (graphs_dir / f"{defn['id']}.yaml").write_text(
            yaml.dump(defn, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )

    settings = Settings(
        mongodb_uri=_MONGO_URI,
        mongodb_database=_TEST_DB,
        openhands_mock_mode=True,
        environment="test",
        graph_definitions_path=str(graphs_dir),
        workflow_backend_type="localfiles",
    )
    mongo_provider = MongoClientProvider(settings)
    repo = mongo_provider.get_repository()
    await repo._collection.delete_many({})

    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])

    openhands_adapter = MagicMock(spec=OpenHandsAdapter)
    openhands_adapter.execute = AsyncMock(return_value={"status": "success"})

    workflow_backend = LocalFilesWorkflowBackend(str(graphs_dir))

    container = ApplicationContainer(
        settings=settings,
        llm=llm,
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=openhands_adapter,
        workflow_backend=workflow_backend,
    )
    # Simulate startup: load registry from backend
    await container._load_registry()

    app = create_app()
    app.state.container = container
    http_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return http_client, mongo_provider


async def _build_client_with_mongo(
    llm: Any,
    seed_definitions: list[dict] | None = None,
) -> tuple[AsyncClient, MongoClientProvider, MongoWorkflowBackend]:
    """Build a test client backed by a MongoWorkflowBackend."""
    import yaml
    from app.domain.models.workflow_definition import WorkflowDefinition

    settings = Settings(
        mongodb_uri=_MONGO_URI,
        mongodb_database=_WORKFLOW_DB,
        openhands_mock_mode=True,
        environment="test",
        workflow_backend_type="mongodb",
    )
    mongo_provider = MongoClientProvider(settings)
    repo = mongo_provider.get_repository()
    await repo._collection.delete_many({})

    workflow_backend = MongoWorkflowBackend(_MONGO_URI, _WORKFLOW_DB)
    await workflow_backend._col.delete_many({})

    for defn_dict in (seed_definitions or []):
        defn = WorkflowDefinition(
            id=defn_dict["id"],
            name=defn_dict.get("name", ""),
            description=defn_dict.get("description", ""),
            steps=defn_dict.get("steps", []),
        )
        await workflow_backend.create(defn)

    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])

    openhands_adapter = MagicMock(spec=OpenHandsAdapter)
    openhands_adapter.execute = AsyncMock(return_value={"status": "success"})

    container = ApplicationContainer(
        settings=settings,
        llm=llm,
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=openhands_adapter,
        workflow_backend=workflow_backend,
    )
    await container._load_registry()

    app = create_app()
    app.state.container = container
    http_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return http_client, mongo_provider, workflow_backend


# ============================================================================
# LocalFiles backend tests
# ============================================================================

class TestLocalFilesWorkflowDefinitionsAPI:
    @pytest.mark.asyncio
    async def test_create_workflow_and_list(self, tmp_path) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo = await _build_client_with_localfiles(tmp_path, llm)
        try:
            resp = await client.post("/api/v1/workflows", json=_GRAPH_V1)
            assert resp.status_code == 201, resp.text
            assert resp.json()["id"] == "versioned-wf"

            list_resp = await client.get("/api/v1/workflows")
            assert list_resp.status_code == 200
            ids = [w["id"] for w in list_resp.json()]
            assert "versioned-wf" in ids
        finally:
            await mongo.close()

    @pytest.mark.asyncio
    async def test_get_workflow_returns_full_definition(self, tmp_path) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo = await _build_client_with_localfiles(tmp_path, llm, seed_definitions=[_GRAPH_V1])
        try:
            resp = await client.get("/api/v1/workflows/versioned-wf")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["id"] == "versioned-wf"
            assert body["name"] == "Versioned Workflow V1"
            assert len(body["steps"]) == 1
            assert body["steps"][0]["id"] == "step_v1"
        finally:
            await mongo.close()

    @pytest.mark.asyncio
    async def test_get_workflow_404_for_missing(self, tmp_path) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo = await _build_client_with_localfiles(tmp_path, llm)
        try:
            resp = await client.get("/api/v1/workflows/no-such-workflow")
            assert resp.status_code == 404
        finally:
            await mongo.close()

    @pytest.mark.asyncio
    async def test_update_workflow(self, tmp_path) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo = await _build_client_with_localfiles(tmp_path, llm, seed_definitions=[_GRAPH_V1])
        try:
            update_body = {
                "name": "Updated Name",
                "description": "Updated",
                "steps": _GRAPH_V2["steps"],
            }
            resp = await client.put("/api/v1/workflows/versioned-wf", json=update_body)
            assert resp.status_code == 200, resp.text
            assert resp.json()["name"] == "Updated Name"

            get_resp = await client.get("/api/v1/workflows/versioned-wf")
            assert get_resp.json()["name"] == "Updated Name"
            assert get_resp.json()["steps"][0]["id"] == "step_v2"
        finally:
            await mongo.close()

    @pytest.mark.asyncio
    async def test_update_workflow_404_for_missing(self, tmp_path) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo = await _build_client_with_localfiles(tmp_path, llm)
        try:
            resp = await client.put(
                "/api/v1/workflows/ghost",
                json={"name": "x", "description": "", "steps": []},
            )
            assert resp.status_code == 404
        finally:
            await mongo.close()

    @pytest.mark.asyncio
    async def test_delete_workflow(self, tmp_path) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo = await _build_client_with_localfiles(tmp_path, llm, seed_definitions=[_GRAPH_V1])
        try:
            resp = await client.delete("/api/v1/workflows/versioned-wf")
            assert resp.status_code == 204

            get_resp = await client.get("/api/v1/workflows/versioned-wf")
            assert get_resp.status_code == 404

            list_resp = await client.get("/api/v1/workflows")
            ids = [w["id"] for w in list_resp.json()]
            assert "versioned-wf" not in ids
        finally:
            await mongo.close()

    @pytest.mark.asyncio
    async def test_create_duplicate_returns_409(self, tmp_path) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo = await _build_client_with_localfiles(tmp_path, llm, seed_definitions=[_GRAPH_V1])
        try:
            resp = await client.post("/api/v1/workflows", json=_GRAPH_V1)
            assert resp.status_code == 409
        finally:
            await mongo.close()

    @pytest.mark.asyncio
    async def test_new_run_uses_latest_definition(self, tmp_path) -> None:
        """After updating a workflow, a new run uses the updated definition."""
        llm = make_mock_llm(text_responses=["v2 response"])
        client, mongo = await _build_client_with_localfiles(tmp_path, llm, seed_definitions=[_GRAPH_V1])
        try:
            # Update to V2 (different step id)
            await client.put(
                "/api/v1/workflows/versioned-wf",
                json={"name": "V2", "description": "", "steps": _GRAPH_V2["steps"]},
            )

            run_resp = await client.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "versioned-wf", "user_request": "hello"},
            )
            assert run_resp.status_code == 200, run_resp.text
            run_id = run_resp.json()["id"]

            get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
            body = get_resp.json()
            # The run's steps should reflect V2 (step_v2, not step_v1)
            step_ids = [s["id"] for s in body["steps"]]
            assert "step_v2" in step_ids
            assert "step_v1" not in step_ids
        finally:
            await mongo.close()

    @pytest.mark.asyncio
    async def test_seeded_yaml_files_loaded_at_startup(self, tmp_path) -> None:
        """YAML files present at startup are loadable via the API."""
        llm = make_mock_llm(text_responses=["ok"])
        client, mongo = await _build_client_with_localfiles(tmp_path, llm, seed_definitions=[_GRAPH_V1])
        try:
            list_resp = await client.get("/api/v1/workflows")
            assert list_resp.status_code == 200
            ids = [w["id"] for w in list_resp.json()]
            assert "versioned-wf" in ids
        finally:
            await mongo.close()

    @pytest.mark.asyncio
    async def test_run_versioning_old_run_uses_old_definition(self, tmp_path) -> None:
        """
        A run started before a definition update must use the original definition.

        Scenario:
        1. Start a run with V1 → reaches human_approval gate.
        2. Update definition to V2.
        3. Approve the V1 run → must succeed with the V1 runner (MemorySaver intact).
        4. Start a new run → uses V2 definition.
        """
        llm = make_mock_llm(
            text_responses=["v1 plan", "v1 execute result"],
        )
        client, mongo = await _build_client_with_localfiles(
            tmp_path, llm, seed_definitions=[_GRAPH_WITH_APPROVAL]
        )
        try:
            # Step 1: Start run — reaches approval gate
            run_resp = await client.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "approval-versioned", "user_request": "do something"},
            )
            assert run_resp.status_code == 200, run_resp.text
            run_id = run_resp.json()["id"]

            get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
            assert get_resp.json()["status"] == "waiting_approval", get_resp.json()

            # Step 2: Update the workflow definition (add a new step)
            updated_steps = _GRAPH_WITH_APPROVAL["steps"] + [
                {"id": "new_step_v2", "type": "llm", "output_key": "extra", "user_template": "x"}
            ]
            await client.put(
                "/api/v1/workflows/approval-versioned",
                json={
                    "name": "V2",
                    "description": "v2",
                    "steps": updated_steps,
                },
            )

            # Step 3: Approve the V1 run — should still work with V1 runner
            approve_resp = await client.post(f"/api/v1/workflows/runs/{run_id}/approve")
            assert approve_resp.status_code == 200, approve_resp.text
            assert approve_resp.json()["status"] == "completed"
            # Response steps should reflect V1 definition (3 steps, no new_step_v2)
            step_ids = [s["id"] for s in approve_resp.json()["steps"]]
            assert "new_step_v2" not in step_ids
            assert "execute" in step_ids

            # Step 4: New run uses V2
            new_llm = make_mock_llm(text_responses=["v2 response", "extra response"])
            # Rebuild the container would normally use a new runner; since the registry
            # was already updated by the PUT, starting a new run should pick up V2.
            new_run_resp = await client.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "approval-versioned", "user_request": "v2 run"},
            )
            assert new_run_resp.status_code == 200, new_run_resp.text
            # V2 has 4 steps including new_step_v2
            new_step_ids = [s["id"] for s in new_run_resp.json()["steps"]]
            assert "new_step_v2" in new_step_ids
        finally:
            await mongo.close()


# ============================================================================
# MongoDB backend tests (skipped when MongoDB unavailable)
# ============================================================================

def _check_mongo() -> bool:
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError
        client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


_mongo_available = _check_mongo()
_skip_mongo = pytest.mark.skipif(not _mongo_available, reason="MongoDB not reachable")


class TestMongoWorkflowDefinitionsAPI:
    @_skip_mongo
    @pytest.mark.asyncio
    async def test_create_and_list_via_mongo_backend(self) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo, wf_backend = await _build_client_with_mongo(llm)
        try:
            resp = await client.post("/api/v1/workflows", json=_GRAPH_V1)
            assert resp.status_code == 201, resp.text

            list_resp = await client.get("/api/v1/workflows")
            ids = [w["id"] for w in list_resp.json()]
            assert "versioned-wf" in ids
        finally:
            await mongo.close()
            await wf_backend.close()

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_run_uses_definition_from_mongo(self) -> None:
        llm = make_mock_llm(text_responses=["mongo response"])
        client, mongo, wf_backend = await _build_client_with_mongo(llm, seed_definitions=[_GRAPH_V1])
        try:
            run_resp = await client.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "versioned-wf", "user_request": "hello from mongo"},
            )
            assert run_resp.status_code == 200, run_resp.text
            run_id = run_resp.json()["id"]

            get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
            assert get_resp.json()["status"] == "completed"
            assert get_resp.json()["intermediate_outputs"]["answer"] == "mongo response"
        finally:
            await mongo.close()
            await wf_backend.close()

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_update_workflow_in_mongo(self) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo, wf_backend = await _build_client_with_mongo(llm, seed_definitions=[_GRAPH_V1])
        try:
            update_body = {
                "name": "Updated Via Mongo",
                "description": "v2",
                "steps": _GRAPH_V2["steps"],
            }
            resp = await client.put("/api/v1/workflows/versioned-wf", json=update_body)
            assert resp.status_code == 200, resp.text
            assert resp.json()["name"] == "Updated Via Mongo"

            # Verify persisted in MongoDB
            defn = await wf_backend.get("versioned-wf")
            assert defn is not None
            assert defn.name == "Updated Via Mongo"
            assert defn.steps[0]["id"] == "step_v2"
        finally:
            await mongo.close()
            await wf_backend.close()

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_delete_workflow_in_mongo(self) -> None:
        llm = make_mock_llm(text_responses=["response"])
        client, mongo, wf_backend = await _build_client_with_mongo(llm, seed_definitions=[_GRAPH_V1])
        try:
            resp = await client.delete("/api/v1/workflows/versioned-wf")
            assert resp.status_code == 204

            # Verify removed from MongoDB
            defn = await wf_backend.get("versioned-wf")
            assert defn is None

            # Verify 404 from API
            get_resp = await client.get("/api/v1/workflows/versioned-wf")
            assert get_resp.status_code == 404
        finally:
            await mongo.close()
            await wf_backend.close()

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_mongo_backend_seeded_at_startup(self) -> None:
        """Definitions in MongoDB are available via the API after startup."""
        llm = make_mock_llm(text_responses=["ok"])
        client, mongo, wf_backend = await _build_client_with_mongo(llm, seed_definitions=[_GRAPH_V1])
        try:
            list_resp = await client.get("/api/v1/workflows")
            ids = [w["id"] for w in list_resp.json()]
            assert "versioned-wf" in ids
        finally:
            await mongo.close()
            await wf_backend.close()
