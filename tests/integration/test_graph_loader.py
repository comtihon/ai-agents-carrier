"""
Integration test: YAML graph loading from the filesystem.

Replaces: test_python_action_loader.py

Scenarios
─────────
1. A valid YAML file on disk is loaded at startup; the graph is runnable
   via the HTTP API end-to-end (real MongoDB, fake LLM).
2. A missing GRAPH_DEFINITIONS_PATH directory → empty registry, no error.
3. A malformed YAML file in the directory is skipped; valid files still load.
4. Multiple graph files in the directory are all registered independently.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer, build_container
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoClientProvider
from app.infrastructure.tools.mcp_client import McpToolsProvider
from tests.integration.conftest import _MONGO_URI, _TEST_DB, make_mock_llm

# ---------------------------------------------------------------------------
# Helper: build a full container from a temp graph directory
# ---------------------------------------------------------------------------


async def _build_from_dir(graph_dir: Path, llm) -> tuple[AsyncClient, MongoClientProvider]:
    settings = Settings(
        mongodb_uri=_MONGO_URI,
        mongodb_database=_TEST_DB,
        openhands_mock_mode=True,
        environment="test",
        graph_definitions_path=str(graph_dir),
    )
    mongo_provider = MongoClientProvider(settings)
    repo = mongo_provider.get_repository()
    await repo._collection.delete_many({})

    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)

    openhands_adapter = MagicMock(spec=OpenHandsAdapter)
    openhands_adapter.execute = AsyncMock(return_value={"status": "success"})

    from app.infrastructure.config.graph_loader import load_yaml_graphs

    registry = load_yaml_graphs(str(graph_dir), llm=llm, mcp_tools_provider=mcp)

    container = ApplicationContainer(
        settings=settings,
        llm=llm,
        mcp_tools_provider=mcp,
        yaml_graph_registry=registry,
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=openhands_adapter,
    )

    app = create_app()
    app.state.container = container

    http_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return http_client, mongo_provider


# ---------------------------------------------------------------------------
# Graph YAML content used across tests
# ---------------------------------------------------------------------------

_SIMPLE_YAML = textwrap.dedent("""\
    id: file-loaded-graph
    steps:
      - id: respond
        type: llm
        output_key: answer
        system_prompt: "Answer concisely."
        user_template: "{request}"
""")

_SECOND_YAML = textwrap.dedent("""\
    id: second-graph
    steps:
      - id: step1
        type: llm
        output_key: out
        user_template: "{request}"
""")

_BROKEN_YAML = "id: broken\nsteps: [{"  # invalid YAML


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_loaded_from_yaml_file_is_runnable(tmp_path) -> None:
    """
    A YAML file written to a temp directory is discovered at startup and the
    graph it defines can be invoked via the REST API.
    """
    (tmp_path / "file-loaded-graph.yaml").write_text(_SIMPLE_YAML)
    llm = make_mock_llm(text_responses=["answer from disk graph"])

    client, mongo = await _build_from_dir(tmp_path, llm)
    try:
        # Graph should appear in the listing
        list_resp = await client.get("/api/v1/workflows")
        assert list_resp.status_code == 200
        workflow_ids = [w["id"] for w in list_resp.json()]
        assert "file-loaded-graph" in workflow_ids

        # Graph should be runnable
        run_resp = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "file-loaded-graph", "user_request": "hello from disk"},
        )
        assert run_resp.status_code == 200, run_resp.text
        assert run_resp.json()["intermediate_outputs"]["answer"] == "answer from disk graph"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_missing_directory_yields_empty_registry(tmp_path) -> None:
    """
    When GRAPH_DEFINITIONS_PATH points to a non-existent directory the
    registry is empty and the list endpoint returns an empty list — no crash.
    """
    missing_dir = tmp_path / "does_not_exist"
    llm = make_mock_llm(text_responses=["x"])

    client, mongo = await _build_from_dir(missing_dir, llm)
    try:
        list_resp = await client.get("/api/v1/workflows")
        assert list_resp.status_code == 200
        assert list_resp.json() == []
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_malformed_yaml_skipped_valid_still_loads(tmp_path) -> None:
    """
    A broken YAML file is skipped with a log warning; valid files in the
    same directory are still registered.
    """
    (tmp_path / "broken.yaml").write_text(_BROKEN_YAML)
    (tmp_path / "file-loaded-graph.yaml").write_text(_SIMPLE_YAML)
    llm = make_mock_llm(text_responses=["fine"])

    client, mongo = await _build_from_dir(tmp_path, llm)
    try:
        list_resp = await client.get("/api/v1/workflows")
        workflow_ids = [w["id"] for w in list_resp.json()]
        assert "file-loaded-graph" in workflow_ids
        assert "broken" not in workflow_ids
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_multiple_yaml_files_all_registered(tmp_path) -> None:
    """
    All .yaml files in the directory are registered as separate graphs.
    """
    (tmp_path / "file-loaded-graph.yaml").write_text(_SIMPLE_YAML)
    (tmp_path / "second-graph.yaml").write_text(_SECOND_YAML)
    llm = make_mock_llm(text_responses=["r1", "r2"])

    client, mongo = await _build_from_dir(tmp_path, llm)
    try:
        list_resp = await client.get("/api/v1/workflows")
        workflow_ids = [w["id"] for w in list_resp.json()]
        assert "file-loaded-graph" in workflow_ids
        assert "second-graph" in workflow_ids
        assert len(workflow_ids) == 2
    finally:
        await mongo.close()
