"""Tests for the /api/v1/scripts CRUD API (the Python script library)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.script_definition import ScriptDefinition
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.persistence.script_backend import ScriptDefinitionBackend
from app.infrastructure.tools.mcp_client import McpToolsProvider


class InMemoryScriptBackend(ScriptDefinitionBackend):
    def __init__(self) -> None:
        self._store: dict[str, ScriptDefinition] = {}

    async def list(self) -> list[ScriptDefinition]:
        return list(self._store.values())

    async def get(self, script_id: str) -> ScriptDefinition | None:
        return self._store.get(script_id)

    async def get_by_name(self, name: str) -> ScriptDefinition | None:
        return next((s for s in self._store.values() if s.name == name), None)

    async def create(self, definition: ScriptDefinition) -> ScriptDefinition:
        definition.touch()
        self._store[definition.id] = definition
        return definition

    async def update(self, script_id: str, definition: ScriptDefinition) -> ScriptDefinition:
        definition.id = script_id
        definition.touch()
        self._store[script_id] = definition
        return definition

    async def delete(self, script_id: str) -> None:
        self._store.pop(script_id, None)


def _build_container(backend: ScriptDefinitionBackend | None) -> ApplicationContainer:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    return ApplicationContainer(
        settings=Settings(),
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=AsyncMock(spec=MongoGraphRunRepository),
        openhands=MagicMock(spec=OpenHandsAdapter),
        script_backend=backend,
    )


@pytest.fixture
async def client():
    backend = InMemoryScriptBackend()
    app = create_app()
    app.state.container = _build_container(backend)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, backend


@pytest.fixture
async def client_without_backend():
    app = create_app()
    app.state.container = _build_container(None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_create_get_list_update_delete_roundtrip(client):
    c, backend = client

    resp = await c.post("/api/v1/scripts", json={
        "name": "Sum Items", "description": "adds them up", "code": "output = 1",
    })
    assert resp.status_code == 201
    # The id is slugified from the name so saving by name is idempotent.
    assert resp.json()["id"] == "sum-items"

    resp = await c.get("/api/v1/scripts")
    assert [s["id"] for s in resp.json()] == ["sum-items"]

    resp = await c.get("/api/v1/scripts/sum-items")
    assert resp.json()["code"] == "output = 1"

    resp = await c.put("/api/v1/scripts/sum-items", json={
        "name": "Sum Items", "description": "adds them up", "code": "output = 2",
    })
    assert resp.status_code == 200
    assert (await backend.get("sum-items")).code == "output = 2"

    resp = await c.delete("/api/v1/scripts/sum-items")
    assert resp.status_code == 204
    assert await backend.get("sum-items") is None


async def test_create_with_existing_name_conflicts_until_overwrite(client):
    c, backend = client

    await c.post("/api/v1/scripts", json={"name": "Dedupe", "code": "output = 1"})

    resp = await c.post("/api/v1/scripts", json={"name": "Dedupe", "code": "output = 2"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]
    assert (await backend.get("dedupe")).code == "output = 1"

    resp = await c.post("/api/v1/scripts", json={
        "name": "Dedupe", "code": "output = 2", "overwrite": True,
    })
    assert resp.status_code == 201
    assert (await backend.get("dedupe")).code == "output = 2"


async def test_rename_onto_another_script_conflicts(client):
    c, _ = client
    await c.post("/api/v1/scripts", json={"name": "First", "code": ""})
    await c.post("/api/v1/scripts", json={"name": "Second", "code": ""})

    resp = await c.put("/api/v1/scripts/second", json={"name": "First", "code": ""})
    assert resp.status_code == 409


async def test_missing_script_is_404(client):
    c, _ = client
    assert (await c.get("/api/v1/scripts/nope")).status_code == 404
    assert (await c.delete("/api/v1/scripts/nope")).status_code == 404


async def test_routes_report_501_without_backend(client_without_backend):
    c = client_without_backend
    assert (await c.get("/api/v1/scripts")).status_code == 501
    assert (await c.post("/api/v1/scripts", json={"name": "x"})).status_code == 501
