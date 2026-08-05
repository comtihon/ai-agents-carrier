"""Tests for the /api/v1/datasources CRUD API."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.datasources.executor import DataSourceExecutor
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.data_source_backend import DataSourceDefinitionBackend
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


class InMemoryDataSourceBackend(DataSourceDefinitionBackend):
    def __init__(self) -> None:
        self._store: dict[str, DataSourceDefinition] = {}

    async def list(self) -> list[DataSourceDefinition]:
        return list(self._store.values())

    async def get(self, source_id: str) -> DataSourceDefinition | None:
        return self._store.get(source_id)

    async def create(self, definition: DataSourceDefinition) -> DataSourceDefinition:
        definition.touch()
        self._store[definition.id] = definition
        return definition

    async def update(self, source_id: str, definition: DataSourceDefinition) -> DataSourceDefinition:
        definition.touch()
        self._store[source_id] = definition
        return definition

    async def delete(self, source_id: str) -> None:
        self._store.pop(source_id, None)


def _build_container(backend: DataSourceDefinitionBackend | None) -> ApplicationContainer:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.refresh_server = AsyncMock()
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
        data_source_backend=backend,
        data_source_executor=DataSourceExecutor() if backend else None,
    )


def _payload(**overrides) -> dict:
    body = {
        "id": "github",
        "name": "GitHub",
        "base_url": "https://api.github.com",
        "auth": {"type": "bearer", "token_env": "GITHUB_TOKEN"},
        "operations": [
            {
                "name": "list_repos",
                "path": "/users/{params.owner}/repos",
                "params": [{"name": "owner"}],
            }
        ],
    }
    body.update(overrides)
    return body


@pytest.fixture
async def client():
    backend = InMemoryDataSourceBackend()
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

    resp = await c.post("/api/v1/datasources", json=_payload())
    assert resp.status_code == 201
    assert resp.json()["auth"]["token_env"] == "GITHUB_TOKEN"

    resp = await c.get("/api/v1/datasources")
    assert resp.status_code == 200
    assert [s["id"] for s in resp.json()] == ["github"]

    resp = await c.get("/api/v1/datasources/github")
    assert resp.status_code == 200
    assert resp.json()["operations"][0]["name"] == "list_repos"

    resp = await c.put("/api/v1/datasources/github", json={"description": "code host"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["description"] == "code host"
    # Omitted fields are preserved.
    assert data["operations"][0]["name"] == "list_repos"
    assert data["auth"]["token_env"] == "GITHUB_TOKEN"

    resp = await c.delete("/api/v1/datasources/github")
    assert resp.status_code == 204
    assert await backend.get("github") is None


async def test_create_duplicate_conflicts(client):
    c, _ = client
    assert (await c.post("/api/v1/datasources", json=_payload())).status_code == 201
    assert (await c.post("/api/v1/datasources", json=_payload())).status_code == 409


async def test_get_unknown_returns_404(client):
    c, _ = client
    assert (await c.get("/api/v1/datasources/nope")).status_code == 404


async def test_create_with_cycle_returns_422(client):
    c, _ = client
    resp = await c.post("/api/v1/datasources", json=_payload(operations=[
        {"name": "a", "path": "/a/{b.id}"},
        {"name": "b", "path": "/b/{a.id}"},
    ]))
    assert resp.status_code == 422
    assert "Cyclic" in resp.json()["detail"]


async def test_create_with_unknown_operation_ref_returns_422(client):
    c, _ = client
    resp = await c.post("/api/v1/datasources", json=_payload(operations=[
        {"name": "a", "path": "/a/{ghost.id}"},
    ]))
    assert resp.status_code == 422
    assert "unknown operation" in resp.json()["detail"]


async def test_create_with_unknown_param_ref_returns_422(client):
    c, _ = client
    resp = await c.post("/api/v1/datasources", json=_payload(operations=[
        {"name": "a", "path": "/a/{params.missing}"},
    ]))
    assert resp.status_code == 422
    assert "unknown param" in resp.json()["detail"]


async def test_routes_return_501_without_backend(client_without_backend):
    c = client_without_backend
    assert (await c.get("/api/v1/datasources")).status_code == 501
    assert (await c.post("/api/v1/datasources", json=_payload())).status_code == 501
    assert (await c.get("/api/v1/datasources/github")).status_code == 501
