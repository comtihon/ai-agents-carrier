"""Regression tests for PUT /api/v1/agents/{agent_id} addons merge behavior.

Bug: update_agent used to build a brand-new AgentDefinition straight from the
request body, so a PUT that only intended to change e.g. system_prompt but
omitted `addons` would silently reset addons to []. Fixed by treating a
missing `addons` field (None) as "preserve existing" while an explicit []
still clears it.
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
from app.domain.models.agent_definition import AgentDefinition
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.agent_backend import AgentDefinitionBackend
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


class InMemoryAgentBackend(AgentDefinitionBackend):
    def __init__(self) -> None:
        self._store: dict[str, AgentDefinition] = {}

    async def list(self) -> list[AgentDefinition]:
        return list(self._store.values())

    async def get(self, agent_id: str) -> AgentDefinition | None:
        return self._store.get(agent_id)

    async def create(self, definition: AgentDefinition) -> AgentDefinition:
        definition.touch()
        self._store[definition.id] = definition
        return definition

    async def update(self, agent_id: str, definition: AgentDefinition) -> AgentDefinition:
        definition.touch()
        self._store[agent_id] = definition
        return definition

    async def delete(self, agent_id: str) -> None:
        self._store.pop(agent_id, None)


def _build_container(agent_backend: AgentDefinitionBackend) -> ApplicationContainer:
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
        agent_backend=agent_backend,
    )


@pytest.fixture
async def client():
    backend = InMemoryAgentBackend()
    await backend.create(
        AgentDefinition(
            id="researcher",
            name="Researcher",
            agent_input={"system_prompt": "old prompt"},
            addons=[
                {"type": "mcp", "servers": {"jira": True, "github": True}},
                {"type": "s3", "bucket": "b", "path": "{workflow_id}"},
            ],
        )
    )
    container = _build_container(backend)
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, backend


@pytest.mark.asyncio
async def test_put_without_addons_preserves_existing_addons(client):
    c, backend = client
    resp = await c.put(
        "/api/v1/agents/researcher",
        json={"agent_input": {"system_prompt": "new prompt"}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_input"]["system_prompt"] == "new prompt"
    assert len(data["addons"]) == 2
    assert {a["type"] for a in data["addons"]} == {"mcp", "s3"}

    stored = await backend.get("researcher")
    assert len(stored.addons) == 2


@pytest.mark.asyncio
async def test_put_with_explicit_empty_addons_clears_them(client):
    c, backend = client
    resp = await c.put(
        "/api/v1/agents/researcher",
        json={"agent_input": {"system_prompt": "new prompt"}, "addons": []},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["addons"] == []

    stored = await backend.get("researcher")
    assert stored.addons == []


@pytest.mark.asyncio
async def test_put_with_new_addons_replaces_existing(client):
    c, backend = client
    resp = await c.put(
        "/api/v1/agents/researcher",
        json={
            "agent_input": {"system_prompt": "new prompt"},
            "addons": [{"type": "s3", "bucket": "other", "path": "{workflow_id}"}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["addons"]) == 1
    assert data["addons"][0]["bucket"] == "other"
