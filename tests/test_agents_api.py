"""Regression tests for PUT /api/v1/agents/{agent_id} addons merge behavior.

Bug: update_agent used to build a brand-new AgentDefinition straight from the
request body, so a PUT that only intended to change e.g. system_prompt but
omitted `addons` would silently reset addons to []. Fixed by treating a
missing `addons` field (None) as "preserve existing" while an explicit []
still clears it.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings, get_settings
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
async def test_summary_view_omits_agent_input_and_addons(client):
    """A list of agents shows name/description/runtime.  ``agent_input`` carries
    the system prompt and ``addons`` the tool wiring; neither is drawn, so
    ``view=summary`` leaves them out and the editor fetches the full agent."""
    c, _ = client

    summary = (await c.get("/api/v1/agents", params={"view": "summary"})).json()

    assert summary == [{
        "id": "researcher",
        "name": "Researcher",
        "description": None,
        "default_runtime": "local",
    }]
    # Default view is unchanged.
    full = (await c.get("/api/v1/agents")).json()[0]
    assert full["agent_input"]["system_prompt"] == "old prompt"
    assert len(full["addons"]) == 2


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


@pytest.mark.asyncio
async def test_put_addons_accepts_tools_addon(client):
    c, backend = client
    resp = await c.put(
        "/api/v1/agents/researcher",
        json={
            "agent_input": {"system_prompt": "new prompt"},
            "addons": [
                {"type": "tools", "tools": {"github": True, "jira": False, "graphify": True}},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["addons"]) == 1
    assert data["addons"][0]["type"] == "tools"
    assert data["addons"][0]["tools"] == {"github": True, "jira": False, "graphify": True}

    # Round-trip: GET returns the persisted tools addon.
    get_resp = await c.get("/api/v1/agents/researcher")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    tools_addon = next(a for a in get_data["addons"] if a["type"] == "tools")
    assert tools_addon["tools"] == {"github": True, "jira": False, "graphify": True}


async def test_list_agent_tools_reports_registry_without_secret_values(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("TRACKER_API_TOKEN", "tracker-secret")
    monkeypatch.setenv(
        "AGENT_TOOLS",
        json.dumps({
            "tracker": {
                "label": "Tracker",
                "description": "issue tracker CLI",
                "env": {
                    "TRACKER_TOKEN": {"from_config": "TRACKER_API_TOKEN"},
                    "TRACKER_URL": {"from_config": "TRACKER_BASE_URL"},
                },
            },
            "grapher": {"command": "grapher", "cli_tools": {"grapher_query": {"args": ["query"]}}},
        }),
    )
    get_settings.cache_clear()
    try:
        resp = await c.get("/api/v1/agents/tools")
        assert resp.status_code == 200
        by_name = {t["name"]: t for t in resp.json()}

        tracker = by_name["tracker"]
        assert tracker["label"] == "Tracker"
        assert tracker["env_keys"] == ["TRACKER_TOKEN", "TRACKER_URL"]
        # TRACKER_BASE_URL is unset, so the tool is declared but incomplete.
        assert tracker["configured"] is False
        assert "tracker-secret" not in resp.text

        assert by_name["grapher"]["command"] == "grapher"
        assert by_name["grapher"]["cli_tools"] == ["grapher_query"]
    finally:
        get_settings.cache_clear()
