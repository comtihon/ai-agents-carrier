"""Tests for the /api/v1/events CRUD API."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.event_definition import EventDefinition
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.event_backend import EventDefinitionBackend
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


class InMemoryEventBackend(EventDefinitionBackend):
    def __init__(self) -> None:
        self._store: dict[str, EventDefinition] = {}

    async def list(self) -> list[EventDefinition]:
        return list(self._store.values())

    async def get(self, event_id: str) -> EventDefinition | None:
        return self._store.get(event_id)

    async def get_by_name(self, name: str) -> EventDefinition | None:
        return next((e for e in self._store.values() if e.name == name), None)

    async def create(self, definition: EventDefinition) -> EventDefinition:
        definition.touch()
        self._store[definition.id] = definition
        return definition

    async def update(self, event_id: str, definition: EventDefinition) -> EventDefinition:
        definition.touch()
        self._store[event_id] = definition
        return definition

    async def delete(self, event_id: str) -> None:
        self._store.pop(event_id, None)


def _build_container(backend: EventDefinitionBackend | None) -> ApplicationContainer:
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
        event_backend=backend,
    )


def _payload(**overrides) -> dict:
    body = {
        "id": "orders-events",
        "name": "Order events",
        "description": "Shop orders",
        "topic": "orders",
        "event_schema": {"type": "object", "required": ["order_id"]},
    }
    body.update(overrides)
    return body


@pytest.fixture
async def client():
    backend = InMemoryEventBackend()
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


# ─── Create / read ────────────────────────────────────────────────────────────

async def test_an_event_roundtrips_without_a_base_url_or_auth(client):
    c, backend = client

    created = await c.post("/api/v1/events", json=_payload())

    assert created.status_code == 201, created.text
    body = created.json()
    assert body["topic"] == "orders"
    assert body["event_schema"] == {"type": "object", "required": ["order_id"]}
    # Nothing named a subscription: one is created on first use.
    assert body["subscription"] == ""
    stored = await backend.get("orders-events")
    assert stored is not None and stored.topic == "orders"


async def test_listing_returns_every_event(client):
    c, _ = client
    await c.post("/api/v1/events", json=_payload())
    await c.post("/api/v1/events", json=_payload(id="shipments", name="Shipments", topic="shipments"))

    listed = await c.get("/api/v1/events")

    assert listed.status_code == 200
    assert {e["id"] for e in listed.json()} == {"orders-events", "shipments"}


async def test_getting_an_unknown_event_is_a_404(client):
    c, _ = client
    assert (await c.get("/api/v1/events/nope")).status_code == 404


async def test_an_event_without_a_topic_is_rejected(client):
    c, backend = client

    response = await c.post("/api/v1/events", json=_payload(topic="  "))

    assert response.status_code == 422
    assert "topic" in response.text
    assert await backend.get("orders-events") is None


async def test_a_duplicate_id_is_a_409(client):
    c, _ = client
    await c.post("/api/v1/events", json=_payload())

    again = await c.post("/api/v1/events", json=_payload(name="Something else"))

    assert again.status_code == 409
    assert "already exists" in again.text


async def test_a_duplicate_name_is_a_409_naming_the_event_it_clashes_with(client):
    """The warning the UI shows before it offers to overwrite."""
    c, _ = client
    await c.post("/api/v1/events", json=_payload())

    clash = await c.post("/api/v1/events", json=_payload(id="orders-v2"))

    assert clash.status_code == 409
    assert "Order events" in clash.text
    assert "orders-events" in clash.text


# ─── Update / delete ──────────────────────────────────────────────────────────

async def test_updating_records_the_subscription_and_keeps_created_at(client):
    c, _ = client
    created = (await c.post("/api/v1/events", json=_payload())).json()

    updated = await c.put(
        "/api/v1/events/orders-events",
        json={"subscription": "projects/p/subscriptions/aac-orders"},
    )

    assert updated.status_code == 200, updated.text
    body = updated.json()
    assert body["subscription"] == "projects/p/subscriptions/aac-orders"
    # Omitted fields survive.
    assert body["topic"] == "orders"
    assert body["event_schema"] == {"type": "object", "required": ["order_id"]}
    assert body["created_at"] == created["created_at"]


async def test_an_update_may_keep_its_own_name(client):
    c, _ = client
    await c.post("/api/v1/events", json=_payload())

    updated = await c.put("/api/v1/events/orders-events", json={"name": "Order events"})

    assert updated.status_code == 200


async def test_an_update_cannot_take_another_events_name(client):
    c, _ = client
    await c.post("/api/v1/events", json=_payload())
    await c.post("/api/v1/events", json=_payload(id="shipments", name="Shipments", topic="shipments"))

    clash = await c.put("/api/v1/events/shipments", json={"name": "Order events"})

    assert clash.status_code == 409


async def test_updating_an_unknown_event_is_a_404(client):
    c, _ = client
    assert (await c.put("/api/v1/events/nope", json={"topic": "x"})).status_code == 404


async def test_deleting_removes_the_event(client):
    c, backend = client
    await c.post("/api/v1/events", json=_payload())

    deleted = await c.delete("/api/v1/events/orders-events")

    assert deleted.status_code == 204
    assert await backend.get("orders-events") is None


async def test_deleting_an_unknown_event_is_a_404(client):
    c, _ = client
    assert (await c.delete("/api/v1/events/nope")).status_code == 404


# ─── No backend ───────────────────────────────────────────────────────────────

async def test_without_a_backend_every_route_is_a_501(client_without_backend):
    c = client_without_backend
    assert (await c.get("/api/v1/events")).status_code == 501
    assert (await c.post("/api/v1/events", json=_payload())).status_code == 501
