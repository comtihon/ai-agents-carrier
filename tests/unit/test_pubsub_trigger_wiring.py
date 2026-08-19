"""The `pubsub` trigger step: graph node, spec resolution, write-back.

Covers the three seams between an arriving event and a workflow run: the node
that republishes the event into state, the container code that turns a step (or
the data source it points at) into a subscriber spec, and the hook that saves a
freshly created subscription back into the data sources.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.core.container import ApplicationContainer, _pubsub_source_id
from app.domain.models.data_source_definition import DataSourceDefinition, PubSubSpec
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.tools.mcp_client import McpToolsProvider
from app.infrastructure.triggers.pubsub_subscriber import PubSubTriggerSpec

pytestmark = pytest.mark.asyncio


def _runner(step: dict) -> YamlGraphRunner:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])
    return YamlGraphRunner({"id": "g", "steps": [step]}, llm=llm, mcp_tools_provider=mcp)


# ─── graph node ───────────────────────────────────────────────────────────────

async def test_the_node_republishes_the_event_under_output_key():
    step = {"id": "on_order", "type": "pubsub", "topic": "orders", "output_key": "event"}
    runner = _runner(step)

    result = await runner._pubsub_trigger_node(step)(
        {"trigger_payload": {"order_id": "A-1"}, "trigger_info": {"message_id": "m-1"}}
    )

    assert result["event"] == {"order_id": "A-1"}


async def test_the_node_fills_request_from_the_event_when_there_is_none():
    step = {"id": "on_order", "type": "pubsub", "topic": "orders"}
    runner = _runner(step)

    result = await runner._pubsub_trigger_node(step)({"trigger_payload": {"order_id": "A-1"}})

    assert result["trigger_payload"] == {"order_id": "A-1"}
    assert result["request"] == '{"order_id": "A-1"}'


async def test_the_node_keeps_an_existing_request():
    step = {"id": "on_order", "type": "pubsub", "topic": "orders"}
    runner = _runner(step)

    result = await runner._pubsub_trigger_node(step)(
        {"trigger_payload": {"order_id": "A-1"}, "request": "handle order A-1"}
    )

    assert "request" not in result


async def test_a_manual_run_sees_an_empty_event():
    step = {"id": "on_order", "type": "pubsub", "topic": "orders", "output_key": "event"}
    runner = _runner(step)

    result = await runner._pubsub_trigger_node(step)({"request": "manual"})

    assert result == {"event": {}}


async def test_the_graph_builds_with_a_pubsub_step():
    # Unknown step types raise in _build_node — this asserts pubsub is wired in.
    runner = _runner({"id": "on_order", "type": "pubsub", "topic": "orders"})
    assert runner.steps[0]["type"] == "pubsub"
    assert runner._pubsub_trigger_node is not None


# ─── spec resolution ──────────────────────────────────────────────────────────

class FakeDataSourceBackend:
    def __init__(self, sources: dict[str, DataSourceDefinition] | None = None) -> None:
        self.sources = sources or {}
        self.created: list[DataSourceDefinition] = []
        self.updated: list[DataSourceDefinition] = []

    async def get(self, source_id):
        return self.sources.get(source_id)

    async def create(self, definition):
        self.created.append(definition)
        self.sources[definition.id] = definition
        return definition

    async def update(self, source_id, definition):
        self.updated.append(definition)
        self.sources[source_id] = definition
        return definition

    async def list(self):
        return list(self.sources.values())


def _container(backend: FakeDataSourceBackend | None = None, **setting_overrides) -> ApplicationContainer:
    settings = Settings(
        PUBSUB_ENABLED=True,
        PUBSUB_PROJECT_ID="proj",
        **setting_overrides,
    )
    container = ApplicationContainer(
        settings=settings,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=MagicMock(spec=McpToolsProvider),
        openhands=MagicMock(),
        run_repository=MagicMock(),
        yaml_graph_registry=MagicMock(),
        mongo_provider=MagicMock(),
    )
    container.data_source_backend = backend
    return container


async def test_a_step_supplies_its_own_topic_and_schema():
    container = _container()

    spec = await container._build_pubsub_spec({
        "id": "on_order",
        "type": "pubsub",
        "topic": "orders",
        "schema": {"type": "object", "required": ["order_id"]},
        "max_messages": 3,
    })

    assert spec.topic == "orders"
    assert spec.project_id == "proj"
    assert spec.event_schema == {"type": "object", "required": ["order_id"]}
    assert spec.max_messages == 3
    assert spec.ack_deadline_seconds == 60


async def test_a_step_can_take_everything_from_a_pubsub_datasource():
    source = DataSourceDefinition(
        id="orders-events",
        kind="pubsub",
        pubsub=PubSubSpec(
            topic="orders",
            subscription="projects/proj/subscriptions/shared",
            project_id="other",
            event_schema={"type": "object"},
        ),
    )
    container = _container(FakeDataSourceBackend({"orders-events": source}))

    spec = await container._build_pubsub_spec({
        "id": "on_order", "type": "pubsub", "datasource": "orders-events",
    })

    assert spec.topic == "orders"
    assert spec.subscription == "projects/proj/subscriptions/shared"
    assert spec.project_id == "other"
    assert spec.event_schema == {"type": "object"}
    assert spec.datasource_id == "orders-events"


async def test_step_fields_override_the_datasource():
    source = DataSourceDefinition(
        id="orders-events",
        kind="pubsub",
        pubsub=PubSubSpec(topic="orders", subscription="sub-a", event_schema={"type": "object"}),
    )
    container = _container(FakeDataSourceBackend({"orders-events": source}))

    spec = await container._build_pubsub_spec({
        "id": "on_order",
        "type": "pubsub",
        "datasource": "orders-events",
        "topic": "orders-v2",
        "subscription": "sub-b",
    })

    assert (spec.topic, spec.subscription) == ("orders-v2", "sub-b")


async def test_a_step_without_a_topic_is_rejected():
    container = _container()

    with pytest.raises(ValueError, match="no topic configured"):
        await container._build_pubsub_spec({"id": "on_order", "type": "pubsub"})


async def test_pointing_at_a_non_pubsub_datasource_is_rejected():
    source = DataSourceDefinition(id="github", kind="http", base_url="https://api.github.com")
    container = _container(FakeDataSourceBackend({"github": source}))

    with pytest.raises(ValueError, match="not a Pub/Sub source"):
        await container._build_pubsub_spec({
            "id": "on_order", "type": "pubsub", "datasource": "github",
        })


async def test_pointing_at_a_missing_datasource_is_rejected():
    container = _container(FakeDataSourceBackend())

    with pytest.raises(ValueError, match="not found"):
        await container._build_pubsub_spec({
            "id": "on_order", "type": "pubsub", "datasource": "ghost",
        })


# ─── write-back ───────────────────────────────────────────────────────────────

async def test_a_created_subscription_lands_on_the_datasource_it_came_from():
    source = DataSourceDefinition(
        id="orders-events", kind="pubsub", pubsub=PubSubSpec(topic="orders"),
    )
    backend = FakeDataSourceBackend({"orders-events": source})
    container = _container(backend)
    spec = PubSubTriggerSpec(topic="orders", project_id="proj", datasource_id="orders-events")

    await container._save_pubsub_subscription(spec, "projects/proj/subscriptions/aac-wf-step")

    assert backend.updated[0].pubsub.subscription == "projects/proj/subscriptions/aac-wf-step"
    assert backend.created == []


async def test_an_inline_trigger_gets_a_new_pubsub_datasource():
    backend = FakeDataSourceBackend()
    container = _container(backend)
    spec = PubSubTriggerSpec(
        topic="projects/proj/topics/orders",
        project_id="proj",
        event_schema={"type": "object"},
    )

    await container._save_pubsub_subscription(spec, "projects/proj/subscriptions/aac-wf-step")

    created = backend.created[0]
    assert created.id == "pubsub-orders"
    assert created.kind == "pubsub"
    assert created.pubsub.topic == "projects/proj/topics/orders"
    assert created.pubsub.subscription == "projects/proj/subscriptions/aac-wf-step"
    assert created.pubsub.event_schema == {"type": "object"}


async def test_write_back_is_idempotent():
    source = DataSourceDefinition(
        id="pubsub-orders",
        kind="pubsub",
        pubsub=PubSubSpec(topic="orders", subscription="projects/proj/subscriptions/aac-wf-step"),
    )
    backend = FakeDataSourceBackend({"pubsub-orders": source})
    container = _container(backend)

    await container._save_pubsub_subscription(
        PubSubTriggerSpec(topic="orders", project_id="proj"),
        "projects/proj/subscriptions/aac-wf-step",
    )

    assert (backend.created, backend.updated) == ([], [])


async def test_write_back_without_a_datasource_backend_is_a_no_op():
    container = _container(None)
    # Must not raise — a Mongo-less deployment still runs pubsub triggers.
    await container._save_pubsub_subscription(
        PubSubTriggerSpec(topic="orders", project_id="proj"), "projects/proj/subscriptions/s",
    )


async def test_source_ids_are_slugged_from_the_topic():
    assert _pubsub_source_id("projects/p/topics/Order.Events") == "pubsub-order-events"
    assert _pubsub_source_id("orders") == "pubsub-orders"
    assert _pubsub_source_id("") == "pubsub-topic"
