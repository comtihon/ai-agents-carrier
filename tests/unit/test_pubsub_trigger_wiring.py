"""The `pubsub` trigger step: graph node, spec resolution, write-back.

Covers the three seams between an arriving event and a workflow run: the node
that republishes the event into state, the container code that turns a step (or
the event it points at) into a subscriber spec, and the hook that saves a
freshly created subscription back onto the event.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.core.container import ApplicationContainer, _pubsub_source_id
from app.domain.models.data_source_definition import DataSourceDefinition, PubSubSpec
from app.domain.models.event_definition import EventDefinition
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


class FakeEventBackend:
    def __init__(self, events: dict[str, EventDefinition] | None = None) -> None:
        self.events = events or {}
        self.created: list[EventDefinition] = []
        self.updated: list[EventDefinition] = []

    async def get(self, event_id):
        return self.events.get(event_id)

    async def get_by_name(self, name):
        return next((e for e in self.events.values() if e.name == name), None)

    async def create(self, definition):
        self.created.append(definition)
        self.events[definition.id] = definition
        return definition

    async def update(self, event_id, definition):
        self.updated.append(definition)
        self.events[event_id] = definition
        return definition

    async def delete(self, event_id):
        self.events.pop(event_id, None)

    async def list(self):
        return list(self.events.values())


def _container(
    backend: FakeEventBackend | None = None,
    data_sources: FakeDataSourceBackend | None = None,
    **setting_overrides,
) -> ApplicationContainer:
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
    container.event_backend = backend
    container.data_source_backend = data_sources
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


async def test_a_step_can_take_everything_from_an_event():
    event = EventDefinition(
        id="orders-events",
        topic="orders",
        subscription="projects/proj/subscriptions/shared",
        project_id="other",
        event_schema={"type": "object"},
    )
    container = _container(FakeEventBackend({"orders-events": event}))

    spec = await container._build_pubsub_spec({
        "id": "on_order", "type": "pubsub", "event": "orders-events",
    })

    assert spec.topic == "orders"
    assert spec.subscription == "projects/proj/subscriptions/shared"
    assert spec.project_id == "other"
    assert spec.event_schema == {"type": "object"}
    assert spec.datasource_id == "orders-events"


async def test_step_fields_override_the_event():
    event = EventDefinition(
        id="orders-events", topic="orders", subscription="sub-a",
        event_schema={"type": "object"},
    )
    container = _container(FakeEventBackend({"orders-events": event}))

    spec = await container._build_pubsub_spec({
        "id": "on_order",
        "type": "pubsub",
        "event": "orders-events",
        "topic": "orders-v2",
        "subscription": "sub-b",
    })

    assert (spec.topic, spec.subscription) == ("orders-v2", "sub-b")


async def test_the_pre_events_datasource_key_still_resolves_against_the_events():
    """Workflows written before events existed keep working unchanged."""
    event = EventDefinition(id="pubsub-orders", topic="orders")
    container = _container(FakeEventBackend({"pubsub-orders": event}))

    spec = await container._build_pubsub_spec({
        "id": "on_order", "type": "pubsub", "datasource": "pubsub-orders",
    })

    assert spec.topic == "orders"


async def test_an_unmigrated_pubsub_datasource_is_still_read():
    """Until the migration script runs, the topic may only exist over there."""
    source = DataSourceDefinition(
        id="orders-events", kind="pubsub", pubsub=PubSubSpec(topic="orders"),
    )
    container = _container(
        FakeEventBackend(), data_sources=FakeDataSourceBackend({"orders-events": source}),
    )

    spec = await container._build_pubsub_spec({
        "id": "on_order", "type": "pubsub", "datasource": "orders-events",
    })

    assert spec.topic == "orders"


async def test_a_step_without_a_topic_is_rejected():
    container = _container()

    with pytest.raises(ValueError, match="no topic configured"):
        await container._build_pubsub_spec({"id": "on_order", "type": "pubsub"})


async def test_pointing_at_a_non_pubsub_datasource_is_rejected():
    source = DataSourceDefinition(id="github", kind="http", base_url="https://api.github.com")
    container = _container(
        FakeEventBackend(), data_sources=FakeDataSourceBackend({"github": source}),
    )

    with pytest.raises(ValueError, match="not an event"):
        await container._build_pubsub_spec({
            "id": "on_order", "type": "pubsub", "event": "github",
        })


async def test_pointing_at_a_missing_event_is_rejected():
    container = _container(FakeEventBackend())

    with pytest.raises(ValueError, match="not found"):
        await container._build_pubsub_spec({
            "id": "on_order", "type": "pubsub", "event": "ghost",
        })


# ─── write-back ───────────────────────────────────────────────────────────────

async def test_a_created_subscription_lands_on_the_event_it_came_from():
    event = EventDefinition(id="orders-events", topic="orders")
    backend = FakeEventBackend({"orders-events": event})
    container = _container(backend)
    spec = PubSubTriggerSpec(topic="orders", project_id="proj", datasource_id="orders-events")

    await container._save_pubsub_subscription(spec, "projects/proj/subscriptions/aac-wf-step")

    assert backend.updated[0].subscription == "projects/proj/subscriptions/aac-wf-step"
    assert backend.created == []


async def test_an_inline_trigger_gets_a_new_event():
    backend = FakeEventBackend()
    container = _container(backend)
    spec = PubSubTriggerSpec(
        topic="projects/proj/topics/orders",
        project_id="proj",
        event_schema={"type": "object"},
    )

    await container._save_pubsub_subscription(spec, "projects/proj/subscriptions/aac-wf-step")

    created = backend.created[0]
    assert created.id == "pubsub-orders"
    assert created.topic == "projects/proj/topics/orders"
    assert created.subscription == "projects/proj/subscriptions/aac-wf-step"
    assert created.event_schema == {"type": "object"}


async def test_write_back_is_idempotent():
    event = EventDefinition(
        id="pubsub-orders",
        topic="orders",
        subscription="projects/proj/subscriptions/aac-wf-step",
    )
    backend = FakeEventBackend({"pubsub-orders": event})
    container = _container(backend)

    await container._save_pubsub_subscription(
        PubSubTriggerSpec(topic="orders", project_id="proj"),
        "projects/proj/subscriptions/aac-wf-step",
    )

    assert (backend.created, backend.updated) == ([], [])


async def test_write_back_without_an_event_backend_is_a_no_op():
    container = _container(None)
    # Must not raise — a Mongo-less deployment still runs pubsub triggers.
    await container._save_pubsub_subscription(
        PubSubTriggerSpec(topic="orders", project_id="proj"), "projects/proj/subscriptions/s",
    )


async def test_event_ids_are_slugged_from_the_topic():
    assert _pubsub_source_id("projects/p/topics/Order.Events") == "pubsub-order-events"
    assert _pubsub_source_id("orders") == "pubsub-orders"
    assert _pubsub_source_id("") == "pubsub-topic"


# ─── syncing registrations with the definition ─────────────────────────────────

from types import SimpleNamespace  # noqa: E402 - test-local helper import

from app.infrastructure.triggers.pubsub_subscriber import PubSubSubscriberManager  # noqa: E402
from tests.unit.test_pubsub_subscriber import FakeSubscriber  # noqa: E402


def _runner_with(steps: list[dict], workflow_id: str = "orders-wf") -> SimpleNamespace:
    return SimpleNamespace(id=workflow_id, steps=steps)


def _with_subscriber(container: ApplicationContainer, client: FakeSubscriber) -> PubSubSubscriberManager:
    manager = PubSubSubscriberManager(client_factory=lambda: client)
    manager.start()
    container.pubsub_subscriber = manager
    return manager


async def test_removing_the_pubsub_node_releases_its_subscription():
    client = FakeSubscriber()
    container = _container()
    manager = _with_subscriber(container, client)
    step = {"id": "on_order", "type": "pubsub", "topic": "orders"}

    await container._register_pubsub_steps(_runner_with([step]))
    path = manager.registrations()["orders-wf:on_order"]

    # Saved again with the trigger node deleted.
    await container._register_pubsub_steps(_runner_with([{"id": "work", "type": "llm"}]))

    assert manager.registrations() == {}
    assert manager._streams == {}


async def test_saving_an_unchanged_workflow_keeps_the_stream():
    client = FakeSubscriber()
    container = _container()
    manager = _with_subscriber(container, client)
    step = {"id": "on_order", "type": "pubsub", "topic": "orders"}

    await container._register_pubsub_steps(_runner_with([step]))
    path = manager.registrations()["orders-wf:on_order"]
    future = manager._streams[path].future

    await container._register_pubsub_steps(_runner_with([step]))

    assert manager._streams[path].future is future
    assert client.subscribed == [path]


async def test_two_workflows_on_one_event_share_the_subscription():
    event = EventDefinition(
        id="orders-events",
        topic="orders",
        subscription="projects/proj/subscriptions/shared",
    )
    client = FakeSubscriber(existing={"projects/proj/subscriptions/shared"})
    container = _container(FakeEventBackend({"orders-events": event}))
    manager = _with_subscriber(container, client)
    step = {"id": "on_order", "type": "pubsub", "event": "orders-events"}

    await container._register_pubsub_steps(_runner_with([step], "wf-a"))
    await container._register_pubsub_steps(_runner_with([step], "wf-b"))

    path = "projects/proj/subscriptions/shared"
    # One stream feeding both workflows, so both see every event.
    assert client.subscribed == [path]
    assert manager.consumers_of(path) == ["wf-a:on_order", "wf-b:on_order"]


async def test_a_broken_step_leaves_the_other_triggers_registered():
    client = FakeSubscriber()
    container = _container(FakeEventBackend())
    manager = _with_subscriber(container, client)

    await container._register_pubsub_steps(_runner_with([
        {"id": "on_order", "type": "pubsub", "topic": "orders"},
        {"id": "broken", "type": "pubsub", "event": "ghost"},
    ]))

    assert list(manager.registrations()) == ["orders-wf:on_order"]


async def test_deleting_the_workflow_releases_its_subscriptions():
    from unittest.mock import AsyncMock

    client = FakeSubscriber()
    container = _container()
    manager = _with_subscriber(container, client)
    await container._register_pubsub_steps(
        _runner_with([{"id": "on_order", "type": "pubsub", "topic": "orders"}])
    )
    path = manager.registrations()["orders-wf:on_order"]

    # refresh_runner is what every delete path funnels through; no definition
    # left means the workflow is gone.
    container.workflow_backend = AsyncMock()
    container.workflow_backend.get = AsyncMock(return_value=None)
    await container.refresh_runner("orders-wf")

    assert manager.registrations() == {}
    assert manager._streams == {}
