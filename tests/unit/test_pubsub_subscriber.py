"""Pub/Sub trigger subscriptions.

The Google client library is replaced by a fake subscriber: it records admin
calls, hands back the streaming-pull callback so tests can deliver messages the
way the library would (from another thread), and records ack/nack per message.
"""
from __future__ import annotations

import asyncio
import json
import threading

import pytest

from app.infrastructure.triggers.event_schema import validate_event_payload
from app.infrastructure.triggers.pubsub_subscriber import (
    PubSubSubscriberManager,
    PubSubTriggerSpec,
)


class NotFound(Exception):
    """Stands in for google.api_core.exceptions.NotFound."""

    code = 404


class FakeMessage:
    def __init__(self, payload, message_id: str = "m-1", attributes: dict | None = None) -> None:
        self.data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode()
        self.message_id = message_id
        self.publish_time = None
        self.attributes = attributes or {}
        self.acked = False
        self.nacked = False

    def ack(self) -> None:
        self.acked = True

    def nack(self) -> None:
        self.nacked = True


class FakeFuture:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeSubscriber:
    """Minimal stand-in for pubsub_v1.SubscriberClient."""

    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = existing or set()
        self.created: list[dict] = []
        self.subscribed: list[str] = []
        self.callbacks: dict[str, callable] = {}
        self.closed = False

    def get_subscription(self, request):  # noqa: ANN001 - mirrors the library
        if request["subscription"] not in self.existing:
            raise NotFound(f"Subscription not found: {request['subscription']}")
        return {"name": request["subscription"]}

    def create_subscription(self, request):  # noqa: ANN001
        self.created.append(request)
        self.existing.add(request["name"])
        return request

    def subscribe(self, subscription, callback, flow_control=None):  # noqa: ANN001
        self.subscribed.append(subscription)
        self.callbacks[subscription] = callback
        return FakeFuture()

    def close(self) -> None:
        self.closed = True


def _manager(client: FakeSubscriber, **kwargs) -> PubSubSubscriberManager:
    return PubSubSubscriberManager(client_factory=lambda: client, **kwargs)


async def _deliver(client: FakeSubscriber, subscription: str, message: FakeMessage) -> None:
    """Push a message the way the library does: from a non-loop thread."""
    handler = client.callbacks[subscription]
    thread = threading.Thread(target=handler, args=(message,))
    thread.start()
    while thread.is_alive():
        # Keep the loop running so run_coroutine_threadsafe can complete.
        await asyncio.sleep(0.01)
    thread.join()


# ─── spec resolution ──────────────────────────────────────────────────────────

def test_short_names_resolve_against_the_configured_project():
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")
    assert spec.topic_path() == "projects/proj/topics/orders"
    assert spec.subscription_path("aac-wf-step") == "projects/proj/subscriptions/aac-wf-step"


def test_full_paths_are_left_alone():
    spec = PubSubTriggerSpec(
        topic="projects/other/topics/orders",
        project_id="proj",
        subscription="projects/other/subscriptions/mine",
    )
    assert spec.topic_path() == "projects/other/topics/orders"
    assert spec.subscription_path("ignored") == "projects/other/subscriptions/mine"


def test_a_short_topic_without_a_project_is_rejected():
    with pytest.raises(ValueError, match="short name"):
        PubSubTriggerSpec(topic="orders", project_id="").topic_path()


# ─── subscription lifecycle ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_creates_a_missing_subscription_and_reports_it():
    client = FakeSubscriber()
    created: list[tuple[str, str]] = []

    async def on_created(spec, path):
        created.append((spec.topic, path))

    manager = _manager(client, on_subscription_created=on_created)
    manager.start()

    path = await manager.register(
        "orders-wf", "on_order",
        PubSubTriggerSpec(topic="orders", project_id="proj"),
        lambda payload, meta: asyncio.sleep(0),
    )

    assert path == "projects/proj/subscriptions/aac-orders-wf-on_order"
    assert client.created[0]["topic"] == "projects/proj/topics/orders"
    assert client.subscribed == [path]
    # The write-back hook fires only for subscriptions created from scratch.
    assert created == [("orders", path)]


@pytest.mark.asyncio
async def test_an_existing_subscription_is_reused_and_not_reported():
    client = FakeSubscriber(existing={"projects/proj/subscriptions/mine"})
    created: list[str] = []

    async def on_created(spec, path):
        created.append(path)

    manager = _manager(client, on_subscription_created=on_created)
    manager.start()

    path = await manager.register(
        "wf", "step",
        PubSubTriggerSpec(topic="orders", project_id="proj", subscription="mine"),
        lambda payload, meta: asyncio.sleep(0),
    )

    assert path == "projects/proj/subscriptions/mine"
    assert client.created == []
    assert created == []


@pytest.mark.asyncio
async def test_re_registering_a_step_replaces_its_stream():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")

    await manager.register("wf", "step", spec, lambda p, m: asyncio.sleep(0))
    first = manager._registrations["wf:step"].future
    await manager.register("wf", "step", spec, lambda p, m: asyncio.sleep(0))

    assert first.cancelled is True
    assert len(manager.registrations()) == 1


@pytest.mark.asyncio
async def test_unregister_workflow_only_drops_that_workflow():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")

    await manager.register("wf-a", "s1", spec, lambda p, m: asyncio.sleep(0))
    await manager.register("wf-a", "s2", spec, lambda p, m: asyncio.sleep(0))
    await manager.register("wf-b", "s1", spec, lambda p, m: asyncio.sleep(0))

    manager.unregister_workflow("wf-a")

    assert list(manager.registrations()) == ["wf-b:s1"]


# ─── message dispatch ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_matching_event_starts_a_run_and_is_acked():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    seen: list[tuple[dict, dict]] = []

    async def callback(payload, meta):
        seen.append((payload, meta))

    path = await manager.register(
        "wf", "step",
        PubSubTriggerSpec(
            topic="orders",
            project_id="proj",
            event_schema={"type": "object", "required": ["order_id"]},
        ),
        callback,
    )
    message = FakeMessage({"order_id": "A-1"}, message_id="m-42", attributes={"src": "shop"})
    await _deliver(client, path, message)

    assert message.acked is True
    payload, meta = seen[0]
    assert payload == {"order_id": "A-1"}
    assert meta["type"] == "pubsub"
    assert meta["topic"] == "orders"
    assert meta["subscription"] == path
    assert meta["message_id"] == "m-42"
    assert meta["attributes"] == {"src": "shop"}


@pytest.mark.asyncio
async def test_a_schema_mismatch_never_starts_a_run_and_is_dropped():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    started: list[dict] = []

    path = await manager.register(
        "wf", "step",
        PubSubTriggerSpec(
            topic="orders",
            project_id="proj",
            event_schema={"type": "object", "required": ["order_id"]},
        ),
        lambda payload, meta: started.append(payload) or asyncio.sleep(0),
    )
    message = FakeMessage({"nope": 1})
    await _deliver(client, path, message)

    assert started == []
    # Dropped, not redelivered forever.
    assert (message.acked, message.nacked) == (True, False)


@pytest.mark.asyncio
async def test_a_schema_mismatch_is_nacked_when_dropping_is_disabled():
    client = FakeSubscriber()
    manager = _manager(client, drop_invalid_messages=False)
    manager.start()

    path = await manager.register(
        "wf", "step",
        PubSubTriggerSpec(topic="orders", project_id="proj", event_schema={"type": "object", "required": ["id"]}),
        lambda payload, meta: asyncio.sleep(0),
    )
    message = FakeMessage({})
    await _deliver(client, path, message)

    assert (message.acked, message.nacked) == (False, True)


@pytest.mark.asyncio
async def test_a_failing_run_start_nacks_so_the_event_is_redelivered():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()

    async def failing(payload, meta):
        raise RuntimeError("mongo down")

    path = await manager.register(
        "wf", "step", PubSubTriggerSpec(topic="orders", project_id="proj"), failing,
    )
    message = FakeMessage({"order_id": "A-1"})
    await _deliver(client, path, message)

    assert (message.acked, message.nacked) == (False, True)


@pytest.mark.asyncio
async def test_a_non_json_body_arrives_as_raw_text():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    seen: list[dict] = []

    async def callback(payload, meta):
        seen.append(payload)

    path = await manager.register(
        "wf", "step", PubSubTriggerSpec(topic="orders", project_id="proj"), callback,
    )
    await _deliver(client, path, FakeMessage(b"not json at all"))

    assert seen == [{"raw": "not json at all"}]


@pytest.mark.asyncio
async def test_stop_cancels_every_stream_and_closes_the_client():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")
    await manager.register("wf", "step", spec, lambda p, m: asyncio.sleep(0))
    future = manager._registrations["wf:step"].future

    manager.stop()

    assert future.cancelled is True
    assert manager.registrations() == {}
    assert client.closed is True


# ─── schema check ─────────────────────────────────────────────────────────────

def test_no_schema_accepts_anything():
    validate_event_payload({"anything": True}, None, "label")
    validate_event_payload([1, 2, 3], {}, "label")


def test_schema_checks_type_required_keys_and_property_types():
    schema = {
        "type": "object",
        "required": ["id"],
        "properties": {"id": {"type": "string"}, "count": {"type": "number"}},
    }
    validate_event_payload({"id": "x", "count": 2}, schema, "label")
    # A null property value is not a type violation.
    validate_event_payload({"id": "x", "count": None}, schema, "label")

    with pytest.raises(ValueError, match="missing required key 'id'"):
        validate_event_payload({"count": 1}, schema, "label")
    with pytest.raises(ValueError, match="key 'id' is int, expected string"):
        validate_event_payload({"id": 5}, schema, "label")
    with pytest.raises(ValueError, match="payload is list, expected object"):
        validate_event_payload([], schema, "label")
