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
        self.deleted: list[str] = []
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

    def delete_subscription(self, request):  # noqa: ANN001
        if request["subscription"] not in self.existing:
            raise NotFound(f"Subscription not found: {request['subscription']}")
        self.existing.discard(request["subscription"])
        self.deleted.append(request["subscription"])

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
async def test_re_registering_a_step_keeps_its_stream_alive():
    """Saving a workflow must not interrupt delivery for its own triggers."""
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")

    path = await manager.register("wf", "step", spec, lambda p, m: asyncio.sleep(0))
    first = manager._streams[path].future
    await manager.register("wf", "step", spec, lambda p, m: asyncio.sleep(0))

    assert first.cancelled is False
    assert client.subscribed == [path]
    assert client.deleted == []
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
async def test_stop_cancels_every_stream_but_keeps_the_subscriptions():
    """Shutdown must not delete: events published while down have to survive."""
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")
    path = await manager.register("wf", "step", spec, lambda p, m: asyncio.sleep(0))
    future = manager._streams[path].future

    manager.stop()

    assert future.cancelled is True
    assert manager.registrations() == {}
    assert client.closed is True
    assert client.deleted == []
    assert path in client.existing


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


# ─── several workflows on one topic ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_two_steps_sharing_a_subscription_both_get_every_event():
    """One subscription, two consumers: Pub/Sub would give the event to one of
    them if we opened two streams, so the manager opens one and fans out."""
    client = FakeSubscriber(existing={"projects/proj/subscriptions/shared"})
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj", subscription="shared")
    started: list[str] = []

    async def cb_a(payload, meta):
        started.append("a")

    async def cb_b(payload, meta):
        started.append("b")

    path = await manager.register("wf-a", "on_order", spec, cb_a)
    assert await manager.register("wf-b", "on_order", spec, cb_b) == path

    # One stream, two consumers.
    assert client.subscribed == [path]
    assert manager.consumers_of(path) == ["wf-a:on_order", "wf-b:on_order"]

    message = FakeMessage({"order_id": "A-1"})
    await _deliver(client, path, message)

    assert sorted(started) == ["a", "b"]
    assert message.acked is True


@pytest.mark.asyncio
async def test_separate_default_subscriptions_are_separate_streams():
    """Without a named subscription each step gets its own, which is Pub/Sub's
    own fan-out — every workflow still sees every event."""
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")

    path_a = await manager.register("wf-a", "on_order", spec, lambda p, m: asyncio.sleep(0))
    path_b = await manager.register("wf-b", "on_order", spec, lambda p, m: asyncio.sleep(0))

    assert path_a != path_b
    assert sorted(client.subscribed) == sorted([path_a, path_b])
    assert [c["topic"] for c in client.created] == ["projects/proj/topics/orders"] * 2


@pytest.mark.asyncio
async def test_a_shared_subscription_applies_each_consumers_own_schema():
    client = FakeSubscriber(existing={"projects/proj/subscriptions/shared"})
    manager = _manager(client)
    manager.start()
    started: list[str] = []

    def spec_with(schema):
        return PubSubTriggerSpec(
            topic="orders", project_id="proj", subscription="shared", event_schema=schema,
        )

    path = await manager.register(
        "wf-orders", "on_order", spec_with({"type": "object", "required": ["order_id"]}),
        lambda p, m: _record(started, "orders"),
    )
    await manager.register(
        "wf-refunds", "on_refund", spec_with({"type": "object", "required": ["refund_id"]}),
        lambda p, m: _record(started, "refunds"),
    )

    message = FakeMessage({"order_id": "A-1"})
    await _deliver(client, path, message)

    # Only the workflow whose schema matches runs; the event is still acked.
    assert started == ["orders"]
    assert message.acked is True


@pytest.mark.asyncio
async def test_one_failing_consumer_does_not_redeliver_to_the_others():
    client = FakeSubscriber(existing={"projects/proj/subscriptions/shared"})
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj", subscription="shared")
    started: list[str] = []

    async def ok(payload, meta):
        started.append("ok")

    async def boom(payload, meta):
        raise RuntimeError("mongo down")

    path = await manager.register("wf-a", "s", spec, ok)
    await manager.register("wf-b", "s", spec, boom)

    message = FakeMessage({"order_id": "A-1"})
    await _deliver(client, path, message)

    # Acked: a redelivery would run wf-a twice, which is worse than wf-b missing
    # one event (logged).
    assert started == ["ok"]
    assert (message.acked, message.nacked) == (True, False)


@pytest.mark.asyncio
async def test_a_shared_subscription_nacks_when_no_consumer_could_start():
    client = FakeSubscriber(existing={"projects/proj/subscriptions/shared"})
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj", subscription="shared")

    async def boom(payload, meta):
        raise RuntimeError("mongo down")

    path = await manager.register("wf-a", "s", spec, boom)
    await manager.register("wf-b", "s", spec, boom)

    message = FakeMessage({"order_id": "A-1"})
    await _deliver(client, path, message)

    assert (message.acked, message.nacked) == (False, True)


async def _record(sink: list[str], name: str) -> None:
    sink.append(name)


# ─── releasing subscriptions ──────────────────────────────────────────────────

def _wait_for(predicate, timeout: float = 2.0) -> bool:
    """Wait for the daemon thread that deletes a subscription."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.mark.asyncio
async def test_the_last_consumer_leaving_stops_and_deletes_our_subscription():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")
    path = await manager.register("wf", "on_order", spec, lambda p, m: asyncio.sleep(0))
    future = manager._streams[path].future

    manager.unregister("wf", "on_order")

    assert future.cancelled is True
    assert manager.registrations() == {}
    assert _wait_for(lambda: client.deleted == [path]), client.deleted


@pytest.mark.asyncio
async def test_a_subscription_survives_while_another_workflow_still_uses_it():
    client = FakeSubscriber(existing={"projects/proj/subscriptions/shared"})
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj", subscription="shared")
    started: list[str] = []

    path = await manager.register("wf-a", "s", spec, lambda p, m: _record(started, "a"))
    await manager.register("wf-b", "s", spec, lambda p, m: _record(started, "b"))

    manager.unregister("wf-a", "s")

    assert manager._streams[path].future.cancelled is False
    assert manager.consumers_of(path) == ["wf-b:s"]
    # wf-b keeps receiving.
    await _deliver(client, path, FakeMessage({"order_id": "A-1"}))
    assert started == ["b"]


@pytest.mark.asyncio
async def test_a_subscription_we_did_not_create_is_never_deleted():
    client = FakeSubscriber(existing={"projects/proj/subscriptions/mine"})
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj", subscription="mine")
    await manager.register("wf", "s", spec, lambda p, m: asyncio.sleep(0))

    manager.unregister("wf", "s")

    assert manager.registrations() == {}
    assert _wait_for(lambda: client.deleted != [], timeout=0.3) is False
    assert "projects/proj/subscriptions/mine" in client.existing


@pytest.mark.asyncio
async def test_deletion_can_be_turned_off():
    client = FakeSubscriber()
    manager = _manager(client, delete_orphaned_subscriptions=False)
    manager.start()
    path = await manager.register(
        "wf", "s", PubSubTriggerSpec(topic="orders", project_id="proj"), lambda p, m: asyncio.sleep(0),
    )

    manager.unregister("wf", "s")

    assert _wait_for(lambda: client.deleted != [], timeout=0.3) is False
    assert path in client.existing


# ─── syncing a workflow's steps ───────────────────────────────────────────────

def _entry(step_id: str, spec: PubSubTriggerSpec, sink: list[str]):
    return (step_id, spec, lambda p, m: _record(sink, step_id))


@pytest.mark.asyncio
async def test_sync_releases_a_step_the_definition_no_longer_has():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")
    sink: list[str] = []

    await manager.sync_workflow("wf", [_entry("on_order", spec, sink), _entry("on_refund", spec, sink)])
    assert sorted(manager.registrations()) == ["wf:on_order", "wf:on_refund"]
    refund_path = manager.registrations()["wf:on_refund"]

    # The user deleted the on_refund node and saved.
    await manager.sync_workflow("wf", [_entry("on_order", spec, sink)])

    assert list(manager.registrations()) == ["wf:on_order"]
    assert _wait_for(lambda: client.deleted == [refund_path]), client.deleted


@pytest.mark.asyncio
async def test_sync_does_not_disturb_the_steps_that_stayed():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")
    sink: list[str] = []

    await manager.sync_workflow("wf", [_entry("on_order", spec, sink)])
    path = manager.registrations()["wf:on_order"]
    future = manager._streams[path].future

    await manager.sync_workflow("wf", [_entry("on_order", spec, sink)])

    # Same stream, no re-subscribe, nothing deleted and recreated.
    assert manager._streams[path].future is future
    assert client.subscribed == [path]
    assert client.created == [client.created[0]]
    assert client.deleted == []


@pytest.mark.asyncio
async def test_sync_with_no_entries_releases_everything():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")
    sink: list[str] = []
    await manager.sync_workflow("wf", [_entry("on_order", spec, sink)])
    path = manager.registrations()["wf:on_order"]

    await manager.sync_workflow("wf", [])

    assert manager.registrations() == {}
    assert _wait_for(lambda: client.deleted == [path]), client.deleted


@pytest.mark.asyncio
async def test_a_step_that_moves_to_another_subscription_releases_the_old_one():
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    sink: list[str] = []

    old_path = await manager.register(
        "wf", "s", PubSubTriggerSpec(topic="orders", project_id="proj"), lambda p, m: asyncio.sleep(0),
    )
    new_path = await manager.register(
        "wf", "s",
        PubSubTriggerSpec(topic="orders", project_id="proj", subscription="shared"),
        lambda p, m: _record(sink, "s"),
    )

    assert new_path != old_path
    assert manager.registrations() == {"wf:s": new_path}
    assert _wait_for(lambda: client.deleted == [old_path]), client.deleted


@pytest.mark.asyncio
async def test_a_failed_registration_keeps_the_previous_one():
    """A transient GCP error on save must not unsubscribe a live trigger."""
    client = FakeSubscriber()
    manager = _manager(client)
    manager.start()
    spec = PubSubTriggerSpec(topic="orders", project_id="proj")
    sink: list[str] = []
    await manager.sync_workflow("wf", [_entry("on_order", spec, sink)])
    path = manager.registrations()["wf:on_order"]

    broken = PubSubTriggerSpec(topic="orders", project_id="")  # topic_path() raises
    await manager.sync_workflow("wf", [_entry("on_order", broken, sink)])

    assert manager.registrations() == {"wf:on_order": path}
    assert client.deleted == []
