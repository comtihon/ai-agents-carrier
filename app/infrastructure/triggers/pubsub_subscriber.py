"""Google Cloud Pub/Sub trigger subscriptions.

Mirrors ``CronScheduler``: the container registers one entry per ``pubsub``
trigger step, keyed ``"workflow_id:step_id"``, so a workflow update replaces
only its own entries.

Two consequences of Pub/Sub's delivery model shape this module:

*Every consumer of one subscription competes for its messages.*  Two workflow
steps that name the same subscription must therefore not open two streaming
pulls — Pub/Sub would hand each event to one of them.  Streams are keyed by
subscription path instead, with a set of consumers behind them, and one arriving
event starts a run for **every** consumer.  Steps that do not name a
subscription each get their own (``{prefix}{workflow}-{step}``), which is
Pub/Sub's own fan-out and needs nothing special.

*A subscription outlives the process.*  Cancelling a streaming pull stops
delivery but leaves the subscription accruing a backlog, so the last consumer
of a subscription the backend created also deletes it.  Shutdown never does:
messages published while the backend is down must still be there on restart.

Threading: the client library invokes callbacks on its own threads while the
runs they start are asyncio work, so every message hops onto the loop captured
at ``start()``.  Acknowledgement happens only after the runs exist — a crash
mid-dispatch redelivers rather than dropping the event.

The client library is imported lazily so a backend without GCP credentials
(tests, local runs, ``PUBSUB_ENABLED=false``) never needs it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.infrastructure.triggers.event_schema import validate_event_payload

logger = logging.getLogger(__name__)

# (payload, event_meta) -> awaitable.  Awaited before the message is acked.
EventCallback = Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]]
# Called with (spec, subscription_path) after the manager creates a
# subscription that did not exist yet — used to save it back to datasources.
SubscriptionCreatedHook = Callable[["PubSubTriggerSpec", str], Awaitable[None]]


@dataclass
class PubSubTriggerSpec:
    """Everything one ``pubsub`` trigger step needs to subscribe.

    ``topic`` and ``subscription`` accept either a short name or a fully
    qualified path; short names resolve against ``project_id``.
    """

    topic: str
    project_id: str
    subscription: str = ""
    event_schema: dict[str, Any] | None = None
    ack_deadline_seconds: int = 60
    max_messages: int = 10
    # Identifies the datasource this spec came from, when it came from one.
    # Carried so the write-back hook knows whether to create or update.
    datasource_id: str = ""

    def topic_path(self) -> str:
        if self.topic.startswith("projects/"):
            return self.topic
        if not self.project_id:
            raise ValueError(
                f"Pub/Sub topic '{self.topic}' is a short name and no project is "
                "configured — set PUBSUB_PROJECT_ID or use a full "
                "projects/<p>/topics/<t> path"
            )
        return f"projects/{self.project_id}/topics/{self.topic}"

    def subscription_path(self, default_name: str) -> str:
        name = self.subscription or default_name
        if name.startswith("projects/"):
            return name
        project = self.project_id or self.topic_path().split("/")[1]
        return f"projects/{project}/subscriptions/{name}"


@dataclass
class _Consumer:
    """One workflow step listening on a subscription."""

    key: str
    label: str
    spec: PubSubTriggerSpec
    callback: EventCallback


@dataclass
class _Stream:
    """One streaming pull, shared by every consumer of that subscription."""

    subscription_path: str
    topic_path: str
    created_by_us: bool
    future: Any  # StreamingPullFuture
    consumers: dict[str, _Consumer] = field(default_factory=dict)


@dataclass
class PubSubSubscriberManager:
    """Owns one streaming pull per subscription, shared across trigger steps."""

    subscription_prefix: str = "aac-"
    drop_invalid_messages: bool = True
    # Delete a subscription the manager created once its last consumer is gone
    # (a trigger step removed, a workflow deleted).  Subscriptions named by a
    # step or datasource are never deleted — they are not ours.
    delete_orphaned_subscriptions: bool = True
    # Injected in tests; None means "use google.cloud.pubsub_v1.SubscriberClient".
    client_factory: Callable[[], Any] | None = None
    on_subscription_created: SubscriptionCreatedHook | None = None

    _client: Any = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    # subscription path -> stream
    _streams: dict[str, _Stream] = field(default_factory=dict, init=False)
    # "workflow:step" -> subscription path
    _keys: dict[str, str] = field(default_factory=dict, init=False)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Capture the loop messages must be dispatched onto."""
        self._loop = asyncio.get_running_loop()
        logger.info("Pub/Sub subscriber manager started")

    def stop(self) -> None:
        """Stop pulling, keeping every subscription so a restart resumes it."""
        for path in list(self._streams):
            self._close_stream(path, delete=False)
        self._keys.clear()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("Pub/Sub client close failed", exc_info=True)
            self._client = None
        logger.info("Pub/Sub subscriber manager stopped")

    # ── registration ─────────────────────────────────────────────────────────

    async def register(
        self,
        workflow_id: str,
        step_id: str,
        spec: PubSubTriggerSpec,
        callback: EventCallback,
    ) -> str:
        """Subscribe one workflow step, returning the subscription path in use.

        Re-registering the same step is idempotent in the way that matters: the
        stream is kept and only the callback and spec are replaced, so saving a
        workflow does not interrupt delivery.  Failures are logged and re-raised
        so the caller can decide what a broken trigger means for the workflow.
        """
        key = f"{workflow_id}:{step_id}"
        label = f"workflow '{workflow_id}' step '{step_id}'"

        client = self._ensure_client()
        topic_path = spec.topic_path()
        default_name = f"{self.subscription_prefix}{workflow_id}-{step_id}"
        subscription_path = spec.subscription_path(default_name)

        previous_path = self._keys.get(key)
        if previous_path is not None and previous_path != subscription_path:
            # The step now points somewhere else — release the old subscription.
            self._detach(key)

        stream = self._streams.get(subscription_path)
        created = False
        if stream is None:
            created = await asyncio.to_thread(
                self._ensure_subscription, client, subscription_path, topic_path, spec
            )
            future = client.subscribe(
                subscription_path,
                callback=self._make_message_handler(subscription_path),
                flow_control=self._flow_control(spec.max_messages),
            )
            stream = _Stream(
                subscription_path=subscription_path,
                topic_path=topic_path,
                created_by_us=created,
                future=future,
            )
            self._streams[subscription_path] = stream
        elif stream.topic_path != topic_path:
            # A subscription is bound to one topic at creation; the step's topic
            # is decoration at this point, and a mismatch means the workflow is
            # misconfigured.
            logger.warning(
                "%s asks for topic %s but subscription %s is attached to %s — "
                "events will arrive from %s",
                label, topic_path, subscription_path, stream.topic_path, stream.topic_path,
            )

        stream.consumers[key] = _Consumer(key=key, label=label, spec=spec, callback=callback)
        self._keys[key] = subscription_path
        logger.info(
            "Pub/Sub trigger registered: %s topic=%s subscription=%s%s (%d consumer(s))",
            label, topic_path, subscription_path,
            " (created)" if created else "", len(stream.consumers),
        )

        if created and self.on_subscription_created is not None:
            # Newly created from scratch — persist it so the same subscription
            # can be picked from the datasources list next time.
            try:
                await self.on_subscription_created(spec, subscription_path)
            except Exception:
                logger.exception("Saving Pub/Sub subscription back to datasources failed: %s", label)
        return subscription_path

    async def sync_workflow(
        self,
        workflow_id: str,
        entries: list[tuple[str, PubSubTriggerSpec, EventCallback]],
    ) -> dict[str, str]:
        """Make *workflow_id*'s registrations exactly *entries*.

        Steps in *entries* are registered (or refreshed in place); registrations
        of that workflow which *entries* does not mention — a ``pubsub`` node the
        user deleted, or every node of a deleted workflow — are released, and any
        subscription left without consumers is torn down.

        A step whose registration raises keeps whatever registration it already
        had: a transient GCP error must not silently unsubscribe a live trigger.
        """
        attempted: set[str] = set()
        registered: dict[str, str] = {}
        for step_id, spec, callback in entries:
            key = f"{workflow_id}:{step_id}"
            attempted.add(key)
            try:
                registered[key] = await self.register(workflow_id, step_id, spec, callback)
            except Exception:
                logger.exception(
                    "Subscribing Pub/Sub step '%s' of workflow '%s' failed", step_id, workflow_id,
                )

        prefix = f"{workflow_id}:"
        for key in [k for k in list(self._keys) if k.startswith(prefix) and k not in attempted]:
            logger.info("Pub/Sub trigger no longer in workflow '%s': releasing %s", workflow_id, key)
            self._detach(key)
        return registered

    def unregister(self, workflow_id: str, step_id: str) -> None:
        self._detach(f"{workflow_id}:{step_id}")

    def unregister_workflow(self, workflow_id: str) -> None:
        for key in [k for k in list(self._keys) if k.startswith(f"{workflow_id}:")]:
            self._detach(key)

    def registrations(self) -> dict[str, str]:
        """``{"workflow:step": subscription_path}`` — for status endpoints/tests."""
        return dict(self._keys)

    def consumers_of(self, subscription_path: str) -> list[str]:
        """Keys sharing one subscription — for status endpoints/tests."""
        stream = self._streams.get(subscription_path)
        return sorted(stream.consumers) if stream else []

    # ── internals ────────────────────────────────────────────────────────────

    def _ensure_client(self) -> Any:
        if self._client is None:
            if self.client_factory is not None:
                self._client = self.client_factory()
            else:
                from google.cloud import pubsub_v1  # imported lazily on purpose

                self._client = pubsub_v1.SubscriberClient()
        return self._client

    @staticmethod
    def _flow_control(max_messages: int) -> Any:
        try:
            from google.cloud import pubsub_v1

            return pubsub_v1.types.FlowControl(max_messages=max_messages)
        except Exception:
            # No client library (tests with a fake factory) — the fake decides
            # what to do with whatever it gets.
            return {"max_messages": max_messages}

    def _ensure_subscription(
        self, client: Any, subscription_path: str, topic_path: str, spec: PubSubTriggerSpec
    ) -> bool:
        """Create the subscription when missing.  Returns True when created.

        Runs in a worker thread — the client library's admin calls are blocking.
        """
        try:
            client.get_subscription(request={"subscription": subscription_path})
            return False
        except Exception as exc:
            if not _is_not_found(exc):
                raise
        client.create_subscription(
            request={
                "name": subscription_path,
                "topic": topic_path,
                "ack_deadline_seconds": spec.ack_deadline_seconds,
            }
        )
        return True

    def _make_message_handler(self, subscription_path: str) -> Callable[[Any], None]:
        """Build the streaming-pull callback for one subscription.

        The consumer set is read at delivery time, so steps registered or
        released later are picked up without touching the stream.
        """

        def handle(message: Any) -> None:
            # Runs on a client-library thread.
            stream = self._streams.get(subscription_path)
            consumers = list(stream.consumers.values()) if stream else []
            if not consumers:
                # Stream cancelled between delivery and dispatch — let the next
                # owner of this subscription have the message.
                message.nack()
                return

            try:
                payload = _decode(message.data)
            except Exception:
                logger.warning(
                    "%s: undecodable Pub/Sub message %s",
                    subscription_path, getattr(message, "message_id", "?"),
                )
                self._reject(message)
                return

            loop = self._loop
            if loop is None:
                logger.error(
                    "%s: event dropped — subscriber manager was never started", subscription_path,
                )
                self._reject(message)
                return

            accepted = 0
            failed = 0
            for consumer in consumers:
                try:
                    validate_event_payload(payload, consumer.spec.event_schema, consumer.label)
                except ValueError as exc:
                    logger.warning("%s", exc)
                    continue
                accepted += 1
                meta = _event_meta(consumer.spec, subscription_path, message)
                try:
                    asyncio.run_coroutine_threadsafe(consumer.callback(payload, meta), loop).result()
                except Exception:
                    failed += 1
                    logger.exception("%s: starting the workflow run failed", consumer.label)

            if accepted == 0:
                # Nobody's schema matched: this event is not for this workflow set.
                self._reject(message)
            elif failed == accepted:
                # Nothing started — redeliver.
                message.nack()
            else:
                # Some runs started. Redelivering would duplicate those, which is
                # worse than the failed consumers missing one event (they are
                # logged above).
                message.ack()

        return handle

    def _reject(self, message: Any) -> None:
        """Ack (drop) or nack an unusable message, per configuration."""
        if self.drop_invalid_messages:
            message.ack()
        else:
            message.nack()

    def _detach(self, key: str) -> None:
        """Release one step's registration, tearing the stream down if it was the last."""
        subscription_path = self._keys.pop(key, None)
        if subscription_path is None:
            return
        stream = self._streams.get(subscription_path)
        if stream is None:
            return
        consumer = stream.consumers.pop(key, None)
        if consumer is not None:
            logger.info("Pub/Sub trigger unregistered: %s", consumer.label)
        if not stream.consumers:
            self._close_stream(subscription_path, delete=True)
        else:
            logger.info(
                "Subscription %s still has %d consumer(s): %s",
                subscription_path, len(stream.consumers), ", ".join(sorted(stream.consumers)),
            )

    def _close_stream(self, subscription_path: str, *, delete: bool) -> None:
        stream = self._streams.pop(subscription_path, None)
        if stream is None:
            return
        try:
            stream.future.cancel()
        except Exception:
            logger.debug("Cancelling Pub/Sub streaming pull failed: %s", subscription_path, exc_info=True)
        logger.info("Pub/Sub streaming pull stopped: %s", subscription_path)
        if delete and stream.created_by_us and self.delete_orphaned_subscriptions:
            self._delete_subscription_async(subscription_path)

    def _delete_subscription_async(self, subscription_path: str) -> None:
        """Delete a subscription we created, off the caller's thread.

        Unregistration happens from both sync and async call sites, and the
        admin call is blocking; a daemon thread keeps both callers free and a
        failed delete only leaves an idle subscription behind.
        """
        client = self._client
        if client is None:
            return

        def run() -> None:
            try:
                client.delete_subscription(request={"subscription": subscription_path})
                logger.info("Pub/Sub subscription deleted (no consumers left): %s", subscription_path)
            except Exception as exc:
                if _is_not_found(exc):
                    return
                logger.warning(
                    "Deleting orphaned Pub/Sub subscription %s failed: %s", subscription_path, exc,
                )

        threading.Thread(target=run, name="pubsub-delete-subscription", daemon=True).start()


def _event_meta(spec: PubSubTriggerSpec, subscription_path: str, message: Any) -> dict[str, Any]:
    return {
        "type": "pubsub",
        "topic": spec.topic,
        "subscription": subscription_path,
        "message_id": getattr(message, "message_id", ""),
        "publish_time": _isoformat(getattr(message, "publish_time", None)),
        "attributes": dict(getattr(message, "attributes", {}) or {}),
    }


def _decode(data: Any) -> dict[str, Any]:
    """Message body as a dict.  Non-JSON bodies become ``{"raw": "..."}``."""
    if isinstance(data, (bytes, bytearray)):
        text = bytes(data).decode("utf-8", errors="replace")
    else:
        text = str(data)
    try:
        parsed = json.loads(text)
    except Exception:
        return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def _isoformat(value: Any) -> str:
    if value is None:
        return ""
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _is_not_found(exc: Exception) -> bool:
    """True when *exc* is the client library's 404 for a missing subscription."""
    try:
        from google.api_core.exceptions import NotFound

        if isinstance(exc, NotFound):
            return True
    except Exception:
        pass
    # Fakes and transport-level variants: fall back to the status code / text.
    if getattr(exc, "code", None) == 404:
        return True
    return "not found" in str(exc).lower() or "notfound" in type(exc).__name__.lower()
