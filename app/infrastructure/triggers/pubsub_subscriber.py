"""Google Cloud Pub/Sub trigger subscriptions.

Mirrors ``CronScheduler``: the container registers one entry per ``pubsub``
trigger step and gets a callback invoked whenever an event arrives, keyed by
``"workflow_id:step_id"`` so a workflow update can replace its own entries
without touching anyone else's.

Threading: ``google-cloud-pubsub``'s streaming pull runs its callback on a
library-owned thread, while the workflow run it has to start is asyncio work.
Every message therefore hops back onto the loop captured at ``start()`` via
``asyncio.run_coroutine_threadsafe``, and the message is only acknowledged
once that coroutine has finished creating the run.  A run that outlives the
ack deadline is fine — the ack covers accepting the event, not completing the
workflow.

The client library is imported lazily so that a backend built without GCP
credentials (tests, local runs, ``PUBSUB_ENABLED=false``) neither needs the
dependency at import time nor pays for it.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.infrastructure.triggers.event_schema import validate_event_payload

logger = logging.getLogger(__name__)

# (payload, event_meta) -> awaitable.  The manager awaits this before acking.
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
class _Registration:
    spec: PubSubTriggerSpec
    subscription_path: str
    future: Any  # StreamingPullFuture
    label: str


@dataclass
class PubSubSubscriberManager:
    """Owns one streaming pull per registered ``pubsub`` trigger step."""

    subscription_prefix: str = "aac-"
    drop_invalid_messages: bool = True
    # Injected in tests; None means "use google.cloud.pubsub_v1.SubscriberClient".
    client_factory: Callable[[], Any] | None = None
    on_subscription_created: SubscriptionCreatedHook | None = None

    _client: Any = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _registrations: dict[str, _Registration] = field(default_factory=dict, init=False)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Capture the loop messages must be dispatched onto."""
        self._loop = asyncio.get_running_loop()
        logger.info("Pub/Sub subscriber manager started")

    def stop(self) -> None:
        for key in list(self._registrations):
            self._cancel(key)
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
        """Subscribe to *spec*'s topic for one workflow step.

        Returns the subscription path in use.  Replaces any previous
        registration for the same ``workflow_id:step_id``.  Failures are logged
        and re-raised so the caller can decide whether a workflow with a broken
        trigger should still load.
        """
        key = f"{workflow_id}:{step_id}"
        label = f"workflow '{workflow_id}' step '{step_id}'"
        self._cancel(key)

        client = self._ensure_client()
        topic_path = spec.topic_path()
        default_name = f"{self.subscription_prefix}{workflow_id}-{step_id}"
        subscription_path = spec.subscription_path(default_name)

        created = await asyncio.to_thread(
            self._ensure_subscription, client, subscription_path, topic_path, spec
        )

        flow_control = self._flow_control(spec.max_messages)
        future = client.subscribe(
            subscription_path,
            callback=self._make_message_handler(label, spec, subscription_path, callback),
            flow_control=flow_control,
        )
        self._registrations[key] = _Registration(
            spec=spec, subscription_path=subscription_path, future=future, label=label,
        )
        logger.info(
            "Pub/Sub trigger registered: %s topic=%s subscription=%s%s",
            label, topic_path, subscription_path, " (created)" if created else "",
        )
        if created and self.on_subscription_created is not None:
            # Newly created from scratch — persist it so the same subscription
            # can be picked from the datasources list next time.
            try:
                await self.on_subscription_created(spec, subscription_path)
            except Exception:
                logger.exception("Saving Pub/Sub subscription back to datasources failed: %s", label)
        return subscription_path

    def unregister(self, workflow_id: str, step_id: str) -> None:
        self._cancel(f"{workflow_id}:{step_id}")

    def unregister_workflow(self, workflow_id: str) -> None:
        for key in [k for k in list(self._registrations) if k.startswith(f"{workflow_id}:")]:
            self._cancel(key)

    def registrations(self) -> dict[str, str]:
        """``{"workflow:step": subscription_path}`` — for status endpoints/tests."""
        return {k: r.subscription_path for k, r in self._registrations.items()}

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

    def _make_message_handler(
        self,
        label: str,
        spec: PubSubTriggerSpec,
        subscription_path: str,
        callback: EventCallback,
    ) -> Callable[[Any], None]:
        def handle(message: Any) -> None:
            # Runs on a client-library thread.
            try:
                payload = _decode(message.data)
            except Exception:
                logger.warning("%s: undecodable Pub/Sub message %s", label, getattr(message, "message_id", "?"))
                self._reject(message)
                return
            try:
                validate_event_payload(payload, spec.event_schema, label)
            except ValueError as exc:
                logger.warning("%s: %s", label, exc)
                self._reject(message)
                return

            meta = {
                "type": "pubsub",
                "topic": spec.topic,
                "subscription": subscription_path,
                "message_id": getattr(message, "message_id", ""),
                "publish_time": _isoformat(getattr(message, "publish_time", None)),
                "attributes": dict(getattr(message, "attributes", {}) or {}),
            }

            loop = self._loop
            if loop is None:
                logger.error("%s: event dropped — subscriber manager was never started", label)
                self._reject(message)
                return
            try:
                asyncio.run_coroutine_threadsafe(callback(payload, meta), loop).result()
            except Exception:
                logger.exception("%s: starting the workflow run failed", label)
                message.nack()
                return
            message.ack()

        return handle

    def _reject(self, message: Any) -> None:
        """Ack (drop) or nack an unusable message, per configuration."""
        if self.drop_invalid_messages:
            message.ack()
        else:
            message.nack()

    def _cancel(self, key: str) -> None:
        registration = self._registrations.pop(key, None)
        if registration is None:
            return
        try:
            registration.future.cancel()
        except Exception:
            logger.debug("Cancelling Pub/Sub streaming pull failed: %s", key, exc_info=True)
        logger.info("Pub/Sub trigger unregistered: %s", registration.label)


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
    if getattr(exc, "code", None) == 404 or getattr(exc, "grpc_status_code", None) is not None and "NOT_FOUND" in str(getattr(exc, "grpc_status_code")):
        return True
    return "not found" in str(exc).lower() or "notfound" in type(exc).__name__.lower()
