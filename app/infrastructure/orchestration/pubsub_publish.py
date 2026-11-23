"""The ``pubsub_publish`` workflow step: publish one message to a Pub/Sub topic.

Publishing is closed by default.  A topic is only reachable when it matches a
glob in ``PUBSUB_PUBLISH_ALLOWED_TOPICS`` (comma-separated, ``fnmatch``
syntax, checked against both the full ``projects/<p>/topics/<t>`` path and the
short topic name).  An empty setting disables the step type entirely, so a
workflow author can never reach a topic the operators did not list.

The client library is imported lazily, and one ``PublisherClient`` is shared
per process: it is thread-safe, and building one per message would open a new
gRPC channel every time.  ``publish().result()`` blocks, so it runs in a worker
thread to keep the event loop free.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import threading
from typing import Any, Callable

DEFAULT_TIMEOUT_SECONDS = 30.0

# Injected in tests; None means google.cloud.pubsub_v1.PublisherClient.
client_factory: Callable[[], Any] | None = None
_client: Any = None
_client_lock = threading.Lock()


def _get_client() -> Any:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if client_factory is not None:
                    _client = client_factory()
                else:
                    from google.cloud import pubsub_v1

                    _client = pubsub_v1.PublisherClient()
    return _client


def reset_client() -> None:
    """Drop the cached client (tests)."""
    global _client
    with _client_lock:
        _client = None


def parse_allowlist(raw: str | None) -> list[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def resolve_topic(topic: str, project_id: str | None) -> str:
    topic = topic.strip()
    if not topic:
        raise ValueError("pubsub_publish needs a non-empty 'topic'")
    if topic.startswith("projects/"):
        parts = topic.split("/")
        if len(parts) != 4 or parts[2] != "topics" or not parts[1] or not parts[3]:
            raise ValueError(f"invalid Pub/Sub topic path '{topic}' (expected projects/<p>/topics/<t>)")
        return topic
    if "/" in topic:
        raise ValueError(f"invalid Pub/Sub topic name '{topic}'")
    if not project_id:
        raise ValueError(
            f"Pub/Sub topic '{topic}' is a short name and no project is configured "
            "— set PUBSUB_PROJECT_ID, the step's project_id, or use a full "
            "projects/<p>/topics/<t> path"
        )
    return f"projects/{project_id}/topics/{topic}"


def check_allowed(topic_path: str, allowlist: list[str]) -> None:
    """Raise ``PermissionError`` unless *topic_path* matches the allowlist."""
    if not allowlist:
        raise PermissionError(
            "publishing to Pub/Sub is disabled on this backend "
            "(PUBSUB_PUBLISH_ALLOWED_TOPICS is empty)"
        )
    short = topic_path.rsplit("/", 1)[-1]
    if any(fnmatch.fnmatchcase(topic_path, p) or fnmatch.fnmatchcase(short, p) for p in allowlist):
        return
    raise PermissionError(
        f"Pub/Sub topic '{topic_path}' is not in PUBSUB_PUBLISH_ALLOWED_TOPICS"
    )


def encode_data(data: Any) -> bytes:
    if data is None:
        return b""
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    return json.dumps(data, default=str).encode("utf-8")


async def publish(
    topic_path: str,
    data: bytes,
    attributes: dict[str, str],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Publish and wait for the server-assigned message id."""

    def _send() -> str:
        future = _get_client().publish(topic_path, data, **attributes)
        return str(future.result(timeout=timeout))

    return await asyncio.to_thread(_send)


def validate_pubsub_publish_steps(steps: Any) -> list[str]:
    """Problems with every ``type: pubsub_publish`` step's config (empty = fine).

    Structural only: the topic is usually templated, and the allowlist is a
    property of the deployment, so it is enforced when the step runs.
    """
    errors: list[str] = []
    if not isinstance(steps, list):
        return errors
    for step in steps:
        if not isinstance(step, dict) or step.get("type") != "pubsub_publish":
            continue
        sid = step.get("id", "?")
        topic = step.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            errors.append(f"step '{sid}': 'topic' is required")
        elif topic.startswith("projects/") and "{" not in topic:
            try:
                resolve_topic(topic, None)
            except ValueError as exc:
                errors.append(f"step '{sid}': {exc}")
        project_id = step.get("project_id")
        if project_id is not None and not isinstance(project_id, str):
            errors.append(f"step '{sid}': 'project_id' must be a string")
        data = step.get("data")
        if data is not None and not isinstance(data, (dict, list, str)):
            errors.append(f"step '{sid}': 'data' must be an object, list or string")
        attributes = step.get("attributes")
        if attributes is not None:
            if not isinstance(attributes, dict):
                errors.append(f"step '{sid}': 'attributes' must be an object of strings")
            else:
                for k, v in attributes.items():
                    if isinstance(v, (dict, list)):
                        errors.append(f"step '{sid}': attribute '{k}' must be a string")
        timeout = step.get("timeout_seconds")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            errors.append(f"step '{sid}': 'timeout_seconds' must be a positive number")
    return errors
