"""Persistent definition of an event a workflow can be triggered by.

An event is a Google Cloud Pub/Sub topic, configured once and reused: the
``pubsub`` trigger steps of several workflows can point at the same event
instead of repeating topic, schema and subscription each time.

Events used to be data sources with ``kind="pubsub"``.  They are their own
resource because they are not callable APIs — nothing about base URL, auth or
operations applies to them, and listing them next to REST sources said
otherwise.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class EventDefinition(BaseModel):
    """One Pub/Sub topic that workflows subscribe to."""

    id: str
    name: str = ""
    description: str | None = None
    # Short name ("orders") or a fully qualified path
    # ("projects/p/topics/orders").  The subscriber resolves short names
    # against the configured project.
    topic: str = ""
    # Subscription to pull from.  Empty means "create one on first use" — the
    # subscriber fills this in and the definition is saved back.
    subscription: str = ""
    # Project override; empty means the backend-wide PUBSUB_PROJECT_ID.
    project_id: str = ""
    # JSON-schema-ish description of the message payload: top-level type,
    # required keys and property types.  Named ``event_schema`` because
    # ``schema`` collides with pydantic's own attribute.
    event_schema: dict[str, Any] | None = None

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def touch(self) -> None:
        from datetime import timezone
        self.updated_at = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = self.updated_at
