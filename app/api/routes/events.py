"""REST API for EventDefinition CRUD.

An event is a Pub/Sub topic a workflow can be triggered by.  Events carry no
credentials, so unlike data sources nothing here is redacted on the way out.

Name conflicts
--------------
Ids are unique by construction (they are the Mongo ``_id``).  Names are not,
but two events sharing one is almost always a mistake — a second "orders"
event silently shadows the first in every picker.  ``POST`` therefore refuses a
duplicate name with 409; a caller that means it sends ``PUT`` to the existing
event instead.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ValidationError

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.event_definition import EventDefinition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


# ─── Request models ───────────────────────────────────────────────────────────

class CreateEventRequest(BaseModel):
    id: str
    name: str = ""
    description: str | None = None
    topic: str = ""
    subscription: str = ""
    project_id: str = ""
    event_schema: dict | None = None


class UpdateEventRequest(BaseModel):
    # None = omitted by the caller -> preserve the existing value.
    name: str | None = None
    description: str | None = None
    topic: str | None = None
    subscription: str | None = None
    project_id: str | None = None
    event_schema: dict | None = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_backend(container: ApplicationContainer) -> None:
    if container.event_backend is None:
        raise HTTPException(status_code=501, detail="Event backend not configured")


def _build_definition(data: dict[str, Any]) -> EventDefinition:
    """Validate the payload into a definition, mapping errors onto HTTP 422."""
    if not (data.get("topic") or "").strip():
        raise HTTPException(status_code=422, detail="An event needs a topic")
    try:
        return EventDefinition.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc


async def _reject_duplicate_name(
    container: ApplicationContainer, name: str, *, allow_id: str | None = None
) -> None:
    """409 when *name* is already taken by a different event."""
    if not name.strip():
        return
    assert container.event_backend is not None
    clash = await container.event_backend.get_by_name(name)
    if clash is not None and clash.id != allow_id:
        raise HTTPException(
            status_code=409,
            detail=f"Event '{name}' already exists (id '{clash.id}')",
        )


# ─── Collection routes ────────────────────────────────────────────────────────

@router.get("")
async def list_events(
    view: Literal["full", "summary"] = "full",
    container: ApplicationContainer = Depends(get_container),
):
    """List all registered events.

    ``view=summary`` leaves out ``event_schema``, which dominates the payload --
    a list of events is drawn from name/topic alone, and the schema is only
    needed once one event is opened.  Default stays ``full``.
    """
    _require_backend(container)
    assert container.event_backend is not None
    events = await container.event_backend.list()
    if view == "summary":
        return [
            {
                "id": e.id,
                "name": e.name,
                "description": e.description,
                "topic": e.topic,
            }
            for e in events
        ]
    return [e.model_dump(mode="json") for e in events]


@router.post("", status_code=201)
async def create_event(
    body: CreateEventRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Register a new event.

    409 when the id exists, and also when the name does — overwriting an event
    someone else's workflow subscribes to has to be asked for explicitly with
    ``PUT``.
    """
    _require_backend(container)
    assert container.event_backend is not None

    existing = await container.event_backend.get(body.id)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Event '{body.id}' already exists")
    await _reject_duplicate_name(container, body.name)

    defn = _build_definition(body.model_dump())
    saved = await container.event_backend.create(defn)
    return saved.model_dump(mode="json")


# ─── Item routes (parameterised — must come AFTER literal-segment routes) ─────

@router.get("/{event_id}")
async def get_event(
    event_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Get a specific event by ID."""
    _require_backend(container)
    assert container.event_backend is not None

    defn = await container.event_backend.get(event_id)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    return defn.model_dump(mode="json")


@router.put("/{event_id}")
async def update_event(
    event_id: str,
    body: UpdateEventRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Update an existing event (omitted fields are preserved)."""
    _require_backend(container)
    assert container.event_backend is not None

    existing = await container.event_backend.get(event_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")

    incoming = body.model_dump(exclude_none=True)
    if "name" in incoming:
        await _reject_duplicate_name(container, incoming["name"], allow_id=event_id)

    payload = existing.model_dump(mode="json")
    payload.update(incoming)
    payload["id"] = event_id
    payload["created_at"] = existing.created_at
    defn = _build_definition(payload)
    saved = await container.event_backend.update(event_id, defn)
    return saved.model_dump(mode="json")


@router.delete("/{event_id}", status_code=204)
async def delete_event(
    event_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Delete an event.

    Workflows pointing at it are left alone: their trigger steps start failing
    to resolve, which is logged per step at registration time.
    """
    _require_backend(container)
    assert container.event_backend is not None

    existing = await container.event_backend.get(event_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")

    await container.event_backend.delete(event_id)
