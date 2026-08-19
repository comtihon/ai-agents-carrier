from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.domain.models.event_definition import EventDefinition

logger = logging.getLogger(__name__)


class EventDefinitionBackend(ABC):
    """Persistent storage for event definitions.

    The only implementation provided is ``MongoEventBackend`` — events are
    always stored in MongoDB, like agent and data source definitions.
    """

    @abstractmethod
    async def list(self) -> list[EventDefinition]: ...

    @abstractmethod
    async def get(self, event_id: str) -> EventDefinition | None: ...

    @abstractmethod
    async def get_by_name(self, name: str) -> EventDefinition | None: ...

    @abstractmethod
    async def create(self, definition: EventDefinition) -> EventDefinition: ...

    @abstractmethod
    async def update(self, event_id: str, definition: EventDefinition) -> EventDefinition: ...

    @abstractmethod
    async def delete(self, event_id: str) -> None: ...


# ---------------------------------------------------------------------------
# MongoDB implementation
# ---------------------------------------------------------------------------

class MongoEventBackend(EventDefinitionBackend):
    """Reads and writes event definitions in a MongoDB collection."""

    _COLLECTION = "event_definitions"

    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(uri)
        self._col = self._client[database][self._COLLECTION]

    async def list(self) -> list[EventDefinition]:
        docs = await self._col.find({}).to_list(None)
        return [self._from_doc(d) for d in docs]

    async def get(self, event_id: str) -> EventDefinition | None:
        doc = await self._col.find_one({"_id": event_id})
        return self._from_doc(doc) if doc else None

    async def get_by_name(self, name: str) -> EventDefinition | None:
        """First event carrying *name*, or None.

        Names are not unique by construction — this is what the API uses to
        warn before a save silently shadows an existing event.
        """
        doc = await self._col.find_one({"name": name})
        return self._from_doc(doc) if doc else None

    async def create(self, definition: EventDefinition) -> EventDefinition:
        definition.touch()
        await self._col.replace_one(
            {"_id": definition.id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def update(self, event_id: str, definition: EventDefinition) -> EventDefinition:
        definition.id = event_id
        definition.touch()
        await self._col.replace_one(
            {"_id": event_id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def delete(self, event_id: str) -> None:
        await self._col.delete_one({"_id": event_id})

    async def close(self) -> None:
        self._client.close()

    @staticmethod
    def _to_doc(defn: EventDefinition) -> dict[str, Any]:
        data = defn.model_dump(mode="python")
        data["_id"] = data.pop("id")
        return data

    @staticmethod
    def _from_doc(doc: dict[str, Any]) -> EventDefinition:
        data = dict(doc)
        data["id"] = data.pop("_id")
        return EventDefinition.model_validate(data)
