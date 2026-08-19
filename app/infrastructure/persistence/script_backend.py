from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.domain.models.script_definition import ScriptDefinition

logger = logging.getLogger(__name__)


class ScriptDefinitionBackend(ABC):
    """Persistent storage for the Python script library.

    Mirrors ``AgentDefinitionBackend`` — the only implementation is
    ``MongoScriptBackend``.
    """

    @abstractmethod
    async def list(self) -> list[ScriptDefinition]: ...

    @abstractmethod
    async def get(self, script_id: str) -> ScriptDefinition | None: ...

    @abstractmethod
    async def get_by_name(self, name: str) -> ScriptDefinition | None: ...

    @abstractmethod
    async def create(self, definition: ScriptDefinition) -> ScriptDefinition: ...

    @abstractmethod
    async def update(self, script_id: str, definition: ScriptDefinition) -> ScriptDefinition: ...

    @abstractmethod
    async def delete(self, script_id: str) -> None: ...


# ---------------------------------------------------------------------------
# MongoDB implementation
# ---------------------------------------------------------------------------

class MongoScriptBackend(ScriptDefinitionBackend):
    """Reads and writes script definitions in a MongoDB collection."""

    _COLLECTION = "script_definitions"

    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(uri)
        self._col = self._client[database][self._COLLECTION]

    async def list(self) -> list[ScriptDefinition]:
        docs = await self._col.find({}).to_list(None)
        return [self._from_doc(d) for d in docs]

    async def get(self, script_id: str) -> ScriptDefinition | None:
        doc = await self._col.find_one({"_id": script_id})
        return self._from_doc(doc) if doc else None

    async def get_by_name(self, name: str) -> ScriptDefinition | None:
        doc = await self._col.find_one({"name": name})
        return self._from_doc(doc) if doc else None

    async def create(self, definition: ScriptDefinition) -> ScriptDefinition:
        definition.touch()
        await self._col.replace_one(
            {"_id": definition.id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def update(self, script_id: str, definition: ScriptDefinition) -> ScriptDefinition:
        definition.id = script_id
        definition.touch()
        await self._col.replace_one(
            {"_id": script_id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def delete(self, script_id: str) -> None:
        await self._col.delete_one({"_id": script_id})

    async def close(self) -> None:
        self._client.close()

    @staticmethod
    def _to_doc(defn: ScriptDefinition) -> dict[str, Any]:
        data = defn.model_dump(mode="python")
        data["_id"] = data.pop("id")
        return data

    @staticmethod
    def _from_doc(doc: dict[str, Any]) -> ScriptDefinition:
        data = dict(doc)
        data["id"] = data.pop("_id")
        return ScriptDefinition.model_validate(data)
