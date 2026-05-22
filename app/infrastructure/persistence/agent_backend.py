from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.domain.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)


class AgentDefinitionBackend(ABC):
    """Persistent storage for agent definitions.

    The only implementation provided is ``MongoAgentBackend`` — agent
    definitions are always stored in MongoDB (unlike workflow definitions
    which also support a local-files backend).
    """

    @abstractmethod
    async def list(self) -> list[AgentDefinition]: ...

    @abstractmethod
    async def get(self, agent_id: str) -> AgentDefinition | None: ...

    @abstractmethod
    async def create(self, definition: AgentDefinition) -> AgentDefinition: ...

    @abstractmethod
    async def update(self, agent_id: str, definition: AgentDefinition) -> AgentDefinition: ...

    @abstractmethod
    async def delete(self, agent_id: str) -> None: ...


# ---------------------------------------------------------------------------
# MongoDB implementation
# ---------------------------------------------------------------------------

class MongoAgentBackend(AgentDefinitionBackend):
    """Reads and writes agent definitions in a MongoDB collection."""

    _COLLECTION = "agent_definitions"

    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(uri)
        self._col = self._client[database][self._COLLECTION]

    async def list(self) -> list[AgentDefinition]:
        docs = await self._col.find({}).to_list(None)
        return [self._from_doc(d) for d in docs]

    async def get(self, agent_id: str) -> AgentDefinition | None:
        doc = await self._col.find_one({"_id": agent_id})
        return self._from_doc(doc) if doc else None

    async def create(self, definition: AgentDefinition) -> AgentDefinition:
        definition.touch()
        await self._col.replace_one(
            {"_id": definition.id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def update(self, agent_id: str, definition: AgentDefinition) -> AgentDefinition:
        definition.id = agent_id
        definition.touch()
        await self._col.replace_one(
            {"_id": agent_id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def delete(self, agent_id: str) -> None:
        await self._col.delete_one({"_id": agent_id})

    async def close(self) -> None:
        self._client.close()

    @staticmethod
    def _to_doc(defn: AgentDefinition) -> dict[str, Any]:
        data = defn.model_dump(mode="python")
        data["_id"] = data.pop("id")
        return data

    @staticmethod
    def _from_doc(doc: dict[str, Any]) -> AgentDefinition:
        data = dict(doc)
        data["id"] = data.pop("_id")
        return AgentDefinition.model_validate(data)
