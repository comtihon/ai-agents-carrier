from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.domain.models.data_source_definition import DataSourceDefinition

logger = logging.getLogger(__name__)


class DataSourceDefinitionBackend(ABC):
    """Persistent storage for data source definitions.

    The only implementation provided is ``MongoDataSourceBackend`` — data
    source definitions are always stored in MongoDB (like agent definitions,
    and unlike workflow definitions which also support a local-files backend).
    """

    @abstractmethod
    async def list(self) -> list[DataSourceDefinition]: ...

    @abstractmethod
    async def get(self, source_id: str) -> DataSourceDefinition | None: ...

    @abstractmethod
    async def create(self, definition: DataSourceDefinition) -> DataSourceDefinition: ...

    @abstractmethod
    async def update(self, source_id: str, definition: DataSourceDefinition) -> DataSourceDefinition: ...

    @abstractmethod
    async def delete(self, source_id: str) -> None: ...


# ---------------------------------------------------------------------------
# MongoDB implementation
# ---------------------------------------------------------------------------

class MongoDataSourceBackend(DataSourceDefinitionBackend):
    """Reads and writes data source definitions in a MongoDB collection."""

    _COLLECTION = "data_source_definitions"

    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(uri)
        self._col = self._client[database][self._COLLECTION]

    async def list(self) -> list[DataSourceDefinition]:
        docs = await self._col.find({}).to_list(None)
        return [self._from_doc(d) for d in docs]

    async def get(self, source_id: str) -> DataSourceDefinition | None:
        doc = await self._col.find_one({"_id": source_id})
        return self._from_doc(doc) if doc else None

    async def create(self, definition: DataSourceDefinition) -> DataSourceDefinition:
        definition.touch()
        await self._col.replace_one(
            {"_id": definition.id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def update(self, source_id: str, definition: DataSourceDefinition) -> DataSourceDefinition:
        definition.id = source_id
        definition.touch()
        await self._col.replace_one(
            {"_id": source_id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def delete(self, source_id: str) -> None:
        await self._col.delete_one({"_id": source_id})

    async def close(self) -> None:
        self._client.close()

    @staticmethod
    def _to_doc(defn: DataSourceDefinition) -> dict[str, Any]:
        data = defn.model_dump(mode="python")
        data["_id"] = data.pop("id")
        return data

    @staticmethod
    def _from_doc(doc: dict[str, Any]) -> DataSourceDefinition:
        data = dict(doc)
        data["id"] = data.pop("_id")
        return DataSourceDefinition.model_validate(data)
