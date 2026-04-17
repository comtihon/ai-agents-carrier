from __future__ import annotations

from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.core.config import Settings
from app.domain.models.graph_run import GraphRun


class MongoGraphRunRepository:
    def __init__(self, collection: AsyncIOMotorCollection) -> None:
        self._collection = collection

    async def create(self, run: GraphRun) -> None:
        run.touch()
        await self._collection.insert_one(self._to_doc(run))

    async def update(self, run: GraphRun) -> None:
        run.touch()
        await self._collection.replace_one({"_id": run.id}, self._to_doc(run), upsert=True)

    async def get(self, run_id: str) -> GraphRun | None:
        doc = await self._collection.find_one({"_id": run_id})
        return self._from_doc(doc) if doc else None

    async def list_incomplete(self) -> list[GraphRun]:
        cursor = self._collection.find({"status": {"$in": ["running", "waiting_approval"]}})
        docs = await cursor.to_list(length=None)
        return [self._from_doc(doc) for doc in docs]

    @staticmethod
    def _to_doc(run: GraphRun) -> dict[str, Any]:
        data = run.model_dump(mode="python")
        data["_id"] = data.pop("id")
        return data

    @staticmethod
    def _from_doc(doc: dict[str, Any]) -> GraphRun:
        data = dict(doc)
        data["id"] = data.pop("_id")
        return GraphRun.model_validate(data)


class MongoClientProvider:
    _COLLECTION = "graph_runs"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncIOMotorClient | None = None

    def get_repository(self) -> MongoGraphRunRepository:
        if self._client is None:
            self._client = AsyncIOMotorClient(self._settings.mongodb_uri)
        db = self._client[self._settings.mongodb_database]
        return MongoGraphRunRepository(db[self._COLLECTION])

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
