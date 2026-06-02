from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import ReturnDocument

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

    async def claim_for_resume(self, run_id: str) -> GraphRun | None:
        """Atomically flip status waiting_approval → running.

        Returns the updated run when the swap succeeds, otherwise None.
        Without this guard the /approve and /reject handlers had a TOCTOU
        race: two near-simultaneous clicks would both read status=
        waiting_approval and both schedule their own resume task on the
        same langgraph runner, which corrupted state and observably left
        the run stuck back at waiting_approval after the user had clicked
        through.
        """
        doc = await self._collection.find_one_and_update(
            {"_id": run_id, "status": "waiting_approval"},
            {"$set": {"status": "running", "updated_at": datetime.now(timezone.utc)}},
            return_document=ReturnDocument.AFTER,
        )
        return self._from_doc(doc) if doc else None

    async def find_by_ask_context_ts(self, thread_ts: str) -> GraphRun | None:
        doc = await self._collection.find_one({
            "status": "waiting_approval",
            "state._slack_ask_context_ts": thread_ts,
        })
        return self._from_doc(doc) if doc else None

    async def list_incomplete(self) -> list[GraphRun]:
        cursor = self._collection.find({"status": {"$in": ["running", "waiting_approval", "waiting_agent"]}})
        docs = await cursor.to_list(length=None)
        return [self._from_doc(doc) for doc in docs]

    def _build_run_query(
        self,
        workflow_id: str | None = None,
        status: str | None = None,
        search: str | None = None,
        exclude_workflow_ids: list[str] | None = None,
    ) -> dict:
        query: dict = {}
        if workflow_id and not exclude_workflow_ids:
            query["graph_id"] = workflow_id
        elif exclude_workflow_ids and not workflow_id:
            query["graph_id"] = {"$nin": exclude_workflow_ids}
        elif workflow_id and exclude_workflow_ids:
            query["$and"] = [
                {"graph_id": workflow_id},
                {"graph_id": {"$nin": exclude_workflow_ids}},
            ]
        if status:
            query["status"] = status
        if search:
            query["user_request"] = {"$regex": search, "$options": "i"}
        return query

    async def list_recent(
        self,
        limit: int = 50,
        offset: int = 0,
        workflow_id: str | None = None,
        status: str | None = None,
        search: str | None = None,
        exclude_workflow_ids: list[str] | None = None,
    ) -> list[GraphRun]:
        query = self._build_run_query(workflow_id, status, search, exclude_workflow_ids)
        cursor = self._collection.find(query).sort("created_at", -1).skip(offset).limit(limit)
        docs = await cursor.to_list(length=None)
        return [self._from_doc(doc) for doc in docs]

    async def count_recent(
        self,
        workflow_id: str | None = None,
        status: str | None = None,
        search: str | None = None,
        exclude_workflow_ids: list[str] | None = None,
    ) -> int:
        query = self._build_run_query(workflow_id, status, search, exclude_workflow_ids)
        return await self._collection.count_documents(query)

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


_PVC_COLLECTION = "pvc_leases"


class MongoPvcLeaseRepository:
    def __init__(self, collection) -> None:
        self._col = collection

    async def save(self, lease: dict) -> None:
        await self._col.replace_one({"_id": lease["pvc_name"]}, {**lease, "_id": lease["pvc_name"]}, upsert=True)

    async def get_expired(self, now) -> list[dict]:
        cursor = self._col.find({"expires_at": {"$lte": now}})
        return await cursor.to_list(length=1000)

    async def delete(self, pvc_name: str) -> None:
        await self._col.delete_one({"_id": pvc_name})

    async def delete_by_run(self, run_id: str) -> list[dict]:
        docs = await self._col.find({"run_id": run_id}).to_list(length=100)
        if docs:
            await self._col.delete_many({"run_id": run_id})
        return docs


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

    def get_pvc_lease_repository(self) -> MongoPvcLeaseRepository:
        if self._client is None:
            self._client = AsyncIOMotorClient(self._settings.mongodb_uri)
        db = self._client[self._settings.mongodb_database]
        return MongoPvcLeaseRepository(db[_PVC_COLLECTION])

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
