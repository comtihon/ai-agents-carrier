from __future__ import annotations

from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.core.config import Settings
from app.domain.interfaces.repositories import WorkflowRunRepository
from app.domain.models.runtime import WorkflowRun


class MongoWorkflowRunRepository(WorkflowRunRepository):
    def __init__(self, collection: AsyncIOMotorCollection) -> None:
        self._collection = collection

    async def create(self, workflow_run: WorkflowRun) -> WorkflowRun:
        workflow_run.touch()
        await self._collection.insert_one(self._serialize(workflow_run))
        return workflow_run

    async def update(self, workflow_run: WorkflowRun) -> WorkflowRun:
        workflow_run.touch()
        await self._collection.replace_one({"_id": workflow_run.id}, self._serialize(workflow_run), upsert=True)
        return workflow_run

    async def get_by_id(self, run_id: str) -> WorkflowRun | None:
        document = await self._collection.find_one({"_id": run_id})
        if document is None:
            return None
        return self._deserialize(document)

    @staticmethod
    def _serialize(workflow_run: WorkflowRun) -> dict[str, Any]:
        payload = workflow_run.model_dump(mode="python")
        payload["_id"] = payload.pop("id")
        return payload

    @staticmethod
    def _deserialize(document: dict) -> WorkflowRun:
        payload = dict(document)
        payload["id"] = payload.pop("_id")
        return WorkflowRun.model_validate(payload)


class MongoClientProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncIOMotorClient | None = None

    def get_client(self) -> AsyncIOMotorClient:
        if self._client is None:
            self._client = AsyncIOMotorClient(self._settings.mongodb_uri)
        return self._client

    def get_repository(self) -> MongoWorkflowRunRepository:
        database = self.get_client()[self._settings.mongodb_database]
        collection = database[self._settings.workflow_runs_collection]
        return MongoWorkflowRunRepository(collection)

    async def ping(self) -> bool:
        try:
            await self.get_client().admin.command("ping")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
