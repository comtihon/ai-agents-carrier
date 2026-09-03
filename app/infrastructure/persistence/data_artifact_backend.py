"""Persistent storage for run download manifests.

One collection, ``data_artifacts``.  One access pattern matters and is indexed:
everything a run exported, newest first — which is what ``GET
/runs/{id}/data`` and the ``list_run_data`` tool both ask for.  Entries are
append-only: a `data` node inside a loop writes a new one on every pass, and
the manifest is meant to list all of them.

Mirrors ``ScriptDefinitionBackend`` / ``ApprovalCaseBackend`` — an ABC, a Mongo
implementation, and an in-memory one used by tests and by Mongo-less
deployments.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.domain.models.data_artifact import DataArtifact

logger = logging.getLogger(__name__)


class DataArtifactBackend(ABC):
    """Persistent storage for :class:`DataArtifact` rows."""

    @abstractmethod
    async def add(self, artifact: DataArtifact) -> DataArtifact:
        """Record one manifest entry."""

    @abstractmethod
    async def list_for_run(self, run_id: str) -> list[DataArtifact]:
        """Every artifact of one run, newest first."""

    @abstractmethod
    async def get(self, run_id: str, artifact_id: str) -> DataArtifact | None:
        """One artifact, looked up *within* its run.

        Scoped on purpose: an artifact id that leaked out of one run must not
        resolve under another, so the run in the URL is part of the key rather
        than decoration on it.
        """

    @abstractmethod
    async def delete_for_run(self, run_id: str) -> list[DataArtifact]:
        """Drop a run's manifest; returns the entries removed.

        The caller unpins the streams the returned entries name — dropping the
        rows without that would leave every stream pinned for good.
        """


# ---------------------------------------------------------------------------
# MongoDB implementation
# ---------------------------------------------------------------------------

class MongoDataArtifactBackend(DataArtifactBackend):
    """Reads and writes manifest entries in the ``data_artifacts`` collection."""

    _COLLECTION = "data_artifacts"

    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(uri)
        self._col = self._client[database][self._COLLECTION]
        self._indexes_ready = False

    async def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        try:
            await self._col.create_index([("run_id", 1), ("created_at", -1)])
            self._indexes_ready = True
        except Exception:
            logger.exception("failed to create data_artifacts indexes")

    async def add(self, artifact: DataArtifact) -> DataArtifact:
        await self.ensure_indexes()
        await self._col.replace_one(
            {"_id": artifact.id}, self._to_doc(artifact), upsert=True
        )
        return artifact

    async def list_for_run(self, run_id: str) -> list[DataArtifact]:
        docs = await self._col.find({"run_id": run_id}).sort("created_at", -1).to_list(None)
        return [self._from_doc(d) for d in docs]

    async def get(self, run_id: str, artifact_id: str) -> DataArtifact | None:
        doc = await self._col.find_one({"_id": artifact_id, "run_id": run_id})
        return self._from_doc(doc) if doc else None

    async def delete_for_run(self, run_id: str) -> list[DataArtifact]:
        rows = await self.list_for_run(run_id)
        if rows:
            await self._col.delete_many({"run_id": run_id})
        return rows

    async def close(self) -> None:
        self._client.close()

    @staticmethod
    def _to_doc(artifact: DataArtifact) -> dict[str, Any]:
        data = artifact.model_dump(mode="python")
        data["_id"] = data.pop("id")
        return data

    @staticmethod
    def _from_doc(doc: dict[str, Any]) -> DataArtifact:
        data = dict(doc)
        data["id"] = data.pop("_id")
        return DataArtifact.model_validate(data)


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryDataArtifactBackend(DataArtifactBackend):
    """Non-persistent backend used by tests and by Mongo-less deployments."""

    def __init__(self) -> None:
        self._rows: dict[str, DataArtifact] = {}

    async def add(self, artifact: DataArtifact) -> DataArtifact:
        self._rows[artifact.id] = artifact
        return artifact

    async def list_for_run(self, run_id: str) -> list[DataArtifact]:
        rows = [a for a in self._rows.values() if a.run_id == run_id]
        rows.sort(key=lambda a: a.created_at, reverse=True)
        return rows

    async def get(self, run_id: str, artifact_id: str) -> DataArtifact | None:
        row = self._rows.get(artifact_id)
        return row if row is not None and row.run_id == run_id else None

    async def delete_for_run(self, run_id: str) -> list[DataArtifact]:
        rows = await self.list_for_run(run_id)
        for row in rows:
            self._rows.pop(row.id, None)
        return rows
