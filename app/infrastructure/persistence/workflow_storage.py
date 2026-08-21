"""Per-workflow key/value storage.

Workflow runs are isolated: each one is a fresh ``GraphRun`` whose state dies
with it.  That is right for most steps, but it leaves nowhere to keep the small
amount of state a *recurring* workflow needs to do its job — "which alerts have
I already reported", "which overrides did I accept", "when did I last see a
usable snapshot".  Without it a scheduled workflow cannot tell its first run
from its thousandth.

This is that place, and deliberately nothing more: a few KB of JSON per key,
read and written by ``storage`` steps.  It is state, not a knowledge base —
there is no indexing, no search and no history.

**Every entry is owned by exactly one workflow.**  ``workflow_id`` is not a
parameter a step can choose: it is taken from the runner that owns the step and
baked into the document ``_id`` as ``<workflow_id>::<key>``, and every query
filters on it as well.  So a workflow cannot read or write another workflow's
entries even by guessing a key name — there is no code path that would let it
name a different owner.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# A single entry is meant for bookkeeping, not for parking a dataset. Mongo's own
# document ceiling is 16 MB; refusing well below it turns "the run mysteriously
# failed to save" into a clear error at the point of the write.
MAX_VALUE_BYTES = 512 * 1024


class WorkflowStorageBackend(ABC):
    """Key/value storage scoped to one workflow.

    Implementations must treat ``workflow_id`` as the owner of the entry and
    must never expose a way to address another workflow's keys.
    """

    @abstractmethod
    async def get(self, workflow_id: str, key: str) -> Any | None: ...

    @abstractmethod
    async def set(self, workflow_id: str, key: str, value: Any) -> None: ...

    @abstractmethod
    async def delete(self, workflow_id: str, key: str) -> None: ...

    @abstractmethod
    async def keys(self, workflow_id: str) -> list[str]: ...

    @abstractmethod
    async def clear(self, workflow_id: str) -> int:
        """Drop every entry of *workflow_id*; returns how many were removed."""


def _doc_id(workflow_id: str, key: str) -> str:
    return f"{workflow_id}::{key}"


def check_value_size(value: Any) -> None:
    """Raise when *value* is too large to be bookkeeping.

    Measured on the JSON encoding rather than the Python object, because that is
    what actually goes to Mongo.
    """
    import json

    try:
        encoded = json.dumps(value, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"storage value is not JSON-serialisable: {exc}") from exc
    if len(encoded.encode("utf-8")) > MAX_VALUE_BYTES:
        raise ValueError(
            f"storage value is {len(encoded)} bytes, over the "
            f"{MAX_VALUE_BYTES}-byte limit for one key. Workflow storage is for "
            f"bookkeeping (a few KB), not for parking a dataset."
        )


class MongoWorkflowStorageBackend(WorkflowStorageBackend):
    """Mongo-backed implementation, one document per (workflow, key)."""

    _COLLECTION = "workflow_storage"

    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(uri)
        self._col = self._client[database][self._COLLECTION]

    async def get(self, workflow_id: str, key: str) -> Any | None:
        # Filtering on workflow_id as well as the composite _id is redundant by
        # construction, and kept so that a future change to the id scheme cannot
        # silently turn into a cross-workflow read.
        doc = await self._col.find_one(
            {"_id": _doc_id(workflow_id, key), "workflow_id": workflow_id}
        )
        return doc.get("value") if doc else None

    async def set(self, workflow_id: str, key: str, value: Any) -> None:
        check_value_size(value)
        now = datetime.now(timezone.utc)
        await self._col.update_one(
            {"_id": _doc_id(workflow_id, key)},
            {
                "$set": {
                    "workflow_id": workflow_id,
                    "key": key,
                    "value": value,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def delete(self, workflow_id: str, key: str) -> None:
        await self._col.delete_one(
            {"_id": _doc_id(workflow_id, key), "workflow_id": workflow_id}
        )

    async def keys(self, workflow_id: str) -> list[str]:
        docs = await self._col.find(
            {"workflow_id": workflow_id}, {"key": 1}
        ).to_list(None)
        return sorted(d["key"] for d in docs if d.get("key"))

    async def clear(self, workflow_id: str) -> int:
        result = await self._col.delete_many({"workflow_id": workflow_id})
        return int(result.deleted_count)

    async def close(self) -> None:
        self._client.close()


class InMemoryWorkflowStorageBackend(WorkflowStorageBackend):
    """Process-local implementation for tests and Mongo-less local runs.

    Keyed the same way as the Mongo one, so a test that reaches across
    workflows fails here exactly as it would in production.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def get(self, workflow_id: str, key: str) -> Any | None:
        return self._data.get(_doc_id(workflow_id, key))

    async def set(self, workflow_id: str, key: str, value: Any) -> None:
        check_value_size(value)
        self._data[_doc_id(workflow_id, key)] = value

    async def delete(self, workflow_id: str, key: str) -> None:
        self._data.pop(_doc_id(workflow_id, key), None)

    async def keys(self, workflow_id: str) -> list[str]:
        prefix = f"{workflow_id}::"
        return sorted(k[len(prefix):] for k in self._data if k.startswith(prefix))

    async def clear(self, workflow_id: str) -> int:
        prefix = f"{workflow_id}::"
        doomed = [k for k in self._data if k.startswith(prefix)]
        for k in doomed:
            del self._data[k]
        return len(doomed)
