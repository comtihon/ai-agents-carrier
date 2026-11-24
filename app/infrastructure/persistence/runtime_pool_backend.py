"""Storage for runtime pools and their run assignments (MongoDB, plus an in-memory twin for tests).

``create``/``update`` write the pool's definition only: ``members`` change
through :meth:`set_members` and ``cursor`` through :meth:`next_cursor`, so a
save built from a request can neither add a member past the membership rules
nor rewind the round robin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from app.domain.models.ownership import COMPUTED_FIELDS
from app.domain.models.runtime_pool import PoolAssignment, PoolMember, RuntimePool

# Assignments outlive their run only to explain it afterwards.
ASSIGNMENT_TTL_S = 30 * 24 * 3600
_STATE_FIELDS = ("members", "cursor")


class RuntimePoolBackend(ABC):
    @abstractmethod
    async def list(self) -> list[RuntimePool]: ...

    @abstractmethod
    async def get(self, pool_id: str) -> RuntimePool | None: ...

    @abstractmethod
    async def create(self, pool: RuntimePool) -> RuntimePool: ...

    @abstractmethod
    async def update(self, pool_id: str, pool: RuntimePool) -> RuntimePool: ...

    @abstractmethod
    async def delete(self, pool_id: str) -> None: ...

    @abstractmethod
    async def set_members(self, pool_id: str, members: list[PoolMember]) -> None: ...

    @abstractmethod
    async def next_cursor(self, pool_id: str) -> int:
        """Advance the round robin; the position before the advance."""

    @abstractmethod
    async def get_assignment(self, pool_id: str, run_id: str) -> PoolAssignment | None: ...

    @abstractmethod
    async def claim_assignment(
        self, pool_id: str, run_id: str, agent_id: str, replacing: str | None,
    ) -> PoolAssignment:
        """Assign *agent_id* to the run unless someone got there first.

        *replacing* None: only if the run has no assignment yet. Otherwise only
        if the run is still assigned *replacing* (which moves to ``previous``).
        Returns the assignment in force afterwards, whoever won.
        """

    @abstractmethod
    async def list_assignments(self, pool_id: str, since: datetime | None = None) -> list[PoolAssignment]: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _pool_doc(pool: RuntimePool) -> dict[str, Any]:
    doc = pool.model_dump(mode="python", exclude=set(COMPUTED_FIELDS))
    doc["_id"] = doc.pop("id")
    return doc


def _pool_from(doc: dict[str, Any]) -> RuntimePool:
    doc = dict(doc)
    doc["id"] = doc.pop("_id")
    for key in COMPUTED_FIELDS:
        doc.pop(key, None)
    return RuntimePool.model_validate(doc)


def _assignment_from(doc: dict[str, Any]) -> PoolAssignment:
    doc = dict(doc)
    doc.pop("_id", None)
    return PoolAssignment.model_validate(doc)


class MongoRuntimePoolBackend(RuntimePoolBackend):
    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient

        self._client = AsyncIOMotorClient(uri, tz_aware=True)
        db = self._client[database]
        self._pools = db["acp_runtime_pools"]
        self._assignments = db["acp_runtime_pool_assignments"]
        self._indexed = False

    async def _ensure_indexes(self) -> None:
        if self._indexed:
            return
        await self._assignments.create_index("assigned_at", expireAfterSeconds=ASSIGNMENT_TTL_S)
        await self._assignments.create_index([("pool_id", 1), ("assigned_at", -1)])
        self._indexed = True

    async def list(self) -> list[RuntimePool]:
        return [_pool_from(d) for d in await self._pools.find({}).to_list(None)]

    async def get(self, pool_id: str) -> RuntimePool | None:
        doc = await self._pools.find_one({"_id": pool_id})
        return _pool_from(doc) if doc else None

    async def create(self, pool: RuntimePool) -> RuntimePool:
        pool.touch()
        doc = _pool_doc(pool)
        state = {k: doc.pop(k) for k in _STATE_FIELDS}
        doc.pop("_id")
        await self._pools.update_one(
            {"_id": pool.id}, {"$set": doc, "$setOnInsert": state}, upsert=True,
        )
        return await self.get(pool.id) or pool

    async def update(self, pool_id: str, pool: RuntimePool) -> RuntimePool:
        pool.id = pool_id
        return await self.create(pool)

    async def delete(self, pool_id: str) -> None:
        await self._pools.delete_one({"_id": pool_id})
        await self._assignments.delete_many({"pool_id": pool_id})

    async def set_members(self, pool_id: str, members: list[PoolMember]) -> None:
        await self._pools.update_one(
            {"_id": pool_id},
            {"$set": {"members": [m.model_dump(mode="python") for m in members], "updated_at": _utcnow()}},
        )

    async def next_cursor(self, pool_id: str) -> int:
        from pymongo import ReturnDocument

        doc = await self._pools.find_one_and_update(
            {"_id": pool_id}, {"$inc": {"cursor": 1}},
            projection={"cursor": 1}, return_document=ReturnDocument.BEFORE,
        )
        return int((doc or {}).get("cursor") or 0)

    async def get_assignment(self, pool_id: str, run_id: str) -> PoolAssignment | None:
        doc = await self._assignments.find_one({"_id": f"{pool_id}:{run_id}"})
        return _assignment_from(doc) if doc else None

    async def claim_assignment(
        self, pool_id: str, run_id: str, agent_id: str, replacing: str | None,
    ) -> PoolAssignment:
        from pymongo.errors import DuplicateKeyError

        await self._ensure_indexes()
        key = f"{pool_id}:{run_id}"
        now = _utcnow()
        if replacing is None or await self.get_assignment(pool_id, run_id) is None:
            try:
                await self._assignments.insert_one({
                    "_id": key, "pool_id": pool_id, "run_id": run_id,
                    "agent_id": agent_id, "assigned_at": now, "previous": [],
                })
            except DuplicateKeyError:
                pass
        else:
            await self._assignments.update_one(
                {"_id": key, "agent_id": replacing},
                {"$set": {"agent_id": agent_id, "assigned_at": now}, "$push": {"previous": replacing}},
            )
        current = await self.get_assignment(pool_id, run_id)
        assert current is not None
        return current

    async def list_assignments(self, pool_id: str, since: datetime | None = None) -> list[PoolAssignment]:
        query: dict[str, Any] = {"pool_id": pool_id}
        if since is not None:
            query["assigned_at"] = {"$gte": since}
        docs = await self._assignments.find(query).sort("assigned_at", -1).to_list(500)
        return [_assignment_from(d) for d in docs]

    async def close(self) -> None:
        self._client.close()


class InMemoryRuntimePoolBackend(RuntimePoolBackend):
    def __init__(self) -> None:
        self.pools: dict[str, RuntimePool] = {}
        self.assignments: dict[tuple[str, str], PoolAssignment] = {}

    async def list(self) -> list[RuntimePool]:
        return [p.model_copy(deep=True) for p in self.pools.values()]

    async def get(self, pool_id: str) -> RuntimePool | None:
        p = self.pools.get(pool_id)
        return p.model_copy(deep=True) if p else None

    async def create(self, pool: RuntimePool) -> RuntimePool:
        pool.touch()
        stored = self.pools.get(pool.id)
        saved = pool.model_copy(deep=True)
        if stored is not None:
            saved.members = [m.model_copy() for m in stored.members]
            saved.cursor = stored.cursor
            saved.created_at = stored.created_at or saved.created_at
        else:
            saved.members = []
            saved.cursor = 0
        self.pools[pool.id] = saved
        return saved.model_copy(deep=True)

    async def update(self, pool_id: str, pool: RuntimePool) -> RuntimePool:
        pool.id = pool_id
        return await self.create(pool)

    async def delete(self, pool_id: str) -> None:
        self.pools.pop(pool_id, None)
        for key in [k for k in self.assignments if k[0] == pool_id]:
            del self.assignments[key]

    async def set_members(self, pool_id: str, members: list[PoolMember]) -> None:
        p = self.pools.get(pool_id)
        if p is not None:
            p.members = [m.model_copy() for m in members]
            p.updated_at = _utcnow()

    async def next_cursor(self, pool_id: str) -> int:
        p = self.pools.get(pool_id)
        if p is None:
            return 0
        before = p.cursor
        p.cursor += 1
        return before

    async def get_assignment(self, pool_id: str, run_id: str) -> PoolAssignment | None:
        a = self.assignments.get((pool_id, run_id))
        return a.model_copy(deep=True) if a else None

    async def claim_assignment(
        self, pool_id: str, run_id: str, agent_id: str, replacing: str | None,
    ) -> PoolAssignment:
        key = (pool_id, run_id)
        current = self.assignments.get(key)
        if replacing is None or current is None:
            if current is None:
                self.assignments[key] = PoolAssignment(
                    pool_id=pool_id, run_id=run_id, agent_id=agent_id, assigned_at=_utcnow(),
                )
        elif current is not None and current.agent_id == replacing:
            current.previous.append(replacing)
            current.agent_id = agent_id
            current.assigned_at = _utcnow()
        return self.assignments[key].model_copy(deep=True)

    async def list_assignments(self, pool_id: str, since: datetime | None = None) -> list[PoolAssignment]:
        found = [
            a.model_copy(deep=True) for (pid, _), a in self.assignments.items()
            if pid == pool_id and (since is None or a.assigned_at >= since)
        ]
        return sorted(found, key=lambda a: a.assigned_at, reverse=True)
