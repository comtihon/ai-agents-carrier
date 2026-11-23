"""Leases on agent runtimes: who runs, who queues (MongoDB, plus an in-memory twin).

One document per runtime key (see app.application.concurrency)::

    {_id: key, limit, updated_at,
     holders: [{id, run_id, step_id, agent_id, workflow_id, acquired_at, heartbeat_at, expires_at}],
     queue:   [{id, run_id, step_id, agent_id, workflow_id, enqueued_at, heartbeat_at, expires_at}]}

Every change is a single-document update, so it is atomic across carrier
replicas: a waiter moves from ``queue`` to ``holders`` only in an update whose
filter checks, on the stored document, that it is among the first
``limit - len(holders)`` of the queue. The queue is kept sorted by
``(enqueued_at, id)``, i.e. FIFO. Entries carry ``expires_at``; whoever touches
the document drops expired ones first, so a crashed replica's slot frees itself
after the TTL.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class AcquireResult:
    acquired: bool
    # 1-based place in the queue while waiting (0 once acquired).
    position: int
    holders: int


class ConcurrencyBackend(ABC):
    @abstractmethod
    async def try_acquire(
        self, key: str, entry: dict[str, Any], limit: int, now: datetime, ttl_s: float, force: bool = False,
    ) -> AcquireResult:
        """Hold a slot of *key* for ``entry["id"]`` if it may, else (stay) queued.

        Already holding: the slot is refreshed. Not queued yet: queued with
        ``entry["enqueued_at"]``. ``force``: take a slot even past the limit
        (a run that is already executing, re-attached after a restart).
        """

    @abstractmethod
    async def heartbeat(self, key: str, entry_id: str, now: datetime, ttl_s: float) -> bool:
        """Extend a held slot; False when it is no longer held (expired, released)."""

    @abstractmethod
    async def release(self, key: str, entry_id: str) -> None:
        """Give up the slot or the queue place of *entry_id* (no-op when it has none)."""

    @abstractmethod
    async def list_keys(self, now: datetime) -> list[dict[str, Any]]:
        """Every key with live holders or waiters: ``{key, limit, holders, queue}``."""


def _new_holder(entry: dict[str, Any], now: datetime, ttl_s: float) -> dict[str, Any]:
    holder = {k: v for k, v in entry.items() if k != "enqueued_at"}
    holder.update(acquired_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=ttl_s))
    return holder


def _new_waiter(entry: dict[str, Any], now: datetime, ttl_s: float) -> dict[str, Any]:
    waiter = dict(entry)
    waiter.setdefault("enqueued_at", now)
    waiter.update(heartbeat_at=now, expires_at=now + timedelta(seconds=ttl_s))
    return waiter


def _live(entries: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    return [e for e in entries or [] if e.get("expires_at") is None or e["expires_at"] >= now]


def _sort_key(e: dict[str, Any]) -> tuple[Any, str]:
    return (e.get("enqueued_at"), str(e.get("id")))


class MongoConcurrencyBackend(ConcurrencyBackend):
    def __init__(self, uri: str, database: str, collection: str = "agent_runtime_leases") -> None:
        from motor.motor_asyncio import AsyncIOMotorClient

        self._client = AsyncIOMotorClient(uri, tz_aware=True)
        self._col = self._client[database][collection]

    async def _ensure_doc(self, key: str, limit: int | None, now: datetime) -> None:
        from pymongo.errors import DuplicateKeyError

        update: dict[str, Any] = {"$setOnInsert": {"holders": [], "queue": []}, "$set": {"updated_at": now}}
        if limit is not None:
            update["$set"]["limit"] = limit
        try:
            await self._col.update_one({"_id": key}, update, upsert=True)
        except DuplicateKeyError:  # two replicas created it at once: it exists now
            pass

    async def try_acquire(
        self, key: str, entry: dict[str, Any], limit: int, now: datetime, ttl_s: float, force: bool = False,
    ) -> AcquireResult:
        entry_id = entry["id"]
        expires = now + timedelta(seconds=ttl_s)
        await self._ensure_doc(key, limit, now)
        # Drop whatever a dead replica left behind.
        await self._col.update_one(
            {"_id": key},
            {"$pull": {"holders": {"expires_at": {"$lt": now}}, "queue": {"expires_at": {"$lt": now}}}},
        )
        held = await self._col.update_one(
            {"_id": key, "holders.id": entry_id},
            {"$set": {"holders.$.expires_at": expires, "holders.$.heartbeat_at": now}},
        )
        if held.matched_count:
            await self._col.update_one({"_id": key}, {"$pull": {"queue": {"id": entry_id}}})
            return await self._result(key, entry_id, True)
        if force:
            await self._col.update_one(
                {"_id": key, "holders.id": {"$ne": entry_id}},
                {"$push": {"holders": _new_holder(entry, now, ttl_s)}, "$pull": {"queue": {"id": entry_id}}},
            )
            return await self._result(key, entry_id, True)
        queued = await self._col.update_one(
            {"_id": key, "queue.id": entry_id},
            {"$set": {"queue.$.expires_at": expires, "queue.$.heartbeat_at": now}},
        )
        if not queued.matched_count:
            await self._col.update_one(
                {"_id": key, "queue.id": {"$ne": entry_id}},
                {"$push": {"queue": {"$each": [_new_waiter(entry, now, ttl_s)], "$sort": {"enqueued_at": 1, "id": 1}}}},
            )
        # Promote only if, on the stored document, this entry is among the first
        # (limit - holders) waiters: atomic, whichever replica gets there first.
        promoted = await self._col.update_one(
            {
                "_id": key,
                "queue.id": entry_id,
                "$expr": {"$lt": [
                    {"$indexOfArray": ["$queue.id", entry_id]},
                    {"$subtract": [limit, {"$size": "$holders"}]},
                ]},
            },
            {"$pull": {"queue": {"id": entry_id}}, "$push": {"holders": _new_holder(entry, now, ttl_s)}},
        )
        return await self._result(key, entry_id, bool(promoted.modified_count))

    async def _result(self, key: str, entry_id: str, acquired: bool) -> AcquireResult:
        doc = await self._col.find_one({"_id": key}, {"holders.id": 1, "queue.id": 1}) or {}
        ids = [q.get("id") for q in doc.get("queue") or []]
        position = 0 if acquired or entry_id not in ids else ids.index(entry_id) + 1
        return AcquireResult(acquired, position, len(doc.get("holders") or []))

    async def heartbeat(self, key: str, entry_id: str, now: datetime, ttl_s: float) -> bool:
        res = await self._col.update_one(
            {"_id": key, "holders": {"$elemMatch": {"id": entry_id, "expires_at": {"$gte": now}}}},
            {"$set": {"holders.$.expires_at": now + timedelta(seconds=ttl_s), "holders.$.heartbeat_at": now}},
        )
        return bool(res.matched_count)

    async def release(self, key: str, entry_id: str) -> None:
        await self._col.update_one({"_id": key}, {"$pull": {"holders": {"id": entry_id}, "queue": {"id": entry_id}}})

    async def list_keys(self, now: datetime) -> list[dict[str, Any]]:
        out = []
        async for doc in self._col.find({}):
            holders, queue = _live(doc.get("holders"), now), _live(doc.get("queue"), now)
            if holders or queue:
                out.append({"key": doc["_id"], "limit": doc.get("limit"), "holders": holders,
                            "queue": sorted(queue, key=_sort_key)})
        # Empty documents of idle keys are left in place: they are tiny and reused.
        return out

    async def close(self) -> None:
        self._client.close()


class InMemoryConcurrencyBackend(ConcurrencyBackend):
    """Single-process twin with the same semantics (tests, no-Mongo mode)."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _doc(self, key: str) -> dict[str, Any]:
        return self.docs.setdefault(key, {"limit": None, "holders": [], "queue": []})

    async def try_acquire(
        self, key: str, entry: dict[str, Any], limit: int, now: datetime, ttl_s: float, force: bool = False,
    ) -> AcquireResult:
        async with self._lock:
            doc = self._doc(key)
            doc["limit"] = limit
            doc["holders"] = _live(doc["holders"], now)
            doc["queue"] = _live(doc["queue"], now)
            entry_id = entry["id"]
            expires = now + timedelta(seconds=ttl_s)
            for h in doc["holders"]:
                if h["id"] == entry_id:
                    h.update(expires_at=expires, heartbeat_at=now)
                    doc["queue"] = [q for q in doc["queue"] if q["id"] != entry_id]
                    return AcquireResult(True, 0, len(doc["holders"]))
            if force:
                doc["queue"] = [q for q in doc["queue"] if q["id"] != entry_id]
                doc["holders"].append(_new_holder(entry, now, ttl_s))
                return AcquireResult(True, 0, len(doc["holders"]))
            mine = next((q for q in doc["queue"] if q["id"] == entry_id), None)
            if mine is None:
                doc["queue"].append(_new_waiter(entry, now, ttl_s))
                doc["queue"].sort(key=_sort_key)
            else:
                mine.update(expires_at=expires, heartbeat_at=now)
            index = [q["id"] for q in doc["queue"]].index(entry_id)
            if index < limit - len(doc["holders"]):
                doc["queue"].pop(index)
                doc["holders"].append(_new_holder(entry, now, ttl_s))
                return AcquireResult(True, 0, len(doc["holders"]))
            return AcquireResult(False, index + 1, len(doc["holders"]))

    async def heartbeat(self, key: str, entry_id: str, now: datetime, ttl_s: float) -> bool:
        async with self._lock:
            for h in self.docs.get(key, {}).get("holders", []):
                if h["id"] == entry_id and h["expires_at"] >= now:
                    h.update(expires_at=now + timedelta(seconds=ttl_s), heartbeat_at=now)
                    return True
            return False

    async def release(self, key: str, entry_id: str) -> None:
        async with self._lock:
            doc = self.docs.get(key)
            if doc is None:
                return
            doc["holders"] = [h for h in doc["holders"] if h["id"] != entry_id]
            doc["queue"] = [q for q in doc["queue"] if q["id"] != entry_id]

    async def list_keys(self, now: datetime) -> list[dict[str, Any]]:
        async with self._lock:
            out = []
            for key, doc in self.docs.items():
                holders, queue = _live(doc["holders"], now), _live(doc["queue"], now)
                if holders or queue:
                    out.append({"key": key, "limit": doc.get("limit"),
                                "holders": [dict(h) for h in holders],
                                "queue": [dict(q) for q in sorted(queue, key=_sort_key)]})
            return out
