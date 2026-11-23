"""MongoConcurrencyBackend against a real MongoDB (skipped without one).

Several backend instances stand in for several carrier replicas sharing the
collection.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.infrastructure.persistence.concurrency_backend import MongoConcurrencyBackend

_URI = "mongodb://localhost:27017"
_DB = "test_langgraph_integration"
T0 = datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc)


def _entry(i: str, at: datetime = T0) -> dict:
    return {"id": i, "run_id": f"run-{i}", "step_id": "s", "agent_id": "a", "workflow_id": "wf", "enqueued_at": at}


@pytest.fixture
async def replicas():
    backends = [MongoConcurrencyBackend(_URI, _DB, collection="test_agent_runtime_leases") for _ in range(4)]
    await backends[0]._col.delete_many({})
    yield backends
    await backends[0]._col.delete_many({})
    for b in backends:
        await b.close()


async def test_fifo_limit_and_ttl(replicas):
    a, b, c, _ = replicas
    assert (await a.try_acquire("k", _entry("a", T0), 1, T0, 30)).acquired
    late = await c.try_acquire("k", _entry("c", T0 + timedelta(seconds=2)), 1, T0, 30)
    early = await b.try_acquire("k", _entry("b", T0 + timedelta(seconds=1)), 1, T0, 30)
    assert not late.acquired and not early.acquired and early.position == 1
    assert (await c.try_acquire("k", _entry("c"), 1, T0, 30)).position == 2
    await a.release("k", "a")
    assert not (await c.try_acquire("k", _entry("c"), 1, T0, 30)).acquired
    assert (await b.try_acquire("k", _entry("b"), 1, T0, 30)).acquired
    assert await b.heartbeat("k", "b", T0 + timedelta(seconds=20), 30)
    # "b"'s replica dies: after its TTL "c" (still polling, so alive) gets in.
    later = T0 + timedelta(seconds=55)
    assert (await c.try_acquire("k", _entry("c"), 1, later, 30)).acquired
    [row] = await a.list_keys(later)
    assert [h["id"] for h in row["holders"]] == ["c"] and row["queue"] == []


async def test_limit_two_and_reacquire(replicas):
    a, b, c, _ = replicas
    assert (await a.try_acquire("k2", _entry("a"), 2, T0, 30)).acquired
    assert (await b.try_acquire("k2", _entry("b", T0 + timedelta(seconds=1)), 2, T0, 30)).acquired
    third = await c.try_acquire("k2", _entry("c", T0 + timedelta(seconds=2)), 2, T0, 30)
    assert not third.acquired and third.holders == 2
    assert (await a.try_acquire("k2", _entry("a"), 2, T0, 30)).acquired  # same id: still held
    assert (await c.try_acquire("k2", _entry("c"), 2, T0, 30, force=True)).acquired
    [row] = await a.list_keys(T0)
    assert len(row["holders"]) == 3


async def test_racing_replicas_admit_exactly_limit(replicas):
    for limit in (1, 2):
        key = f"race-{limit}"
        results = await asyncio.gather(*[
            replicas[n % 4].try_acquire(key, _entry(f"e{n}", T0 + timedelta(milliseconds=n)), limit, T0, 30)
            for n in range(12)
        ])
        assert sum(r.acquired for r in results) <= limit
        # Everyone polls again: exactly `limit` hold, the rest queue in order.
        for _ in range(2):
            results = [
                await replicas[n % 4].try_acquire(key, _entry(f"e{n}"), limit, T0, 30) for n in range(12)
            ]
        assert sum(r.acquired for r in results) == limit
        [row] = [r for r in await replicas[0].list_keys(T0) if r["key"] == key]
        assert len(row["holders"]) == limit
        ids = [q["id"] for q in row["queue"]]
        assert ids == sorted(ids, key=lambda i: int(i[1:]))
