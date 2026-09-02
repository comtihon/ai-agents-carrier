"""The run list must stay cheap.

`graph_runs` documents are large — step_inputs alone averaged 471 KB of a
611 KB document in production — and the collection had no index on the sort
key. Together that made `GET /workflows/runs` answer 500 past the first pages
(Mongo aborts an in-memory sort over 32 MB) and ship megabytes when it did
answer. These tests pin both halves of the fix.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.persistence.mongo import _LIST_PROJECTION, MongoGraphRunRepository


def _collection_with(docs: list[dict]) -> MagicMock:
    """A collection whose find() records its call and answers with `docs`."""
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.to_list = AsyncMock(return_value=docs)
    col = MagicMock()
    col.find.return_value = cursor
    col.create_index = AsyncMock()
    return col


@pytest.mark.asyncio
async def test_summary_list_leaves_the_heavy_fields_in_the_database():
    col = _collection_with([{"_id": "r1", "graph_id": "wf", "status": "completed"}])
    repo = MongoGraphRunRepository(col)

    runs = await repo.list_recent(limit=13, offset=39, summary=True)

    _, projection = col.find.call_args[0]
    assert projection == _LIST_PROJECTION
    for field in ("state", "step_inputs", "step_outputs", "routing_log", "trace_data"):
        assert projection[field] == 0
    # The model defaults stand in for what was not read.
    assert runs[0].step_inputs == {}
    assert runs[0].state == {}


@pytest.mark.asyncio
async def test_full_list_still_reads_whole_documents():
    col = _collection_with([])
    repo = MongoGraphRunRepository(col)

    await repo.list_recent(limit=5)

    _, projection = col.find.call_args[0]
    assert projection is None


@pytest.mark.asyncio
async def test_summary_list_still_pages_newest_first():
    col = _collection_with([])
    repo = MongoGraphRunRepository(col)

    await repo.list_recent(limit=13, offset=39, summary=True)

    cursor = col.find.return_value
    cursor.sort.assert_called_once_with("created_at", -1)
    cursor.skip.assert_called_once_with(39)
    cursor.limit.assert_called_once_with(13)


@pytest.mark.asyncio
async def test_ensure_indexes_covers_the_list_sort_and_its_filters():
    col = _collection_with([])
    repo = MongoGraphRunRepository(col)

    await repo.ensure_indexes()

    created = [call.args[0] for call in col.create_index.call_args_list]
    assert [("created_at", -1)] in created
    assert [("graph_id", 1), ("created_at", -1)] in created
    assert [("status", 1), ("created_at", -1)] in created
