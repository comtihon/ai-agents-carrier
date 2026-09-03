"""The result size budget, `total_path` projection, and spilling to disk.

The failure these guard against is not a wrong answer, it is a dead pod: a
paginated read of a few tens of MB parses into several hundred MB of Python
objects inside a 1 GiB limit, and the single uvicorn worker that dies takes
every in-flight run with it, leaving them at status "running" for ever.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    OperationDefinition,
    Paginate,
)
from app.domain.models.datastream import as_data_ref, is_data_ref
from app.infrastructure.datasources.executor import DataSourceExecutor
from app.infrastructure.datasources.datastream import (
    LocalDiskStreamStore,
    NotStreamable,
    ResultTooLarge,
    StreamGone,
    StreamReadTooLarge,
)

PAGE_ITEMS = 200
TOTAL_PAGES = 20
TOTAL_ITEMS = PAGE_ITEMS * TOTAL_PAGES


def _row(i: int) -> dict:
    return {"id": i, "name": f"row-{i}", "blob": "x" * 300}


def _source(*, total_path=None, **kw) -> DataSourceDefinition:
    body = dict(
        id="big",
        base_url="https://api.test",
        operations=[
            OperationDefinition(
                name="list_things",
                path="/things",
                paginate=Paginate(
                    type="page", param="page", max_pages=50,
                    items_path="items", total_path=total_path,
                ),
            )
        ],
    )
    body.update(kw)
    return DataSourceDefinition(**body)


def _transport(report_total: int | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(dict(request.url.params).get("page", 1))
        if page > TOTAL_PAGES:
            return httpx.Response(200, json={"items": []})
        start = (page - 1) * PAGE_ITEMS
        body: dict = {"items": [_row(start + i) for i in range(PAGE_ITEMS)]}
        if report_total is not None:
            body["total"] = report_total
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.fixture
def patched_client(monkeypatch):
    """Point the executor's httpx client at the mock transport."""
    import app.infrastructure.datasources.executor as mod

    def install(report_total: int | None = None) -> None:
        real = httpx.AsyncClient

        def factory(**kwargs):
            return real(transport=_transport(report_total), **kwargs)

        monkeypatch.setattr(mod.httpx, "AsyncClient", factory)

    return install


@pytest.fixture
def store(tmp_path) -> LocalDiskStreamStore:
    return LocalDiskStreamStore(tmp_path / "streams")


# ---------------------------------------------------------------------------
# everything is a stream
# ---------------------------------------------------------------------------

async def test_even_a_one_record_result_is_a_stream_reference(patched_client, store):
    """No threshold. A tiny result goes to a file like any other.

    This is the whole design: one path, so a workflow behaves the same in test
    as in production instead of flipping shape at a size nobody can predict.
    """
    patched_client()
    tiny = DataSourceDefinition(
        id="tiny", base_url="https://api.test",
        operations=[OperationDefinition(
            name="one", path="/things",
            mapping="items[0:1]",
        )],
    )

    result = await DataSourceExecutor(stream_store=store).execute(tiny, "one", {})

    assert is_data_ref(result)
    ref = as_data_ref(result)
    assert ref.items == 1
    assert [r async for r in store.stream(ref)] == [_row(0)]


async def test_a_large_result_is_the_same_shape_as_a_small_one(patched_client, store):
    patched_client()
    result = await DataSourceExecutor(stream_store=store).execute(
        _source(), "list_things", {}
    )

    assert is_data_ref(result)
    ref = as_data_ref(result)
    assert ref.items == TOTAL_ITEMS
    # What enters workflow state is tiny either way.
    assert len(json.dumps(result)) < 2000


async def test_an_unpaginated_dict_response_keeps_its_document_shape(
    patched_client, store
):
    """One dict response is a document, not a one-record list.

    `result_mode: ram` and every existing `mapping` depend on getting the dict
    back, so the ref records which it was.
    """
    patched_client()
    plain = DataSourceDefinition(
        id="plain", base_url="https://api.test",
        operations=[OperationDefinition(name="one", path="/things")],
    )

    ref = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(plain, "one", {})
    )

    assert ref.shape == "value"
    assert ref.items == 1
    whole = await store.read_all(ref, max_bytes=100 * 1024 * 1024)
    assert isinstance(whole, dict)
    assert len(whole["items"]) == PAGE_ITEMS


async def test_a_document_shaped_stream_cannot_be_iterated(patched_client, store):
    patched_client()
    plain = DataSourceDefinition(
        id="plain", base_url="https://api.test",
        operations=[OperationDefinition(name="one", path="/things")],
    )
    ref = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(plain, "one", {})
    )

    with pytest.raises(NotStreamable, match="not a list of records"):
        [x async for x in store.stream(ref)]


async def test_no_store_configured_is_refused_not_worked_around(patched_client):
    """Without a store there is nowhere for a result to go. Say so."""
    patched_client()
    with pytest.raises(ValueError, match="no data.*stream store is configured"):
        await DataSourceExecutor(stream_store=None).execute(
            _source(), "list_things", {}
        )


# ---------------------------------------------------------------------------
# reading a stream back
# ---------------------------------------------------------------------------

async def test_a_stream_reference_is_small_whatever_the_result(
    patched_client, store
):
    patched_client()
    result = await DataSourceExecutor(stream_store=store).execute(
        _source(), "list_things", {}
    )

    assert is_data_ref(result)
    handle = as_data_ref(result)
    assert handle.items == TOTAL_ITEMS
    assert handle.bytes > 1_000_000
    assert not handle.truncated
    # The whole point: what goes into the checkpoint is tiny regardless.
    assert len(json.dumps(result)) < 2000


async def test_a_stream_reads_back_complete_and_in_order(
    patched_client, store
):
    patched_client()
    result = await DataSourceExecutor(stream_store=store).execute(
        _source(), "list_things", {}
    )
    handle = as_data_ref(result)

    seen = [item async for item in store.stream(handle)]

    assert len(seen) == TOTAL_ITEMS
    assert seen[0] == _row(0)
    assert seen[-1] == _row(TOTAL_ITEMS - 1)


async def test_chunks_respect_the_item_cap_and_the_byte_cap(patched_client, store):
    patched_client()
    handle = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(
            _source(), "list_things", {}
        )
    )

    by_items = [len(c) async for c in store.chunks(handle, size=150)]
    assert max(by_items) == 150
    assert sum(by_items) == TOTAL_ITEMS

    by_bytes = [c async for c in store.chunks(handle, size=10**9, max_bytes=20 * 1024)]
    assert sum(len(c) for c in by_bytes) == TOTAL_ITEMS
    # Each chunk is under the cap, give or take the one record that is allowed
    # to exceed it alone rather than be split.
    assert all(len(json.dumps(c).encode()) <= 21 * 1024 for c in by_bytes)


async def test_a_cached_reference_whose_file_is_gone_is_refetched(
    patched_client, store, caplog
):
    """A live cache entry is not proof the stream is still readable.

    Files are swept on a TTL and lost on a restart, so a hit is verified
    before it is trusted -- otherwise a cache hit would hand back a reference
    to nothing.
    """
    patched_client()
    executor = DataSourceExecutor(stream_store=store)
    source = _source()
    source.cache.ttl_seconds = 600

    first = as_data_ref(await executor.execute(source, "list_things", {}))
    assert executor._cache, "expected the reference to be cached"

    # Same call again: served from cache, same stream.
    again = as_data_ref(await executor.execute(source, "list_things", {}))
    assert again.id == first.id

    # Now the file goes away, as a TTL sweep or a restart would take it.
    await store.delete(first)
    with caplog.at_level("INFO"):
        third = as_data_ref(await executor.execute(source, "list_things", {}))

    assert third.id != first.id
    assert any("is gone -- refetching" in r.message for r in caplog.records)
    assert third.items == TOTAL_ITEMS


# ---------------------------------------------------------------------------
# total_path
# ---------------------------------------------------------------------------

async def test_total_path_projects_the_finished_size_from_the_first_page(
    patched_client, store, caplog
):
    patched_client(report_total=TOTAL_ITEMS)
    with caplog.at_level("INFO"):
        result = await DataSourceExecutor(stream_store=store).execute(
            _source(total_path="total"), "list_things", {}
        )

    assert as_data_ref(result).items == TOTAL_ITEMS
    assert any(
        f"API reports {TOTAL_ITEMS} total records" in r.message for r in caplog.records
    )


async def test_a_projection_over_the_ceiling_is_reported_at_page_one(
    patched_client, store, caplog
):
    """The value of knowing early: warn before the walk, not after it."""
    patched_client(report_total=TOTAL_ITEMS)
    with caplog.at_level("WARNING"):
        await DataSourceExecutor(stream_store=store).execute(
            _source(total_path="total", max_result_bytes=200 * 1024),
            "list_things", {},
        )

    projected = [r for r in caplog.records if "exceeds max_result_bytes" in r.message]
    assert projected, "expected a projection warning"
    # Warned on the projection, before the ceiling was actually hit.
    hit = [r for r in caplog.records if "reached max_result_bytes" in r.message]
    assert caplog.records.index(projected[0]) < caplog.records.index(hit[0])


async def test_a_missing_or_unusable_total_is_simply_ignored(patched_client, store):
    """No total reported: the ceiling still works, just reactively."""
    patched_client(report_total=None)
    result = await DataSourceExecutor(stream_store=store).execute(
        _source(total_path="total"), "list_things", {}
    )

    assert as_data_ref(result).items == TOTAL_ITEMS


# ---------------------------------------------------------------------------
# ceilings
# ---------------------------------------------------------------------------

async def test_max_result_bytes_truncates_and_says_so_on_the_handle(
    patched_client, store
):
    patched_client()
    result = await DataSourceExecutor(stream_store=store).execute(
        _source(max_result_bytes=300 * 1024),
        "list_things", {},
    )

    handle = as_data_ref(result)
    assert handle.truncated is True
    assert 0 < handle.items < TOTAL_ITEMS
    assert handle.bytes < 500 * 1024


async def test_max_pages_marks_the_stream_truncated(patched_client, store):
    """The log always said "probably incomplete"; the handle now says it too."""
    patched_client()
    source = _source()
    source.operations[0].paginate.max_pages = 5

    handle = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(source, "list_things", {})
    )

    assert handle.truncated is True
    assert handle.items == 5 * PAGE_ITEMS


async def test_a_failed_fetch_leaves_no_orphan_stream_file(patched_client, store, tmp_path):
    import app.infrastructure.datasources.executor as mod

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] > 2:
            return httpx.Response(500, json={"boom": True})
        page = int(dict(request.url.params).get("page", 1))
        start = (page - 1) * PAGE_ITEMS
        return httpx.Response(
            200, json={"items": [_row(start + i) for i in range(PAGE_ITEMS)]}
        )

    real = httpx.AsyncClient
    mod.httpx.AsyncClient = lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await DataSourceExecutor(stream_store=store).execute(
                _source(), "list_things", {}
            )
    finally:
        mod.httpx.AsyncClient = real

    assert list((tmp_path / "streams").glob("ds_*.jsonl")) == []


# ---------------------------------------------------------------------------
# reading it back
# ---------------------------------------------------------------------------

async def test_read_all_refuses_past_its_limit(patched_client, store):
    patched_client()
    handle = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(
            _source(), "list_things", {}
        )
    )

    with pytest.raises(StreamReadTooLarge):
        await store.read_all(handle, max_bytes=1024)

    whole = await store.read_all(handle, max_bytes=100 * 1024 * 1024)
    assert len(whole) == TOTAL_ITEMS


async def test_a_purged_stream_reports_itself_gone_rather_than_reading_short(
    patched_client, store
):
    patched_client()
    handle = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(
            _source(), "list_things", {}
        )
    )
    await store.delete(handle)

    with pytest.raises(StreamGone, match="does not survive a pod restart"):
        [x async for x in store.stream(handle)]


async def test_purge_older_than_sweeps_by_age(patched_client, store, tmp_path):
    patched_client()
    await DataSourceExecutor(stream_store=store).execute(
        _source(), "list_things", {}
    )

    assert await store.purge_older_than(3600) == 0
    assert await store.purge_older_than(0) == 1
    assert list((tmp_path / "streams").glob("ds_*.jsonl")) == []


# ---------------------------------------------------------------------------
# fan-out over a spilled upstream
# ---------------------------------------------------------------------------

def _fanout_source(**kw) -> DataSourceDefinition:
    body = dict(
        id="fan",
        base_url="https://api.test",
        operations=[
            OperationDefinition(
                name="list_things", path="/things",
                paginate=Paginate(
                    type="page", param="page", max_pages=50, items_path="items"
                ),
            ),
            OperationDefinition(name="detail", path="/things/{list_things.id}"),
        ],
    )
    body.update(kw)
    return DataSourceDefinition(**body)


@pytest.fixture
def fanout_client(monkeypatch):
    import app.infrastructure.datasources.executor as mod

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/things":
            page = int(dict(request.url.params).get("page", 1))
            if page > 2:
                return httpx.Response(200, json={"items": []})
            start = (page - 1) * PAGE_ITEMS
            return httpx.Response(
                200, json={"items": [_row(start + i) for i in range(PAGE_ITEMS)]}
            )
        seen.append(path)
        return httpx.Response(200, json={"detail_for": path.rsplit("/", 1)[-1]})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        mod.httpx, "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    return seen


async def test_fanout_over_a_streamed_upstream_calls_every_element(
    fanout_client, store
):
    """Without this the handle would render into the URL and run once."""
    result = await DataSourceExecutor(stream_store=store).execute(
        _fanout_source(), "detail", {}
    )

    calls = fanout_client
    assert len(calls) == 2 * PAGE_ITEMS
    assert "/things/0" in calls and f"/things/{2 * PAGE_ITEMS - 1}" in calls
    # The fan-out result itself went through the same budget.
    handle = as_data_ref(result)
    if handle is not None:
        assert handle.items == 2 * PAGE_ITEMS
    else:
        assert len(result) == 2 * PAGE_ITEMS


async def test_the_approval_preview_counts_a_streamed_upstream_exactly(
    fanout_client, store
):
    """An approver must see 400 rows, not "1 affected row"."""
    source = _fanout_source()
    source.operations[1].method = "DELETE"

    plan = await DataSourceExecutor(stream_store=store).preview(source, "detail", {})

    assert plan.affected_rows == 2 * PAGE_ITEMS
    # Only the capped preview is read off disk, not the whole upstream.
    assert len(plan.targets) == 20
    assert len(plan.sample) == 20
