"""Paginated reads against a Django REST Framework endpoint.

DRF's PageNumberPagination raises NotFound past the last page, so such an
endpoint never returns the empty page the executor's stop-check waits for -- it
404s. Before this was handled, every paginated DRF read failed at the final page
boundary, and the failure looked like a broken request rather than a fetch that
had actually finished.

The distinction that matters: a 404 *after* at least one good page is the end of
the data; a 404 on the first request is a wrong path and must still fail.
"""
from __future__ import annotations

import httpx
import pytest

from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.datasources.executor import DataSourceExecutor


def _source(paginate: dict | None = None, **op_extra) -> DataSourceDefinition:
    op: dict = {
        "name": "list_things",
        "method": "GET",
        "path": "things/",
        "params": [],
        **op_extra,
    }
    if paginate is not None:
        op["paginate"] = paginate
    return DataSourceDefinition.model_validate({
        "id": "drf-api",
        "name": "DRF API",
        "kind": "http",
        "base_url": "https://example.test",
        "operations": [op],
    })


def _drf_pages(pages: list[list[dict]], total: int) -> httpx.MockTransport:
    """Serve *pages* 1-indexed, 404ing past the end exactly as DRF does."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", 1))
        calls.append(page)
        if page < 1 or page > len(pages):
            return httpx.Response(404, json={"detail": "Invalid page."})
        return httpx.Response(200, json={
            "count": total,
            "next": None if page == len(pages) else "…",
            "previous": None,
            "results": pages[page - 1],
        })

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    return transport


async def _run(source: DataSourceDefinition, transport: httpx.MockTransport):
    executor = DataSourceExecutor()
    # The executor opens its own client, so hand it the mock transport.
    import app.infrastructure.datasources.executor as mod

    real_client = mod.httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    mod.httpx.AsyncClient = _client  # type: ignore[assignment]
    try:
        return await executor.execute(source, "list_things", {})
    finally:
        mod.httpx.AsyncClient = real_client  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_a_404_past_the_last_page_ends_the_fetch_instead_of_failing():
    """Two full pages then a 404: the caller should get all four rows."""
    transport = _drf_pages([
        [{"id": 1}, {"id": 2}],
        [{"id": 3}, {"id": 4}],
    ], total=4)
    source = _source({"type": "page", "param": "page", "items_path": "results", "max_pages": 10})

    rows = await _run(source, transport)

    assert rows == [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    assert transport.calls == [1, 2, 3], "it must probe page 3 and stop on the 404"


@pytest.mark.asyncio
async def test_a_404_on_the_very_first_page_still_fails():
    """Otherwise a wrong path would quietly read as "no data"."""
    transport = _drf_pages([], total=0)
    source = _source({"type": "page", "param": "page", "items_path": "results", "max_pages": 10})

    with pytest.raises(httpx.HTTPStatusError):
        await _run(source, transport)


@pytest.mark.asyncio
async def test_max_pages_caps_the_fetch():
    """The cap is a real ceiling -- the executor logs, but returns a short read."""
    transport = _drf_pages([[{"id": i}] for i in range(1, 11)], total=10)
    source = _source({"type": "page", "param": "page", "items_path": "results", "max_pages": 3})

    rows = await _run(source, transport)

    assert rows == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert transport.calls == [1, 2, 3], "must not request a fourth page"


@pytest.mark.asyncio
async def test_an_empty_page_still_ends_the_fetch():
    """Endpoints that answer with an empty list rather than 404 keep working."""
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", 1))
        results = [{"id": 1}] if page == 1 else []
        return httpx.Response(200, json={"count": 1, "results": results})

    source = _source({"type": "page", "param": "page", "items_path": "results", "max_pages": 10})

    rows = await _run(source, httpx.MockTransport(handler))

    assert rows == [{"id": 1}]


@pytest.mark.asyncio
async def test_an_unpaginated_operation_is_untouched():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "page" not in request.url.params
        return httpx.Response(200, json={"results": [{"id": 1}]})

    rows = await _run(_source(None), httpx.MockTransport(handler))

    assert rows == {"results": [{"id": 1}]}
