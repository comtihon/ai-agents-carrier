"""A caller says how many rows it wants; the data source does the paging.

Pagination is the data source's business. Before this, a workflow had to pass
`pagination: {limit: 50, skip: 0}` itself -- which meant the editor showed
`[object Object]`, an unpaginated read of control-center 502'd, and a step
could only ever get one page. Now the operation declares how to page and the
step declares how much it wants, or nothing at all for everything.
"""
from __future__ import annotations

import httpx
import pytest

from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    OperationDefinition,
    Paginate,
)
from app.domain.models.datastream import as_data_ref
from app.infrastructure.datasources.datastream import LocalDiskStreamStore
from app.infrastructure.datasources.executor import DataSourceExecutor, _set_path

TOTAL = 250


def _row(i: int) -> dict:
    return {"id": i, "name": f"p-{i}"}


@pytest.fixture
def store(tmp_path) -> LocalDiskStreamStore:
    return LocalDiskStreamStore(tmp_path / "streams")


def _graphql_source(**kw) -> DataSourceDefinition:
    """Control-center's shape: pagination is a nested PaginationInput."""
    return DataSourceDefinition(
        id="cc", kind="graphql", base_url="https://cc/graphql",
        operations=[
            OperationDefinition(
                name="projects",
                method="POST",
                query="query projects($pagination: PaginationInput) "
                      "{ projects(pagination: $pagination) { id name } }",
                variables={"pagination": None},
                mapping="data.projects",
                paginate=Paginate(
                    type="offset",
                    param="pagination.skip",
                    size_param="pagination.limit",
                    page_size=100,
                    max_pages=kw.pop("max_pages", 0),
                ),
            )
        ],
        **kw,
    )


@pytest.fixture
def gql(monkeypatch):
    """A GraphQL endpoint that pages via pagination.{limit, skip}."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        pag = (body.get("variables") or {}).get("pagination") or {}
        skip = int(pag.get("skip") or 0)
        limit = int(pag.get("limit") or 100)
        seen.append({"skip": skip, "limit": limit})
        rows = [_row(i) for i in range(skip, min(skip + limit, TOTAL))]
        return httpx.Response(200, json={"data": {"projects": rows}})

    import app.infrastructure.datasources.executor as mod
    real = httpx.AsyncClient
    monkeypatch.setattr(
        mod.httpx, "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    return seen


# ---------------------------------------------------------------------------
# nested pagination variables
# ---------------------------------------------------------------------------

def test_set_path_reaches_a_nested_input_field():
    v: dict = {"filterBy": {"x": 1}}
    _set_path(v, "pagination.skip", 100)
    _set_path(v, "pagination.limit", 50)
    _set_path(v, "page", 3)

    assert v == {"filterBy": {"x": 1}, "pagination": {"skip": 100, "limit": 50}, "page": 3}


def test_set_path_replaces_a_scalar_sitting_where_an_object_belongs():
    v: dict = {"pagination": "oops"}
    _set_path(v, "pagination.skip", 0)
    assert v == {"pagination": {"skip": 0}}


# ---------------------------------------------------------------------------
# no limit → every page
# ---------------------------------------------------------------------------

async def test_no_limit_walks_every_page(gql, store):
    ref = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(
            _graphql_source(), "projects", {}
        )
    )

    assert ref.items == TOTAL
    assert ref.truncated is False
    # 100 + 100 + 50, then one more that comes back empty and stops it.
    assert [c["skip"] for c in gql] == [0, 100, 200, 250]
    assert all(c["limit"] == 100 for c in gql)

    rows = [r async for r in store.stream(ref)]
    assert rows[0] == _row(0)
    assert rows[-1] == _row(TOTAL - 1)


async def test_max_pages_zero_means_no_ceiling(gql, store):
    """The default of 10 would have silently stopped at 1000 rows."""
    ref = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(
            _graphql_source(), "projects", {}
        )
    )
    assert ref.items == TOTAL
    assert ref.truncated is False


# ---------------------------------------------------------------------------
# a limit → exactly that many
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("limit,expected_pages", [(10, 1), (100, 1), (150, 2), (250, 3)])
async def test_a_limit_collects_exactly_that_many(gql, store, limit, expected_pages):
    ref = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(
            _graphql_source(), "projects", {}, limit=limit
        )
    )

    assert ref.items == limit
    # A limit is not a truncation: the caller got what it asked for.
    assert ref.truncated is False
    assert len(gql) == expected_pages


async def test_a_small_limit_asks_the_api_for_only_that_many(gql, store):
    """A limit of 10 must not fetch a 100-row page and discard 90."""
    await DataSourceExecutor(stream_store=store).execute(
        _graphql_source(), "projects", {}, limit=10
    )

    assert gql == [{"skip": 0, "limit": 10}]


async def test_a_limit_over_the_total_stops_at_the_total(gql, store):
    ref = as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(
            _graphql_source(), "projects", {}, limit=10_000
        )
    )
    assert ref.items == TOTAL


async def test_a_limit_of_zero_or_none_means_everything(gql, store):
    for limit in (None, 0, -5):
        ref = as_data_ref(
            await DataSourceExecutor(stream_store=store).execute(
                _graphql_source(), "projects", {}, limit=limit
            )
        )
        assert ref.items == TOTAL, limit


# ---------------------------------------------------------------------------
# the step field
# ---------------------------------------------------------------------------

def test_the_step_reads_a_blank_limit_as_no_limit():
    from app.infrastructure.orchestration.yaml_graph import _positive_int

    for blank in (None, "", 0, -3, "x"):
        assert _positive_int(blank) is None, blank
    assert _positive_int("10") == 10
    assert _positive_int(10) == 10


async def test_a_limited_and_an_unlimited_read_do_not_share_a_cache_entry(gql, store):
    """Serving one for the other would be silent and wrong."""
    source = _graphql_source()
    source.cache.ttl_seconds = 600
    ex = DataSourceExecutor(stream_store=store)

    small = as_data_ref(await ex.execute(source, "projects", {}, limit=10))
    everything = as_data_ref(await ex.execute(source, "projects", {}))

    assert small.items == 10
    assert everything.items == TOTAL
