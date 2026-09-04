"""Every pagination shape a real API uses, walked automatically.

Three kinds (cursor, page, offset) across three places the arguments can go
(query string, JSON body, GraphQL variables), flat and nested. The body case
is the one that was missing: `extra` only ever reached the query string, so
HubSpot's POST object search -- which takes `after` and `limit` in the body --
could not declare pagination at all and had to be walked by hand.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    OperationDefinition,
    Paginate,
    validate_operations,
)
from app.domain.models.datastream import as_data_ref
from app.infrastructure.datasources.datastream import LocalDiskStreamStore
from app.infrastructure.datasources.executor import DataSourceExecutor

TOTAL = 250
PAGE = 100


def _row(i: int) -> dict:
    return {"id": i}


@pytest.fixture
def store(tmp_path) -> LocalDiskStreamStore:
    return LocalDiskStreamStore(tmp_path / "streams")


def _slice(skip: int, limit: int) -> list[dict]:
    return [_row(i) for i in range(skip, min(skip + limit, TOTAL))]


def _install(monkeypatch, handler):
    import app.infrastructure.datasources.executor as mod

    real = httpx.AsyncClient
    monkeypatch.setattr(
        mod.httpx, "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )


async def _run(store, source, op="items", **kw):
    return as_data_ref(
        await DataSourceExecutor(stream_store=store).execute(source, op, {}, **kw)
    )


# ---------------------------------------------------------------------------
# HTTP, query string
# ---------------------------------------------------------------------------

async def test_http_query_offset(monkeypatch, store):
    seen: list[dict] = []

    def handler(req):
        q = dict(req.url.params)
        seen.append(q)
        return httpx.Response(200, json={"items": _slice(int(q["offset"]), int(q["size"]))})

    _install(monkeypatch, handler)
    source = DataSourceDefinition(
        id="s", base_url="https://api", operations=[OperationDefinition(
            name="items", method="GET", path="/items", mapping="items",
            paginate=Paginate(type="offset", param="offset", size_param="size",
                              page_size=PAGE, max_pages=0),
        )])

    ref = await _run(store, source)

    assert ref.items == TOTAL
    assert [int(c["offset"]) for c in seen] == [0, 100, 200, 250]


async def test_http_query_page_numbers(monkeypatch, store):
    seen: list[int] = []

    def handler(req):
        q = dict(req.url.params)
        page = int(q["page"])
        seen.append(page)
        return httpx.Response(200, json={"items": _slice((page - 1) * PAGE, PAGE)})

    _install(monkeypatch, handler)
    source = DataSourceDefinition(
        id="s", base_url="https://api", operations=[OperationDefinition(
            name="items", method="GET", path="/items", mapping="items",
            paginate=Paginate(type="page", param="page", max_pages=0),
        )])

    ref = await _run(store, source)

    assert ref.items == TOTAL
    assert seen == [1, 2, 3, 4]


async def test_http_query_cursor(monkeypatch, store):
    """HubSpot's GET list shape: paging.next.after."""
    def handler(req):
        q = dict(req.url.params)
        skip = int(q.get("after") or 0)
        rows = _slice(skip, PAGE)
        nxt = skip + len(rows)
        body: dict = {"results": rows}
        if nxt < TOTAL:
            body["paging"] = {"next": {"after": str(nxt)}}
        return httpx.Response(200, json=body)

    _install(monkeypatch, handler)
    source = DataSourceDefinition(
        id="s", base_url="https://api", operations=[OperationDefinition(
            name="items", method="GET", path="/items", mapping="results",
            paginate=Paginate(type="cursor", param="after",
                              cursor_path="paging.next.after",
                              items_path="results", max_pages=0),
        )])

    ref = await _run(store, source)
    assert ref.items == TOTAL


# ---------------------------------------------------------------------------
# HTTP, JSON body — the gap this closes
# ---------------------------------------------------------------------------

async def test_http_body_cursor(monkeypatch, store):
    """HubSpot's POST search: `after` and `limit` live in the body."""
    bodies: list[dict] = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        skip = int(body.get("after") or 0)
        rows = _slice(skip, int(body.get("limit") or PAGE))
        nxt = skip + len(rows)
        out: dict = {"results": rows, "total": TOTAL}
        if nxt < TOTAL:
            out["paging"] = {"next": {"after": str(nxt)}}
        return httpx.Response(200, json=out)

    _install(monkeypatch, handler)
    source = DataSourceDefinition(
        id="s", base_url="https://api", operations=[OperationDefinition(
            name="items", method="POST", path="/items/search", mapping="results",
            paginate=Paginate(type="cursor", param="after", size_param="limit",
                              cursor_path="paging.next.after",
                              items_path="results", page_size=PAGE, max_pages=0),
        )])

    ref = await _run(store, source)

    assert ref.items == TOTAL
    # The arguments reached the BODY, not the query string.
    assert [b.get("after") for b in bodies] == [None, "100", "200"]
    assert all(b["limit"] == PAGE for b in bodies)


async def test_http_body_pagination_never_lands_in_the_query_string(monkeypatch, store):
    urls: list[str] = []

    def handler(req):
        urls.append(str(req.url))
        body = json.loads(req.content)
        return httpx.Response(200, json={"items": _slice(int(body.get("offset") or 0), PAGE)})

    _install(monkeypatch, handler)
    source = DataSourceDefinition(
        id="s", base_url="https://api", operations=[OperationDefinition(
            name="items", method="POST", path="/items", mapping="items",
            paginate=Paginate(type="offset", param="offset", max_pages=0),
        )])

    await _run(store, source)
    assert all("offset" not in u for u in urls)


async def test_http_body_nested_pagination(monkeypatch, store):
    """A body whose paging sits inside an object."""
    bodies: list[dict] = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        page = body["paging"]
        return httpx.Response(200, json={"items": _slice(page["skip"], page["take"])})

    _install(monkeypatch, handler)
    source = DataSourceDefinition(
        id="s", base_url="https://api", operations=[OperationDefinition(
            name="items", method="POST", path="/items", mapping="items",
            paginate=Paginate(type="offset", param="paging.skip",
                              size_param="paging.take", page_size=PAGE,
                              max_pages=0),
        )])

    ref = await _run(store, source)

    assert ref.items == TOTAL
    assert bodies[0]["paging"] == {"skip": 0, "take": PAGE}


async def test_a_body_paginated_post_keeps_its_own_params(monkeypatch, store):
    """Pagination must not displace the operation's real body arguments."""
    bodies: list[dict] = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        return httpx.Response(200, json={"items": _slice(int(body.get("offset") or 0), PAGE)})

    _install(monkeypatch, handler)
    source = DataSourceDefinition(
        id="s", base_url="https://api", operations=[OperationDefinition(
            name="items", method="POST", path="/items", mapping="items",
            params=[{"name": "query", "type": "string", "required": False}],
            paginate=Paginate(type="offset", param="offset", max_pages=0),
        )])

    await DataSourceExecutor(stream_store=store).execute(
        source, "items", {"query": "acme"}
    )

    assert bodies[0]["query"] == "acme"
    assert bodies[0]["offset"] == 0


# ---------------------------------------------------------------------------
# GraphQL variables, flat and nested
# ---------------------------------------------------------------------------

def _gql_source(paginate: Paginate) -> DataSourceDefinition:
    return DataSourceDefinition(
        id="g", kind="graphql", base_url="https://g/graphql",
        operations=[OperationDefinition(
            name="items", method="POST",
            query="query items { items { id } }",
            mapping="data.items", paginate=paginate,
        )])


@pytest.mark.parametrize("paginate,read", [
    (
        Paginate(type="offset", param="skip", size_param="limit",
                 page_size=PAGE, max_pages=0),
        lambda v: (int(v.get("skip") or 0), int(v.get("limit") or PAGE)),
    ),
    (
        Paginate(type="offset", param="pagination.skip",
                 size_param="pagination.limit", page_size=PAGE, max_pages=0),
        lambda v: (int(v["pagination"]["skip"]), int(v["pagination"]["limit"])),
    ),
    (
        Paginate(type="page", param="page", max_pages=0),
        lambda v: ((int(v["page"]) - 1) * PAGE, PAGE),
    ),
])
async def test_graphql_variable_pagination(monkeypatch, store, paginate, read):
    def handler(req):
        variables = json.loads(req.content)["variables"]
        skip, limit = read(variables)
        return httpx.Response(200, json={"data": {"items": _slice(skip, limit)}})

    _install(monkeypatch, handler)
    ref = await _run(store, _gql_source(paginate))
    assert ref.items == TOTAL


async def test_graphql_relay_cursor(monkeypatch, store):
    """first / after / pageInfo.endCursor."""
    def handler(req):
        variables = json.loads(req.content)["variables"]
        skip = int(variables.get("after") or 0)
        size = int(variables.get("first") or PAGE)
        rows = _slice(skip, size)
        nxt = skip + len(rows)
        return httpx.Response(200, json={"data": {"items": {
            "nodes": rows,
            "pageInfo": {"endCursor": str(nxt) if nxt < TOTAL else None},
        }}})

    _install(monkeypatch, handler)
    source = DataSourceDefinition(
        id="g", kind="graphql", base_url="https://g/graphql",
        operations=[OperationDefinition(
            name="items", method="POST",
            query="query items($first: Int, $after: String) { items(first: $first, after: $after) { nodes { id } pageInfo { endCursor } } }",
            mapping="data.items.nodes",
            paginate=Paginate(type="cursor", param="after", size_param="first",
                              cursor_path="data.items.pageInfo.endCursor",
                              page_size=PAGE, max_pages=0),
        )])

    ref = await _run(store, source)
    assert ref.items == TOTAL


# ---------------------------------------------------------------------------
# a row limit applies to every kind
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("paginate,method,kind,mapping", [
    (Paginate(type="offset", param="offset", size_param="size", page_size=PAGE,
              max_pages=0), "GET", "http", "items"),
    (Paginate(type="page", param="page", max_pages=0), "GET", "http", "items"),
    (Paginate(type="offset", param="offset", size_param="size", page_size=PAGE,
              max_pages=0), "POST", "http", "items"),
    (Paginate(type="offset", param="pagination.skip",
              size_param="pagination.limit", page_size=PAGE, max_pages=0),
     "POST", "graphql", "data.items"),
])
async def test_a_row_limit_works_for_every_kind(monkeypatch, store, paginate,
                                                method, kind, mapping):
    def handler(req):
        if kind == "graphql":
            v = json.loads(req.content)["variables"]
            pag = v.get("pagination") or v
            skip, size = int(pag.get("skip") or 0), int(pag.get("limit") or PAGE)
            return httpx.Response(200, json={"data": {"items": _slice(skip, size)}})
        src = dict(req.url.params) if method == "GET" else json.loads(req.content)
        if "page" in src:
            skip, size = (int(src["page"]) - 1) * PAGE, PAGE
        else:
            skip, size = int(src.get("offset") or 0), int(src.get("size") or PAGE)
        return httpx.Response(200, json={"items": _slice(skip, size)})

    _install(monkeypatch, handler)
    source = DataSourceDefinition(
        id="s", kind=kind, base_url="https://api",
        operations=[OperationDefinition(
            name="items", method=method, path="/items",
            query="query items { items { id } }" if kind == "graphql" else None,
            mapping=mapping, paginate=paginate,
        )])

    ref = await _run(store, source, limit=120)
    assert ref.items == 120
    assert ref.truncated is False


# ---------------------------------------------------------------------------
# declarations that cannot work are refused at save time
# ---------------------------------------------------------------------------

def _validate(kind, method, **pg) -> str:
    op = OperationDefinition(
        name="op", method=method, path="/x",
        query="query { x }" if kind == "graphql" else None,
        paginate=Paginate(**pg),
    )
    d = DataSourceDefinition(id="d", kind=kind, base_url="https://x", operations=[op])
    try:
        validate_operations(d)
        return "ok"
    except ValueError as exc:
        return str(exc)


def test_a_nested_path_in_a_query_string_is_refused():
    msg = _validate("http", "GET", type="offset", param="pagination.skip")
    assert "nesting has no meaning" in msg


def test_a_nested_path_in_a_body_is_fine():
    assert _validate("http", "POST", type="offset", param="paging.skip") == "ok"


def test_a_cursor_without_a_cursor_path_is_refused():
    msg = _validate("http", "GET", type="cursor", param="after")
    assert "no cursor_path" in msg


def test_graphql_pagination_cannot_be_told_to_use_the_query_string():
    msg = _validate("graphql", "POST", type="page", param="page", location="query")
    assert "goes in its variables" in msg
