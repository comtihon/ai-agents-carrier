"""Tests for the data source DAG executor.

httpx is stubbed with a fake AsyncClient so no network access is needed: each
test supplies a handler that maps (method, url) onto a JSON payload and records
every request.
"""
from __future__ import annotations

import base64

import pytest

from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    validate_operations,
)
from app.infrastructure.datasources import executor as executor_module
from app.infrastructure.datasources.executor import DataSourceExecutor


# ---------------------------------------------------------------------------
# httpx stub
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, handler, calls: list[dict]) -> None:
        self._handler = handler
        self._calls = calls

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def request(self, method, url, params=None, headers=None, json=None):
        call = {
            "method": method,
            "url": url,
            "params": dict(params or {}),
            "headers": dict(headers or {}),
            "json": json,
        }
        self._calls.append(call)
        payload = self._handler(call)
        if isinstance(payload, FakeResponse):
            return payload
        return FakeResponse(payload)

    async def post(self, url, json=None, headers=None):
        return await self.request("POST", url, json=json, headers=headers)


@pytest.fixture
def http(monkeypatch):
    """Install the fake client; returns a recorder with a settable handler."""
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.handler = lambda call: {}

        def count(self, fragment: str) -> int:
            return sum(1 for c in self.calls if fragment in c["url"])

    recorder = Recorder()

    def _factory(*args, **kwargs):
        return FakeClient(lambda call: recorder.handler(call), recorder.calls)

    monkeypatch.setattr(executor_module.httpx, "AsyncClient", _factory)
    return recorder


def _source(**overrides) -> DataSourceDefinition:
    data = {
        "id": "api",
        "name": "API",
        "base_url": "https://api.test",
        "operations": [],
    }
    data.update(overrides)
    return DataSourceDefinition.model_validate(data)


# ---------------------------------------------------------------------------
# DAG resolution
# ---------------------------------------------------------------------------

async def test_linear_dag_passes_upstream_field_downstream(http):
    source = _source(operations=[
        {"name": "whoami", "path": "/me"},
        {"name": "profile", "path": "/users/{whoami.id}/profile"},
    ])

    def handler(call):
        if call["url"].endswith("/me"):
            return {"id": "u7"}
        return {"bio": "hello"}

    http.handler = handler
    result = await DataSourceExecutor().execute(source, "profile", {})
    assert result == {"bio": "hello"}
    assert http.calls[-1]["url"] == "https://api.test/users/u7/profile"


async def test_diamond_dag_memoises_shared_upstream(http):
    source = _source(operations=[
        {"name": "base", "path": "/base"},
        {"name": "left", "path": "/left/{base.id}"},
        {"name": "right", "path": "/right/{base.id}"},
        {"name": "merge", "path": "/merge/{left.v}/{right.v}"},
    ])

    def handler(call):
        if "/base" in call["url"]:
            return {"id": "b1"}
        if "/left" in call["url"]:
            return {"v": "L"}
        if "/right" in call["url"]:
            return {"v": "R"}
        return {"done": True}

    http.handler = handler
    result = await DataSourceExecutor().execute(source, "merge", {})
    assert result == {"done": True}
    assert http.count("/base") == 1
    assert http.calls[-1]["url"] == "https://api.test/merge/L/R"


async def test_fanout_over_array_upstream(http):
    source = _source(operations=[
        {"name": "list_repos", "path": "/repos", "mapping": "items"},
        {"name": "languages", "path": "/repos/{list_repos.name}/languages"},
    ])

    def handler(call):
        if call["url"].endswith("/repos"):
            return {"items": [{"name": "a"}, {"name": "b"}]}
        return {"Python": 100}

    http.handler = handler
    result = await DataSourceExecutor().execute(source, "languages", {})
    assert result == [
        {"name": "a", "result": {"Python": 100}},
        {"name": "b", "result": {"Python": 100}},
    ]
    assert http.count("/languages") == 2


async def test_fanout_rejects_two_array_upstreams(http):
    source = _source(operations=[
        {"name": "xs", "path": "/xs", "mapping": "items"},
        {"name": "ys", "path": "/ys", "mapping": "items"},
        {"name": "combine", "path": "/c/{xs.name}/{ys.name}"},
    ])
    http.handler = lambda call: {"items": [{"name": "a"}]}

    with pytest.raises(ValueError, match="more than one array upstream"):
        await DataSourceExecutor().execute(source, "combine", {})


async def test_unknown_operation_raises(http):
    with pytest.raises(ValueError, match="no operation 'nope'"):
        await DataSourceExecutor().execute(_source(), "nope", {})


async def test_missing_required_param_raises(http):
    source = _source(operations=[
        {"name": "op", "path": "/x/{params.owner}", "params": [{"name": "owner"}]},
    ])
    with pytest.raises(ValueError, match="missing required param"):
        await DataSourceExecutor().execute(source, "op", {})


async def test_missing_required_param_on_upstream_op_raises_before_any_call(http):
    """Required params must be validated for every op in the dependency
    closure, not just the target — so a missing upstream param fails clearly
    up front instead of surfacing as a confusing downstream error."""
    source = _source(operations=[
        {"name": "up", "path": "/up/{params.owner}", "params": [{"name": "owner"}]},
        {"name": "down", "path": "/down/{up.id}"},
    ])
    with pytest.raises(ValueError, match="Operation 'up' is missing required param.*owner"):
        await DataSourceExecutor().execute(source, "down", {})
    assert http.calls == []


async def test_loose_params_become_query_string(http):
    source = _source(operations=[
        {"name": "search", "path": "/search", "params": [
            {"name": "q"},
            {"name": "limit", "type": "number", "required": False},
        ]},
    ])
    http.handler = lambda call: {"ok": True}
    await DataSourceExecutor().execute(source, "search", {"q": "cats", "limit": 5})
    assert http.calls[0]["params"] == {"q": "cats", "limit": 5}


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

async def test_cursor_pagination_follows_cursor_path(http):
    source = _source(operations=[
        {"name": "items", "path": "/items", "mapping": "items",
         "paginate": {"type": "cursor", "param": "after", "cursor_path": "next", "max_pages": 5}},
    ])
    pages = [
        {"items": [1, 2], "next": "c1"},
        {"items": [3], "next": None},
    ]
    http.handler = lambda call: pages[0] if "after" not in call["params"] else pages[1]

    result = await DataSourceExecutor().execute(source, "items", {})
    assert result == [1, 2, 3]
    assert http.calls[1]["params"] == {"after": "c1"}


async def test_page_pagination_stops_on_empty_page(http):
    source = _source(operations=[
        {"name": "items", "path": "/items", "mapping": "items",
         "paginate": {"type": "page", "param": "page", "max_pages": 5}},
    ])
    payloads = {1: {"items": ["a"]}, 2: {"items": ["b"]}, 3: {"items": []}}
    http.handler = lambda call: payloads[call["params"]["page"]]

    result = await DataSourceExecutor().execute(source, "items", {})
    assert result == ["a", "b"]
    assert [c["params"]["page"] for c in http.calls] == [1, 2, 3]


async def test_offset_pagination_advances_by_item_count(http):
    source = _source(operations=[
        {"name": "items", "path": "/items", "mapping": "items",
         "paginate": {"type": "offset", "param": "offset", "max_pages": 5}},
    ])
    payloads = {0: {"items": ["a", "b"]}, 2: {"items": ["c"]}, 3: {"items": []}}
    http.handler = lambda call: payloads[call["params"]["offset"]]

    result = await DataSourceExecutor().execute(source, "items", {})
    assert result == ["a", "b", "c"]
    assert [c["params"]["offset"] for c in http.calls] == [0, 2, 3]


async def test_page_pagination_without_mapping_stops_via_items_path(http):
    """Without a `mapping`, a dict-shaped page never looks "empty" and
    pagination used to silently run to max_pages, returning raw page dicts.
    `items_path` fixes this for both the stop-check and concatenation."""
    source = _source(operations=[
        {"name": "items", "path": "/items",
         "paginate": {"type": "page", "param": "page", "max_pages": 5, "items_path": "data"}},
    ])
    payloads = {
        1: {"data": ["a"], "total": 3},
        2: {"data": ["b"], "total": 3},
        3: {"data": [], "total": 3},
    }
    http.handler = lambda call: payloads[call["params"]["page"]]

    result = await DataSourceExecutor().execute(source, "items", {})
    assert result == ["a", "b"]
    assert [c["params"]["page"] for c in http.calls] == [1, 2, 3]


async def test_pagination_respects_max_pages(http):
    source = _source(operations=[
        {"name": "items", "path": "/items", "mapping": "items",
         "paginate": {"type": "page", "param": "page", "max_pages": 2}},
    ])
    http.handler = lambda call: {"items": ["x"]}
    result = await DataSourceExecutor().execute(source, "items", {})
    assert result == ["x", "x"]
    assert len(http.calls) == 2


# ---------------------------------------------------------------------------
# Cache / retry
# ---------------------------------------------------------------------------

async def test_cache_hit_within_ttl_then_miss_after_expiry(http, monkeypatch):
    source = _source(
        cache={"ttl_seconds": 60},
        operations=[{"name": "op", "path": "/x"}],
    )
    http.handler = lambda call: {"v": 1}
    now = {"t": 1000.0}
    monkeypatch.setattr(executor_module.time, "monotonic", lambda: now["t"])

    ex = DataSourceExecutor()
    assert await ex.execute(source, "op", {}) == {"v": 1}
    assert await ex.execute(source, "op", {}) == {"v": 1}
    assert len(http.calls) == 1  # served from cache

    now["t"] += 61
    await ex.execute(source, "op", {})
    assert len(http.calls) == 2  # TTL expired


async def test_cache_key_includes_resolved_upstream_value(http):
    """Regression: an operation whose template binds an upstream operation's
    result must not serve one upstream value's cached response for another."""
    source = _source(
        cache={"ttl_seconds": 60},
        operations=[
            {"name": "up", "path": "/up/{params.id}", "params": [{"name": "id"}]},
            {"name": "down", "path": "/down/{up.value}"},
        ],
    )
    responses = {"A": "down-for-A", "B": "down-for-B"}

    def handler(call):
        if "/up/" in call["url"]:
            uid = call["url"].rsplit("/", 1)[-1]
            return {"value": uid}
        upstream_value = call["url"].rsplit("/", 1)[-1]
        return {"result": responses[upstream_value]}

    http.handler = handler
    ex = DataSourceExecutor()

    result_a = await ex.execute(source, "down", {"id": "A"})
    assert result_a == {"result": "down-for-A"}
    result_b = await ex.execute(source, "down", {"id": "B"})
    assert result_b == {"result": "down-for-B"}
    # Both up() and down() must have been called twice — once per distinct id.
    assert http.count("/up/") == 2
    assert http.count("/down/") == 2

    # Re-running with the same id as before must still hit the cache.
    await ex.execute(source, "down", {"id": "A"})
    assert http.count("/up/") == 2
    assert http.count("/down/") == 2


async def test_cache_disabled_by_default(http):
    source = _source(operations=[{"name": "op", "path": "/x"}])
    http.handler = lambda call: {"v": 1}
    ex = DataSourceExecutor()
    await ex.execute(source, "op", {})
    await ex.execute(source, "op", {})
    assert len(http.calls) == 2


async def test_retry_recovers_after_transient_failure(http, monkeypatch):
    source = _source(
        retries={"attempts": 3, "backoff": 0.0},
        operations=[{"name": "op", "path": "/x"}],
    )
    calls = {"n": 0}

    def handler(call):
        calls["n"] += 1
        if calls["n"] < 3:
            return FakeResponse({}, status_code=500)
        return {"ok": True}

    http.handler = handler
    async def _no_sleep(_delay):
        return None
    monkeypatch.setattr(executor_module.asyncio, "sleep", _no_sleep)

    assert await DataSourceExecutor().execute(source, "op", {}) == {"ok": True}
    assert calls["n"] == 3


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

async def test_bearer_auth_reads_env_at_call_time(http, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "sekret")
    source = _source(
        auth={"type": "bearer", "token_env": "MY_TOKEN"},
        default_headers={"Accept": "application/json"},
        operations=[{"name": "op", "path": "/x"}],
    )
    http.handler = lambda call: {}
    await DataSourceExecutor().execute(source, "op", {})
    assert http.calls[0]["headers"] == {
        "Accept": "application/json",
        "Authorization": "Bearer sekret",
    }


async def test_basic_auth_encodes_env_credentials(http, monkeypatch):
    monkeypatch.setenv("U", "alice")
    monkeypatch.setenv("P", "pw")
    source = _source(
        auth={"type": "basic", "username_env": "U", "password_env": "P"},
        operations=[{"name": "op", "path": "/x"}],
    )
    http.handler = lambda call: {}
    await DataSourceExecutor().execute(source, "op", {})
    expected = base64.b64encode(b"alice:pw").decode()
    assert http.calls[0]["headers"]["Authorization"] == f"Basic {expected}"


async def test_header_auth_uses_configured_header(http, monkeypatch):
    monkeypatch.setenv("API_KEY_ENV", "k123")
    source = _source(
        auth={"type": "header", "header_name": "X-Api-Key", "value_env": "API_KEY_ENV"},
        operations=[{"name": "op", "path": "/x"}],
    )
    http.handler = lambda call: {}
    await DataSourceExecutor().execute(source, "op", {})
    assert http.calls[0]["headers"]["X-Api-Key"] == "k123"


# ---------------------------------------------------------------------------
# GraphQL / mapping / schema
# ---------------------------------------------------------------------------

async def test_graphql_posts_query_and_variables(http):
    source = _source(
        kind="graphql",
        base_url="https://api.test/graphql",
        operations=[{
            "name": "repo",
            "query": "query($owner:String!){ repo(owner:$owner){ name } }",
            "variables": {"owner": "{params.owner}"},
            "params": [{"name": "owner"}],
            "mapping": "data.repo.name",
        }],
    )
    http.handler = lambda call: {"data": {"repo": {"name": "langgraph"}}}

    result = await DataSourceExecutor().execute(source, "repo", {"owner": "acme"})
    assert result == "langgraph"
    call = http.calls[0]
    assert call["url"] == "https://api.test/graphql"
    assert call["json"]["variables"] == {"owner": "acme"}


async def test_mapping_extracts_with_jmespath(http):
    source = _source(operations=[
        {"name": "op", "path": "/x", "mapping": "items[].name"},
    ])
    http.handler = lambda call: {"items": [{"name": "a"}, {"name": "b"}]}
    assert await DataSourceExecutor().execute(source, "op", {}) == ["a", "b"]


async def test_response_schema_missing_required_key_raises(http):
    source = _source(operations=[
        {"name": "op", "path": "/x",
         "response_schema": {"type": "object", "required": ["id"]}},
    ])
    http.handler = lambda call: {"other": 1}
    with pytest.raises(ValueError, match="missing required key 'id'"):
        await DataSourceExecutor().execute(source, "op", {})


async def test_response_schema_wrong_property_type_raises(http):
    source = _source(operations=[
        {"name": "op", "path": "/x",
         "response_schema": {"type": "object", "properties": {"id": {"type": "number"}}}},
    ])
    http.handler = lambda call: {"id": "not-a-number"}
    with pytest.raises(ValueError, match="expected number"):
        await DataSourceExecutor().execute(source, "op", {})


async def test_response_schema_accepts_valid_payload(http):
    source = _source(operations=[
        {"name": "op", "path": "/x",
         "response_schema": {"type": "object", "required": ["id"],
                             "properties": {"id": {"type": "string"}}}},
    ])
    http.handler = lambda call: {"id": "abc"}
    assert await DataSourceExecutor().execute(source, "op", {}) == {"id": "abc"}


# ---------------------------------------------------------------------------
# Save-time validation
# ---------------------------------------------------------------------------

def test_validate_operations_rejects_cycles():
    source = _source(operations=[
        {"name": "a", "path": "/a/{b.id}"},
        {"name": "b", "path": "/b/{a.id}"},
    ])
    with pytest.raises(ValueError, match="Cyclic"):
        validate_operations(source)


def test_validate_operations_rejects_self_reference():
    source = _source(operations=[{"name": "a", "path": "/a/{a.id}"}])
    with pytest.raises(ValueError, match="references itself"):
        validate_operations(source)


def test_validate_operations_rejects_unknown_operation():
    source = _source(operations=[{"name": "a", "path": "/a/{ghost.id}"}])
    with pytest.raises(ValueError, match="unknown operation 'ghost'"):
        validate_operations(source)


def test_validate_operations_rejects_unknown_param():
    source = _source(operations=[{"name": "a", "path": "/a/{params.nope}"}])
    with pytest.raises(ValueError, match="unknown param 'nope'"):
        validate_operations(source)


def test_validate_operations_accepts_valid_dag():
    source = _source(operations=[
        {"name": "a", "path": "/a/{params.x}", "params": [{"name": "x"}]},
        {"name": "b", "path": "/b/{a.id}"},
        {"name": "c", "path": "/c/{a.id}/{b.id}"},
    ])
    validate_operations(source)  # does not raise
