"""Tests for the /api/v1/datasources CRUD API."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.datasources.executor import DataSourceExecutor
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.data_source_backend import DataSourceDefinitionBackend
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


class InMemoryDataSourceBackend(DataSourceDefinitionBackend):
    def __init__(self) -> None:
        self._store: dict[str, DataSourceDefinition] = {}

    async def list(self) -> list[DataSourceDefinition]:
        return list(self._store.values())

    async def get(self, source_id: str) -> DataSourceDefinition | None:
        return self._store.get(source_id)

    async def create(self, definition: DataSourceDefinition) -> DataSourceDefinition:
        definition.touch()
        self._store[definition.id] = definition
        return definition

    async def update(self, source_id: str, definition: DataSourceDefinition) -> DataSourceDefinition:
        definition.touch()
        self._store[source_id] = definition
        return definition

    async def delete(self, source_id: str) -> None:
        self._store.pop(source_id, None)


def _build_container(backend: DataSourceDefinitionBackend | None) -> ApplicationContainer:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.refresh_server = AsyncMock()
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    return ApplicationContainer(
        settings=Settings(),
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=AsyncMock(spec=MongoGraphRunRepository),
        openhands=MagicMock(spec=OpenHandsAdapter),
        data_source_backend=backend,
        data_source_executor=DataSourceExecutor() if backend else None,
    )


def _payload(**overrides) -> dict:
    body = {
        "id": "github",
        "name": "GitHub",
        "base_url": "https://api.github.com",
        "auth": {"type": "bearer", "token": "gh-secret-token"},
        "operations": [
            {
                "name": "list_repos",
                "path": "/users/{params.owner}/repos",
                "params": [{"name": "owner"}],
            }
        ],
    }
    body.update(overrides)
    return body


@pytest.fixture
async def client():
    backend = InMemoryDataSourceBackend()
    app = create_app()
    app.state.container = _build_container(backend)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, backend


@pytest.fixture
async def client_without_backend():
    app = create_app()
    app.state.container = _build_container(None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_create_get_list_update_delete_roundtrip(client):
    c, backend = client

    resp = await c.post("/api/v1/datasources", json=_payload())
    assert resp.status_code == 201
    # Secrets are stored but never echoed back.
    assert resp.json()["auth"]["token"] == "********"
    assert (await backend.get("github")).auth.token == "gh-secret-token"

    resp = await c.get("/api/v1/datasources")
    assert resp.status_code == 200
    assert [s["id"] for s in resp.json()] == ["github"]

    resp = await c.get("/api/v1/datasources/github")
    assert resp.status_code == 200
    assert resp.json()["operations"][0]["name"] == "list_repos"

    resp = await c.put("/api/v1/datasources/github", json={"description": "code host"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["description"] == "code host"
    # Omitted fields are preserved (auth secret intact in the store, redacted here).
    assert data["operations"][0]["name"] == "list_repos"
    assert data["auth"]["token"] == "********"
    assert (await backend.get("github")).auth.token == "gh-secret-token"

    resp = await c.delete("/api/v1/datasources/github")
    assert resp.status_code == 204
    assert await backend.get("github") is None


async def test_list_and_get_redact_auth_secrets(client):
    c, _ = client
    await c.post("/api/v1/datasources", json=_payload())

    listed = (await c.get("/api/v1/datasources")).json()
    assert listed[0]["auth"] == {"type": "bearer", "token": "********"}

    fetched = (await c.get("/api/v1/datasources/github")).json()
    assert fetched["auth"] == {"type": "bearer", "token": "********"}


async def test_summary_view_keeps_operation_methods_and_drops_the_rest(client):
    """The list view aggregates a risk badge over every operation's method, so a
    summary has to keep name+method -- but not the paths, params or response
    schemas, which are what make an imported OpenAPI source large.  ``auth`` is
    dropped outright rather than redacted: a summary carries no secret shape."""
    c, _ = client
    await c.post("/api/v1/datasources", json=_payload())

    summary = (await c.get("/api/v1/datasources", params={"view": "summary"})).json()

    assert summary == [{
        "id": "github",
        "name": "GitHub",
        "description": None,
        "kind": "http",
        "base_url": "https://api.github.com",
        "operations": [{"name": "list_repos", "method": "GET"}],
    }]
    # Default view is unchanged, secrets still redacted.
    full = (await c.get("/api/v1/datasources")).json()[0]
    assert full["auth"] == {"type": "bearer", "token": "********"}
    assert full["operations"][0]["path"] == "/users/{params.owner}/repos"


async def test_update_with_redacted_secret_preserves_stored_value(client):
    c, backend = client
    await c.post("/api/v1/datasources", json=_payload())

    resp = await c.put(
        "/api/v1/datasources/github",
        json={"auth": {"type": "bearer", "token": "********"}, "name": "GH"},
    )
    assert resp.status_code == 200
    assert resp.json()["auth"]["token"] == "********"
    assert (await backend.get("github")).auth.token == "gh-secret-token"


async def test_update_with_real_secret_overwrites(client):
    c, backend = client
    await c.post("/api/v1/datasources", json=_payload())

    resp = await c.put(
        "/api/v1/datasources/github",
        json={"auth": {"type": "bearer", "token": "new-token"}},
    )
    assert resp.status_code == 200
    assert (await backend.get("github")).auth.token == "new-token"


async def test_update_auth_type_switch_requires_real_secret(client):
    c, _ = client
    await c.post("/api/v1/datasources", json=_payload())

    resp = await c.put(
        "/api/v1/datasources/github",
        json={"auth": {"type": "header", "header_name": "X-Api-Key", "value": "********"}},
    )
    assert resp.status_code == 422


async def test_create_rejects_redaction_placeholder_as_secret(client):
    c, _ = client
    resp = await c.post(
        "/api/v1/datasources",
        json=_payload(auth={"type": "bearer", "token": "********"}),
    )
    assert resp.status_code == 422


async def test_create_resolves_bearer_token_from_config(client, monkeypatch):
    c, backend = client
    monkeypatch.setenv("AFP_SERVICE_TOKEN", "resolved-service-token")

    resp = await c.post(
        "/api/v1/datasources",
        json=_payload(auth={"type": "bearer", "from_config": "AFP_SERVICE_TOKEN"}),
    )
    assert resp.status_code == 201
    # The reference never reaches the stored definition, and the resolved value
    # is redacted in the response like any other secret.
    assert resp.json()["auth"] == {"type": "bearer", "token": "********"}
    assert (await backend.get("github")).auth.token == "resolved-service-token"


async def test_create_resolves_header_value_from_config(client, monkeypatch):
    c, backend = client
    monkeypatch.setenv("AFP_SERVICE_TOKEN", "resolved-service-token")

    resp = await c.post(
        "/api/v1/datasources",
        json=_payload(auth={"type": "header", "header_name": "X-Api-Key", "from_config": "AFP_SERVICE_TOKEN"}),
    )
    assert resp.status_code == 201
    stored = (await backend.get("github")).auth
    assert stored.header_name == "X-Api-Key"
    assert stored.value == "resolved-service-token"


async def test_update_resolves_token_from_config(client, monkeypatch):
    c, backend = client
    await c.post("/api/v1/datasources", json=_payload())
    monkeypatch.setenv("AFP_SERVICE_TOKEN", "rotated-service-token")

    resp = await c.put(
        "/api/v1/datasources/github",
        json={"auth": {"type": "bearer", "from_config": "AFP_SERVICE_TOKEN"}},
    )
    assert resp.status_code == 200
    assert (await backend.get("github")).auth.token == "rotated-service-token"


async def test_create_with_unset_config_key_returns_422(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("AFP_SERVICE_TOKEN", raising=False)

    resp = await c.post(
        "/api/v1/datasources",
        json=_payload(auth={"type": "bearer", "from_config": "AFP_SERVICE_TOKEN"}),
    )
    assert resp.status_code == 422
    assert "AFP_SERVICE_TOKEN" in resp.text


async def test_create_with_blank_config_key_returns_422(client):
    c, _ = client
    resp = await c.post(
        "/api/v1/datasources",
        json=_payload(auth={"type": "bearer", "from_config": "  "}),
    )
    assert resp.status_code == 422


async def test_from_config_on_secretless_auth_type_returns_422(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("AFP_SERVICE_TOKEN", "resolved-service-token")

    resp = await c.post(
        "/api/v1/datasources",
        json=_payload(auth={"type": "none", "from_config": "AFP_SERVICE_TOKEN"}),
    )
    assert resp.status_code == 422


async def test_create_duplicate_conflicts(client):
    c, _ = client
    assert (await c.post("/api/v1/datasources", json=_payload())).status_code == 201
    assert (await c.post("/api/v1/datasources", json=_payload())).status_code == 409


async def test_get_unknown_returns_404(client):
    c, _ = client
    assert (await c.get("/api/v1/datasources/nope")).status_code == 404


async def test_create_with_cycle_returns_422(client):
    c, _ = client
    resp = await c.post("/api/v1/datasources", json=_payload(operations=[
        {"name": "a", "path": "/a/{b.id}"},
        {"name": "b", "path": "/b/{a.id}"},
    ]))
    assert resp.status_code == 422
    assert "Cyclic" in resp.json()["detail"]


async def test_create_with_unknown_operation_ref_returns_422(client):
    c, _ = client
    resp = await c.post("/api/v1/datasources", json=_payload(operations=[
        {"name": "a", "path": "/a/{ghost.id}"},
    ]))
    assert resp.status_code == 422
    assert "unknown operation" in resp.json()["detail"]


async def test_create_with_unknown_param_ref_returns_422(client):
    c, _ = client
    resp = await c.post("/api/v1/datasources", json=_payload(operations=[
        {"name": "a", "path": "/a/{params.missing}"},
    ]))
    assert resp.status_code == 422
    assert "unknown param" in resp.json()["detail"]


async def test_routes_return_501_without_backend(client_without_backend):
    c = client_without_backend
    assert (await c.get("/api/v1/datasources")).status_code == 501
    assert (await c.post("/api/v1/datasources", json=_payload())).status_code == 501
    assert (await c.get("/api/v1/datasources/github")).status_code == 501


# ─── /datasources/try-operation ───────────────────────────────────────────────

def _try_payload(**overrides) -> dict:
    body = {
        "base_url": "https://api.example.com",
        "auth": {"type": "bearer", "token": "real-token"},
        "operations": [{"name": "list_items", "path": "/items"}],
        "operation": "list_items",
    }
    body.update(overrides)
    return body


async def test_try_operation_returns_sample_and_mapping(client, monkeypatch):
    c, _ = client

    async def fake_execute(self, source, operation, params):
        assert operation == "list_items"
        # mapping/schema/pagination must be stripped from the target op
        op = source.get_operation(operation)
        assert op.mapping is None and op.response_schema is None and op.paginate is None
        return {"content": [{"id": 1}, {"id": 2}]}

    async def fake_suggest(sample, settings):
        return "content[].{id: id}"

    monkeypatch.setattr(DataSourceExecutor, "execute", fake_execute)
    monkeypatch.setattr("app.infrastructure.datasources.try_run.suggest_mapping", fake_suggest)

    resp = await c.post(
        "/api/v1/datasources/try-operation",
        json=_try_payload(operations=[{"name": "list_items", "path": "/items", "mapping": "content", "paginate": {"param": "page"}}]),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["api_output"] == {"content": [{"id": 1}, {"id": 2}]}
    assert data["suggested_mapping"] == "content[].{id: id}"


async def test_try_operation_encodes_target_failure(client, monkeypatch):
    c, _ = client

    async def fake_execute(self, source, operation, params):
        raise ValueError("boom")

    monkeypatch.setattr(DataSourceExecutor, "execute", fake_execute)

    resp = await c.post("/api/v1/datasources/try-operation", json=_try_payload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "error"
    assert "boom" in data["error"]
    assert data["api_output"] is None


async def test_try_operation_unknown_operation_returns_422(client):
    c, _ = client
    resp = await c.post("/api/v1/datasources/try-operation", json=_try_payload(operation="ghost"))
    assert resp.status_code == 422


async def test_try_operation_merges_stored_secret_for_redacted_auth(client, monkeypatch):
    c, backend = client
    await c.post("/api/v1/datasources", json=_payload())  # stores github with real token

    seen: dict = {}

    async def fake_execute(self, source, operation, params):
        seen["auth"] = source.auth
        return {"ok": True}

    async def fake_suggest(sample, settings):
        return None

    monkeypatch.setattr(DataSourceExecutor, "execute", fake_execute)
    monkeypatch.setattr("app.infrastructure.datasources.try_run.suggest_mapping", fake_suggest)

    resp = await c.post(
        "/api/v1/datasources/try-operation",
        json=_try_payload(
            source_id="github",
            auth={"type": "bearer", "token": "********"},
            operations=[{"name": "list_repos", "path": "/repos"}],
            operation="list_repos",
        ),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert seen["auth"].token == "gh-secret-token"


async def test_try_operation_rejects_placeholder_without_source(client):
    c, _ = client
    resp = await c.post(
        "/api/v1/datasources/try-operation",
        json=_try_payload(auth={"type": "bearer", "token": "********"}),
    )
    assert resp.status_code == 422


def test_shrink_sample_caps_lists_and_strings():
    from app.infrastructure.datasources.try_run import shrink_sample

    value = {"items": list(range(10)), "text": "x" * 1000, "nested": [{"a": list(range(5))}] * 4}
    out = shrink_sample(value)
    assert out["items"] == [0, 1, 2]
    assert len(out["text"]) == 501
    assert len(out["nested"]) == 3
    assert out["nested"][0]["a"] == [0, 1, 2]


# ─── Schema import ────────────────────────────────────────────────────────────

_OPENAPI_DOC = {
    "openapi": "3.0.0",
    "servers": [{"url": "https://api.test/v1"}],
    "paths": {
        "/pets": {
            "get": {"operationId": "listPets", "summary": "List pets"},
            "post": {
                "operationId": "createPet",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {"name": {"type": "string"}},
                            },
                        },
                    },
                },
            },
        },
    },
}


async def test_schema_fetch_returns_the_operation_pick_list(client, monkeypatch):
    c, backend = client
    seen: dict = {}

    async def fake_fetch(schema_url, kind="http", auth=None, max_operations=0):
        seen.update(url=schema_url, kind=kind, auth=auth)
        from app.infrastructure.datasources.spec import openapi_to_operations
        return {
            "kind": "openapi",
            "source": schema_url,
            "base_url": "https://api.test/v1",
            "operations": openapi_to_operations(_OPENAPI_DOC, max_operations=max_operations),
        }

    monkeypatch.setattr("app.api.routes.datasources.fetch_and_parse_spec", fake_fetch)

    resp = await c.post(
        "/api/v1/datasources/schema/fetch",
        json={
            "schema_url": "https://api.test/openapi.json",
            "kind": "http",
            "auth": {"type": "bearer", "token": "spec-token"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "openapi"
    assert body["base_url"] == "https://api.test/v1"
    assert [op["name"] for op in body["operations"]] == ["listpets", "createpet"]
    # Nothing is stored — the caller saves the subset it wants.
    assert await backend.list() == []
    assert seen["auth"].token == "spec-token"


async def test_schema_fetch_maps_a_bad_url_onto_422(client, monkeypatch):
    c, _ = client

    async def fake_fetch(*args, **kwargs):
        from app.infrastructure.datasources.discovery import SpecFetchError
        raise SpecFetchError("Schema URL returned HTTP 404 (request failed)")

    monkeypatch.setattr("app.api.routes.datasources.fetch_and_parse_spec", fake_fetch)

    resp = await c.post(
        "/api/v1/datasources/schema/fetch",
        json={"schema_url": "https://api.test/nope.json"},
    )
    assert resp.status_code == 422
    assert "404" in resp.json()["detail"]


async def test_schema_fetch_resolves_the_secret_from_backend_config(client, monkeypatch):
    c, _ = client
    seen: dict = {}

    monkeypatch.setattr(
        Settings, "get_forwardable_config", lambda self: {"SPEC_TOKEN": "from-config"}
    )

    async def fake_fetch(schema_url, kind="http", auth=None, max_operations=0):
        seen["auth"] = auth
        return {"kind": "openapi", "source": schema_url, "base_url": None, "operations": []}

    monkeypatch.setattr("app.api.routes.datasources.fetch_and_parse_spec", fake_fetch)

    resp = await c.post(
        "/api/v1/datasources/schema/fetch",
        json={
            "schema_url": "https://api.test/openapi.json",
            "auth": {"type": "bearer", "from_config": "SPEC_TOKEN"},
        },
    )
    assert resp.status_code == 200
    assert seen["auth"].token == "from-config"


async def test_schema_upload_parses_an_uploaded_document(client):
    import json as _json

    c, _ = client
    resp = await c.post(
        "/api/v1/datasources/schema/upload",
        files={"file": ("petstore.json", _json.dumps(_OPENAPI_DOC), "application/json")},
        data={"kind": "http"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "petstore.json"
    assert body["base_url"] == "https://api.test/v1"
    create = next(op for op in body["operations"] if op["name"] == "createpet")
    assert create["params"] == [
        {"name": "name", "type": "string", "required": True, "description": ""}
    ]


async def test_schema_upload_rejects_a_non_specification(client):
    c, _ = client
    resp = await c.post(
        "/api/v1/datasources/schema/upload",
        files={"file": ("notes.txt", "just some notes", "text/plain")},
    )
    assert resp.status_code == 422
    assert "Unrecognised specification" in resp.json()["detail"]


async def test_probe_no_longer_hunts_for_an_openapi_document(client, monkeypatch):
    """The probe reports reachability; an HTTP source's schema is imported explicitly."""
    c, _ = client
    requested: list[str] = []

    class _Resp:
        status_code = 200

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            requested.append(url)
            return _Resp()

        async def request(self, method, url, headers=None, json=None):
            requested.append(url)
            return _Resp()

    monkeypatch.setattr(
        "app.infrastructure.datasources.discovery.httpx.AsyncClient",
        lambda **kwargs: _Client(),
    )

    resp = await c.post(
        "/api/v1/datasources/probe",
        json={"base_url": "https://api.test", "kind": "http"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["url_status"] == "ok"
    assert body["discovered"] is None
    # Only the base URL was touched — no /openapi.json, /swagger.json, …
    assert requested == ["https://api.test"]


# ─── Pub/Sub sources moved out ────────────────────────────────────────────────

async def test_a_pubsub_datasource_is_refused_and_points_at_events(client):
    """Topics are events now — see tests/test_events_api.py."""
    c, backend = client

    resp = await c.post("/api/v1/datasources", json={
        "id": "orders-events",
        "name": "Order events",
        "kind": "pubsub",
        "pubsub": {"topic": "orders"},
    })

    assert resp.status_code == 422
    assert "/events" in resp.json()["detail"]
    assert await backend.get("orders-events") is None
