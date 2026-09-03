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


# ─── Google Sheets ────────────────────────────────────────────────────────────
#
# Nothing here reaches Google: the impersonated-token mint is stubbed at
# app.infrastructure.auth.google_token_provider._mint_token and the Drive call
# goes through a fake httpx client, the same way the probe tests fake theirs.

GOOGLE_SA = "copilot@engineering-368717.iam.gserviceaccount.com"

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789/edit#gid=0"
)
SHEET_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


def _google_settings(**overrides) -> Settings:
    return Settings(GOOGLE_IMPERSONATE_SA=GOOGLE_SA, **overrides)


@pytest.fixture
async def google_client(monkeypatch):
    """Client whose backend has GOOGLE_IMPERSONATE_SA set and no real Google."""
    from app.core.config import get_settings
    from app.infrastructure.auth import google_token_provider

    google_token_provider.reset_token_cache()
    monkeypatch.setattr(
        google_token_provider, "_mint_token", lambda subject, scopes: ("tok", 3600.0)
    )

    settings = _google_settings()
    backend = InMemoryDataSourceBackend()
    app = create_app()
    container = _build_container(backend)
    object.__setattr__(container, "settings", settings) if hasattr(
        container, "__dataclass_fields__"
    ) else setattr(container, "settings", settings)
    app.state.container = container
    app.dependency_overrides[get_settings] = lambda: settings
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, backend
    google_token_provider.reset_token_cache()


def _fake_drive(status_code: int, payload: dict | None = None, recorder: list | None = None):
    """A fake httpx.AsyncClient factory answering the Drive files.get call."""
    class _Resp:
        def __init__(self) -> None:
            self.status_code = status_code
            self.text = "drive says no"

        def json(self):
            return payload or {}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, params=None, headers=None):
            if recorder is not None:
                recorder.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
            return _Resp()

    return lambda **kwargs: _Client()


async def test_google_resolve_returns_the_document_it_can_see(google_client, monkeypatch):
    c, _ = google_client
    calls: list[dict] = []
    monkeypatch.setattr(
        "app.infrastructure.datasources.google_sheets.httpx.AsyncClient",
        _fake_drive(200, {
            "id": SHEET_ID,
            "name": "Q3 pipeline",
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "capabilities": {"canEdit": True},
        }, calls),
    )

    resp = await c.post("/api/v1/datasources/google/resolve", json={"ref": SHEET_URL})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "status": "ok",
        "error": None,
        "file_id": SHEET_ID,
        "name": "Q3 pipeline",
        "mime_type": "application/vnd.google-apps.spreadsheet",
        "can_edit": True,
        "service_account": GOOGLE_SA,
    }
    # Asked Drive as the impersonated account, for exactly the fields the UI
    # shows, and in a way that also works for a file on a shared drive.
    assert calls[0]["url"].endswith(f"/drive/v3/files/{SHEET_ID}")
    assert calls[0]["params"] == {
        "fields": "id,name,mimeType,capabilities/canEdit",
        "supportsAllDrives": "true",
    }
    assert calls[0]["headers"]["Authorization"] == "Bearer tok"


async def test_google_resolve_accepts_a_bare_file_id(google_client, monkeypatch):
    """The Picker hands back an id, not a URL."""
    c, _ = google_client
    monkeypatch.setattr(
        "app.infrastructure.datasources.google_sheets.httpx.AsyncClient",
        _fake_drive(200, {
            "id": SHEET_ID, "name": "Sheet", "capabilities": {"canEdit": False},
            "mimeType": "application/vnd.google-apps.spreadsheet",
        }),
    )
    resp = await c.post("/api/v1/datasources/google/resolve", json={"ref": SHEET_ID})
    assert resp.json()["status"] == "ok"
    assert resp.json()["can_edit"] is False


async def test_google_resolve_reports_no_access_with_the_address_to_share_with(
    google_client, monkeypatch
):
    """403/404 is the expected first answer, not a 500."""
    c, _ = google_client
    monkeypatch.setattr(
        "app.infrastructure.datasources.google_sheets.httpx.AsyncClient",
        _fake_drive(404),
    )
    resp = await c.post("/api/v1/datasources/google/resolve", json={"ref": SHEET_URL})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_access"
    assert body["service_account"] == GOOGLE_SA
    assert GOOGLE_SA in body["error"]


async def test_google_resolve_names_a_doc_as_a_doc(google_client):
    c, _ = google_client
    resp = await c.post(
        "/api/v1/datasources/google/resolve",
        json={"ref": "https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUv/edit"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "wrong_type"
    assert "Google Doc" in body["error"]


async def test_google_resolve_names_a_slides_deck_as_a_slides_deck(google_client):
    c, _ = google_client
    resp = await c.post(
        "/api/v1/datasources/google/resolve",
        json={"ref": "https://docs.google.com/presentation/d/1AbCdEfGhIjKlMnOpQ/edit"},
    )
    assert resp.json()["status"] == "wrong_type"
    assert "Slides" in resp.json()["error"]


async def test_google_resolve_rejects_something_that_is_not_a_drive_ref(google_client):
    c, _ = google_client
    resp = await c.post(
        "/api/v1/datasources/google/resolve", json={"ref": "https://example.com/nope"}
    )
    assert resp.json()["status"] == "invalid"


async def test_google_resolve_says_so_when_the_backend_is_not_configured(client):
    """No GOOGLE_IMPERSONATE_SA — a deployment gap, still not a 5xx."""
    c, _ = client
    resp = await c.post("/api/v1/datasources/google/resolve", json={"ref": SHEET_URL})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_configured"
    assert "GOOGLE_IMPERSONATE_SA" in body["error"]


async def test_sheets_template_is_saveable_through_the_normal_create_path(google_client):
    c, backend = google_client
    template = (await c.get("/api/v1/datasources/google/sheets-template")).json()
    assert template["base_url"] == "https://sheets.googleapis.com"
    assert template["auth"]["type"] == "google"
    assert template["service_account"] == GOOGLE_SA
    assert template["default_value_input_option"] == "RAW"

    payload = {k: template[k] for k in
               ("id", "name", "description", "kind", "base_url", "auth", "operations")}
    resp = await c.post("/api/v1/datasources", json=payload)
    assert resp.status_code == 201, resp.text

    stored = await backend.get("google-sheets")
    assert stored is not None
    names = [op.name for op in stored.operations]
    assert names == [
        "get_metadata", "get_values", "batch_get_values",
        "update_values", "batch_update_values", "append_values",
    ]
    # Every write is gated; no read is.
    gated = {op.name for op in stored.operations if op.destructive}
    assert gated == {"update_values", "batch_update_values", "append_values"}
    # Append is not idempotent, so it is not retried.
    append = stored.get_operation("append_values")
    assert append is not None and append.retries is not None
    assert append.retries.attempts == 1
    # RAW, not USER_ENTERED — a value starting with "=" must stay text.
    value_input = next(
        p for p in append.params if p.name == "value_input_option"
    )
    assert value_input.default == "RAW"


async def test_google_auth_carries_no_secret_to_redact(google_client):
    c, _ = google_client
    template = (await c.get("/api/v1/datasources/google/sheets-template")).json()
    payload = {k: template[k] for k in
               ("id", "name", "description", "kind", "base_url", "auth", "operations")}
    await c.post("/api/v1/datasources", json=payload)

    body = (await c.get("/api/v1/datasources/google-sheets")).json()
    assert body["auth"]["type"] == "google"
    # Nothing to redact, and nothing invented either.
    assert "********" not in str(body["auth"])
    assert body["auth"]["impersonate_subject"] is None


async def test_create_refuses_a_google_block_naming_another_service_account(google_client):
    """The auth block is caller-supplied; the target principal is not."""
    c, _ = google_client
    resp = await c.post("/api/v1/datasources", json={
        "id": "sneaky",
        "base_url": "https://sheets.googleapis.com",
        "auth": {
            "type": "google",
            "impersonate_subject": "prod-admin@engineering-368717.iam.gserviceaccount.com",
        },
        "operations": [],
    })
    assert resp.status_code == 422
    assert GOOGLE_SA in resp.json()["detail"]


async def test_create_accepts_a_google_block_naming_the_configured_account(google_client):
    c, _ = google_client
    resp = await c.post("/api/v1/datasources", json={
        "id": "sheets",
        "base_url": "https://sheets.googleapis.com",
        "auth": {"type": "google", "impersonate_subject": GOOGLE_SA},
        "operations": [],
    })
    assert resp.status_code == 201, resp.text


async def test_update_refuses_a_google_block_naming_another_service_account(google_client):
    c, backend = google_client
    await c.post("/api/v1/datasources", json={
        "id": "sheets", "base_url": "https://sheets.googleapis.com",
        "auth": {"type": "google"}, "operations": [],
    })
    resp = await c.put("/api/v1/datasources/sheets", json={
        "auth": {"type": "google", "impersonate_subject": "other@x.iam.gserviceaccount.com"},
    })
    assert resp.status_code == 422
    stored = await backend.get("sheets")
    assert stored is not None and stored.auth.impersonate_subject is None


async def test_google_auth_cannot_take_its_subject_from_backend_config(google_client):
    """`from_config` is for secrets, and this auth type has none."""
    c, _ = google_client
    resp = await c.post("/api/v1/datasources", json={
        "id": "sheets", "base_url": "https://sheets.googleapis.com",
        "auth": {"type": "google", "from_config": "SOME_KEY"}, "operations": [],
    })
    assert resp.status_code == 422
    assert "no secret" in resp.json()["detail"]
