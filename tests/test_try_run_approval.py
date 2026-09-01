"""The try-run button's gate on a destructive operation.

Try run is the one surface where the approver is already present — a person in
the editor, one click from deleting whatever the operation points at. So the
gate is a two-step rather than a wait: preview and refuse, then run on a
confirmed retry. These tests are about that handshake and about the
self-approval it leaves behind.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.application.approval_service import ApprovalService
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.datasources import executor as executor_module
from app.infrastructure.datasources.executor import DataSourceExecutor
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.approval_backend import InMemoryApprovalBackend
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


# ── httpx stub (same shape as tests/test_data_source_executor.py) ─────────────

class FakeResponse:
    def __init__(self, payload) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

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
        self._calls.append({"method": method, "url": url})
        return FakeResponse(self._handler(method, url))

    async def post(self, url, json=None, headers=None):
        return await self.request("POST", url, json=json, headers=headers)


@pytest.fixture
def http(monkeypatch):
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.handler = lambda method, url: (
                [{"id": "f1"}, {"id": "f2"}, {"id": "f3"}]
                if method == "GET" else {"ok": True}
            )

        def methods(self) -> list[str]:
            return [c["method"] for c in self.calls]

    recorder = Recorder()
    monkeypatch.setattr(
        executor_module.httpx,
        "AsyncClient",
        lambda *a, **k: FakeClient(recorder.handler, recorder.calls),
    )
    return recorder


# ── App ───────────────────────────────────────────────────────────────────────

def _container(backend) -> ApplicationContainer:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    settings = Settings()
    return ApplicationContainer(
        settings=settings,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=AsyncMock(spec=MongoGraphRunRepository),
        openhands=MagicMock(spec=OpenHandsAdapter),
        data_source_executor=DataSourceExecutor(),
        approval_backend=backend,
        approval_service=ApprovalService(backend, settings),
    )


@pytest.fixture
async def client(monkeypatch):
    # The mapping suggestion is a live meta-LLM call; it is not what is under
    # test and it must not reach a provider.
    monkeypatch.setattr(
        "app.infrastructure.datasources.try_run.suggest_mapping",
        AsyncMock(return_value=None),
    )
    backend = InMemoryApprovalBackend()
    app = create_app()
    app.state.container = _container(backend)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, backend


def _body(**overrides) -> dict:
    body = {
        "base_url": "https://files.test",
        "kind": "http",
        "operations": [
            {"name": "stale", "path": "/files?stale=true"},
            {"name": "drop", "method": "DELETE", "path": "/files/{stale.id}"},
            {"name": "list", "path": "/files"},
        ],
        "operation": "drop",
        "params": {},
    }
    body.update(overrides)
    return body


# ── The handshake ─────────────────────────────────────────────────────────────

async def test_a_destructive_try_run_previews_and_refuses(client, http):
    c, backend = client

    body = (await c.post("/api/v1/datasources/try-operation", json=_body())).json()

    assert body["status"] == "confirmation_required"
    assert body["destructive"]["affected_rows"] == 3
    assert body["destructive"]["method"] == "DELETE"
    assert body["destructive"]["affected_sample"] == ["f1", "f2", "f3"]
    # The upstream read happened; nothing was deleted.
    assert http.methods() == ["GET"]
    assert await backend.count() == 0


async def test_a_confirmed_try_run_deletes(client, http):
    c, _ = client

    body = (await c.post(
        "/api/v1/datasources/try-operation",
        json=_body(confirm_destructive=True),
    )).json()

    assert body["status"] == "ok"
    assert "DELETE" in http.methods()


async def test_a_confirmed_try_run_leaves_a_case_behind(client, http):
    c, backend = client

    await c.post(
        "/api/v1/datasources/try-operation",
        json=_body(confirm_destructive=True),
    )

    cases = await backend.list()
    assert len(cases) == 1
    assert cases[0].status == "approved"
    assert cases[0].surface == "try_run"
    assert cases[0].affected_rows == 3
    assert cases[0].operation == "drop"


async def test_a_read_operation_runs_straight_through(client, http):
    c, backend = client

    body = (await c.post(
        "/api/v1/datasources/try-operation",
        json=_body(operation="list"),
    )).json()

    assert body["status"] == "ok"
    assert await backend.count() == 0


async def test_an_operation_flagged_not_destructive_runs_straight_through(client, http):
    c, backend = client
    ops = [
        {"name": "stale", "path": "/files?stale=true"},
        {"name": "drop", "method": "DELETE", "path": "/files/{stale.id}",
         "destructive": False},
    ]

    body = (await c.post(
        "/api/v1/datasources/try-operation", json=_body(operations=ops),
    )).json()

    assert body["status"] == "ok"
    assert await backend.count() == 0


async def test_a_flagged_post_is_gated_even_though_it_is_not_a_delete(client, http):
    c, _ = client
    ops = [
        {"name": "stale", "path": "/files?stale=true"},
        {"name": "purge", "method": "POST", "path": "/files/{stale.id}/purge",
         "destructive": True},
    ]

    body = (await c.post(
        "/api/v1/datasources/try-operation",
        json=_body(operations=ops, operation="purge"),
    )).json()

    assert body["status"] == "confirmation_required"
    assert body["destructive"]["affected_rows"] == 3


async def test_a_delete_that_matches_nothing_needs_no_confirmation(client, http):
    c, backend = client
    http.handler = lambda method, url: [] if method == "GET" else {"ok": True}

    body = (await c.post("/api/v1/datasources/try-operation", json=_body())).json()

    # Confirming a no-op only teaches people to click through the dialog.
    assert body["status"] == "ok"
    assert await backend.count() == 0


async def test_a_targeted_single_delete_still_asks(client, http):
    c, _ = client
    ops = [{
        "name": "drop", "method": "DELETE", "path": "/files/{params.id}",
        "params": [{"name": "id"}],
    }]

    body = (await c.post(
        "/api/v1/datasources/try-operation",
        json=_body(operations=ops, params={"id": "f9"}),
    )).json()

    assert body["status"] == "confirmation_required"
    assert body["destructive"]["affected_rows"] == 1
    assert http.methods() == []


async def test_a_failing_preview_is_reported_not_raised(client, monkeypatch):
    c, _ = client

    async def _boom(*a, **k):
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(DataSourceExecutor, "preview", _boom)
    body = (await c.post("/api/v1/datasources/try-operation", json=_body())).json()

    assert body["status"] == "error"
    assert "upstream 503" in body["error"]
