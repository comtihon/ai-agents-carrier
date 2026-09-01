"""Tests for the /api/v1/approvals API."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.application.approval_service import ApprovalService
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.approval_case import ApprovalCase, history_key_for
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.approval_backend import InMemoryApprovalBackend
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


def _container(backend, settings: Settings | None = None) -> ApplicationContainer:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    settings = settings or Settings()
    service = (
        ApprovalService(backend, settings) if backend is not None else None
    )
    return ApplicationContainer(
        settings=settings,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=AsyncMock(spec=MongoGraphRunRepository),
        openhands=MagicMock(spec=OpenHandsAdapter),
        approval_backend=backend,
        approval_service=service,
    )


def _case(**overrides) -> ApprovalCase:
    data = dict(
        id="apr_1",
        status="pending",
        workflow_id="wf",
        datasource_id="files",
        datasource_name="File store",
        operation="drop",
        method="DELETE",
        affected_rows=7,
        # An MCP-surface case has no run to resume, which keeps these tests
        # about the API rather than about the graph.
        surface="mcp",
        history_key=history_key_for("wf", "files", "drop"),
    )
    data.update(overrides)
    return ApprovalCase(**data)


@pytest.fixture
async def client():
    backend = InMemoryApprovalBackend()
    app = create_app()
    app.state.container = _container(backend)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, backend


@pytest.fixture
async def client_without_backend():
    app = create_app()
    app.state.container = _container(None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_list_returns_the_queue_newest_first(client):
    c, backend = client
    base = datetime.now(timezone.utc)
    await backend.create(_case(id="old", created_at=base - timedelta(hours=1)))
    await backend.create(_case(id="new", created_at=base))

    resp = await c.get("/api/v1/approvals")

    assert resp.status_code == 200
    body = resp.json()
    assert [i["id"] for i in body["items"]] == ["new", "old"]
    assert body["total"] == 2


async def test_list_filters_by_status(client):
    c, backend = client
    await backend.create(_case(id="p"))
    await backend.create(_case(id="a", status="approved"))

    resp = await c.get("/api/v1/approvals", params={"status": "approved"})

    assert [i["id"] for i in resp.json()["items"]] == ["a"]


async def test_pending_count_feeds_the_dock_badge(client):
    c, backend = client
    await backend.create(_case(id="p1"))
    await backend.create(_case(id="p2"))
    await backend.create(_case(id="done", status="approved"))

    resp = await c.get("/api/v1/approvals/pending/count")
    assert resp.json() == {"count": 2}


async def test_get_one_case_carries_its_summary(client):
    c, backend = client
    await backend.create(_case())

    body = (await c.get("/api/v1/approvals/apr_1")).json()

    assert body["affected_rows"] == 7
    assert body["summary"] == "File store.drop [DELETE] — 7 rows"


async def test_get_unknown_case_is_404(client):
    c, _ = client
    assert (await c.get("/api/v1/approvals/nope")).status_code == 404


async def test_decide_records_the_answer(client):
    c, backend = client
    await backend.create(_case())

    resp = await c.post(
        "/api/v1/approvals/apr_1/decide",
        json={"approved": True, "reason": "expected nightly cleanup"},
    )

    assert resp.status_code == 200
    stored = await backend.get("apr_1")
    assert stored.status == "approved"
    assert stored.reason == "expected nightly cleanup"
    assert stored.decision_source == "ui"


async def test_deciding_twice_is_a_conflict(client):
    c, backend = client
    await backend.create(_case())

    assert (await c.post("/api/v1/approvals/apr_1/decide", json={"approved": True})).status_code == 200
    second = await c.post("/api/v1/approvals/apr_1/decide", json={"approved": False})

    assert second.status_code == 409


async def test_history_reports_the_streak_and_the_threshold(client):
    c, backend = client
    base = datetime.now(timezone.utc)
    for i in range(4):
        await backend.create(_case(
            id=f"h{i}", status="approved", decision_source="ui",
            decided_at=base - timedelta(minutes=i),
        ))

    body = (await c.get("/api/v1/approvals/history", params={
        "workflow_id": "wf", "datasource_id": "files", "operation": "drop",
    })).json()

    assert body["streak"] == 4
    assert body["streak_decision"] == "approved"
    assert body["threshold"] == 10
    assert body["autonomous_next"] is False


async def test_history_streak_stops_at_a_disagreement(client):
    c, backend = client
    base = datetime.now(timezone.utc)
    await backend.create(_case(id="h0", status="approved", decision_source="ui",
                               decided_at=base))
    await backend.create(_case(id="h1", status="rejected", decision_source="ui",
                               decided_at=base - timedelta(minutes=1)))
    await backend.create(_case(id="h2", status="approved", decision_source="ui",
                               decided_at=base - timedelta(minutes=2)))

    body = (await c.get("/api/v1/approvals/history", params={
        "workflow_id": "wf", "datasource_id": "files", "operation": "drop",
    })).json()

    assert body["streak"] == 1


async def test_veto_cancels_inside_the_window(client):
    c, backend = client
    await backend.create(_case(
        status="approved",
        decision_source="meta_llm",
        veto_deadline=datetime.now(timezone.utc) + timedelta(seconds=30),
    ))

    resp = await c.post("/api/v1/approvals/apr_1/veto", json={"by": "ada"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
    assert (await backend.get("apr_1")).vetoed_by == "ada"


async def test_veto_of_a_human_decision_is_refused(client):
    c, backend = client
    await backend.create(_case(status="approved", decision_source="ui"))

    resp = await c.post("/api/v1/approvals/apr_1/veto", json={"by": "ada"})

    assert resp.status_code == 409


async def test_every_route_reports_a_missing_backend_as_501(client_without_backend):
    c = client_without_backend
    assert (await c.get("/api/v1/approvals")).status_code == 501
    assert (await c.get("/api/v1/approvals/apr_1")).status_code == 501
    # The badge count is the exception: a dock that cannot draw a number must
    # still draw the rail.
    assert (await c.get("/api/v1/approvals/pending/count")).json() == {"count": 0}
