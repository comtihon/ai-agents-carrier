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


# ─── Summary reads ────────────────────────────────────────────────────────────
#
# The queue row shows a data source, an operation, a row count and a verdict.
# The fields it does *not* show are the unbounded ones — the caller's inputs,
# every resolved target, a per-row (on a write, per-cell) sample, and the
# details map — and fetching fifty of those to draw fifty one-line rows is what
# made the panel slow to open. `view=summary` leaves them out.


def _fat_case(**overrides) -> ApprovalCase:
    return _case(
        params={"body": "x" * 500},
        targets=[f"https://files.test/f{i}" for i in range(50)],
        affected_sample=[{"row": i, "before": "a", "after": "b"} for i in range(50)],
        details={"document": "Q3 sheet", "values_from": "generated code"},
        **overrides,
    )


async def test_summary_view_leaves_out_the_bulky_fields(client):
    c, backend = client
    await backend.create(_fat_case())

    row = (await c.get("/api/v1/approvals", params={"view": "summary"})).json()["items"][0]

    for field in ("params", "targets", "affected_sample", "details"):
        assert field not in row, f"{field} should not be in a summary row"


async def test_summary_view_keeps_everything_a_row_renders(client):
    c, backend = client
    await backend.create(_fat_case(
        workflow_name="Nightly cleanup",
        affected_rows=1240,
        change_kind="write",
    ))

    row = (await c.get("/api/v1/approvals", params={"view": "summary"})).json()["items"][0]

    assert row["id"] == "apr_1"
    assert row["status"] == "pending"
    assert row["datasource_name"] == "File store"
    assert row["operation"] == "drop"
    assert row["method"] == "DELETE"
    assert row["affected_rows"] == 1240
    assert row["workflow_name"] == "Nightly cleanup"
    assert row["change_kind"] == "write"
    assert row["summary"]


async def test_full_view_is_the_default_so_old_callers_keep_their_fields(client):
    c, backend = client
    await backend.create(_fat_case())

    row = (await c.get("/api/v1/approvals")).json()["items"][0]

    assert row["params"] == {"body": "x" * 500}
    assert len(row["targets"]) == 50
    assert len(row["affected_sample"]) == 50
    assert row["details"]["document"] == "Q3 sheet"


async def test_reading_one_case_always_carries_the_values(client):
    c, backend = client
    await backend.create(_fat_case())
    # The panel lists summaries and the detail page reads the case by id, so
    # this is the request that has to hold everything the reviewer decides on.
    await c.get("/api/v1/approvals", params={"view": "summary"})

    case = (await c.get("/api/v1/approvals/apr_1")).json()

    assert case["params"] == {"body": "x" * 500}
    assert len(case["affected_sample"]) == 50


async def test_a_summary_read_does_not_empty_the_stored_case(client):
    c, backend = client
    await backend.create(_fat_case())

    await c.get("/api/v1/approvals", params={"view": "summary"})

    stored = await backend.get("apr_1")
    assert stored is not None
    assert stored.params == {"body": "x" * 500}
    assert len(stored.targets) == 50


async def test_history_can_be_read_as_summaries(client):
    c, backend = client
    await backend.create(_fat_case(
        id="done", status="approved", decided_by_name="ada", reason="routine",
        decided_at=datetime.now(timezone.utc),
    ))

    body = (await c.get("/api/v1/approvals/history", params={
        "workflow_id": "wf", "datasource_id": "files", "operation": "drop",
        "view": "summary",
    })).json()

    row = body["items"][0]
    assert row["decided_by_name"] == "ada"
    assert row["reason"] == "routine"
    assert "affected_sample" not in row
    # The streak is what the page reads off this response; it must survive the
    # projection, because it is computed from fields a summary keeps.
    assert body["streak"] == 1
    assert body["streak_decision"] == "approved"


# ─── A run that died while parked on its gate ────────────────────────────────
#
# A workflow-surface case has no timeout on purpose: the run is parked inside a
# LangGraph interrupt and the case waits for a person for as long as it takes.
# The hole that leaves is a run that dies while parked — terminated, evicted,
# lost to a restart. The case stayed pending forever, the queue kept offering
# Approve and Reject, and both answered "Run is not awaiting approval (status:
# failed)" because there was no run left to resume.


async def test_deciding_a_case_whose_run_is_gone_cancels_it(client):
    c, backend = client
    await backend.create(_case(surface="workflow", run_id="run-dead"))
    # claim_for_resume returning None is how run_control reports "not parked".
    container = c._transport.app.state.container  # type: ignore[attr-defined]
    container.run_repository.claim_for_resume = AsyncMock(return_value=None)
    container.run_repository.get = AsyncMock(return_value=MagicMock(status="failed"))

    resp = await c.post("/api/v1/approvals/apr_1/decide", json={"approved": True})

    assert resp.status_code == 409
    assert "has been cancelled" in resp.json()["detail"]
    stored = await backend.get("apr_1")
    assert stored is not None
    assert stored.status == "cancelled"


async def test_a_cancelled_orphan_leaves_the_pending_queue(client):
    c, backend = client
    await backend.create(_case(surface="workflow", run_id="run-dead"))
    container = c._transport.app.state.container  # type: ignore[attr-defined]
    container.run_repository.claim_for_resume = AsyncMock(return_value=None)
    container.run_repository.get = AsyncMock(return_value=MagicMock(status="failed"))

    await c.post("/api/v1/approvals/apr_1/decide", json={"approved": False})

    queue = (await c.get("/api/v1/approvals", params={"status": "pending"})).json()
    assert queue["items"] == []


async def test_the_run_that_vanished_entirely_is_treated_the_same(client):
    c, backend = client
    await backend.create(_case(surface="workflow", run_id="run-gone"))
    container = c._transport.app.state.container  # type: ignore[attr-defined]
    container.run_repository.claim_for_resume = AsyncMock(return_value=None)
    container.run_repository.get = AsyncMock(return_value=None)

    resp = await c.post("/api/v1/approvals/apr_1/decide", json={"approved": True})

    assert resp.status_code == 404
    stored = await backend.get("apr_1")
    assert stored is not None and stored.status == "cancelled"
