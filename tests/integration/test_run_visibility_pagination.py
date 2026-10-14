"""Run and approval listings filter hidden workflows in the query, not after it.

A private workflow's runs and approval cases are its owner's. The filter has to
be part of the Mongo query (``graph_id`` / ``workflow_id`` ``$nin``) so skip,
limit and the total are computed on what the caller may see: a page filtered
after it was fetched comes back short, and its total counts runs nobody shown
the page can open. Against a real MongoDB, with hidden runs interleaved.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.approval_case import ApprovalCase
from app.domain.models.graph_run import GraphRun
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.auth.authorization import (
    Subject,
    reset_current_subject,
    set_current_subject,
)
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.approval_backend import MongoApprovalBackend
from app.infrastructure.persistence.mongo import MongoClientProvider
from app.infrastructure.persistence.workflow_backend import MongoWorkflowBackend
from app.infrastructure.tools.mcp_client import McpToolsProvider

_DB = "test_langgraph_integration"
_URI = "mongodb://localhost:27017"


def _with_test_subject(app):
    async def wrapped(scope, receive, send):
        user = dict(scope.get("headers") or ()).get(b"x-test-user") if scope["type"] == "http" else None
        if not user:
            return await app(scope, receive, send)
        name = user.decode()
        token = set_current_subject(Subject(id=name, name=name, is_super_admin=name == "admin"))
        try:
            return await app(scope, receive, send)
        finally:
            reset_current_subject(token)
    return wrapped


@pytest.fixture
async def env():
    settings = Settings(mongodb_uri=_URI, mongodb_database=_DB, environment="test")
    mongo_provider = MongoClientProvider(settings)
    runs = mongo_provider.get_repository()
    await runs._collection.delete_many({})
    workflows = MongoWorkflowBackend(_URI, _DB)
    await workflows._col.delete_many({})
    approvals = MongoApprovalBackend(_URI, _DB)
    await approvals._col.delete_many({})

    # Alice owns both; only "pub" is public. Bob sees "pub" alone.
    await workflows.create(WorkflowDefinition(id="pub", visibility="public", owner_id="alice"))
    await workflows.create(WorkflowDefinition(id="priv", visibility="private", owner_id="alice"))

    # 12 runs, newest first alternating priv/pub: every page of the unfiltered
    # list holds hidden runs.
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(12):
        graph_id = "priv" if i % 2 == 0 else "pub"
        run = GraphRun(id=f"r{i:02d}", graph_id=graph_id, user_request=f"req {i}", status="completed")
        run.created_at = base + timedelta(minutes=12 - i)
        await runs.create(run)
        await approvals.create(ApprovalCase(
            id=f"c{i:02d}", workflow_id=graph_id, run_id=run.id,
            created_at=base + timedelta(minutes=12 - i),
        ))

    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    container = ApplicationContainer(
        settings=settings,
        llm=MagicMock(),
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=runs,
        openhands=MagicMock(spec=OpenHandsAdapter),
        workflow_backend=workflows,
        approval_backend=approvals,
        approval_service=MagicMock(),  # only the read side is exercised
    )
    container.refresh_runner = AsyncMock()
    app = create_app()
    app.state.container = container
    async with AsyncClient(
        transport=ASGITransport(app=_with_test_subject(app)), base_url="http://test"
    ) as client:
        yield client
    await workflows.close()
    await approvals.close()
    await mongo_provider.close()


async def _runs(client, user, **params):
    resp = await client.get(
        "/api/v1/workflows/runs", params=params, headers={"X-Test-User": user}
    )
    assert resp.status_code == 200
    return resp.json()


async def test_a_page_is_full_of_visible_runs_and_the_total_excludes_hidden_ones(env):
    page = await _runs(env, "bob", limit=4, offset=0)
    assert [r["workflow_id"] for r in page["runs"]] == ["pub"] * 4
    assert page["total"] == 6

    rest = await _runs(env, "bob", limit=4, offset=4)
    assert [r["workflow_id"] for r in rest["runs"]] == ["pub"] * 2
    assert rest["total"] == 6
    ids = {r["id"] for r in page["runs"] + rest["runs"]}
    assert len(ids) == 6


async def test_owner_and_admin_see_every_run(env):
    for user in ("alice", "admin"):
        page = await _runs(env, user, limit=4)
        assert len(page["runs"]) == 4
        assert {r["workflow_id"] for r in page["runs"]} == {"priv", "pub"}
        assert page["total"] == 12


async def test_search_and_status_filters_compose_with_visibility(env):
    page = await _runs(env, "bob", limit=50, status="completed", search="req")
    assert page["total"] == 6
    assert {r["workflow_id"] for r in page["runs"]} == {"pub"}


async def test_a_workflow_turning_private_takes_its_runs_with_it(env):
    """Nothing is denormalised onto runs: visibility is read off the workflow at
    query time, so a change applies to every existing run at once."""
    resp = await env.put(
        "/api/v1/workflows/pub",
        json={"name": "pub", "steps": [], "visibility": "private"},
        headers={"X-Test-User": "alice"},
    )
    assert resp.status_code == 200
    page = await _runs(env, "bob", limit=4)
    assert page == {"runs": [], "total": 0}


async def test_approval_list_is_filtered_before_pagination(env):
    resp = await env.get(
        "/api/v1/approvals", params={"limit": 4}, headers={"X-Test-User": "bob"}
    )
    body = resp.json()
    assert resp.status_code == 200, body
    assert [c["workflow_id"] for c in body["items"]] == ["pub"] * 4
    assert body["total"] == 6
    admin = (await env.get(
        "/api/v1/approvals", params={"limit": 4}, headers={"X-Test-User": "admin"}
    )).json()
    assert admin["total"] == 12
    pending = (await env.get(
        "/api/v1/approvals/pending/count", headers={"X-Test-User": "bob"}
    )).json()
    assert pending == {"count": 6}
