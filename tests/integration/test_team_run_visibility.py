"""Team workflows' runs: visible to the team, filtered in the query.

Against a real MongoDB, with the real Mongo user directory: runs of a team
workflow, a private one and a public one interleaved on every page. A team
member's page is full of what they may see, the total excludes the rest, and
leaving the team takes effect on the next request.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.application import teams
from app.application.teams import resolve_subject
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.auth.authorization import (
    Subject,
    reset_current_subject,
    set_current_subject,
)
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoClientProvider
from app.infrastructure.persistence.team_directory import MongoTeamDirectory
from app.infrastructure.persistence.workflow_backend import MongoWorkflowBackend
from app.infrastructure.tools.mcp_client import McpToolsProvider

_DB = "test_langgraph_integration"
_URI = "mongodb://localhost:27017"


def _shim(app, directory):
    async def wrapped(scope, receive, send):
        user = dict(scope.get("headers") or ()).get(b"x-test-user") if scope["type"] == "http" else None
        if not user:
            return await app(scope, receive, send)
        name = user.decode()
        subject = await resolve_subject(
            directory, Subject(id=name, name=name, is_super_admin=name == "admin")
        )
        token = set_current_subject(subject)
        try:
            return await app(scope, receive, send)
        finally:
            reset_current_subject(token)
    return wrapped


@pytest.fixture
async def env():
    teams._NAMES.clear()
    teams._NAMES_LOADED_AT = 0.0
    settings = Settings(mongodb_uri=_URI, mongodb_database=_DB, environment="test")
    mongo_provider = MongoClientProvider(settings)
    runs = mongo_provider.get_repository()
    await runs._collection.delete_many({})
    workflows = MongoWorkflowBackend(_URI, _DB)
    await workflows._col.delete_many({})
    directory = MongoTeamDirectory(_URI, _DB)
    await directory._users.delete_many({})
    await directory._teams.delete_many({})
    await directory.ensure_indexes()

    red = await directory.create_team("Red")
    await directory.set_membership("alice", red.id, "admin")
    await directory.set_membership("bob", red.id, "member")

    await workflows.create(WorkflowDefinition(
        id="team", visibility="team", team_id=red.id, owner_id="alice",
    ))
    await workflows.create(WorkflowDefinition(id="priv", visibility="private", owner_id="alice"))
    await workflows.create(WorkflowDefinition(id="pub", visibility="public", owner_id="alice"))

    # 12 runs cycling team/priv/pub, newest first: every page mixes all three.
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cycle = ["team", "priv", "pub"]
    for i in range(12):
        run = GraphRun(
            id=f"r{i:02d}", graph_id=cycle[i % 3], user_request=f"req {i}", status="completed"
        )
        run.created_at = base + timedelta(minutes=12 - i)
        await runs.create(run)

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
        team_directory=directory,
    )
    container.refresh_runner = AsyncMock()
    app = create_app()
    app.state.container = container
    async with AsyncClient(
        transport=ASGITransport(app=_shim(app, directory)), base_url="http://test"
    ) as client:
        yield client, directory
    await workflows.close()
    await directory.close()
    await mongo_provider.close()


async def _runs(client, user, **params):
    resp = await client.get(
        "/api/v1/workflows/runs", params=params, headers={"X-Test-User": user}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_a_team_members_pages_are_full_and_the_total_is_theirs(env):
    client, _ = env
    first = await _runs(client, "bob", limit=3, offset=0)
    assert len(first["runs"]) == 3
    assert {r["workflow_id"] for r in first["runs"]} <= {"team", "pub"}
    assert first["total"] == 8
    seen = []
    for offset in range(0, 8, 3):
        page = await _runs(client, "bob", limit=3, offset=offset)
        seen += [r["id"] for r in page["runs"]]
        assert page["total"] == 8
    assert len(seen) == len(set(seen)) == 8


async def test_outsiders_see_only_public_runs_and_owner_and_super_admin_all(env):
    client, _ = env
    carol = await _runs(client, "carol", limit=50)
    assert {r["workflow_id"] for r in carol["runs"]} == {"pub"} and carol["total"] == 4
    for user in ("alice", "admin"):
        assert (await _runs(client, user, limit=3))["total"] == 12


async def test_leaving_the_team_applies_on_the_next_request(env):
    client, directory = env
    assert (await _runs(client, "bob", limit=50))["total"] == 8
    await directory.set_membership("bob", None, None)
    after = await _runs(client, "bob", limit=50)
    assert after["total"] == 4
    assert {r["workflow_id"] for r in after["runs"]} == {"pub"}
