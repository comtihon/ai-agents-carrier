"""Public/private agents, data sources, scripts and workflows.

A private resource is visible to its owner and to admins only; anyone else gets
a 404 and never sees it listed. Only the owner (or an admin) may change or
delete a resource. References are checked on save: you can only use what you
can see, and a public resource may not use a private one. With no subject bound
— authentication off — every check is a no-op.

The caller is bound by a small ASGI shim keyed on an ``X-Test-User`` header,
standing in for OAuthMiddleware (whose own binding is covered by its tests).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.application import management_tools as core
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.graph_run import GraphRun
from app.infrastructure.auth.authorization import (
    Subject,
    reset_current_subject,
    set_current_subject,
)
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


class MemoryBackend:
    """One in-memory store shape for all four definition kinds."""

    def __init__(self) -> None:
        self.docs: dict = {}

    async def list(self):
        return list(self.docs.values())

    async def get(self, resource_id):
        return self.docs.get(resource_id)

    async def get_by_name(self, name):
        return next((d for d in self.docs.values() if d.name == name), None)

    async def create(self, definition):
        definition.touch()
        self.docs[definition.id] = definition
        return definition

    async def update(self, resource_id, definition):
        definition.id = resource_id
        definition.touch()
        self.docs[resource_id] = definition
        return definition

    async def delete(self, resource_id):
        self.docs.pop(resource_id, None)


def _subject(user: str) -> Subject:
    if user == "apikey":
        return Subject()
    return Subject(id=user, name=user.title(), is_admin=user == "admin")


def _with_test_subject(app):
    """Bind the subject named by X-Test-User, as OAuthMiddleware would."""
    async def wrapped(scope, receive, send):
        user = None
        if scope["type"] == "http":
            user = dict(scope.get("headers") or ()).get(b"x-test-user")
        if not user:
            return await app(scope, receive, send)
        token = set_current_subject(_subject(user.decode()))
        try:
            return await app(scope, receive, send)
        finally:
            reset_current_subject(token)
    return wrapped


def _container() -> ApplicationContainer:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    run_repository = AsyncMock(spec=MongoGraphRunRepository)
    run_repository.list_recent = AsyncMock(return_value=[])
    run_repository.count_recent = AsyncMock(return_value=0)
    container = ApplicationContainer(
        settings=Settings(),
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=run_repository,
        openhands=MagicMock(spec=OpenHandsAdapter),
        workflow_backend=MemoryBackend(),
        agent_backend=MemoryBackend(),
        data_source_backend=MemoryBackend(),
        script_backend=MemoryBackend(),
    )
    container.refresh_runner = AsyncMock()  # the registry is not under test
    return container


@pytest.fixture
async def env():
    container = _container()
    app = create_app()
    app.state.container = container
    transport = ASGITransport(app=_with_test_subject(app))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        def as_(user: str | None):
            headers = {"X-Test-User": user} if user else {}

            class _Caller:
                async def get(self, url, **kw):
                    return await client.get(url, headers=headers, **kw)

                async def post(self, url, **kw):
                    return await client.post(url, headers=headers, **kw)

                async def put(self, url, **kw):
                    return await client.put(url, headers=headers, **kw)

                async def delete(self, url, **kw):
                    return await client.delete(url, headers=headers, **kw)
            return _Caller()
        yield as_, container


def _agent(agent_id: str, **extra) -> dict:
    return {"id": agent_id, "name": agent_id.title(), **extra}


def _workflow(workflow_id: str, agent_id: str | None = None, **extra) -> dict:
    steps = []
    if agent_id:
        steps.append({"id": "a", "type": "langgraph-agent", "agent_id": agent_id})
    return {"id": workflow_id, "name": workflow_id, "steps": steps, **extra}


# ─── Defaults, listing, 404 ──────────────────────────────────────────────────

async def test_a_user_creates_private_by_default_and_owns_it(env):
    as_, _ = env
    resp = await as_("alice").post("/api/v1/agents", json=_agent("helper"))
    assert resp.status_code == 201
    body = resp.json()
    assert body["visibility"] == "private"
    assert body["owner_id"] == "alice"
    assert body["owner_name"] == "Alice"
    assert body["can_edit"] is True


async def test_private_resources_are_invisible_to_other_users(env):
    as_, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("secret"))
    await as_("alice").post("/api/v1/agents", json=_agent("shared", visibility="public"))

    ids = {a["id"] for a in (await as_("bob").get("/api/v1/agents")).json()}
    assert ids == {"shared"}
    summary = (await as_("bob").get("/api/v1/agents", params={"view": "summary"})).json()
    assert [a["id"] for a in summary] == ["shared"]
    assert summary[0]["can_edit"] is False
    assert (await as_("bob").get("/api/v1/agents/secret")).status_code == 404
    assert (await as_("alice").get("/api/v1/agents/secret")).status_code == 200


async def test_admin_sees_everything(env):
    as_, _ = env
    await as_("alice").post("/api/v1/scripts", json={"name": "Mine", "code": "output = 1"})
    listed = (await as_("admin").get("/api/v1/scripts")).json()
    assert [s["id"] for s in listed] == ["mine"]
    assert listed[0]["can_edit"] is True


async def test_api_key_caller_sees_public_only_and_creates_public(env):
    as_, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("secret"))
    assert (await as_("apikey").get("/api/v1/agents")).json() == []
    created = (await as_("apikey").post("/api/v1/agents", json=_agent("bot"))).json()
    assert created["visibility"] == "public"
    assert created["owner_id"] is None


async def test_auth_off_is_a_no_op(env):
    """No subject bound: everything visible and editable, new resources public."""
    as_, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("secret"))
    created = (await as_(None).post("/api/v1/agents", json=_agent("anon"))).json()
    assert created["visibility"] == "public" and created["owner_id"] is None
    listed = {a["id"]: a for a in (await as_(None).get("/api/v1/agents")).json()}
    assert set(listed) == {"secret", "anon"}
    assert listed["secret"]["can_edit"] is True
    resp = await as_(None).put("/api/v1/agents/secret", json={"name": "changed"})
    assert resp.status_code == 200


# ─── Ownership on writes ─────────────────────────────────────────────────────

async def test_only_the_owner_or_an_admin_edits_a_public_resource(env):
    as_, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("shared", visibility="public"))

    resp = await as_("bob").put("/api/v1/agents/shared", json={"name": "mine now"})
    assert resp.status_code == 403
    assert "owner" in resp.json()["detail"]
    assert (await as_("bob").delete("/api/v1/agents/shared")).status_code == 403

    assert (await as_("admin").put("/api/v1/agents/shared", json={"name": "fixed"})).status_code == 200
    assert (await as_("alice").put("/api/v1/agents/shared", json={"name": "again"})).status_code == 200


async def test_owner_and_visibility_survive_a_put_that_rebuilds_the_model(env):
    as_, container = env
    await as_("alice").post("/api/v1/agents", json=_agent("helper"))
    # An admin edit does not transfer ownership, and an omitted visibility is kept.
    await as_("admin").put("/api/v1/agents/helper", json={"name": "renamed"})
    stored = await container.agent_backend.inner.get("helper")
    assert stored.owner_id == "alice"
    assert stored.visibility == "private"


async def test_a_hidden_id_cannot_be_taken_over_by_create(env):
    as_, container = env
    await as_("alice").post("/api/v1/agents", json=_agent("helper", agent_input={"v": 1}))
    resp = await as_("bob").post("/api/v1/agents", json=_agent("helper"))
    assert resp.status_code == 409
    assert "alice" not in resp.json()["detail"].lower()
    assert (await container.agent_backend.inner.get("helper")).agent_input == {"v": 1}


async def test_script_name_of_a_hidden_script_is_refused_without_details(env):
    """Names are unique within what a caller sees, but ids are global: the slug
    of a hidden script's name is taken, and the refusal says nothing about it."""
    as_, container = env
    await as_("alice").post("/api/v1/scripts", json={"name": "Sum", "code": "a"})
    resp = await as_("bob").post("/api/v1/scripts", json={"name": "Sum", "code": "b"})
    assert resp.status_code == 409
    assert "Alice" not in resp.json()["detail"]
    # Even with overwrite: no takeover.
    resp = await as_("bob").post(
        "/api/v1/scripts", json={"name": "Sum", "code": "b", "overwrite": True}
    )
    assert resp.status_code == 409
    assert (await container.script_backend.inner.get("sum")).code == "a"


async def test_overwriting_someone_elses_public_script_is_forbidden(env):
    as_, _ = env
    await as_("alice").post("/api/v1/scripts", json={"name": "Sum", "code": "a", "visibility": "public"})
    resp = await as_("bob").post(
        "/api/v1/scripts", json={"name": "Sum", "code": "b", "overwrite": True}
    )
    assert resp.status_code == 403


async def test_turning_a_legacy_ownerless_resource_private_claims_it(env):
    as_, container = env
    await container.agent_backend.inner.create(AgentDefinition(id="legacy"))
    assert (await as_("bob").put("/api/v1/agents/legacy", json={"name": "x"})).status_code == 200
    resp = await as_("bob").put("/api/v1/agents/legacy", json={"visibility": "private"})
    assert resp.status_code == 200
    assert resp.json()["owner_id"] == "bob"
    assert (await as_("carol").get("/api/v1/agents/legacy")).status_code == 404


# ─── References ──────────────────────────────────────────────────────────────

async def test_a_public_workflow_cannot_use_a_private_agent(env):
    as_, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("helper"))
    resp = await as_("alice").post(
        "/api/v1/workflows", json=_workflow("wf", "helper", visibility="public")
    )
    assert resp.status_code == 422
    assert "public workflow cannot use private agent 'helper'" in resp.json()["detail"]

    ok = await as_("alice").post("/api/v1/workflows", json=_workflow("wf", "helper"))
    assert ok.status_code == 201
    # ...and it cannot be published while it still uses the private agent.
    resp = await as_("alice").put("/api/v1/workflows/wf", json={
        **_workflow("wf", "helper"), "visibility": "public",
    })
    assert resp.status_code == 422


async def test_you_cannot_reference_what_you_cannot_see(env):
    as_, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("helper"))
    resp = await as_("bob").post("/api/v1/workflows", json=_workflow("wf", "helper"))
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Agent 'helper' not found"


async def test_a_private_workflow_cannot_use_another_users_private_agent_even_via_admin(env):
    as_, _ = env
    await as_("bob").post("/api/v1/agents", json=_agent("bobs"))
    await as_("alice").post("/api/v1/workflows", json=_workflow("wf"))
    resp = await as_("admin").put("/api/v1/workflows/wf", json=_workflow("wf", "bobs"))
    assert resp.status_code == 422
    assert "private to another user" in resp.json()["detail"]


async def test_a_public_agent_cannot_attach_a_private_data_source(env):
    as_, _ = env
    await as_("alice").post("/api/v1/datasources", json={"id": "crm", "base_url": "https://x"})
    resp = await as_("alice").post("/api/v1/agents", json=_agent(
        "helper", visibility="public",
        addons=[{"type": "datasource", "source_id": "crm", "allowed_operations": []}],
    ))
    assert resp.status_code == 422
    assert "private data source 'crm'" in resp.json()["detail"]


async def test_making_a_used_resource_private_is_refused(env):
    as_, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("shared", visibility="public"))
    await as_("bob").post("/api/v1/workflows", json=_workflow("bobs-wf", "shared"))

    resp = await as_("alice").put("/api/v1/agents/shared", json={"visibility": "private"})
    assert resp.status_code == 409
    # Bob's workflow is private: it is not named to Alice.
    assert "another user" in resp.json()["detail"]
    assert "bobs-wf" not in resp.json()["detail"]


async def test_making_private_is_allowed_when_only_own_private_resources_use_it(env):
    as_, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("shared", visibility="public"))
    await as_("alice").post("/api/v1/workflows", json=_workflow("mine", "shared"))
    resp = await as_("alice").put("/api/v1/agents/shared", json={"visibility": "private"})
    assert resp.status_code == 200
    assert resp.json()["visibility"] == "private"


async def test_making_private_is_refused_while_a_public_workflow_uses_it(env):
    as_, _ = env
    await as_("alice").post("/api/v1/agents", json=_agent("shared", visibility="public"))
    await as_("alice").post("/api/v1/workflows", json=_workflow("pub", "shared", visibility="public"))
    resp = await as_("alice").put("/api/v1/agents/shared", json={"visibility": "private"})
    assert resp.status_code == 409
    assert "public workflow 'pub'" in resp.json()["detail"]


# ─── Workflows list and runs ─────────────────────────────────────────────────

async def test_runs_of_hidden_workflows_are_excluded_in_the_query(env):
    as_, container = env
    await as_("alice").post("/api/v1/workflows", json=_workflow("private-wf"))
    await as_("alice").post("/api/v1/workflows", json=_workflow("public-wf", visibility="public"))

    await as_("bob").get("/api/v1/workflows/runs", params={"limit": 5})
    kwargs = container.run_repository.list_recent.await_args.kwargs
    assert kwargs["exclude_workflow_ids"] == ["private-wf"]
    count_kwargs = container.run_repository.count_recent.await_args.kwargs
    assert count_kwargs["exclude_workflow_ids"] == ["private-wf"]

    # Asking for a hidden workflow's runs by id returns nothing at all.
    container.run_repository.list_recent.reset_mock()
    resp = await as_("bob").get("/api/v1/workflows/runs", params={"workflow_id": "private-wf"})
    assert resp.json() == {"runs": [], "total": 0}
    container.run_repository.list_recent.assert_not_awaited()

    # The owner and admins are not filtered.
    await as_("alice").get("/api/v1/workflows/runs")
    assert not container.run_repository.list_recent.await_args.kwargs.get("exclude_workflow_ids")


async def test_a_hidden_workflows_run_is_404(env):
    as_, container = env
    await as_("alice").post("/api/v1/workflows", json=_workflow("private-wf"))
    container.run_repository.get = AsyncMock(
        return_value=GraphRun(id="r1", graph_id="private-wf", user_request="x", status="completed")
    )
    assert (await as_("bob").get("/api/v1/workflows/runs/r1")).status_code == 404
    assert (await as_("bob").get("/api/v1/workflows/runs/r1/trace")).status_code == 404
    assert (await as_("bob").post("/api/v1/workflows/runs/r1/terminate")).status_code == 404
    assert (await as_("bob").get("/api/v1/runs/r1/data")).status_code == 404
    assert (await as_("alice").get("/api/v1/workflows/runs/r1/trace")).status_code == 200


async def test_starting_a_hidden_workflow_is_404(env):
    as_, _ = env
    await as_("alice").post("/api/v1/workflows", json=_workflow("private-wf"))
    resp = await as_("bob").post(
        "/api/v1/workflows/runs", json={"workflow_id": "private-wf", "user_request": "x"}
    )
    assert resp.status_code == 404


async def test_me(env):
    as_, _ = env
    assert (await as_("alice").get("/api/v1/me")).json() == {
        "sub": "alice", "name": "Alice", "is_admin": False, "auth_enabled": False,
    }
    assert (await as_("admin").get("/api/v1/me")).json()["is_admin"] is True
    assert (await as_(None).get("/api/v1/me")).json()["sub"] is None


# ─── Management tools (shared by the MCP server and the chat agent) ─────────

async def test_management_tools_see_and_change_only_what_the_caller_may(env):
    as_, container = env
    await as_("alice").post("/api/v1/agents", json=_agent("secret"))
    await as_("alice").post("/api/v1/agents", json=_agent("shared", visibility="public"))
    deps = core.deps_from_container(container)

    token = set_current_subject(_subject("bob"))
    try:
        listing = await core.list_agents(deps)
        assert "shared" in listing and "secret" not in listing
        assert "no agent found" in (await core.get_agent(deps, "secret")).lower()
        refused = await core.delete_agent(deps, "shared")
        assert refused.startswith("Refused:")
    finally:
        reset_current_subject(token)
    assert await container.agent_backend.inner.get("shared") is not None


async def test_management_run_listing_excludes_hidden_workflows(env):
    as_, container = env
    await as_("alice").post("/api/v1/workflows", json=_workflow("private-wf"))
    deps = core.deps_from_container(container)
    token = set_current_subject(_subject("bob"))
    try:
        await core.list_runs(deps)
        assert container.run_repository.list_recent.await_args.kwargs[
            "exclude_workflow_ids"
        ] == ["private-wf"]
        assert await core.list_runs(deps, workflow_id="private-wf") == "No runs found."
    finally:
        reset_current_subject(token)
