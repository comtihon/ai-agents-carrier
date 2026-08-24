"""Tests for the /mcp/management FastMCP tool surface.

The management MCP server and the internal chat agent must expose the same
tools, backed by the same shared cores — the only intended difference is
``ask_user``, which is agent-only.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.api.mcp.management_server import build_management_mcp, register_management_tools
from app.application import run_control
from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.default_workflow import build_default_workflow
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from tests.test_datasources_api import InMemoryDataSourceBackend
from tests.test_events_api import InMemoryEventBackend

_EXPECTED_TOOLS = {
    "list_workflows", "run_workflow", "list_runs", "get_run",
    "create_workflow", "update_workflow", "delete_workflow",
    "list_agents", "get_agent", "create_agent", "update_agent", "delete_agent",
    "list_datasources", "get_datasource", "create_datasource", "update_datasource",
    "create_pubsub_datasource", "list_pubsub_subscriptions",
    "list_scripts", "get_script", "create_script", "update_script", "delete_script",
    "list_events", "get_event", "create_event", "update_event", "delete_event",
    "delete_datasource", "import_datasource_schema",
    "create_datasource_from_schema", "add_datasource_operations_from_schema",
    "terminate_run", "retry_run", "restart_from_step", "approve_run", "reject_run",
}


class _Container:
    """Minimal ApplicationContainer stand-in for the management handlers."""

    def __init__(self, *, data_source_backend=None, event_backend=None, run_repository=None) -> None:
        self.yaml_graph_registry = MagicMock()
        self.yaml_graph_registry.list_definitions.return_value = []
        self.run_repository = run_repository or AsyncMock()
        self.workflow_backend = None
        self.agent_backend = None
        self.data_source_backend = data_source_backend
        self.event_backend = event_backend
        self.refresh_runner = None
        self.settings = MagicMock()
        self.live_runners: dict = {}


@pytest.fixture
def mcp():
    server = build_management_mcp()
    return server


def _register(server, container) -> None:
    register_management_tools(server, lambda: container)


def _agent_platform_tools(container=None, **build_kwargs) -> tuple:
    """The chat agent's platform tools, captured from bind_tools()."""
    captured: dict = {}
    llm = MagicMock()

    def bind_tools(tools):
        captured.setdefault("tools", tools)
        bound = MagicMock()
        bound.ainvoke = AsyncMock(return_value=AIMessage(content="hi"))
        return bound

    llm.bind_tools = bind_tools
    registry = MagicMock()
    registry.list_definitions.return_value = []
    graph = build_default_workflow(
        llm, registry, AsyncMock(), container=container, **build_kwargs
    )
    return captured, graph


async def _agent_tools(container=None, **build_kwargs) -> list:
    captured, graph = _agent_platform_tools(container, **build_kwargs)
    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")],
         "copilotkit": {"actions": [], "context": []}},
        {"configurable": {"thread_id": "parity"}},
    )
    return captured["tools"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

async def test_registers_the_full_tool_set(mcp):
    _register(mcp, _Container())
    tools = await mcp.list_tools()
    assert {t.name for t in tools} == _EXPECTED_TOOLS
    assert len(tools) == 37


async def test_ask_user_is_not_exposed_over_mcp(mcp):
    _register(mcp, _Container())
    assert "ask_user" not in {t.name for t in await mcp.list_tools()}


# ---------------------------------------------------------------------------
# Agent <-> MCP parity
# ---------------------------------------------------------------------------

async def test_agent_and_mcp_expose_the_same_tools(mcp):
    _register(mcp, _Container())
    mcp_names = {t.name for t in await mcp.list_tools()}
    agent_names = {t.name for t in await _agent_tools(container=MagicMock())}
    # ask_user is the only agent-only tool.
    assert agent_names == mcp_names | {"ask_user"}


def _norm(text: str) -> str:
    """Compare docstrings ignoring indentation/trailing whitespace."""
    return "\n".join(line.strip() for line in (text or "").split("\n")).strip()


# approve_run/reject_run document the approver attribution of their own
# surface (agent vs mcp), so only their summary line is shared.
_SURFACE_SPECIFIC = {"approve_run", "reject_run"}


async def test_agent_and_mcp_descriptions_match(mcp):
    _register(mcp, _Container())
    mcp_desc = {t.name: t.description for t in await mcp.list_tools()}
    agent_desc = {t.name: t.description for t in await _agent_tools(container=MagicMock())}

    diffs = [
        n for n in mcp_desc
        if n not in _SURFACE_SPECIFIC and _norm(mcp_desc[n]) != _norm(agent_desc[n])
    ]
    assert diffs == []
    for n in _SURFACE_SPECIFIC:
        assert mcp_desc[n].splitlines()[0] == agent_desc[n].splitlines()[0]


def _norm_schema(schema: dict) -> dict:
    """Comparable arg schema: drop the wrapper title/description and the
    per-property titles (pydantic derives those from the surface's own model
    name), keep names, types, defaults and required-ness."""
    out = {k: v for k, v in schema.items() if k not in ("title", "description")}
    out["properties"] = {
        name: {k: v for k, v in spec.items() if k != "title"}
        for name, spec in (schema.get("properties") or {}).items()
    }
    out["required"] = sorted(schema.get("required") or [])
    return out


async def test_agent_and_mcp_arg_schemas_match(mcp):
    """Names and descriptions matching is not enough — the arg schemas are just
    as much part of the tool contract, and a drifting default or a param renamed
    on one surface only would otherwise go unnoticed."""
    _register(mcp, _Container())
    mcp_schemas = {t.name: _norm_schema(t.inputSchema) for t in await mcp.list_tools()}
    agent_schemas = {
        t.name: _norm_schema(t.tool_call_schema.model_json_schema())
        for t in await _agent_tools(container=MagicMock())
    }

    diffs = {
        n: (mcp_schemas[n], agent_schemas[n])
        for n in mcp_schemas
        if mcp_schemas[n] != agent_schemas[n]
    }
    assert diffs == {}


async def test_agent_without_container_has_no_run_control_tools():
    names = {t.name for t in await _agent_tools(container=None)}
    assert not names & {
        "terminate_run", "retry_run", "restart_from_step", "approve_run", "reject_run"
    }
    assert "list_workflows" in names


# ---------------------------------------------------------------------------
# A CRUD tool returns exactly what the agent tool returns
# ---------------------------------------------------------------------------

async def test_list_datasources_matches_the_agent_tool(mcp):
    backend = InMemoryDataSourceBackend()
    from app.domain.models.data_source_definition import DataSourceDefinition
    await backend.create(DataSourceDefinition.model_validate({
        "id": "github",
        "name": "GitHub",
        "description": "Code host",
        "base_url": "https://api.github.com",
        "operations": [{"name": "list_repos", "path": "/repos"}],
    }))

    _register(mcp, _Container(data_source_backend=backend))
    mcp_result = str(await mcp.call_tool("list_datasources", {}))

    agent_tools = await _agent_tools(container=MagicMock(), data_source_backend=backend)
    agent_tool = next(t for t in agent_tools if t.name == "list_datasources")
    agent_result = await agent_tool.ainvoke({})

    assert agent_result == "- **github** (GitHub, http): Code host — operations: list_repos"
    assert agent_result in mcp_result


async def test_create_event_stores_topic_and_schema(mcp):
    backend = InMemoryEventBackend()
    _register(mcp, _Container(event_backend=backend))

    result = str(await mcp.call_tool("create_event", {
        "event_id": "orders-events",
        "name": "Order events",
        "topic": "orders",
        "event_schema_json": '{"type": "object", "required": ["order_id"]}',
    }))

    assert "created" in result
    stored = await backend.get("orders-events")
    assert stored.topic == "orders"
    assert stored.event_schema == {"type": "object", "required": ["order_id"]}
    # No subscription named: one gets created (and saved back) on first use.
    assert stored.subscription == ""


async def test_create_event_rejects_a_missing_topic(mcp):
    backend = InMemoryEventBackend()
    _register(mcp, _Container(event_backend=backend))

    result = str(await mcp.call_tool("create_event", {
        "event_id": "orders-events", "name": "Order events", "topic": "  ",
    }))

    assert "needs a topic" in result
    assert await backend.get("orders-events") is None


async def test_create_event_rejects_invalid_schema_json(mcp):
    backend = InMemoryEventBackend()
    _register(mcp, _Container(event_backend=backend))

    result = str(await mcp.call_tool("create_event", {
        "event_id": "orders-events", "name": "n", "topic": "orders",
        "event_schema_json": "{not json",
    }))

    assert "Invalid event_schema_json" in result
    assert await backend.get("orders-events") is None


async def test_create_event_refuses_a_name_another_event_already_uses(mcp):
    backend = InMemoryEventBackend()
    _register(mcp, _Container(event_backend=backend))
    await mcp.call_tool("create_event", {
        "event_id": "orders-events", "name": "Order events", "topic": "orders",
    })

    result = str(await mcp.call_tool("create_event", {
        "event_id": "orders-v2", "name": "Order events", "topic": "orders-v2",
    }))

    assert "already exists" in result
    assert await backend.get("orders-v2") is None


async def test_create_pubsub_datasource_still_writes_an_event(mcp):
    """The pre-events tool name keeps working for agents mid-conversation."""
    backend = InMemoryEventBackend()
    _register(mcp, _Container(event_backend=backend))

    result = str(await mcp.call_tool("create_pubsub_datasource", {
        "source_id": "orders-events", "name": "Order events", "topic": "orders",
    }))

    assert "created" in result
    assert (await backend.get("orders-events")).topic == "orders"


async def test_update_event_changes_only_what_it_names(mcp):
    backend = InMemoryEventBackend()
    _register(mcp, _Container(event_backend=backend))
    await mcp.call_tool("create_event", {
        "event_id": "orders-events", "name": "Order events", "topic": "orders",
        "event_schema_json": '{"type": "object"}',
    })

    result = str(await mcp.call_tool("update_event", {
        "event_id": "Order events", "subscription": "projects/p/subscriptions/mine",
    }))

    assert "updated" in result
    stored = await backend.get("orders-events")
    assert stored.subscription == "projects/p/subscriptions/mine"
    assert stored.topic == "orders"
    assert stored.event_schema == {"type": "object"}


async def test_delete_event_removes_it(mcp):
    backend = InMemoryEventBackend()
    _register(mcp, _Container(event_backend=backend))
    await mcp.call_tool("create_event", {
        "event_id": "orders-events", "name": "Order events", "topic": "orders",
    })

    result = str(await mcp.call_tool("delete_event", {"event_id": "orders-events"}))

    assert "deleted" in result
    assert await backend.get("orders-events") is None


async def test_list_events_describes_an_event_by_topic(mcp):
    backend = InMemoryEventBackend()
    _register(mcp, _Container(event_backend=backend))
    await mcp.call_tool("create_event", {
        "event_id": "orders-events", "name": "Order events", "topic": "orders",
        "description": "Shop orders",
    })

    result = str(await mcp.call_tool("list_events", {}))

    assert "topic: orders" in result
    assert "subscription: (created on first use)" in result


async def test_list_pubsub_subscriptions_reports_disabled_triggers(mcp):
    _register(mcp, _Container())

    result = str(await mcp.call_tool("list_pubsub_subscriptions", {}))

    assert "PUBSUB_ENABLED" in result


async def test_list_pubsub_subscriptions_lists_live_registrations(mcp):
    container = _Container()
    container.pubsub_subscriber = MagicMock()
    container.pubsub_subscriber.registrations.return_value = {
        "orders-wf:on_order": "projects/p/subscriptions/aac-orders-wf-on_order",
    }
    _register(mcp, container)

    result = str(await mcp.call_tool("list_pubsub_subscriptions", {}))

    assert "orders-wf:on_order" in result
    assert "aac-orders-wf-on_order" in result


# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------

def _run(status: str = "running") -> GraphRun:
    return GraphRun(id="run-1", graph_id="wf", user_request="do it", status=status)


async def test_terminate_run_happy_path(mcp, monkeypatch):
    run = _run("running")
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=run)
    container = _Container(run_repository=repo)
    _register(mcp, container)

    monkeypatch.setattr(
        "app.services.agent_cleanup.cleanup_run_agents", AsyncMock()
    )
    result = str(await mcp.call_tool("terminate_run", {"run_id": "run-1"}))

    assert "terminated and marked failed" in result
    assert run.status == "failed"
    repo.update.assert_awaited()


async def test_terminate_run_not_found_returns_error_string(mcp):
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    _register(mcp, _Container(run_repository=repo))

    result = str(await mcp.call_tool("terminate_run", {"run_id": "nope"}))
    assert "Error (404)" in result


async def test_terminate_run_conflict_returns_error_string(mcp):
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=_run("completed"))
    _register(mcp, _Container(run_repository=repo))

    result = str(await mcp.call_tool("terminate_run", {"run_id": "run-1"}))
    assert "Error (409)" in result


async def test_approve_run_schedules_resume_and_returns_immediately(mcp, monkeypatch):
    run = _run("waiting_approval")
    run.current_step = "gate"
    run.step_statuses = {"gate": "waiting_approval"}
    repo = AsyncMock()
    repo.claim_for_resume = AsyncMock(return_value=run)
    container = _Container(run_repository=repo)
    container.live_runners[run.id] = MagicMock()
    _register(mcp, container)

    resumed: dict = {}

    async def fake_resume(runner, r, c, corrections, **kwargs):
        resumed.update({"corrections": corrections, **kwargs})

    monkeypatch.setattr(run_control, "_resume_approved", fake_resume)

    result = str(await mcp.call_tool(
        "approve_run", {"run_id": "run-1", "corrections_json": '{"plan": "ok"}'}
    ))
    assert "approved; resuming" in result
    # the approval step flips synchronously, the resume runs as a task
    assert run.step_statuses["gate"] == "finished"
    await asyncio.sleep(0)
    assert resumed == {
        "corrections": {"plan": "ok"},
        "approver_name": "management-mcp",
        "approver_id": None,
        "approver_source": "mcp",
    }


async def test_approve_run_conflict_returns_error_string(mcp):
    repo = AsyncMock()
    repo.claim_for_resume = AsyncMock(return_value=None)
    repo.get = AsyncMock(return_value=_run("running"))
    _register(mcp, _Container(run_repository=repo))

    result = str(await mcp.call_tool("approve_run", {"run_id": "run-1"}))
    assert "Error (409)" in result


async def test_reject_run_records_mcp_approver(mcp, monkeypatch):
    run = _run("waiting_approval")
    repo = AsyncMock()
    repo.claim_for_resume = AsyncMock(return_value=run)
    container = _Container(run_repository=repo)
    container.live_runners[run.id] = MagicMock()
    _register(mcp, container)

    seen: dict = {}

    async def fake_resume(runner, r, c, reason, **kwargs):
        seen.update({"reason": reason, **kwargs})

    monkeypatch.setattr(run_control, "_resume_rejected", fake_resume)

    result = str(await mcp.call_tool("reject_run", {"run_id": "run-1", "reason": "nah"}))
    assert "rejected" in result
    await asyncio.sleep(0)
    assert seen == {
        "reason": "nah",
        "approver_name": "management-mcp",
        "approver_id": None,
        "approver_source": "mcp",
    }


async def test_list_runs_calls_the_real_repository_interface(mcp):
    """``list_runs`` must call a method the concrete repository actually has.

    Regression test: the core used to call ``run_repository.list()``, which has
    never existed on ``MongoGraphRunRepository`` (the only implementation) — the
    real method is ``list_recent``. Every other test fakes the repository with a
    bare ``AsyncMock``, where any attribute name resolves, so the broken call
    passed the suite and only failed against a live backend. Speccing the mock
    to the real class is what makes a wrong name fail here.
    """
    repo = AsyncMock(spec=MongoGraphRunRepository)
    repo.list_recent = AsyncMock(return_value=[_run("running")])
    _register(mcp, _Container(run_repository=repo))

    result = str(await mcp.call_tool("list_runs", {"limit": 5}))

    repo.list_recent.assert_awaited_once_with(limit=5, workflow_id=None)
    assert "run-1" in result


async def test_list_runs_clamps_limit_and_passes_workflow_filter(mcp):
    repo = AsyncMock(spec=MongoGraphRunRepository)
    repo.list_recent = AsyncMock(return_value=[])
    _register(mcp, _Container(run_repository=repo))

    assert "No runs found." in str(
        await mcp.call_tool("list_runs", {"workflow_id": "wf1", "limit": 999})
    )
    repo.list_recent.assert_awaited_once_with(limit=20, workflow_id="wf1")


# ---------------------------------------------------------------------------
# JSON-carrying params must survive FastMCP's argument pre-parsing
# ---------------------------------------------------------------------------
# FastMCP pre-parses a string argument whenever the annotation is not exactly
# `str` (mcp/server/fastmcp/utilities/func_metadata.py: `field_info.annotation
# is not str`). Annotating a JSON param `str | None` therefore turned the
# payload into a list/dict and then failed validation against that same
# annotation, so the field could not be set at all — update_datasource could
# not change operations or credentials, and a stored token could never be
# rotated. These tests go through `call_tool`, which is the path that pre-parses.

async def test_update_datasource_can_replace_its_operations(mcp):
    backend = InMemoryDataSourceBackend()
    _register(mcp, _Container(data_source_backend=backend))
    await mcp.call_tool("create_datasource", {
        "source_id": "api", "name": "API", "base_url": "https://api.example",
        "operations_json": '[{"name": "list_things", "path": "/things"}]',
    })

    result = str(await mcp.call_tool("update_datasource", {
        "source_id": "api",
        "operations_json": '[{"name": "list_things", "path": "/things"},'
                           ' {"name": "get_thing", "path": "/things/{params.id}",'
                           '  "params": [{"name": "id", "type": "string", "required": true}]}]',
    }))

    assert "updated" in result.lower() or "error" not in result.lower()
    stored = await backend.get("api")
    assert [op.name for op in stored.operations] == ["list_things", "get_thing"]


async def test_update_datasource_can_rotate_the_stored_credential(mcp):
    backend = InMemoryDataSourceBackend()
    _register(mcp, _Container(data_source_backend=backend))
    await mcp.call_tool("create_datasource", {
        "source_id": "api", "name": "API", "base_url": "https://api.example",
        "operations_json": '[{"name": "list_things", "path": "/things"}]',
        "auth_json": '{"type": "bearer", "token": "old-token"}',
    })

    await mcp.call_tool("update_datasource", {
        "source_id": "api", "auth_json": '{"type": "bearer", "token": "new-token"}',
    })

    stored = await backend.get("api")
    assert stored.auth.token == "new-token"
    # Untouched fields keep their stored values.
    assert [op.name for op in stored.operations] == ["list_things"]
    assert stored.base_url == "https://api.example"


async def test_update_datasource_without_json_fields_keeps_them(mcp):
    backend = InMemoryDataSourceBackend()
    _register(mcp, _Container(data_source_backend=backend))
    await mcp.call_tool("create_datasource", {
        "source_id": "api", "name": "API", "base_url": "https://api.example",
        "operations_json": '[{"name": "list_things", "path": "/things"}]',
        "auth_json": '{"type": "bearer", "token": "keep-me"}',
    })

    await mcp.call_tool("update_datasource", {"source_id": "api", "name": "Renamed"})

    stored = await backend.get("api")
    assert stored.name == "Renamed"
    assert stored.auth.token == "keep-me"
    assert [op.name for op in stored.operations] == ["list_things"]


async def test_update_event_can_replace_its_schema(mcp):
    backend = InMemoryEventBackend()
    _register(mcp, _Container(event_backend=backend))
    await mcp.call_tool("create_event", {
        "event_id": "orders-events", "name": "Order events", "topic": "orders",
        "event_schema_json": '{"type": "object"}',
    })

    await mcp.call_tool("update_event", {
        "event_id": "orders-events",
        "event_schema_json": '{"type": "object", "required": ["orderId"]}',
    })

    stored = await backend.get("orders-events")
    assert stored.event_schema == {"type": "object", "required": ["orderId"]}


async def test_update_workflow_can_replace_its_steps(mcp):
    container = _Container()
    workflow_backend = AsyncMock()
    container.workflow_backend = workflow_backend
    _register(mcp, container)

    result = str(await mcp.call_tool("update_workflow", {
        "workflow_id": "wf",
        "steps_json": '[{"id": "one", "type": "python"}, {"id": "two", "type": "python"}]',
    }))

    # The payload reached the core function as a string it could parse — the
    # pre-parse bug surfaced here as a validation error before any call landed.
    assert "Input should be a valid string" not in result


async def test_create_workflow_can_be_called_disabled_and_with_storage():
    """Both flags must be reachable from the MCP surface, not just the core.

    They were added to the core first and the wrapper kept its old four-argument
    signature, so `enabled=False` and `use_storage=True` were silently dropped: a
    workflow carrying a cron trigger came out live, and its `storage` steps came
    out failing. Neither shows up as an error at creation time, which is what
    makes it worth pinning here.
    """
    mcp = build_management_mcp()
    _register(mcp, _Container())
    schemas = {t.name: t.inputSchema for t in await mcp.list_tools()}

    create_props = schemas["create_workflow"]["properties"]
    assert "enabled" in create_props, "cannot create a workflow already disabled"
    assert "use_storage" in create_props, "cannot create a workflow with storage on"

    update_props = schemas["update_workflow"]["properties"]
    assert "use_storage" in update_props, "cannot turn storage on after the fact"
