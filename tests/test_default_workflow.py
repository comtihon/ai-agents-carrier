"""
Unit tests for the default ReAct chat agent.

Covers:
- "develop a feature X" → LLM calls run_workflow tool → GraphRun created, background task spawned.
- "2+2" → LLM replies directly, no tool calls, no workflow spawned.
- "do the thing" (ambiguous) → LLM calls ask_user → graph pauses at interrupt,
  resumes with answers, LLM produces final reply.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.default_workflow import build_default_workflow


# ── helpers ───────────────────────────────────────────────────────────────────

def _tool_call_msg(tool_name: str, args: dict, call_id: str = "tc_1") -> AIMessage:
    """AIMessage carrying a single tool call — triggers the tools node."""
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _fake_registry(workflow_ids: list[str] | None = None) -> MagicMock:
    workflow_ids = workflow_ids or ["develop-a-ticket"]
    runner = MagicMock()
    runner.name = "Develop a Ticket"
    runner.steps = [{"id": "analyze", "type": "llm"}, {"id": "plan", "type": "llm"}]

    registry = MagicMock()
    registry.list_ids.return_value = workflow_ids
    registry.list_definitions.return_value = [
        {"id": wid, "name": wid.replace("-", " ").title(), "description": f"Workflow {wid}.", "steps": []}
        for wid in workflow_ids
    ]
    registry.get.side_effect = lambda wid: runner if wid in workflow_ids else None
    return registry


def _fake_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.list = AsyncMock(return_value=[])
    repo.get = AsyncMock(return_value=None)
    return repo


def _llm_with_responses(responses: list) -> MagicMock:
    """LLM whose bind_tools(…).ainvoke(…) returns *responses* in sequence."""
    bound = MagicMock()
    bound.ainvoke = AsyncMock(side_effect=responses)
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    return llm


_BASE_STATE: dict = {"messages": [], "copilotkit": {"actions": [], "context": []}}
_STREAM_FN = "app.infrastructure.orchestration.default_workflow.stream_graph_to_pause"


# ── test: run_workflow tool spawns child workflow ─────────────────────────────

@pytest.mark.asyncio
async def test_run_workflow_tool_spawns_child_workflow():
    """
    Input  : "develop feature X"
    LLM 1  : calls run_workflow(workflow_id="develop-a-ticket", request="develop feature X")
    LLM 2  : confirmation reply after tool result
    Expect : GraphRun created with correct fields, stream_graph_to_pause called.
    """
    llm = _llm_with_responses([
        _tool_call_msg("run_workflow", {
            "workflow_id": "develop-a-ticket",
            "request": "develop feature X",
        }),
        AIMessage(content="Started the workflow."),
    ])
    registry = _fake_registry(["develop-a-ticket"])
    repo = _fake_repo()

    graph = build_default_workflow(llm, registry, repo)

    with patch(_STREAM_FN, new_callable=AsyncMock) as mock_stream:
        result = await graph.ainvoke(
            {**_BASE_STATE, "messages": [HumanMessage(content="develop feature X")]},
            {"configurable": {"thread_id": "test-spawn"}},
        )
        await asyncio.sleep(0)

    # child run persisted
    repo.create.assert_awaited_once()
    created_run: GraphRun = repo.create.call_args[0][0]
    assert created_run.graph_id == "develop-a-ticket"
    assert created_run.user_request == "develop feature X"
    assert created_run.status == "running"
    assert set(created_run.step_statuses.values()) == {"pending"}

    # background streaming scheduled
    mock_stream.assert_called_once()
    stream_args = mock_stream.call_args[0]
    assert stream_args[1] is created_run
    assert stream_args[2] is repo
    assert stream_args[3] == {"request": "develop feature X"}

    # final AIMessage present
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    assert ai_msgs, "Expected at least one final AIMessage"


# ── test: direct reply without tool calls ────────────────────────────────────

@pytest.mark.asyncio
async def test_arithmetic_question_returns_direct_reply():
    """
    Input  : "2+2"
    LLM    : AIMessage(content="4") — no tool calls
    Expect : "4" returned, no GraphRun created, no task scheduled.
    """
    llm = _llm_with_responses([AIMessage(content="4")])
    registry = _fake_registry(["develop-a-ticket"])
    repo = _fake_repo()

    graph = build_default_workflow(llm, registry, repo)

    with patch(_STREAM_FN, new_callable=AsyncMock) as mock_stream:
        result = await graph.ainvoke(
            {**_BASE_STATE, "messages": [HumanMessage(content="2+2")]},
            {"configurable": {"thread_id": "test-reply"}},
        )
        await asyncio.sleep(0)

    repo.create.assert_not_awaited()
    mock_stream.assert_not_called()

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert ai_msgs, "Expected at least one AIMessage"
    assert ai_msgs[-1].content == "4"


# ── test: ask_user tool pauses and resumes ────────────────────────────────────

@pytest.mark.asyncio
async def test_ask_context_pauses_and_resumes():
    """
    Input  : "do the thing" (ambiguous)
    LLM 1  : calls ask_user(questions=["Which thing?"])  → interrupt fires
    Resume : answers={"0": "deploy the app"}
    LLM 2  : AIMessage("Got it!")
    Expect : graph pauses with ask_context interrupt, resumes to produce reply.
    """
    from langgraph.types import Command

    llm = _llm_with_responses([
        _tool_call_msg("ask_user", {"questions": ["Which thing?"]}, call_id="tc_ask"),
        AIMessage(content="Got it!"),
    ])
    registry = _fake_registry(["develop-a-ticket"])
    repo = _fake_repo()

    graph = build_default_workflow(llm, registry, repo)
    config = {"configurable": {"thread_id": "test-ask"}}

    with patch(_STREAM_FN, new_callable=AsyncMock):
        await graph.ainvoke(
            {**_BASE_STATE, "messages": [HumanMessage(content="do the thing")]},
            config,
        )

    # graph should have paused at ask_context interrupt
    snap = graph.get_state(config)
    interrupt_vals = [
        intr.value
        for task in snap.tasks
        for intr in getattr(task, "interrupts", [])
    ]
    assert any(
        isinstance(v, dict) and v.get("type") == "ask_context"
        for v in interrupt_vals
    ), f"Expected ask_context interrupt, got: {interrupt_vals}"

    # resume with user answers
    with patch(_STREAM_FN, new_callable=AsyncMock):
        result2 = await graph.ainvoke(Command(resume={"0": "deploy the app"}), config)
        await asyncio.sleep(0)

    ai_msgs = [m for m in result2["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    assert ai_msgs, "Expected AIMessage after resume"
    assert ai_msgs[-1].content == "Got it!"


# ── the internal agent: MCP tools + schema-driven data source tools ───────────

class _FakeMcpProvider:
    """Stand-in for McpToolsProvider with a mutable tool list."""

    def __init__(self, tools: list) -> None:
        self.tools = tools

    def get_tools(self) -> list:
        return list(self.tools)

    def get_tool_server(self, name: str) -> str | None:
        return next((server for tool, server in self.tools_with_servers if tool.name == name), None)

    @property
    def tools_with_servers(self):
        return [(t, getattr(t, "_server", "datasources")) for t in self.tools]


def _fake_mcp_tool(name: str, server: str = "datasources"):
    from langchain_core.tools import tool as make_tool

    @make_tool(name)
    def _t(query: str = "") -> str:
        """Fake MCP tool."""
        return f"{name} ran"

    _t._server = server
    return _t


def _bound_tool_names(llm) -> list[str]:
    """Tool names of the most recent bind_tools() call."""
    return [t.name for t in llm.bind_tools.call_args[0][0]]


@pytest.mark.asyncio
async def test_mcp_tools_are_exposed_to_the_chat_agent():
    llm = _llm_with_responses([AIMessage(content="ok")])
    provider = _FakeMcpProvider([_fake_mcp_tool("jira_search", server="jira")])

    graph = build_default_workflow(
        llm, _fake_registry(), _fake_repo(), mcp_tools_provider=provider
    )
    await graph.ainvoke(
        {**_BASE_STATE, "messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": "test-mcp"}},
    )

    names = _bound_tool_names(llm)
    assert "jira_search" in names
    assert "create_datasource_from_schema" in names


@pytest.mark.asyncio
async def test_mcp_tool_list_is_resolved_per_invocation():
    """A data source saved mid-session republishes its MCP tools; the agent must see them."""
    llm = _llm_with_responses([AIMessage(content="one"), AIMessage(content="two")])
    provider = _FakeMcpProvider([])

    graph = build_default_workflow(
        llm, _fake_registry(), _fake_repo(), mcp_tools_provider=provider
    )
    config = {"configurable": {"thread_id": "test-refresh"}}
    await graph.ainvoke({**_BASE_STATE, "messages": [HumanMessage(content="hi")]}, config)
    assert "datasource_github_list_repos" not in _bound_tool_names(llm)

    provider.tools.append(_fake_mcp_tool("datasource_github_list_repos"))
    await graph.ainvoke({**_BASE_STATE, "messages": [HumanMessage(content="again")]}, config)
    assert "datasource_github_list_repos" in _bound_tool_names(llm)


@pytest.mark.asyncio
async def test_mcp_servers_allow_list_narrows_the_tools():
    llm = _llm_with_responses([AIMessage(content="ok")])
    provider = _FakeMcpProvider([
        _fake_mcp_tool("datasource_call", server="datasources"),
        _fake_mcp_tool("jira_search", server="jira"),
    ])

    graph = build_default_workflow(
        llm, _fake_registry(), _fake_repo(),
        agent_config={"system_prompt": "p", "mcp_servers": ["datasources"]},
        mcp_tools_provider=provider,
    )
    await graph.ainvoke(
        {**_BASE_STATE, "messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": "test-allow"}},
    )

    names = _bound_tool_names(llm)
    assert "datasource_call" in names
    assert "jira_search" not in names


@pytest.mark.asyncio
async def test_platform_tool_wins_over_a_colliding_mcp_tool():
    llm = _llm_with_responses([AIMessage(content="ok")])
    provider = _FakeMcpProvider([_fake_mcp_tool("list_workflows", server="rogue")])

    graph = build_default_workflow(
        llm, _fake_registry(), _fake_repo(), mcp_tools_provider=provider
    )
    await graph.ainvoke(
        {**_BASE_STATE, "messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": "test-collision"}},
    )

    assert _bound_tool_names(llm).count("list_workflows") == 1


_SPEC_RESULT = {
    "kind": "openapi",
    "source": "https://api.test/openapi.json",
    "base_url": "https://api.test/v1",
    "operations": [
        {
            "name": "listpets",
            "method": "GET",
            "path": "/pets?limit={params.limit}",
            "params": [{"name": "limit", "type": "number", "required": False, "description": ""}],
            "response_schema": {"type": "array"},
            "mapping": None,
            "summary": "List pets",
        },
        {
            "name": "deletepet",
            "method": "DELETE",
            "path": "/pets/{params.id}",
            "params": [{"name": "id", "type": "string", "required": True, "description": ""}],
            "response_schema": None,
            "mapping": None,
            "summary": "",
        },
    ],
}

_FETCH_FN = "app.infrastructure.datasources.discovery.fetch_and_parse_spec"


def _ds_backend() -> AsyncMock:
    backend = AsyncMock()
    backend.get = AsyncMock(return_value=None)
    backend.list = AsyncMock(return_value=[])
    backend.create = AsyncMock()
    backend.update = AsyncMock()
    return backend


async def _run_tool(graph_llm_messages, tool_name: str, args: dict, **build_kwargs) -> str:
    """Drive one tool call through the graph and return its ToolMessage content."""
    from langchain_core.messages import ToolMessage

    llm = _llm_with_responses([
        _tool_call_msg(tool_name, args, call_id="tc_ds"),
        AIMessage(content="done"),
    ])
    graph = build_default_workflow(llm, _fake_registry(), _fake_repo(), **build_kwargs)
    result = await graph.ainvoke(
        {**_BASE_STATE, "messages": [HumanMessage(content=graph_llm_messages)]},
        {"configurable": {"thread_id": f"test-{tool_name}"}},
    )
    return next(m.content for m in result["messages"] if isinstance(m, ToolMessage))


@pytest.mark.asyncio
async def test_import_datasource_schema_lists_operations_without_storing():
    backend = _ds_backend()
    with patch(_FETCH_FN, new=AsyncMock(return_value=_SPEC_RESULT)):
        content = await _run_tool(
            "import the petstore",
            "import_datasource_schema",
            {"schema_url": "https://api.test/openapi.json"},
            data_source_backend=backend,
        )

    assert "Declared base URL: https://api.test/v1" in content
    assert "- listpets [GET] /pets?limit={params.limit} params: limit? — List pets" in content
    backend.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_datasource_from_schema_copies_the_parsed_operations():
    backend = _ds_backend()
    with patch(_FETCH_FN, new=AsyncMock(return_value=_SPEC_RESULT)):
        content = await _run_tool(
            "create the petstore source",
            "create_datasource_from_schema",
            {
                "source_id": "petstore",
                "name": "Petstore",
                "schema_url": "https://api.test/openapi.json",
                "operation_names": "listpets",
            },
            data_source_backend=backend,
        )

    assert "created" in content
    backend.create.assert_awaited_once()
    defn = backend.create.call_args[0][0]
    assert defn.id == "petstore"
    # Base URL defaults to the one the specification declares.
    assert defn.base_url == "https://api.test/v1"
    # Only the requested operation, copied verbatim (params and schema intact).
    assert [op.name for op in defn.operations] == ["listpets"]
    assert defn.operations[0].path == "/pets?limit={params.limit}"
    assert defn.operations[0].response_schema == {"type": "array"}
    assert [p.name for p in defn.operations[0].params] == ["limit"]


@pytest.mark.asyncio
async def test_create_datasource_from_schema_rejects_unknown_operation_names():
    backend = _ds_backend()
    with patch(_FETCH_FN, new=AsyncMock(return_value=_SPEC_RESULT)):
        content = await _run_tool(
            "create it",
            "create_datasource_from_schema",
            {
                "source_id": "petstore",
                "name": "Petstore",
                "schema_url": "https://api.test/openapi.json",
                "operation_names": "list_pets",
            },
            data_source_backend=backend,
        )

    assert "Unknown operation(s): list_pets" in content
    assert "listpets" in content
    backend.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_datasource_operations_from_schema_extends_without_overwriting():
    from app.domain.models.data_source_definition import DataSourceDefinition

    stored = DataSourceDefinition.model_validate({
        "id": "petstore",
        "name": "Petstore",
        "base_url": "https://api.test/v1",
        "operations": [{"name": "listpets", "path": "/pets"}],
    })
    backend = _ds_backend()
    backend.get = AsyncMock(return_value=stored)
    backend.list = AsyncMock(return_value=[stored])

    with patch(_FETCH_FN, new=AsyncMock(return_value=_SPEC_RESULT)):
        content = await _run_tool(
            "add delete",
            "add_datasource_operations_from_schema",
            {
                "source_id": "petstore",
                "schema_url": "https://api.test/openapi.json",
                "operation_names": "listpets,deletepet",
            },
            data_source_backend=backend,
        )

    assert "Added 1 operation(s)" in content
    assert "skipped, already present: listpets" in content
    defn = backend.update.call_args[0][1]
    assert [op.name for op in defn.operations] == ["listpets", "deletepet"]
    # The pre-existing operation keeps its own definition.
    assert defn.operations[0].path == "/pets"
