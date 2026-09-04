"""Tests for the `data_source` yaml_graph step type."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.tools.mcp_client import McpToolsProvider

_DEFINITION = DataSourceDefinition.model_validate({
    "id": "github",
    "name": "GitHub",
    "base_url": "https://api.github.com",
    "operations": [
        {"name": "list_repos", "path": "/users/{params.owner}/repos",
         "params": [{"name": "owner"}]},
    ],
})


def _make_runner(*, backend, executor) -> YamlGraphRunner:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="x")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    definition = {"id": "ds-graph", "steps": [
        {
            "id": "fetch",
            "type": "data_source",
            "source": "github",
            "operation": "list_repos",
            "params": {"owner": "{request}"},
            "output_key": "repos",
        },
    ]}
    runner = YamlGraphRunner(definition, llm=llm, mcp_tools_provider=mcp)
    runner._data_source_backend = backend
    runner._data_source_executor = executor
    return runner


async def test_data_source_node_stores_executor_result():
    backend = AsyncMock()
    backend.get = AsyncMock(return_value=_DEFINITION)
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=[{"name": "repo-a"}])

    runner = _make_runner(backend=backend, executor=executor)
    state = await runner.graph.ainvoke(
        {"request": "acme"},
        {"configurable": {"thread_id": "ds1"}},
    )

    assert state["repos"] == [{"name": "repo-a"}]
    backend.get.assert_awaited_once_with("github")
    executor.execute.assert_awaited_once_with(_DEFINITION, "list_repos", {"owner": "acme"}, limit=None)


async def test_data_source_node_captures_executor_error():
    backend = AsyncMock()
    backend.get = AsyncMock(return_value=_DEFINITION)
    executor = AsyncMock()
    executor.execute = AsyncMock(side_effect=RuntimeError("boom"))

    runner = _make_runner(backend=backend, executor=executor)
    state = await runner.graph.ainvoke(
        {"request": "acme"},
        {"configurable": {"thread_id": "ds2"}},
    )
    assert state["repos"] == {"error": "boom"}


async def test_data_source_node_errors_when_source_missing():
    backend = AsyncMock()
    backend.get = AsyncMock(return_value=None)

    runner = _make_runner(backend=backend, executor=AsyncMock())
    state = await runner.graph.ainvoke(
        {"request": "acme"},
        {"configurable": {"thread_id": "ds3"}},
    )
    assert "not found" in state["repos"]["error"]


async def test_data_source_node_errors_without_injected_dependencies():
    runner = _make_runner(backend=None, executor=None)
    state = await runner.graph.ainvoke(
        {"request": "acme"},
        {"configurable": {"thread_id": "ds4"}},
    )
    assert "not configured" in state["repos"]["error"]
