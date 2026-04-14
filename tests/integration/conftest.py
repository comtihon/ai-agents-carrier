"""
Shared fixtures for integration tests.

Prerequisites
─────────────
A real MongoDB instance must be reachable at mongodb://localhost:27017.
All integration tests are skipped automatically when it is not.

Pattern
───────
Each test calls ``build_int_client(graph_def, llm, mcp_tools={})`` which:
  - wires up a real MongoGraphRunRepository against the test database,
  - drops the collection for isolation,
  - injects the graph runner + fake dependencies into the FastAPI app,
  - returns (AsyncClient, MongoClientProvider) so callers can close cleanly.

``make_mock_llm(structured_responses, text_responses)`` builds a MagicMock
LLM that:
  - drives ``.ainvoke()`` calls from *text_responses* (returns AIMessage),
  - drives ``.with_structured_output(schema).ainvoke()`` from *structured_responses*
    (returns a populated Pydantic instance of *schema*).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, YamlGraphRunner as _Runner
from app.infrastructure.persistence.mongo import MongoClientProvider
from app.infrastructure.tools.mcp_client import McpToolsProvider

_TEST_DB = "test_langgraph_integration"
_MONGO_URI = "mongodb://localhost:27017"


# ---------------------------------------------------------------------------
# Session-scoped connectivity check — skip all integration tests when MongoDB
# is not available so the suite still passes in environments without it.
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require a running MongoDB instance",
    )


@pytest.fixture(scope="session", autouse=True)
def _require_mongodb():
    """Skip integration tests when MongoDB is not reachable."""
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError

    try:
        client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        client.close()
    except ServerSelectionTimeoutError:
        pytest.skip("MongoDB not reachable at localhost:27017 — skipping integration tests")


# ---------------------------------------------------------------------------
# LLM mock factory
# ---------------------------------------------------------------------------

def make_mock_llm(
    structured_responses: list[dict[str, Any]] | None = None,
    text_responses: list[str] | None = None,
) -> MagicMock:
    """
    Build a mock LLM that returns deterministic responses without a real API key.

    *structured_responses* — consumed in order for ``.bind_tools(...).ainvoke()`` calls
                             (llm_structured steps). Returned as a submit_output tool call.
    *text_responses*       — consumed in order for direct ``.ainvoke()`` calls (llm steps).
    Both lists cycle back to their last element once exhausted.
    """
    s_iter = _cycling_iter(structured_responses or [{}])
    t_iter = _cycling_iter(text_responses or ["default response"])

    llm = MagicMock()

    async def _ainvoke(messages, **kwargs):
        return AIMessage(content=next(t_iter))

    llm.ainvoke = AsyncMock(side_effect=_ainvoke)

    def _bind_tools(tools, **kwargs):
        chain = MagicMock()

        async def _chain_ainvoke(messages, **kwargs):
            data = next(s_iter)
            return AIMessage(
                content="",
                tool_calls=[{"name": _Runner._SUBMIT_TOOL, "args": data, "id": "mock-tc-id"}],
            )

        chain.ainvoke = AsyncMock(side_effect=_chain_ainvoke)
        return chain

    llm.bind_tools = MagicMock(side_effect=_bind_tools)
    return llm


def _cycling_iter(lst: list):
    """Yield items from *lst*, repeating the last one indefinitely."""
    for item in lst:
        yield item
    while True:
        yield lst[-1]


# ---------------------------------------------------------------------------
# Client builder (called directly in each test)
# ---------------------------------------------------------------------------

async def build_int_client(
    graph_def: dict[str, Any],
    llm: Any,
    mcp_tools: dict[str, Any] | None = None,
) -> tuple[AsyncClient, MongoClientProvider]:
    """
    Build a full integration client with real MongoDB persistence.

    Returns (AsyncClient, MongoClientProvider).
    The caller is responsible for closing the MongoClientProvider when done.
    """
    settings = Settings(
        mongodb_uri=_MONGO_URI,
        mongodb_database=_TEST_DB,
        openhands_mock_mode=True,
        environment="test",
    )
    mongo_provider = MongoClientProvider(settings)
    repo = mongo_provider.get_repository()
    await repo._collection.delete_many({})

    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(side_effect=lambda name: (mcp_tools or {}).get(name))
    mcp.get_tools = MagicMock(return_value=[])

    runner = YamlGraphRunner(graph_def, llm=llm, mcp_tools_provider=mcp)
    registry = YamlGraphRegistry({runner.id: runner})

    openhands_adapter = MagicMock(spec=OpenHandsAdapter)
    openhands_adapter.execute = AsyncMock(return_value={"status": "success", "mock": True})

    container = ApplicationContainer(
        settings=settings,
        llm=llm,
        mcp_tools_provider=mcp,
        yaml_graph_registry=registry,
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=openhands_adapter,
    )

    app = create_app()
    app.state.container = container

    http_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return http_client, mongo_provider
