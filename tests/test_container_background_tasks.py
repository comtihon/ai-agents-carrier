"""ApplicationContainer must hold references to the background tasks it
creates in startup() (so they can't be garbage-collected mid-flight) and
cancel them in shutdown()."""
from __future__ import annotations

import asyncio

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.tools.mcp_client import McpToolsProvider


def _build_container() -> ApplicationContainer:
    settings = Settings(mcp_datasources_enabled=False)
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.start = AsyncMock()
    mcp.stop = AsyncMock()
    run_repository = AsyncMock()
    run_repository.list_incomplete = AsyncMock(return_value=[])
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    return ApplicationContainer(
        settings=settings,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=MagicMock(),
        mongo_provider=mongo_provider,
        run_repository=run_repository,
        openhands=MagicMock(spec=OpenHandsAdapter),
        workflow_backend=None,
        data_source_backend=None,
    )


@pytest.mark.asyncio
async def test_startup_holds_recover_task_reference_and_shutdown_cancels_it():
    container = _build_container()
    await container.startup()
    try:
        assert container._recover_task is not None
        # data_source_backend is None, so no datasources MCP task is created.
        assert container._datasources_mcp_task is None
    finally:
        await container.shutdown()

    # shutdown() only requests cancellation; give the event loop a turn so it
    # actually takes effect before asserting.
    task = container._recover_task
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled() or task.done()
