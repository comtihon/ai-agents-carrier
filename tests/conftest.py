from __future__ import annotations

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock

from app.api.app import create_app
from app.core.container import ApplicationContainer
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.core.config import Settings


def _fake_llm() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(
        responses=[AIMessage(content="fake response")]
    )


def _fake_container(registry: YamlGraphRegistry) -> ApplicationContainer:
    settings = Settings(
        llm_provider=None,
        mongodb_uri="mongodb://localhost:27017",
        mongodb_database="test",
    )
    repo = AsyncMock(spec=MongoGraphRunRepository)
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    openhands = MagicMock(spec=OpenHandsAdapter)
    mongo_provider = MagicMock()
    return ApplicationContainer(
        settings=settings,
        llm=_fake_llm(),
        mcp_tools_provider=mcp,
        yaml_graph_registry=registry,
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=openhands,
    )


@pytest.fixture
def fake_llm():
    return _fake_llm()
