from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest_asyncio
from asgi_lifespan import LifespanManager

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import build_container

# Isolated test database — never touches the real one.
_TEST_SETTINGS = Settings(
    mongodb_uri="mongodb://localhost:27017",
    mongodb_database="test_langgraph_integration",
    openhands_mock_mode=True,
    environment="test",
    workflow_definitions_path="workflows",
)


@pytest_asyncio.fixture
async def client():
    """
    Spin up the full FastAPI app against a real MongoDB instance.

    build_container is patched so the lifespan always wires up to the test
    database regardless of whatever get_settings() returns from its lru_cache.
    The collection is wiped before each test to guarantee isolation.
    """
    with (
        patch("app.api.app.build_container", side_effect=lambda _: build_container(_TEST_SETTINGS)),
        patch("app.api.app._register_langserve_routes"),  # not under test; skips py3.14 compat issue
    ):
        app = create_app()
        async with LifespanManager(app):
            await app.state.container.workflow_run_repository._collection.delete_many({})
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as http_client:
                yield http_client
