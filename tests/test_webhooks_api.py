from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider

# The webhook workflow uses a bearer-authenticated http trigger step so the
# request can be signed with a simple Authorization header.
_WEBHOOK_WORKFLOW = {
    "id": "hooked",
    "steps": [
        {
            "id": "trigger",
            "type": "http",
            "auth_mode": "bearer",
            "bearer_token": "test-token",
            "output_key": "payload",
        }
    ],
}


class _CapturingLiveRunners(dict):
    """Remembers the last inserted runner even after it gets popped — the
    webhook run completes synchronously within the request/response cycle, so
    live_runners is already empty by the time the POST resolves."""

    def __setitem__(self, key, value):
        self.last_inserted = value
        super().__setitem__(key, value)


def _make_defn():
    defn = MagicMock()
    defn.id = "hooked"
    defn.readonly = False
    defn.steps = _WEBHOOK_WORKFLOW["steps"]
    defn.to_raw_dict = MagicMock(return_value=_WEBHOOK_WORKFLOW)
    return defn


def _make_repo() -> AsyncMock:
    repo = AsyncMock(spec=MongoGraphRunRepository)
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.get = AsyncMock(
        return_value=GraphRun(id="tid1", graph_id="hooked", user_request="hello", status="running")
    )
    return repo


def _make_mcp() -> MagicMock:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.start = AsyncMock()
    mcp.stop = AsyncMock()
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])
    return mcp


def _build_container(*, with_repos: bool) -> ApplicationContainer:
    settings = Settings()
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    workflow_backend = MagicMock()
    workflow_backend.get = AsyncMock(return_value=_make_defn())

    kwargs = dict(
        settings=settings,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=_make_mcp(),
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=_make_repo(),
        openhands=MagicMock(spec=OpenHandsAdapter),
        workflow_backend=workflow_backend,
    )
    if with_repos:
        kwargs["warm_pod_repository"] = MagicMock()
        kwargs["pvc_lease_repository"] = MagicMock()
        kwargs["agent_task_repository"] = MagicMock()
    container = ApplicationContainer(**kwargs)
    container.live_runners = _CapturingLiveRunners()
    return container


async def _post(container: ApplicationContainer):
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.post(
            "/api/v1/webhooks/hooked",
            json={"request": "hello"},
            headers={"Authorization": "Bearer test-token"},
        )


@pytest.mark.asyncio
async def test_webhook_injects_runner_dependencies():
    """A webhook-triggered run must get the shared repositories injected and the
    callback base URL wired from settings."""
    container = _build_container(with_repos=True)
    resp = await _post(container)
    assert resp.status_code == 202, resp.text

    runner = container.live_runners.last_inserted
    assert runner._agent_task_repository is container.agent_task_repository
    assert runner._pvc_lease_repository is container.pvc_lease_repository
    assert runner._warm_pod_repository is container.warm_pod_repository
    assert runner._callback_base_url == (
        container.settings.agent_callback_url or container.settings.base_url
    )


@pytest.mark.asyncio
async def test_webhook_prefers_agent_callback_url():
    """When AGENT_CALLBACK_URL is configured, the runner's callback base URL must
    resolve to it rather than the (possibly external/OAuth-protected) base_url."""
    container = _build_container(with_repos=True)
    container.settings.agent_callback_url = "http://agent-callback.internal"
    resp = await _post(container)
    assert resp.status_code == 202, resp.text

    runner = container.live_runners.last_inserted
    assert runner._callback_base_url == "http://agent-callback.internal"
    assert runner._callback_base_url != container.settings.base_url


@pytest.mark.asyncio
async def test_webhook_does_not_force_inject_none_dependencies():
    """When the container has no repos configured, the runner's dependency attrs
    must stay None rather than being overwritten."""
    container = _build_container(with_repos=False)
    resp = await _post(container)
    assert resp.status_code == 202, resp.text

    runner = container.live_runners.last_inserted
    assert runner._agent_task_repository is None
    assert runner._pvc_lease_repository is None
    assert runner._warm_pod_repository is None


@pytest.mark.asyncio
async def test_webhook_rejects_bad_bearer_token():
    container = _build_container(with_repos=True)
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(
            "/api/v1/webhooks/hooked",
            json={"request": "hello"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert resp.status_code == 403
