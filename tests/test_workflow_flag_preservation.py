"""A PUT that omits a boolean flag must not switch that feature off.

There is no partial-update route: ``PUT /workflows/{id}`` replaces the whole
definition. So every flag needs "field omitted means keep the stored value",
otherwise any client that hand-assembles its payload and forgets one field
turns that feature off behind the user's back.

This is not hypothetical. ``use_storage`` was applied unconditionally from a
``bool = False`` default while the UI's update call did not send the field at
all, so every save from the workflow editor switched storage back off — the
checkbox looked like it would not stick, and a workflow whose steps depend on
storage silently lost it.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider

_STEPS = [{"id": "s1", "type": "llm", "output_key": "answer", "user_template": "{request}"}]


def _mcp() -> MagicMock:
    provider = MagicMock(spec=McpToolsProvider)
    provider.get_tools = AsyncMock(return_value=[])
    return provider


def _stored() -> WorkflowDefinition:
    """A workflow with both flags set away from their field defaults."""
    return WorkflowDefinition(
        id="wf",
        name="WF",
        description="",
        steps=_STEPS,
        use_meta_llm=False,   # field default is True
        use_storage=True,     # field default is False
        enabled=False,        # field default is True
    )


def _container() -> tuple[ApplicationContainer, MagicMock]:
    repo = AsyncMock(spec=MongoGraphRunRepository)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()

    backend = MagicMock()
    backend.get = AsyncMock(return_value=_stored())
    # update() echoes what it was handed, which is what the assertions read.
    backend.update = AsyncMock(side_effect=lambda _id, defn: defn)

    container = ApplicationContainer(
        settings=Settings(),
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=_mcp(),
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=MagicMock(spec=OpenHandsAdapter),
        workflow_backend=backend,
    )
    container.refresh_runner = AsyncMock()
    container.script_backend = None
    return container, backend


async def _client(container: ApplicationContainer) -> AsyncClient:
    app = create_app()
    app.state.container = container
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


_MINIMAL_PUT = {"name": "WF", "description": "", "steps": _STEPS}


@pytest.mark.asyncio
async def test_omitted_use_storage_keeps_it_on() -> None:
    """The exact regression: the UI used to PUT without this field."""
    container, backend = _container()
    async with await _client(container) as c:
        resp = await c.put("/api/v1/workflows/wf", json=_MINIMAL_PUT)
    assert resp.status_code == 200
    assert backend.update.await_args.args[1].use_storage is True
    assert resp.json()["use_storage"] is True


@pytest.mark.asyncio
async def test_omitted_use_meta_llm_keeps_it_off() -> None:
    """Same class of bug in the other direction: the default here is True."""
    container, backend = _container()
    async with await _client(container) as c:
        resp = await c.put("/api/v1/workflows/wf", json=_MINIMAL_PUT)
    assert resp.status_code == 200
    assert backend.update.await_args.args[1].use_meta_llm is False
    assert resp.json()["use_meta_llm"] is False


@pytest.mark.asyncio
async def test_omitted_enabled_keeps_it_disabled() -> None:
    """Already guarded before this change — pinned so it stays guarded."""
    container, backend = _container()
    async with await _client(container) as c:
        resp = await c.put("/api/v1/workflows/wf", json=_MINIMAL_PUT)
    assert resp.status_code == 200
    assert backend.update.await_args.args[1].enabled is False
    assert resp.json()["enabled"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["use_storage", "use_meta_llm", "enabled"])
async def test_explicitly_sent_false_still_turns_a_flag_off(field: str) -> None:
    """Preserving on omission must not make a flag impossible to clear."""
    container, backend = _container()
    stored = _stored()
    setattr(stored, field, True)
    backend.get = AsyncMock(return_value=stored)

    async with await _client(container) as c:
        resp = await c.put("/api/v1/workflows/wf", json={**_MINIMAL_PUT, field: False})
    assert resp.status_code == 200
    assert getattr(backend.update.await_args.args[1], field) is False
    assert resp.json()[field] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["use_storage", "use_meta_llm", "enabled"])
async def test_explicitly_sent_true_still_turns_a_flag_on(field: str) -> None:
    container, backend = _container()
    stored = _stored()
    setattr(stored, field, False)
    backend.get = AsyncMock(return_value=stored)

    async with await _client(container) as c:
        resp = await c.put("/api/v1/workflows/wf", json={**_MINIMAL_PUT, field: True})
    assert resp.status_code == 200
    assert getattr(backend.update.await_args.args[1], field) is True
    assert resp.json()[field] is True


@pytest.mark.asyncio
async def test_get_returns_the_flags_so_a_client_can_echo_them_back() -> None:
    """The read side has to expose the flags, or no client can round-trip them.

    The list endpoint is a summary and deliberately omits them; the single-
    workflow GET is what the editor loads from, so that one must carry them.
    """
    container, _ = _container()
    async with await _client(container) as c:
        resp = await c.get("/api/v1/workflows/wf")
    assert resp.status_code == 200
    body = resp.json()
    assert body["use_storage"] is True
    assert body["use_meta_llm"] is False
    assert body["enabled"] is False
