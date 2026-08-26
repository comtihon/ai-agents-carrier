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


# ─── The management tool core (chat agent + management MCP) ───────────────────
#
# The surface-parity test compares the MCP tool against the chat-agent tool, so
# a parameter missing from BOTH looks perfectly consistent. `use_meta_llm` was
# missing from both for exactly that reason: readable via get_workflow, but
# impossible to set from either surface. These tests assert behaviour instead.

from tests.test_script_capture import InMemoryWorkflowBackend  # noqa: E402
from tests.test_scripts_api import InMemoryScriptBackend  # noqa: E402


def _deps(scripts, workflows):
    from app.application import management_tools

    registry = MagicMock()
    registry.list_definitions.return_value = []
    return management_tools.ManagementDeps(
        registry=registry,
        run_repository=AsyncMock(),
        workflow_backend=workflows,
        script_backend=scripts,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, (True, False)),                                   # field defaults
        ({"use_meta_llm": False}, (False, False)),
        ({"use_storage": True}, (True, True)),
        ({"use_meta_llm": False, "use_storage": True}, (False, True)),
    ],
)
async def test_core_create_honours_both_flags(kwargs, expected) -> None:
    from app.application import management_tools

    scripts, workflows = InMemoryScriptBackend(), InMemoryWorkflowBackend()
    await management_tools.create_workflow(
        _deps(scripts, workflows), "wf", "WF", "", "[]", **kwargs
    )
    stored = await workflows.get("wf")
    assert (stored.use_meta_llm, stored.use_storage) == expected


@pytest.mark.asyncio
async def test_core_update_can_turn_meta_llm_off_without_touching_steps() -> None:
    """The exact request this was added for: flip one flag, disturb nothing else."""
    from app.application import management_tools

    scripts, workflows = InMemoryScriptBackend(), InMemoryWorkflowBackend()
    await workflows.create(
        WorkflowDefinition(
            id="wf", name="WF", steps=_STEPS, use_meta_llm=True, use_storage=True
        )
    )

    await management_tools.update_workflow(
        _deps(scripts, workflows), "wf", use_meta_llm=False
    )

    stored = await workflows.get("wf")
    assert stored.use_meta_llm is False
    assert stored.use_storage is True   # untouched
    assert stored.steps == _STEPS       # untouched
    assert stored.name == "WF"          # untouched


@pytest.mark.asyncio
async def test_core_update_omitting_a_flag_keeps_it() -> None:
    from app.application import management_tools

    scripts, workflows = InMemoryScriptBackend(), InMemoryWorkflowBackend()
    await workflows.create(
        WorkflowDefinition(
            id="wf", name="WF", steps=_STEPS, use_meta_llm=False, use_storage=True
        )
    )

    await management_tools.update_workflow(
        _deps(scripts, workflows), "wf", name="Renamed"
    )

    stored = await workflows.get("wf")
    assert stored.name == "Renamed"
    assert stored.use_meta_llm is False   # not reset to the True default
    assert stored.use_storage is True
