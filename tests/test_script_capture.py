"""Inline ``python`` step bodies become library scripts when a workflow is saved.

Covers the shared capture helper, the REST save routes, and the management tool
cores the chat agent and the management MCP server both call.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.application import management_tools
from app.application.script_capture import capture_inline_scripts
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.persistence.workflow_backend import WorkflowDefinitionBackend
from app.infrastructure.tools.mcp_client import McpToolsProvider
from tests.test_scripts_api import InMemoryScriptBackend


class InMemoryWorkflowBackend(WorkflowDefinitionBackend):
    def __init__(self) -> None:
        self._store: dict[str, WorkflowDefinition] = {}

    async def list(self) -> list[WorkflowDefinition]:
        return list(self._store.values())

    async def get(self, workflow_id: str) -> WorkflowDefinition | None:
        return self._store.get(workflow_id)

    async def create(self, definition: WorkflowDefinition) -> WorkflowDefinition:
        self._store[definition.id] = definition
        return definition

    async def update(self, workflow_id: str, definition: WorkflowDefinition) -> WorkflowDefinition:
        self._store[workflow_id] = definition
        return definition

    async def delete(self, workflow_id: str) -> None:
        self._store.pop(workflow_id, None)


def _python_step(step_id: str = "transform", code: str = "output = 1", **extra) -> dict:
    # sandbox_runtime k8s keeps the step out of the admin-only class, so these
    # tests exercise capture rather than the sandbox guard.
    return {
        "id": step_id, "type": "python", "code": code,
        "sandbox_runtime": "k8s", **extra,
    }


# ─── The capture helper ───────────────────────────────────────────────────────

async def test_inline_code_becomes_a_library_script():
    backend = InMemoryScriptBackend()
    steps = [_python_step(name="Transform")]

    captured = await capture_inline_scripts("orders-wf", steps, backend)

    assert captured == ["orders-wf-transform"]
    # The step now points at the library; the inline copy stays as the fallback
    # body the UI shows, and script_id is what actually runs.
    assert steps[0]["script_id"] == "orders-wf-transform"
    assert steps[0]["code"] == "output = 1"
    script = await backend.get("orders-wf-transform")
    assert script.name == "Transform"
    assert script.code == "output = 1"


async def test_the_id_is_scoped_to_the_workflow_and_step():
    """Two workflows with a same-named step must not share one library entry."""
    backend = InMemoryScriptBackend()
    a = [_python_step("transform", "output = 'a'")]
    b = [_python_step("transform", "output = 'b'")]

    await capture_inline_scripts("wf-a", a, backend)
    await capture_inline_scripts("wf-b", b, backend)

    assert (await backend.get("wf-a-transform")).code == "output = 'a'"
    assert (await backend.get("wf-b-transform")).code == "output = 'b'"


async def test_resaving_updates_the_same_script():
    backend = InMemoryScriptBackend()
    steps = [_python_step(code="output = 1")]
    await capture_inline_scripts("wf", steps, backend)

    # Editing the node body and saving again must not create a second entry.
    edited = [_python_step(code="output = 2")]
    captured = await capture_inline_scripts("wf", edited, backend)

    assert captured == ["wf-transform"]
    assert len(await backend.list()) == 1
    assert (await backend.get("wf-transform")).code == "output = 2"


async def test_steps_already_referencing_a_script_are_left_alone():
    backend = InMemoryScriptBackend()
    steps = [_python_step(script_id="hand-picked", code="output = 1")]

    assert await capture_inline_scripts("wf", steps, backend) == []
    assert steps[0]["script_id"] == "hand-picked"
    assert await backend.list() == []


@pytest.mark.parametrize("steps", [
    [{"id": "s", "type": "python", "code": "   "}],       # nothing to save
    [{"id": "s", "type": "python"}],                      # no code at all
    [{"id": "s", "type": "llm", "code": "output = 1"}],   # not a python step
    [{"id": "", "type": "python", "code": "output = 1"}], # no id to scope on
])
async def test_nothing_is_captured_for(steps):
    backend = InMemoryScriptBackend()
    assert await capture_inline_scripts("wf", steps, backend) == []
    assert await backend.list() == []


async def test_without_a_script_backend_the_save_is_untouched():
    steps = [_python_step()]
    assert await capture_inline_scripts("wf", steps, None) == []
    assert "script_id" not in steps[0]


async def test_a_failing_library_does_not_break_the_save():
    """The step keeps running from its inline code when the library is down."""
    backend = InMemoryScriptBackend()
    backend.create = AsyncMock(side_effect=RuntimeError("mongo is gone"))
    steps = [_python_step()]

    assert await capture_inline_scripts("wf", steps, backend) == []
    assert "script_id" not in steps[0]
    assert steps[0]["code"] == "output = 1"


# ─── The REST save routes ─────────────────────────────────────────────────────

def _build_container(
    script_backend: InMemoryScriptBackend, workflow_backend: InMemoryWorkflowBackend
) -> ApplicationContainer:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    container = ApplicationContainer(
        settings=Settings(),
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=AsyncMock(spec=MongoGraphRunRepository),
        openhands=MagicMock(spec=OpenHandsAdapter),
        script_backend=script_backend,
        workflow_backend=workflow_backend,
    )
    # Registering a runner needs the full graph machinery; the save path only has
    # to reach the backend for these tests.
    container.refresh_runner = AsyncMock()
    return container


@pytest.fixture
async def client():
    scripts = InMemoryScriptBackend()
    workflows = InMemoryWorkflowBackend()
    app = create_app()
    app.state.container = _build_container(scripts, workflows)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, scripts, workflows


async def test_post_workflow_captures_inline_python(client):
    c, scripts, workflows = client

    resp = await c.post("/api/v1/workflows", json={
        "id": "orders-wf",
        "name": "Orders",
        "steps": [_python_step(name="Transform")],
    })

    assert resp.status_code == 201
    assert resp.json()["steps"][0]["script_id"] == "orders-wf-transform"
    assert (await scripts.get("orders-wf-transform")).code == "output = 1"
    # Persisted, not only echoed back.
    assert (await workflows.get("orders-wf")).steps[0]["script_id"] == "orders-wf-transform"


async def test_put_workflow_captures_inline_python(client):
    c, scripts, workflows = client
    await workflows.create(WorkflowDefinition(id="orders-wf", name="Orders", steps=[]))

    resp = await c.put("/api/v1/workflows/orders-wf", json={
        "name": "Orders",
        "steps": [_python_step(code="output = 2")],
    })

    assert resp.status_code == 200
    assert resp.json()["steps"][0]["script_id"] == "orders-wf-transform"
    assert (await scripts.get("orders-wf-transform")).code == "output = 2"


# ─── The management tool cores (chat agent + management MCP) ──────────────────

def _deps(scripts, workflows) -> management_tools.ManagementDeps:
    registry = MagicMock()
    registry.list_definitions.return_value = [{"id": "orders-wf", "name": "Orders"}]
    return management_tools.ManagementDeps(
        registry=registry,
        run_repository=AsyncMock(),
        workflow_backend=workflows,
        script_backend=scripts,
    )


async def test_agent_created_workflow_captures_inline_python():
    scripts, workflows = InMemoryScriptBackend(), InMemoryWorkflowBackend()

    result = await management_tools.create_workflow(
        _deps(scripts, workflows),
        "orders-wf", "Orders", "",
        '[{"id": "transform", "type": "python", "code": "output = 1", "sandbox_runtime": "k8s"}]',
    )

    assert "orders-wf-transform" in result
    assert (await scripts.get("orders-wf-transform")).code == "output = 1"
    assert (await workflows.get("orders-wf")).steps[0]["script_id"] == "orders-wf-transform"


async def test_agent_updated_workflow_captures_inline_python():
    scripts, workflows = InMemoryScriptBackend(), InMemoryWorkflowBackend()
    await workflows.create(WorkflowDefinition(id="orders-wf", name="Orders", steps=[]))

    result = await management_tools.update_workflow(
        _deps(scripts, workflows),
        "orders-wf",
        steps_json='[{"id": "transform", "type": "python", "code": "output = 2",'
                   ' "sandbox_runtime": "k8s"}]',
    )

    assert "orders-wf-transform" in result
    assert (await scripts.get("orders-wf-transform")).code == "output = 2"


async def test_update_without_steps_says_nothing_about_scripts():
    scripts, workflows = InMemoryScriptBackend(), InMemoryWorkflowBackend()
    await workflows.create(WorkflowDefinition(id="orders-wf", name="Orders", steps=[]))

    result = await management_tools.update_workflow(
        _deps(scripts, workflows), "orders-wf", name="Renamed",
    )

    assert result == "Workflow 'orders-wf' updated."
