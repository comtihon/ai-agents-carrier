"""Canvas positions on the management-API path, and only on that path.

A definition arriving through the MCP tools carries no coordinates, so the
editor fell back to its six-column snake -- which reads the step array, not the
graph. Writing the layout at create/update time is what closes that gap.

The other half of this is the boundary: the REST routes are what the canvas
itself saves through, and a person who dragged their nodes into place there did
so on purpose. Those handlers store `ui` verbatim, and these tests pin that they
still do.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application import management_tools
from app.domain.models.workflow_definition import WorkflowDefinition
from tests.test_script_capture import InMemoryWorkflowBackend
from tests.test_scripts_api import InMemoryScriptBackend

pytestmark = pytest.mark.asyncio

_FAN_OUT = (
    '[{"id": "trigger", "type": "cron", "next": "fan"},'
    ' {"id": "fan", "type": "parallel", "targets": ["a", "b"]},'
    ' {"id": "a", "type": "python", "code": "output = 1",'
    '  "sandbox_runtime": "k8s", "next": "gather"},'
    ' {"id": "b", "type": "python", "code": "output = 2",'
    '  "sandbox_runtime": "k8s", "next": "gather"},'
    ' {"id": "gather", "type": "join", "next": "END"}]'
)


def _deps(scripts=None, workflows=None):
    registry = MagicMock()
    registry.list_definitions.return_value = []
    return management_tools.ManagementDeps(
        registry=registry,
        run_repository=AsyncMock(),
        workflow_backend=workflows if workflows is not None else InMemoryWorkflowBackend(),
        script_backend=scripts if scripts is not None else InMemoryScriptBackend(),
    )


async def test_create_stores_a_position_for_every_step():
    workflows = InMemoryWorkflowBackend()
    await management_tools.create_workflow(
        _deps(workflows=workflows), "wf", "WF", "", _FAN_OUT
    )
    nodes = (await workflows.get("wf")).ui["nodes"]
    assert set(nodes) == {"trigger", "fan", "a", "b", "gather"}


async def test_created_positions_follow_the_graph_not_the_array():
    """The branches share a column and the join sits between them.

    Under the snake fallback these five steps came out on one row in array
    order, which draws a fan-out as a sequence.
    """
    workflows = InMemoryWorkflowBackend()
    await management_tools.create_workflow(
        _deps(workflows=workflows), "wf", "WF", "", _FAN_OUT
    )
    nodes = (await workflows.get("wf")).ui["nodes"]

    assert nodes["trigger"]["x"] < nodes["fan"]["x"] < nodes["a"]["x"]
    assert nodes["a"]["x"] == nodes["b"]["x"] < nodes["gather"]["x"]
    assert nodes["a"]["y"] != nodes["b"]["y"]
    assert min(nodes["a"]["y"], nodes["b"]["y"]) < nodes["gather"]["y"] \
        < max(nodes["a"]["y"], nodes["b"]["y"])


async def test_a_workflow_with_no_steps_stores_no_layout():
    workflows = InMemoryWorkflowBackend()
    await management_tools.create_workflow(
        _deps(workflows=workflows), "wf", "WF", "", "[]"
    )
    assert "nodes" not in ((await workflows.get("wf")).ui or {})


async def test_update_places_the_added_step_and_leaves_the_others_alone():
    workflows = InMemoryWorkflowBackend()
    arranged = {"nodes": {"one": {"x": 900, "y": 700}, "two": {"x": 1180, "y": 700}}}
    await workflows.create(WorkflowDefinition(
        id="wf", name="WF",
        steps=[{"id": "one", "type": "llm", "next": "two"},
               {"id": "two", "type": "llm", "next": "END"}],
        ui=arranged,
    ))

    await management_tools.update_workflow(
        _deps(workflows=workflows), "wf",
        steps_json='[{"id": "one", "type": "llm", "next": "two"},'
                   ' {"id": "two", "type": "llm", "next": "three"},'
                   ' {"id": "three", "type": "llm", "next": "END"}]',
    )

    nodes = (await workflows.get("wf")).ui["nodes"]
    assert nodes["one"] == {"x": 900, "y": 700}
    assert nodes["two"] == {"x": 1180, "y": 700}
    assert "three" in nodes


async def test_relayout_recomputes_the_whole_arrangement():
    workflows = InMemoryWorkflowBackend()
    await workflows.create(WorkflowDefinition(
        id="wf", name="WF",
        steps=[{"id": "one", "type": "llm", "next": "two"},
               {"id": "two", "type": "llm", "next": "END"}],
        ui={"nodes": {"one": {"x": 900, "y": 700}, "two": {"x": 1180, "y": 700}}},
    ))

    await management_tools.update_workflow(
        _deps(workflows=workflows), "wf", relayout=True
    )

    nodes = (await workflows.get("wf")).ui["nodes"]
    assert nodes["one"] == {"x": 40, "y": 40}
    assert nodes["two"]["x"] > nodes["one"]["x"]
    assert nodes["two"]["y"] == nodes["one"]["y"]


async def test_an_update_that_touches_no_steps_touches_no_positions():
    workflows = InMemoryWorkflowBackend()
    arranged = {"nodes": {"one": {"x": 900, "y": 700}}}
    await workflows.create(WorkflowDefinition(
        id="wf", name="WF", steps=[{"id": "one", "type": "llm", "next": "END"}],
        ui=arranged,
    ))

    await management_tools.update_workflow(
        _deps(workflows=workflows), "wf", name="Renamed"
    )

    assert (await workflows.get("wf")).ui == arranged


async def test_layout_survives_a_round_trip_through_update():
    """Two updates in a row must not drift the positions the first one chose."""
    workflows = InMemoryWorkflowBackend()
    deps = _deps(workflows=workflows)
    await management_tools.create_workflow(deps, "wf", "WF", "", _FAN_OUT)
    first = dict((await workflows.get("wf")).ui["nodes"])

    await management_tools.update_workflow(deps, "wf", steps_json=_FAN_OUT)

    assert (await workflows.get("wf")).ui["nodes"] == first


async def test_the_mcp_update_tool_exposes_relayout():
    from app.api.mcp.management_server import build_management_mcp, register_management_tools

    class _Container:
        def __init__(self) -> None:
            self.yaml_graph_registry = MagicMock()
            self.yaml_graph_registry.list_definitions.return_value = []
            self.run_repository = AsyncMock()
            self.workflow_backend = None
            self.agent_backend = None
            self.data_source_backend = None
            self.event_backend = None
            self.refresh_runner = None
            self.settings = MagicMock()
            self.live_runners: dict = {}

    server = build_management_mcp()
    register_management_tools(server, _Container)
    schemas = {t.name: t.inputSchema for t in await server.list_tools()}
    assert "relayout" in schemas["update_workflow"]["properties"]


async def test_the_rest_routes_store_ui_verbatim():
    """The canvas's own save path must not be reformatted.

    Positions arriving over REST came from the editor, where a person may have
    arranged them by hand; the layout pass belongs to the path that has no
    positions at all.
    """
    import inspect

    from app.api.routes import workflows as routes

    for handler in (routes.create_workflow, routes.update_workflow):
        source = inspect.getsource(handler)
        assert "ui=body.ui" in source
        assert "apply_layout" not in source
