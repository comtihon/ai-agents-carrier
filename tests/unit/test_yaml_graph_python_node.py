"""Tests for the `python` step type: inline code, library scripts, sandboxing."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.domain.models.script_definition import ScriptDefinition
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.tools.mcp_client import McpToolsProvider


def _make_runner(steps: list[dict], script_backend=None) -> YamlGraphRunner:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="x")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner({"id": "test-graph", "steps": steps}, llm=llm, mcp_tools_provider=mcp)
    if script_backend is not None:
        runner._script_backend = script_backend
    return runner


async def test_inline_code_runs_sandboxed_by_default():
    runner = _make_runner([
        {"id": "calc", "type": "python", "code": 'output = state["request"].upper()',
         "output_key": "result"},
    ])
    state = await runner.graph.ainvoke({"request": "hi"}, {"configurable": {"thread_id": "t1"}})
    assert state["result"] == "HI"


async def test_sandboxed_step_cannot_see_backend_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")
    runner = _make_runner([
        {"id": "peek", "type": "python", "code": "import os\noutput = dict(os.environ)",
         "output_key": "env"},
    ])
    state = await runner.graph.ainvoke({"request": "x"}, {"configurable": {"thread_id": "t2"}})
    assert state["env"] == {}


async def test_unsandboxed_step_runs_in_process(monkeypatch):
    monkeypatch.setenv("SANDBOX_PROBE", "visible")
    runner = _make_runner([
        {"id": "peek", "type": "python", "sandbox": False,
         "code": 'import os\noutput = os.environ.get("SANDBOX_PROBE")',
         "output_key": "probe"},
    ])
    state = await runner.graph.ainvoke({"request": "x"}, {"configurable": {"thread_id": "t3"}})
    assert state["probe"] == "visible"


async def test_script_id_loads_code_from_the_library():
    backend = MagicMock()
    backend.get = AsyncMock(return_value=ScriptDefinition(
        id="doubler", name="Doubler", code='output = state["request"] * 2',
    ))
    runner = _make_runner(
        [{"id": "run-script", "type": "python", "script_id": "doubler", "output_key": "result"}],
        script_backend=backend,
    )
    state = await runner.graph.ainvoke({"request": "ab"}, {"configurable": {"thread_id": "t4"}})
    assert state["result"] == "abab"
    backend.get.assert_awaited_once_with("doubler")


async def test_script_id_wins_over_stale_inline_code():
    backend = MagicMock()
    backend.get = AsyncMock(return_value=ScriptDefinition(id="s", name="S", code='output = "library"'))
    runner = _make_runner(
        [{"id": "s1", "type": "python", "script_id": "s", "code": 'output = "inline"',
          "output_key": "result"}],
        script_backend=backend,
    )
    state = await runner.graph.ainvoke({"request": "x"}, {"configurable": {"thread_id": "t5"}})
    assert state["result"] == "library"


async def test_missing_script_is_reported_under_output_key():
    backend = MagicMock()
    backend.get = AsyncMock(return_value=None)
    runner = _make_runner(
        [{"id": "s1", "type": "python", "script_id": "gone", "output_key": "result"}],
        script_backend=backend,
    )
    state = await runner.graph.ainvoke({"request": "x"}, {"configurable": {"thread_id": "t6"}})
    assert "not found" in state["result"]["error"]


async def test_script_id_without_backend_is_reported():
    runner = _make_runner(
        [{"id": "s1", "type": "python", "script_id": "gone", "output_key": "result"}],
    )
    state = await runner.graph.ainvoke({"request": "x"}, {"configurable": {"thread_id": "t7"}})
    assert "no script backend is configured" in state["result"]["error"]


@pytest.mark.parametrize("runtime", ["docker", "k8s"])
async def test_sandbox_runtime_is_passed_through(runtime, monkeypatch):
    captured: dict = {}

    async def _fake_run_script(code, state, **kwargs):
        captured.update(kwargs, code=code)
        return "ok"

    monkeypatch.setattr(
        "app.infrastructure.orchestration.script_sandbox.run_script", _fake_run_script,
    )
    runner = _make_runner([
        {"id": "s1", "type": "python", "code": "output = 1", "sandbox_runtime": runtime,
         "timeout_seconds": 12, "sandbox_image": "python:3.11-slim", "output_key": "result"},
    ])
    state = await runner.graph.ainvoke({"request": "x"}, {"configurable": {"thread_id": "t8"}})
    assert state["result"] == "ok"
    assert captured["runtime"] == runtime
    assert captured["timeout"] == 12
    assert captured["image"] == "python:3.11-slim"
