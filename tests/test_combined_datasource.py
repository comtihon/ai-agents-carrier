"""combined_datasource: ordered parts, cross-part refs, list -> loop, nesting, validation."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.domain.models.data_source_definition import DataSourceDefinition
from app.domain.models.datastream import as_data_ref
from app.domain.models.graph_run import GraphRun
from app.infrastructure.auth.sandbox_guard import find_admin_only_python_steps
from app.infrastructure.datasources.datastream import LocalDiskStreamStore
from app.infrastructure.orchestration.combined_datasource import (
    combined_step_errors,
    validate_combined_steps,
)
from app.infrastructure.orchestration.step_outputs import StepOutputStore, is_output_ref
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
from app.infrastructure.tools.mcp_client import McpToolsProvider

PROJECTS = [{"id": i, "status": 10, "open": i % 2 == 0} for i in range(1, 11)]

_SOURCES = {
    "afp": DataSourceDefinition.model_validate({
        "id": "afp", "base_url": "https://afp.test",
        "operations": [
            {"name": "list_projects", "path": "/projects",
             "params": [{"name": "status", "type": "array"}, {"name": "kind", "type": "string"}]},
            {"name": "get_classification", "path": "/projects/{id}/classification",
             "params": [{"name": "id", "type": "number", "required": True}]},
            {"name": "bulk_classification", "path": "/classifications",
             "params": [{"name": "ids", "type": "array"}]},
            {"name": "delete_project", "path": "/projects/{id}", "method": "DELETE",
             "params": [{"name": "id", "type": "number"}]},
        ],
    }),
}


class _Backend:
    async def get(self, source_id):
        return _SOURCES.get(source_id)


class _Executor:
    """Records every call and writes results to the real store, like the executor."""

    _fanout_concurrency = 3

    def __init__(self, store) -> None:
        self.store = store
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, source, operation, params, *, limit=None):
        self.calls.append((operation, dict(params)))
        if operation == "list_projects":
            rows = [p for p in PROJECTS if p["status"] in params.get("status", [])]
            if params.get("kind"):
                rows = [r for r in rows if r["open"]]
            shape, data = "list", rows
        elif operation == "get_classification":
            if params["id"] == 7:
                raise RuntimeError("404 not found")
            shape, data = "value", [{"name": f"Paket {'s' if params['id'] < 5 else 'm'}", "price": 100}]
        elif operation == "bulk_classification":
            shape, data = "list", [{"id": i} for i in params["ids"]]
        else:
            raise AssertionError(operation)
        writer = await self.store.open_writer(source_id=source.id, operation=operation, shape=shape)
        await writer.append_many(data)
        return (await writer.close()).to_state()


class _Scripts:
    def __init__(self, scripts: dict[str, str]) -> None:
        self.scripts = scripts

    async def get(self, script_id):
        code = self.scripts.get(script_id)
        return None if code is None else MagicMock(code=code)


class _Repo:
    def __init__(self) -> None:
        self.runs: dict[str, GraphRun] = {}

    async def update(self, run):
        self.runs[run.id] = run.model_copy(deep=True)

    async def get(self, run_id):
        return self.runs.get(run_id)


@pytest.fixture
def store(tmp_path):
    return LocalDiskStreamStore(tmp_path / "s")


def _runner(steps, store, *, scripts=None):
    mcp = MagicMock(spec=McpToolsProvider)
    runner = YamlGraphRunner({"id": "g", "steps": steps}, llm=MagicMock(), mcp_tools_provider=mcp)
    runner._stream_store = store
    runner._step_output_store = StepOutputStore(store)
    runner._data_source_backend = _Backend()
    runner._data_source_executor = _Executor(store)
    runner._script_backend = _Scripts(scripts or {})
    return runner


async def _go(runner, run_id="r1"):
    repo = _Repo()
    run = GraphRun(id=run_id, graph_id="g", status="running",
                   step_statuses={s["id"]: "pending" for s in runner.steps})
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})
    return repo.runs[run_id]


def _py(pid, code, **kw):
    return {"id": pid, "type": "python", "sandbox": False, "code": code, **kw}


def _step(parts, **kw):
    return {"id": "cands", "type": "combined_datasource", "output_key": "cands", "parts": parts, **kw}


async def _value(runner, value):
    ref = as_data_ref(value)
    if ref is not None:
        return await runner._stream_store.read_all(ref, max_bytes=0)
    if is_output_ref(value):
        return await runner._step_output_store.resolve(value)
    return value


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def test_parts_run_in_order_with_cross_part_refs_and_a_loop(store):
    parts = [
        {"id": "fetch", "type": "data_source", "source": "afp", "operation": "list_projects",
         "params": {"status": [10]}},
        _py("sel", "output = {'ids': [r['id'] for r in records() if r['open']]}", stream="fetch"),
        {"id": "cls", "type": "data_source", "source": "afp", "operation": "get_classification",
         "params": {"id": "{sel[ids]}"}},
    ]
    runner = _runner([_step(parts)], store)
    run = await _go(runner)
    calls = runner._data_source_executor.calls
    # array-typed `status` is passed whole (bulk); `id` is looped per element.
    assert calls[0] == ("list_projects", {"status": [10]})
    assert [c[1]["id"] for c in calls[1:]] == [2, 4, 6, 8, 10]
    out = await _value(runner, run.state["cands"])
    assert [r["id"] for r in out] == [2, 4, 6, 8, 10]
    assert out[0]["result"] == {"name": "Paket s", "price": 100}
    trace = run.step_outputs["cands"]["_combined_cands"]
    assert [p["id"] for p in trace["parts"]] == ["fetch", "sel", "cls"]
    assert all(p["status"] == "finished" for p in trace["parts"])
    assert trace["parts"][0]["records"] == 10 and trace["parts"][2]["calls"] == 5
    assert trace["output_part"] == "cls"
    assert run.step_statuses["cands"] == "finished"


async def test_loop_collects_per_element_errors_without_failing(store):
    parts = [
        _py("ids", "output = [6, 7, 8]"),
        {"id": "cls", "type": "data_source", "source": "afp", "operation": "get_classification",
         "params": {"id": "{ids}"}},
    ]
    runner = _runner([_step(parts)], store)
    run = await _go(runner)
    out = await _value(runner, run.state["cands"])
    assert [r["id"] for r in out] == [6, 8]
    cls = run.step_outputs["cands"]["_combined_cands"]["parts"][1]
    assert cls["error_count"] == 1 and "404" in cls["errors"][0]


async def test_fail_on_error_fails_the_step_with_trace(store):
    parts = [
        _py("ids", "output = [6, 7]"),
        {"id": "cls", "type": "data_source", "source": "afp", "operation": "get_classification",
         "params": {"id": "{ids}"}, "fail_on_error": True},
        _py("after", "output = 1"),
    ]
    runner = _runner([_step(parts)], store)
    run = await _go(runner)
    assert run.step_statuses["cands"] == "failed"
    trace = run.step_outputs["cands"]["_combined_cands"]["parts"]
    assert [p["status"] for p in trace] == ["finished", "failed", "skipped"]


async def test_array_param_is_called_once_in_bulk(store):
    parts = [
        _py("ids", "output = {'ids': [1, 2, 3]}"),
        {"id": "bulk", "type": "data_source", "source": "afp", "operation": "bulk_classification",
         "params": {"ids": "{ids[ids]}"}},
    ]
    runner = _runner([_step(parts)], store)
    await _go(runner)
    assert runner._data_source_executor.calls == [("bulk_classification", {"ids": [1, 2, 3]})]


async def test_python_part_uses_several_outputs_and_order_matters(store):
    parts = [
        {"id": "all", "type": "data_source", "source": "afp", "operation": "list_projects",
         "params": {"status": [10]}},
        {"id": "open", "type": "data_source", "source": "afp", "operation": "list_projects",
         "params": {"status": [10], "kind": "open"}},
        _py("a", "output = {'ids': [r['id'] for r in parts['all'] if r['id'] < 5]}"),
        _py("b", "output = {'ids': [r['id'] for r in parts['open']]}"),
        _py("cand", "output = {'project_ids': sorted(set(parts['a']['ids']) | set(parts['b']['ids']))}"),
    ]
    runner = _runner([_step(parts)], store)
    run = await _go(runner)
    assert await _value(runner, run.state["cands"]) == {"project_ids": [1, 2, 3, 4, 6, 8, 10]}
    # Intermediate outputs live in the store, not in the run document.
    trace = run.step_outputs["cands"]["_combined_cands"]["parts"]
    assert trace[0]["output"]["__stream__"] == 1
    data = await store.get_object("runs/r1/cands.cand.md")
    assert b"project_ids" in data


async def test_nested_combined_and_explicit_output(store):
    parts = [
        _py("seed", "output = {'ids': [2, 3]}"),
        {"id": "inner", "type": "combined_datasource", "parts": [
            {"id": "cls", "type": "data_source", "source": "afp", "operation": "get_classification",
             "params": {"id": "{seed[ids]}"}},
            _py("names", "output = [r['result']['name'] for r in parts['cls']]"),
        ]},
        _py("unused", "output = 'last'"),
    ]
    runner = _runner([_step(parts, output="inner")], store)
    run = await _go(runner)
    assert await _value(runner, run.state["cands"]) == ["Paket s", "Paket s"]
    inner = run.step_outputs["cands"]["_combined_cands"]["parts"][1]
    assert [p["id"] for p in inner["parts"]] == ["cls", "names"]
    assert inner["output_part"] == "names"


async def test_script_library_part(store):
    parts = [
        {"id": "fetch", "type": "data_source", "source": "afp", "operation": "list_projects",
         "params": {"status": [10]}},
        {"id": "sel", "type": "python", "script_id": "count-open", "sandbox": False, "stream": "fetch"},
    ]
    runner = _runner([_step(parts)], store,
                     scripts={"count-open": "output = sum(1 for r in records() if r['open'])"})
    run = await _go(runner)
    assert run.state["cands"] == 5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_structural_validation_errors():
    steps = [_step([
        {"id": "a", "type": "data_source", "source": "afp", "params": {"x": "{c[ids]}"}},
        {"id": "b", "type": "python", "code": "output = parts['zzz']", "stream": "c"},
        {"id": "c", "type": "python"},
        {"id": "c", "type": "weird"},
        {"id": "n", "type": "combined_datasource", "parts": []},
    ], output="nope")]
    errors = "\n".join(validate_combined_steps(steps))
    assert "missing `operation`" in errors
    assert "forward reference" in errors
    assert "`stream: c` names a later part" in errors
    assert "parts['zzz'], which is no earlier part" in errors
    assert "needs `script_id` or `code`" in errors
    assert "duplicate part id 'c'" in errors
    assert "unknown part type 'weird'" in errors
    assert "non-empty list" in errors
    assert "`output: nope` names no part" in errors


def test_nesting_cycle_is_reported():
    parts: list = [{"id": "x", "type": "combined_datasource"}]
    parts[0]["parts"] = parts
    assert any("cycle" in e for e in validate_combined_steps([_step(parts)]))


async def test_reference_validation_against_backends():
    steps = [_step([
        {"id": "a", "type": "data_source", "source": "missing", "operation": "x"},
        {"id": "b", "type": "data_source", "source": "afp", "operation": "nope"},
        {"id": "c", "type": "data_source", "source": "afp", "operation": "delete_project"},
        {"id": "d", "type": "python", "script_id": "ghost"},
    ])]
    errors = "\n".join(await combined_step_errors(
        steps, data_source_backend=_Backend(), script_backend=_Scripts({})))
    assert "data source 'missing' not found" in errors
    assert "has no operation 'nope'" in errors
    assert "destructive" in errors
    assert "script 'ghost' not found" in errors


def test_unsandboxed_python_part_needs_admin():
    steps = [_step([{"id": "p", "type": "python", "code": "output=1", "sandbox": False}])]
    assert "p" in find_admin_only_python_steps(steps)


async def test_mcp_create_workflow_rejects_invalid_combined(monkeypatch):
    from app.application import management_tools as mt
    from app.infrastructure.auth.authorization import Permission

    monkeypatch.setattr(
        "app.infrastructure.auth.authorization.get_current_permissions",
        lambda: frozenset(Permission),
    )
    deps = MagicMock()
    deps.data_source_backend = _Backend()
    deps.script_backend = _Scripts({})
    import json
    out = await mt.create_workflow(
        deps, "wf", "wf", "d",
        json.dumps([_step([{"id": "a", "type": "data_source", "source": "afp", "operation": "nope"}])]),
    )
    assert out.startswith("Invalid combined_datasource step") and "no operation 'nope'" in out
