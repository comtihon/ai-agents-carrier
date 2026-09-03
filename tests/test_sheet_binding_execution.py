"""A compiled binding is reachable the same way every other operation is.

That is the whole design claim: because a binding compiles into an
``OperationDefinition``, ``DataSourceExecutor.execute`` serves it, the approval
gate previews it, and the ``datasources`` MCP surface publishes it — with no
new step type and no new gate.  These tests exercise those three paths against
a stubbed HTTP layer so nothing reaches Google.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.api.mcp.datasources_server import build_datasources_mcp, rebuild_datasource_tools
from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.datasources.destructive import is_destructive
from app.infrastructure.datasources.executor import DataSourceExecutor
from app.infrastructure.datasources.sheet_binding_compile import (
    compile_binding,
    param_name_for,
)
from app.infrastructure.datasources.sheet_binding_library import (
    sheet_binding_library_scripts,
)
from app.domain.models.sheet_binding import SheetBinding
from app.infrastructure.datasources.google_sheets import google_sheets_operations
from tests.test_datasources_api import InMemoryDataSourceBackend
from tests.test_sheet_bindings_api import GRID, _read_binding, _write_binding


def _executor(**kwargs):
    """An executor with a throw-away stream store.

    Every data source result is written to a stream and returned as a
    reference, so an executor needs somewhere to write. Tests that assert on
    records call ``execute_value``, which reads the stream back.
    """
    import tempfile

    from app.infrastructure.datasources.datastream import LocalDiskStreamStore

    kwargs.setdefault("stream_store", LocalDiskStreamStore(tempfile.mkdtemp()))
    return DataSourceExecutor(**kwargs)


class _RecordingExecutor(DataSourceExecutor):
    """A real executor with only the raw Sheets operations stubbed out.

    Subclassing rather than faking, so the binding interception really is the
    one in ``DataSourceExecutor.execute`` and not a test-only shortcut.
    """

    def __init__(self) -> None:
        super().__init__()
        self.grid = GRID
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_value(self, source, operation, params, *, max_bytes=0):
        # The binding runtime asks for values; this double answers with
        # the canned responses instead of going near a stream store.
        return await self.execute(source, operation, params)

    async def execute(self, source, operation, params):
        if source.get_binding(operation) is None:
            self.calls.append((operation, params))
            if operation == "get_values":
                return {"range": params["range"], "values": self.grid}
            if operation in ("batch_update_values", "append_values"):
                return {"totalUpdatedCells": 2}
            raise AssertionError(f"unexpected raw operation {operation}")
        return await super().execute(source, operation, params)


def _source(*bindings: dict) -> DataSourceDefinition:
    models = [SheetBinding.model_validate(b) for b in bindings]
    return DataSourceDefinition.model_validate({
        "id": "google-sheets",
        "name": "Google Sheets",
        "description": "Read and write Google Sheets.",
        "base_url": "https://sheets.googleapis.com",
        "auth": {"type": "google", "scopes": []},
        "operations": [
            *google_sheets_operations(),
            *[compile_binding(b).model_dump() for b in models],
        ],
        "bindings": [b.model_dump(mode="json") for b in models],
    })


# ─── execute() ────────────────────────────────────────────────────────────────

async def test_a_read_binding_is_executed_by_its_operation_name():
    source = _source(_read_binding())
    executor = _RecordingExecutor()

    result = await executor.execute_value(source, "read_open_projects", {"assignee": "ann"})

    assert result == [{"project_id": "P-1", "status": "open", "owner": "ann"}]
    # One read, and it carried the header row so the fingerprint could be
    # checked against the very grid the records came from.
    assert [name for name, _ in executor.calls] == ["get_values"]
    assert executor.calls[0][1]["range"] == "Projects!A1:Z"


async def test_a_write_binding_runs_the_five_steps_in_order():
    source = _source(_write_binding())
    executor = _RecordingExecutor()

    result = await executor.execute_value(source, "update_project", {
        "project_id": "P-1", "classification_status": "closed",
    })

    assert result["status"] == "ok"
    assert result["row_number"] == 2
    assert result["cells_written"] == 2
    # read, then write — the row number was resolved inside this one execution.
    assert [name for name, _ in executor.calls] == ["get_values", "batch_update_values"]
    written = executor.calls[1][1]
    assert written["data"] == [
        {"range": "Projects!B2", "values": [["closed"]]},
        {"range": "Projects!E2", "values": [["Reviewed automatically"]]},
    ]
    assert written["value_input_option"] == "RAW"


async def test_a_write_stops_at_the_fingerprint_before_touching_anything():
    source = _source(_write_binding())
    executor = _RecordingExecutor()
    executor.grid = [["id", "project_id", "status"], ["x", "P-1", "open"]]

    from app.infrastructure.datasources.sheet_binding_resolver import SheetBindingError
    with pytest.raises(SheetBindingError, match="header row has changed"):
        await executor.execute_value(source, "update_project", {
            "project_id": "P-1", "classification_status": "closed",
        })

    # It read, and then it stopped. No write, and no fallback to position.
    assert [name for name, _ in executor.calls] == ["get_values"]


async def test_a_missing_param_is_named_rather_than_silently_blank():
    source = _source(_write_binding())
    executor = _RecordingExecutor()

    from app.infrastructure.datasources.sheet_binding_runtime import BindingRuntimeError
    with pytest.raises(BindingRuntimeError, match="classification_status"):
        await executor.execute_value(source, "update_project", {"project_id": "P-1"})

    assert executor.calls == []


# ─── The approval gate ────────────────────────────────────────────────────────

async def test_a_write_binding_is_destructive_and_previews_as_cells():
    source = _source(_write_binding())
    executor = _RecordingExecutor()
    op = source.get_operation("update_project")

    # The gate's own question.
    assert is_destructive(op, source) is True

    plan = await executor.preview(source, "update_project", {
        "project_id": "P-1", "classification_status": "closed",
    })

    assert plan.affected_rows == 2
    assert plan.targets == ["RC Projects Tracker · Projects row 2"]
    # An approver reads cells, not a batchUpdate body.
    assert plan.sample == [
        "Projects!B2 (status): 'open' → 'closed'",
        "Projects!E2 (notes): 'first note' → 'Reviewed automatically'",
    ]
    # A preview never writes.
    assert [name for name, _ in executor.calls] == ["get_values"]


async def test_a_read_binding_is_not_gated():
    source = _source(_read_binding())

    assert is_destructive(source.get_operation("read_open_projects"), source) is False


async def test_a_write_whose_columns_all_skip_affects_no_rows():
    """Nothing for a person to approve, so the gate lets it through."""
    binding = _write_binding(
        blank_policy="skip",
        columns={"status": {"from": "state.classification.status"}},
    )
    source = _source(binding)
    executor = _RecordingExecutor()

    plan = await executor.preview(source, "update_project", {
        "project_id": "P-1", "classification_status": "",
    })

    assert plan.affected_rows == 0


# ─── The MCP tool surface ─────────────────────────────────────────────────────

class _Container:
    def __init__(self, backend) -> None:
        self.data_source_backend = backend
        self.data_source_executor = _executor()


async def test_a_compiled_binding_is_published_as_an_mcp_tool():
    backend = InMemoryDataSourceBackend()
    await backend.create(_source(_read_binding(), _write_binding()))
    mcp = build_datasources_mcp()

    await rebuild_datasource_tools(mcp, backend, lambda: _Container(backend))
    tools = {t.name: t for t in await mcp.list_tools()}

    read = tools["ds_google-sheets_read_open_projects"]
    # The param is the state path the binding references, and it is the only one.
    assert set(read.inputSchema["properties"]) == {"assignee"}
    assert "[GET · READ]" in read.description
    assert "Read columns project_id, status, owner" in read.description

    write = tools["ds_google-sheets_update_project"]
    assert set(write.inputSchema["properties"]) == {"project_id", "classification_status"}
    # The marker a model reads before deciding how careful to be.
    assert "[POST · WRITE]" in write.description
    assert "left untouched" in write.description


# ─── Param naming ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("state.assignee", "assignee"),
    ("state.project.id", "project_id"),
    ("state.classification.status", "classification_status"),
    ("$.state.qc.rtk_label", "qc_rtk_label"),
    ("state.items[0].id", "items_0_id"),
])
def test_state_paths_flatten_to_usable_param_names(path, expected):
    assert param_name_for(path) == expected


# ─── The script library ───────────────────────────────────────────────────────

async def test_the_resolver_entries_run_in_the_sandbox():
    """The padding rule has to work where a workflow step would call it.

    The sandbox has no site-packages and no network, which is why the resolver
    module is stdlib-only and shipped as source rather than imported.
    """
    from app.infrastructure.orchestration.script_sandbox import run_script

    scripts = {s["id"]: s for s in sheet_binding_library_scripts()}

    result = await run_script(
        scripts["sheets-rows-to-records"]["code"],
        {"values": [["P-1", "open"], ["P-2"]], "headers": ["project_id", "status"]},
        runtime="local",
        timeout=30,
    )

    assert result == [
        {"project_id": "P-1", "status": "open"},
        {"project_id": "P-2", "status": ""},
    ]


async def test_the_fingerprint_entry_raises_in_the_sandbox_on_drift():
    from app.infrastructure.orchestration.script_sandbox import run_script

    scripts = {s["id"]: s for s in sheet_binding_library_scripts()}

    with pytest.raises(Exception, match="header row has changed"):
        await run_script(
            scripts["sheets-check-fingerprint"]["code"],
            {"headers": ["a", "b"], "fingerprint": "sha256:deadbeef"},
            runtime="local",
            timeout=30,
        )
