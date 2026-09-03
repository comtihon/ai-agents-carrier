"""Sheet bindings on the management MCP surface.

The point of these is parity: an agent authoring a binding must be held to the
same validation as a person authoring one in the form, and must not be able to
reach a write through a tool classified READ.  Google is mocked — a fake
executor answers the raw Sheets operations.
"""
from __future__ import annotations

import json

import pytest

from app.application import management_tools as core
from app.application.management_tools import ManagementDeps
from app.infrastructure.auth.authorization import (
    Permission,
    reset_current_permissions,
    set_current_permissions,
)
from tests.test_datasources_api import InMemoryDataSourceBackend
from tests.test_sheet_bindings_api import (
    GOOGLE_SA,
    FakeSheetsExecutor,
    _read_binding,
    _sheets_source,
    _write_binding,
)


@pytest.fixture(autouse=True)
def _all_permissions():
    token = set_current_permissions({Permission.READ, Permission.WRITE, Permission.DELETE})
    yield
    reset_current_permissions(token)


@pytest.fixture(autouse=True)
def _google_configured(monkeypatch):
    """A `google` auth block may name only the configured principal."""
    from app.core.config import Settings, get_settings
    from app.infrastructure.auth import google_token_provider

    settings = Settings(GOOGLE_IMPERSONATE_SA=GOOGLE_SA)
    get_settings.cache_clear()
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    # google_token_provider imported get_settings by name, so the module's own
    # reference has to be replaced as well as the one in app.core.config.
    monkeypatch.setattr(google_token_provider, "get_settings", lambda: settings)
    yield
    get_settings.cache_clear()


@pytest.fixture
def executor(monkeypatch):
    """Route every binding tool through one recording fake executor."""
    fake = FakeSheetsExecutor()
    monkeypatch.setattr(core, "_binding_executor", lambda deps: fake)
    return fake


@pytest.fixture
async def deps():
    backend = InMemoryDataSourceBackend()
    await backend.create(_sheets_source())
    return ManagementDeps(
        registry=None,  # type: ignore[arg-type]
        run_repository=None,  # type: ignore[arg-type]
        data_source_backend=backend,
    )


async def test_probe_reports_tabs_headers_and_the_fingerprint(deps, executor):
    result = await core.probe_google_sheet(deps, "1AbC", "Projects")

    assert "Tab: Projects (sheet_id 1234567)" in result
    assert "Headers: project_id, status, owner, rtk_flag, notes" in result
    assert "Fingerprint: sha256:" in result
    # The sample rows are what makes an agent's binding correspond to reality.
    assert '["P-2", "closed", "", "", ""]' in result


async def test_save_compiles_a_binding_into_an_operation(deps, executor):
    result = await core.save_sheet_binding(deps, json.dumps(_read_binding()))

    assert "compiled into operation 'read_open_projects'" in result
    assert "params: assignee" in result
    stored = await deps.data_source_backend.get("google-sheets")
    assert stored.get_operation("read_open_projects") is not None


async def test_saving_a_write_says_it_goes_through_the_approval_gate(deps, executor):
    result = await core.save_sheet_binding(deps, json.dumps(_write_binding()))

    assert "approval gate" in result
    stored = await deps.data_source_backend.get("google-sheets")
    assert stored.get_operation("update_project").destructive is True


async def test_the_mcp_path_rejects_an_unknown_column_too(deps, executor):
    binding = _read_binding()
    binding["read"]["columns"] = ["project_id", "assignee"]

    result = await core.save_sheet_binding(deps, json.dumps(binding))

    assert "unknown column 'assignee'" in result
    assert (await deps.data_source_backend.get("google-sheets")).bindings == []


async def test_the_mcp_path_will_not_shadow_a_raw_operation(deps, executor):
    result = await core.save_sheet_binding(
        deps, json.dumps(_read_binding(name="get_values"))
    )

    assert "already an operation" in result


async def test_save_is_idempotent_by_name(deps, executor):
    await core.save_sheet_binding(deps, json.dumps(_read_binding()))

    result = await core.save_sheet_binding(deps, json.dumps(_read_binding()))

    assert "replaced" in result
    stored = await deps.data_source_backend.get("google-sheets")
    assert len(stored.bindings) == 1


async def test_provenance_is_not_caller_supplied(deps, executor):
    binding = _read_binding()
    binding["resolution"] = {
        "tier": "binding",
        "instruction": "read the open ones",
        "authored_by": "llm",
        "model_id": "some-model",
    }

    await core.save_sheet_binding(deps, json.dumps(binding))

    stored = (await deps.data_source_backend.get("google-sheets")).bindings[0]
    assert stored.resolution.authored_by == "human"
    assert stored.resolution.instruction is None
    assert stored.resolution.model_id is None


async def test_list_and_get_round_trip(deps, executor):
    await core.save_sheet_binding(deps, json.dumps(_read_binding()))

    listed = await core.list_sheet_bindings(deps)
    assert "**read_open_projects** (read, mode rows)" in listed

    fetched = json.loads(await core.get_sheet_binding(deps, "read_open_projects"))
    assert fetched["schema"]["headers"] == _read_binding()["schema"]["headers"]
    # A get returns exactly what save accepts.
    assert "replaced" in await core.save_sheet_binding(deps, json.dumps(fetched))


async def test_delete_removes_the_binding_and_its_operation(deps, executor):
    await core.save_sheet_binding(deps, json.dumps(_read_binding()))

    result = await core.delete_sheet_binding(deps, "read_open_projects")

    assert "deleted" in result
    stored = await deps.data_source_backend.get("google-sheets")
    assert stored.bindings == []
    assert stored.get_operation("read_open_projects") is None


async def test_preview_reports_cells_and_writes_nothing(deps, executor):
    await core.save_sheet_binding(deps, json.dumps(_write_binding()))

    result = await core.preview_sheet_binding(
        deps,
        "update_project",
        json.dumps({"project": {"id": "P-1"}, "classification": {"status": "closed"}}),
    )

    assert "row 2" in result
    assert "Projects!B2 (status): 'open' → 'closed'" in result
    assert "Columns not listed above are not touched." in result
    # READ-tier, and honestly so: no write operation was reached.
    assert executor.operations_called.count("get_values") == 1
    assert "batch_update_values" not in executor.operations_called


async def test_preview_surfaces_header_drift(deps, executor):
    await core.save_sheet_binding(deps, json.dumps(_write_binding()))
    executor.grid = [["id", "project_id", "status"], ["x", "P-1", "open"]]

    result = await core.preview_sheet_binding(
        deps,
        "update_project",
        json.dumps({"project": {"id": "P-1"}, "classification": {"status": "closed"}}),
    )

    assert "header row has changed" in result
    assert "batch_update_values" not in executor.operations_called


async def test_a_reader_cannot_save_a_binding(deps, executor):
    token = set_current_permissions({Permission.READ})
    try:
        result = await core.save_sheet_binding(deps, json.dumps(_read_binding()))
    finally:
        reset_current_permissions(token)

    assert "Not permitted" in result
    assert (await deps.data_source_backend.get("google-sheets")).bindings == []


async def test_a_writer_cannot_delete_a_binding(deps, executor):
    await core.save_sheet_binding(deps, json.dumps(_read_binding()))
    token = set_current_permissions({Permission.READ, Permission.WRITE})
    try:
        result = await core.delete_sheet_binding(deps, "read_open_projects")
    finally:
        reset_current_permissions(token)

    assert "Not permitted" in result
    assert len((await deps.data_source_backend.get("google-sheets")).bindings) == 1
