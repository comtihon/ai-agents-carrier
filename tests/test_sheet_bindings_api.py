"""Sheet bindings over the REST API: probe, CRUD, compilation, preview.

Google is mocked throughout — a fake executor answers the ``google-sheets``
source's raw operations with canned grids, so no test here reaches a real API.
The fake also records every operation it was asked for, which is how the
"a preview never writes" tests can prove a negative.
"""
from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.core.config import Settings
from app.domain.models.data_source_definition import DataSourceDefinition
from app.domain.models.sheet_binding import header_fingerprint
from app.infrastructure.datasources.google_sheets import (
    DEFAULT_VALUE_INPUT_OPTION,
    google_sheets_operations,
)
from tests.test_datasources_api import InMemoryDataSourceBackend, _build_container

HEADERS = ["project_id", "status", "owner", "rtk_flag", "notes"]
FINGERPRINT = header_fingerprint(HEADERS)

# Row P-2 is deliberately ragged: the Sheets API omits trailing empty cells.
GRID = [
    HEADERS,
    ["P-1", "open", "ann", "yes", "first note"],
    ["P-2", "closed"],
    ["P-3", "open", "bob", "", "third note"],
]

METADATA = {
    "sheets": [
        {"properties": {"sheetId": 1234567, "title": "Projects"}},
        {"properties": {"sheetId": 99, "title": "Archive"}},
    ],
    "namedRanges": [{"name": "monthly_total", "range": {"sheetId": 1234567}}],
}


class FakeSheetsExecutor:
    """Answers the raw Sheets operations; records what it was asked to do."""

    def __init__(self, grid: list[list[Any]] | None = None) -> None:
        self.grid = GRID if grid is None else grid
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def operations_called(self) -> list[str]:
        return [name for name, _ in self.calls]

    async def execute(self, source, operation: str, params: dict[str, Any]):
        self.calls.append((operation, params))
        if operation == "get_metadata":
            return METADATA
        if operation == "get_values":
            return {"range": params["range"], "values": self.grid}
        if operation == "batch_get_values":
            return {
                "valueRanges": [
                    {"range": r, "values": [["before"]]} for r in params["ranges"]
                ]
            }
        if operation in ("batch_update_values", "append_values"):
            return {"totalUpdatedCells": 1}
        raise AssertionError(f"unexpected operation {operation}")


def _sheets_source() -> DataSourceDefinition:
    return DataSourceDefinition.model_validate({
        "id": "google-sheets",
        "name": "Google Sheets",
        "base_url": "https://sheets.googleapis.com",
        "auth": {"type": "google", "scopes": ["https://www.googleapis.com/auth/spreadsheets"]},
        "operations": google_sheets_operations(),
    })


# The impersonated principal.  A `google` auth block may name only this one,
# and the probe / preview routes re-check that against the *stored* definition
# before minting a token, so the backend has to be configured for these tests
# even though no real token is ever minted.
GOOGLE_SA = "copilot@engineering-368717.iam.gserviceaccount.com"


@pytest.fixture
async def client():
    backend = InMemoryDataSourceBackend()
    await backend.create(_sheets_source())
    app = create_app()
    container = _build_container(backend)
    container.settings = Settings(GOOGLE_IMPERSONATE_SA=GOOGLE_SA)
    executor = FakeSheetsExecutor()
    container.data_source_executor = executor
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, backend, executor


def _read_binding(**overrides) -> dict:
    binding = {
        "version": 1,
        "name": "read_open_projects",
        "document": {
            "provider": "google_sheets",
            "file_id": "1AbC",
            "name": "RC Projects Tracker",
            "sheet": "Projects",
            "sheet_id": 1234567,
        },
        "schema": {"header_row": 1, "headers": HEADERS, "fingerprint": FINGERPRINT},
        "operation": "read",
        "read": {
            "mode": "rows",
            "columns": ["project_id", "status", "owner"],
            "filter": {
                "op": "and",
                "clauses": [
                    {"column": "status", "op": "eq", "value": {"literal": "open"}},
                    {"column": "owner", "op": "eq", "value": {"from": "state.assignee"}},
                ],
            },
            "limit": 200,
        },
        "output": {"key": "projects"},
    }
    binding.update(overrides)
    return binding


def _write_binding(**write_overrides) -> dict:
    write = {
        "mode": "update_by_key",
        "key_column": "project_id",
        "key_value": {"from": "state.project.id"},
        "on_missing": "error",
        "value_input_option": "RAW",
        "blank_policy": "skip",
        "columns": {
            "status": {"from": "state.classification.status"},
            "notes": {"literal": "Reviewed automatically"},
        },
    }
    write.update(write_overrides)
    return {
        "version": 1,
        "name": "update_project",
        "document": {
            "provider": "google_sheets",
            "file_id": "1AbC",
            "name": "RC Projects Tracker",
            "sheet": "Projects",
            "sheet_id": 1234567,
        },
        "schema": {"header_row": 1, "headers": HEADERS, "fingerprint": FINGERPRINT},
        "operation": "write",
        "write": write,
    }


# ─── Probe ────────────────────────────────────────────────────────────────────

async def test_probe_sheet_drives_the_whole_form(client):
    """One call returns every dropdown the editor needs, plus real rows."""
    c, _, executor = client

    resp = await c.post(
        "/api/v1/datasources/google/probe-sheet",
        json={"file_id": "1AbC", "sheet": "Projects"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["tabs"] == [
        {"sheet_id": 1234567, "title": "Projects"},
        {"sheet_id": 99, "title": "Archive"},
    ]
    assert [n["name"] for n in data["named_ranges"]] == ["monthly_total"]
    assert data["sheet"] == "Projects"
    # The numeric id is what a binding stores; the title is display only.
    assert data["sheet_id"] == 1234567
    assert data["headers"] == HEADERS
    assert data["fingerprint"] == FINGERPRINT
    # Sample rows are padded to the header width — a ragged table is unreadable.
    assert all(len(row) == len(HEADERS) for row in data["sample_rows"])
    assert data["sample_rows"][1] == ["P-2", "closed", "", "", ""]
    assert executor.operations_called == ["get_metadata", "get_values"]


async def test_probe_sheet_defaults_to_the_first_tab(client):
    c, _, _ = client

    data = (await c.post(
        "/api/v1/datasources/google/probe-sheet", json={"file_id": "1AbC"}
    )).json()

    assert data["sheet"] == "Projects"


async def test_probe_sheet_names_the_tabs_when_the_one_asked_for_is_gone(client):
    c, _, _ = client

    data = (await c.post(
        "/api/v1/datasources/google/probe-sheet",
        json={"file_id": "1AbC", "sheet": "Renamed"},
    )).json()

    assert data["status"] == "error"
    assert "Archive, Projects" in data["error"]


async def test_probe_sheet_404s_when_the_source_is_not_there(client):
    c, backend, _ = client
    await backend.delete("google-sheets")

    resp = await c.post("/api/v1/datasources/google/probe-sheet", json={"file_id": "1AbC"})

    assert resp.status_code == 404


# ─── CRUD + compilation ───────────────────────────────────────────────────────

async def test_saving_a_read_binding_compiles_it_into_an_operation(client):
    c, backend, _ = client

    resp = await c.post(
        "/api/v1/datasources/google-sheets/bindings", json={"binding": _read_binding()}
    )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    # The wire shape keeps the documented key names.
    assert body["schema"]["fingerprint"] == FINGERPRINT
    assert body["read"]["filter"]["clauses"][1]["value"] == {"from": "state.assignee"}
    assert body["resolution"]["tier"] == "binding"
    assert body["resolution"]["authored_by"] == "human"
    assert body["resolution"]["compiled_at"] is not None
    # No natural-language path exists, so nothing claims one did.
    assert body["resolution"]["instruction"] is None

    stored = await backend.get("google-sheets")
    op = stored.get_operation("read_open_projects")
    assert op is not None
    assert op.method == "GET"
    # Params are exactly the {"from": …} state paths the binding references.
    assert [p.name for p in op.params] == ["assignee"]
    assert op.destructive is None
    assert "Read columns project_id, status, owner" in op.description


async def test_saving_a_write_binding_compiles_a_destructive_operation(client):
    c, backend, _ = client

    resp = await c.post(
        "/api/v1/datasources/google-sheets/bindings", json={"binding": _write_binding()}
    )

    assert resp.status_code == 201, resp.text
    stored = await backend.get("google-sheets")
    op = stored.get_operation("update_project")
    # destructive: true is what puts it behind the existing approval gate — the
    # verb alone would not, because only DELETE gates by default.
    assert op.destructive is True
    assert set(p.name for p in op.params) == {"project_id", "classification_status"}
    assert "left untouched" in op.description


async def test_append_row_carries_a_single_retry_attempt(client):
    c, backend, _ = client
    binding = _write_binding(
        mode="append_row",
        columns={"project_id": {"from": "state.project.id"}, "notes": {"literal": "new"}},
    )
    binding["write"].pop("key_column")
    binding["write"].pop("key_value")
    binding["write"].pop("on_missing")

    resp = await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    assert resp.status_code == 201, resp.text
    op = (await backend.get("google-sheets")).get_operation("update_project")
    # An append is not idempotent: a retry after a timeout that in fact reached
    # Sheets appends the row twice.
    assert op.retries is not None and op.retries.attempts == 1


async def test_list_update_delete_roundtrip(client):
    c, backend, _ = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _read_binding()})

    listed = (await c.get("/api/v1/datasources/google-sheets/bindings")).json()
    assert [b["name"] for b in listed] == ["read_open_projects"]
    assert listed[0]["compiled_operation"] == "read_open_projects"

    changed = _read_binding()
    changed["read"]["columns"] = ["project_id", "notes"]
    resp = await c.put(
        "/api/v1/datasources/google-sheets/bindings/read_open_projects",
        json={"binding": changed},
    )
    assert resp.status_code == 200
    op = (await backend.get("google-sheets")).get_operation("read_open_projects")
    assert "columns project_id, notes" in op.description

    resp = await c.delete("/api/v1/datasources/google-sheets/bindings/read_open_projects")
    assert resp.status_code == 204
    stored = await backend.get("google-sheets")
    assert stored.bindings == []
    # The operation goes with the binding that was the only reason it existed…
    assert stored.get_operation("read_open_projects") is None
    # …and the raw operations are untouched.
    assert {op.name for op in stored.operations} == {
        o["name"] for o in google_sheets_operations()
    }


async def test_renaming_a_binding_takes_its_operation_with_it(client):
    c, backend, _ = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _read_binding()})

    renamed = _read_binding(name="read_projects")
    resp = await c.put(
        "/api/v1/datasources/google-sheets/bindings/read_open_projects",
        json={"binding": renamed},
    )

    assert resp.status_code == 200
    stored = await backend.get("google-sheets")
    assert stored.get_operation("read_open_projects") is None
    assert stored.get_operation("read_projects") is not None


async def test_a_put_on_the_source_does_not_strip_compiled_operations(client):
    """Replacing ``operations`` wholesale must not orphan the bindings."""
    c, backend, _ = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _read_binding()})

    resp = await c.put(
        "/api/v1/datasources/google-sheets",
        json={"operations": [o for o in google_sheets_operations()]},
    )

    assert resp.status_code == 200
    stored = await backend.get("google-sheets")
    assert stored.get_operation("read_open_projects") is not None


# ─── Validation ───────────────────────────────────────────────────────────────

async def test_an_unknown_column_is_rejected_by_name(client):
    c, _, _ = client
    binding = _read_binding()
    binding["read"]["columns"] = ["project_id", "assignee"]

    resp = await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "unknown column 'assignee' in read.columns" in detail
    # The message lists what the sheet does have, so the fix is obvious.
    assert "'owner'" in detail


async def test_an_unknown_filter_column_names_its_clause(client):
    c, _, _ = client
    binding = _read_binding()
    binding["read"]["filter"]["clauses"][0]["column"] = "state"

    resp = await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    assert resp.status_code == 422
    assert "read.filter.clauses[0].column" in resp.json()["detail"]


async def test_an_unknown_write_key_column_is_rejected(client):
    c, _, _ = client
    binding = _write_binding(key_column="ticket_id")

    resp = await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    assert resp.status_code == 422
    assert "unknown column 'ticket_id' in write.key_column" in resp.json()["detail"]


@pytest.mark.parametrize("name", ["Read Projects", "read-projects", "1read", "params", ""])
async def test_an_unusable_binding_name_is_rejected(client, name):
    c, _, _ = client

    resp = await c.post(
        "/api/v1/datasources/google-sheets/bindings", json={"binding": _read_binding(name=name)}
    )

    assert resp.status_code == 422


async def test_a_binding_name_cannot_shadow_a_raw_operation(client):
    c, _, _ = client

    resp = await c.post(
        "/api/v1/datasources/google-sheets/bindings",
        json={"binding": _read_binding(name="get_values")},
    )

    assert resp.status_code == 409
    assert "already an operation" in resp.json()["detail"]


async def test_a_duplicate_binding_name_is_a_conflict(client):
    c, _, _ = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _read_binding()})

    resp = await c.post(
        "/api/v1/datasources/google-sheets/bindings", json={"binding": _read_binding()}
    )

    assert resp.status_code == 409


async def test_rows_mode_requires_a_column_projection(client):
    """Forty columns by five hundred rows must not be the default."""
    c, _, _ = client
    binding = _read_binding()
    binding["read"]["columns"] = []

    resp = await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    assert resp.status_code == 422
    assert "read.columns is required" in resp.json()["detail"]


async def test_update_by_key_requires_its_key(client):
    c, _, _ = client
    binding = _write_binding()
    binding["write"].pop("key_value")

    resp = await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    assert resp.status_code == 422
    assert "key_value is required" in resp.json()["detail"]


async def test_a_value_carrying_both_tags_is_refused(client):
    c, _, _ = client
    binding = _write_binding(columns={"status": {"literal": "open", "from": "state.x"}})

    resp = await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    assert resp.status_code == 422


async def test_a_missing_fingerprint_is_filled_from_the_headers(client):
    """A binding with no drift protection at all must not be storable."""
    c, backend, _ = client
    binding = _read_binding()
    binding["schema"].pop("fingerprint")

    resp = await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    assert resp.status_code == 201
    assert resp.json()["schema"]["fingerprint"] == FINGERPRINT


# ─── Preview ──────────────────────────────────────────────────────────────────

async def test_previewing_a_read_returns_the_matching_rows(client):
    c, _, executor = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _read_binding()})

    resp = await c.post(
        "/api/v1/datasources/google-sheets/bindings/read_open_projects/preview",
        json={"state": {"assignee": "bob"}},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["output"] == [{"project_id": "P-3", "status": "open", "owner": "bob"}]
    assert data["params"] == {"assignee": "bob"}
    assert executor.operations_called == ["get_values"]


async def test_previewing_a_write_shows_the_target_range_and_before_after(client):
    c, _, executor = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _write_binding()})

    resp = await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-1"}, "classification": {"status": "closed"}}},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["row_number"] == 2
    assert data["range"] == "Projects!B2, Projects!E2"
    assert data["value_input_option"] == "RAW"
    by_column = {cell["column"]: cell for cell in data["cells"]}
    assert by_column["status"]["before"] == "open"
    assert by_column["status"]["after"] == "closed"
    assert by_column["notes"]["before"] == "first note"
    assert by_column["notes"]["after"] == "Reviewed automatically"
    # An approver reads cells, not JSON.
    assert data["changes"] == [
        "Projects!B2 (status): 'open' → 'closed'",
        "Projects!E2 (notes): 'first note' → 'Reviewed automatically'",
    ]


async def test_a_preview_never_reaches_a_write_operation(client):
    """The tool is READ-tier, and this is what makes that honest."""
    c, _, executor = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _write_binding()})

    await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-1"}, "classification": {"status": "closed"}}},
    )

    assert executor.operations_called == ["get_values"]
    assert not any(
        name in ("batch_update_values", "append_values", "update_values")
        for name in executor.operations_called
    )


async def test_a_column_absent_from_the_map_is_absent_from_the_preview(client):
    """The safety property, end to end."""
    c, _, _ = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _write_binding()})

    data = (await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-1"}, "classification": {"status": "closed"}}},
    )).json()

    touched = {cell["column"] for cell in data["cells"]}
    assert touched == {"status", "notes"}
    # owner and rtk_flag are named nowhere in the binding, so nothing in the
    # plan mentions them — no range, no cell, no "unchanged" entry.
    assert "owner" not in touched and "rtk_flag" not in touched
    assert all("C2" not in (cell["a1"] or "") for cell in data["cells"])


async def test_blank_policy_skip_shows_the_cell_as_left_alone(client):
    c, _, _ = client
    binding = _write_binding(
        blank_policy="skip",
        columns={"status": {"from": "state.classification.status"}},
    )
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    data = (await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-1"}, "classification": {"status": ""}}},
    )).json()

    assert data["cells"][0]["action"] == "skip"
    assert data["cells"][0]["before"] == data["cells"][0]["after"] == "open"
    assert data["changes"] == ["Projects!B2 (status): left unchanged ('open')"]


async def test_blank_policy_clear_writes_an_empty_cell(client):
    c, _, _ = client
    binding = _write_binding(
        blank_policy="clear",
        columns={"status": {"from": "state.classification.status"}},
    )
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    data = (await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-1"}, "classification": {"status": None}}},
    )).json()

    assert data["cells"][0]["action"] == "write"
    assert data["cells"][0]["after"] == ""


async def test_on_missing_error_says_which_key_was_not_found(client):
    c, _, _ = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _write_binding()})

    data = (await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-99"}, "classification": {"status": "closed"}}},
    )).json()

    assert data["status"] == "error"
    assert "P-99" in data["error"] and "on_missing is 'error'" in data["error"]


async def test_on_missing_skip_reports_that_nothing_would_change(client):
    c, _, _ = client
    binding = _write_binding(on_missing="skip")
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    data = (await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-99"}, "classification": {"status": "closed"}}},
    )).json()

    assert data["status"] == "skipped"
    assert data["cells"] == []


async def test_on_missing_append_plans_a_new_row(client):
    c, _, _ = client
    binding = _write_binding(
        on_missing="append",
        columns={
            "project_id": {"from": "state.project.id"},
            "status": {"from": "state.classification.status"},
        },
    )
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    data = (await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-99"}, "classification": {"status": "open"}}},
    )).json()

    assert data["status"] == "ready"
    assert data["mode"] == "append_row"
    assert data["row_number"] is None
    assert {cell["column"] for cell in data["cells"]} == {"project_id", "status"}
    # Nothing existed there, so every "before" is empty by definition.
    assert all(cell["before"] == "" for cell in data["cells"])


async def test_on_missing_append_needs_the_key_column_in_the_map(client):
    """Appending a row with a blank key column would be unfindable afterwards."""
    c, _, _ = client
    binding = _write_binding(on_missing="append")

    resp = await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    assert resp.status_code == 422
    assert "no value in the key column" in resp.json()["detail"]


async def test_a_drifted_header_row_fails_the_preview_loudly(client):
    """The binding is saved against one header row and run against another."""
    c, _, executor = client
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": _write_binding()})
    # Somebody inserted a column in front of the table.
    executor.grid = [["id", *HEADERS], ["x", "P-1", "open", "ann", "yes", "note"]]

    data = (await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-1"}, "classification": {"status": "closed"}}},
    )).json()

    assert data["status"] == "error"
    assert "header row has changed" in data["error"]
    assert data["cells"] == []
    # Nothing was written, and nothing fell back to writing by position.
    assert executor.operations_called == ["get_values"]


async def test_user_entered_prefix_guards_a_formula_in_the_preview(client):
    c, _, _ = client
    binding = _write_binding(
        value_input_option="USER_ENTERED",
        columns={"notes": {"from": "state.note"}},
    )
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    data = (await c.post(
        "/api/v1/datasources/google-sheets/bindings/update_project/preview",
        json={"state": {"project": {"id": "P-1"}, "note": '=IMPORTRANGE("evil","A1")'}},
    )).json()

    assert data["cells"][0]["after"] == '\'=IMPORTRANGE("evil","A1")'


async def test_raw_is_the_default_value_input_option(client):
    """And it is the same default the raw Sheets operations carry."""
    c, backend, _ = client
    binding = _write_binding()
    binding["write"].pop("value_input_option")
    await c.post("/api/v1/datasources/google-sheets/bindings", json={"binding": binding})

    stored = await backend.get("google-sheets")
    assert stored.bindings[0].write.value_input_option == "RAW"
    assert DEFAULT_VALUE_INPUT_OPTION == "RAW"


async def test_preview_404s_for_an_unknown_binding(client):
    c, _, _ = client

    resp = await c.post(
        "/api/v1/datasources/google-sheets/bindings/nope/preview", json={"state": {}}
    )

    assert resp.status_code == 404
