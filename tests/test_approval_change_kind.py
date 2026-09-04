"""The gate fires on what a call does, and says what it does.

Two faults seen in the Slack channel, from one gated Google Sheets append:

    Data deletion awaiting approval — 1 row
    Workflow: AFP Delivered -> Google Sheet (temp)
    Operation: Google Sheets · append_values

Nothing was going to be deleted, and it was ten rows, not one.
"""
from __future__ import annotations

import httpx
import pytest

from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    OperationDefinition,
)
from app.infrastructure.datasources.datastream import LocalDiskStreamStore
from app.infrastructure.datasources.destructive import is_destructive
from app.infrastructure.datasources.executor import (
    DataSourceExecutor,
    _change_kind_for,
    _rows_from_params,
)
from app.infrastructure.datasources.google_sheets import google_sheets_operations


def _sheets() -> DataSourceDefinition:
    return DataSourceDefinition(
        id="google-sheets", base_url="https://sheets.googleapis.com",
        operations=[OperationDefinition.model_validate(o)
                    for o in google_sheets_operations()],
    )


# ---------------------------------------------------------------------------
# what is gated
# ---------------------------------------------------------------------------

def test_appending_rows_is_not_gated():
    """It adds rows after the last one and removes nothing."""
    src = _sheets()
    op = src.get_operation("append_values")

    assert op.destructive is False, "recorded as a decision, not left unset"
    assert is_destructive(op, src) is False


def test_overwriting_writes_are_still_gated():
    src = _sheets()
    for name in ("update_values", "batch_update_values"):
        assert is_destructive(src.get_operation(name), src) is True, name


def test_reads_are_not_gated():
    src = _sheets()
    for name in ("get_metadata", "get_values", "batch_get_values"):
        assert is_destructive(src.get_operation(name), src) is False, name


# ---------------------------------------------------------------------------
# what the approver is told it is
# ---------------------------------------------------------------------------

def test_a_gated_write_is_called_a_write_not_a_deletion():
    src = _sheets()
    assert _change_kind_for(src.get_operation("update_values"), src) == "write"
    assert _change_kind_for(src.get_operation("batch_update_values"), src) == "write"


def test_a_delete_is_still_called_a_delete():
    op = OperationDefinition(name="drop", method="DELETE", path="/x/{params.id}")
    src = DataSourceDefinition(id="s", base_url="https://x", operations=[op])
    assert _change_kind_for(op, src) == "delete"


def test_a_gated_graphql_mutation_is_a_write():
    op = OperationDefinition(name="m", method="POST",
                             query="mutation { deleteMachine(id: 1) { ok } }")
    src = DataSourceDefinition(id="g", kind="graphql", base_url="https://g",
                               operations=[op])
    # It is gated (a mutation), and it reaches the gate as a write: POST is
    # not a removal verb, whatever the field happens to be called.
    assert is_destructive(op, src) is True
    assert _change_kind_for(op, src) == "write"


# ---------------------------------------------------------------------------
# how many rows
# ---------------------------------------------------------------------------

def test_the_row_count_comes_from_the_payload_array():
    """"1 row" for a ten-row write was the one number the message exists for."""
    op = _sheets().get_operation("update_values")
    rows, label = _rows_from_params(op, {
        "spreadsheet_id": "x", "range": "Sheet1!A1",
        "values": [["a"], ["b"], ["c"]],
    })

    assert rows == 3
    assert "values" in label


def test_an_ambiguous_payload_does_not_guess():
    """Two arrays, or none: fall back rather than pick one."""
    op = OperationDefinition(name="op", method="POST", path="/x", params=[
        {"name": "a", "type": "array", "required": False},
        {"name": "b", "type": "array", "required": False},
    ])
    assert _rows_from_params(op, {"a": [1, 2], "b": [3]}) == (1, "")
    assert _rows_from_params(op, {}) == (1, "")


def test_a_single_empty_array_still_reports_at_least_one():
    op = _sheets().get_operation("update_values")
    rows, _ = _rows_from_params(op, {"values": []})
    assert rows == 1


# ---------------------------------------------------------------------------
# end to end through preview
# ---------------------------------------------------------------------------

async def test_preview_of_a_sheets_overwrite_reads_as_a_write_of_n_rows(tmp_path):
    src = _sheets()
    store = LocalDiskStreamStore(tmp_path / "s")

    plan = await DataSourceExecutor(stream_store=store).preview(
        src, "update_values",
        {"spreadsheet_id": "abc", "range": "Sheet1!A1",
         "values": [["a"], ["b"], ["c"], ["d"]]},
    )

    assert plan.change_kind == "write"
    assert plan.affected_rows == 4
