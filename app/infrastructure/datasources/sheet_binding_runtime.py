"""Running a sheet binding: the Sheets calls, in the right order.

This is the only place that combines the two halves — the raw Sheets
operations of the ``google-sheets`` data source (which hold the impersonated
credential) and the pure transforms in ``sheet_binding_resolver`` (which hold
the logic).  Nothing here decides anything: it reads, hands the grid to the
pure functions, and writes back what they composed.

Run-time order for a write, which is the correctness core of the whole feature
-----------------------------------------------------------------------------
1. **read** the table (header row included) with ``get_values``;
2. **resolve the row** whose key column matches, from that same response;
3. **check the fingerprint** of the header row in that same response;
4. **build** the write from the binding's column map;
5. **call** ``batch_update_values`` / ``append_values``.

All five in one execution, always.  A row number is never carried across steps
or cached: a person inserting a row above it — which they may do at any moment,
in a document they own — silently makes it point at somebody else's row, and a
write that lands on the wrong row is exactly the failure this design exists to
prevent.  The fingerprint check sits inside the same window for the same
reason: checking it against a header row read earlier would prove nothing about
the grid being written to now.
"""
from __future__ import annotations

import logging
from typing import Any

from app.domain.models.sheet_binding import (
    COMPUTE_PATH_PREFIX,
    SheetBinding,
    header_fingerprint,
)
from app.infrastructure.datasources.sheet_binding_compile import (
    LAST_PROBE_COLUMN,
    binding_key_range,
    binding_param_map,
    binding_read_range,
)
from app.infrastructure.datasources.sheet_binding_resolver import (
    SheetBindingError,
    sheets_a1_range,
    sheets_build_write,
    sheets_check_fingerprint,
    sheets_column_letter,
    sheets_filter_records,
    sheets_project_records,
    sheets_resolve_row,
    sheets_resolve_tagged,
    sheets_rows_to_records,
)

logger = logging.getLogger(__name__)

# Raw operations of the google-sheets source this module drives.  Named rather
# than inlined so a source missing one fails with "the source has no operation
# 'get_values'" instead of a KeyError somewhere deeper.
OP_GET_METADATA = "get_metadata"
OP_GET_VALUES = "get_values"
OP_BATCH_GET_VALUES = "batch_get_values"
OP_BATCH_UPDATE_VALUES = "batch_update_values"
OP_APPEND_VALUES = "append_values"

# Rows of sample data a probe returns under the header row.  Enough for the
# author to recognise their own sheet, few enough that the response stays small.
PROBE_SAMPLE_ROWS = 5

# Cap on the cell-by-cell change list a preview / approval case carries, so one
# binding cannot flood the approval panel.
_PREVIEW_CELL_CAP = 50


class BindingRuntimeError(SheetBindingError):
    """A binding could not be run against the source as configured."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def values_by_state_path(binding: SheetBinding, params: dict[str, Any]) -> dict[str, Any]:
    """Re-key the caller's params by the state paths the binding names.

    The compiled operation declares one param per referenced state path (see
    ``sheet_binding_compile.binding_params``); the pure resolvers look values up
    by path.  This is the single translation between the two, so the two cannot
    drift.
    """
    resolved: dict[str, Any] = {}
    missing: list[str] = []
    for name, path in binding_param_map(binding).items():
        if name in params:
            resolved[path] = params[name]
        elif path in params:
            # Tolerated so a caller that already speaks in paths (the preview
            # endpoint, a test) does not need the slug mapping.
            resolved[path] = params[path]
        else:
            missing.append(f"{name} (= {path})")
    if missing:
        raise BindingRuntimeError(
            f"binding '{binding.name}' needs value(s) for {', '.join(missing)}"
        )
    return resolved


def params_from_state(binding: SheetBinding, state: dict[str, Any]) -> dict[str, Any]:
    """Build the compiled operation's params out of a (sample) state dict.

    Only used by the preview endpoint and the MCP preview tool, where a person
    or an agent supplies a state snapshot rather than the param names.  A path
    that is absent resolves to ``None`` rather than raising, because previewing
    an incomplete state is a legitimate thing to do — the preview then shows a
    blank value, which is itself informative under ``blank_policy``.
    """
    params: dict[str, Any] = {}
    for name, path in binding_param_map(binding).items():
        params[name] = _traverse(state, path)
    return params


def _traverse(state: dict[str, Any], path: str) -> Any:
    """Walk a dotted state path.  ``state.a.b`` and ``a.b`` both work."""
    parts = [p for p in path.replace("$.", "").split(".") if p]
    if parts and parts[0] == "state":
        parts = parts[1:]
    current: Any = state
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def resolved_columns(
    binding: SheetBinding, by_path: dict[str, Any]
) -> dict[str, Any]:
    """``column -> value`` (or ``range -> value`` for ``set_cells``).

    A column absent from ``write.columns`` is absent here too — the invariant
    that makes a write leave the rest of the row alone starts at this dict.
    """
    write = binding.write
    if write is None:
        return {}
    if write.mode == "set_cells":
        return {
            entry.range.render(): sheets_resolve_tagged(
                entry.value.model_dump(), by_path
            )
            for entry in write.cells
        }
    return {
        column: sheets_resolve_tagged(value.model_dump(), by_path)
        for column, value in write.columns.items()
    }


# ---------------------------------------------------------------------------
# Talking to the source
# ---------------------------------------------------------------------------

async def _call(source: Any, executor: Any, operation: str, params: dict[str, Any]) -> Any:
    if source.get_operation(operation) is None:
        raise BindingRuntimeError(
            f"data source '{source.id}' has no '{operation}' operation — a "
            "binding needs the Google Sheets operations on the same source"
        )
    # A binding's raw calls are internal steps with pure transforms between
    # them, so this code needs the response in hand rather than a stream
    # reference -- they are reads and writes of one range, bounded by
    # construction. `execute_value` is asked for by name when the executor
    # offers it, and anything executor-shaped that only implements `execute`
    # (a test double, an older adapter) still works.
    execute_value = getattr(executor, "execute_value", None)
    if execute_value is not None:
        return await execute_value(source, operation, params)
    return await executor.execute(source, operation, params)


async def _read_grid(
    source: Any, executor: Any, binding: SheetBinding, a1_range: str
) -> tuple[list[list[Any]], dict[str, Any]]:
    """One ``get_values`` call, returning ``(values, raw response)``."""
    read = binding.read
    payload = await _call(source, executor, OP_GET_VALUES, {
        "spreadsheet_id": binding.document.file_id,
        "range": a1_range,
        # A binding feeds a program, so unformatted values by default — see
        # ReadSpec.value_render.  A write's key lookup uses the same rendering
        # so the key it compares against is the one a read would have returned.
        "value_render_option": read.value_render if read else "UNFORMATTED_VALUE",
        "date_time_render_option": read.date_render if read else "FORMATTED_STRING",
    })
    if not isinstance(payload, dict):
        raise BindingRuntimeError(
            f"unexpected response reading {a1_range}: {type(payload).__name__}"
        )
    return list(payload.get("values") or []), payload


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

async def probe_sheet(
    source: Any,
    executor: Any,
    file_id: str,
    sheet: str | None = None,
    header_row: int = 1,
) -> dict[str, Any]:
    """Everything the binding editor's form needs, in one call.

    Two Sheets calls: ``get_metadata`` for the tab list and the named ranges,
    and ``get_values`` over ``A<header_row>:Z<header_row+5>`` of the chosen tab
    for the headers plus a few real rows.

    The sample rows are the whole UX of the editor: the author sees their own
    column names and their own data, so no screen has to explain what a
    "column" is.  They are also why the probe reads six rows and not one — a
    header row alone is not recognisable as the right tab.
    """
    metadata = await _call(source, executor, OP_GET_METADATA, {"spreadsheet_id": file_id})
    if not isinstance(metadata, dict):
        raise BindingRuntimeError("unexpected response from get_metadata")

    tabs = [
        {
            "sheet_id": (s.get("properties") or {}).get("sheetId"),
            "title": (s.get("properties") or {}).get("title") or "",
        }
        for s in (metadata.get("sheets") or [])
    ]
    named_ranges = [
        {
            "name": nr.get("name") or "",
            "range": nr.get("range") or {},
        }
        for nr in (metadata.get("namedRanges") or [])
    ]

    title = sheet or (tabs[0]["title"] if tabs else "")
    if not title:
        raise BindingRuntimeError("that spreadsheet has no tabs")
    known = {t["title"] for t in tabs}
    if title not in known:
        raise BindingRuntimeError(
            f"'{title}' is not a tab of that spreadsheet. Tabs: "
            f"{', '.join(sorted(known)) or '(none)'}"
        )
    sheet_id = next((t["sheet_id"] for t in tabs if t["title"] == title), None)

    start = max(1, int(header_row))
    a1 = sheets_a1_range(title, f"A{start}", f"{LAST_PROBE_COLUMN}{start + PROBE_SAMPLE_ROWS}")
    payload = await _call(source, executor, OP_GET_VALUES, {
        "spreadsheet_id": file_id,
        "range": a1,
        "value_render_option": "UNFORMATTED_VALUE",
        "date_time_render_option": "FORMATTED_STRING",
    })
    grid = list((payload or {}).get("values") or []) if isinstance(payload, dict) else []
    headers = [str(h) for h in (grid[0] if grid else [])]
    # Sheets truncates trailing empty cells, so the sample rows are padded to
    # the header width here — the editor renders them as a table under the
    # headers, and a ragged table is unreadable.
    sample_rows = [
        [row[i] if i < len(row) else "" for i in range(len(headers))]
        for row in grid[1:]
    ]

    return {
        "file_id": file_id,
        "tabs": tabs,
        "named_ranges": named_ranges,
        "sheet": title,
        "sheet_id": sheet_id,
        "header_row": start,
        "headers": headers,
        "sample_rows": sample_rows,
        "fingerprint": header_fingerprint(headers),
        "range": a1,
    }


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

async def run_read_binding(
    source: Any,
    executor: Any,
    binding: SheetBinding,
    params: dict[str, Any],
) -> Any:
    """Execute a read binding and return its result.

    ``rows`` gives a list of records, ``row_by_key`` a single record (or
    ``None`` / an error, per ``on_missing``), ``cells`` the raw value grid of
    the named range.
    """
    read = binding.read
    if read is None:
        raise BindingRuntimeError(f"binding '{binding.name}' is not a read binding")
    by_path = values_by_state_path(binding, params)
    a1 = binding_read_range(binding)
    grid, payload = await _read_grid(source, executor, binding, a1)

    if read.mode == "cells":
        # A named range is returned as-is: there is no header row to check, and
        # nothing is resolved by column, so a fingerprint would be checking
        # something this mode never uses.
        return {"range": payload.get("range") or a1, "values": grid}

    headers = [str(h) for h in (grid[0] if grid else [])]
    sheets_check_fingerprint(headers, binding.sheet_schema.fingerprint)
    records = sheets_rows_to_records(grid[1:], headers)

    # Tier 2: a generated transform replaces the filter/project half of the
    # read, and nothing else.  It runs here -- after the fingerprint check, on
    # records built by tier-1 code -- so the columns it sees are the columns the
    # binding was authored against, or the run has already failed.
    if binding.compute is not None:
        return await run_compute_read(binding, records, params)

    if read.mode == "row_by_key":
        wanted = sheets_resolve_tagged(
            read.key_value.model_dump() if read.key_value else None, by_path
        )
        match = next(
            (
                r for r in records
                if _same(r.get(read.key_column or ""), wanted)
            ),
            None,
        )
        if match is None:
            if read.on_missing == "error":
                raise BindingRuntimeError(
                    f"no row in '{binding.document.sheet}' has "
                    f"{read.key_column} = {wanted!r}"
                )
            return None
        return sheets_project_records([match], read.columns)[0]

    filter_spec = read.filter.model_dump() if read.filter is not None else None
    matched = sheets_filter_records(records, filter_spec, by_path)
    if read.limit:
        matched = matched[: read.limit]
    return sheets_project_records(matched, read.columns)


def _same(left: Any, right: Any) -> bool:
    """The lenient equality the resolver uses, for the key lookup."""
    from app.infrastructure.datasources.sheet_binding_resolver import _compare
    return _compare(left, "eq", right)


# ---------------------------------------------------------------------------
# Tier 2: the generated transform
# ---------------------------------------------------------------------------
# Both helpers below run the *pinned* code stored on the binding, in the same
# seccomp sandbox as a user-authored `python` step, through the same
# `run_script` entry point. There is no separate path for code the platform
# generated itself and no `exec` anywhere: code a model wrote from an
# instruction a user typed is the least trusted code here, not the most.

async def _run_pinned(
    binding: SheetBinding,
    records: list[dict[str, Any]],
    params: dict[str, Any],
) -> Any:
    """Run the binding's stored transform, with the per-run audit line.

    The log line is the whole of this feature's debuggability: six months on,
    a binding that wrote a surprising value is explained by which script id and
    which content hash ran, over how many rows, and what came back.
    """
    from app.core.config import get_settings
    from app.infrastructure.datasources.sheet_compute import (
        ComputeValidationError,
        run_transform,
        validate_transform_source,
    )

    compute = binding.compute
    assert compute is not None

    if not compute.activated:
        raise BindingRuntimeError(
            f"binding '{binding.name}' holds generated code that has not been "
            "activated. Open it, review the code and its output on your own "
            "sample rows, and confirm it — generated code never starts running "
            "just because it compiled."
        )
    if compute.stale:
        raise BindingRuntimeError(
            f"binding '{binding.name}' is marked stale"
            + (f" ({compute.stale_reason})" if compute.stale_reason else "")
            + ". Re-test it and confirm the result before it runs again."
        )

    settings = get_settings()
    if not settings.sheets_compute_enabled:
        raise BindingRuntimeError(
            "generated sheet transforms are disabled on this backend "
            "(SHEETS_COMPUTE_ENABLED)"
        )
    if binding.operation == "write" and not settings.sheets_compute_writes_enabled:
        # The flag is checked at run time as well as at save time: a binding
        # stored while it was on must stop working when it is turned off, or
        # the flag would only ever have been an authoring formality.
        raise BindingRuntimeError(
            f"binding '{binding.name}' is a generated-code WRITE, and those "
            "are disabled on this backend (SHEETS_COMPUTE_WRITES_ENABLED). "
            "The read side of tier 2 is unaffected."
        )

    # Re-validated on the way in, every run. Tightening the AST allow-list must
    # retroactively stop code an earlier, looser version accepted.
    try:
        validate_transform_source(compute.code)
    except ComputeValidationError as exc:
        raise BindingRuntimeError(
            f"binding '{binding.name}' holds a transform that no longer passes "
            f"validation: {exc}"
        ) from exc

    output = await run_transform(
        compute.code,
        records,
        params,
        timeout=float(settings.sheets_compute_timeout_seconds),
        validate=False,
    )
    logger.info(
        "tier-2 %s '%s': script=%s hash=%s rows_in=%d out=%s",
        binding.operation, binding.name, compute.script_id,
        compute.content_hash, len(records), _output_summary(output),
    )
    return output


def _output_summary(output: Any) -> str:
    """A short, log-safe description of a transform's result.

    Deliberately a shape rather than the values: a read binding's output can be
    a thousand rows of somebody's data, and the log line is for debugging what
    ran, not for keeping a copy of the spreadsheet.
    """
    if isinstance(output, list):
        keys = sorted(output[0]) if output and isinstance(output[0], dict) else []
        return f"{len(output)} row(s)" + (f" keyed {keys}" if keys else "")
    if isinstance(output, dict):
        return f"1 row keyed {sorted(output)}"
    return type(output).__name__


async def run_compute_read(
    binding: SheetBinding,
    records: list[dict[str, Any]],
    params: dict[str, Any],
) -> Any:
    """A tier-2 read: transform the records, then check the declared shape."""
    from app.infrastructure.datasources.sheet_compute import (
        ComputeValidationError,
        check_read_output,
    )
    from app.infrastructure.datasources.sheet_compute_generate import records_for

    compute = binding.compute
    assert compute is not None
    # `read.columns`, when the binding names any, limits what the transform can
    # even see -- so a generated read cannot start reporting a column the
    # author never put in the binding.
    read = binding.read
    if read is not None and read.columns:
        records = sheets_project_records(records, read.columns)
    output = await _run_pinned(binding, records, params)
    try:
        return check_read_output(output, compute.output_shape)
    except ComputeValidationError as exc:
        raise BindingRuntimeError(
            f"binding '{binding.name}' returned the wrong shape: {exc}"
        ) from exc


async def run_compute_write(
    binding: SheetBinding,
    grid: list[list[Any]],
    headers: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    """A tier-2 write: transform the rows into ``{column: value}``, whitelist-checked.

    The whitelist check is the load-bearing line of the whole feature, and it
    is a hard runtime error rather than a filter: a transform that returns a
    column the binding does not declare has misunderstood the job, and silently
    dropping the extra value would leave half the author's intent unwritten
    while looking like success.
    """
    from app.infrastructure.datasources.sheet_compute import (
        ComputeValidationError,
        check_write_output,
    )

    write = binding.write
    assert write is not None
    records = sheets_rows_to_records(grid[1:], headers)
    output = await _run_pinned(binding, records, params)
    try:
        return check_write_output(output, list(write.columns))
    except ComputeValidationError as exc:
        raise BindingRuntimeError(
            f"binding '{binding.name}' produced values it may not write: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

async def plan_write_binding(
    source: Any,
    executor: Any,
    binding: SheetBinding,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Steps 1-4 of the write order: read, resolve, fingerprint, build.

    Returns everything the write needs plus what a person needs to see before
    it happens — ``cells`` carries the before/after of every cell — and makes
    no change.  The preview endpoint and the approval gate call exactly this,
    so an approver is shown the write that will actually be attempted rather
    than a reconstruction of it.
    """
    write = binding.write
    if write is None:
        raise BindingRuntimeError(f"binding '{binding.name}' is not a write binding")
    by_path = values_by_state_path(binding, params)
    # A tier-2 write cannot resolve its columns yet: the values for its
    # `compute.*` references do not exist until the transform has run, which
    # cannot happen until the grid is read and its fingerprint checked. So the
    # resolution is deferred to step 3b below, after both.
    values = {} if binding.compute is not None else resolved_columns(binding, by_path)

    if write.mode == "set_cells":
        return await _plan_set_cells(source, executor, binding, values)

    # 1. read the table, header row included.
    a1 = binding_key_range(binding)
    grid, _payload = await _read_grid(source, executor, binding, a1)
    headers = [str(h) for h in (grid[0] if grid else [])]

    # 2. resolve the row -- from this very response, never a cached number.
    row_number: int | None = None
    action = write.mode
    if write.mode == "update_by_key":
        wanted = sheets_resolve_tagged(
            write.key_value.model_dump() if write.key_value else None, by_path
        )
        row_number = sheets_resolve_row(
            grid, write.key_column or "", wanted, binding.sheet_schema.header_row
        )
        if row_number is None:
            if write.on_missing == "error":
                raise BindingRuntimeError(
                    f"no row in '{binding.document.sheet}' has "
                    f"{write.key_column} = {wanted!r}, and this binding's "
                    "on_missing is 'error'"
                )
            if write.on_missing == "skip":
                return {
                    "status": "skipped",
                    "reason": (
                        f"no row has {write.key_column} = {wanted!r}; "
                        "on_missing is 'skip'"
                    ),
                    "binding": binding.name,
                    "mode": write.mode,
                    "row_number": None,
                    "cells": [],
                    "call": None,
                }
            action = "append_row"

    # 3. check the fingerprint, against the header row just read.
    sheets_check_fingerprint(headers, binding.sheet_schema.fingerprint)

    # 3b. Tier 2: run the generated transform over the rows just read and let
    # it supply the values for the columns it is allowed to.  It sits *between*
    # the fingerprint check and the build on purpose: it computes values from a
    # grid already proven to match the binding's schema, and the build that
    # turns those values into A1 ranges is still tier-1 code.
    if binding.compute is not None:
        computed = await run_compute_write(binding, grid, headers, params)
        for field, value in computed.items():
            by_path[f"{COMPUTE_PATH_PREFIX}{field}"] = value
        values = resolved_columns(binding, by_path)
        # A transform that named a column directly wins over that column's
        # declared tagged value; the whitelist check above has already refused
        # any name that is not a declared column.
        values.update(computed)

    # 4. build the write.
    payload = binding.model_dump(mode="json")
    if action == "append_row":
        payload["write"] = {**payload["write"], "mode": "append_row"}
    built = sheets_build_write(payload, values, row_number)
    cells = _with_before(built["cells"], grid, headers, binding, row_number)
    return {
        "status": "ready",
        "reason": None,
        "binding": binding.name,
        "mode": built["mode"],
        "row_number": built["row_number"],
        "cells": cells[:_PREVIEW_CELL_CAP],
        "cells_total": len(cells),
        "written_columns": built["written_columns"],
        "value_input_option": write.value_input_option,
        "blank_policy": write.blank_policy,
        "call": _write_call(binding, built),
    }


async def _plan_set_cells(
    source: Any, executor: Any, binding: SheetBinding, values: dict[str, Any]
) -> dict[str, Any]:
    """``set_cells`` needs no row lookup, but still needs the before values."""
    write = binding.write
    assert write is not None
    built = sheets_build_write(binding.model_dump(mode="json"), values, None)
    ranges = [entry["a1"] for entry in built["cells"]]
    before: dict[str, Any] = {}
    if ranges:
        payload = await _call(source, executor, OP_BATCH_GET_VALUES, {
            "spreadsheet_id": binding.document.file_id,
            "ranges": ranges,
            "value_render_option": "UNFORMATTED_VALUE",
            "date_time_render_option": "FORMATTED_STRING",
        })
        for wanted, result in zip(ranges, (payload or {}).get("valueRanges") or []):
            grid = result.get("values") or []
            before[wanted] = (grid[0][0] if grid and grid[0] else "")
    cells = [
        {**cell, "before": before.get(cell["a1"], ""), "after": cell["value"]}
        for cell in built["cells"]
    ]
    return {
        "status": "ready",
        "reason": None,
        "binding": binding.name,
        "mode": "set_cells",
        "row_number": None,
        "cells": cells[:_PREVIEW_CELL_CAP],
        "cells_total": len(cells),
        "written_columns": [],
        "value_input_option": write.value_input_option,
        "blank_policy": write.blank_policy,
        "call": _write_call(binding, built),
    }


def _with_before(
    cells: list[dict[str, Any]],
    grid: list[list[Any]],
    headers: list[str],
    binding: SheetBinding,
    row_number: int | None,
) -> list[dict[str, Any]]:
    """Attach the current cell value to each planned change.

    Read out of the grid already in hand rather than with a second call, so the
    "before" an approver sees is the same snapshot the row number came from.
    """
    if row_number is None:
        # An append: every cell of the new row is empty by definition.
        return [
            {**cell, "before": "", "after": cell["value"]}
            for cell in cells
        ]
    data_index = row_number - binding.sheet_schema.header_row
    row = list(grid[data_index]) if 0 <= data_index < len(grid) else []
    out: list[dict[str, Any]] = []
    for cell in cells:
        column = cell.get("column")
        index = headers.index(column) if column in headers else -1
        current = row[index] if 0 <= index < len(row) else ""
        out.append({
            **cell,
            "before": current,
            "after": current if cell["action"] == "skip" else cell["value"],
        })
    return out


def _write_call(binding: SheetBinding, built: dict[str, Any]) -> dict[str, Any]:
    """The raw operation call the plan resolves to, as ``{operation, params}``."""
    write = binding.write
    assert write is not None
    if built["mode"] == "append_row":
        return {
            "operation": OP_APPEND_VALUES,
            "params": {
                "spreadsheet_id": binding.document.file_id,
                "range": binding_key_range(binding),
                "values": built["values"],
                "value_input_option": write.value_input_option,
                # INSERT_ROWS rather than OVERWRITE: an append that overwrites
                # is not an append, and OVERWRITE will happily land on a row
                # somebody added below the table.
                "insert_data_option": "INSERT_ROWS",
            },
        }
    return {
        "operation": OP_BATCH_UPDATE_VALUES,
        "params": {
            "spreadsheet_id": binding.document.file_id,
            "data": built["data"],
            "value_input_option": write.value_input_option,
        },
    }


async def run_write_binding(
    source: Any,
    executor: Any,
    binding: SheetBinding,
    params: dict[str, Any],
) -> Any:
    """The full write: plan (steps 1-4) then call (step 5), one execution."""
    plan = await plan_write_binding(source, executor, binding, params)
    if plan["status"] == "skipped":
        return {
            "status": "skipped",
            "reason": plan["reason"],
            "cells_written": 0,
        }
    call = plan["call"]
    if not call or not (call["params"].get("data") or call["params"].get("values")):
        # Every named column resolved blank under blank_policy "skip".  Not an
        # error: the binding did what it says, which is to leave those cells
        # alone.
        return {
            "status": "no_change",
            "reason": "every named column resolved to a blank value and blank_policy is 'skip'",
            "cells_written": 0,
        }
    response = await _call(source, executor, call["operation"], call["params"])
    written = [c for c in plan["cells"] if c["action"] == "write"]
    return {
        "status": "ok",
        "mode": plan["mode"],
        "row_number": plan["row_number"],
        "cells_written": len(written),
        "cells": written,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Entry points used by the executor
# ---------------------------------------------------------------------------

async def run_binding(
    source: Any,
    executor: Any,
    binding: SheetBinding,
    params: dict[str, Any],
) -> Any:
    """Execute *binding* -- the single entry point the executor delegates to."""
    if binding.operation == "read":
        return await run_read_binding(source, executor, binding, params)
    return await run_write_binding(source, executor, binding, params)


def render_cell_changes(cells: list[dict[str, Any]]) -> list[str]:
    """Cell changes as lines a person can read.

    The approval panel and the Slack message show this instead of the JSON body
    of a ``batchUpdate``: an approver has to decide whether *this change to this
    cell* is right, and ``{"data":[{"range":"Projects!C7","values":[["closed"]]}]}``
    does not answer that question.
    """
    lines: list[str] = []
    for cell in cells:
        column = f" ({cell['column']})" if cell.get("column") else ""
        if cell.get("action") == "skip":
            lines.append(f"{cell['a1']}{column}: left unchanged ({cell.get('before')!r})")
        else:
            lines.append(
                f"{cell['a1']}{column}: {cell.get('before')!r} → {cell.get('after')!r}"
            )
    return lines


async def binding_destructive_plan(
    source: Any,
    executor: Any,
    binding: SheetBinding,
    params: dict[str, Any],
) -> tuple[int, list[str], list[Any], str]:
    """``(affected_rows, targets, sample, rows_label)`` for the approval gate.

    ``affected_rows`` counts cells actually being written, so a write whose
    every column was skipped reports zero and the gate lets it through — there
    is nothing for a person to approve.

    ``rows_label`` names WHICH rows are touched without saying anything about
    their contents ("row 7", "a new row"). It exists so the Slack message can
    state the blast radius without carrying data values into a channel — the
    values stay in ``sample``, which only the authenticated editor and the
    management MCP read.
    """
    plan = await plan_write_binding(source, executor, binding, params)
    if plan["status"] == "skipped":
        return 0, [], [], ""
    written = [c for c in plan["cells"] if c["action"] == "write"]
    target = (
        f"{binding.document.name or binding.document.file_id} · "
        f"{binding.document.sheet}"
    )
    rows_label = ""
    if plan.get("row_number"):
        rows_label = f"row {plan['row_number']}"
        target = f"{target} {rows_label}"
    elif plan["mode"] == "append_row":
        rows_label = "a new row"
        target = f"{target} (new row)"
    return len(written), [target], render_cell_changes(plan["cells"]), rows_label


def a1_for_column(binding: SheetBinding, column: str, row_number: int) -> str:
    """A1 reference of one cell of one row, for a message or a log line."""
    headers = binding.sheet_schema.headers
    if column not in headers:
        raise BindingRuntimeError(f"column '{column}' is not in the binding's schema")
    letter = sheets_column_letter(headers.index(column))
    return sheets_a1_range(binding.document.sheet, f"{letter}{row_number}")
