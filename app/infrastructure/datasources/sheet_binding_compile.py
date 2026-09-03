"""Compiling a sheet binding into a named ``OperationDefinition``.

A binding is only useful if the rest of the platform can *call* it, and the
platform already has exactly one way of naming something callable on a data
source: an operation.  So saving a binding generates (or refreshes) an
operation of the same name on the same source, and every existing surface picks
it up for free —

* a workflow ``data_source`` step (``source: google-sheets``,
  ``operation: read_open_projects``),
* the approval gate, because a write binding compiles to
  ``destructive: true``,
* ``POST /datasources/try-operation``,
* the ``datasources`` MCP tool list, republished by
  ``_refresh_datasource_tools`` on every mutation.

None of them needed a new step type, a new gate or a new tool kind.

The operation's **params are exactly the state paths the binding references**.
A binding that reads ``{"from": "state.project.id"}`` compiles to an operation
with one param, ``project_id``; the caller (a workflow step's ``params``, an
agent calling the MCP tool) supplies the value, and the binding runtime
resolves the ``from`` against it.  So the datasource layer never reaches into
workflow state — it is handed values, like every other operation.

The templated ``path`` on a compiled operation is documentary: the request is
intercepted by ``DataSourceExecutor`` and served by
``sheet_binding_runtime``, which composes the real Sheets calls.  It is written
out accurately anyway so ``get_datasource`` and the editor show something true
about where the operation points.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    OperationDefinition,
    ParamSpec,
    RetryPolicy,
)
from app.domain.models.sheet_binding import SheetBinding
from app.infrastructure.datasources.sheet_binding_resolver import sheets_a1_range

# Longest column a probe reads / a binding addresses.  Sheets' own default
# grid is 26 columns wide, and a binding is authored from a probe that reads
# A..Z, so the compiled read range matches what the author saw.
LAST_PROBE_COLUMN = "Z"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def param_name_for(state_path: str) -> str:
    """A param name for a state path.

    ``state.project.id`` -> ``project_id``.  The leading ``state.`` is dropped
    because every path has it and it would make every param name start the same
    way; the rest is flattened to lower snake case, which is what both the
    reference syntax and an MCP tool signature accept.
    """
    trimmed = state_path.strip()
    for prefix in ("state.", "$.state.", "$."):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix):]
            break
    slug = _SLUG_RE.sub("_", trimmed.lower()).strip("_")
    if not slug:
        slug = "value"
    if slug[0].isdigit():
        slug = f"v_{slug}"
    return slug


def binding_params(binding: SheetBinding) -> list[tuple[str, str]]:
    """``(param_name, state_path)`` pairs, in the binding's own stable order.

    Two different paths that flatten to the same name (``state.a.id`` and
    ``state.b.id`` both give ``a_id``/``b_id``, but ``state.a.id`` and
    ``state.a_id`` do collide) are disambiguated by suffixing the second, so a
    binding is never rejected for a naming accident and the mapping stays
    one-to-one.
    """
    pairs: list[tuple[str, str]] = []
    taken: set[str] = set()
    for path in binding.state_paths():
        name = param_name_for(path)
        if name in taken:
            index = 2
            while f"{name}_{index}" in taken:
                index += 1
            name = f"{name}_{index}"
        taken.add(name)
        pairs.append((name, path))
    return pairs


def binding_param_map(binding: SheetBinding) -> dict[str, str]:
    """``param_name -> state_path``, the runtime's key to the caller's inputs."""
    return {name: path for name, path in binding_params(binding)}


def binding_read_range(binding: SheetBinding) -> str:
    """The A1 range a read binding's underlying ``get_values`` call reads.

    ``rows`` / ``row_by_key`` read the header row and everything below it, so
    the header row can be fingerprinted against the same response the records
    come from — two calls would leave a window in which the headers changed
    between them.  ``cells`` reads exactly what the binding names.
    """
    read = binding.read
    if read is not None and read.mode == "cells" and read.range is not None:
        return read.range.render()
    header_row = binding.sheet_schema.header_row
    return sheets_a1_range(
        binding.document.sheet, f"A{header_row}", LAST_PROBE_COLUMN
    )


def binding_key_range(binding: SheetBinding) -> str:
    """The range a write binding reads before it resolves its row.

    The whole table, not just the key column: the read has to carry the header
    row (for the fingerprint check) and the key column, and the key column's
    position is only known from the headers.  One call for both is also one
    consistent snapshot.
    """
    header_row = binding.sheet_schema.header_row
    return sheets_a1_range(
        binding.document.sheet, f"A{header_row}", LAST_PROBE_COLUMN
    )


def _param_specs(binding: SheetBinding) -> list[ParamSpec]:
    return [
        ParamSpec(
            name=name,
            type="string",
            required=True,
            description=f"Value of {path} (used by binding '{binding.name}')",
        )
        for name, path in binding_params(binding)
    ]


def _read_description(binding: SheetBinding) -> str:
    read = binding.read
    assert read is not None
    document = binding.document.name or binding.document.file_id
    where = f"tab '{binding.document.sheet}' of '{document}'"
    if read.mode == "cells":
        what = f"cells {read.range.render() if read.range else ''}"
    else:
        what = f"columns {', '.join(read.columns)}"
        if read.mode == "row_by_key":
            what = f"the row where {read.key_column} matches, {what}"
        elif read.filter is not None:
            what = f"{what}, filtered"
        if read.limit:
            what = f"{what} (at most {read.limit} rows)"
    tail = f" Result is published under '{binding.output.key}'." if binding.output.key else ""
    if binding.compute is not None:
        # Said plainly, in the one string that reaches an agent's tool list and
        # the editor's operation view: a caller deciding whether to trust this
        # operation should not have to go and look up the binding to find out
        # that a model wrote its computation.
        what = (
            f"columns {', '.join(read.columns)}" if read.columns else "every column"
        )
        return (
            f"Read {what} from {where} and compute a result with a "
            f"GENERATED transform ({binding.compute.script_id}, output shape "
            f"'{binding.compute.output_shape}'). The computation was written by "
            f"a language model from a natural-language instruction and runs "
            f"sandboxed.{tail}"
        )
    return f"Read {what} from {where}.{tail}"


def _write_description(binding: SheetBinding) -> str:
    write = binding.write
    assert write is not None
    document = binding.document.name or binding.document.file_id
    where = f"tab '{binding.document.sheet}' of '{document}'"
    if write.mode == "set_cells":
        what = f"{len(write.cells)} explicit cell range(s)"
    else:
        what = f"columns {', '.join(sorted(write.columns))}"
        if write.mode == "append_row":
            what = f"a new row with {what}"
        else:
            what = f"{what} of the row where {write.key_column} matches"
    guard = (
        "Values are stored as text (RAW)."
        if write.value_input_option == "RAW"
        else "Values are parsed by Sheets (USER_ENTERED)."
    )
    generated = ""
    if binding.compute is not None:
        generated = (
            f" The values come from a GENERATED transform "
            f"({binding.compute.script_id}), written by a language model from a "
            f"natural-language instruction and run sandboxed; it can only "
            f"produce values for the columns named above."
        )
    return (
        f"WRITE: set {what} in {where}. Columns not named here are left "
        f"untouched. {guard}{generated} Needs approval."
    )


def compile_binding(binding: SheetBinding) -> OperationDefinition:
    """The operation a binding compiles to.

    Read bindings compile to a GET, write bindings to a POST marked
    ``destructive: true`` — not because POST is destructive (it is not, by the
    verb rule in ``datasources.destructive``) but because overwriting somebody's
    spreadsheet is exactly what the approval gate exists for.  ``append_row``
    additionally carries ``retries.attempts = 1``: an append is not idempotent,
    so a retry after a timeout that in fact reached Sheets appends the row
    twice.
    """
    params = _param_specs(binding)
    file_id = binding.document.file_id
    if binding.operation == "read":
        return OperationDefinition(
            name=binding.name,
            method="GET",
            description=_read_description(binding),
            path=f"/v4/spreadsheets/{file_id}/values/{binding_read_range(binding)}",
            params=params,
        )

    write = binding.write
    assert write is not None
    if write.mode == "append_row":
        path = f"/v4/spreadsheets/{file_id}/values/{binding_key_range(binding)}:append"
        # Not idempotent — see the docstring.
        retries: RetryPolicy | None = RetryPolicy(attempts=1)
    else:
        path = f"/v4/spreadsheets/{file_id}/values:batchUpdate"
        # An `update_by_key` write is idempotent (it overwrites one known row),
        # so it may use the source's retry policy — unless `on_missing` lets it
        # fall back to appending, which is not.  A retried timeout on that path
        # would add the row twice, so pin the same single attempt here: the
        # binding cannot know whether the source policy is a safe 1.
        retries = RetryPolicy(attempts=1) if write.on_missing == "append" else None
    return OperationDefinition(
        name=binding.name,
        method="POST",
        description=_write_description(binding),
        path=path,
        params=params,
        destructive=True,
        retries=retries,
    )


def stamp_compiled(binding: SheetBinding) -> SheetBinding:
    """Record when the binding was last compiled, leaving provenance alone."""
    binding.resolution.compiled_at = datetime.now(timezone.utc)
    return binding


def refresh_binding_operations(
    definition: DataSourceDefinition,
    bindings: list[SheetBinding],
) -> list[OperationDefinition]:
    """The source's operation list with every binding's operation up to date.

    Raw operations (``get_values``, ``batch_update_values``, …) are kept
    untouched and in place; a binding's operation replaces the one of the same
    name, or is appended.  An operation left over from a binding that no longer
    exists is dropped — but only if it *was* one: an operation whose name
    happens to match nothing is a hand-written operation and stays.

    Left-over detection works from the previous binding set, which is why this
    takes the whole definition rather than just its operations.
    """
    previous = {b.name for b in getattr(definition, "bindings", []) or []}
    current = {b.name for b in bindings}
    compiled = {b.name: compile_binding(b) for b in bindings}

    operations: list[OperationDefinition] = []
    for op in definition.operations:
        if op.name in compiled:
            operations.append(compiled.pop(op.name))
        elif op.name in previous and op.name not in current:
            # Its binding was deleted or renamed; the operation goes with it.
            continue
        else:
            operations.append(op)
    # Anything not already in the list, in binding order.
    operations.extend(compiled[b.name] for b in bindings if b.name in compiled)
    return operations


def binding_for_operation(
    definition: Any, operation: str
) -> SheetBinding | None:
    """The binding *operation* was compiled from, or ``None``.

    The lookup the executor makes on every call, so it is a name comparison and
    nothing more.
    """
    for binding in getattr(definition, "bindings", None) or []:
        if binding.name == operation:
            return binding
    return None
