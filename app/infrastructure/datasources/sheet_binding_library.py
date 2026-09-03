"""The binding resolvers, packaged as script-library entries.

``sheet_binding_resolver`` is imported directly by the binding runtime, which
runs in the backend process.  The same functions are *also* published into the
script library so that:

* a ``python`` workflow step can call them, sandboxed, on a grid it obtained
  some other way — the seccomp sandbox has no network and no site-packages, so
  a step cannot ``import`` a backend module and this is the only way code gets
  in there (see ``app.infrastructure.orchestration.script_sandbox``);
* the padding rule and the fingerprint check exist once.  A workflow that
  hand-rolled "zip headers against rows" would reintroduce exactly the ragged-row
  bug ``sheets_rows_to_records`` exists to prevent.

Each entry's ``code`` is the resolver module's own source followed by a short
block that reads ``state`` and assigns ``output``.  The source is read from disk
rather than retyped, so a library entry can never drift from the module the
backend runs — and the module is deliberately stdlib-only for this reason.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RESOLVER_PATH = Path(__file__).with_name("sheet_binding_resolver.py")

# Script ids.  Stable, because a workflow step references one by id.
SCRIPT_IDS = (
    "sheets-rows-to-records",
    "sheets-filter-records",
    "sheets-resolve-row",
    "sheets-build-write",
    "sheets-check-fingerprint",
)

_ENTRIES: tuple[dict[str, str], ...] = (
    {
        "id": "sheets-rows-to-records",
        "name": "Sheets: rows to records",
        "description": (
            "Turn a Google Sheets value grid into records keyed by header name. "
            "state: {values (the grid BELOW the header row), headers}. "
            "Pads every row to the header width first — the Sheets API drops "
            "trailing empty cells, so naive zipping shifts data silently."
        ),
        "body": 'output = sheets_rows_to_records(state["values"], state["headers"])',
    },
    {
        "id": "sheets-filter-records",
        "name": "Sheets: filter records",
        "description": (
            "Apply a binding filter tree (and/or over eq ne lt lte gt gte in "
            "contains) to records. state: {records, filter, params (values for "
            "any {\"from\": …} clause, keyed by state path)}."
        ),
        "body": (
            'output = sheets_filter_records(\n'
            '    state["records"], state.get("filter"), state.get("params") or {}\n'
            ')'
        ),
    },
    {
        "id": "sheets-resolve-row",
        "name": "Sheets: resolve row by key",
        "description": (
            "Find the 1-based sheet row number whose key column matches a value. "
            "state: {values (the grid INCLUDING its header row), key_column, "
            "key_value, header_row}. Returns null when nothing matches. Resolve "
            "and write in the same step: an inserted row invalidates the number."
        ),
        "body": (
            'output = sheets_resolve_row(\n'
            '    state["values"], state["key_column"], state["key_value"],\n'
            '    state.get("header_row") or 1,\n'
            ')'
        ),
    },
    {
        "id": "sheets-build-write",
        "name": "Sheets: build write payload",
        "description": (
            "Compose the {range, values} entries for batch_update_values (or a "
            "row for append_values) from a binding and its resolved column "
            "values. state: {binding, resolved_values, row_number}. A column "
            "absent from binding.write.columns is never touched."
        ),
        "body": (
            'output = sheets_build_write(\n'
            '    state["binding"], state["resolved_values"], state.get("row_number")\n'
            ')'
        ),
    },
    {
        "id": "sheets-check-fingerprint",
        "name": "Sheets: check header fingerprint",
        "description": (
            "Raise unless the sheet's header row still hashes to the binding's "
            "recorded fingerprint. state: {headers, fingerprint}. There is no "
            "fallback: a mismatch means every column position is suspect, and "
            "writing by position anyway is how a sheet quietly gets corrupted."
        ),
        "body": (
            'sheets_check_fingerprint(state["headers"], state["fingerprint"])\n'
            'output = {"ok": True}'
        ),
    },
)


# The sandbox's import guard refuses `__future__` along with everything else
# that is not pre-imported, so the line has to come out of the shipped copy.
# Nothing depends on it: the module's annotations are all builtin generics and
# PEP 604 unions, which Python evaluates natively.
_FUTURE_IMPORT = "from __future__ import annotations\n"


def _resolver_source() -> str:
    source = _RESOLVER_PATH.read_text(encoding="utf-8")
    return source.replace(_FUTURE_IMPORT, "")


def sheet_binding_library_scripts() -> list[dict[str, Any]]:
    """The library entries, ready to be stored as ``ScriptDefinition``s."""
    source = _resolver_source()
    scripts: list[dict[str, Any]] = []
    for entry in _ENTRIES:
        code = (
            f"{source}\n\n"
            "# --- library entry -------------------------------------------------\n"
            "# Everything above is app/infrastructure/datasources/"
            "sheet_binding_resolver.py,\n"
            "# copied verbatim so this runs in the sandbox (no imports available "
            "beyond\n# the standard library). Do not edit it here — edit the module.\n"
            f"{entry['body']}\n"
        )
        scripts.append({
            "id": entry["id"],
            "name": entry["name"],
            "description": entry["description"],
            "code": code,
        })
    return scripts


async def ensure_binding_scripts(script_backend: Any) -> list[str]:
    """Upsert the resolver entries into the script library.

    Called whenever a binding is saved: the entries are how a hand-written
    workflow reuses the same logic, and a library that only fills up if somebody
    remembers to seed it is a library nobody trusts.  Idempotent — the ids are
    fixed, so a save overwrites in place — and never fatal: a binding is
    perfectly usable through its compiled operation with no library entry at
    all, so a failure here is logged and the save proceeds.
    """
    if script_backend is None:
        return []
    from app.domain.models.script_definition import ScriptDefinition

    written: list[str] = []
    for payload in sheet_binding_library_scripts():
        try:
            existing = await script_backend.get(payload["id"])
            defn = ScriptDefinition(
                id=payload["id"],
                name=payload["name"],
                description=payload["description"],
                code=payload["code"],
                created_at=existing.created_at if existing else None,
            )
            if existing is None:
                await script_backend.create(defn)
            elif existing.code != defn.code or existing.description != defn.description:
                await script_backend.update(defn.id, defn)
            written.append(payload["id"])
        except Exception:
            logger.warning(
                "could not publish binding resolver script '%s' to the library",
                payload["id"], exc_info=True,
            )
    return written
