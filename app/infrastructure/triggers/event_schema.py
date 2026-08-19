"""Payload validation for event triggers.

Deliberately the same minimal JSON-schema-ish check the data source executor
applies to operation responses (``_validate_response`` there): top-level type,
required keys, property types.  Event schemas are authored by the same people
in the same UI, so the two must accept the same documents — a full JSON Schema
implementation here would reject shapes the datasource side happily takes.
"""
from __future__ import annotations

from typing import Any

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def validate_event_payload(payload: Any, schema: dict[str, Any] | None, label: str) -> None:
    """Raise ``ValueError`` when *payload* does not match *schema*.

    A falsy schema accepts anything — a trigger without a schema is a valid
    configuration, not an error.  *label* names the trigger in messages, e.g.
    ``"workflow 'orders' step 'on_order'"``.
    """
    if not schema:
        return

    expected = schema.get("type")
    if expected and expected in _JSON_TYPES and not isinstance(payload, _JSON_TYPES[expected]):
        raise ValueError(
            f"{label}: event payload is {type(payload).__name__}, expected {expected}"
        )
    if not isinstance(payload, dict):
        return

    for key in schema.get("required", []) or []:
        if key not in payload:
            raise ValueError(f"{label}: event payload is missing required key '{key}'")

    for key, spec in (schema.get("properties") or {}).items():
        if key not in payload or not isinstance(spec, dict):
            continue
        prop_type = spec.get("type")
        if (
            prop_type in _JSON_TYPES
            and payload[key] is not None
            and not isinstance(payload[key], _JSON_TYPES[prop_type])
        ):
            raise ValueError(
                f"{label}: event payload key '{key}' is "
                f"{type(payload[key]).__name__}, expected {prop_type}"
            )
