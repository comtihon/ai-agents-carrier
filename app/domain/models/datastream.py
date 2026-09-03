"""References to data source results, which always live in a file.

A data source result never travels as a value.  The executor writes every
result -- one record or four million -- to the data stream store and returns a
:class:`DataRef`: an id, counts, and a hard-capped preview.  The ref is what
enters workflow state, so state, the LangGraph checkpoint and the Mongo run
document are never functions of result size.

Consumers open a file descriptor and read.  ``shape`` says whether iterating
makes sense: ``"list"`` is one record per line, ``"value"`` is a single JSON
document on one line -- readable, but with no record boundary to iterate.

See ``app.infrastructure.datasources.datastream`` for the store, and for why
this is unconditional rather than a size threshold.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# Marker key. Present (and truthy) on a spill handle and on nothing else, so
# `is_data_ref` never has to guess at a dict that merely looks similar.
STREAM_MARKER = "__stream__"
STREAM_VERSION = 1

# How many leading items the handle carries inline. Deliberately tiny: the
# handle's whole purpose is to be small enough for a checkpoint, and a preview
# is for recognising the data, not for working with it.
PREVIEW_ITEMS = 3
PREVIEW_STRING_CAP = 200


class DataRef(BaseModel):
    """Reference to a spilled data source result.

    Serialises to a plain dict small enough to sit in workflow state, be
    replaced into a Mongo run document, and be rendered into a sandbox
    payload, regardless of how large the data it points at is.
    """

    # Aliased so the marker key survives a round-trip through `model_dump`.
    version: int = Field(default=STREAM_VERSION, alias=STREAM_MARKER)

    id: str
    shape: Literal["list", "value"] = "list"
    # Item count for shape="list"; always 1 for shape="value".
    items: int = 0
    # Size of the stored JSONL payload in bytes.
    bytes: int = 0
    # True when the fetch stopped at a limit — `max_result_bytes` or
    # `max_pages` — so the data behind this handle is a prefix, not the whole
    # answer. Consumers surface it; nothing silently treats a truncated read
    # as complete.
    truncated: bool = False
    # Where it came from, for logs, the run UI and an approver's context.
    source_id: str = ""
    operation: str = ""
    # First few items, size-capped. What an `llm` step is shown and what the
    # run UI renders.
    preview: list[Any] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}

    def to_state(self) -> dict[str, Any]:
        """The dict form that goes into workflow state."""
        return self.model_dump(by_alias=True, mode="json")

    def summary(self) -> str:
        """One line naming what this points at, for a prompt or a log.

        Used wherever a handle would otherwise be interpolated into a string
        template: rendering the payload there is exactly the mistake spilling
        exists to prevent, so the template gets this instead.
        """
        what = f"{self.items} items" if self.shape == "list" else "1 document"
        size = _human_bytes(self.bytes)
        tail = ", truncated (incomplete)" if self.truncated else ""
        origin = f"{self.source_id}.{self.operation}".strip(".")
        return (
            f"<spilled result{f' from {origin}' if origin else ''}: "
            f"{what}, {size}{tail}; not inlined — read it in chunks>"
        )


def is_data_ref(value: Any) -> bool:
    """True when *value* is the state form of a :class:`DataRef`."""
    return isinstance(value, dict) and bool(value.get(STREAM_MARKER)) and "id" in value


def as_data_ref(value: Any) -> DataRef | None:
    """Parse the state form back into a handle; ``None`` when it is not one."""
    if isinstance(value, DataRef):
        return value
    if not is_data_ref(value):
        return None
    try:
        return DataRef.model_validate(value)
    except Exception:  # noqa: BLE001 — a malformed handle is simply not one
        return None


def find_data_refs(value: Any) -> dict[str, DataRef]:
    """Every handle reachable in *value*, keyed by the state key holding it.

    Only the top level of a state dict is scanned: a handle is what a
    ``data_source`` step returns under its ``output_key``, so that is where it
    lives.  Walking arbitrarily deep would mean treating a nested user dict
    that happens to carry the marker as a handle.
    """
    if not isinstance(value, dict):
        return {}
    found: dict[str, DataRef] = {}
    for key, item in value.items():
        handle = as_data_ref(item)
        if handle is not None:
            found[key] = handle
    return found


def shrink_preview(items: list[Any]) -> list[Any]:
    """Cap a preview so a handle stays small whatever the items look like."""
    from app.infrastructure.datasources.try_run import shrink_sample

    return [
        shrink_sample(item, list_limit=PREVIEW_ITEMS, string_cap=PREVIEW_STRING_CAP)
        for item in items[:PREVIEW_ITEMS]
    ]


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{count} B"
