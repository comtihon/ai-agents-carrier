"""Step outputs live in object storage; workflow state carries references.

Why
---
A run document used to hold every step's output three times over: in
``state``, in ``step_outputs`` and -- worst -- in ``step_inputs``, which kept a
*full state snapshot* per step.  A workflow with 170 steps and a 200 KB state
therefore grew to ``steps x state`` and crossed Mongo's 16 MB document limit;
the failed write (``DocumentTooLarge``) left the run resumable-on-startup, and
the backend crash-looped on it.

The rule
--------
* Every step's output is written to the stream store (GCS in production) as a
  markdown file under the run's folder: ``runs/<run_id>/<step_id>[-<n>].md``.
  A short header names the step, its type and when it ran; the values follow in
  one fenced ``json`` block, so the file is readable by a person *and* parses
  back exactly.
* What goes back into workflow state -- and so into the LangGraph checkpoint,
  ``run.state`` and ``run.step_outputs`` -- is a small reference::

      {"__ref__": true, "uri": "gs://bucket/runs/r1/plan.md", "key": "runs/r1/plan.md",
       "field": "plan", "bytes": 48213, "sha256": "...", "preview": "...", "kind": "dict"}

  Values whose JSON encoding is at most ``inline_max_bytes`` (default 256 B) stay
  inline as well: they are no bigger than their own reference, and keeping them
  is what lets ``when`` guards, route conditions and Slack templates outside the
  node wrapper keep reading flags and ids directly.  The file still gets them.
* A step never fetches anything itself.  Before a node runs, the engine
  resolves the references the step needs (see :func:`referenced_keys`) and
  hands the node a hydrated copy of state; templates, ``when`` guards, switch
  routes and a python step's ``state`` all see plain values.

Old run documents carry inline values everywhere; hydration leaves a value that
is not a reference untouched, so they load and resume unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from app.domain.models.datastream import as_data_ref

logger = logging.getLogger(__name__)

REF_MARKER = "__ref__"
PREVIEW_CHARS = 200

# Never offloaded: run-level bookkeeping every surface reads straight out of
# state (``error`` is the run's failure message, ``request`` its input).
_NEVER_OFFLOAD = frozenset({"error", "request"})

# Step types whose code can read *any* state key, so "which keys does it need"
# cannot be answered from the definition: everything is resolved for them,
# under ``hydrate_max_bytes``.
ALL_STATE_TYPES = frozenset({
    "python", "langgraph-agent", "claude-agent", "dynamic", "dynamic_gate",
})

# Definition fields that name steps or describe layout, never state. A token
# found only here is not a reference to a state key (an output_key defaults to
# the step id, so `next: fetch` would otherwise hydrate `fetch` for nothing).
_STRUCTURAL_FIELDS = frozenset({
    "id", "type", "name", "description", "next", "targets", "output_key",
    "position", "ui", "layout", "sandbox", "sandbox_runtime", "sandbox_image",
    "timeout_seconds", "max_loops", "result_mode", "limit",
})

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]")


class StepOutputUnavailable(RuntimeError):
    """A referenced step output could not be read back."""


class StepInputTooLarge(RuntimeError):
    """Resolving a step's inputs would load more than the configured cap."""


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def is_output_ref(value: Any) -> bool:
    """True when *value* is a stored-step-output reference."""
    return (
        isinstance(value, dict)
        and value.get(REF_MARKER) is True
        and isinstance(value.get("key"), str)
        and isinstance(value.get("field"), str)
    )


def find_output_refs(state: Any) -> dict[str, dict[str, Any]]:
    """Top-level state keys holding a reference (same scope as DataRefs)."""
    if not isinstance(state, dict):
        return {}
    return {k: v for k, v in state.items() if is_output_ref(v)}


def offloadable(key: str, value: Any) -> bool:
    """Whether *key* of a node update is a step output to be stored."""
    if not isinstance(key, str) or key.startswith("_") or key in _NEVER_OFFLOAD:
        return False
    # Already a reference to stored bytes: a DataRef is the stream store's own
    # handle, and an output ref passed through is already in a file.
    if is_output_ref(value) or as_data_ref(value) is not None:
        return False
    return True


def _segment(text: str) -> str:
    cleaned = _SEGMENT_RE.sub("_", str(text or "")).strip(".")
    return (cleaned or "x")[:120]


def object_key(run_id: str, step_id: str, iteration: int = 1) -> str:
    """``runs/<run_id>/<step_id>[-<n>].md`` -- first execution has no suffix."""
    suffix = f"-{iteration}" if iteration and iteration > 1 else ""
    return f"runs/{_segment(run_id)}/{_segment(step_id)}{suffix}.md"


def run_prefix(run_id: str) -> str:
    return f"runs/{_segment(run_id)}/"


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _encode(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def make_ref(*, uri: str, key: str, field: str, value: Any, text: str) -> dict[str, Any]:
    raw = text.encode("utf-8")
    ref: dict[str, Any] = {
        REF_MARKER: True,
        "uri": uri,
        "key": key,
        "field": field,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "kind": _kind(value),
        "preview": text[:PREVIEW_CHARS],
    }
    if isinstance(value, (list, tuple, dict)):
        ref["items"] = len(value)
    return ref


# ---------------------------------------------------------------------------
# The markdown file
# ---------------------------------------------------------------------------

_FENCE_OPEN = "```json\n"
_FENCE_CLOSE = "\n```"


def render_markdown(
    *, run_id: str, step_id: str, step_type: str, iteration: int,
    values: dict[str, Any], written_at: datetime | None = None,
) -> bytes:
    """A short header, then every value in one fenced json block."""
    ts = (written_at or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    body = json.dumps(values, default=str, ensure_ascii=False, indent=2)
    lines = [
        f"# Step output: {step_id}",
        "",
        f"- step: `{step_id}`",
        f"- type: `{step_type}`",
        f"- run: `{run_id}`",
        f"- iteration: {iteration}",
        f"- written_at: {ts}",
        f"- keys: {', '.join(f'`{k}`' for k in values)}",
        "",
        _FENCE_OPEN + body + _FENCE_CLOSE,
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def parse_markdown(data: bytes | str) -> dict[str, Any]:
    """The ``{key: value}`` mapping stored by :func:`render_markdown`.

    The json block is the last thing in the file, so it is located from the
    first opening fence to the *last* closing one: a value containing three
    backticks cannot end it early.
    """
    text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
    start = text.find(_FENCE_OPEN)
    end = text.rfind(_FENCE_CLOSE)
    if start < 0 or end <= start:
        raise StepOutputUnavailable("stored step output has no ```json block")
    payload = json.loads(text[start + len(_FENCE_OPEN):end])
    if not isinstance(payload, dict):
        raise StepOutputUnavailable("stored step output is not a mapping")
    return payload


# ---------------------------------------------------------------------------
# What a step needs
# ---------------------------------------------------------------------------

def _strings(value: Any, *, top: bool = False) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            if top and k in _STRUCTURAL_FIELDS:
                continue
            if k == "next" or k == "targets":
                continue
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _has_python_part(parts: Any) -> bool:
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "python":
            return True
        if part.get("type") == "combined_datasource" and _has_python_part(part.get("parts")):
            return True
    return False


def referenced_keys(step: dict[str, Any]) -> set[str] | None:
    """State keys *step* may read, or ``None`` for "any of them".

    Deliberately over-inclusive: every identifier-shaped token in the step's
    non-structural config counts (``{plan[items]}``, ``when: approved and
    score > 3``, ``from: state.projects``).  Hydrating a key the step did not
    need costs one read; missing one it did need would hand a template a
    reference instead of a value.
    """
    step_type = step.get("type")
    if step_type in ALL_STATE_TYPES:
        return None
    if step_type == "combined_datasource" and _has_python_part(step.get("parts")):
        return None
    tokens: set[str] = set()
    for text in _strings(step, top=True):
        tokens.update(_TOKEN_RE.findall(text))
    return tokens


def condition_keys(conditions: Iterable[Any]) -> set[str]:
    tokens: set[str] = set()
    for cond in conditions:
        if cond is not None:
            tokens.update(_TOKEN_RE.findall(str(cond)))
    return tokens


# ---------------------------------------------------------------------------
# step_inputs: what the run document records about a step's input
# ---------------------------------------------------------------------------

STEP_INPUT_MAX_KEYS = 30
STEP_INPUT_VALUE_MAX_BYTES = 1024


def _compact_value(value: Any) -> Any:
    """A value small enough for the run document, or a stub describing it."""
    if is_output_ref(value):
        return value
    handle = as_data_ref(value)
    if handle is not None:
        return {
            "__stream__": 1, "id": handle.id, "shape": handle.shape,
            "items": handle.items, "bytes": handle.bytes,
            "truncated": handle.truncated, "source_id": handle.source_id,
            "operation": handle.operation,
        }
    text = _encode(value)
    if len(text.encode("utf-8")) <= STEP_INPUT_VALUE_MAX_BYTES:
        return value
    return {
        "__truncated__": True, "kind": _kind(value),
        "bytes": len(text.encode("utf-8")), "preview": text[:PREVIEW_CHARS],
    }


# Bookkeeping a node update carries that the run document must not copy per
# step: the loop guard returns the *whole* visit-count map on every update, so
# keeping it in step_outputs made that field grow steps x steps.
_STEP_OUTPUT_DROP = frozenset({"_visit_counts"})


def step_output_record(output: Any) -> Any:
    """What ``run.step_outputs[step]`` keeps of a node update."""
    if not isinstance(output, dict):
        return output
    return {k: v for k, v in output.items() if k not in _STEP_OUTPUT_DROP}


def step_input_snapshot(step: dict[str, Any] | None, state: dict[str, Any]) -> dict[str, Any]:
    """The part of *state* a step read, compacted -- never a full snapshot.

    ``step_inputs`` used to be ``dict(state)`` per step, which is what made a
    run document ``steps x state``.  It now holds only the keys the step's
    definition references (for code-driven steps: the keys its inline code
    names), each capped at ``STEP_INPUT_VALUE_MAX_BYTES`` -- references stay
    references -- and at most ``STEP_INPUT_MAX_KEYS`` of them.
    """
    if not isinstance(state, dict):
        return {}
    step = step or {}
    tokens: set[str] = set()
    for text in _strings(step, top=True):
        tokens.update(_TOKEN_RE.findall(text))
    keys = [k for k in state if k in tokens and not k.startswith("__")]
    shown = keys[:STEP_INPUT_MAX_KEYS]
    snapshot: dict[str, Any] = {k: _compact_value(state[k]) for k in shown}
    if len(keys) > len(shown):
        snapshot["__more_keys__"] = keys[len(shown):]
    return snapshot


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

class StepOutputStore:
    """Writes step outputs to the stream store and reads them back."""

    def __init__(
        self,
        store: Any,
        *,
        inline_max_bytes: int = 256,
        hydrate_max_bytes: int = 32 * 1024 * 1024,
        cache_max_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self._store = store
        self.inline_max_bytes = max(0, int(inline_max_bytes))
        self.hydrate_max_bytes = max(0, int(hydrate_max_bytes))
        self._cache_max_bytes = max(0, int(cache_max_bytes))
        # key -> raw bytes; bounded LRU so a long run re-reading the same
        # outputs step after step pays the round trip once.
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_bytes = 0

    @property
    def store(self) -> Any:
        return self._store

    # -- cache ----------------------------------------------------------
    def _cache_put(self, key: str, data: bytes) -> None:
        if not self._cache_max_bytes or len(data) > self._cache_max_bytes:
            return
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_bytes -= len(old)
        self._cache[key] = data
        self._cache_bytes += len(data)
        while self._cache_bytes > self._cache_max_bytes and self._cache:
            _, dropped = self._cache.popitem(last=False)
            self._cache_bytes -= len(dropped)

    async def read_file(self, key: str) -> bytes:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        try:
            data = await self._store.get_object(key)
        except Exception as exc:  # noqa: BLE001 — re-raised with the key named
            raise StepOutputUnavailable(
                f"stored step output '{key}' could not be read: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._cache_put(key, data)
        return data

    # -- write ----------------------------------------------------------
    async def offload(
        self,
        *,
        run_id: str,
        step_id: str,
        step_type: str,
        iteration: int,
        update: Any,
    ) -> Any:
        """Write the outputs in *update* to a file; return it with refs in place."""
        if not isinstance(update, dict) or not update or not run_id:
            return update
        values = {k: v for k, v in update.items() if offloadable(k, v)}
        if not values:
            return update
        key = object_key(run_id, step_id, iteration)
        data = render_markdown(
            run_id=run_id, step_id=step_id, step_type=step_type,
            iteration=iteration, values=values,
        )
        uri = await self._store.put_object(key, data, content_type="text/markdown; charset=utf-8")
        self._cache_put(key, data)
        out = dict(update)
        for field, value in values.items():
            text = _encode(value)
            if len(text.encode("utf-8")) > self.inline_max_bytes:
                out[field] = make_ref(uri=uri, key=key, field=field, value=value, text=text)
        logger.debug(
            "run %s step '%s': %d output(s) -> %s (%d bytes)",
            run_id, step_id, len(values), key, len(data),
        )
        return out

    # -- read -----------------------------------------------------------
    async def resolve(self, ref: dict[str, Any]) -> Any:
        values = parse_markdown(await self.read_file(ref["key"]))
        field = ref["field"]
        if field not in values:
            raise StepOutputUnavailable(
                f"stored step output '{ref['key']}' has no value for '{field}'"
            )
        return values[field]

    async def hydrate(
        self,
        state: Any,
        keys: set[str] | None = None,
        *,
        step_id: str = "",
        max_bytes: int | None = None,
    ) -> Any:
        """*state* with the references named by *keys* (all when ``None``) resolved.

        Returns *state* itself when there is nothing to resolve, so the common
        case allocates nothing.
        """
        if not isinstance(state, dict):
            return state
        targets = {
            k: v for k, v in state.items()
            if is_output_ref(v) and (keys is None or k in keys)
        }
        if not targets:
            return state
        cap = self.hydrate_max_bytes if max_bytes is None else max_bytes
        total = sum(int(v.get("bytes") or 0) for v in targets.values())
        if cap and total > cap:
            biggest = sorted(targets.items(), key=lambda kv: -int(kv[1].get("bytes") or 0))[:5]
            listing = ", ".join(f"{k} ({int(v.get('bytes') or 0)} B)" for k, v in biggest)
            raise StepInputTooLarge(
                f"step '{step_id}' would load {total} bytes of stored step outputs "
                f"into memory, over the {cap}-byte STEP_INPUT_HYDRATE_MAX_BYTES cap. "
                f"Largest: {listing}. Reduce what earlier steps return (e.g. "
                f"return ids, or keep data in a data_source stream and read it "
                f"with `stream:`) or raise the cap."
            )
        files: dict[str, dict[str, Any]] = {}
        out = dict(state)
        for k, ref in targets.items():
            file_key = ref["key"]
            if file_key not in files:
                files[file_key] = parse_markdown(await self.read_file(file_key))
            values = files[file_key]
            field = ref["field"]
            if field not in values:
                raise StepOutputUnavailable(
                    f"stored step output '{file_key}' has no value for '{field}' "
                    f"(needed by step '{step_id}' as state key '{k}')"
                )
            out[k] = values[field]
        return out

    async def delete_run(self, run_id: str) -> int:
        deleter = getattr(self._store, "delete_objects", None)
        if deleter is None:
            return 0
        return await deleter(run_prefix(run_id))


async def resolve_for_display(
    store: StepOutputStore | None,
    value: Any,
    *,
    budget: list[int],
    download_url: Any = None,
) -> Any:
    """Resolve refs in a mapping for an API response, within a byte *budget*.

    *budget* is a one-element list so several calls share one allowance.
    A ref that does not fit (or cannot be read) stays a ref, with a
    ``download_url`` added when *download_url* can build one.
    """
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for k, v in value.items():
        if not is_output_ref(v):
            out[k] = v
            continue
        size = int(v.get("bytes") or 0)
        if store is not None and size <= budget[0]:
            try:
                out[k] = await store.resolve(v)
                budget[0] -= size
                continue
            except Exception:  # noqa: BLE001 — a response must render without it
                logger.debug("could not resolve %s for display", v.get("key"), exc_info=True)
        ref = dict(v)
        if download_url is not None:
            ref["download_url"] = download_url(v)
        out[k] = ref
    return out
