"""``combined_datasource``: one step that fetches and filters through several parts.

Why
---
Acquiring a data set often means: fetch A, filter it, fetch B for what is left,
filter again, look up one record per remaining id.  Without a loop construct a
workflow did that with one step per call -- 80 copy-pasted "slot" steps in one
production workflow -- and every one of those outputs went into the run
document.  A ``combined_datasource`` step is that whole pipeline as one step:

.. code-block:: yaml

    - id: invoice_candidates
      type: combined_datasource
      output_key: invoice_candidates
      output: get_classification          # optional; default = last part
      parts:
        - {id: fetch_at, type: data_source, source: crm-projects,
           operation: list_projects_operations, params: {status: [10]}}
        - {id: sel_at, type: python, script_id: invoice-candidates-select, stream: fetch_at}
        - {id: cand, type: python, code: "output = {'project_ids': parts['sel_at']['ids']}"}
        - {id: get_classification, type: data_source, source: crm-order-details,
           operation: get_order_details, params: {id: "{cand[project_ids]}"}}

Rules
-----
* Parts run strictly in the order they are listed.
* A part sees the workflow state plus every *earlier* part's output, by part
  id: ``{cand[project_ids]}`` in a data source part's params; ``parts["sel_at"]``
  in a python part (``records()`` reads the part named by ``stream:``).  A data
  source output that a python part names in ``parts[...]`` (other than its
  ``stream``) is loaded for it as a list, so one script can combine several.
* A data source param that resolves to a list where the operation does not
  declare that param ``array`` is a *loop*: one call per element, bounded
  concurrency, results collected into one stream of
  ``{"<param>": element, "result": <response>}`` records.  Per-element errors
  are collected in the trace and are not fatal unless the part sets
  ``fail_on_error: true``.
* ``for_each: "{part[field]}"`` on a data source part loops over a list of
  *param sets* instead: each element is a mapping merged over ``params`` for
  that call, so one call can differ in several params at once (one POST body
  per element).  Records are ``{"params": element, "result": <response>}``.
  ``for_each`` excludes a param-driven loop in the same part.
* A part may itself be a ``combined_datasource`` (its own ordered ``parts``);
  its output is its chosen part's output.
* Every part's output is written to the run's folder in the stream store
  (``runs/<run_id>/<step_id>.<part path>.md``) when step outputs are stored;
  only the chosen output becomes the step's ``output_key``.  The run document
  gets a small per-part trace: status, timing, record counts, errors.
* Destructive operations are refused: this is a data *acquisition* step, and a
  deletion goes through a ``data_source`` step and its approval gate.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from app.domain.models.datastream import as_data_ref

PART_TYPES = ("data_source", "python", "combined_datasource")
MAX_DEPTH = 6
_TRACE_ERRORS = 10

_TEMPLATE_HEAD_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)")
_PARTS_ACCESS_RE = re.compile(
    r"""parts\s*(?:\[\s*['"]([A-Za-z_][A-Za-z0-9_\-]*)['"]\s*\]|\.get\(\s*['"]([A-Za-z_][A-Za-z0-9_\-]*)['"])"""
)
_WHOLE_REF_RE = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class CombinedPartError(RuntimeError):
    """A part failed; carries the part path for the trace and the error."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"part '{path}': {message}")
        self.path = path


# ---------------------------------------------------------------------------
# Validation (save time)
# ---------------------------------------------------------------------------

def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def template_heads(value: Any) -> set[str]:
    """Names referenced as ``{name...}`` anywhere in *value*."""
    heads: set[str] = set()
    for text in _strings(value):
        heads.update(_TEMPLATE_HEAD_RE.findall(text))
    return heads


def code_part_refs(code: str) -> set[str]:
    """Part ids a python part reads as ``parts["id"]`` / ``parts.get("id")``."""
    return {a or b for a, b in _PARTS_ACCESS_RE.findall(code or "")}


def output_part_id(spec: dict[str, Any]) -> str | None:
    parts = spec.get("parts") or []
    chosen = spec.get("output")
    if chosen:
        return str(chosen)
    last = parts[-1] if parts else None
    return str(last.get("id")) if isinstance(last, dict) and last.get("id") else None


def _validate_parts(
    where: str, parts: Any, outer: set[str], errors: list[str],
    depth: int, seen: set[int],
) -> None:
    if id(parts) in seen:
        errors.append(f"{where}: nesting cycle (a parts list contains itself)")
        return
    if depth > MAX_DEPTH:
        errors.append(f"{where}: nested deeper than {MAX_DEPTH} levels")
        return
    if not isinstance(parts, list) or not parts:
        errors.append(f"{where}: `parts` must be a non-empty list")
        return
    seen = seen | {id(parts)}
    level_ids = [p.get("id") for p in parts if isinstance(p, dict)]
    earlier: set[str] = set()
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            errors.append(f"{where}: part {index} is not a mapping")
            continue
        pid = part.get("id")
        here = f"{where} part '{pid}'" if pid else f"{where} part {index}"
        if not pid or not isinstance(pid, str):
            errors.append(f"{here}: missing `id`")
            continue
        if pid in earlier or pid in outer:
            errors.append(f"{here}: duplicate part id '{pid}'")
        ptype = part.get("type")
        if ptype not in PART_TYPES:
            errors.append(
                f"{here}: unknown part type {ptype!r} (expected one of {', '.join(PART_TYPES)})"
            )
            earlier.add(pid)
            continue
        visible = outer | earlier
        later = {i for i in level_ids[index + 1:] if isinstance(i, str)}

        def _check_ref(name: str, what: str) -> None:
            if name == pid:
                errors.append(f"{here}: {what} references itself")
            elif name in later and name not in visible:
                errors.append(
                    f"{here}: {what} references part '{name}', which runs later "
                    f"(forward reference) -- parts run in the order listed"
                )

        if ptype == "data_source":
            if not part.get("source"):
                errors.append(f"{here}: missing `source`")
            if not part.get("operation"):
                errors.append(f"{here}: missing `operation`")
            params = part.get("params") or {}
            if not isinstance(params, dict):
                errors.append(f"{here}: `params` must be a mapping")
            for head in template_heads(params):
                _check_ref(head, "params template")
            for_each = part.get("for_each")
            if for_each is not None:
                if not isinstance(for_each, (str, list)):
                    errors.append(f"{here}: `for_each` must be a template or a list of mappings")
                for head in template_heads(for_each):
                    _check_ref(head, "for_each template")
        elif ptype == "python":
            if not part.get("script_id") and not part.get("code"):
                errors.append(f"{here}: needs `script_id` or `code`")
            stream = part.get("stream")
            if stream:
                if stream not in visible:
                    kind = "a later part" if stream in later else "no earlier part"
                    errors.append(f"{here}: `stream: {stream}` names {kind}")
            for name in code_part_refs(part.get("code") or ""):
                if name not in visible:
                    if name in later or name == pid:
                        _check_ref(name, "code")
                    else:
                        errors.append(f"{here}: code reads parts['{name}'], which is no earlier part")
        else:  # nested combined_datasource
            _validate_parts(f"{here}", part.get("parts"), visible, errors, depth + 1, seen)
            nested = part.get("parts") if isinstance(part.get("parts"), list) else []
            nested_ids = {p.get("id") for p in nested if isinstance(p, dict)}
            if part.get("output") and part["output"] not in nested_ids:
                errors.append(f"{here}: `output: {part['output']}` names no part of it")
        earlier.add(pid)


def validate_combined_steps(steps: Any) -> list[str]:
    """Structural problems of every ``combined_datasource`` step in *steps*."""
    errors: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict) or step.get("type") != "combined_datasource":
            continue
        where = f"combined_datasource step '{step.get('id')}'"
        _validate_parts(where, step.get("parts"), set(), errors, 1, set())
        parts = step.get("parts") if isinstance(step.get("parts"), list) else []
        ids = {p.get("id") for p in parts if isinstance(p, dict)}
        if step.get("output") and step["output"] not in ids:
            errors.append(f"{where}: `output: {step['output']}` names no part of this step")
    return errors


def iter_parts(parts: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    """Every part, depth first, with its dotted path."""
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        path = f"{prefix}{part.get('id')}"
        yield path, part
        if part.get("type") == "combined_datasource":
            yield from iter_parts(part.get("parts"), f"{path}.")


async def validate_combined_references(
    steps: Any, *, data_source_backend: Any = None, script_backend: Any = None,
) -> list[str]:
    """Missing sources / operations / scripts, and destructive operations."""
    errors: list[str] = []
    for step in steps or []:
        if not isinstance(step, dict) or step.get("type") != "combined_datasource":
            continue
        for path, part in iter_parts(step.get("parts")):
            here = f"combined_datasource step '{step.get('id')}' part '{path}'"
            if part.get("type") == "data_source" and data_source_backend is not None:
                source_id, op_name = part.get("source"), part.get("operation")
                if not source_id or not op_name:
                    continue
                source = await data_source_backend.get(source_id)
                if source is None:
                    errors.append(f"{here}: data source '{source_id}' not found")
                    continue
                op = source.get_operation(op_name)
                bindings = {getattr(b, "name", None) for b in (getattr(source, "sheet_bindings", None) or [])}
                if op is None and op_name not in bindings:
                    errors.append(f"{here}: data source '{source_id}' has no operation '{op_name}'")
                    continue
                if op is not None:
                    from app.infrastructure.datasources.destructive import is_destructive
                    if is_destructive(op, source):
                        errors.append(
                            f"{here}: operation '{op_name}' is destructive; combined_datasource "
                            f"only acquires data -- use a data_source step (it has the approval gate)"
                        )
            elif part.get("type") == "python" and part.get("script_id") and script_backend is not None:
                if await script_backend.get(part["script_id"]) is None:
                    errors.append(f"{here}: script '{part['script_id']}' not found in the library")
    return errors


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _record_count(value: Any) -> int | None:
    ref = as_data_ref(value)
    if ref is not None:
        return ref.items if ref.shape == "list" else 1
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("project_ids", "ids", "items", "rows", "records"):
            if isinstance(value.get(key), list):
                return len(value[key])
    return None


class CombinedExecutor:
    """Runs one ``combined_datasource`` step against a :class:`YamlGraphRunner`."""

    # One element's response read back into the loop's aggregate. Matches the
    # executor's ceiling for an intermediate document.
    ELEMENT_MAX_BYTES = 8 * 1024 * 1024

    def __init__(self, runner: Any, step: dict[str, Any], *, run_id: str, iteration: int) -> None:
        self.runner = runner
        self.step = step
        self.run_id = run_id
        self.iteration = iteration

    # -- entry ----------------------------------------------------------
    async def run(self, state: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """``(output, trace)``; raises :class:`CombinedPartError` with the trace attached."""
        trace: list[dict[str, Any]] = []
        try:
            output = await self._run_level(self.step, state, {}, trace, prefix="")
        except CombinedPartError as exc:
            exc.trace = {"parts": trace, "output_part": output_part_id(self.step)}  # type: ignore[attr-defined]
            raise
        return output, {"parts": trace, "output_part": output_part_id(self.step)}

    async def _run_level(
        self, spec: dict[str, Any], state: dict[str, Any], outer: dict[str, Any],
        trace: list[dict[str, Any]], *, prefix: str,
    ) -> Any:
        outputs: dict[str, Any] = dict(outer)
        level: dict[str, Any] = {}
        parts = spec.get("parts") or []
        for part in parts:
            pid = str(part["id"])
            path = f"{prefix}{pid}"
            entry: dict[str, Any] = {
                "id": pid, "path": path, "type": part.get("type"), "status": "running",
                "started_at": _now_iso(), "duration_ms": None, "records": None,
                "calls": 0, "error_count": 0, "errors": [],
            }
            trace.append(entry)
            started = time.monotonic()
            try:
                if part["type"] == "data_source":
                    value = await self._data_source(part, path, state, outputs, entry)
                elif part["type"] == "python":
                    value = await self._python(part, path, state, outputs, entry)
                else:
                    nested: list[dict[str, Any]] = []
                    entry["parts"] = nested
                    entry["output_part"] = output_part_id(part)
                    value = await self._run_level(part, state, outputs, nested, prefix=f"{path}.")
            except CombinedPartError as exc:
                entry["status"] = "failed"
                if not entry["errors"]:
                    entry["errors"] = [str(exc)]
                    entry["error_count"] = 1
                entry["duration_ms"] = int((time.monotonic() - started) * 1000)
                self._mark_rest_pending(parts[parts.index(part) + 1:], trace, prefix)
                raise
            except Exception as exc:  # noqa: BLE001 — attributed to the part
                entry["status"] = "failed"
                entry["errors"] = [f"{type(exc).__name__}: {exc}"]
                entry["error_count"] = 1
                entry["duration_ms"] = int((time.monotonic() - started) * 1000)
                self._mark_rest_pending(parts[parts.index(part) + 1:], trace, prefix)
                raise CombinedPartError(path, f"{type(exc).__name__}: {exc}") from exc
            entry["duration_ms"] = int((time.monotonic() - started) * 1000)
            entry["status"] = "finished"
            entry["records"] = _record_count(value)
            entry["output"] = await self._store(path, part.get("type", ""), pid, value)
            outputs[pid] = value
            level[pid] = value
        chosen = output_part_id(spec)
        return level.get(chosen) if chosen else None

    @staticmethod
    def _mark_rest_pending(rest: list[Any], trace: list[dict[str, Any]], prefix: str) -> None:
        for part in rest:
            if isinstance(part, dict):
                trace.append({"id": part.get("id"), "path": f"{prefix}{part.get('id')}",
                              "type": part.get("type"), "status": "skipped"})

    async def _store(self, path: str, ptype: str, pid: str, value: Any) -> Any:
        """Write the part output to the run's folder; the trace keeps the ref."""
        handle = as_data_ref(value)
        if handle is not None:
            return {"__stream__": 1, "id": handle.id, "items": handle.items,
                    "bytes": handle.bytes, "truncated": handle.truncated}
        store = getattr(self.runner, "_step_output_store", None)
        if store is None or not self.run_id:
            return None
        written = await store.offload(
            run_id=self.run_id, step_id=f"{self.step['id']}.{path}", step_type=ptype,
            iteration=self.iteration, update={pid: value},
        )
        out = written.get(pid)
        # Small values stay inline in the store's update; the trace only keeps refs.
        from app.infrastructure.orchestration.step_outputs import is_output_ref
        return out if is_output_ref(out) else None

    # -- data source parts ---------------------------------------------
    async def _data_source(
        self, part: dict[str, Any], path: str, state: dict[str, Any],
        outputs: dict[str, Any], entry: dict[str, Any],
    ) -> Any:
        runner = self.runner
        if runner._data_source_backend is None or runner._data_source_executor is None:
            raise CombinedPartError(path, "data source backend/executor not configured")
        source = await runner._data_source_backend.get(part.get("source", ""))
        if source is None:
            raise CombinedPartError(path, f"data source '{part.get('source')}' not found")
        operation = part.get("operation", "")
        op = source.get_operation(operation)
        if op is not None:
            from app.infrastructure.datasources.destructive import is_destructive
            if is_destructive(op, source):
                raise CombinedPartError(
                    path, f"operation '{operation}' is destructive; use a data_source step"
                )
        ctx = {**state, **outputs}
        raw_params = part.get("params") or {}
        params = runner._render_deep(raw_params, ctx)
        from app.infrastructure.orchestration.yaml_graph import _positive_int
        limit = _positive_int(part.get("limit"))
        array_params = {p.name for p in (op.params if op is not None else []) if p.type == "array"}

        if part.get("for_each") is not None:
            raw = part["for_each"]
            match = _WHOLE_REF_RE.match(raw) if isinstance(raw, str) else None
            handle = as_data_ref(ctx.get(match.group(1))) if match else None
            elements = handle if handle is not None else runner._render_deep(raw, ctx)
            if handle is None and not isinstance(elements, list):
                raise CombinedPartError(
                    path, f"`for_each` resolved to {type(elements).__name__}, expected a list of mappings"
                )
            looped = [n for n, v in params.items() if isinstance(v, list) and n not in array_params]
            if looped:
                raise CombinedPartError(
                    path, f"`for_each` and list param {looped[0]!r} both drive a loop; use one"
                )
            entry["loop_param"] = "for_each"
            return await self._loop(
                part, path, source, operation, params, None, elements, entry, limit,
            )

        # Loop detection: a list value for a param the operation does not take
        # as an array -- or a whole `{part}` placeholder naming a list stream.
        loop_name: str | None = None
        loop_source: Any = None
        for name, value in params.items():
            if isinstance(value, list) and name not in array_params:
                loop_name, loop_source = name, value
            else:
                template = raw_params.get(name)
                match = _WHOLE_REF_RE.match(template) if isinstance(template, str) else None
                handle = as_data_ref(ctx.get(match.group(1))) if match else None
                if handle is not None and handle.shape == "list" and name not in array_params:
                    loop_name, loop_source = name, handle
            if loop_name is not None:
                others = [
                    n for n, v in params.items()
                    if n != loop_name and isinstance(v, list) and n not in array_params
                ]
                if others:
                    raise CombinedPartError(
                        path, f"params {loop_name!r} and {others[0]!r} are both lists; "
                        f"only one param may drive a loop"
                    )
                break

        executor = runner._data_source_executor
        if loop_name is None:
            entry["calls"] = 1
            result = await executor.execute(source, operation, params, limit=limit)
            return await self._maybe_ram(part, result, path)
        entry["loop_param"] = loop_name
        return await self._loop(part, path, source, operation, params, loop_name, loop_source, entry, limit)

    async def _maybe_ram(self, part: dict[str, Any], result: Any, path: str) -> Any:
        ref = as_data_ref(result)
        if ref is not None and (part.get("result_mode") or "auto").lower() == "ram":
            return await self.runner._load_stream(ref, path)
        return result

    async def _loop(
        self, part: dict[str, Any], path: str, source: Any, operation: str,
        params: dict[str, Any], name: str | None, elements: Any, entry: dict[str, Any],
        limit: int | None,
    ) -> Any:
        # ``name`` is the looping param; ``None`` means a ``for_each`` loop whose
        # elements are whole param sets merged over ``params``.
        runner = self.runner
        executor = runner._data_source_executor
        store = runner._require_stream_store(path)
        concurrency = max(1, int(part.get("concurrency") or getattr(executor, "_fanout_concurrency", 5) or 5))
        semaphore = asyncio.Semaphore(concurrency)
        fail_on_error = bool(part.get("fail_on_error", False))
        include_errors = bool(part.get("include_errors", False))
        errors: list[str] = []
        error_count = 0
        calls = 0

        async def _one(element: Any) -> dict[str, Any] | None:
            nonlocal error_count, calls
            async with semaphore:
                calls += 1
                key = name or "params"
                try:
                    if name is None:
                        if not isinstance(element, dict):
                            raise TypeError(
                                f"for_each element is {type(element).__name__}, expected a mapping"
                            )
                        call_params = {**params, **element}
                    else:
                        call_params = {**params, name: element}
                    result = await executor.execute(source, operation, call_params, limit=limit)
                    handle = as_data_ref(result)
                    if handle is not None:
                        result = await store.read_all(handle, max_bytes=self.ELEMENT_MAX_BYTES)
                    return {key: element, "result": result}
                except Exception as exc:  # noqa: BLE001 — collected, see fail_on_error
                    error_count += 1
                    message = f"{key}={element!r}: {type(exc).__name__}: {exc}"
                    if len(errors) < _TRACE_ERRORS:
                        errors.append(message)
                    if fail_on_error:
                        raise CombinedPartError(path, message) from exc
                    return {key: element, "error": str(exc)} if include_errors else None

        from app.infrastructure.datasources.datastream import StreamBuilder
        builder = StreamBuilder(
            store=store, max_result_bytes=source.max_result_bytes,
            source_id=source.id, operation=operation,
        )
        window = concurrency * 4

        async def _windows():
            handle = as_data_ref(elements)
            if handle is not None:
                async for chunk in store.chunks(handle, size=window):
                    yield chunk
            else:
                items = list(elements)
                for i in range(0, len(items), window):
                    yield items[i:i + window]

        try:
            async for chunk in _windows():
                results = await asyncio.gather(*(_one(e) for e in chunk))
                kept = [r for r in results if r is not None]
                if kept:
                    await builder.add(kept)
                if builder.full:
                    entry.setdefault("warnings", []).append("stopped at max_result_bytes")
                    break
            result = await builder.finish()
        except BaseException:
            await builder.abort()
            entry["calls"], entry["error_count"], entry["errors"] = calls, error_count, errors
            raise
        entry["calls"], entry["error_count"], entry["errors"] = calls, error_count, errors
        return await self._maybe_ram(part, result, path)

    # -- python parts ---------------------------------------------------
    async def _python(
        self, part: dict[str, Any], path: str, state: dict[str, Any],
        outputs: dict[str, Any], entry: dict[str, Any],
    ) -> Any:
        runner = self.runner
        code = await runner._resolve_script_code(part)
        stream_id = part.get("stream")
        ref = None
        if stream_id:
            ref = as_data_ref(outputs.get(stream_id))
            if ref is None:
                raise CombinedPartError(
                    path, f"`stream: {stream_id}` does not name an earlier data source part output"
                )
        read = code_part_refs(code)
        visible: dict[str, Any] = {}
        limit = int(getattr(runner._stream_conf(), "stream_read_all_max_bytes", 0) or 0)
        budget = limit
        for pid, value in outputs.items():
            handle = as_data_ref(value)
            if handle is not None and pid in read and pid != stream_id:
                # Named by the script: hand it the records, within the ram cap.
                if limit and handle.bytes > budget:
                    raise CombinedPartError(
                        path, f"parts['{pid}'] is {handle.bytes} bytes; loading it would pass "
                        f"STREAM_READ_ALL_MAX_BYTES -- read it with `stream: {pid}` instead"
                    )
                budget -= handle.bytes
                value = await runner._require_stream_store(path).read_all(handle, max_bytes=limit)
            visible[pid] = value
        script_state = {**state, "parts": visible}
        prelude = "parts = state.get('parts') or {}\n"
        entry["calls"] = 1
        result = await runner._run_python({**part, "id": f"{self.step['id']}.{path}"},
                                          prelude + code, script_state, ref)
        if isinstance(result, dict) and set(result) == {"error"}:
            raise CombinedPartError(path, f"script returned error: {result['error']}")
        return result


async def combined_step_errors(
    steps: Any, *, data_source_backend: Any = None, script_backend: Any = None,
) -> list[str]:
    """Every save-time problem of the ``combined_datasource`` steps in *steps*.

    Structure first; the backend lookups only run on a structurally valid
    definition, so one typo does not produce a cascade of "not found"s.
    """
    errors = validate_combined_steps(steps)
    if errors:
        return errors
    return await validate_combined_references(
        steps, data_source_backend=data_source_backend, script_backend=script_backend,
    )
