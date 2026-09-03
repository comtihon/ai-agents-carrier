"""The tier-2 lifecycle, once, for every surface that drives it.

REST, the management MCP and the chat agent all offer the same tier-2
operations, and every one of them is a decision about executable code: compile,
answer the ambiguity questions, read the code, edit it by hand, activate it,
recompile it, mark it stale.  Putting that logic here rather than in a route
means the gates cannot differ between surfaces — which is the failure mode
worth designing against, because a caller who finds one lenient surface does
not use the strict one.

What each function guarantees, wherever it is called from:

* **ADMIN to store code.**  ``assert_generated_code_allowed`` on compile, on a
  hand edit and on activation.  Reading, previewing and running are not gated
  here; see ``auth.sandbox_guard``.
* **Every gate, every time.**  Static AST allow-list, seccomp sandbox,
  determinism double-run, output shape, column whitelist — via the single
  ``verify_transform`` entry point, never re-implemented.
* **Nothing is activated by compiling.**  A fresh compile stores inert code.
  Escalating to generated code is authorised by the user in the UI; *turning it
  on* is a second, separate confirmation, and both are explicit.
* **A human edit ends generation.**  ``edited_by_human`` is one-way: once set,
  compile refuses rather than overwriting somebody's fix.
* **A write is never silently recompiled.**  Invalidation marks a write binding
  stale and stops.  A read may be recompiled on request, but not on its own.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.core.config import Settings
from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    validate_operations,
)
from app.domain.models.sheet_binding import (
    BindingValidationError,
    ComputeSpec,
    SheetBinding,
    header_fingerprint,
    validate_bindings,
)
from app.infrastructure.auth.sandbox_guard import assert_generated_code_allowed
from app.infrastructure.datasources.sheet_binding_compile import (
    refresh_binding_operations,
    stamp_compiled,
)
from app.infrastructure.datasources.sheet_binding_runtime import probe_sheet
from app.infrastructure.datasources.sheet_compute import (
    ComputeRuntimeError,
    ComputeValidationError,
    cache_key,
    content_hash,
    script_id_for,
)
from app.infrastructure.datasources.sheet_compute_generate import (
    compile_transform,
    records_for,
    resolved_model,
    rerun_golden,
    verify_transform,
)

logger = logging.getLogger(__name__)


class ComputeServiceError(Exception):
    """A tier-2 lifecycle operation that cannot proceed, with a message for its caller."""


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compute_status(binding: SheetBinding) -> dict[str, Any]:
    """What every surface reports about a binding's tier and code state.

    One function so the REST payload, the MCP tool text and the editor badge
    are describing the same thing.  ``tier`` is an *outcome* here, exactly as it
    is in the UI: derived from whether the binding carries code, never from a
    field a caller set.
    """
    compute = binding.compute
    resolution = binding.resolution
    golden = resolution.golden

    verified_at = golden.verified_at if golden else None
    age_days: float | None = None
    if verified_at is not None:
        delta = datetime.now(timezone.utc) - verified_at
        age_days = round(delta.total_seconds() / 86400, 2)

    status: dict[str, Any] = {
        "name": binding.name,
        "operation": binding.operation,
        "tier": "script" if compute is not None else "binding",
        "generated": compute is not None,
        "instruction": resolution.instruction,
        "answers": dict(resolution.answers),
        "authored_by": resolution.authored_by,
        "model_id": resolution.model_id,
        "edited_by_human": resolution.edited_by_human,
        "compiled_at": resolution.compiled_at.isoformat() if resolution.compiled_at else None,
        "schema_fingerprint": binding.sheet_schema.fingerprint,
    }
    if compute is None:
        return status

    status.update({
        "script_id": compute.script_id,
        "content_hash": compute.content_hash,
        "signature_version": compute.signature_version,
        "output_shape": compute.output_shape,
        "rationale": compute.rationale,
        "activated": compute.activated,
        "stale": compute.stale,
        "stale_reason": compute.stale_reason,
        "golden": {
            "rows": len(golden.input_rows) if golden else 0,
            "input_hash": golden.input_hash if golden else "",
            "output_hash": golden.output_hash if golden else "",
            "verified_at": verified_at.isoformat() if verified_at else None,
            "verified_days_ago": age_days,
        } if golden else None,
    })
    return status


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

async def persist_binding(
    *,
    backend: Any,
    source: DataSourceDefinition,
    binding: SheetBinding,
    script_backend: Any = None,
    publish: Any = None,
) -> DataSourceDefinition:
    """Store one changed binding, recompiling the operation it feeds.

    The same two-step every binding write does: validate the whole set (a
    binding is only legal in the context of its source's other operations),
    then refresh the compiled operations so the runtime and the operation list
    cannot disagree about what the binding does.
    """
    bindings = [binding if b.name == binding.name else b for b in source.bindings]
    if all(b.name != binding.name for b in source.bindings):
        bindings = [*source.bindings, binding]

    try:
        validate_bindings(bindings)
    except BindingValidationError as exc:
        raise ComputeServiceError(str(exc)) from exc

    updated = source.model_copy(update={
        "bindings": [stamp_compiled(b) for b in bindings],
        "operations": refresh_binding_operations(source, bindings),
    })
    try:
        validate_operations(updated)
    except ValueError as exc:
        raise ComputeServiceError(str(exc)) from exc

    saved = await backend.update(updated.id, updated)
    if script_backend is not None:
        from app.infrastructure.datasources.sheet_binding_library import (
            ensure_binding_scripts,
        )
        await ensure_binding_scripts(script_backend)
    if publish is not None:
        await publish()
    return saved


def _require_binding(source: DataSourceDefinition, name: str) -> SheetBinding:
    binding = source.get_binding(name)
    if binding is None:
        raise ComputeServiceError(f"Binding '{name}' not found on '{source.id}'.")
    return binding


def _check_writes_enabled(binding: SheetBinding, settings: Settings) -> None:
    """Refuse to author a tier-2 write while the flag is off.

    Checked at authoring time *and* at run time (see
    ``sheet_binding_runtime._run_pinned``).  Authoring-time alone would let a
    binding be stored now and start working the moment somebody flipped the
    flag for an unrelated reason; run-time alone would let a user build the
    whole thing and only discover at the end that it can never run.
    """
    if binding.operation == "write" and not settings.sheets_compute_writes_enabled:
        raise ComputeServiceError(
            "Generated-code WRITE bindings are disabled on this backend "
            "(SHEETS_COMPUTE_WRITES_ENABLED is off). A generated read is wrong "
            "in its response; a generated write is wrong in somebody's "
            "spreadsheet, so writes are opt-in per deployment. Compose this as "
            "a tier-2 read plus a tier-1 write binding, or ask an operator to "
            "enable the flag."
        )


# ---------------------------------------------------------------------------
# Probe + staleness
# ---------------------------------------------------------------------------

async def _probe(
    source: DataSourceDefinition, executor: Any, binding: SheetBinding
) -> dict[str, Any]:
    if executor is None:
        raise ComputeServiceError("Data source executor not configured.")
    try:
        return await probe_sheet(
            source,
            executor,
            binding.document.file_id,
            binding.document.sheet or None,
            binding.sheet_schema.header_row,
        )
    except Exception as exc:  # noqa: BLE001 — reported, never a 500
        raise ComputeServiceError(f"Could not probe that spreadsheet: {exc}") from exc


def _mark_stale(binding: SheetBinding, reason: str) -> None:
    if binding.compute is not None:
        binding.compute.stale = True
        binding.compute.stale_reason = reason


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

async def compile_compute(
    *,
    source: DataSourceDefinition,
    name: str,
    instruction: str | None,
    answers: dict[str, str] | None,
    settings: Settings,
    executor: Any,
    backend: Any,
    script_backend: Any = None,
    publish: Any = None,
    params: dict[str, Any] | None = None,
    ask: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate (or regenerate) the transform of binding *name*.

    Returns ``{"status": "needs"|"ok"|"cached"|"error", ...}``.  On ``ok`` the
    code is stored **inert**: ``compute.activated`` is false until
    :func:`activate_compute`, because compiling successfully is not the same
    event as a person deciding to run it.

    *force* re-runs the model even when the cache key is unchanged, which is
    what a "Recompile" button means; without it an unchanged request returns
    the stored code and calls no model.
    """
    assert_generated_code_allowed("a generated sheet transform")

    binding = _require_binding(source, name)
    if not settings.sheets_compute_enabled:
        raise ComputeServiceError(
            "Generated sheet transforms are disabled on this backend "
            "(SHEETS_COMPUTE_ENABLED)."
        )
    _check_writes_enabled(binding, settings)

    # A hand-edited transform is somebody's fix. Regenerating over it is the
    # single most annoying thing this feature could do, so it is refused --
    # loudly, and with the way out named.
    if binding.resolution.edited_by_human:
        raise ComputeServiceError(
            f"Binding '{name}' has been edited by hand, so it is no longer "
            "regenerated — a recompile would overwrite that edit. Edit the code "
            "directly, or delete the binding and start again if you want the "
            "model to have another go."
        )

    text = (instruction if instruction is not None else binding.resolution.instruction) or ""
    if not text.strip():
        raise ComputeServiceError(
            "An instruction is required: it is what the transform is generated "
            "from, and what a recompile reproduces it from."
        )
    merged = {**dict(binding.resolution.answers), **(answers or {})}

    # Probe first. The headers the model is shown have to be the sheet's
    # headers *now*, or the compile is against a schema that no longer exists.
    probe = await _probe(source, executor, binding)
    fresh_fingerprint = probe.get("fingerprint") or ""
    drifted = (
        bool(binding.sheet_schema.fingerprint)
        and fresh_fingerprint != binding.sheet_schema.fingerprint
    )
    if drifted and binding.operation == "write":
        # Never silently re-author a write against a changed sheet: the column
        # the binding meant may not be the column of that name any more.
        _mark_stale(binding, "the sheet's header row changed since this binding was saved")
        await persist_binding(
            backend=backend, source=source, binding=binding,
            script_backend=script_backend, publish=publish,
        )
        raise ComputeServiceError(
            f"The header row of '{binding.document.sheet}' has changed since "
            f"binding '{name}' was saved, and it is a write. It has been marked "
            "stale rather than regenerated: check the column mappings against "
            "the new headers, save the binding, then recompile. Headers now: "
            + ", ".join(repr(h) for h in probe.get("headers") or [])
        )

    headers = [str(h) for h in (probe.get("headers") or [])]
    sample_rows = [list(r) for r in (probe.get("sample_rows") or [])]
    if drifted:
        # A read may be re-authored against the new headers on request: the
        # worst case is a wrong number in a response, and the fixture below is
        # re-frozen against the new schema anyway.
        binding.sheet_schema.headers = headers
        binding.sheet_schema.fingerprint = fresh_fingerprint or header_fingerprint(headers)
        binding.sheet_schema.probed_at = datetime.now(timezone.utc)

    _provider, model_id = resolved_model(settings)
    key = cache_key(
        instruction=text,
        answers=merged,
        schema_fingerprint=binding.sheet_schema.fingerprint,
        model_id=model_id,
    )

    # Cache hit: the same request against the same schema and the same model.
    # Re-run the fixture rather than trusting the key -- the key proves the
    # *inputs* did not change, the fixture proves the code still computes what
    # it computed.
    if (
        not force
        and binding.compute is not None
        and binding.compute.cache_key == key
        and not binding.compute.stale
    ):
        try:
            await rerun_golden(binding, settings=settings, params=params)
        except (ComputeValidationError, ComputeRuntimeError) as exc:
            _mark_stale(binding, f"the golden fixture stopped reproducing: {exc}")
            await persist_binding(
                backend=backend, source=source, binding=binding,
                script_backend=script_backend, publish=publish,
            )
            raise ComputeServiceError(str(exc)) from exc
        return {
            "status": "cached",
            "binding": name,
            "compute": compute_status(binding),
            "code": binding.compute.code,
            "needs": [],
        }

    result = await compile_transform(
        binding,
        instruction=text,
        answers=merged,
        sample_rows=sample_rows,
        params=params,
        settings=settings,
        ask=ask,
    )

    if result["status"] == "needs":
        # Questions are not a failure and nothing is stored: the instruction and
        # the answers so far are already on the binding (or supplied by the
        # caller), and the next call folds the new answers in.
        return {
            "status": "needs",
            "binding": name,
            "needs": result["needs"],
            "instruction": text,
            "answers": merged,
        }

    if result["status"] != "ok":
        return {
            "status": "error",
            "binding": name,
            "error": result.get("error") or "compilation failed",
            "code": result.get("code"),
            "attempts": result.get("attempts"),
        }

    compute: ComputeSpec = result["compute"]
    compute.cache_key = key
    # A recompile of an already-activated binding does NOT stay activated: the
    # code changed, so the person who approved the old code has not approved
    # this one. Re-confirmation is the point.
    compute.activated = False

    binding.compute = compute
    binding.resolution.tier = "script"
    binding.resolution.authored_by = "llm"
    binding.resolution.instruction = text
    binding.resolution.answers = merged
    binding.resolution.model_id = result["model_id"]
    binding.resolution.script_id = compute.script_id
    binding.resolution.golden = result["golden"]
    binding.resolution.compiled_at = datetime.now(timezone.utc)

    await persist_binding(
        backend=backend, source=source, binding=binding,
        script_backend=script_backend, publish=publish,
    )
    logger.info(
        "tier-2 binding '%s' compiled (script=%s, model=%s, attempts=%s) — inert "
        "until activated",
        name, compute.script_id, result["model_id"], result.get("attempts"),
    )
    return {
        "status": "ok",
        "binding": name,
        "compute": compute_status(binding),
        "code": compute.code,
        "rationale": compute.rationale,
        "output": result["output"],
        "attempts": result.get("attempts"),
        "needs": [],
    }


# ---------------------------------------------------------------------------
# Hand edit
# ---------------------------------------------------------------------------

async def edit_compute_code(
    *,
    source: DataSourceDefinition,
    name: str,
    code: str,
    settings: Settings,
    backend: Any,
    script_backend: Any = None,
    publish: Any = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replace a binding's transform with a hand-written one.

    Held to **exactly** the gates a generated transform is: the same AST
    allow-list, the same sandbox, the same determinism double-run, the same
    shape and whitelist checks.  A human author is not more trusted here — they
    are differently trusted, and the checks are about what the code does, not
    who wrote it.

    Sets ``edited_by_human``, which permanently stops regeneration, and re-freezes
    the golden fixture against the edited code (the old fixture described the
    model's version, and keeping it would fail on the next re-run for the wrong
    reason).  Activation is cleared: an edit is new code, and the previous
    approval was of the previous code.
    """
    assert_generated_code_allowed("an edited sheet transform")

    binding = _require_binding(source, name)
    if binding.compute is None:
        raise ComputeServiceError(
            f"Binding '{name}' is a tier-1 binding (a form, no code) — there is "
            "no transform to edit. Compile one from an instruction first."
        )
    _check_writes_enabled(binding, settings)

    golden = binding.resolution.golden
    if golden is None or not golden.input_rows:
        raise ComputeServiceError(
            f"Binding '{name}' has no golden fixture rows to verify an edit "
            "against — recompile it first."
        )

    records = records_for(binding, golden.input_rows)
    try:
        output = await verify_transform(
            code, binding, records, dict(params or {}),
            timeout=float(settings.sheets_compute_timeout_seconds),
        )
    except (ComputeValidationError, ComputeRuntimeError) as exc:
        raise ComputeServiceError(str(exc)) from exc

    from app.infrastructure.datasources.sheet_compute import fixture_hash

    binding.compute.code = code
    binding.compute.script_id = script_id_for(code)
    binding.compute.content_hash = content_hash(code)
    binding.compute.activated = False
    binding.compute.stale = False
    binding.compute.stale_reason = ""
    binding.resolution.edited_by_human = True
    binding.resolution.script_id = binding.compute.script_id
    golden.output = output
    golden.output_hash = fixture_hash(output)
    golden.verified_at = datetime.now(timezone.utc)
    binding.resolution.compiled_at = datetime.now(timezone.utc)

    await persist_binding(
        backend=backend, source=source, binding=binding,
        script_backend=script_backend, publish=publish,
    )
    logger.info(
        "tier-2 binding '%s' edited by hand (script=%s) — regeneration disabled",
        name, binding.compute.script_id,
    )
    return {
        "status": "ok",
        "binding": name,
        "compute": compute_status(binding),
        "code": code,
        "output": output,
    }


# ---------------------------------------------------------------------------
# Activate / stale
# ---------------------------------------------------------------------------

async def activate_compute(
    *,
    source: DataSourceDefinition,
    name: str,
    settings: Settings,
    backend: Any,
    script_backend: Any = None,
    publish: Any = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn a compiled transform on, after re-proving it against its fixture.

    The fixture is re-run here rather than trusted from compile time, because
    activation can happen much later than the compile — a person leaves the
    review panel open, comes back after lunch, and in between the sheet may
    have changed.  Activating on a stale claim is exactly what this feature
    must not do.
    """
    assert_generated_code_allowed("a generated sheet transform")

    binding = _require_binding(source, name)
    if binding.compute is None:
        raise ComputeServiceError(f"Binding '{name}' carries no generated code.")
    _check_writes_enabled(binding, settings)

    try:
        output = await rerun_golden(binding, settings=settings, params=params)
    except (ComputeValidationError, ComputeRuntimeError) as exc:
        _mark_stale(binding, f"verification failed at activation: {exc}")
        await persist_binding(
            backend=backend, source=source, binding=binding,
            script_backend=script_backend, publish=publish,
        )
        raise ComputeServiceError(str(exc)) from exc

    binding.compute.activated = True
    binding.compute.stale = False
    binding.compute.stale_reason = ""
    if binding.resolution.golden is not None:
        binding.resolution.golden.verified_at = datetime.now(timezone.utc)

    await persist_binding(
        backend=backend, source=source, binding=binding,
        script_backend=script_backend, publish=publish,
    )
    logger.info("tier-2 binding '%s' activated (script=%s)", name, binding.compute.script_id)
    return {
        "status": "ok",
        "binding": name,
        "compute": compute_status(binding),
        "output": output,
    }


async def mark_compute_stale(
    *,
    source: DataSourceDefinition,
    name: str,
    reason: str,
    backend: Any,
    script_backend: Any = None,
    publish: Any = None,
) -> dict[str, Any]:
    """Mark a binding's transform stale, stopping it until it is re-confirmed.

    Not gated on ADMIN: turning generated code *off* is a safety action, and
    requiring a privileged role to stop something suspicious would be exactly
    the wrong trade.
    """
    binding = _require_binding(source, name)
    if binding.compute is None:
        raise ComputeServiceError(f"Binding '{name}' carries no generated code.")

    _mark_stale(binding, reason or "marked stale manually")
    binding.compute.activated = False
    await persist_binding(
        backend=backend, source=source, binding=binding,
        script_backend=script_backend, publish=publish,
    )
    logger.info("tier-2 binding '%s' marked stale: %s", name, binding.compute.stale_reason)
    return {"status": "ok", "binding": name, "compute": compute_status(binding)}


# ---------------------------------------------------------------------------
# Re-test
# ---------------------------------------------------------------------------

async def retest_compute(
    *,
    source: DataSourceDefinition,
    name: str,
    settings: Settings,
    executor: Any,
    backend: Any,
    script_backend: Any = None,
    publish: Any = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-run the fixture, and check the sheet's schema has not moved.

    What the editor's **Re-test** button calls.  Two independent questions, both
    answered here: does the code still compute the frozen answer, and is the
    schema it was authored against still the schema of the sheet?  Either one
    failing marks the binding stale.
    """
    binding = _require_binding(source, name)
    if binding.compute is None:
        raise ComputeServiceError(f"Binding '{name}' carries no generated code.")

    probe = await _probe(source, executor, binding)
    fresh = probe.get("fingerprint") or ""
    if binding.sheet_schema.fingerprint and fresh != binding.sheet_schema.fingerprint:
        _mark_stale(binding, "the sheet's header row changed since this binding was saved")
        binding.compute.activated = False
        await persist_binding(
            backend=backend, source=source, binding=binding,
            script_backend=script_backend, publish=publish,
        )
        return {
            "status": "stale",
            "binding": name,
            "error": (
                "The sheet's header row has changed, so every column this "
                "binding resolves is suspect. Nothing was read or written. "
                "Headers now: "
                + ", ".join(repr(h) for h in probe.get("headers") or [])
            ),
            "compute": compute_status(binding),
        }

    try:
        output = await rerun_golden(binding, settings=settings, params=params)
    except (ComputeValidationError, ComputeRuntimeError) as exc:
        _mark_stale(binding, f"the golden fixture stopped reproducing: {exc}")
        binding.compute.activated = False
        await persist_binding(
            backend=backend, source=source, binding=binding,
            script_backend=script_backend, publish=publish,
        )
        return {
            "status": "stale",
            "binding": name,
            "error": str(exc),
            "compute": compute_status(binding),
        }

    binding.compute.stale = False
    binding.compute.stale_reason = ""
    if binding.resolution.golden is not None:
        binding.resolution.golden.verified_at = datetime.now(timezone.utc)
    await persist_binding(
        backend=backend, source=source, binding=binding,
        script_backend=script_backend, publish=publish,
    )
    return {
        "status": "ok",
        "binding": name,
        "compute": compute_status(binding),
        "output": output,
    }
