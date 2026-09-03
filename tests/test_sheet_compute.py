"""Tier 2: what generated code may do, and everything it may not.

The claim tier 2 makes is narrow and worth testing literally: a model writes
the *computation* of a binding and nothing else, so generated code produces
values and never addresses.  These tests are mostly about the "never" half —
the gates that hold when the model, or the instruction behind it, misbehaves.

No test here calls a live model.  The single seam is
``sheet_compute_generate.compile_transform(..., ask=...)`` (and
``_ask_model``, monkeypatched where a surface does not expose the seam), so a
compile is driven by a canned reply and the gates are exercised for real.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.config import Settings
from app.domain.models.sheet_binding import (
    SheetBinding,
    header_fingerprint,
    validate_binding,
    BindingValidationError,
)
from app.infrastructure.datasources.sheet_binding_resolver import (
    SheetBindingError,
    sheets_build_write,
    sheets_rows_to_records,
)
from app.infrastructure.datasources.sheet_binding_runtime import (
    BindingRuntimeError,
    plan_write_binding,
    run_read_binding,
)
from app.infrastructure.datasources.sheet_compute import (
    MAX_SOURCE_CHARS,
    SIGNATURE_VERSION,
    ComputeValidationError,
    adversarial_rows,
    assert_deterministic,
    cache_key,
    check_read_output,
    check_write_output,
    compare_golden,
    content_hash,
    run_transform,
    script_id_for,
    validate_transform_source,
)
from tests.test_sheet_bindings_api import (
    FINGERPRINT,
    GRID,
    HEADERS,
    FakeSheetsExecutor,
    _read_binding,
    _sheets_source,
    _write_binding,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

# A hand-authored transform against the fixed signature. Landed and tested
# before any generation existed, because the sandwich has to be proven to run
# end to end independently of whether a model can write the middle of it.
SUM_BY_OWNER = '''
def transform(records, params):
    """Total open rows per owner."""
    totals = {}
    for row in records:
        if str(row.get("status", "")).strip().lower() != "open":
            continue
        owner = str(row.get("owner", "")).strip()
        if not owner:
            continue
        totals[owner] = totals.get(owner, 0) + 1
    # Sorted, so two runs cannot disagree about order.
    return [{"owner": o, "open_rows": totals[o]} for o in sorted(totals)]
'''

WRITE_STATUS = '''
def transform(records, params):
    open_rows = 0
    for row in records:
        if str(row.get("status", "")).strip().lower() == "open":
            open_rows += 1
    return {"status": "reviewed", "notes": "%d open" % open_rows}
'''


def _settings(**overrides: Any) -> Settings:
    values = {
        "sheets_compute_enabled": True,
        "sheets_compute_writes_enabled": True,
        "sheets_compute_max_attempts": 3,
        "sheets_compute_timeout_seconds": 15.0,
        "google_impersonate_sa": "copilot@engineering-368717.iam.gserviceaccount.com",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def compute_writes_on(monkeypatch):
    """Turn the tier-2 write flag on for one test.

    The flag is read at *run* time by the binding runtime, not captured at save
    time, which is the behaviour the test below the fixture pins: a binding
    stored while it was on stops working when it is turned off.
    """
    monkeypatch.setattr("app.core.config.get_settings", lambda: _settings())
    return _settings()


def _compute_read_binding(code: str = SUM_BY_OWNER, **compute_overrides: Any) -> dict:
    binding = _read_binding()
    # A tier-2 read has no filter of its own -- the transform is the filter.
    binding["read"] = {
        "mode": "rows",
        "columns": ["project_id", "status", "owner"],
    }
    binding["output"] = {"key": "totals"}
    compute = {
        "script_id": script_id_for(code),
        "content_hash": content_hash(code),
        "signature_version": SIGNATURE_VERSION,
        "code": code,
        "output_shape": "records",
        "activated": True,
    }
    compute.update(compute_overrides)
    binding["compute"] = compute
    binding["resolution"] = {
        "tier": "script",
        "authored_by": "llm",
        "instruction": "count open rows per owner",
        "model_id": "test/model",
    }
    return binding


def _compute_write_binding(code: str = WRITE_STATUS, **compute_overrides: Any) -> dict:
    binding = _write_binding(
        columns={
            "status": {"from": "compute.status"},
            "notes": {"from": "compute.notes"},
        },
    )
    compute = {
        "script_id": script_id_for(code),
        "content_hash": content_hash(code),
        "signature_version": SIGNATURE_VERSION,
        "code": code,
        "activated": True,
    }
    compute.update(compute_overrides)
    binding["compute"] = compute
    binding["resolution"] = {
        "tier": "script",
        "authored_by": "llm",
        "instruction": "mark reviewed and note the open count",
        "model_id": "test/model",
    }
    return binding


# ─── The sandwich, hand-authored ──────────────────────────────────────────────

async def test_the_hand_authored_sandwich_runs_end_to_end():
    """rows -> records -> transform -> whitelist -> build_write -> A1.

    The shape of the whole feature in one test, with no model anywhere near it:
    tier-1 code turns the grid into records, the transform computes values,
    the whitelist check passes them, and tier-1 code turns them into a range.
    """
    records = sheets_rows_to_records(GRID[1:], HEADERS)
    values = await assert_deterministic(WRITE_STATUS, records, {})
    checked = check_write_output(values, ["status", "notes"])
    assert checked == {"status": "reviewed", "notes": "2 open"}

    built = sheets_build_write(
        {
            "document": {"sheet": "Projects"},
            "schema": {"headers": HEADERS, "header_row": 1},
            "write": {
                "mode": "update_by_key",
                "columns": {"status": {"from": "compute.status"},
                            "notes": {"from": "compute.notes"}},
                "value_input_option": "RAW",
                "blank_policy": "skip",
            },
        },
        checked,
        7,
    )
    # The transform never saw a range, a tab or a file id; tier-1 code computed
    # both A1 addresses from the column names.
    assert [entry["a1"] for entry in built["cells"]] == ["Projects!B7", "Projects!E7"]


# ─── Static gate: the AST allow-list ─────────────────────────────────────────

@pytest.mark.parametrize(
    "code, expected",
    [
        pytest.param(
            "import os\ndef transform(records, params):\n    return []",
            "import of 'os' is not allowed",
            id="import-outside-allowlist",
        ),
        pytest.param(
            "import subprocess\ndef transform(records, params):\n    return []",
            "import of 'subprocess' is not allowed",
            id="import-subprocess",
        ),
        pytest.param(
            "from os import path\ndef transform(records, params):\n    return []",
            "import from 'os' is not allowed",
            id="import-from",
        ),
        pytest.param(
            "def transform(records, params):\n    return records.__class__",
            "attribute '__class__' is not allowed",
            id="dunder-attribute",
        ),
        pytest.param(
            "def transform(records, params):\n    return eval('1')",
            "'eval' is not allowed",
            id="eval",
        ),
        pytest.param(
            "def transform(records, params):\n    exec('x = 1')\n    return []",
            "'exec' is not allowed",
            id="exec",
        ),
        pytest.param(
            "def transform(records, params):\n    return getattr(records, 'x')",
            "'getattr' is not allowed",
            id="getattr-defeats-syntax-checks",
        ),
        pytest.param(
            "def transform(records, params):\n    return open('/etc/passwd').read()",
            "'open' is not allowed",
            id="open",
        ),
        pytest.param(
            "def transform(records, params):\n    return __import__('os')",
            "'__import__' is not allowed",
            id="dunder-import",
        ),
        pytest.param(
            "def transform(records, params):\n    while True:\n        pass",
            "never terminates on its own",
            id="unbounded-while",
        ),
        pytest.param(
            "def transform(records, params):\n"
            "    f = lambda: 1\n"
            "    return f.__globals__",
            "attribute '__globals__' is not allowed",
            id="reflection-via-globals",
        ),
        pytest.param(
            "def transform(records, params):\n"
            "    try:\n        pass\n    except Exception as e:\n"
            "        return e.__traceback__.tb_frame\n    return []",
            "not allowed",
            id="frame-via-traceback",
        ),
        pytest.param(
            "def helper():\n    return 1\n",
            "no 'transform' function was found",
            id="no-entry-point",
        ),
        pytest.param(
            "def transform(rows, params):\n    return []",
            "must take exactly ['records', 'params']",
            id="wrong-param-names",
        ),
        pytest.param(
            "def transform(records, params, *extra):\n    return []",
            "must not take *args",
            id="varargs",
        ),
        pytest.param(
            "print('hi')\ndef transform(records, params):\n    return []",
            "top-level Expr is not allowed",
            id="top-level-statement-runs-before-any-gate",
        ),
        pytest.param(
            "def transform(records, params):\n    return []\n"
            "def transform(records, params):\n    return [1]",
            "defined more than once",
            id="two-entry-points",
        ),
        pytest.param("", "the transform is empty", id="empty"),
    ],
)
def test_the_ast_gate_refuses(code: str, expected: str):
    with pytest.raises(ComputeValidationError) as exc:
        validate_transform_source(code)
    assert expected in str(exc.value)


def test_the_ast_gate_accepts_a_real_transform():
    """The allowed imports are genuinely usable, or the gate is useless."""
    code = (
        "import datetime\nimport math\nimport re\n\n"
        "TAGS = re.compile(r'[a-z]+')\n\n"
        "def _num(text):\n"
        "    try:\n        return float(str(text).replace(',', ''))\n"
        "    except ValueError:\n        return 0.0\n\n"
        "def transform(records, params):\n"
        "    total = math.fsum(_num(r.get('hours')) for r in records)\n"
        "    return {'total': round(total, 2)}\n"
    )
    validate_transform_source(code)


def test_the_source_length_cap_is_enforced():
    code = "def transform(records, params):\n    return []\n" + "# pad\n" * MAX_SOURCE_CHARS
    with pytest.raises(ComputeValidationError, match="over the"):
        validate_transform_source(code)


# ─── Dynamic gate: the sandbox ───────────────────────────────────────────────

async def test_a_transform_runs_in_the_seccomp_sandbox():
    records = sheets_rows_to_records(GRID[1:], HEADERS)
    out = await run_transform(SUM_BY_OWNER, records, {})
    assert out == [{"owner": "ann", "open_rows": 1}, {"owner": "bob", "open_rows": 1}]


async def test_the_sandbox_denies_the_filesystem_even_if_the_ast_gate_is_bypassed():
    """The AST gate is not the boundary; the kernel is.

    Validation is skipped deliberately here, to prove that a transform which
    somehow got past the static gate still cannot read a file -- which is what
    stops generated code reaching the pod's service-account token.
    """
    code = (
        "def transform(records, params):\n"
        "    try:\n"
        "        return open('/etc/passwd').read()[:10]\n"
        "    except Exception as exc:\n"
        "        return 'denied: %s' % type(exc).__name__\n"
    )
    out = await run_transform(code, [], {}, validate=False)
    assert isinstance(out, str) and out.startswith("denied:")


# ─── Determinism ─────────────────────────────────────────────────────────────

async def test_a_non_deterministic_transform_is_rejected():
    code = (
        "import math\n"
        "def transform(records, params):\n"
        "    # A wall-clock read reaching the output: legal Python, useless binding.\n"
        "    import datetime\n"
        "    return {'total': datetime.datetime.now().microsecond}\n"
    )
    with pytest.raises(ComputeValidationError, match="not deterministic"):
        await assert_deterministic(code, [{"a": 1}], {})


async def test_set_iteration_order_is_caught_as_non_determinism():
    """The subtlest of the three, and the reason the check is a double-run.

    A transform that returns an unsorted set's contents passes every static
    check and looks right in review; it just answers differently between runs.
    """
    code = (
        "def transform(records, params):\n"
        "    owners = set(str(r.get('owner', '')) for r in records)\n"
        "    return [{'owner': o} for o in owners]\n"
    )
    records = [{"owner": f"o{i}"} for i in range(24)]
    with pytest.raises(ComputeValidationError, match="not deterministic"):
        await assert_deterministic(code, records, {})


async def test_a_deterministic_transform_over_the_same_rows_passes():
    records = sheets_rows_to_records(GRID[1:], HEADERS)
    assert await assert_deterministic(SUM_BY_OWNER, records, {}) is not None


# ─── Output shape and the column whitelist ───────────────────────────────────

def test_a_write_may_not_produce_a_column_outside_the_whitelist():
    with pytest.raises(ComputeValidationError) as exc:
        check_write_output({"status": "ok", "owner_email": "x@y.z"}, ["status", "notes"])
    message = str(exc.value)
    assert "owner_email" in message
    # Refused, not trimmed: the allowed set is quoted so the author can see it.
    assert "'notes'" in message and "'status'" in message


def test_a_write_output_is_refused_rather_than_trimmed():
    """Dropping the extra key would look like success with half the intent lost."""
    with pytest.raises(ComputeValidationError):
        check_write_output({"notes": "n", "unknown": 1}, ["notes"])


def test_a_single_element_list_is_unwrapped_but_a_longer_one_is_refused():
    assert check_write_output([{"notes": "n"}], ["notes"]) == {"notes": "n"}
    with pytest.raises(ComputeValidationError, match="writes one resolved row"):
        check_write_output([{"notes": "a"}, {"notes": "b"}], ["notes"])


def test_a_cell_may_not_hold_structure():
    with pytest.raises(ComputeValidationError, match="flatten it"):
        check_write_output({"notes": {"nested": 1}}, ["notes"])


@pytest.mark.parametrize(
    "shape, output, ok",
    [
        ("records", [{"a": 1}], True),
        ("records", {"a": 1}, False),
        ("record", {"a": 1}, True),
        ("record", [{"a": 1}], False),
        ("value", 42, True),
        ("value", [1, 2], False),
    ],
)
def test_the_declared_read_shape_is_enforced(shape: str, output: Any, ok: bool):
    if ok:
        assert check_read_output(output, shape) == output
    else:
        with pytest.raises(ComputeValidationError):
            check_read_output(output, shape)


async def test_the_whitelist_is_enforced_at_run_time_not_only_at_compile_time(
    compute_writes_on,
):
    """A stored transform that reaches for a new column fails the run.

    This is the case the whole design turns on: the code was fine when it was
    approved, somebody changed it in the database, or the model was always
    going to return one extra key on rows it had not seen. Either way the run
    stops rather than writing somewhere the binding never declared.
    """
    rogue = (
        "def transform(records, params):\n"
        "    return {'status': 'reviewed', 'rtk_flag': 'yes'}\n"
    )
    source = _sheets_source().model_copy(update={
        "bindings": [SheetBinding.model_validate(_compute_write_binding(rogue))],
    })
    binding = source.bindings[0]
    with pytest.raises(BindingRuntimeError) as exc:
        await plan_write_binding(
            source, FakeSheetsExecutor(), binding, {"project_id": "P-1"},
        )
    assert "rtk_flag" in str(exc.value)
    assert "may not write" in str(exc.value)


# ─── Fingerprint ─────────────────────────────────────────────────────────────

async def test_a_tier2_read_fails_loudly_when_the_header_row_changed():
    """No fallback, and no transform run: the columns are all suspect."""
    drifted = [["project_id", "status", "owner_name", "rtk_flag", "notes"], *GRID[1:]]
    source = _sheets_source().model_copy(update={
        "bindings": [SheetBinding.model_validate(_compute_read_binding())],
    })
    with pytest.raises(SheetBindingError, match="header row has changed"):
        await run_read_binding(
            source, FakeSheetsExecutor(grid=drifted), source.bindings[0], {"assignee": "ann"},
        )


async def test_a_tier2_write_fails_loudly_when_the_header_row_changed(compute_writes_on):
    drifted = [["project_id", "state", "owner", "rtk_flag", "notes"], *GRID[1:]]
    source = _sheets_source().model_copy(update={
        "bindings": [SheetBinding.model_validate(_compute_write_binding())],
    })
    with pytest.raises(SheetBindingError, match="header row has changed"):
        await plan_write_binding(
            source, FakeSheetsExecutor(grid=drifted), source.bindings[0],
            {"project_id": "P-1"},
        )


# ─── Golden fixture ──────────────────────────────────────────────────────────

def test_adversarial_rows_add_the_four_shapes_real_sheets_produce():
    rows = adversarial_rows(HEADERS, [GRID[1]])
    assert rows[0] == GRID[1]
    # A fully empty row, a truncated row, a number stored as text, a duplicate.
    assert [""] * len(HEADERS) in rows
    assert any(len(r) < len(HEADERS) for r in rows[1:])
    assert any("1,234" in r for r in rows)
    assert rows[-1] == rows[0]


def test_a_transform_survives_the_adversarial_rows():
    """The synthetic rows are the half of a fixture that finds bugs."""
    rows = adversarial_rows(HEADERS, GRID[1:])
    records = sheets_rows_to_records(rows, HEADERS)
    assert len(records) == len(rows)
    # Padded, so no record is short and nothing shifted into the wrong column.
    assert all(set(r) == set(HEADERS) for r in records)


def test_compare_golden_accepts_a_reproduction_and_rejects_a_drift():
    compare_golden([{"a": 1}], [{"a": 1}])
    with pytest.raises(ComputeValidationError, match="no longer reproduces"):
        compare_golden([{"a": 1}], [{"a": 2}])


def test_the_cache_key_changes_with_every_input_that_matters():
    base = dict(
        instruction="sum hours",
        answers={"which date?": "created_at"},
        schema_fingerprint=FINGERPRINT,
        model_id="test/model",
    )
    key = cache_key(**base)
    assert key == cache_key(**base)
    assert key != cache_key(**{**base, "instruction": "sum hours by owner"})
    assert key != cache_key(**{**base, "answers": {"which date?": "closed_at"}})
    assert key != cache_key(**{**base, "schema_fingerprint": header_fingerprint(["a"])})
    assert key != cache_key(**{**base, "model_id": "other/model"})
    assert key != cache_key(**base, signature_version=99)


def test_answers_order_does_not_change_the_cache_key():
    a = cache_key(instruction="i", answers={"x": "1", "y": "2"},
                  schema_fingerprint="f", model_id="m")
    b = cache_key(instruction="i", answers={"y": "2", "x": "1"},
                  schema_fingerprint="f", model_id="m")
    assert a == b


# ─── Model consistency ───────────────────────────────────────────────────────

def test_the_signature_version_constants_cannot_drift():
    """The model mirrors the infrastructure constant as a literal; pin them."""
    from app.domain.models.sheet_binding import COMPUTE_SIGNATURE_VERSION

    assert COMPUTE_SIGNATURE_VERSION == SIGNATURE_VERSION


def test_a_stored_script_from_another_signature_version_is_refused():
    binding = SheetBinding.model_validate(
        _compute_read_binding(signature_version=SIGNATURE_VERSION + 1)
    )
    with pytest.raises(BindingValidationError, match="signature version"):
        validate_binding(binding)


def test_compute_and_tier_must_agree_in_both_directions():
    # Code present, tier claims "no code".
    payload = _compute_read_binding()
    payload["resolution"]["tier"] = "binding"
    with pytest.raises(BindingValidationError, match="must be 'script'"):
        validate_binding(SheetBinding.model_validate(payload))

    # Tier claims code, none present.
    plain = _read_binding()
    plain["resolution"] = {"tier": "script"}
    with pytest.raises(BindingValidationError, match="carries no 'compute' block"):
        validate_binding(SheetBinding.model_validate(plain))


def test_a_compute_reference_without_a_compute_block_is_refused():
    payload = _write_binding(columns={"status": {"from": "compute.status"}})
    with pytest.raises(BindingValidationError, match="nothing would ever produce them"):
        validate_binding(SheetBinding.model_validate(payload))


def test_a_compute_field_must_be_a_declared_column():
    payload = _compute_write_binding()
    payload["write"]["columns"] = {
        "status": {"from": "compute.status"},
        "notes": {"from": "compute.something_else"},
    }
    with pytest.raises(BindingValidationError, match="are not columns of this write"):
        validate_binding(SheetBinding.model_validate(payload))


def test_a_compute_path_is_not_a_param_of_the_compiled_operation():
    """Or a tier-2 write would demand its own computed values from the caller."""
    from app.infrastructure.datasources.sheet_binding_compile import compile_binding

    binding = SheetBinding.model_validate(_compute_write_binding())
    params = {p.name for p in compile_binding(binding).params}
    assert params == {"project_id"}
    assert binding.compute_paths() == ["notes", "status"]


def test_set_cells_cannot_be_generated():
    payload = _compute_write_binding()
    payload["write"] = {
        "mode": "set_cells",
        "cells": [{"range": {"a1": "Projects!A1"}, "value": {"literal": "x"}}],
    }
    with pytest.raises(BindingValidationError, match="do not combine"):
        validate_binding(SheetBinding.model_validate(payload))


# ─── The tier-2 write flag ───────────────────────────────────────────────────

async def test_a_tier2_write_is_blocked_when_the_flag_is_off(monkeypatch):
    """Default-off, and enforced at run time rather than only at save time.

    A generated read is wrong in its response; a generated write is wrong in
    somebody's spreadsheet. So the write half is opt-in per deployment, and the
    check lives on the run path — a binding stored while the flag was on has to
    stop the moment it is turned off, or the flag was only ever an authoring
    formality.
    """
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: _settings(sheets_compute_writes_enabled=False),
    )
    source = _sheets_source().model_copy(update={
        "bindings": [SheetBinding.model_validate(_compute_write_binding())],
    })
    with pytest.raises(BindingRuntimeError, match="SHEETS_COMPUTE_WRITES_ENABLED"):
        await plan_write_binding(
            source, FakeSheetsExecutor(), source.bindings[0], {"project_id": "P-1"},
        )


async def test_a_tier2_read_is_unaffected_by_the_write_flag(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: _settings(sheets_compute_writes_enabled=False),
    )
    source = _sheets_source().model_copy(update={
        "bindings": [SheetBinding.model_validate(_compute_read_binding())],
    })
    result = await run_read_binding(
        source, FakeSheetsExecutor(), source.bindings[0], {"assignee": "ann"},
    )
    assert result == [{"owner": "ann", "open_rows": 1}, {"owner": "bob", "open_rows": 1}]


async def test_generated_code_does_not_run_until_it_is_activated(monkeypatch):
    """Compiling is not the same event as somebody deciding to run it."""
    monkeypatch.setattr("app.core.config.get_settings", lambda: _settings())
    source = _sheets_source().model_copy(update={
        "bindings": [
            SheetBinding.model_validate(_compute_read_binding(activated=False)),
        ],
    })
    with pytest.raises(BindingRuntimeError, match="has not been activated"):
        await run_read_binding(
            source, FakeSheetsExecutor(), source.bindings[0], {"assignee": "ann"},
        )


async def test_a_stale_binding_refuses_to_run(monkeypatch):
    monkeypatch.setattr("app.core.config.get_settings", lambda: _settings())
    source = _sheets_source().model_copy(update={
        "bindings": [
            SheetBinding.model_validate(
                _compute_read_binding(stale=True, stale_reason="headers changed")
            ),
        ],
    })
    with pytest.raises(BindingRuntimeError, match="marked stale"):
        await run_read_binding(
            source, FakeSheetsExecutor(), source.bindings[0], {"assignee": "ann"},
        )


async def test_a_stored_transform_is_revalidated_on_every_run(monkeypatch):
    """Tightening the allow-list must retroactively stop code it once accepted.

    A gate that only ran at authoring time would leave whatever an earlier,
    looser version let through running forever.
    """
    monkeypatch.setattr("app.core.config.get_settings", lambda: _settings())
    payload = _compute_read_binding()
    # Stored directly, as an older backend version might have accepted it.
    payload["compute"]["code"] = (
        "import os\ndef transform(records, params):\n    return []\n"
    )
    source = _sheets_source().model_copy(update={
        "bindings": [SheetBinding.model_validate(payload)],
    })
    with pytest.raises(BindingRuntimeError, match="no longer passes validation"):
        await run_read_binding(
            source, FakeSheetsExecutor(), source.bindings[0], {"assignee": "ann"},
        )
