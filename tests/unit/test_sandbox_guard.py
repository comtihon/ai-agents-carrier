"""The ADMIN gate on unsandboxed `python` steps.

`sandbox: false` execs inside the backend process, next to its credentials, so
this is a code-execution boundary rather than a data one.
"""
from __future__ import annotations

import pytest

from app.infrastructure.auth.authorization import Permission
from app.infrastructure.auth.sandbox_guard import (
    SandboxNotPermittedError,
    assert_sandbox_allowed,
    find_unsandboxed_python_steps,
)

ADMIN = frozenset(Permission)
WRITER = frozenset({Permission.ACCESS, Permission.READ, Permission.WRITE, Permission.DELETE})


# ── detection ────────────────────────────────────────────────────────────────

def test_explicit_false_is_detected() -> None:
    steps = [{"id": "danger", "type": "python", "sandbox": False}]
    assert find_unsandboxed_python_steps(steps) == ["danger"]


def test_omitted_sandbox_is_sandboxed() -> None:
    """Sandboxing is the documented default; absence must not read as opt-out."""
    assert find_unsandboxed_python_steps([{"id": "s", "type": "python"}]) == []


def test_sandbox_true_is_allowed() -> None:
    assert find_unsandboxed_python_steps([{"id": "s", "type": "python", "sandbox": True}]) == []


def test_non_python_step_with_sandbox_false_is_ignored() -> None:
    """The flag only means anything on a python step."""
    assert find_unsandboxed_python_steps([{"id": "s", "type": "http", "sandbox": False}]) == []


@pytest.mark.parametrize("falsy", [0, "", None, "false", "no"])
def test_only_a_real_boolean_false_counts(falsy: object) -> None:
    """`is False` on purpose: a string "false" is not the documented opt-out, and
    treating it as one would let a typo silently disable the sandbox."""
    steps = [{"id": "s", "type": "python", "sandbox": falsy}]
    assert find_unsandboxed_python_steps(steps) == []


def test_multiple_offending_steps_are_all_reported() -> None:
    steps = [
        {"id": "a", "type": "python", "sandbox": False},
        {"id": "ok", "type": "python"},
        {"id": "b", "type": "python", "sandbox": False},
    ]
    assert find_unsandboxed_python_steps(steps) == ["a", "b"]


def test_unnamed_step_still_reported() -> None:
    assert find_unsandboxed_python_steps([{"type": "python", "sandbox": False}]) == ["<unnamed>"]


def test_step_name_used_when_id_absent() -> None:
    steps = [{"name": "by-name", "type": "python", "sandbox": False}]
    assert find_unsandboxed_python_steps(steps) == ["by-name"]


# ── definition shapes ────────────────────────────────────────────────────────

def test_dict_definition_with_steps_key() -> None:
    defn = {"id": "wf", "steps": [{"id": "s", "type": "python", "sandbox": False}]}
    assert find_unsandboxed_python_steps(defn) == ["s"]


def test_definition_exposing_to_raw_dict() -> None:
    class Defn:
        def to_raw_dict(self) -> dict:
            return {"steps": [{"id": "s", "type": "python", "sandbox": False}]}

    assert find_unsandboxed_python_steps(Defn()) == ["s"]


def test_definition_whose_to_raw_dict_raises_is_treated_as_empty() -> None:
    """Cannot approve what it could not read: an unreadable definition also has no
    `sandbox: false` to approve, so yielding nothing is safe."""
    class Defn:
        def to_raw_dict(self) -> dict:
            raise RuntimeError("boom")

    assert find_unsandboxed_python_steps(Defn()) == []


def test_mapping_of_step_id_to_step() -> None:
    defn = {"first": {"id": "first", "type": "python", "sandbox": False}}
    assert find_unsandboxed_python_steps(defn) == ["first"]


@pytest.mark.parametrize("junk", [None, 42, "steps", [], {}, [1, 2, 3], ["a"]])
def test_unrecognised_shapes_yield_nothing(junk: object) -> None:
    assert find_unsandboxed_python_steps(junk) == []


# ── the gate ─────────────────────────────────────────────────────────────────

def test_admin_may_disable_the_sandbox() -> None:
    assert_sandbox_allowed([{"id": "s", "type": "python", "sandbox": False}], ADMIN)


def test_writer_may_not_disable_the_sandbox() -> None:
    with pytest.raises(SandboxNotPermittedError) as excinfo:
        assert_sandbox_allowed([{"id": "s", "type": "python", "sandbox": False}], WRITER)
    assert excinfo.value.step_ids == ["s"]


def test_writer_may_submit_sandboxed_steps() -> None:
    assert_sandbox_allowed([{"id": "s", "type": "python"}], WRITER)


def test_caller_with_no_permissions_is_denied() -> None:
    """An unauthenticated path reaching the guard must not be treated as admin."""
    with pytest.raises(SandboxNotPermittedError):
        assert_sandbox_allowed([{"id": "s", "type": "python", "sandbox": False}], frozenset())


def test_error_message_names_the_steps_and_says_how_to_proceed() -> None:
    with pytest.raises(SandboxNotPermittedError) as excinfo:
        assert_sandbox_allowed(
            [{"id": "alpha", "type": "python", "sandbox": False}], WRITER
        )
    message = str(excinfo.value)
    assert "alpha" in message
    assert "admin" in message.lower()
    assert "sandbox" in message.lower()
