"""Tests for the local Python script sandbox.

Only the ``local`` runtime is exercised here — ``docker`` and ``k8s`` need a
daemon / cluster and are covered by integration runs.
"""
from __future__ import annotations

import pytest

from app.infrastructure.orchestration.script_sandbox import (
    ScriptSandboxError,
    run_script,
)


async def test_runs_code_with_state_and_returns_output():
    result = await run_script('output = state["x"] * 2', {"x": 21})
    assert result == 42


async def test_stdlib_is_available():
    result = await run_script("import math\noutput = round(math.pi, 3)", {})
    assert result == 3.142


async def test_backend_environment_is_not_visible(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret")
    result = await run_script("import os\noutput = dict(os.environ)", {})
    assert result == {}


async def test_process_spawning_modules_are_blocked():
    with pytest.raises(ScriptSandboxError, match="not available in sandboxed scripts"):
        await run_script("import subprocess\noutput = 1", {})


async def test_network_modules_are_blocked():
    with pytest.raises(ScriptSandboxError, match="not available in sandboxed scripts"):
        await run_script("import socket\noutput = 1", {})


async def test_backend_libraries_are_not_importable():
    # -S keeps site-packages off sys.path, and the bootstrap then tears down the
    # loaders entirely, so anything not pre-imported is refused by name.
    with pytest.raises(ScriptSandboxError, match="'pydantic' is not available"):
        await run_script("import pydantic\noutput = 1", {})


async def test_import_machinery_cannot_be_used_to_reach_a_blocked_module():
    """``importlib.import_module`` never consults ``builtins.__import__``.

    Filtering the builtin alone therefore left a three-line path to arbitrary
    process execution, which is why the loaders are removed as well.
    """
    escape = (
        "import importlib\n"
        "output = importlib.import_module('subprocess').run(['id'])\n"
    )
    with pytest.raises(ScriptSandboxError, match="not available in sandboxed scripts"):
        await run_script(escape, {})


async def test_os_backing_module_cannot_be_reached():
    """``posix.system`` survives deleting ``os.system``, so posix is blocked too."""
    with pytest.raises(ScriptSandboxError, match="not available in sandboxed scripts"):
        await run_script("import posix\noutput = 1", {})


async def test_timeout_kills_a_runaway_script():
    with pytest.raises(ScriptSandboxError, match="timed out"):
        await run_script("while True: pass", {}, timeout=3)


async def test_script_stdout_does_not_break_result_parsing():
    result = await run_script('print("chatty")\noutput = "done"', {})
    assert result == "done"


async def test_script_error_is_reported():
    with pytest.raises(ScriptSandboxError, match="ZeroDivisionError"):
        await run_script("output = 1 / 0", {})


async def test_unknown_runtime_is_rejected():
    with pytest.raises(ValueError, match="Unknown sandbox runtime"):
        await run_script("output = 1", {}, runtime="vm")
