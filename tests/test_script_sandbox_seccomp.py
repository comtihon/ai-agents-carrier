"""The sandbox's kernel boundary.

The in-interpreter controls — cleared `sys.meta_path`, a blocked-module list, a
guarded `__import__` — keep an honest script from wandering off. They are not a
boundary, because they live in the same interpreter as the code they filter and
reflection reaches around them: `().__class__.__mro__[1].__subclasses__()` walks
to classes no deny-list removed.

So the sandbox installs a seccomp-bpf allow-list before any script runs. These
tests are the ones that matter: they assume the interpreter layer has already
been defeated and check that the kernel still says no.

Linux x86_64 only, which is what the filter targets and what the nodes run.
"""
from __future__ import annotations

import os
import platform

import pytest

from app.infrastructure.orchestration.script_sandbox import (
    _SECCOMP_ALLOWED_SYSCALLS,
    ScriptSandboxError,
    run_script,
)

pytestmark = pytest.mark.skipif(
    not (os.name == "posix" and platform.system() == "Linux" and platform.machine() == "x86_64"),
    reason="the seccomp filter targets Linux x86_64",
)


async def _run(code: str, state: dict | None = None):
    return await run_script(code, state or {}, runtime="local", timeout=30)


async def _expect_blocked(code: str) -> None:
    """The script must not complete. Either the kernel kills the syscall and the
    exception propagates, or the process dies -- both surface as a sandbox error."""
    with pytest.raises(ScriptSandboxError):
        await _run(code)


# ---------------------------------------------------------------------------
# Scripts must still be able to do their job
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_computation_still_works():
    assert await _run("output = sum(i * i for i in range(1000))") == 332833500


@pytest.mark.asyncio
async def test_state_goes_in_and_output_comes_back():
    out = await _run(
        "output = {'n': len(state['items']), 'up': state['name'].upper()}",
        {"items": [1, 2, 3], "name": "ada"},
    )
    assert out == {"n": 3, "up": "ADA"}


@pytest.mark.asyncio
async def test_pre_imported_stdlib_is_usable_after_the_filter():
    """Everything a script may use is imported before the filter goes up, since
    afterwards openat is denied and nothing can be loaded from disk. A regression
    here looks like a stdlib module mysteriously vanishing."""
    out = await _run(
        "import json, hashlib, re, datetime\n"
        "output = [\n"
        "  hashlib.sha256(json.dumps({'a': 1}).encode()).hexdigest()[:8],\n"
        "  re.sub('x', 'y', 'x'),\n"
        "  datetime.datetime(2026, 1, 2).strftime('%Y-%m-%d'),\n"
        "]"
    )
    assert out == ["f9d86028", "y", "2026-01-02"]


@pytest.mark.asyncio
async def test_random_and_time_work():
    """getrandom and clock_gettime are on the allow-list; if either were dropped
    a great many ordinary scripts would fail."""
    out = await _run(
        "import random, time\n"
        "random.seed(1)\n"
        "output = isinstance(random.random(), float) and time.time() > 0"
    )
    assert out is True


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cannot_read_a_file():
    """The service-account token case. A syscall filter cannot inspect a path,
    so the only way to refuse one file is to refuse them all."""
    await _expect_blocked("output = open('/etc/hostname').read()")


@pytest.mark.asyncio
async def test_cannot_read_the_process_environment():
    await _expect_blocked("output = open('/proc/self/environ','rb').read().decode()")


@pytest.mark.asyncio
async def test_cannot_open_a_socket():
    await _expect_blocked("import socket; output = socket.socket()")


@pytest.mark.asyncio
async def test_cannot_spawn_a_process():
    await _expect_blocked("import subprocess; output = subprocess.run(['id'])")


@pytest.mark.asyncio
async def test_cannot_import_ctypes_to_rebuild_the_syscall_bridge():
    """ctypes is how the bootstrap installs the filter; the script must not get
    it back afterwards, or it could call syscalls directly."""
    await _expect_blocked("import ctypes; output = 'got ctypes'")


# ---------------------------------------------------------------------------
# Assume the interpreter layer has already lost
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflection_reaching_open_is_stopped_by_the_kernel():
    """The whole reason the kernel layer exists. Reflection still finds `open` --
    the deny-list never removed it -- and the syscall is refused anyway."""
    out = await _run(
        "try:\n"
        "    f = __builtins__.open if hasattr(__builtins__, 'open') else __builtins__['open']\n"
        "    output = 'ESCAPED:' + f('/etc/hostname').read()\n"
        "except OSError as e:\n"
        "    output = 'contained:%s' % e.errno"
    )
    assert out == "contained:1", out  # EPERM, from the kernel


@pytest.mark.asyncio
async def test_reflection_cannot_reach_a_process_spawner():
    out = await _run(
        "subs = ().__class__.__mro__[1].__subclasses__()\n"
        "output = [c.__name__ for c in subs if c.__name__ == 'Popen']"
    )
    assert out == []


@pytest.mark.asyncio
async def test_builtin_importer_cannot_load_a_blocked_module():
    out = await _run(
        "try:\n"
        "    L = [c for c in ().__class__.__mro__[1].__subclasses__()\n"
        "         if c.__name__ == 'BuiltinImporter'][0]\n"
        "    output = 'ESCAPED:' + str(L.load_module('socket'))\n"
        "except Exception as e:\n"
        "    output = 'contained:%s' % type(e).__name__"
    )
    assert out.startswith("contained:"), out


# ---------------------------------------------------------------------------
# The filter itself
# ---------------------------------------------------------------------------

def test_allow_list_denies_the_syscalls_the_boundary_rests_on():
    """An allow-list is only as good as what it leaves out. These four are what
    the WRITE-level guarantee depends on; a well-meaning addition would quietly
    remove the boundary, so pin them."""
    allowed = set(_SECCOMP_ALLOWED_SYSCALLS)
    for name, nr in (
        ("open", 2), ("openat", 257), ("openat2", 437),
        ("socket", 41), ("connect", 42),
        ("execve", 59), ("execveat", 322),
        ("fork", 57), ("vfork", 58), ("clone", 56), ("clone3", 435),
        ("ptrace", 101), ("process_vm_readv", 310),
    ):
        assert nr not in allowed, f"{name} ({nr}) must not be on the allow-list"


def test_allow_list_keeps_what_a_script_needs():
    allowed = set(_SECCOMP_ALLOWED_SYSCALLS)
    for name, nr in (
        ("read", 0), ("write", 1), ("exit_group", 231),
        ("mmap", 9), ("brk", 12), ("futex", 202), ("getrandom", 318),
    ):
        assert nr in allowed, f"{name} ({nr}) must stay on the allow-list"
