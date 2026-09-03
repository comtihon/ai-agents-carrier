"""Sandboxed execution of workflow ``python`` steps.

A ``python`` step can run in two modes:

* ``sandbox: false`` — legacy behaviour: ``exec`` inside the backend process,
  with full access to the backend's environment, installed libraries and
  filesystem.  Kept for trusted infrastructure scripts.
* ``sandbox: true`` (default) — the script runs in an isolated Python
  interpreter that has **no** access to the backend's env vars, tools, bash,
  installed libraries or system dependencies.

Three sandbox runtimes are supported, mirroring the agent runtimes:

``local``
    A child ``python -I -S -B`` process: empty environment, no ``site``
    packages (so none of the backend's dependencies are importable), a
    throw-away working directory, CPU/memory/file-size rlimits and a wall-clock
    timeout.  Process-level isolation only — no kernel namespaces.
``docker``
    A throw-away container from ``script_sandbox_image`` with networking
    disabled, a read-only root filesystem, no inherited environment and a
    memory limit.
``k8s``
    A one-shot ``Never``-restart Pod in the agent namespace with the service
    account token unmounted, a read-only root filesystem and resource limits.

Payload transport
-----------------
``{"code": ..., "state": ...}`` is JSON-encoded and handed to the interpreter
via stdin (local), a read-only bind mount (docker) or a ConfigMap (k8s), so
script and state size are not limited by ``ARG_MAX``.  The result is printed on
the last stdout line prefixed with ``__SCRIPT_RESULT__`` so that anything the
script itself prints is preserved and ignored by the parser.
"""
from __future__ import annotations

import ast
import asyncio
import base64
import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
from typing import Any

logger = logging.getLogger(__name__)

RESULT_MARKER = "__SCRIPT_RESULT__"

# Modules that would hand a sandboxed script a way out (process spawning,
# network access, raw memory).  Third-party libraries are already unreachable
# because the interpreter runs with ``-S`` (no site-packages).
#
# The import machinery is on the list as well, and that is the important part:
# ``importlib.import_module`` does not consult ``builtins.__import__``, so
# filtering the builtin alone left a three-line path to ``subprocess`` (
# ``importlib.import_module("subprocess")``, verified).  ``posix``/``nt`` are
# listed because they are what ``os`` is a wrapper around — ``posix.system``
# survives deleting ``os.system``.
_BLOCKED_MODULES = (
    "subprocess", "socket", "ssl", "ctypes", "multiprocessing", "http",
    "urllib", "urllib3", "requests", "httpx", "aiohttp", "ftplib", "smtplib",
    "poplib", "imaplib", "telnetlib", "webbrowser", "xmlrpc", "pty", "tty",
    "fcntl", "mmap", "site",
    # import machinery and other module loaders
    "importlib", "imp", "pkgutil", "runpy", "zipimport", "code", "codeop",
    "_frozen_importlib", "_frozen_importlib_external",
    # os' backing modules, and the object graph that leads back to them
    "posix", "nt", "_posixsubprocess", "_socket", "_ssl", "gc", "signal",
    "resource",
)

# Pre-imported for the script's benefit before the loaders are torn down: after
# that only modules already resident in ``sys.modules`` can be imported at all,
# which is what makes the deny-list above enforceable.  Kept to stdlib that
# computes rather than reaches out.
_SANDBOX_STDLIB = (
    "abc", "array", "base64", "binascii", "bisect", "calendar", "cmath",
    "collections", "collections.abc", "contextlib", "copy", "csv", "dataclasses",
    "datetime", "decimal", "difflib", "enum", "fnmatch", "fractions",
    "functools", "graphlib", "hashlib", "heapq", "hmac", "html", "io",
    "itertools", "json", "math", "numbers", "operator", "os", "os.path",
    "pprint", "random", "re", "reprlib", "secrets", "statistics", "string",
    "struct", "textwrap", "time", "types", "typing", "unicodedata", "uuid",
    "warnings", "zlib",
    # Private modules the public ones import lazily, on first use rather than at
    # import time. They have to be pre-imported here or the loaders are gone by
    # the time they are wanted: datetime.strptime reaches for _strptime the first
    # time it is called, which made strptime raise ImportError in every sandboxed
    # script while `import datetime` looked perfectly fine.
    "_strptime", "_pydatetime", "_pydecimal", "_compat_pickle", "encodings.idna",
)

# ---------------------------------------------------------------------------
# seccomp-bpf allow-list (x86_64)
# ---------------------------------------------------------------------------
# The in-interpreter deny-list below keeps an honest script from wandering off.
# It is not a boundary: a deny-list lives in the same interpreter as the code it
# filters, and reflection reaches around it (a str.format() chain is enough).
# This is the boundary — the kernel refusing the syscall, whatever the
# interpreter thinks.
#
# An ALLOW-list, so a syscall nobody thought about is denied rather than
# forgotten. The set is what CPython needs to compute and to write its result to
# stdout, and nothing else. Absent, and therefore refused:
#
#   openat/open/openat2   no filesystem at all — this is what stops a script
#                         reading the pod's service-account token, which no
#                         path-blind syscall filter could otherwise prevent
#   socket/connect/…      no network
#   execve/fork/clone     no processes
#   ptrace/process_vm_*   no reaching into other processes
#
# Denial is EPERM rather than SIGSYS: a script that trips it gets a
# PermissionError it can be debugged from, instead of dying without a word.
_SECCOMP_ALLOWED_SYSCALLS = (
    0,    # read
    1,    # write
    3,    # close
    5,    # fstat
    8,    # lseek
    9,    # mmap
    10,   # mprotect
    11,   # munmap
    12,   # brk
    13,   # rt_sigaction
    14,   # rt_sigprocmask
    15,   # rt_sigreturn
    16,   # ioctl        (isatty on the inherited fds)
    24,   # sched_yield
    25,   # mremap
    28,   # madvise
    32,   # dup
    35,   # nanosleep
    39,   # getpid
    60,   # exit
    63,   # uname
    72,   # fcntl
    79,   # getcwd
    96,   # gettimeofday
    131,  # sigaltstack
    202,  # futex
    219,  # restart_syscall
    228,  # clock_gettime
    229,  # clock_getres
    230,  # clock_nanosleep   (time.sleep)
    231,  # exit_group
    262,  # newfstatat        (metadata on an already-open fd; opens nothing)
    273,  # set_robust_list
    302,  # prlimit64
    318,  # getrandom         (random, secrets, hash seeding)
    334,  # rseq
)

# Installed by the bootstrap once the interpreter is warm. Kept as source text
# because the sandboxed interpreter runs with -S: it cannot import a helper from
# site-packages, so the filter is built with ctypes from the stdlib instead of
# libseccomp. ctypes itself is torn out of sys.modules immediately afterwards.
_SECCOMP_INSTALL = f'''
def _install_seccomp():
    """Return None on success, or a string explaining why it could not install.

    Imports nothing: it runs after the import machinery has been dismantled, so
    it uses the ctypes/struct bound at the top of this bootstrap. os.uname()
    rather than platform, to avoid needing one more module resident.
    """
    if os.uname().machine != "x86_64":
        return "unsupported architecture %r" % os.uname().machine

    AUDIT_ARCH_X86_64 = 0xC000003E
    LD_W_ABS, JEQ_K, RET_K = 0x20, 0x15, 0x06
    RET_ALLOW, RET_ERRNO, RET_KILL = 0x7FFF0000, 0x00050000, 0x80000000
    EPERM = 1

    allowed = {tuple(_SECCOMP_ALLOWED_SYSCALLS)!r}
    n = len(allowed)

    # arch guard first: syscall numbers are meaningless without it, so a
    # mismatched personality is killed rather than filtered by the wrong table.
    ins = [
        (LD_W_ABS, 0, 0, 4),                     # load arch
        (JEQ_K, 0, n + 3, AUDIT_ARCH_X86_64),    # wrong arch -> KILL
        (LD_W_ABS, 0, 0, 0),                     # load syscall nr
    ]
    for i, nr in enumerate(allowed):
        ins.append((JEQ_K, n - i, 0, nr))        # match -> ALLOW
    ins.append((RET_K, 0, 0, RET_ERRNO | EPERM))  # fall-through -> EPERM
    ins.append((RET_K, 0, 0, RET_ALLOW))
    ins.append((RET_K, 0, 0, RET_KILL))

    blob = b"".join(struct.pack("HBBI", *x) for x in ins)
    # Both buffers must outlive the prctl call; keep them referenced until it
    # returns, or the kernel reads freed memory.
    buf = ctypes.create_string_buffer(blob, len(blob))
    prog = struct.pack("HxxxxxxP", len(ins), ctypes.addressof(buf))

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError as exc:
        return "libc not loadable: %s" % exc

    # no_new_privs is what lets an unprivileged process install a filter at all.
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        return "PR_SET_NO_NEW_PRIVS failed (errno %d)" % ctypes.get_errno()
    if libc.prctl(22, 2, ctypes.c_char_p(prog), 0, 0) != 0:
        return "PR_SET_SECCOMP failed (errno %d)" % ctypes.get_errno()
    return None
'''


# Executed by the sandboxed interpreter.  argv[1] is "-" (read payload from
# stdin) or a path to a JSON payload file.
_BOOTSTRAP = f'''
# ctypes is imported here, at the top, because the seccomp filter is built with
# it *after* the import machinery has been torn down -- by then nothing new can
# be loaded. It is dropped from sys.modules again once the filter is up, and it
# never reaches the script's namespace.
import base64, builtins, ctypes, json, os, struct, sys

{_SECCOMP_INSTALL}

_arg = sys.argv[1] if len(sys.argv) > 1 else "-"
_raw = sys.stdin.read() if _arg == "-" else open(_arg, "r", encoding="utf-8").read()
_payload = json.loads(_raw)

# Nothing from the caller's environment leaks in.
os.environ.clear()
for _name in (
    "system", "popen", "execl", "execle", "execlp", "execv", "execve",
    "execvp", "execvpe", "spawnl", "spawnv", "spawnve", "fork", "forkpty",
    "posix_spawn", "putenv",
):
    if hasattr(os, _name):
        try:
            delattr(os, _name)
        except Exception:
            pass

_blocked = {set(_BLOCKED_MODULES)!r}
_real_import = builtins.__import__

# Load what a script is allowed to use *before* the loaders go away.
for _name in {tuple(_SANDBOX_STDLIB)!r}:
    try:
        _real_import(_name)
    except Exception:
        pass

# Remove every way to load a module that is not already resident, so the
# deny-list below cannot be walked around by asking a different importer.
sys.meta_path.clear()
sys.path_hooks.clear()
sys.path_importer_cache.clear()
del sys.path[:]
for _name in list(sys.modules):
    if _name.split(".")[0] in _blocked:
        sys.modules.pop(_name, None)


def _guarded_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in _blocked or root not in sys.modules:
        raise ImportError(
            "module '%s' is not available in sandboxed scripts" % name
        )
    return _real_import(name, *args, **kwargs)


builtins.__import__ = _guarded_import

# The kernel boundary goes up last, once every module the script may use is
# resident: the filter denies openat, so nothing can be loaded from disk after
# this point -- including the .py files of a stdlib module imported lazily.
#
# Refusing to run is the only honest failure here. A script that ran unfiltered
# because the filter would not install is exactly the situation the caller was
# told could not happen, and `local` is only a WRITE-level runtime because this
# succeeds.
# --- data stream -------------------------------------------------------
# Opened HERE, before the filter goes up, because the filter denies openat:
# after it no path can be opened at all. An already-open descriptor stays
# readable -- read, lseek and fstat are on the allow-list -- and that is what
# lets a script stream a result far larger than its own memory limit.
_data_fh = None
_data_path = _payload.get("data_path")
_data_dest = _payload.get("data_dest")
if _payload.get("data_from_stdin") and _data_dest:
    # The stream lives on another pod, so the backend is pushing its bytes in
    # on stdin. They land in a file first, so the script can seek and make
    # more than one pass.
    _sink = open(_data_dest, "wb")
    try:
        while True:
            _block = sys.stdin.buffer.read(262144)
            if not _block:
                break
            _sink.write(_block)
    finally:
        _sink.close()
    _data_path = _data_dest
if _data_path:
    _data_fh = open(_data_path, "r", encoding="utf-8")


def _records():
    """Yield the stream's records, one at a time, from the open descriptor.

    A generator, and it seeks to the start on every call, so a script may
    iterate more than once without the data ever being resident.
    """
    if _data_fh is None:
        raise RuntimeError(
            "no data stream is attached to this step. Add `stream: <state key>` "
            "naming the data_source output this script should read."
        )
    _data_fh.seek(0)
    for _line in _data_fh:
        _line = _line.strip()
        if _line:
            yield json.loads(_line)


_seccomp_error = _install_seccomp()
if _seccomp_error is not None:
    sys.stderr.write("sandbox: refusing to run, seccomp unavailable: %s\\n" % _seccomp_error)
    raise SystemExit(3)

# ctypes was the means of installing the filter; it is not for the script.
sys.modules.pop("ctypes", None)
sys.modules.pop("_ctypes", None)

# One namespace for globals AND locals. Passing two made every top-level `def`
# bind into locals while its body resolved names against globals, so any script
# whose helpers called each other died with NameError -- which is most scripts
# doing real work. Module-level code expects a single namespace; give it one.
# A data source result reaches a script as a file, never as a value in
# `state`, so the script reads it:
#
#     total = 0
#     for row in records():
#         total += row["amount"]
#     output = total
#
# `records()` is a generator over an already-open descriptor, so the script's
# memory is one record at a time whatever the result's size. `stream` is the
# raw file object for a script that wants to do its own parsing or seeking,
# and `stream_records` is the record count, known up front.
_scope = {{
    "__name__": "__sandbox__",
    "__builtins__": builtins,
    "state": _payload.get("state") or {{}},
    "records": _records,
    "stream": _data_fh,
    "stream_records": _payload.get("stream_records") or 0,
    "stream_truncated": bool(_payload.get("stream_truncated")),
    "output": None,
}}

exec(compile(_payload.get("code") or "", "<script>", "exec"), _scope, _scope)

sys.stdout.flush()
_result = json.dumps({{"output": _scope.get("output")}}, default=str)
sys.stdout.write("\\n{RESULT_MARKER}" + _result + "\\n")
sys.stdout.flush()
'''


class ScriptSandboxError(RuntimeError):
    """Raised when a sandboxed script fails to run or returns no result."""


def _as_text(value: Any) -> str:
    """Coerce a log payload to str, whether the client returned bytes or str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _unwrap_bytes_repr(text: str) -> str:
    """Undo a bytes value that was str()'d into "b'...'" before reaching us.

    Belt and braces for the client quirk described at the log-read site: if a
    caller ever gets the deserialized form anyway, the marker is still in there,
    just escaped. Only attempted when the text has exactly that shape, so real
    script output that merely happens to begin with b' is left alone.
    """
    stripped = text.strip()
    if len(stripped) > 3 and stripped[0] == "b" and stripped[1] in "'\"" and stripped[-1] == stripped[1]:
        try:
            value = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return text
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    return text


def _parse_result(stdout: Any) -> Any:
    """Extract the JSON result the bootstrap printed on its marker line."""
    stdout = _as_text(stdout)
    # Scan as-is first, then again after unwrapping a bytes repr. Testing
    # `RESULT_MARKER in stdout` to decide whether to unwrap does not work: in the
    # trapped form the marker IS present as literal characters, just never at the
    # start of a line, so that check skips the very case it means to catch.
    for candidate in (stdout, _unwrap_bytes_repr(stdout)):
        for line in reversed(candidate.splitlines()):
            if line.startswith(RESULT_MARKER):
                return json.loads(line[len(RESULT_MARKER):]).get("output")
    # Include what the script did print. The bootstrap writes the marker as its
    # last act, so reaching here means it exited before that -- and the reason is
    # almost always sitting in the output we were throwing away.
    tail = stdout.strip()[-1000:]
    raise ScriptSandboxError(
        "sandboxed script produced no result marker; output was: "
        + (repr(tail) if tail else "(empty)")
    )


def _payload_json(
    code: str,
    state: dict[str, Any],
    *,
    data_path: str | None = None,
    data_dest: str | None = None,
    data_from_stdin: bool = False,
    stream_records: int = 0,
    stream_truncated: bool = False,
) -> str:
    # ``default=str`` keeps non-JSON state values (datetimes, ObjectIds, …)
    # transportable instead of failing the whole step.
    body: dict[str, Any] = {"code": code, "state": state}
    if data_path or data_from_stdin:
        body.update(
            data_path=data_path,
            data_dest=data_dest,
            data_from_stdin=data_from_stdin,
            stream_records=stream_records,
            stream_truncated=stream_truncated,
        )
    return json.dumps(body, default=str)


# Where a streamed-in data file lands inside the sandbox container, and how
# much room it is given there. The emptyDir is disk-backed, so this is a
# quota against the node's disk rather than against the pod's memory.
_K8S_DATA_DEST = "/data/stream.jsonl"
_K8S_DATA_LIMIT_MB = int(os.environ.get("SANDBOX_DATA_LIMIT_MB", "2048"))
_DOCKER_DATA_MOUNT = "/sandbox-data/stream.jsonl"


async def run_script(
    code: str,
    state: dict[str, Any],
    *,
    runtime: str = "local",
    timeout: float = 60.0,
    image: str = "python:3.12-slim",
    memory_mb: int = 512,
    namespace: str = "langgraph",
    stream_path: str | None = None,
    stream_copy: Any = None,
    stream_records: int = 0,
    stream_truncated: bool = False,
) -> Any:
    """Run *code* in a sandbox and return the value it assigned to ``output``.

    A data source result is handed over as a file, never as a value.  How it
    gets there depends on where the sandbox runs:

    ``stream_path``
        The stream is on this pod's filesystem.  ``local`` opens the path
        directly; ``docker`` bind-mounts it read-only.  Both then have a real
        descriptor open before the seccomp filter denies ``openat``.
    ``stream_copy``
        An awaitable ``(sink) -> bytes_written`` that writes the stream's raw
        bytes into a binary sink.  Used by ``k8s``, where the sandbox is a
        different pod with no network and no shared filesystem: the bytes are
        pushed into the pod's stdin and it writes its own file.  This is the
        "stream it from one file to another" path.

    ``stream_records`` and ``stream_truncated`` are handed to the script so it
    knows how many records to expect and whether the read was cut short.
    """
    if runtime == "k8s":
        payload = _payload_json(
            code, state,
            data_dest=_K8S_DATA_DEST,
            data_from_stdin=stream_copy is not None,
            stream_records=stream_records,
            stream_truncated=stream_truncated,
        )
    elif runtime == "docker":
        payload = _payload_json(
            code, state,
            data_path=_DOCKER_DATA_MOUNT if stream_path else None,
            stream_records=stream_records,
            stream_truncated=stream_truncated,
        )
    else:
        payload = _payload_json(
            code, state,
            data_path=stream_path,
            stream_records=stream_records,
            stream_truncated=stream_truncated,
        )
    if runtime == "local":
        return await _run_local(payload, timeout=timeout, memory_mb=memory_mb)
    if runtime == "docker":
        return await _run_docker(
            payload, timeout=timeout, image=image, memory_mb=memory_mb,
            stream_path=stream_path,
        )
    if runtime == "k8s":
        return await _run_k8s(
            payload, timeout=timeout, image=image, memory_mb=memory_mb,
            namespace=namespace, stream_copy=stream_copy,
        )
    raise ValueError(
        f"Unknown sandbox runtime '{runtime}'. Valid values are: 'local', 'docker', 'k8s'."
    )


# ---------------------------------------------------------------------------
# local — child interpreter
# ---------------------------------------------------------------------------

def _rlimit_preexec(memory_mb: int, cpu_seconds: int):
    """Return a preexec_fn applying rlimits; None on platforms without them."""
    try:
        import resource  # noqa: PLC0415 — POSIX only
    except ImportError:  # pragma: no cover — Windows
        return None

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        mem = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass
        # No core dumps, and a cap on what a runaway script can write out.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))

    return _apply


async def _run_local(payload: str, *, timeout: float, memory_mb: int) -> Any:
    workdir = tempfile.mkdtemp(prefix="script-sandbox-")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-S", "-B", "-c", _BOOTSTRAP, "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env={},  # no backend env vars, no PATH → no bash, no tools
            preexec_fn=_rlimit_preexec(memory_mb, int(timeout) + 5),
            start_new_session=True,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(payload.encode()), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise ScriptSandboxError(f"sandboxed script timed out after {timeout}s") from None

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        if proc.returncode != 0:
            raise ScriptSandboxError(
                f"sandboxed script exited with code {proc.returncode}: {stderr.strip()[-2000:]}"
            )
        if stderr.strip():
            logger.debug("sandboxed script stderr: %s", stderr.strip()[-2000:])
        return _parse_result(stdout)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# docker — throw-away container
# ---------------------------------------------------------------------------

async def _run_docker(
    payload: str,
    *,
    timeout: float,
    image: str,
    memory_mb: int,
    stream_path: str | None = None,
) -> Any:
    import aiodocker

    workdir = tempfile.mkdtemp(prefix="script-sandbox-")
    payload_path = os.path.join(workdir, "payload.json")
    with open(payload_path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    os.chmod(workdir, 0o755)
    os.chmod(payload_path, 0o644)

    docker = aiodocker.Docker()
    container = None
    try:
        config = {
            "Image": image,
            "Cmd": ["python", "-I", "-S", "-B", "-c", _BOOTSTRAP, "/sandbox/payload.json"],
            "Env": [],
            "NetworkDisabled": True,
            "AttachStdout": True,
            "AttachStderr": True,
            "WorkingDir": "/tmp",
            "HostConfig": {
                # The data stream is bind-mounted read-only rather than copied
                # in: the container is on this host, so there is nothing to
                # transfer, and the bootstrap opens the path before seccomp
                # denies openat.
                "Binds": (
                    [f"{workdir}:/sandbox:ro"]
                    + ([f"{stream_path}:{_DOCKER_DATA_MOUNT}:ro"] if stream_path else [])
                ),
                "ReadonlyRootfs": True,
                "Tmpfs": {"/tmp": "rw,size=64m"},
                "Memory": memory_mb * 1024 * 1024,
                "NetworkMode": "none",
                "AutoRemove": False,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "PidsLimit": 128,
            },
        }
        container = await docker.containers.create(config=config)
        await container.start()
        try:
            await asyncio.wait_for(container.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            await container.kill()
            raise ScriptSandboxError(f"sandboxed script timed out after {timeout}s") from None

        logs = await container.log(stdout=True, stderr=True)
        stdout = "".join(logs)
        info = await container.show()
        exit_code = info.get("State", {}).get("ExitCode", 0)
        if exit_code != 0:
            raise ScriptSandboxError(
                f"sandboxed script exited with code {exit_code}: {stdout.strip()[-2000:]}"
            )
        return _parse_result(stdout)
    finally:
        if container is not None:
            try:
                await container.delete(force=True)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("docker sandbox: container cleanup failed", exc_info=True)
        await docker.close()
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# k8s — one-shot pod
# ---------------------------------------------------------------------------

class _AttachSink:
    """Binary sink that forwards each block to a pod's stdin over attach.

    A file-like ``write`` so a store's ``copy_to`` needs to know nothing about
    Kubernetes; the websocket's own buffering bounds how much is in flight.
    """

    __slots__ = ("_ws", "_written")

    def __init__(self, ws: Any) -> None:
        self._ws = ws
        self._written = 0

    def write(self, block: bytes) -> int:
        self._ws.write_stdin(block)
        self._written += len(block)
        return len(block)

    @property
    def written(self) -> int:
        return self._written


async def _stream_into_pod(
    core: Any,
    name: str,
    namespace: str,
    stream_copy: Any,
    *,
    loop: Any,
    timeout: float,
    api_timeout: float,
) -> int:
    """Write a data stream into a running pod's stdin, then close it.

    Blocks until the container is Running, because attach has nowhere to write
    before that.  Closing the socket is what gives the script's read loop its
    EOF, so it happens in a ``finally`` -- leaking an open stdin would hang the
    sandbox until its timeout rather than fail it.
    """
    from kubernetes.stream import stream as k8s_stream

    deadline = loop.time() + timeout
    while loop.time() < deadline:
        pod = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: core.read_namespaced_pod(
                    name=name, namespace=namespace, _request_timeout=api_timeout,
                ),
            ),
            timeout=api_timeout,
        )
        phase = pod.status.phase or "Pending"
        if phase == "Running":
            break
        if phase in ("Succeeded", "Failed"):
            # It finished without reading stdin at all. Nothing to send, and
            # the caller's own log read will report why.
            logger.warning(
                "k8s sandbox %s reached %s before stdin could be attached",
                name, phase,
            )
            return 0
        await asyncio.sleep(0.25)
    else:
        raise ScriptSandboxError(
            f"sandboxed script never reached Running, so its data stream could "
            f"not be delivered (waited {timeout}s)"
        )

    ws = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: k8s_stream(
                core.connect_get_namespaced_pod_attach,
                name, namespace,
                stderr=False, stdin=True, stdout=False, tty=False,
                _preload_content=False,
            ),
        ),
        timeout=api_timeout,
    )
    sink = _AttachSink(ws)
    try:
        written = await stream_copy(sink)
    finally:
        try:
            await loop.run_in_executor(None, ws.close)
        except Exception:  # noqa: BLE001 — the pod is going away regardless
            logger.debug("k8s sandbox %s: stdin close failed", name, exc_info=True)
    logger.info(
        "k8s sandbox %s: streamed %d bytes of data in on stdin", name, written
    )
    return written


async def _run_k8s(
    payload: str,
    *,
    timeout: float,
    image: str,
    memory_mb: int,
    namespace: str,
    stream_copy: Any = None,
) -> Any:
    """Run a script in a one-shot pod, streaming its data stream in on stdin.

    The sandbox pod shares nothing with the backend: no network (seccomp denies
    ``socket``), no service account, no filesystem in common.  So a data stream
    cannot be fetched by the pod and cannot be mounted into it -- the backend
    has to push it.  It goes in over the pod's stdin, in 256 KB blocks, and the
    pod's bootstrap writes it to a disk-backed ``emptyDir`` before installing
    the filter that would forbid opening it.

    stdin rather than a shared volume because the alternative on GKE is an RWX
    volume, which means Filestore: a paid NFS appliance and an IAM binding, for
    a transfer that lasts seconds.
    """
    from kubernetes import client as k8s_client, config as k8s_config

    try:
        k8s_config.load_incluster_config()
    except Exception:  # noqa: BLE001 — fall back to a local kubeconfig
        k8s_config.load_kube_config()

    core = k8s_client.CoreV1Api()
    name = f"script-sandbox-{uuid.uuid4().hex[:10]}"
    loop = asyncio.get_event_loop()

    cm_body = k8s_client.V1ConfigMap(
        metadata=k8s_client.V1ObjectMeta(name=name, labels={"app": "script-sandbox"}),
        # base64 so a payload with arbitrary bytes survives the round-trip.
        binary_data={"payload.json": base64.b64encode(payload.encode()).decode()},
    )
    pod_body = k8s_client.V1Pod(
        metadata=k8s_client.V1ObjectMeta(name=name, labels={"app": "script-sandbox"}),
        spec=k8s_client.V1PodSpec(
            restart_policy="Never",
            automount_service_account_token=False,
            enable_service_links=False,
            volumes=[
                k8s_client.V1Volume(
                    name="payload",
                    config_map=k8s_client.V1ConfigMapVolumeSource(name=name),
                ),
                k8s_client.V1Volume(
                    name="tmp", empty_dir=k8s_client.V1EmptyDirVolumeSource(medium="Memory"),
                ),
                # Deliberately NOT medium="Memory" like /tmp above: a streamed
                # data file lands here, and on a tmpfs those bytes would count
                # against the pod's own memory limit -- which is the whole
                # thing this design avoids.
                k8s_client.V1Volume(
                    name="data",
                    empty_dir=k8s_client.V1EmptyDirVolumeSource(
                        size_limit=f"{_K8S_DATA_LIMIT_MB}Mi"
                    ),
                ),
            ],
            containers=[
                k8s_client.V1Container(
                    name="script",
                    image=image,
                    command=["python", "-I", "-S", "-B", "-c", _BOOTSTRAP, "/sandbox/payload.json"],
                    env=[],
                    working_dir="/tmp",
                    volume_mounts=[
                        k8s_client.V1VolumeMount(name="payload", mount_path="/sandbox", read_only=True),
                        k8s_client.V1VolumeMount(name="tmp", mount_path="/tmp"),
                        k8s_client.V1VolumeMount(name="data", mount_path="/data"),
                    ],
                    # stdin stays open until the backend has finished writing
                    # the stream; stdin_once closes it after that single
                    # attach, so the script's read loop sees a clean EOF.
                    stdin=stream_copy is not None,
                    stdin_once=stream_copy is not None,
                    resources=k8s_client.V1ResourceRequirements(
                        limits={
                            "memory": f"{memory_mb}Mi",
                            "cpu": "1",
                            "ephemeral-storage": f"{_K8S_DATA_LIMIT_MB + 64}Mi",
                        },
                        requests={
                            "memory": f"{min(memory_mb, 128)}Mi",
                            "cpu": "100m",
                            "ephemeral-storage": "64Mi",
                        },
                    ),
                    security_context=k8s_client.V1SecurityContext(
                        allow_privilege_escalation=False,
                        read_only_root_filesystem=True,
                        run_as_non_root=True,
                        run_as_user=1000,
                        capabilities=k8s_client.V1Capabilities(drop=["ALL"]),
                    ),
                ),
            ],
        ),
    )

    # Every call below is bounded twice: `_request_timeout` stops the client from
    # blocking on a stalled connection, and asyncio.wait_for stops a wedged worker
    # thread from holding the step open.
    api_timeout = min(30.0, timeout)

    async def _api(call) -> Any:
        return await asyncio.wait_for(loop.run_in_executor(None, call), timeout=api_timeout)

    async def _cleanup() -> None:
        # Bounded like the rest, and for the same reason: cleanup is awaited in a
        # `finally`, so an unreachable API server here delays the step's error by
        # as long as the deletes take to give up -- turning a prompt failure back
        # into the hang this whole path was fixed for.
        for call in (
            lambda: core.delete_namespaced_pod(
                name=name, namespace=namespace, grace_period_seconds=0,
                _request_timeout=api_timeout,
            ),
            lambda: core.delete_namespaced_config_map(
                name=name, namespace=namespace, _request_timeout=api_timeout,
            ),
        ):
            try:
                await _api(call)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("k8s sandbox: cleanup failed for %s", name, exc_info=True)

    try:
        await _api(
            lambda: core.create_namespaced_config_map(
                namespace=namespace, body=cm_body, _request_timeout=api_timeout,
            )
        )
        await _api(
            lambda: core.create_namespaced_pod(
                namespace=namespace, body=pod_body, _request_timeout=api_timeout,
            )
        )

        # Push the data stream in before waiting for the pod to finish: the
        # bootstrap blocks on stdin until EOF, so nothing runs until this is
        # done. attach() has to wait for Running first -- there is no stdin to
        # write to while the container is still Pending.
        if stream_copy is not None:
            await _stream_into_pod(
                core, name, namespace, stream_copy, loop=loop,
                timeout=timeout, api_timeout=api_timeout,
            )

        deadline = loop.time() + timeout
        phase = "Pending"
        while loop.time() < deadline:
            pod = await _api(
                lambda: core.read_namespaced_pod(
                    name=name, namespace=namespace, _request_timeout=api_timeout,
                )
            )
            phase = pod.status.phase or "Pending"
            if phase in ("Succeeded", "Failed"):
                break
            await asyncio.sleep(1.0)
        else:
            raise ScriptSandboxError(f"sandboxed script timed out after {timeout}s")

        # A pod can report Succeeded a moment before its logs are retrievable,
        # and an empty read here is indistinguishable from a script that printed
        # nothing -- which surfaced as the useless "produced no result". Retry a
        # few times before believing the emptiness.
        stdout = ""
        for _attempt in range(5):
            # `_preload_content=False` returns the raw HTTPResponse instead of
            # letting the client deserialize. That matters: this endpoint's
            # declared response type is `str`, and the client's primitive
            # deserializer applies str() to the raw bytes -- yielding the *text*
            # "b'...\\n'", a bytes repr trapped in a string, which no amount of
            # decoding downstream can undo.
            raw_log = await _api(
                lambda: core.read_namespaced_pod_log(
                    name=name, namespace=namespace,
                    _preload_content=False, _request_timeout=api_timeout,
                )
            )
            raw_log = getattr(raw_log, "data", raw_log)
            # The client hands back bytes here, not str. Left undecoded, every
            # `line.startswith(RESULT_MARKER)` compared bytes to str and was
            # quietly False, so a script that had run perfectly was reported as
            # producing no result -- and bytes.strip() is truthy, so the
            # empty-output check waved it through too.
            stdout = _as_text(raw_log)
            if stdout.strip():
                break
            await asyncio.sleep(0.5)

        if phase == "Failed":
            raise ScriptSandboxError(f"sandboxed script failed: {stdout.strip()[-2000:]}")
        if not stdout.strip():
            # Say what the container actually did. Without this the error names
            # only the symptom, and the cause (OOMKilled, an image that never
            # started, a truncated log) is invisible.
            state = ""
            try:
                statuses = pod.status.container_statuses or []
                if statuses:
                    term = statuses[0].state.terminated if statuses[0].state else None
                    if term is not None:
                        state = (
                            f" container exit={term.exit_code} reason={term.reason!r}"
                        )
            except Exception:  # noqa: BLE001 — diagnostics must not mask the error
                pass
            raise ScriptSandboxError(
                f"sandboxed script produced no output (phase={phase}{state})"
            )
        return _parse_result(stdout)
    finally:
        await _cleanup()
