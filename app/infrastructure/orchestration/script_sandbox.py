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
)

# Executed by the sandboxed interpreter.  argv[1] is "-" (read payload from
# stdin) or a path to a JSON payload file.
_BOOTSTRAP = f'''
import base64, builtins, json, os, sys

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

_scope = {{"state": _payload.get("state") or {{}}, "output": None}}
_globals = {{"__name__": "__sandbox__", "__builtins__": builtins}}

exec(compile(_payload.get("code") or "", "<script>", "exec"), _globals, _scope)

sys.stdout.flush()
_result = json.dumps({{"output": _scope.get("output")}}, default=str)
sys.stdout.write("\\n{RESULT_MARKER}" + _result + "\\n")
sys.stdout.flush()
'''


class ScriptSandboxError(RuntimeError):
    """Raised when a sandboxed script fails to run or returns no result."""


def _parse_result(stdout: str) -> Any:
    """Extract the JSON result the bootstrap printed on its marker line."""
    for line in reversed(stdout.splitlines()):
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


def _payload_json(code: str, state: dict[str, Any]) -> str:
    # ``default=str`` keeps non-JSON state values (datetimes, ObjectIds, …)
    # transportable instead of failing the whole step.
    return json.dumps({"code": code, "state": state}, default=str)


async def run_script(
    code: str,
    state: dict[str, Any],
    *,
    runtime: str = "local",
    timeout: float = 60.0,
    image: str = "python:3.12-slim",
    memory_mb: int = 512,
    namespace: str = "langgraph",
) -> Any:
    """Run *code* in a sandbox and return the value it assigned to ``output``."""
    payload = _payload_json(code, state)
    if runtime == "local":
        return await _run_local(payload, timeout=timeout, memory_mb=memory_mb)
    if runtime == "docker":
        return await _run_docker(payload, timeout=timeout, image=image, memory_mb=memory_mb)
    if runtime == "k8s":
        return await _run_k8s(
            payload, timeout=timeout, image=image, memory_mb=memory_mb, namespace=namespace,
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

async def _run_docker(payload: str, *, timeout: float, image: str, memory_mb: int) -> Any:
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
                "Binds": [f"{workdir}:/sandbox:ro"],
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

async def _run_k8s(
    payload: str, *, timeout: float, image: str, memory_mb: int, namespace: str,
) -> Any:
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
                    ],
                    resources=k8s_client.V1ResourceRequirements(
                        limits={"memory": f"{memory_mb}Mi", "cpu": "1"},
                        requests={"memory": f"{min(memory_mb, 128)}Mi", "cpu": "100m"},
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
            stdout = await _api(
                lambda: core.read_namespaced_pod_log(
                    name=name, namespace=namespace, _request_timeout=api_timeout,
                )
            ) or ""
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
