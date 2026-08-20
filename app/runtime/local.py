from __future__ import annotations

"""LocalRuntime — pi-cloud-agent as a child process on the host.

The local runtime runs the same agent as ``docker`` and ``k8s``:
`pi-cloud-agent <https://github.com/comtihon/pi-cloud-agent>`_, an HTTP server
around the pi coding agent.  Instead of a container or a Helm release it is
started as a subprocess of the backend, listening on a free localhost port, and
driven over the identical HTTP protocol (``/health``, ``/start``, ``/poll``,
``/terminate``).

Configure where the checkout lives with ``LOCAL_AGENT_DIR`` (and, if the
entrypoint differs from ``node src/server.js``, ``LOCAL_AGENT_COMMAND``).  There
is no inline in-process agent any more: one agent implementation serves all
three runtimes, so a workflow behaves the same locally as in a cluster.
"""

import asyncio
import contextlib
import logging
import os
import shlex
import socket
from collections import deque
from typing import TYPE_CHECKING, Any

import httpx

from app.runtime.base import AgentRuntime

if TYPE_CHECKING:
    from app.domain.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)

DEFAULT_AGENT_COMMAND = "node src/server.js"
_HEALTH_TIMEOUT = 300.0       # default seconds to wait for /health to return 200
_HEALTH_POLL_INTERVAL = 0.5   # seconds between health-check attempts
_STATE_CHECK_INTERVAL = 5.0   # seconds between "did the process die?" checks
_LOG_TAIL_LINES = 30          # lines of agent stdout/stderr kept for error reports
_TERMINATE_GRACE = 5.0        # seconds to wait for a graceful exit before SIGKILL


class _AgentProcess:
    """A running agent subprocess and the tail of its output."""

    def __init__(self, process: asyncio.subprocess.Process, agent_url: str, run_id: str) -> None:
        self.process = process
        self.agent_url = agent_url
        self.run_id = run_id
        self.log_tail: deque[str] = deque(maxlen=_LOG_TAIL_LINES)
        self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        """Keep the last lines of the agent's output so failures are explainable.

        Draining is not optional: an unread pipe fills and the agent blocks on
        its own logging.
        """
        stream = self.process.stdout
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                self.log_tail.append(text)
                logger.debug("LocalRuntime[%s]: %s", self.run_id, text)
        except Exception:  # pragma: no cover - drain must never kill a run
            pass

    def tail(self) -> str:
        return "\n".join(self.log_tail).strip() or "(no output)"

    def is_running(self) -> bool:
        return self.process.returncode is None

    async def stop(self) -> None:
        """Terminate the process, then kill it if it ignores SIGTERM."""
        self._drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._drain_task
        if self.process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=_TERMINATE_GRACE)
        except asyncio.TimeoutError:
            logger.warning(
                "LocalRuntime: agent for run %s ignored SIGTERM — killing", self.run_id,
            )
            with contextlib.suppress(ProcessLookupError):
                self.process.kill()
            with contextlib.suppress(Exception):
                await self.process.wait()


# Process registry, module level on purpose: ``get_runtime()`` builds a fresh
# LocalRuntime for every step execution, so per-instance state would be lost
# between the spawn and the resume of the same run.  Docker and K8s get this for
# free from the daemon / API server; a subprocess has only this process to
# remember it.
_PROCESSES: dict[str, "_AgentProcess"] = {}   # agent_url → process
_RUNS: dict[str, str] = {}                    # run_id → agent_url


def _free_port() -> int:
    """Return a port the agent can bind.

    Bind-and-release rather than a fixed port so several agents can run at once;
    the window between release and the agent's own bind is the usual accepted
    race for this pattern.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalRuntime(AgentRuntime):
    """Spawn pi-cloud-agent as a subprocess and speak the agent HTTP protocol.

    Environment variables injected into the child
    ---------------------------------------------
    AGENT_PORT            — TCP port the server binds (a free localhost port)
    BACKEND_CALLBACK_URL  — backend base URL for agent-to-backend callbacks
    RUN_ID                — workflow run identifier

    The backend's own environment is inherited so that ``node``, ``git`` and the
    credentials the agent needs are on PATH, with ``extra_env`` (the step's
    resolved ``env_vars``) layered on top.
    """

    def __init__(
        self,
        agent_dir: str | None = None,
        agent_command: str | None = None,
    ) -> None:
        self._agent_dir = agent_dir or ""
        self._agent_command = agent_command or DEFAULT_AGENT_COMMAND

    async def spawn(
        self,
        agent_def: "AgentDefinition",
        step: dict[str, Any],
        run_id: str,
        callback_base_url: str,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        """Start the agent process and return its base URL once healthy.

        Raises
        ------
        ValueError
            When no agent directory is configured, or it does not exist.
        RuntimeError
            When the process exits or fails to answer ``/health`` in time.
        """
        if not self._agent_dir:
            raise ValueError(
                "LocalRuntime: no agent directory configured. Set LOCAL_AGENT_DIR to a "
                "pi-cloud-agent checkout, or run the agent with runtime 'docker' / 'k8s'."
            )
        if not os.path.isdir(self._agent_dir):
            raise ValueError(
                f"LocalRuntime: LOCAL_AGENT_DIR '{self._agent_dir}' is not a directory."
            )

        health_timeout = float(
            step.get("health_timeout") or agent_def.health_timeout or _HEALTH_TIMEOUT
        )
        port = _free_port()
        agent_url = f"http://127.0.0.1:{port}"

        env = {
            **os.environ,
            **(extra_env or {}),
            "AGENT_PORT": str(port),
            "BACKEND_CALLBACK_URL": callback_base_url,
            "RUN_ID": run_id,
        }

        argv = shlex.split(self._agent_command)
        logger.info(
            "LocalRuntime: starting %s in %s on port %d (run_id=%s)",
            self._agent_command, self._agent_dir, port, run_id,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self._agent_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise RuntimeError(
                f"LocalRuntime: failed to start '{self._agent_command}' in "
                f"'{self._agent_dir}': {exc}"
            ) from exc

        agent_process = _AgentProcess(process, agent_url, run_id)
        _PROCESSES[agent_url] = agent_process
        _RUNS[run_id] = agent_url

        # Poll GET /health until the server is ready, checking every few seconds
        # that the process is still alive so a crash fails fast instead of
        # waiting out the whole timeout.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + health_timeout
        last_state_check = loop.time()
        async with httpx.AsyncClient() as client:
            while loop.time() < deadline:
                try:
                    resp = await client.get(f"{agent_url}/health", timeout=2.0)
                    if resp.status_code == 200:
                        logger.info("LocalRuntime: agent at %s is healthy", agent_url)
                        return agent_url
                except Exception:
                    pass

                now = loop.time()
                if now - last_state_check >= _STATE_CHECK_INTERVAL or not agent_process.is_running():
                    last_state_check = now
                    if not agent_process.is_running():
                        tail = agent_process.tail()
                        code = process.returncode
                        await self.terminate(agent_url)
                        raise RuntimeError(
                            f"LocalRuntime: agent process exited before becoming healthy "
                            f"(exit code {code}, run_id={run_id}). Last output:\n{tail}"
                        )

                await asyncio.sleep(_HEALTH_POLL_INTERVAL)

        tail = agent_process.tail()
        await self.terminate(agent_url)
        raise RuntimeError(
            f"LocalRuntime: agent at {agent_url} did not become healthy within "
            f"{health_timeout}s (run_id={run_id}). Last output:\n{tail}"
        )

    async def terminate(self, agent_url: str) -> None:
        """Call POST /terminate, then stop the process. Idempotent."""
        with contextlib.suppress(Exception):
            async with httpx.AsyncClient() as client:
                await client.post(f"{agent_url}/terminate", timeout=3.0)

        agent_process = _PROCESSES.pop(agent_url, None)
        if agent_process is None:
            logger.debug("LocalRuntime.terminate: no process tracked for %s", agent_url)
            return
        for rid in [rid for rid, url in _RUNS.items() if url == agent_url]:
            _RUNS.pop(rid, None)
        await agent_process.stop()
        logger.info(
            "LocalRuntime: agent for run %s stopped (url=%s)", agent_process.run_id, agent_url,
        )

    async def is_alive(self, agent_url: str) -> bool:
        """Call GET {agent_url}/health and return True if status is 200."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{agent_url}/health", timeout=2.0)
                return resp.status_code == 200
        except Exception:
            return False

    async def has_container_for_run(self, agent_def: "AgentDefinition", run_id: str) -> bool:
        """Return True while this run's agent process is still alive.

        The executor uses this to tell a LangGraph node re-execution (resume)
        from a first execution and skip a second spawn.  ``agent_def`` is taken
        for signature parity with the other runtimes.
        """
        agent_url = _RUNS.get(run_id)
        if agent_url is None:
            return False
        agent_process = _PROCESSES.get(agent_url)
        return agent_process is not None and agent_process.is_running()

    async def get_agent_url_for_run(
        self, agent_def: "AgentDefinition", run_id: str
    ) -> str | None:
        """Return the URL of this run's agent, or None when it is gone."""
        agent_url = _RUNS.get(run_id)
        if agent_url is None:
            return None
        agent_process = _PROCESSES.get(agent_url)
        return agent_url if agent_process is not None and agent_process.is_running() else None

    async def terminate_by_run_id(
        self, agent_def: "AgentDefinition | None", run_id: str
    ) -> None:
        """Stop this run's agent whatever URL it was given."""
        agent_url = _RUNS.get(run_id)
        if agent_url is None:
            return
        await self.terminate(agent_url)
