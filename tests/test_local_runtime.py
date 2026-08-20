"""LocalRuntime runs pi-cloud-agent as a child process, not an inline agent.

The tests stand in a tiny Python HTTP server for the real Node agent: what
matters here is the contract the runtime relies on (a free port, ``AGENT_PORT``,
``/health``, ``/terminate``, run-scoped bookkeeping), not pi itself.
"""
from __future__ import annotations

import os
import textwrap

import pytest

from app.domain.models.agent_definition import AgentDefinition
from app.runtime import local as local_runtime
from app.runtime.factory import get_runtime
from app.runtime.local import LocalRuntime

# A stand-in agent: serves /health, records the env it was given, and exits on
# /terminate. Written to disk per test so the runtime's cwd/argv path is real.
_FAKE_AGENT = textwrap.dedent('''
    import json, os
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({
                "status": "ok",
                "run_id": os.environ.get("RUN_ID"),
                "callback": os.environ.get("BACKEND_CALLBACK_URL"),
                "extra": os.environ.get("EXTRA_TOKEN"),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            raise SystemExit(0)

        def log_message(self, *_args):
            pass

    HTTPServer(("127.0.0.1", int(os.environ["AGENT_PORT"])), Handler).serve_forever()
''')

_CRASHING_AGENT = textwrap.dedent('''
    import sys
    print("boom: cannot find pi", file=sys.stdout, flush=True)
    sys.exit(3)
''')


@pytest.fixture(autouse=True)
def _clean_registry():
    """The process registry is module state — keep tests independent."""
    yield
    local_runtime._PROCESSES.clear()
    local_runtime._RUNS.clear()


@pytest.fixture
def agent_dir(tmp_path):
    def _write(source: str) -> str:
        (tmp_path / "server.py").write_text(source)
        return str(tmp_path)

    return _write


def _agent_def() -> AgentDefinition:
    return AgentDefinition(id="researcher", name="Researcher", health_timeout=20)


async def test_spawn_starts_the_agent_and_waits_for_health(agent_dir):
    import httpx

    runtime = LocalRuntime(
        agent_dir=agent_dir(_FAKE_AGENT), agent_command="python3 server.py",
    )

    url = await runtime.spawn(
        _agent_def(), {}, "run-1", "http://localhost:8000",
        extra_env={"EXTRA_TOKEN": "from-step"},
    )
    try:
        assert url.startswith("http://127.0.0.1:")
        async with httpx.AsyncClient() as client:
            payload = (await client.get(f"{url}/health", timeout=5.0)).json()
        # The child gets the run id, the callback URL and the step's env_vars.
        assert payload["run_id"] == "run-1"
        assert payload["callback"] == "http://localhost:8000"
        assert payload["extra"] == "from-step"
        assert await runtime.is_alive(url) is True
    finally:
        await runtime.terminate(url)

    assert await runtime.is_alive(url) is False


async def test_two_agents_get_different_ports(agent_dir):
    runtime = LocalRuntime(agent_dir=agent_dir(_FAKE_AGENT), agent_command="python3 server.py")

    first = await runtime.spawn(_agent_def(), {}, "run-1", "http://localhost:8000")
    second = await runtime.spawn(_agent_def(), {}, "run-2", "http://localhost:8000")
    try:
        assert first != second
    finally:
        await runtime.terminate(first)
        await runtime.terminate(second)


async def test_a_crashing_agent_fails_fast_with_its_output(agent_dir):
    runtime = LocalRuntime(agent_dir=agent_dir(_CRASHING_AGENT), agent_command="python3 server.py")

    with pytest.raises(RuntimeError) as exc:
        await runtime.spawn(
            # A long health timeout must not be waited out when the process is gone.
            AgentDefinition(id="a", health_timeout=300), {}, "run-1", "http://localhost:8000",
        )

    assert "exited before becoming healthy" in str(exc.value)
    assert "boom: cannot find pi" in str(exc.value)


async def test_health_timeout_is_reported(agent_dir):
    """A process that never binds the port must not hang the run forever."""
    runtime = LocalRuntime(
        agent_dir=agent_dir("import time; time.sleep(30)"), agent_command="python3 server.py",
    )

    with pytest.raises(RuntimeError) as exc:
        await runtime.spawn(_agent_def(), {"health_timeout": 1}, "run-1", "http://localhost:8000")

    assert "did not become healthy within 1.0s" in str(exc.value)


async def test_run_scoped_lookups_survive_a_fresh_runtime_instance(agent_dir):
    """The executor builds a new runtime per step: resume must still find the agent.

    Docker and K8s recover this from the daemon / API server; a subprocess has
    only the module-level registry.
    """
    directory = agent_dir(_FAKE_AGENT)
    spawner = LocalRuntime(agent_dir=directory, agent_command="python3 server.py")
    url = await spawner.spawn(_agent_def(), {}, "run-1", "http://localhost:8000")

    resumer = LocalRuntime(agent_dir=directory, agent_command="python3 server.py")
    try:
        assert await resumer.has_container_for_run(_agent_def(), "run-1") is True
        assert await resumer.get_agent_url_for_run(_agent_def(), "run-1") == url
        assert await resumer.has_container_for_run(_agent_def(), "other-run") is False
    finally:
        await resumer.terminate_by_run_id(_agent_def(), "run-1")

    assert await resumer.has_container_for_run(_agent_def(), "run-1") is False
    assert await resumer.get_agent_url_for_run(_agent_def(), "run-1") is None


async def test_terminate_is_idempotent(agent_dir):
    runtime = LocalRuntime(agent_dir=agent_dir(_FAKE_AGENT), agent_command="python3 server.py")
    url = await runtime.spawn(_agent_def(), {}, "run-1", "http://localhost:8000")

    await runtime.terminate(url)
    await runtime.terminate(url)  # must not raise
    await runtime.terminate_by_run_id(_agent_def(), "run-1")


async def test_spawn_without_a_configured_directory_is_a_config_error():
    with pytest.raises(ValueError, match="LOCAL_AGENT_DIR"):
        await LocalRuntime().spawn(_agent_def(), {}, "run-1", "http://localhost:8000")


async def test_spawn_with_a_missing_directory_is_a_config_error(tmp_path):
    missing = str(tmp_path / "nope")
    with pytest.raises(ValueError, match="is not a directory"):
        await LocalRuntime(agent_dir=missing).spawn(
            _agent_def(), {}, "run-1", "http://localhost:8000",
        )


def test_the_factory_passes_the_local_agent_configuration():
    runtime = get_runtime(
        "local", local_agent_dir="/opt/pi-cloud-agent", local_agent_command="npm start",
    )

    assert isinstance(runtime, LocalRuntime)
    assert runtime._agent_dir == "/opt/pi-cloud-agent"
    assert runtime._agent_command == "npm start"


def test_the_default_command_starts_the_pi_cloud_agent_server():
    assert local_runtime.DEFAULT_AGENT_COMMAND == "node src/server.js"


def test_there_is_no_inline_local_agent_left():
    """The `local` runtime must be the same agent as docker/k8s, not a second one."""
    assert not os.path.exists(
        os.path.join(os.path.dirname(local_runtime.__file__), "..", "agents", "local_agent.py")
    )
