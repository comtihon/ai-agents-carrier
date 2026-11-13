"""A sandboxed python step can work on the run's workspace archive.

The docker/k8s container restores ``workspace.tar.gz`` from a pre-signed GET
URL into a volume of its own, runs the script there (file syscalls allowed,
still no network and no processes), and PUTs the archive back only when a file
changed. These tests run that wrapper on the host against a local HTTP server
standing in for GCS -- the same code the container runs.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.infrastructure.orchestration import script_sandbox as sb

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or __import__("platform").machine() != "x86_64",
    reason="the sandbox's seccomp filter is x86_64 Linux only",
)


def _archive(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo("./" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _names(blob: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        return {
            m.name[2:]: tar.extractfile(m).read()
            for m in tar.getmembers() if m.isfile()
        }


@pytest.fixture()
def gcs():
    """A one-object store: GET (404 when empty) and PUT, with generations."""
    store: dict = {"obj": None, "gen": 0, "puts": 0, "put_status": 200}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if store["obj"] is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("x-goog-generation", str(store["gen"]))
            self.end_headers()
            self.wfile.write(store["obj"])

        def do_PUT(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            if store["put_status"] != 200:
                self.send_response(store["put_status"])
                self.end_headers()
                return
            store["obj"], store["gen"], store["puts"] = body, store["gen"] + 1, store["puts"] + 1
            store["content_type"] = self.headers["content-type"]
            self.send_response(200)
            self.send_header("x-goog-generation", str(store["gen"]))
            self.end_headers()

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    store["url"] = f"http://127.0.0.1:{srv.server_port}/ws.tar.gz"
    yield store
    srv.shutdown()


def _run(tmp_path, gcs, code, exclude=None):
    """Run the container's wrapper on the host; return (exit, output, report, stdout)."""
    ws = {"download_url": gcs["url"], "upload_url": gcs["url"],
          "upload_headers": {"content-type": "application/gzip"}}
    if exclude:
        ws["exclude"] = exclude
    payload = json.loads(sb._payload_json(code, {"n": 2}, workspace=ws))
    # The container's paths, moved under tmp_path.
    payload["workspace_dir"] = payload["workspace"]["dir"] = str(tmp_path / "workspace")
    payload["workspace"]["scratch"] = str(tmp_path / "scratch")
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))
    proc = subprocess.run(
        [sys.executable, "-I", "-c", sb._WORKSPACE_WRAPPER, str(path)],
        capture_output=True, text=True, timeout=60,
    )
    out = proc.stdout + proc.stderr
    report = sb._parse_workspace(out)
    output = sb._parse_result(out) if proc.returncode == 0 else None
    return proc.returncode, output, report, out


def test_script_reads_and_changes_the_workspace_and_it_is_uploaded(tmp_path, gcs):
    gcs["obj"] = _archive({"a.txt": b"hello", "data/b.txt": b"world", "x.log": b"noise"})
    code, output, report, _ = _run(tmp_path, gcs, """
import os
seen = sorted(os.path.join(r, f)[2:] for r, _, fs in os.walk('.') for f in fs)
with open('a.txt', 'a') as fh:
    fh.write('!')
with open('summary.txt', 'w') as fh:
    fh.write(open('data/b.txt').read().upper())
os.remove('data/b.txt')
output = {'seen': seen, 'cwd': workspace, 'n': state['n']}
""", exclude=["*.log"])

    assert code == 0
    assert output["seen"] == ["a.txt", "data/b.txt", "x.log"]
    assert output["cwd"] == str(tmp_path / "workspace") and output["n"] == 2
    assert report["restore"] == "restored" and report["uploaded"] is True
    assert report["changes"] == {
        "added": ["summary.txt"], "modified": ["a.txt"], "deleted": ["data/b.txt"],
        "added_count": 1, "modified_count": 1, "deleted_count": 1,
    }
    assert report["version"] == "1" and report["bytes"] > 0
    assert gcs["content_type"] == "application/gzip"
    # Excluded files stay out of the archive, like the runner's tar --exclude.
    assert _names(gcs["obj"]) == {"a.txt": b"hello!", "summary.txt": b"WORLD"}


def test_nothing_is_uploaded_when_no_file_changed(tmp_path, gcs):
    gcs["obj"] = _archive({"a.txt": b"hello"})
    code, output, report, _ = _run(tmp_path, gcs, "output = open('a.txt').read()")

    assert (code, output) == (0, "hello")
    assert report["uploaded"] is False and report["changes"]["modified"] == []
    assert gcs["puts"] == 0


def test_the_first_step_of_a_run_starts_empty(tmp_path, gcs):
    code, _, report, _ = _run(tmp_path, gcs, "open('out.json', 'w').write('{}')\noutput = 1")

    assert code == 0 and report["restore"] == "empty"
    assert report["changes"]["added"] == ["out.json"] and gcs["puts"] == 1


def test_a_failing_script_uploads_nothing_and_keeps_its_traceback(tmp_path, gcs):
    gcs["obj"] = _archive({"a.txt": b"hello"})
    code, _, report, out = _run(tmp_path, gcs, "open('a.txt', 'w').write('half')\nraise ValueError('boom')")

    assert code != 0 and report["script_exit"] == code
    assert "ValueError: boom" in out
    assert gcs["puts"] == 0
    assert sb._workspace_error(out) is None  # the script's own error is reported, not the wrapper's


def test_the_script_still_has_no_network(tmp_path, gcs):
    code, _, _, out = _run(tmp_path, gcs, "import socket")
    assert code != 0 and "not available in sandboxed scripts" in out


def test_a_failed_upload_fails_the_step(tmp_path, gcs):
    gcs["put_status"] = 403
    code, _, report, out = _run(tmp_path, gcs, "open('x', 'w').write('1')\noutput = 1")
    assert code == 5 and sb._workspace_error(out) == "PUT workspace archive: HTTP 403"


async def test_a_workspace_is_refused_on_the_local_runtime():
    with pytest.raises(ValueError, match="docker' or 'k8s"):
        await sb.run_script("output = 1", {}, runtime="local", workspace={"download_url": "x"})


def test_the_bootstrap_alone_still_cannot_open_files(tmp_path):
    """Without a workspace the seccomp allow-list is unchanged: no openat."""
    payload = tmp_path / "p.json"
    payload.write_text(sb._payload_json("output = open('/etc/hostname').read()", {}))
    proc = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", sb._BOOTSTRAP, str(payload)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0 and "PermissionError" in proc.stderr


# ─── The python step ──────────────────────────────────────────────────────────

async def _run_step(step: dict, monkeypatch, state: dict | None = None):
    """Run one python step through the graph with run_script stubbed out."""
    from types import SimpleNamespace

    from app.infrastructure.orchestration import yaml_graph
    from app.steps import acp_runner_workspace

    calls: list[dict] = []

    async def fake_run_script(code, script_state, **kw):
        calls.append(kw)
        if kw.get("workspace_report") is not None:
            kw["workspace_report"].update({"restore": "restored", "uploaded": True,
                                          "changes": {"added": ["x"]}})
        return "done"

    monkeypatch.setattr(sb, "run_script", fake_run_script)
    monkeypatch.setattr(acp_runner_workspace, "_sign_urls",
                        lambda b, o, t: (f"https://get/{b}/{o}", f"https://put/{b}/{o}"))
    runner = yaml_graph.YamlGraphRunner({"id": "g", "steps": [step]}, object(), lambda *a, **k: [])
    runner._current_run = SimpleNamespace(id="run-7")
    final: dict = {}
    async for chunk in runner.graph.astream(
        {"request": "go", **(state or {})}, {"configurable": {"thread_id": "t"}}, stream_mode="updates",
    ):
        for update in chunk.values():
            final.update(update or {})
    return final, calls


async def test_a_python_step_gets_the_runs_workspace(monkeypatch):
    step = {"id": "crunch", "type": "python", "code": "output = 1", "sandbox_runtime": "k8s", "output_key": "crunch",
            "workspace": {"bucket": "b", "path": "runs/{workflow_id}/", "exclude": ["*.tmp"]}}
    final, calls = await _run_step(step, monkeypatch)

    assert calls[0]["runtime"] == "k8s"
    assert calls[0]["workspace"] == {
        "download_url": "https://get/b/runs/run-7/workspace.tar.gz",
        "upload_url": "https://put/b/runs/run-7/workspace.tar.gz",
        "upload_headers": {"content-type": "application/gzip"},
        "exclude": ["*.tmp"],
    }
    assert final["crunch"] == "done"
    assert final["_workspace_crunch"] == {
        "path": "gs://b/runs/run-7/workspace.tar.gz",
        "restore": "restored", "uploaded": True, "changes": {"added": ["x"]},
    }


@pytest.mark.parametrize("extra", [{"sandbox_runtime": "local"}, {}, {"sandbox": False}])
async def test_a_workspace_is_refused_outside_docker_and_k8s(monkeypatch, extra):
    step = {"id": "crunch", "type": "python", "code": "output = 1", "output_key": "crunch",
            "workspace": {"bucket": "b", "path": "p"}, **extra}
    final, calls = await _run_step(step, monkeypatch)

    assert calls == []
    assert "docker' or 'k8s" in final["crunch"]["error"]
