"""Runners: number-matching enrollment, runner sockets, and runner agent steps
(including parking on a usage limit and resuming the same session)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import uvicorn
import websockets
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

import app.steps.acp_executor as acp_executor
import app.steps.acp_runner_workspace as acp_runner_workspace
from app.api.app import create_app
from app.application.runners import get_runner_hub, sha256
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.runner import Runner
from app.infrastructure.auth.authorization import Subject, reset_current_subject, set_current_subject
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.agent_backend import AgentDefinitionBackend
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.persistence.runner_backend import InMemoryRunnerBackend
from app.infrastructure.tools.mcp_client import McpToolsProvider
from app.steps.acp_executor import CONTINUE_PROMPT, run_acp_agent


class MemoryAgents(AgentDefinitionBackend):
    def __init__(self) -> None:
        self.docs: dict[str, AgentDefinition] = {}

    async def list(self):
        return list(self.docs.values())

    async def get(self, agent_id):
        return self.docs.get(agent_id)

    async def create(self, definition):
        definition.touch()
        self.docs[definition.id] = definition
        return definition

    async def update(self, agent_id, definition):
        definition.touch()
        self.docs[agent_id] = definition
        return definition

    async def delete(self, agent_id):
        self.docs.pop(agent_id, None)


def _with_test_subject(app):
    """Bind the subject named by X-Test-User, as OAuthMiddleware would."""

    async def wrapped(scope, receive, send):
        user = dict(scope.get("headers") or ()).get(b"x-test-user") if scope["type"] == "http" else None
        if not user:
            return await app(scope, receive, send)
        name = user.decode()
        token = set_current_subject(Subject(id=name, name=name.title(), is_super_admin=name == "admin"))
        try:
            return await app(scope, receive, send)
        finally:
            reset_current_subject(token)

    return wrapped


class Server:
    def __init__(self) -> None:
        self.runners = InMemoryRunnerBackend()
        self.agents = MemoryAgents()
        self.port = 0

    @property
    def http(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1/acp/runners"

    @property
    def ws(self) -> str:
        return f"ws://127.0.0.1:{self.port}/api/v1/acp/runners/connect"

    def client(self, user: str | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.http, headers={"x-test-user": user} if user else {}, timeout=10)


@pytest.fixture
async def server():
    s = Server()
    mcp = MagicMock(spec=McpToolsProvider)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    container = ApplicationContainer(
        settings=Settings(),
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=AsyncMock(spec=MongoGraphRunRepository),
        openhands=MagicMock(spec=OpenHandsAdapter),
        agent_backend=s.agents,
        runner_backend=s.runners,
    )
    app = create_app()
    app.state.container = container
    hub = get_runner_hub()
    hub.backend = s.runners
    hub.sessions.clear()
    srv = uvicorn.Server(uvicorn.Config(_with_test_subject(app), host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.01)
    s.port = srv.servers[0].sockets[0].getsockname()[1]
    yield s
    srv.should_exit = True
    await task
    hub.sessions.clear()


# ─── enrollment ──────────────────────────────────────────────────────────────


async def _enroll(s: Server) -> dict:
    async with s.client() as c:
        r = await c.post("/enrollments", json={
            "hostname": "val-laptop", "os": "linux", "arch": "x64", "tz": "Europe/Berlin", "version": "0.2.0",
            "agents": ["claude", "codex", "bad name!"], "default_agent": "claude",
        })
    assert r.status_code == 201, r.text
    return r.json()


async def test_enrollment_by_number(server):
    e = await _enroll(server)
    assert len(e["code"]) == 6 and e["code"].isdigit()
    poll_headers = {"Authorization": f"Bearer {e['poll_token']}"}
    async with server.client() as anon, server.client("alice") as alice, server.client("bob") as bob:
        assert (await anon.get(f"/enrollments/{e['enrollment_id']}/poll", headers=poll_headers)).json() == {"status": "pending"}
        assert (await anon.get(f"/enrollments/{e['enrollment_id']}/poll")).status_code == 404

        wrong = "000000" if e["code"] != "000000" else "111111"
        assert (await alice.post("/enrollments/lookup", json={"code": wrong})).status_code == 404
        shown = f"{e['code'][:3]} {e['code'][3:]}"
        found = (await alice.post("/enrollments/lookup", json={"code": shown})).json()
        assert found["hostname"] == "val-laptop"
        assert found["agents"] == ["claude", "codex"]
        assert found["request_ip"] == "127.0.0.1"

        assert (await alice.post(f"/enrollments/{e['enrollment_id']}/approve", json={"code": wrong})).status_code == 404
        r = await alice.post(f"/enrollments/{e['enrollment_id']}/approve", json={"code": e["code"], "name": "Alice laptop", "agents": ["claude"]})
        assert r.status_code == 200, r.text
        body = r.json()
        runner = body["runner"]
        assert runner["name"] == "Alice laptop" and runner["owner_id"] == "alice" and runner["visibility"] == "private"
        assert "secret_hash" not in runner
        assert [a["acp_agent"] for a in body["agents"]] == ["claude"]
        agent = body["agents"][0]
        assert agent["default_runtime"] == "runner" and agent["runner_id"] == runner["id"] and agent["protocol"] == "acp"
        assert agent["owner_id"] == "alice" and agent["visibility"] == "private"

        # A decided enrollment cannot be approved again.
        assert (await bob.post(f"/enrollments/{e['enrollment_id']}/approve", json={"code": e["code"]})).status_code == 404

        approved = (await anon.get(f"/enrollments/{e['enrollment_id']}/poll", headers=poll_headers)).json()
        assert approved["status"] == "approved" and approved["runner_id"] == runner["id"]
        assert approved["token"].startswith(runner["id"] + ".")
        assert (await anon.get(f"/enrollments/{e['enrollment_id']}/poll", headers=poll_headers)).status_code == 410

        assert [r["id"] for r in (await alice.get(server.http)).json()] == [runner["id"]]
        assert (await bob.get(server.http)).json() == []
        listed = (await alice.get(server.http)).json()[0]
        assert listed["online"] is False
        assert listed["agent_definitions"] == [{"id": agent["id"], "name": agent["name"], "acp_agent": "claude"}]

        # bob cannot point his own agent at alice's machine
        r = await bob.post("http://127.0.0.1:%d/api/v1/agents" % server.port, json={
            "id": "sneaky", "default_runtime": "runner", "runner_id": runner["id"], "acp_agent": "claude",
        })
        assert r.status_code == 400
        r = await alice.post(f"/{runner['id']}/agents", json={"acp_agent": "codex"})
        assert r.status_code == 201 and r.json()["acp_agent"] == "codex"
        assert (await bob.delete(f"/{runner['id']}")).status_code == 404


async def test_wrong_numbers_are_rate_limited(server):
    async with server.client("mallory") as c:
        e = await _enroll(server)
        wrong = [f"{i:06d}" for i in range(10) if f"{i:06d}" != e["code"]][:5]
        for code in wrong:
            assert (await c.post("/enrollments/lookup", json={"code": code})).status_code == 404
        r = await c.post("/enrollments/lookup", json={"code": e["code"]})
        assert r.status_code == 429


async def test_deny_and_expiry(server):
    e = await _enroll(server)
    async with server.client("alice") as alice, server.client() as anon:
        assert (await alice.post(f"/enrollments/{e['enrollment_id']}/deny", json={"code": e["code"]})).json() == {"status": "denied"}
        r = await anon.get(f"/enrollments/{e['enrollment_id']}/poll", headers={"Authorization": f"Bearer {e['poll_token']}"})
        assert r.json() == {"status": "denied"}
    e2 = await _enroll(server)
    server.runners.enrollments[e2["enrollment_id"]].expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    async with server.client() as anon, server.client("alice") as alice:
        r = await anon.get(f"/enrollments/{e2['enrollment_id']}/poll", headers={"Authorization": f"Bearer {e2['poll_token']}"})
        assert r.json() == {"status": "expired"}
        assert (await alice.post("/enrollments/lookup", json={"code": e2["code"]})).status_code == 404


async def test_public_endpoints(server):
    async with server.client() as anon:
        info = (await anon.get("/info")).json()
        assert info["runners"] is True
        text = (await anon.get("/install.md")).text
        assert "acp-web-proxy install" in text


# ─── sockets ─────────────────────────────────────────────────────────────────


async def _runner(s: Server, owner: str = "alice", **over: Any) -> tuple[Runner, str]:
    r = Runner(
        id="rnr-1", name="laptop", hostname="val-laptop", os="linux", tz="UTC",
        agents=["claude", "codex"], default_agent="claude", secret_hash=sha256("sekret"),
        owner_id=owner, visibility="private", **over,
    )
    await s.runners.save_runner(r)
    return r, "rnr-1.sekret"


def _chunk(text: str) -> dict:
    return {"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "s1", "update": {
        "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}}}


class FakeRunner:
    """acp-web-proxy in dial mode with a scripted ACP agent behind it."""

    LIMIT_ERROR = {"code": -32603, "message": "Internal error", "data": {
        "message": "You've hit your usage limit. Try again at 3:42 PM.", "codex_error_info": "usage_limit_exceeded"}}

    def __init__(self, s: Server, token: str, prompts: list[Any]) -> None:
        self.s, self.token = s, token
        self.script = list(prompts)  # per session/prompt: "ok" | "limit"
        self.prompts: list[dict] = []
        self.calls: list[tuple[str, dict]] = []
        self.conns: dict[str, dict] = {}  # connection id → {"frames": [...], "ws": ...}
        self.control = None
        self.task: asyncio.Task | None = None
        self.forget_after_limit = False
        self.upload_result: dict = {"event": "uploaded", "bytes": 10}

    async def start(self) -> None:
        self.control = await websockets.connect(self.s.ws, additional_headers={"Authorization": f"Bearer {self.token}"})
        await self.control.send(json.dumps({"jsonrpc": "2.0", "id": "register", "method": "_proxy/register", "params": {
            "runner_id": "rnr-1", "agents": ["claude", "codex"], "default_agent": "claude", "version": "0.2.0",
            "os": "linux", "tz": "UTC", "workspace": "/home/val/carrier-workspaces"}}))
        assert json.loads(await self.control.recv())["result"]["runner_id"] == "rnr-1"
        self.task = asyncio.create_task(self._control_loop())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
        for c in self.conns.values():
            if c.get("ws") is not None:
                await c["ws"].close()
        if self.control is not None:
            await self.control.close()

    async def _control_loop(self) -> None:
        async for raw in self.control:
            msg = json.loads(raw)
            if msg.get("method") == "_proxy/open":
                p = msg["params"]
                cid = p["connection_id"]
                if p.get("resume") and cid not in self.conns:
                    await self.control.send(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32602, "message": f"unknown connection {cid}"}}))
                    continue
                self.conns.setdefault(cid, {"frames": []})
                await self.control.send(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {"connection_id": cid}}))
                asyncio.create_task(self._data(cid, int(p.get("last_seq") or 0)))
            elif msg.get("method") == "_proxy/close":
                await self.control.send(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {}}))

    async def _data(self, cid: str, last_seq: int) -> None:
        conn = self.conns[cid]
        ws = await websockets.connect(f"{self.s.ws}/connections/{cid}", additional_headers={"Authorization": f"Bearer {self.token}"})
        conn["ws"] = ws
        for frame in conn["frames"][last_seq:]:
            await ws.send(frame)
        try:
            async for raw in ws:
                await self._on_frame(cid, json.loads(raw))
        except websockets.ConnectionClosed:
            pass

    async def emit(self, cid: str, obj: dict) -> None:
        conn = self.conns.get(cid)
        if conn is None:
            return
        frame = json.dumps(obj)
        conn["frames"].append(frame)
        try:
            await conn["ws"].send(frame)
        except Exception:
            pass

    async def _on_frame(self, cid: str, msg: dict) -> None:
        m, rid, p = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if m is None:
            return
        self.calls.append((m, p))
        reply = lambda result: self.emit(cid, {"jsonrpc": "2.0", "id": rid, "result": result})  # noqa: E731
        if m == "_proxy/launch":
            ws = (p.get("config") or {}).get("workspace")
            self.conns[cid]["workspace"] = ws
            await reply({"connection_id": cid, "agent": p.get("agent"),
                         **({"workspace": {"dir": ws["dir"], "restore": "restored"}} if ws else {})})
        elif m == "initialize":
            await reply({"protocolVersion": 1, "agentCapabilities": {"loadSession": True, "mcpCapabilities": {"http": True}}})
        elif m == "session/new":
            await reply({"sessionId": "s1"})
        elif m == "session/load":
            await self.emit(cid, _chunk("HISTORY "))
            await reply({})
        elif m == "session/prompt":
            self.prompts.append(p)
            step = self.script.pop(0) if self.script else "ok"
            if step == "limit":
                await self.emit(cid, _chunk("half done. "))
                await self.emit(cid, {"jsonrpc": "2.0", "id": rid, "error": self.LIMIT_ERROR})
                if self.forget_after_limit:
                    # The proxy restarted: its connections are gone.
                    conn = self.conns.pop(cid)
                    await conn["ws"].close()
                return
            await self.emit(cid, _chunk('{"summary": "done"}'))
            if self.conns[cid].get("workspace"):
                await self.emit(cid, {"jsonrpc": "2.0", "method": "_proxy/workspace", "params": self.upload_result})
            await reply({"stopReason": "end_turn"})
        elif m == "_proxy/shutdown":
            await reply({})


class Runs:
    def __init__(self) -> None:
        self.run = SimpleNamespace(id="run-1", state={}, touch=lambda: None)
        self.history: list[dict] = []

    async def get(self, run_id):
        return self.run

    async def update(self, run):
        self.history.append(dict(run.state))


class Tasks:
    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}

    async def get_task(self, key):
        return self.tasks.get(key)

    async def save_task(self, doc):
        self.tasks[doc["_id"]] = dict(doc)

    async def update_task(self, key, fields):
        self.tasks.setdefault(key, {}).update(fields)


SETTINGS = SimpleNamespace(acp_runner_wait_s=5, acp_quota_max_wait_s=86400, acp_quota_retry_s=60)


async def _step(
    runs: Runs | None = None, tasks: Tasks | None = None, acp_agent: str | None = "codex",
    agent_config: dict | None = None, settings: Any = SETTINGS,
) -> dict:
    return await run_acp_agent(
        step={"id": "code"},
        agent_def=AgentDefinition(id="laptop-codex", default_runtime="runner", runner_id="rnr-1", protocol="acp", acp_agent=acp_agent),
        input_data={"task": "fix it"},
        agent_config=agent_config or {"env_vars": {"SECRET": "x"}, "system_prompt": "be good"},
        runtime=None,
        run_id="run-1",
        task_key="run-1_code_0",
        callback_base_url="http://carrier",
        settings=settings,  # type: ignore[arg-type]
        run_repository=runs or Runs(),
        agent_task_repository=tasks or Tasks(),
    )


async def test_runner_step_runs_on_the_machine(server):
    _, token = await _runner(server)
    fake = FakeRunner(server, token, ["ok"])
    await fake.start()
    try:
        assert get_runner_hub().is_online("rnr-1")
        stored = await server.runners.get_runner("rnr-1")
        assert stored.labels["workspace"] == "/home/val/carrier-workspaces"
        out = await _step()
        assert out["result"] == '{"summary": "done"}'
        methods = [m for m, _ in fake.calls]
        assert methods[:4] == ["_proxy/launch", "initialize", "session/new", "session/prompt"]
        launch = dict(fake.calls)["_proxy/launch"]
        assert launch == {"agent": "codex"}
        new = dict(fake.calls)["session/new"]
        assert new["cwd"] == "/home/val/carrier-workspaces/run-1"
        assert "env_vars" not in new["_meta"]["carrier"]
    finally:
        await fake.stop()


S3_CONFIG = {"system_prompt": "be good", "extra": {"s3_bucket": "ws-bucket", "s3_path": "runs/run-1/", "workspace_exclude": ["*.jpg"]}}


async def test_runner_syncs_the_s3_workspace_through_signed_urls(server, monkeypatch):
    signed: list[tuple] = []

    def fake_sign(bucket, obj, ttl):
        signed.append((bucket, obj, ttl))
        return "https://gcs/get?sig", "https://gcs/put?sig"

    monkeypatch.setattr(acp_runner_workspace, "_sign_urls", fake_sign)
    _, token = await _runner(server)
    fake = FakeRunner(server, token, ["ok"])
    await fake.start()
    try:
        runs = Runs()
        out = await _step(runs, agent_config=S3_CONFIG)
        launch = dict(fake.calls)["_proxy/launch"]
        assert launch["config"]["workspace"] == {
            "dir": "/home/val/carrier-workspaces/run-1",
            "download_url": "https://gcs/get?sig",
            "upload_url": "https://gcs/put?sig",
            "upload_headers": {"content-type": "application/gzip"},
            "exclude": ["*.jpg"],
        }
        assert signed == [("ws-bucket", "runs/run-1/workspace.tar.gz", 86400)]
        assert out["workspace_s3_path"] == "gs://ws-bucket/runs/run-1/workspace.tar.gz"
        progress = runs.run.state["_agent_progress_code"]
        assert "workspace restored from gs://ws-bucket/runs/run-1/workspace.tar.gz" in progress
    finally:
        await fake.stop()


async def test_runner_step_runs_without_sync_when_carrier_cannot_sign(server, monkeypatch):
    def no_key(bucket, obj, ttl):
        raise RuntimeError("no signer")

    monkeypatch.setattr(acp_runner_workspace, "_sign_urls", no_key)
    _, token = await _runner(server)
    fake = FakeRunner(server, token, ["ok"])
    await fake.start()
    try:
        runs = Runs()
        out = await _step(runs, agent_config=S3_CONFIG)
        assert "config" not in dict(fake.calls)["_proxy/launch"]
        assert "workspace_s3_path" not in out
        assert any(line.startswith("workspace not synced: carrier cannot sign") for line in runs.run.state["_agent_progress_code"])
    finally:
        await fake.stop()


async def test_runner_gets_no_backend_secrets_and_the_public_datasources_url(server):
    _, token = await _runner(server)
    fake = FakeRunner(server, token, ["ok"])
    await fake.start()
    settings = SimpleNamespace(
        **vars(SETTINGS),
        resolved_runner_mcp_datasources_url=lambda: "https://carrier.example.com/mcp/datasources",
    )
    try:
        await _step(settings=settings, agent_config={
            "env_vars": {"SECRET": "x"},
            "credentials": {"ANTHROPIC_API_KEY": "sk-backend"},
            "tools": [{"name": "github", "command": "gh", "env": {"GH_TOKEN": "ghp_backend"}}],
            "mcp_servers": [
                {"name": "datasources", "transport": "streamable_http", "url": "http://carrier.svc.cluster.local:8000/mcp/datasources", "api_key": "grant"},
                {"name": "github", "transport": "streamable_http", "url": "https://api.githubcopilot.com/mcp/"},
            ],
        })
        new = dict(fake.calls)["session/new"]
        meta = new["_meta"]["carrier"]
        assert "env_vars" not in meta and "credentials" not in meta
        assert meta["tools"] == [{"name": "github", "command": "gh", "env": {}}]
        urls = {s["name"]: s["url"] for s in new["mcpServers"]}
        assert urls == {"datasources": "https://carrier.example.com/mcp/datasources", "github": "https://api.githubcopilot.com/mcp/"}
        assert new["mcpServers"][0]["headers"] == [{"name": "Authorization", "value": "Bearer grant"}]
    finally:
        await fake.stop()


async def test_offline_runner_fails_the_step_after_waiting(server):
    await _runner(server)
    settings = SETTINGS
    try:
        SETTINGS.acp_runner_wait_s = 0.3
        runs = Runs()
        with pytest.raises(RuntimeError, match="offline"):
            await _step(runs)
        assert any("_runner_wait_code" in h for h in runs.history)
        assert "_runner_wait_code" not in runs.run.state
    finally:
        settings.acp_runner_wait_s = 5


async def test_revoking_closes_the_control_socket(server):
    _, token = await _runner(server)
    fake = FakeRunner(server, token, [])
    await fake.start()
    async with server.client("alice") as alice:
        assert (await alice.delete("/rnr-1")).status_code == 204
    await asyncio.sleep(0.1)
    assert fake.control.close_code == 4403
    with pytest.raises(websockets.InvalidStatus) as exc:
        await websockets.connect(server.ws, additional_headers={"Authorization": f"Bearer {token}"})
    assert exc.value.response.status_code == 403


async def test_usage_limit_parks_and_continues_in_the_same_session(server, monkeypatch):
    _, token = await _runner(server)
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(acp_executor, "_sleep", fake_sleep)
    fake = FakeRunner(server, token, ["limit", "ok"])
    await fake.start()
    try:
        runs = Runs()
        out = await _step(runs)
        assert out["result"] == '{"summary": "done"}'
        assert len(waits) == 1 and waits[0] > 0
        assert [p["prompt"][0]["text"] for p in fake.prompts][1] == CONTINUE_PROMPT
        assert [m for m, _ in fake.calls].count("session/new") == 1
        parked = [h["_quota_wait_code"] for h in runs.history if "_quota_wait_code" in h]
        assert parked and parked[0]["kind"] == "usage_limit" and parked[0]["resume_at"]
        assert "_quota_wait_code" not in runs.run.state
        assert (await server.runners.get_runner("rnr-1")).limits == {}
    finally:
        await fake.stop()


async def test_usage_limit_reloads_the_session_when_the_connection_is_gone(server, monkeypatch):
    _, token = await _runner(server)

    async def slow_sleep(seconds: float) -> None:
        await asyncio.sleep(2.5)  # long enough for the old connection to be declared lost

    monkeypatch.setattr(acp_executor, "_sleep", slow_sleep)
    fake = FakeRunner(server, token, ["limit", "ok"])
    fake.forget_after_limit = True
    await fake.start()
    try:
        out = await _step()
        assert out["result"] == '{"summary": "done"}'
        methods = [m for m, _ in fake.calls]
        assert "session/load" in methods
        assert methods.index("session/load") < len(methods) - 1
        load = dict(fake.calls)["session/load"]
        assert load["sessionId"] == "s1"
        assert fake.prompts[-1]["prompt"][0]["text"] == CONTINUE_PROMPT
        assert "HISTORY" not in out["result"]
    finally:
        await fake.stop()


async def test_parked_step_resumes_after_a_carrier_restart(server, monkeypatch):
    _, token = await _runner(server)

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(acp_executor, "_sleep", fake_sleep)
    fake = FakeRunner(server, token, ["ok"])
    await fake.start()
    try:
        # The connection the step had before the restart, with one frame read.
        fake.conns["c-old"] = {"frames": [json.dumps(_chunk("before "))]}
        tasks = Tasks()
        tasks.tasks["run-1_code_0"] = {"_id": "run-1_code_0", "status": "waiting_for_quota", "acp": {
            "agent_url": None, "token": None, "runner_id": "rnr-1", "connection_id": "c-old", "session_id": "s1",
            "prompt_id": 5, "next_id": 6, "received": 1,
            "quota": {"kind": "usage_limit", "message": "limit", "resets_at": None, "resume_at": None},
            "quota_started": datetime.now(timezone.utc).timestamp(),
        }}
        out = await _step(tasks=tasks)
        assert out["result"] == '{"summary": "done"}'
        assert [m for m, _ in fake.calls] == ["session/prompt", "_proxy/shutdown"]
        assert fake.prompts[0]["prompt"][0]["text"] == CONTINUE_PROMPT
        assert tasks.tasks["run-1_code_0"]["status"] == "finished"
    finally:
        await fake.stop()


async def test_quota_exhausted_fails_and_marks_the_agent(server):
    _, token = await _runner(server)
    fake = FakeRunner(server, token, ["limit"])
    fake.LIMIT_ERROR = {"code": -32603, "message": "Internal error", "data": {
        "message": "You hit your spend cap set in your workspace. Increase your spend cap to continue.",
        "codex_error_info": "usage_limit_exceeded"}}
    await fake.start()
    try:
        with pytest.raises(RuntimeError, match="no quota left"):
            await _step()
        limits = (await server.runners.get_runner("rnr-1")).limits
        assert limits["codex"].kind == "quota_exhausted"
    finally:
        await fake.stop()
