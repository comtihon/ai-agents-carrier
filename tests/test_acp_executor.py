"""ACP agent steps: client, executor, and execute_agent_step routing."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
import websockets

from app.domain.models.agent_definition import AgentDefinition
from app.infrastructure.acp.client import AcpConnection, AcpConnectionLost, to_ws_url
from app.steps.acp_executor import (
    build_prompt_text,
    has_expected_fields,
    pick_permission_option,
    run_acp_agent,
    to_acp_mcp_servers,
)


# ─── fake acp-web-proxy + agent ──────────────────────────────────────────────


class FakeProxy:
    """Speaks the proxy's wire protocol: numbered frames, replay on resume."""

    def __init__(self, *, drop_after_frames: int | None = None, answer_text: str = '{"summary": "done"}') -> None:
        self.frames: list[str] = []  # everything sent to the client, in order
        self.received: list[dict] = []
        self.headers_seen: list[dict] = []
        self.drop_after = drop_after_frames
        self.answer_text = answer_text
        self.client_answers: dict[Any, Any] = {}
        self._answer_events: dict[Any, asyncio.Event] = {}
        self.ws = None
        self.server = None
        self.port = 0
        self.prompts = 0
        self.token = None

    async def start(self) -> None:
        self.server = await websockets.serve(self._handler, "127.0.0.1", 0, process_request=self._process, process_response=self._respond)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        self.server.close()
        await self.server.wait_closed()

    def _process(self, connection, request):
        self.headers_seen.append(dict(request.headers))
        if request.headers.get("Authorization") != f"Bearer {self.token}":
            return connection.respond(401, "nope")
        return None

    def _respond(self, connection, request, response):
        response.headers["Acp-Connection-Id"] = "conn-1"
        return response

    async def _handler(self, ws):
        self.ws = ws
        last = int(ws.request.headers.get("Acp-Last-Seq") or 0)
        for frame in self.frames[last:]:
            await ws.send(frame)
        try:
            async for raw in ws:
                msg = json.loads(raw)
                self.received.append(msg)
                asyncio.ensure_future(self._on_client(msg))
        except websockets.ConnectionClosed:
            pass

    async def emit(self, obj: dict) -> None:
        frame = json.dumps(obj)
        self.frames.append(frame)
        if self.ws is not None:
            try:
                await self.ws.send(frame)
            except websockets.ConnectionClosed:
                pass
        if self.drop_after is not None and len(self.frames) == self.drop_after:
            self.drop_after = None
            await self.ws.close()

    async def ask_client(self, rid: Any, method: str, params: dict) -> Any:
        self._answer_events[rid] = asyncio.Event()
        await self.emit({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        await self._answer_events[rid].wait()
        return self.client_answers[rid]

    async def _on_client(self, msg: dict) -> None:
        if "method" not in msg:
            self.client_answers[msg["id"]] = msg.get("result", msg.get("error"))
            if msg["id"] in self._answer_events:
                self._answer_events[msg["id"]].set()
            return
        m, rid, p = msg["method"], msg.get("id"), msg.get("params") or {}
        reply = lambda result: self.emit({"jsonrpc": "2.0", "id": rid, "result": result})  # noqa: E731
        if m == "_proxy/launch":
            await reply({"connection_id": "conn-1", "agent": p.get("agent", "pi")})
        elif m == "initialize":
            await reply({"protocolVersion": 1, "agentCapabilities": {"mcpCapabilities": {"http": True, "sse": False}}})
        elif m == "session/new":
            self.session_new = p
            await reply({"sessionId": "sess-1"})
        elif m == "session/prompt":
            self.prompts += 1
            if self.prompts > 1:  # the output-protocol nudge
                await self.emit(_chunk('{"summary": "fixed"}'))
                await reply({"stopReason": "end_turn"})
                return
            await self.emit(_chunk("Looking around. "))
            await self.emit({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "sess-1", "update": {
                "sessionUpdate": "tool_call", "toolCallId": "t1", "title": "bash ls", "kind": "execute", "status": "pending",
                "rawInput": {"command": "ls"}}}})
            perm = await self.ask_client(900, "session/request_permission", {"sessionId": "sess-1", "toolCall": {"toolCallId": "t1", "rawInput": {"command": "rm -rf /"}},
                                                                                 "options": [{"optionId": "y", "kind": "allow_once"}, {"optionId": "n", "kind": "reject_once"}]})
            self.permission_answer = perm
            await self.emit({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "sess-1", "update": {
                "sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed"}}})
            await self.emit(_chunk(self.answer_text))
            await reply({"stopReason": "end_turn", "_meta": {"carrier": {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}, "workspace_path": "gs://b/p"}}})
        elif m == "_proxy/shutdown":
            await reply({})


def _chunk(text: str) -> dict:
    return {"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "sess-1", "update": {
        "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}}}


class FakeRuntime:
    def __init__(self, proxy: FakeProxy) -> None:
        self.proxy = proxy
        self.spawned: list[dict] = []
        self.terminated = 0

    async def spawn(self, agent_def, step, run_id, callback_base_url, extra_env=None):
        self.spawned.append(extra_env or {})
        self.proxy.token = (extra_env or {})["ACP_PROXY_TOKEN"]
        return f"http://127.0.0.1:{self.proxy.port}"

    async def terminate_by_run_id(self, agent_def, run_id):
        self.terminated += 1


class Repo:
    def __init__(self) -> None:
        self.runs: dict[str, Any] = {}

    async def get(self, run_id):
        return self.runs.get(run_id)

    async def update(self, run):
        self.runs[run.id] = run


class TaskRepo:
    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}

    async def get_task(self, key):
        return self.tasks.get(key)

    async def save_task(self, doc):
        self.tasks[doc["_id"]] = dict(doc)

    async def update_task(self, key, fields):
        self.tasks.setdefault(key, {}).update(fields)


def _agent() -> AgentDefinition:
    return AgentDefinition(id="pi", default_runtime="k8s", protocol="acp", acp_agent="pi")


AGENT_CONFIG = {
    "system_prompt": "You are a coder.",
    "blocked_commands": ["rm"],
    "mcp_servers": [
        {"name": "datasources", "transport": "streamable_http", "url": "http://carrier/mcp/datasources", "api_key": "grant-jwt", "env": {}},
        {"name": "jira", "transport": "stdio", "command": ["jira-mcp", "--x"], "env": {"A": "1"}},
        {"name": "old", "transport": "sse", "url": "http://x/sse", "env": {}},
    ],
    "env_vars": {"FOO": "bar"},
}


async def _run(proxy: FakeProxy, step: dict | None = None, task_repo: TaskRepo | None = None):
    runtime = FakeRuntime(proxy)
    repo = Repo()
    repo.runs["run-1"] = SimpleNamespace(id="run-1", state={}, touch=lambda: None)
    out = await run_acp_agent(
        step=step or {"id": "code"},
        agent_def=_agent(),
        input_data={"task": "list files", "context": {"repo": "x"}},
        agent_config=AGENT_CONFIG,
        runtime=runtime,
        run_id="run-1",
        task_key="run-1_code_0",
        callback_base_url="http://carrier",
        settings=SimpleNamespace(),
        run_repository=repo,
        agent_task_repository=task_repo,
    )
    return out, runtime, repo


# ─── pure helpers ────────────────────────────────────────────────────────────


def test_mcp_servers_translate_to_acp_shapes():
    out = to_acp_mcp_servers(AGENT_CONFIG["mcp_servers"])
    assert out[0] == {"type": "http", "name": "datasources", "url": "http://carrier/mcp/datasources",
                      "headers": [{"name": "Authorization", "value": "Bearer grant-jwt"}]}
    assert out[1] == {"name": "jira", "command": "jira-mcp", "args": ["--x"], "env": [{"name": "A", "value": "1"}]}
    assert out[2]["type"] == "sse"


def test_prompt_text_puts_the_ask_first_and_sections_after():
    text = build_prompt_text({"request": "Fix bug", "plan": {"steps": [1]}, "_visit_counts": {}, "clarification_context": {"q": "a"}})
    assert text.startswith("Fix bug")
    assert "## plan" in text and "## Clarification context" in text and "_visit_counts" not in text


def test_permission_allows_unless_a_blocked_command_runs():
    opts = [{"optionId": "y", "kind": "allow_once"}, {"optionId": "n", "kind": "reject_once"}]
    assert pick_permission_option({"options": opts, "toolCall": {"rawInput": {"command": "ls -la"}}}, ["rm"])["outcome"]["optionId"] == "y"
    assert pick_permission_option({"options": opts, "toolCall": {"rawInput": {"command": "rm -rf /x"}}}, ["rm"])["outcome"]["optionId"] == "n"
    assert pick_permission_option({"options": opts, "toolCall": {"rawInput": {"command": "git rm-cache"}}}, ["rm"])["outcome"]["optionId"] == "y"
    assert pick_permission_option({"options": [], "toolCall": {}}, [])["outcome"] == {"outcome": "cancelled"}


def test_expected_fields_detection():
    assert has_expected_fields('```json\n{"summary": 1}\n```', ["summary"])
    assert has_expected_fields("summary: ok\nother: 1", ["summary"])
    assert not has_expected_fields("I did it.", ["summary"])
    assert has_expected_fields("anything", [])


def test_ws_url():
    assert to_ws_url("http://10.0.0.1:8000") == "ws://10.0.0.1:8000/acp"
    assert to_ws_url("https://h/") == "wss://h/acp"


# ─── end to end ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_turn_over_acp():
    proxy = FakeProxy()
    await proxy.start()
    try:
        out, runtime, repo = await _run(proxy, task_repo=TaskRepo())
    finally:
        await proxy.stop()
    assert out == {
        "result": '{"summary": "done"}',
        "token_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "workspace_s3_path": "gs://b/p",
    }
    # A fresh per-run token reached the pod env and the handshake.
    env = runtime.spawned[0]
    assert env["FOO"] == "bar" and len(env["ACP_PROXY_TOKEN"]) > 20
    # session/new carried the converted MCP servers (SSE dropped: agent lacks it) and the config.
    sn = proxy.session_new
    assert [s["name"] for s in sn["mcpServers"]] == ["datasources", "jira"]
    assert sn["_meta"]["carrier"]["system_prompt"] == "You are a coder."
    assert sn["_meta"]["carrier"]["run_id"] == "run-1"
    # The blocked command was refused.
    assert proxy.permission_answer == {"outcome": {"outcome": "selected", "optionId": "n"}}
    # Progress reached the run, then the pod was shut down and terminated.
    progress = repo.runs["run-1"].state["_agent_progress_code"]
    assert any("bash ls" in p for p in progress) and any("Looking around" in p for p in progress)
    assert any(m.get("method") == "_proxy/shutdown" for m in proxy.received)
    assert runtime.terminated == 1


@pytest.mark.asyncio
async def test_output_protocol_nudge_asks_again_in_the_same_session():
    proxy = FakeProxy(answer_text="All done, files listed.")
    await proxy.start()
    try:
        out, _, _ = await _run(proxy, step={"id": "code", "output_mapping": {"summary": "summary"}})
    finally:
        await proxy.stop()
    assert proxy.prompts == 2
    assert out["result"] == '{"summary": "fixed"}'


@pytest.mark.asyncio
async def test_survives_a_socket_drop_mid_turn_via_replay():
    proxy = FakeProxy(drop_after_frames=6)
    await proxy.start()
    try:
        out, _, _ = await _run(proxy)
    finally:
        await proxy.stop()
    assert out["result"] == '{"summary": "done"}'
    resumed = [h for h in proxy.headers_seen if h.get("acp-connection-id")]
    assert resumed and resumed[0]["acp-connection-id"] == "conn-1"
    assert int(resumed[0]["acp-last-seq"]) >= 5  # resumed after what it had seen, not from scratch
    # The permission answer survived the drop.
    assert proxy.permission_answer == {"outcome": {"outcome": "selected", "optionId": "n"}}


@pytest.mark.asyncio
async def test_client_gives_up_when_proxy_refuses():
    proxy = FakeProxy()
    await proxy.start()
    proxy.token = "right"
    try:
        conn = AcpConnection(f"ws://127.0.0.1:{proxy.port}/acp", "wrong")
        with pytest.raises(AcpConnectionLost):
            await conn.connect()
    finally:
        await proxy.stop()


@pytest.mark.asyncio
async def test_execute_agent_step_routes_acp_agents(monkeypatch):
    from app.steps import agent_executor

    captured = {}

    async def fake_run_acp_agent(**kw):
        captured.update(kw)
        return {"result": '{"summary": "ok"}'}

    monkeypatch.setattr("app.steps.acp_executor.run_acp_agent", fake_run_acp_agent)

    class Backend:
        async def get(self, _id):
            return _agent()

    settings = agent_executor.get_settings()
    out = await agent_executor.execute_agent_step(
        {"id": "code", "agent_id": "pi", "output_mapping": {"summary": "summary"}},
        {"request": "go"}, Backend(), "run-9", "http://carrier", settings=settings, use_meta_llm=False,
    )
    assert out["summary"] == "ok"
    assert captured["run_id"] == "run-9" and captured["agent_config"]["mcp_servers"] is not None
