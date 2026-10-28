"""Drive an agent step over ACP (Agent Client Protocol) through acp-web-proxy.

Used for agent definitions with ``protocol: "acp"``. The pod runs
acp-web-proxy in front of an ACP agent (pi-carrier-agent, claude-agent-acp,
codex-acp, …); carrier is the ACP client over a WebSocket:

    spawn (existing runtime, ACP_PROXY_TOKEN in env)
    → _proxy/launch → initialize → session/new {cwd, mcpServers, _meta}
    → session/prompt, streaming session/update into run state
    → stopReason → raw output {result, token_usage, …}

The raw output then goes through the same post-processing as every agent
(``_finalize_agent_output``): structured extraction, meta-LLM gate, mapping.

Replaces the HTTP poll protocol's moving parts: no /poll loop (updates are
pushed), no idle meta-LLM recovery (the proxy buffers and replays frames over a
reconnect; a carrier restart re-attaches by connection id), no /start resend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from app.application.runners import RunnerOffline
from app.core.config import DATASOURCES_MCP_NAME
from app.domain.models.runner import AgentLimit
from app.infrastructure.acp.client import AcpConnection, AcpError, AcpMethodError, to_ws_url
from app.services import agent_inbox
from app.steps.acp_limits import LimitInfo, classify_limit
from app.steps.acp_runner_workspace import WorkspaceUnavailable, runner_workspace

if TYPE_CHECKING:  # pragma: no cover
    from app.core.config import Settings
    from app.domain.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
CLIENT_INFO = {"name": "ai-agents-carrier", "version": "1"}
DEFAULT_CWD = "/workspace"
PROGRESS_FLUSH_S = 2.0
ANSWER_TIMEOUT_S = 30 * 60
# Patched by tests: how a parked step waits.
_sleep = asyncio.sleep

CONTINUE_PROMPT = (
    "You were interrupted by a usage limit before you finished. Continue the task from where "
    "you stopped; do not redo work that is already done."
)


# ─── translation ─────────────────────────────────────────────────────────────


def to_acp_mcp_servers(mcp_servers: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """carrier agent_config.mcp_servers → ACP session/new mcpServers."""
    out: list[dict[str, Any]] = []
    for s in mcp_servers or []:
        name = s.get("name")
        if not name:
            continue
        env = [{"name": k, "value": str(v)} for k, v in (s.get("env") or {}).items()]
        if s.get("transport") == "stdio":
            cmd = list(s.get("command") or [])
            if not cmd:
                continue
            out.append({"name": name, "command": cmd[0], "args": cmd[1:], "env": env})
        else:
            headers = []
            if s.get("api_key"):
                headers.append({"name": "Authorization", "value": f"Bearer {s['api_key']}"})
            kind = "sse" if s.get("transport") == "sse" else "http"
            out.append({"type": kind, "name": name, "url": s.get("url", ""), "headers": headers})
    return out


def build_prompt_text(input_data: dict[str, Any]) -> str:
    """The task as one user message: the main ask first, then named sections.

    The system prompt is NOT part of it — it travels in session/new ``_meta``
    and the agent applies it as a real system prompt.
    """
    main_keys = ("request", "task", "prompt")
    main = next((input_data[k] for k in main_keys if isinstance(input_data.get(k), str) and input_data[k].strip()), None)
    parts: list[str] = [main] if main else []
    for key, value in input_data.items():
        if key in main_keys and value == main:
            continue
        if key.startswith("_") or key == "clarification_context" or value in (None, "", [], {}):
            continue
        body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)
        parts.append(f"## {key}\n{body}")
    if input_data.get("clarification_context"):
        ctx = input_data["clarification_context"]
        body = ctx if isinstance(ctx, str) else json.dumps(ctx, ensure_ascii=False, indent=2, default=str)
        parts.append(f"## Clarification context\n{body}")
    return "\n\n".join(parts) if parts else json.dumps(input_data, ensure_ascii=False, default=str)


def has_expected_fields(text: str, fields: list[str]) -> bool:
    """Whether *text* carries at least one of the output-protocol fields as JSON/YAML."""
    if not fields:
        return True
    candidates = [text]
    fence = re.search(r"```(?:yaml|json)?\s*\n?(.*)\n?\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1))
    for c in candidates:
        for loader in (json.loads, _yaml_load):
            try:
                parsed = loader(c.strip())
            except Exception:
                continue
            if isinstance(parsed, dict) and any(f in parsed for f in fields):
                return True
    return False


def _yaml_load(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def pick_permission_option(params: dict[str, Any], blocked_commands: list[str]) -> dict[str, Any]:
    """Answer session/request_permission for a headless run.

    Allow (once) unless the tool call runs a blocked command — then reject.
    """
    options = params.get("options") or []
    tool_call = params.get("toolCall") or {}
    raw = json.dumps(tool_call.get("rawInput") or {}, default=str) + " " + str(tool_call.get("title") or "")
    blocked = any(re.search(rf"(^|[\s\"'/;&|]){re.escape(cmd)}($|[\s\"';&|])", raw) for cmd in blocked_commands if cmd)

    def first(kinds: tuple[str, ...]) -> dict[str, Any] | None:
        for kind in kinds:
            for opt in options:
                if opt.get("kind") == kind:
                    return opt
        return None

    choice = first(("reject_once", "reject_always")) if blocked else first(("allow_once", "allow_always"))
    if choice is None and options and not blocked:
        choice = options[0]
    if choice is None:
        return {"outcome": {"outcome": "cancelled"}}
    return {"outcome": {"outcome": "selected", "optionId": choice.get("optionId")}}


# ─── run state sink ──────────────────────────────────────────────────────────


class _ProgressSink:
    """Batches agent progress into run.state, the way the poll loop did."""

    def __init__(self, run_repository: Any, run_id: str, step_id: str) -> None:
        self.repo = run_repository
        self.run_id = run_id
        self.step_id = step_id
        self.pending: list[str] = []
        self.active_tools: dict[str, str] = {}
        self.context_usage: dict[str, Any] | None = None
        # The last `_claude/rateLimit` meta (status, resetsAt, …) of the turn.
        self.rate_limit: dict[str, Any] | None = None
        self._last_flush = 0.0
        self._lock = asyncio.Lock()

    def add(self, line: str) -> None:
        line = line.strip()
        if line:
            self.pending.append(line[:4000])

    async def maybe_flush(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._last_flush < PROGRESS_FLUSH_S:
            return
        await self.flush()

    async def flush(self) -> None:
        if self.repo is None:
            self.pending.clear()
            return
        async with self._lock:
            self._last_flush = time.monotonic()
            lines, self.pending = self.pending, []
            try:
                run = await self.repo.get(self.run_id)
                if run is None:
                    return
                state = dict(run.state or {})
                key = f"_agent_progress_{self.step_id}"
                if lines:
                    state[key] = list(state.get(key) or []) + lines
                state["_active_tools"] = sorted(set(self.active_tools.values()))
                if self.context_usage is not None:
                    state[f"_live_context_usage_{self.step_id}"] = self.context_usage
                run.state = state
                run.touch()
                await self.repo.update(run)
            except Exception:
                logger.debug("progress flush failed for run %s", self.run_id, exc_info=True)

    async def set_state(self, key: str, value: Any) -> None:
        """Set (None: remove) one key of run.state right away."""
        if self.repo is None:
            return
        async with self._lock:
            try:
                run = await self.repo.get(self.run_id)
                if run is None:
                    return
                state = dict(run.state or {})
                if value is None:
                    state.pop(key, None)
                else:
                    state[key] = value
                run.state = state
                run.touch()
                await self.repo.update(run)
            except Exception:
                logger.debug("state update failed for run %s", self.run_id, exc_info=True)


# ─── executor ────────────────────────────────────────────────────────────────


async def run_acp_agent(
    *,
    step: dict[str, Any],
    agent_def: "AgentDefinition",
    input_data: dict[str, Any],
    agent_config: dict[str, Any],
    runtime: Any,
    run_id: str,
    task_key: str,
    callback_base_url: str,
    settings: "Settings",
    run_repository: Any = None,
    agent_task_repository: Any = None,
    runner_hub: Any = None,
) -> dict[str, Any]:
    """Run one agent step over ACP; returns the raw output for _finalize_agent_output.

    The agent runs in a pod (``runtime.spawn``) or, with ``default_runtime:
    "runner"``, on a machine connected with acp-web-proxy, which dials carrier
    (app.application.runners). When the agent runs out of quota the step is
    parked until the limit resets and then continues in the same session
    (``session/load`` on a new connection when the old one is gone).
    """
    step_id = step["id"]
    sink = _ProgressSink(run_repository, run_id, step_id)
    chunks: list[str] = []
    turn_text: list[str] = []
    blocked = list(agent_config.get("blocked_commands") or [])
    # While session/load replays the conversation, its updates are history.
    loading = False
    # gs:// URI of a runner's workspace archive, once the proxy uploaded it.
    workspace_uri: str | None = None
    workspace_uploaded = False

    async def on_notification(method: str, params: dict[str, Any]) -> None:
        nonlocal workspace_uploaded
        if method == "session/update":
            if loading:
                return
            _apply_update(params.get("update") or {}, sink, chunks, turn_text)
            await sink.maybe_flush()
        elif method == "_proxy/agent_exited":
            sink.add(f"agent exited (code {params.get('code')})")
        elif method == "_proxy/workspace":
            if params.get("event") == "uploaded":
                workspace_uploaded = True
            else:
                sink.add(f"workspace upload failed: {params.get('error')}")

    async def on_request(method: str, params: dict[str, Any]) -> Any:
        if method == "session/request_permission":
            return pick_permission_option(params, blocked)
        if method == "_carrier/ask":
            answer = await _ask_human(run_repository, run_id, str(params.get("question") or ""), params.get("options"))
            return {"cancelled": True} if answer is None else {"answer": answer}
        raise AcpError(f"carrier does not implement {method}")

    agent_input = agent_def.agent_input or {}
    on_runner = agent_def.default_runtime == "runner"
    hub = None
    runner = None
    if on_runner:
        from app.application.runners import get_runner_hub
        hub = runner_hub or get_runner_hub()
        runner = await hub.backend.get_runner(agent_def.runner_id or "") if hub.backend is not None else None
        if runner is None:
            raise RuntimeError(
                f"[step '{step_id}'] runner '{agent_def.runner_id}' of agent '{agent_def.id}' no longer exists "
                "(it was disconnected in carrier); connect the machine again or pick another agent"
            )
    runner_agent = agent_def.acp_agent or (runner.default_agent if runner else None)
    runner_wait_s = float(agent_input.get("runner_wait_s") or getattr(settings, "acp_runner_wait_s", 600))
    quota_max_wait_s = float(agent_input.get("quota_max_wait_s") or getattr(settings, "acp_quota_max_wait_s", 86400))
    quota_retry_s = float(getattr(settings, "acp_quota_retry_s", 1800))
    cwd: str | None = None
    agent_tz = runner.tz if runner else None

    # Re-attach to a turn still running (or parked) from before a restart.
    stored = await agent_task_repository.get_task(task_key) if agent_task_repository is not None else None
    acp_state = (stored or {}).get("acp") if isinstance(stored, dict) else None

    agent_url: str | None = None
    token: str | None = None
    conn: AcpConnection | None = None
    session_id: str | None = None
    prompt_id: int | None = None
    prompt_future: asyncio.Future | None = None
    caps: dict[str, Any] = {}
    quota_started: float | None = None
    transient_attempts = 0
    reattach_seq = 0

    def new_conn(connection_id: str | None = None, first_id: int = 1) -> AcpConnection:
        if runner is not None:
            return AcpConnection(
                None, None,
                opener=hub.opener(runner.id, wait_s=runner_wait_s),
                connection_id=connection_id, first_id=first_id,
                on_notification=on_notification, on_request=on_request,
                send_timeout=runner_wait_s + 60,
            )
        return AcpConnection(
            to_ws_url(agent_url or ""), token or "",
            connection_id=connection_id, first_id=first_id,
            on_notification=on_notification, on_request=on_request,
        )

    async def open_connection() -> AcpConnection:
        """A new proxy connection with the agent launched and initialized."""
        nonlocal caps
        wait_key = f"_runner_wait_{step_id}"
        waiting = runner is not None and not hub.is_online(runner.id)
        if waiting:
            sink.add(f"waiting for machine '{runner.name}' to come online")
            await sink.set_state(wait_key, {"runner_id": runner.id, "runner_name": runner.name, "since": _iso(_utcnow())})
        c = new_conn()
        try:
            await c.connect()
        except Exception as exc:
            if runner is not None and isinstance(exc, RunnerOffline):
                raise RuntimeError(f"[step '{step_id}'] machine '{runner.name}' is offline: {exc}") from exc
            raise
        finally:
            if waiting:
                await sink.set_state(wait_key, None)
        launch_params: dict[str, Any] = {}
        if runner_agent:
            launch_params["agent"] = runner_agent
        if runner is not None:
            workspace = await launch_workspace()
            if workspace is not None:
                launch_params["config"] = {"workspace": workspace}
        launched = await c.request("_proxy/launch", launch_params, timeout=600)
        c.connection_id = c.connection_id or (launched or {}).get("connection_id")
        restore = ((launched or {}).get("workspace") or {}).get("restore")
        if restore:
            sink.add({"restored": f"workspace restored from {workspace_uri}",
                      "empty": "workspace starts empty (nothing uploaded yet)",
                      "kept": "workspace kept from the previous launch"}.get(restore, f"workspace: {restore}"))
        init = await c.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
            "clientInfo": CLIENT_INFO,
        }, timeout=300)
        auth_method = agent_input.get("acp_auth_method")
        if auth_method:
            await c.request("authenticate", {"methodId": auth_method}, timeout=120)
        caps = (init or {}).get("agentCapabilities") or {}
        return c

    async def session_cwd() -> str:
        nonlocal runner, cwd
        if cwd is None:
            if runner is not None:
                # The machine reports its workspace when it connects.
                runner = await hub.backend.get_runner(runner.id) or runner
            cwd = _session_cwd(agent_def, runner, run_id)
        return cwd

    async def launch_workspace() -> dict[str, Any] | None:
        """Signed URLs the runner syncs the session directory with (``s3`` addon)."""
        nonlocal workspace_uri
        cwd = await session_cwd()
        root = ((runner.labels or {}).get("workspace") or "").rstrip("/\\")
        try:
            found = await runner_workspace(agent_config, cwd, settings)
        except WorkspaceUnavailable as exc:
            sink.add(f"workspace not synced: {exc}")
            return None
        if found is None:
            return None
        if not root or not cwd.startswith(root + ("\\" if runner.os == "win32" else "/")):
            sink.add(f"workspace not synced: {cwd} is outside the machine's workspace {root or '(none reported)'}")
            return None
        workspace, workspace_uri = found
        return workspace

    def session_mcp_servers() -> list[dict[str, Any]]:
        servers = agent_config.get("mcp_servers") or []
        if runner is not None and hasattr(settings, "resolved_runner_mcp_datasources_url"):
            # The bridge's agent address is cluster-internal; a machine dials the public one.
            runner_url = settings.resolved_runner_mcp_datasources_url()
            servers = [{**s, "url": runner_url} if s.get("name") == DATASOURCES_MCP_NAME else s for s in servers]
        mcp_servers = to_acp_mcp_servers(servers)
        mcp_caps = caps.get("mcpCapabilities") or {}
        dropped = [
            s["name"] for s in mcp_servers
            if (s.get("type") == "http" and not mcp_caps.get("http"))
            or (s.get("type") == "sse" and not mcp_caps.get("sse"))
        ]
        if dropped:
            sink.add(f"agent cannot reach MCP servers over HTTP/SSE, skipped: {', '.join(dropped)}")
            mcp_servers = [s for s in mcp_servers if s["name"] not in dropped]
        return mcp_servers

    def acp_record(**extra: Any) -> dict[str, Any]:
        assert conn is not None
        return {
            "agent_url": agent_url, "token": token, "runner_id": runner.id if runner else None,
            "connection_id": conn.connection_id, "session_id": session_id, "prompt_id": prompt_id,
            "next_id": conn.peek_next_id(), "received": conn.received, **extra,
        }

    async def save_acp(status: str, **extra: Any) -> None:
        if agent_task_repository is not None:
            await agent_task_repository.update_task(task_key, {"status": status, "acp": acp_record(**extra)})

    async def send_prompt(text: str) -> None:
        nonlocal prompt_id, prompt_future
        assert conn is not None
        prompt_id = conn.next_id()
        await save_acp("working")
        prompt_future = asyncio.ensure_future(conn.request("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": text}],
        }, req_id=prompt_id))

    async def park(limit: LimitInfo) -> None:
        """Wait out a usage limit (state, progress and the task say until when)."""
        nonlocal quota_started, transient_attempts
        now = _utcnow()
        quota_started = quota_started or now.timestamp()
        if limit.kind == "usage_limit":
            if limit.resets_at is not None:
                resume_at = max(limit.resets_at + timedelta(seconds=60), now + timedelta(seconds=30))
            else:
                resume_at = now + timedelta(seconds=quota_retry_s)
            if runner is not None and runner_agent:
                await hub.set_limit(runner.id, runner_agent, AgentLimit(until=limit.resets_at, kind=limit.kind, message=limit.message, updated_at=now))
        else:
            transient_attempts += 1
            resume_at = now + timedelta(seconds=min(60 * 2 ** (transient_attempts - 1), 900))
        if resume_at.timestamp() - quota_started > quota_max_wait_s:
            raise RuntimeError(
                f"[step '{step_id}'] agent hit its usage limit ({limit.message}); it would resume at "
                f"{_iso(resume_at)}, past the {int(quota_max_wait_s)}s carrier waits (quota_max_wait_s)"
            )
        record = {"kind": limit.kind, "message": limit.message,
                  "resets_at": _iso(limit.resets_at) if limit.resets_at else None, "resume_at": _iso(resume_at)}
        if conn is not None:
            await save_acp("waiting_for_quota", quota=record, quota_started=quota_started)
        wait_key = f"_quota_wait_{step_id}"
        await sink.set_state(wait_key, {
            "resume_at": _iso(resume_at) if limit.kind != "usage_limit" or limit.resets_at else None,
            "kind": limit.kind, "message": limit.message, "agent_id": agent_def.id, "since": _iso(now),
        })
        sink.add(f"⏸ {_limit_words(limit.kind)} — resuming at {resume_at:%Y-%m-%d %H:%M} UTC" + (f": {limit.message}" if limit.message else ""))
        await sink.flush()
        logger.info("[step '%s'] %s; parked until %s", step_id, limit.kind, _iso(resume_at))
        await _sleep(max(0.0, (resume_at - _utcnow()).total_seconds()))
        await sink.set_state(wait_key, None)
        sink.add("▶ resuming")

    async def resume_session() -> None:
        """Make ``conn`` a live connection to ``session_id`` again."""
        nonlocal conn, loading
        if conn is not None and not conn.started:
            # Parked before a restart: re-attach where we stopped reading.
            try:
                await conn.connect(last_seq=reattach_seq)
                return
            except Exception as exc:
                logger.info("[step '%s'] stored ACP connection is gone (%s); reloading the session", step_id, exc)
                await conn.close()
                conn = None
        if conn is not None and not conn.lost:
            return
        if conn is not None:
            await conn.close()
        conn = await open_connection()
        if not caps.get("loadSession"):
            raise RuntimeError(f"[step '{step_id}'] the agent's connection was lost and it cannot reload its session (no loadSession)")
        loading = True
        try:
            await conn.request("session/load", {"sessionId": session_id, "cwd": await session_cwd(), "mcpServers": session_mcp_servers()}, timeout=900)
        finally:
            loading = False

    try:
        if acp_state and acp_state.get("prompt_id") is not None:
            agent_url, token = acp_state.get("agent_url"), acp_state.get("token")
            session_id = acp_state["session_id"]
            prompt_id = acp_state["prompt_id"]
            conn = new_conn(acp_state["connection_id"], first_id=int(acp_state.get("next_id") or 1000))
            parked = acp_state.get("quota")
            if parked:
                quota_started = acp_state.get("quota_started")
                reattach_seq = int(acp_state.get("received") or 0)
                logger.info("[step '%s'] still parked on a usage limit; resuming the wait", step_id)
            else:
                logger.info("[step '%s'] re-attaching to ACP connection %s", step_id, acp_state["connection_id"])
                # Replay from the start: the turn's text is rebuilt from its updates.
                await conn.connect(last_seq=0)
                prompt_future = conn.expect(prompt_id)
        else:
            parked = None
            if runner is not None and runner_agent:
                known = await hub.active_limit(runner.id, runner_agent)
                if known is not None:
                    # Known to be out of quota: wait before taking the machine's time.
                    await park(LimitInfo(known.kind, known.until, known.message))
            if runner is None:
                token = secrets.token_urlsafe(32)
                env = dict(agent_config.get("env_vars") or {})
                env["ACP_PROXY_TOKEN"] = token
                agent_url = await runtime.spawn(agent_def, step, run_id, callback_base_url, extra_env=env)
            conn = await open_connection()
            carrier_meta = {**agent_config, "run_id": run_id, "task_id": task_key}
            if runner is not None:
                # A user's machine gets no carrier-side credentials: no step env,
                # no backend API keys, no tool secrets.
                carrier_meta.pop("env_vars", None)
                carrier_meta.pop("credentials", None)
                carrier_meta["tools"] = [{**t, "env": {}} for t in carrier_meta.get("tools") or []]
            meta: dict[str, Any] = {"carrier": carrier_meta}
            if agent_config.get("system_prompt"):
                # claude-agent-acp's slot for a system prompt; others ignore it.
                meta["systemPrompt"] = {"type": "preset", "preset": "claude_code", "append": agent_config["system_prompt"]}
            new = await conn.request("session/new", {
                "cwd": await session_cwd(),
                "mcpServers": session_mcp_servers(),
                "_meta": meta,
            }, timeout=900)
            session_id = new["sessionId"]
            if agent_task_repository is not None:
                await agent_task_repository.save_task({
                    "_id": task_key, "run_id": run_id, "step_id": step_id, "status": "working",
                    "input": input_data, "outputs": [],
                })
            await send_prompt(build_prompt_text(input_data))

        while True:
            if parked:
                limit = LimitInfo(parked.get("kind", "usage_limit"), _parse_iso(parked.get("resets_at")), parked.get("message", ""))
                parked = None
                await park(limit)
                await resume_session()
                await send_prompt(CONTINUE_PROMPT)
                continue
            assert prompt_future is not None
            try:
                result = await _await_turn(conn, session_id, prompt_future, run_id, sink)
                break
            except AcpMethodError as exc:
                limit = classify_limit(str(exc), exc.data, rate_limit=sink.rate_limit, tz=agent_tz)
                if limit is None:
                    raise
                if limit.kind == "quota_exhausted":
                    if runner is not None and runner_agent:
                        await hub.set_limit(runner.id, runner_agent, AgentLimit(until=None, kind=limit.kind, message=limit.message, updated_at=_utcnow()))
                    raise RuntimeError(f"[step '{step_id}'] the agent's account has no quota left: {limit.message}") from exc
                # What the agent said before the limit is progress, not the answer.
                _flush_text(sink, chunks)
                turn_text.clear()
                await park(limit)
                await resume_session()
                await send_prompt(CONTINUE_PROMPT)

        if runner is not None and runner_agent and quota_started is not None:
            await hub.set_limit(runner.id, runner_agent, None)
        assert conn is not None
        final_text, meta_out = _turn_result(result, turn_text, chunks)

        # Messages that reached the agent while it worked are its next turns,
        # in the same session: it keeps everything it did so far.
        for _ in range(MAX_MESSAGE_TURNS):
            queued = agent_inbox.take_messages(run_id)
            if not queued:
                break
            sink.add(f"passing {len(queued)} message(s) from the orchestrator to the agent")
            turn_text.clear()
            follow = asyncio.ensure_future(conn.request("session/prompt", {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": message_prompt(queued)}],
            }))
            result = await _await_turn(conn, session_id, follow, run_id, sink)
            follow_text, follow_meta = _turn_result(result, turn_text, chunks)
            if follow_text:
                final_text = follow_text
            meta_out = _merge_meta(meta_out, follow_meta)

        # The output protocol was not met: ask once more in the same session.
        expected = list((step.get("output_mapping") or {}).keys())
        if expected and final_text and not has_expected_fields(final_text, expected) and (result or {}).get("stopReason") == "end_turn":
            sink.add("asking the agent for the structured output fields")
            turn_text.clear()
            nudge = await conn.request("session/prompt", {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": (
                    "Your answer did not include the required output. Reply with ONLY a JSON object "
                    f"with these fields: {', '.join(expected)}. No prose."
                )}],
            })
            nudge_text, nudge_meta = _turn_result(nudge, turn_text, chunks)
            if nudge_text:
                final_text = nudge_text
            meta_out = _merge_meta(meta_out, nudge_meta)

        await sink.flush()
        stop = (result or {}).get("stopReason")
        if stop == "cancelled":
            raise RuntimeError(f"[step '{step_id}'] agent turn was cancelled")
        if agent_task_repository is not None:
            await agent_task_repository.update_task(task_key, {"status": "finished"})

        raw: dict[str, Any] = {"result": final_text or "(no output)"}
        if meta_out.get("usage"):
            raw["token_usage"] = meta_out["usage"]
        if meta_out.get("meta_usage"):
            raw["meta_token_usage"] = meta_out["meta_usage"]
        if meta_out.get("workspace_path"):
            raw["workspace_s3_path"] = meta_out["workspace_path"]
        elif workspace_uploaded and workspace_uri:
            raw["workspace_s3_path"] = workspace_uri
        if stop in ("max_tokens", "max_turn_requests", "refusal"):
            raw["stop_reason"] = stop
        return raw
    except asyncio.CancelledError:
        if conn is not None and acp_state is None and session_id is not None:
            try:
                await asyncio.wait_for(conn.notify("session/cancel", {"sessionId": session_id}), 5)
            except Exception:
                pass
        raise
    except Exception:
        if agent_task_repository is not None:
            try:
                await agent_task_repository.update_task(task_key, {"status": "failed"})
            except Exception:
                pass
        raise
    finally:
        await sink.flush()
        if conn is not None:
            if conn.started:
                try:
                    await asyncio.wait_for(conn.request("_proxy/shutdown", {}), 15)
                except Exception:
                    pass
            await conn.close()
        if agent_url is not None and runner is None:
            await _terminate(runtime, agent_def, run_id, agent_url)


def _session_cwd(agent_def: Any, runner: Any, run_id: str) -> str:
    """The session's working directory: ``agent_input.acp_cwd``, else on a runner
    a directory per run under the workspace it reported, else the pod's."""
    configured = (agent_def.agent_input or {}).get("acp_cwd")
    if configured:
        return configured
    if runner is None:
        return DEFAULT_CWD
    workspace = (runner.labels or {}).get("workspace")
    if not workspace:
        raise RuntimeError(
            f"machine '{runner.name}' did not report a workspace (acp-web-proxy older than 0.2?); "
            "set agent_input.acp_cwd on the agent to a directory on that machine"
        )
    sep = "\\" if runner.os == "win32" else "/"
    return workspace.rstrip("/\\") + sep + re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _limit_words(kind: str) -> str:
    return {
        "usage_limit": "usage limit reached",
        "rate_limited": "rate limited",
        "overloaded": "provider overloaded",
    }.get(kind, kind)


async def _terminate(runtime: Any, agent_def: Any, run_id: str, agent_url: str) -> None:
    try:
        if hasattr(runtime, "terminate_by_run_id"):
            await runtime.terminate_by_run_id(agent_def, run_id)
        else:
            await runtime.terminate(agent_url)
    except Exception:
        logger.warning("failed to terminate ACP agent at %s", agent_url, exc_info=True)


def _apply_update(update: dict[str, Any], sink: _ProgressSink, chunks: list[str], turn_text: list[str]) -> None:
    kind = update.get("sessionUpdate")
    if kind == "agent_message_chunk":
        content = update.get("content") or {}
        if content.get("type") == "text":
            text = content.get("text") or ""
            chunks.append(text)
            turn_text.append(text)
    elif kind == "agent_thought_chunk":
        return
    elif kind == "tool_call":
        # A new tool call ends the current stretch of assistant text: surface it.
        _flush_text(sink, turn_text_holder=chunks)
        call_id = str(update.get("toolCallId") or "")
        title = str(update.get("title") or update.get("kind") or "tool")
        server = ((update.get("_meta") or {}).get("carrier") or {}).get("mcp_server")
        sink.active_tools[call_id] = server or title.split(" ")[0]
        sink.add(f"🔧 {title}")
    elif kind == "tool_call_update":
        status = update.get("status")
        call_id = str(update.get("toolCallId") or "")
        if status in ("completed", "failed"):
            sink.active_tools.pop(call_id, None)
            if status == "failed":
                sink.add(f"⚠️ tool failed: {update.get('title') or call_id}")
    elif kind == "plan":
        entries = update.get("entries") or []
        lines = [f"- [{e.get('status', '')}] {e.get('content', '')}" for e in entries]
        if lines:
            sink.add("plan:\n" + "\n".join(lines))
    elif kind == "usage_update":
        sink.context_usage = {"used": update.get("used"), "size": update.get("size"), "cost": update.get("cost")}
        rate_limit = (update.get("_meta") or {}).get("_claude/rateLimit")
        if isinstance(rate_limit, dict):
            sink.rate_limit = rate_limit


def _flush_text(sink: _ProgressSink, turn_text_holder: list[str]) -> None:
    text = "".join(turn_text_holder).strip()
    turn_text_holder.clear()
    if text:
        sink.add(text if len(text) <= 500 else text[:500] + "…")


def _turn_result(result: Any, turn_text: list[str], chunks: list[str]) -> tuple[str, dict[str, Any]]:
    meta = (((result or {}).get("_meta") or {}).get("carrier") or {}) if isinstance(result, dict) else {}
    usage = meta.get("usage") or (result or {}).get("usage") if isinstance(result, dict) else None
    if usage and "total_tokens" not in usage and ("inputTokens" in usage or "input_tokens" in usage):
        i = usage.get("input_tokens", usage.get("inputTokens", 0)) or 0
        o = usage.get("output_tokens", usage.get("outputTokens", 0)) or 0
        usage = {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}
    # The answer is what the agent said after its last tool call; fall back to
    # the whole turn when the turn ended on a tool call.
    tail = "".join(chunks).strip()
    text = meta.get("final_text") or tail or "".join(turn_text).strip()
    turn_text.clear()
    chunks.clear()
    return text, {"usage": usage, "meta_usage": meta.get("meta_usage"), "workspace_path": meta.get("workspace_path")}


def _merge_meta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for key in ("usage", "meta_usage"):
        if a.get(key) and b.get(key):
            out[key] = {k: (a[key].get(k, 0) or 0) + (b[key].get(k, 0) or 0) for k in set(a[key]) | set(b[key])}
        elif b.get(key):
            out[key] = b[key]
    if b.get("workspace_path"):
        out["workspace_path"] = b["workspace_path"]
    return out


MAX_MESSAGE_TURNS = 10


def message_prompt(queued: list[dict[str, Any]]) -> str:
    lines = "\n\n".join(m["text"] for m in queued)
    return (
        f"Message from the orchestrator:\n{lines}\n\n"
        "Take it into account and continue the task. End with the required output, as before."
    )


async def _await_turn(conn: Any, session_id: str, turn: "asyncio.Future", run_id: str, sink: "_ProgressSink") -> Any:
    """Wait for a prompt turn; an urgent message cancels it so the next turn can carry the message."""
    while True:
        if agent_inbox.has_urgent_message(run_id) and not turn.done():
            sink.add("urgent message from the orchestrator: interrupting the current turn")
            try:
                await asyncio.wait_for(conn.notify("session/cancel", {"sessionId": session_id}), 5)
            except Exception:
                logger.debug("session/cancel failed", exc_info=True)
            result = await turn
            return {**(result or {}), "stopReason": "end_turn"} if (result or {}).get("stopReason") == "cancelled" else result
        waiter = asyncio.ensure_future(agent_inbox.message_event(run_id).wait())
        try:
            await asyncio.wait({turn, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if turn.done():
            return turn.result()
        if not agent_inbox.has_urgent_message(run_id):
            # A queued (not urgent) message: it waits for the turn to end.
            await turn
            return turn.result()


async def _notify_slack(run_repository: Any, run_id: str, question: str) -> None:
    try:
        from app.core.config import get_settings
        from app.infrastructure.notifications.webhook_notifier import post_slack_ask_context

        s = get_settings()
        if s.slack_bot_token and s.slack_approvals_channel and run_repository is not None:
            run = await run_repository.get(run_id)
            await post_slack_ask_context(s.slack_bot_token, s.slack_approvals_channel, [question], run_id, (run.state if run else {}) or {})
    except Exception:
        logger.debug("slack notification for ACP question failed", exc_info=True)


async def _ask_human(run_repository: Any, run_id: str, question: str, options: Any) -> str | None:
    """Surface an agent's free-text question the way /agent/question does, and wait.

    The answer arrives through POST /runs/{id}/agent/reply, which wakes the
    in-process event and also writes ``_pending_answer`` on the run — polled
    here so a reply handled by another replica is seen too.
    """
    from app.api.routes import agent_callbacks as cb

    opts = options if isinstance(options, list) else None
    cb._questions[run_id] = {"question": question, "options": opts}
    event = cb._get_or_create_event(run_id)
    event.clear()
    cb._answers.pop(run_id, None)
    if run_repository is not None:
        run = await run_repository.get(run_id)
        if run is not None:
            run.state = {k: v for k, v in (run.state or {}).items() if k != "_pending_answer"}
            run.state["_pending_question"] = {"question": question, "options": opts}
            run.touch()
            await run_repository.update(run)
    routed = False
    if run_repository is not None:
        # A dynamic-workflow job: its meta-agent answers (or asks a human itself).
        run = await run_repository.get(run_id)
        if run is not None and run.kind == "job" and run.parent_run_id:
            from app.infrastructure.orchestration.dynamic.hub import route_live_question

            routed = route_live_question(run.parent_run_id, run_id, question)
    if not routed:
        await _notify_slack(run_repository, run_id, question)

    deadline = time.monotonic() + ANSWER_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            await asyncio.wait_for(event.wait(), timeout=3)
        except asyncio.TimeoutError:
            pass
        answer = cb._answers.pop(run_id, None)
        if answer is None and run_repository is not None:
            run = await run_repository.get(run_id)
            if run is not None and (run.state or {}).get("_pending_answer") is not None:
                answer = str(run.state["_pending_answer"])
        if answer is not None:
            if run_repository is not None:
                run = await run_repository.get(run_id)
                if run is not None:
                    run.state = {k: v for k, v in (run.state or {}).items() if k not in ("_pending_answer", "_pending_question")}
                    run.touch()
                    await run_repository.update(run)
            cb._questions.pop(run_id, None)
            return answer
    return None
