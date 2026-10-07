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
from typing import TYPE_CHECKING, Any

from app.infrastructure.acp.client import AcpConnection, AcpError, to_ws_url

if TYPE_CHECKING:  # pragma: no cover
    from app.core.config import Settings
    from app.domain.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
CLIENT_INFO = {"name": "ai-agents-carrier", "version": "1"}
DEFAULT_CWD = "/workspace"
PROGRESS_FLUSH_S = 2.0
ANSWER_TIMEOUT_S = 30 * 60


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
) -> dict[str, Any]:
    """Run one agent step over ACP; returns the raw output for _finalize_agent_output."""
    step_id = step["id"]
    sink = _ProgressSink(run_repository, run_id, step_id)
    chunks: list[str] = []
    turn_text: list[str] = []
    blocked = list(agent_config.get("blocked_commands") or [])

    async def on_notification(method: str, params: dict[str, Any]) -> None:
        if method == "session/update":
            _apply_update(params.get("update") or {}, sink, chunks, turn_text)
            await sink.maybe_flush()
        elif method == "_proxy/agent_exited":
            sink.add(f"agent exited (code {params.get('code')})")

    async def on_request(method: str, params: dict[str, Any]) -> Any:
        if method == "session/request_permission":
            return pick_permission_option(params, blocked)
        if method == "_carrier/ask":
            answer = await _ask_human(run_repository, run_id, str(params.get("question") or ""), params.get("options"))
            return {"cancelled": True} if answer is None else {"answer": answer}
        raise AcpError(f"carrier does not implement {method}")

    # Re-attach to a turn still running in a pod we started before a restart.
    stored = await agent_task_repository.get_task(task_key) if agent_task_repository is not None else None
    acp_state = (stored or {}).get("acp") if isinstance(stored, dict) else None

    agent_url: str | None = None
    conn: AcpConnection | None = None
    try:
        if acp_state and acp_state.get("prompt_id") is not None:
            agent_url = acp_state["agent_url"]
            conn = AcpConnection(
                to_ws_url(agent_url), acp_state["token"],
                connection_id=acp_state["connection_id"],
                on_notification=on_notification, on_request=on_request,
            )
            logger.info("[step '%s'] re-attaching to ACP connection %s", step_id, acp_state["connection_id"])
            # Replay from the start: the turn's text is rebuilt from its updates.
            await conn.connect(last_seq=0)
            prompt_future = conn.expect(acp_state["prompt_id"])
            session_id = acp_state["session_id"]
        else:
            token = secrets.token_urlsafe(32)
            env = dict(agent_config.get("env_vars") or {})
            env["ACP_PROXY_TOKEN"] = token
            agent_url = await runtime.spawn(agent_def, step, run_id, callback_base_url, extra_env=env)
            conn = AcpConnection(to_ws_url(agent_url), token, on_notification=on_notification, on_request=on_request)
            await conn.connect()

            launch_params: dict[str, Any] = {}
            if agent_def.acp_agent:
                launch_params["agent"] = agent_def.acp_agent
            launched = await conn.request("_proxy/launch", launch_params, timeout=600)
            conn.connection_id = conn.connection_id or (launched or {}).get("connection_id")
            init = await conn.request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False}, "terminal": False},
                "clientInfo": CLIENT_INFO,
            }, timeout=300)
            auth_method = (agent_def.agent_input or {}).get("acp_auth_method")
            if auth_method:
                await conn.request("authenticate", {"methodId": auth_method}, timeout=120)
            caps = (init or {}).get("agentCapabilities") or {}
            mcp_servers = to_acp_mcp_servers(agent_config.get("mcp_servers"))
            mcp_caps = caps.get("mcpCapabilities") or {}
            dropped = [
                s["name"] for s in mcp_servers
                if (s.get("type") == "http" and not mcp_caps.get("http"))
                or (s.get("type") == "sse" and not mcp_caps.get("sse"))
            ]
            if dropped:
                sink.add(f"agent cannot reach MCP servers over HTTP/SSE, skipped: {', '.join(dropped)}")
                mcp_servers = [s for s in mcp_servers if s["name"] not in dropped]
            meta: dict[str, Any] = {"carrier": {**agent_config, "run_id": run_id, "task_id": task_key}}
            if agent_config.get("system_prompt"):
                # claude-agent-acp's slot for a system prompt; others ignore it.
                meta["systemPrompt"] = {"type": "preset", "preset": "claude_code", "append": agent_config["system_prompt"]}
            new = await conn.request("session/new", {
                "cwd": (agent_def.agent_input or {}).get("acp_cwd") or DEFAULT_CWD,
                "mcpServers": mcp_servers,
                "_meta": meta,
            }, timeout=900)
            session_id = new["sessionId"]

            prompt_id = conn.next_id()
            if agent_task_repository is not None:
                await agent_task_repository.save_task({
                    "_id": task_key, "run_id": run_id, "step_id": step_id, "status": "working",
                    "input": input_data, "outputs": [],
                    "acp": {"agent_url": agent_url, "token": token, "connection_id": conn.connection_id,
                            "session_id": session_id, "prompt_id": prompt_id},
                })
            prompt_future = asyncio.ensure_future(conn.request("session/prompt", {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": build_prompt_text(input_data)}],
            }, req_id=prompt_id))

        result = await prompt_future
        final_text, meta_out = _turn_result(result, turn_text, chunks)

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
        if stop in ("max_tokens", "max_turn_requests", "refusal"):
            raw["stop_reason"] = stop
        return raw
    except asyncio.CancelledError:
        if conn is not None and acp_state is None:
            try:
                await asyncio.wait_for(conn.notify("session/cancel", {"sessionId": locals().get("session_id")}), 5)
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
            try:
                await asyncio.wait_for(conn.request("_proxy/shutdown", {}), 15)
            except Exception:
                pass
            await conn.close()
        if agent_url is not None:
            await _terminate(runtime, agent_def, run_id, agent_url)


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
    try:
        from app.core.config import get_settings
        from app.infrastructure.notifications.webhook_notifier import post_slack_ask_context

        s = get_settings()
        if s.slack_bot_token and s.slack_approvals_channel and run_repository is not None:
            run = await run_repository.get(run_id)
            await post_slack_ask_context(s.slack_bot_token, s.slack_approvals_channel, [question], run_id, (run.state if run else {}) or {})
    except Exception:
        logger.debug("slack notification for ACP question failed", exc_info=True)

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
