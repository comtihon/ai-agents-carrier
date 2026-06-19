"""Built-in local LangGraph ReAct agent.

Runs inline inside the langgraph-backend process — no subprocess, no HTTP.
Called directly by ``app.steps.agent_executor`` when ``runtime == "local"``.

The implementation mirrors ``langgraph-agent/langgraph/agent.py`` but is
self-contained: it reads credentials and MCP server configs from ``settings``
instead of receiving them via an HTTP payload.
"""
from __future__ import annotations

import json
import subprocess
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

if TYPE_CHECKING:
    from app.core.config import Settings

_DEFAULT_SYSTEM = (
    "You are a capable DevOps and software engineering agent with access to a "
    "bash shell (kubectl, gcloud, helm, git are all installed) and code-search "
    "tools. Complete the requested task step by step. Be concise and precise."
)

_DEFAULT_MODEL = "claude-opus-4-7"
_DEFAULT_MAX_TOKENS = 8096


@tool
def bash(command: str) -> str:
    """Execute a shell command and return stdout + stderr.

    Use this for kubectl, gcloud, helm, git, and any other CLI tools.
    Commands run with a 120-second timeout.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if err:
            out = f"{out}\n[stderr]\n{err}".strip()
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] command timed out after 120 s"
    except Exception as exc:
        return f"[error] {exc}"


@asynccontextmanager
async def _mcp_tools(settings: "Settings", allowed_tools: set[str] | None):
    """Yield a list of LangChain tools from all configured MCP servers."""
    from app.core.config import McpIntegrationConfig

    raw_integrations: list[McpIntegrationConfig] = settings.get_mcp_integrations()

    server_map: dict[str, Any] = {}
    for intg in raw_integrations:
        if allowed_tools is not None and intg.name not in allowed_tools:
            continue
        if intg.transport == "stdio" and intg.command:
            server_map[intg.name] = {
                "command": intg.command,
                "args": intg.args,
                "env": intg.env or {},
                "transport": "stdio",
            }
        elif intg.url:
            server_map[intg.name] = {
                "url": intg.url,
                "transport": intg.transport or "sse",
            }

    if not server_map:
        yield []
        return

    async with MultiServerMCPClient(server_map) as mcp_client:
        yield mcp_client.get_tools()


async def run_local_agent(
    agent_input: dict[str, Any],
    input_data: dict[str, Any],
    settings: "Settings",
    progress_cb: Callable[[str], Awaitable[None]] | None = None,
    compression_level: str = "none",
    allowed_mcp: set[str] | None = None,
) -> dict[str, Any]:
    """Run a LangGraph ReAct agent inline and return its output dict.

    Parameters
    ----------
    agent_input:
        ``AgentDefinition.agent_input`` — runtime configuration for the agent.
        Recognised keys: ``system_prompt``, ``model``, ``max_tokens``, ``tools``
        (list of MCP server names to allow; ``None`` means all enabled).
    input_data:
        The step's input dict built from the workflow state via ``input_mapping``.
        The request / task is read from ``input_data["request"]``,
        ``input_data["task"]``, ``input_data["prompt"]``, or the whole dict is
        JSON-serialised as a fallback.
    settings:
        App ``Settings`` instance — used to resolve LLM credentials and MCP
        server configurations.
    progress_cb:
        Optional async callback called with intermediate AI message text as the
        agent streams its reasoning steps.

    Returns
    -------
    dict
        ``{"answer": <final AI message content>}``
    """
    async def _progress(msg: str) -> None:
        if progress_cb is not None:
            await progress_cb(msg)

    # --- Extract config from agent_input ---
    from app.steps.agent_executor import _COMPRESSION_INSTRUCTIONS
    base_system = agent_input.get("system_prompt") or _DEFAULT_SYSTEM
    compression_instruction = _COMPRESSION_INSTRUCTIONS.get(compression_level or "none", "")
    system_prompt: str = (
        f"{compression_instruction}\n\n{base_system}" if compression_instruction else base_system
    )
    provider: str | None = agent_input.get("llm_provider") or agent_input.get("provider")
    model: str | None = agent_input.get("model") or _DEFAULT_MODEL
    max_tokens: int = int(agent_input.get("max_tokens") or _DEFAULT_MAX_TOKENS)
    if allowed_mcp is not None:
        allowed_tools: set[str] | None = allowed_mcp
    else:
        allowed_tools_list: list[str] | None = agent_input.get("tools")
        allowed_tools = set(allowed_tools_list) if allowed_tools_list is not None else None

    # --- Resolve the task / request ---
    request: str = (
        input_data.get("request")
        or input_data.get("task")
        or input_data.get("prompt")
        or json.dumps(input_data)
    )

    await _progress("Initialising LangGraph ReAct agent…")

    from app.core.container import build_llm_native
    llm = build_llm_native(provider, model, settings, max_tokens=max_tokens)

    async with _mcp_tools(settings, allowed_tools) as extra_tools:
        all_tools = [bash, *extra_tools]
        if extra_tools:
            await _progress(
                f"Connected to {len(extra_tools)} MCP tool(s); building graph…"
            )

        agent = create_react_agent(llm, all_tools, prompt=system_prompt)

        await _progress("Running…")

        # Stream so we can forward intermediate AI messages as progress updates.
        final_messages: list[Any] = []
        async for chunk in agent.astream({"messages": [("human", request)]}):
            for key in ("agent", "tools"):
                if key not in chunk:
                    continue
                msgs = chunk[key].get("messages", [])
                final_messages = msgs  # keep updating; last batch wins
                if key == "agent":
                    for msg in msgs:
                        if (
                            isinstance(msg, AIMessage)
                            and isinstance(msg.content, str)
                            and msg.content.strip()
                        ):
                            await _progress(msg.content[:300])

    # Last AI message is the final answer.
    answer = ""
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage):
            answer = (
                msg.content
                if isinstance(msg.content, str)
                else json.dumps(msg.content)
            )
            break

    return {"answer": answer or "(no output)"}
