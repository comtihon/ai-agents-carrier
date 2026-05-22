"""Claude SDK agent using the langgraph-backend HTTP protocol.

This example shows how to write a Claude-based agent that:
1. Receives input from the backend via POST /start
2. Optionally asks clarifying questions via BackendClient.ask_question
3. Reports its output via BackendClient.send_output

The agent coroutine signature must be::

    async def run(input_data: dict, config: AgentConfig, client: BackendClient) -> None

Run locally with the agent server
----------------------------------
    AGENT_PORT=18001 \\
    BACKEND_CALLBACK_URL=http://localhost:8000 \\
    RUN_ID=test-run-1 \\
    ANTHROPIC_API_KEY=sk-ant-... \\
    python -m agent_server --port 18001 --agent agent:run

Docker
------
    docker build -t my-claude-agent .
    # The runtime injects env vars automatically when spawning the container.
"""
from __future__ import annotations

import os
import sys

# Add the parent directory to sys.path so we can import agent_server when
# running the agent standalone (outside of a proper package install).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from typing import Any

# ---------------------------------------------------------------------------
# Anthropic SDK — install with:  pip install anthropic
# ---------------------------------------------------------------------------
import anthropic

from agent_server import AgentConfig
from agent_server.backend_client import BackendClient


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------

def call_claude(
    request: str,
    system_prompt: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """Call Claude and return the text response."""
    claude = anthropic.Anthropic(api_key=api_key)  # api_key=None → uses ANTHROPIC_API_KEY env var

    effective_system = system_prompt or "You are a helpful assistant. Answer the user's request concisely."
    effective_model = model or "claude-3-5-haiku-20241022"

    message = claude.messages.create(
        model=effective_model,
        max_tokens=max_tokens,
        system=effective_system,
        messages=[{"role": "user", "content": request}],
    )

    for block in message.content:
        if block.type == "text":
            return block.text

    return ""


# ---------------------------------------------------------------------------
# Agent entrypoint — called by the agent server on POST /start
# ---------------------------------------------------------------------------

async def run(input_data: dict[str, Any], config: AgentConfig, client: BackendClient) -> None:
    """Main agent coroutine.

    Parameters
    ----------
    input_data:
        The input dict sent by the backend (derived from workflow state via
        ``input_mapping``).
    config:
        Agent configuration forwarded from the backend, including model,
        system_prompt, and resolved API credentials.
    client:
        BackendClient instance pre-configured with the run_id and callback URL.
    """
    await client.send_progress("Processing request…")

    request = input_data.get("request", "")
    if not request:
        await client.send_output({"error": "No 'request' key found in input"})
        return

    # Use credentials forwarded from the backend when available.
    # Falls back to the ANTHROPIC_API_KEY environment variable if not provided.
    api_key = config.credentials.get("ANTHROPIC_API_KEY")

    # Example: ask a clarifying question before running (optional)
    # scope = await client.ask_question(
    #     "What level of detail do you want?",
    #     options=["brief", "detailed"],
    # )

    try:
        answer = call_claude(
            request,
            system_prompt=config.system_prompt,
            model=config.model,
            api_key=api_key,
            max_tokens=int(config.extra.get("max_tokens", 1024)),
        )
    except Exception as exc:
        await client.send_output({"error": str(exc)})
        return

    await client.send_output({"answer": answer})
