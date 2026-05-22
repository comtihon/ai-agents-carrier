"""LangGraph agent using the langgraph-backend HTTP protocol.

This example shows how to write a LangGraph-based agent that:
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
    python -m agent_server --port 18001 --agent agent:run

Docker
------
    docker build -t my-langgraph-agent .
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
# LangGraph imports — install with:  pip install langgraph langchain-openai
# ---------------------------------------------------------------------------
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from agent_server import AgentConfig
from agent_server.backend_client import BackendClient


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    request: str
    answer: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def call_llm(state: AgentState, model: str = "gpt-4o-mini") -> AgentState:
    """Call an LLM and store the response in ``answer``."""
    llm = ChatOpenAI(model=model)
    response = llm.invoke([HumanMessage(content=state["request"])])
    return {"answer": response.content}


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_graph(model: str = "gpt-4o-mini"):
    sg = StateGraph(AgentState)
    sg.add_node("call_llm", lambda s: call_llm(s, model=model))
    sg.add_edge(START, "call_llm")
    sg.add_edge("call_llm", END)
    return sg.compile()


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
        Agent configuration forwarded from the backend, including model, system
        prompt, MCP server configs, and resolved API credentials.
    client:
        BackendClient instance pre-configured with the run_id and callback URL.
    """
    await client.send_progress("Building graph…")

    request = input_data.get("request", "")
    if not request:
        await client.send_output({"error": "No 'request' key found in input"})
        return

    # Use the model specified in AgentConfig (falls back to gpt-4o-mini).
    model = config.model or "gpt-4o-mini"

    # Use credentials forwarded from the backend if available.
    openai_api_key = config.credentials.get("OPENAI_API_KEY")
    if openai_api_key:
        import os
        os.environ.setdefault("OPENAI_API_KEY", openai_api_key)

    # Example: ask a clarifying question before running (optional)
    # scope = await client.ask_question(
    #     "What scope should I search?",
    #     options=["broad", "narrow"],
    # )
    # You can use `scope` to influence the agent's behaviour.

    await client.send_progress("Invoking LangGraph…")
    graph = build_graph(model=model)
    result = graph.invoke({"request": request})

    await client.send_output({"answer": result["answer"]})
