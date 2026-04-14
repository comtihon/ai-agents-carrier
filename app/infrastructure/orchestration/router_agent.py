"""
LangGraph router agent for the CopilotKit chat runtime.

The agent has a single ``router_node`` that invokes the LLM with a system
prompt and the current conversation messages.  Workflow operations
(start, approve, reject, status) are handled by CopilotKit backend
actions registered in ``app.api.app``; the router itself is a
conversational assistant that explains the platform and guides users.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from copilotkit import CopilotKitState
from copilotkit.langgraph import copilotkit_customize_config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an AI assistant for a software engineering team's LangGraph workflow platform.

## What you can do

- **Explain workflows** — describe what each available workflow does and when to use it.
- **Guide users** — help users understand how to start, approve, reject, or check the
  status of workflow runs via the UI controls.
- **Answer questions** — answer general questions about the platform, LangGraph, or
  software engineering topics.

## Guidelines

- Be concise and action-oriented.
- When a user asks to start a workflow, confirm the workflow name and describe what
  the agent will do, then invite them to use the Start button.
- When a run is waiting for approval, summarise the plan and invite them to approve
  or reject via the UI controls.
- Always confirm destructive actions (reject) before the user proceeds.
"""


def build_router_graph(llm: BaseChatModel):
    """Compile and return the CopilotKit router LangGraph."""

    async def router_node(state: CopilotKitState, config: RunnableConfig) -> dict:  # type: ignore[type-arg]
        ck_config = copilotkit_customize_config(config, emit_messages=True)
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state["messages"])
        response = await llm.ainvoke(messages, config=ck_config)
        return {"messages": [response]}

    sg: StateGraph = StateGraph(CopilotKitState)
    sg.add_node("router", router_node)
    sg.add_edge(START, "router")
    sg.add_edge("router", END)
    return sg.compile()
