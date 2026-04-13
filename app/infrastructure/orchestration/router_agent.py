"""
LangGraph router agent for the CopilotKit chat runtime.

The agent has a single `router_node` that:
  1. Reads the frontend actions injected by CopilotKit
     (startWorkflow, approveWorkflow, rejectWorkflow, getWorkflowStatus, …)
  2. Converts them to LangChain tool specs
  3. Calls the LLM — which either:
       a. Invokes a frontend action (workflow start, approval, status query)
       b. Answers the user directly from its knowledge / context

This IS the "context node" that understands the user's intent and routes
the request.  No explicit decision tree is needed: the LLM reasons over the
system prompt, the readable state (workflow runs, available definitions) and
the tool definitions, then picks the right path.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig

from copilotkit.langgraph import CopilotKitState, copilotkit_customize_config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an AI assistant for a software engineering team's LangGraph workflow platform.

## What you can do

- **Start a workflow** — when the user wants to create a feature, fix a bug, run an agent
  task, or anything that maps to a known workflow → call `startWorkflow`.
  Pick the most suitable `workflowId` from the available workflows in context.
  Write a clear, detailed `userRequest` describing exactly what the agent should do.

- **Approve or reject a pending workflow** → call `approveWorkflow` or `rejectWorkflow`.

- **Show pending approvals** → call `reviewPendingApprovals` to render inline approval cards.

- **Check run status** → call `getWorkflowStatus` with the run ID.

- **Approve / reject a specific gate** → call `approveGate` or `rejectGate`.

- **Answer directly** — for general questions about workflows, the platform, or anything
  that doesn't require triggering an action.

## Guidelines

- Be concise and action-oriented.
- When starting a workflow, confirm the workflow name and what the agent will do.
- When a run is waiting_approval, proactively offer to show the approval UI.
- Always confirm destructive actions (reject) before proceeding.
"""


def _action_to_tool(action: dict[str, Any]) -> dict[str, Any]:
    """Convert a CopilotKit frontend action spec to an OpenAI-style tool dict."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in action.get("parameters", []):
        name = param["name"]
        prop: dict[str, Any] = {"type": param.get("type", "string")}
        if "description" in param:
            prop["description"] = param["description"]
        properties[name] = prop
        if param.get("required", False):
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": action["name"],
            "description": action.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                **({"required": required} if required else {}),
            },
        },
    }


def build_router_graph(llm: BaseChatModel):
    """Compile and return the CopilotKit router LangGraph."""

    async def router_node(state: CopilotKitState, config: RunnableConfig) -> dict:  # type: ignore[type-arg]
        # Customise config so CopilotKit can stream messages back to the frontend
        ck_config = copilotkit_customize_config(config, emit_messages=True)

        # Extract frontend actions from the CopilotKit state field
        ck_props = state.get("copilotkit") or {}
        raw_actions: list[dict[str, Any]] = (
            ck_props.get("actions", [])
            if isinstance(ck_props, dict)
            else getattr(ck_props, "actions", [])
        )
        tools = [_action_to_tool(a) for a in raw_actions]

        # Prepend system message with routing instructions
        messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state["messages"])

        if tools:
            active_llm = llm.bind_tools(tools)  # type: ignore[arg-type]
        else:
            active_llm = llm

        response = await active_llm.ainvoke(messages, config=ck_config)
        return {"messages": [response]}

    sg: StateGraph = StateGraph(CopilotKitState)
    sg.add_node("router", router_node)
    sg.add_edge(START, "router")
    sg.add_edge("router", END)
    return sg.compile()
