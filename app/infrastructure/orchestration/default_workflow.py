"""Default chat agent — the entry point for all copilot chat interactions.

A ReAct-style LangGraph agent backed by the bundled workflow_assistant.yaml
config.  The agent has built-in tools for listing, running, and inspecting
workflow runs, plus an ask_user tool that pauses execution via interrupt() to
collect clarifying answers from the user.

The system prompt and LLM provider are loaded from the YAML config so the
agent can be customised without code changes.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Annotated, Any
from uuid import uuid4

from copilotkit import CopilotKitState
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig, interrupt

from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.yaml_graph import stream_graph_to_pause

if TYPE_CHECKING:
    from app.infrastructure.config.graph_loader import YamlGraphRegistry
    from app.infrastructure.persistence.mongo import MongoGraphRunRepository

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """\
You are the Workflow Assistant for this workflow automation platform.
Use your tools to help users run, inspect, and understand workflows.
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AssistantState(CopilotKitState, total=False):  # type: ignore[misc]
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_default_workflow(
    llm: BaseChatModel,
    registry: "YamlGraphRegistry",
    run_repository: "MongoGraphRunRepository",
    checkpointer: BaseCheckpointSaver | None = None,
    agent_config: dict | None = None,
):
    """Build and compile the default ReAct chat agent."""

    config = agent_config or {}
    system_prompt_template = config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT).strip()

    # ── tools ────────────────────────────────────────────────────────────────

    @tool
    def list_workflows() -> str:
        """List all available workflow IDs, names, and descriptions."""
        defs = registry.list_definitions()
        if not defs:
            return "No workflows are currently configured."
        lines = [
            f"- **{d['id']}** ({d.get('name', d['id'])}): {(d.get('description') or '').strip()}"
            for d in defs
        ]
        return "\n".join(lines)

    @tool
    async def run_workflow(workflow_id: str, request: str) -> str:
        """Start a workflow run.

        Args:
            workflow_id: The workflow ID (from list_workflows).
            request: A detailed description of the task to execute.

        Returns:
            JSON with run_id, workflow_id, workflow_name, and __event__ = workflow_started.
        """
        runner = registry.get(workflow_id)
        if runner is None:
            available = ", ".join(registry.list_ids()) or "none"
            return f"Workflow '{workflow_id}' not found. Available: {available}"

        run_id = str(uuid4())
        child_run = GraphRun(
            id=run_id,
            graph_id=workflow_id,
            user_request=request,
            status="running",
            step_statuses={s["id"]: "pending" for s in runner.steps},
        )
        await run_repository.create(child_run)
        asyncio.create_task(
            stream_graph_to_pause(runner, child_run, run_repository, {"request": request})
        )
        logger.info("chat_agent: spawned '%s' as run %s", workflow_id, run_id)
        return json.dumps({
            "__event__": "workflow_started",
            "run_id": run_id,
            "workflow_id": workflow_id,
            "workflow_name": runner.name,
        })

    @tool
    async def list_runs(workflow_id: str | None = None, limit: int = 10) -> str:
        """List recent workflow runs.

        Args:
            workflow_id: Filter to a specific workflow (optional).
            limit: Maximum number of runs to return (default 10).
        """
        try:
            runs = await run_repository.list(
                workflow_id=workflow_id,
                limit=min(limit, 20),
            )
        except TypeError:
            # fallback for backends that don't support keyword args
            runs = await run_repository.list()
            if workflow_id:
                runs = [r for r in runs if r.graph_id == workflow_id]
            runs = runs[:limit]

        if not runs:
            return "No runs found."
        lines = [
            f"- **{r.id}** ({r.graph_id}) — status: {r.status}"
            + (f", started: {r.created_at}" if getattr(r, "created_at", None) else "")
            for r in runs
        ]
        return "\n".join(lines)

    @tool
    async def get_run(run_id: str) -> str:
        """Get detailed status and step-level output for a specific workflow run.

        Args:
            run_id: The run ID to inspect.
        """
        run = await run_repository.get(run_id)
        if run is None:
            return f"Run '{run_id}' not found."
        parts = [
            f"Run: {run.id}",
            f"Workflow: {run.graph_id}",
            f"Status: {run.status}",
        ]
        if run.step_statuses:
            parts.append("Steps:")
            for step_id, status in run.step_statuses.items():
                parts.append(f"  - {step_id}: {status}")
        if run.state:
            # Include output values for failed/finished steps — skip internal keys
            output_keys = [k for k in run.state if not k.startswith("_")]
            if output_keys:
                parts.append("State keys: " + ", ".join(output_keys))
                for k in output_keys[:8]:  # cap to avoid huge responses
                    v = run.state[k]
                    if isinstance(v, dict) and "error" in v:
                        parts.append(f"  {k}.error: {v['error'][:300]}")
                    elif isinstance(v, dict) and "status" in v:
                        parts.append(f"  {k}.status: {v.get('status')} {str(v.get('body',''))[:200]}")
        if run.error:
            parts.append(f"Error: {run.error[:500]}")
        return "\n".join(parts)

    @tool
    def ask_user(questions: list[str]) -> str:
        """Pause and ask the user clarifying questions before proceeding.

        Use this only when you genuinely cannot act without more information.
        Ask 1-3 focused questions.

        Args:
            questions: List of questions to ask the user.
        """
        answers: dict = interrupt({"type": "ask_context", "questions": questions})
        return "\n".join(
            f"Q: {q}\nA: {answers.get(str(i), '').strip()}"
            for i, q in enumerate(questions)
        )

    tools = [list_workflows, run_workflow, list_runs, get_run, ask_user]
    llm_with_tools = llm.bind_tools(tools)

    # ── nodes ────────────────────────────────────────────────────────────────

    def _build_system_prompt() -> str:
        """Build system prompt with current workflow list injected."""
        defs = registry.list_definitions()
        if defs:
            workflow_lines = "\n".join(
                f"- **{d['id']}**: {(d.get('description') or '').strip()}"
                for d in defs
            )
            return f"{system_prompt_template}\n\nAvailable workflows:\n{workflow_lines}"
        return system_prompt_template

    async def agent(state: AssistantState, config: RunnableConfig) -> dict:
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=_build_system_prompt())] + list(state.get("messages", []))
        response = await llm_with_tools.ainvoke(messages, config)
        return {"messages": [response]}

    def route(state: AssistantState) -> str:
        msgs = state.get("messages", [])
        if not msgs:
            return END
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    # ── graph ────────────────────────────────────────────────────────────────

    sg: StateGraph = StateGraph(AssistantState)
    sg.add_node("agent", agent)
    sg.add_node("tools", ToolNode(tools))

    sg.add_edge(START, "agent")
    sg.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    sg.add_edge("tools", "agent")

    return sg.compile(checkpointer=checkpointer or MemorySaver())
