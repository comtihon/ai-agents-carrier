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
    from collections.abc import Awaitable, Callable

    from app.infrastructure.config.graph_loader import YamlGraphRegistry
    from app.infrastructure.persistence.agent_backend import AgentDefinitionBackend
    from app.infrastructure.persistence.mongo import MongoGraphRunRepository
    from app.infrastructure.persistence.workflow_backend import WorkflowDefinitionBackend

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
    workflow_backend: "WorkflowDefinitionBackend | None" = None,
    refresh_runner: "Callable[[str], Awaitable[None]] | None" = None,
    agent_backend: "AgentDefinitionBackend | None" = None,
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

    async def _resolve_workflow_id(query: str):
        """Returns (resolved_id, None) or (None, error_str)."""
        defs = registry.list_definitions()
        for d in defs:
            if d["id"] == query:
                return d["id"], None
        for d in defs:
            if d.get("name", "").lower() == query.lower():
                return d["id"], None
        matches = [d for d in defs if query.lower() in d["id"].lower() or query.lower() in d.get("name", "").lower()]
        if len(matches) == 1:
            return matches[0]["id"], None
        if matches:
            cands = ", ".join(f"{d['id']} ({d.get('name', d['id'])})" for d in matches)
            return None, f"Ambiguous — multiple matches: {cands}"
        available = ", ".join(f"{d['id']} ({d.get('name', d['id'])})" for d in defs) or "none"
        return None, f"Workflow '{query}' not found. Available: {available}"

    @tool
    async def run_workflow(workflow_id: str, request: str) -> str:
        """Start a workflow run.

        Args:
            workflow_id: The workflow ID (from list_workflows).
            request: A detailed description of the task to execute.

        Returns:
            JSON with run_id, workflow_id, workflow_name, and __event__ = workflow_started.
        """
        resolved, err = await _resolve_workflow_id(workflow_id)
        if err:
            return err
        workflow_id = resolved
        runner = registry.get(workflow_id)
        if runner is None:
            return f"Workflow '{workflow_id}' not found."

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

    @tool
    async def create_workflow(workflow_id: str, name: str, description: str, steps_json: str) -> str:
        """Create a new workflow definition and register it immediately.

        Args:
            workflow_id: Unique kebab-case identifier (e.g. "send-slack-report").
            name: Human-readable display name.
            description: What this workflow does.
            steps_json: JSON array of step objects. Each step must have "id" and "type".
                Supported types: http (webhook trigger), cron (scheduled trigger),
                llm_structured (LLM with structured output), llm (free-form LLM),
                mcp (single MCP tool call), human_approval (pause for approval),
                execute (OpenHands code execution), workflow (child workflow),
                http_call (outbound HTTP), langgraph-agent, claude-agent.
                Example: [{"id": "trigger", "type": "http"}, {"id": "research", "type": "llm_structured", "system_prompt": "...", "output": [{"name": "summary", "type": "str", "description": "..."}]}]
        """
        if workflow_backend is None:
            return "Workflow creation unavailable: no persistent backend configured."
        try:
            steps = json.loads(steps_json)
        except json.JSONDecodeError as exc:
            return f"Invalid steps_json: {exc}"
        if not isinstance(steps, list):
            return "steps_json must be a JSON array."

        existing = await workflow_backend.get(workflow_id)
        if existing is not None:
            return f"Workflow '{workflow_id}' already exists. Use update_workflow to modify it."

        from app.domain.models.workflow_definition import WorkflowDefinition
        defn = WorkflowDefinition(id=workflow_id, name=name, description=description, steps=steps)
        await workflow_backend.create(defn)
        if refresh_runner is not None:
            await refresh_runner(workflow_id)
        return f"Workflow '{workflow_id}' created with {len(steps)} step(s)."

    @tool
    async def update_workflow(
        workflow_id: str,
        name: str | None = None,
        description: str | None = None,
        steps_json: str | None = None,
    ) -> str:
        """Update an existing workflow definition (name, description, and/or steps).

        Args:
            workflow_id: The workflow ID to update.
            name: New display name (omit to keep current).
            description: New description (omit to keep current).
            steps_json: JSON array replacing ALL steps (omit to keep current).
        """
        if workflow_backend is None:
            return "Workflow updates unavailable: no persistent backend configured."

        resolved, err = await _resolve_workflow_id(workflow_id)
        if err:
            return err
        workflow_id = resolved
        defn = await workflow_backend.get(workflow_id)
        if defn is None:
            return f"Workflow '{workflow_id}' not found."
        if defn.readonly:
            return f"Workflow '{workflow_id}' is read-only and cannot be modified."

        if name is not None:
            defn.name = name
        if description is not None:
            defn.description = description
        if steps_json is not None:
            try:
                steps = json.loads(steps_json)
            except json.JSONDecodeError as exc:
                return f"Invalid steps_json: {exc}"
            if not isinstance(steps, list):
                return "steps_json must be a JSON array."
            defn.steps = steps

        await workflow_backend.update(workflow_id, defn)
        if refresh_runner is not None:
            await refresh_runner(workflow_id)
        return f"Workflow '{workflow_id}' updated."

    @tool
    async def delete_workflow(workflow_id: str) -> str:
        """Permanently delete a workflow definition.

        Args:
            workflow_id: The workflow ID to delete.
        """
        if workflow_backend is None:
            return "Workflow deletion unavailable: no persistent backend configured."

        resolved, err = await _resolve_workflow_id(workflow_id)
        if err:
            return err
        workflow_id = resolved
        defn = await workflow_backend.get(workflow_id)
        if defn is None:
            return f"Workflow '{workflow_id}' not found."
        if defn.readonly:
            return f"Workflow '{workflow_id}' is read-only and cannot be deleted."

        await workflow_backend.delete(workflow_id)
        registry._runners.pop(workflow_id, None)
        return f"Workflow '{workflow_id}' deleted."

    # --- Agent tools ---

    async def _resolve_agent_id(query: str):
        """Returns (resolved_id, None) or (None, error_str)."""
        if agent_backend is None:
            return None, "agent_backend not configured"
        agents = await agent_backend.list()
        # exact id match
        for a in agents:
            if a.id == query:
                return a.id, None
        # exact name match (case-insensitive)
        for a in agents:
            if a.name.lower() == query.lower():
                return a.id, None
        # substring match
        matches = [a for a in agents if query.lower() in a.id.lower() or query.lower() in a.name.lower()]
        if len(matches) == 1:
            return matches[0].id, None
        if matches:
            cands = ", ".join(f"{a.id} ({a.name})" for a in matches)
            return None, f"Ambiguous — multiple matches: {cands}"
        return None, f"No agent found matching '{query}'. Available: {', '.join(f'{a.id} ({a.name})' for a in agents)}"

    @tool
    async def list_agents() -> str:
        """List all available agent definitions."""
        if agent_backend is None:
            return "Agent backend not configured."
        agents = await agent_backend.list()
        if not agents:
            return "No agents found."
        lines = [f"- **{a.id}** ({a.name}): {a.description or '(no description)'}" for a in agents]
        return "\n".join(lines)

    @tool
    async def get_agent(agent_id: str) -> str:
        """Get full agent definition by id or name."""
        if agent_backend is None:
            return "Agent backend not configured."
        resolved, err = await _resolve_agent_id(agent_id)
        if err:
            return err
        agent = await agent_backend.get(resolved)
        if agent is None:
            return f"Agent '{resolved}' not found."
        import json as _json
        return _json.dumps(agent.model_dump(mode="json"), indent=2, default=str)

    @tool
    async def create_agent(agent_id: str, name: str, description: str = "", default_runtime: str = "local", agent_input_json: str = "{}") -> str:
        """Create a new agent definition. agent_input_json is a JSON object of default input overrides."""
        if agent_backend is None:
            return "Agent backend not configured."
        import json as _json
        try:
            agent_input = _json.loads(agent_input_json)
            if not isinstance(agent_input, dict):
                return "agent_input_json must be a JSON object."
        except Exception as e:
            return f"Invalid agent_input_json: {e}"
        if default_runtime not in ("local", "docker", "k8s"):
            return f"Invalid default_runtime '{default_runtime}'. Must be one of: local, docker, k8s."
        existing = await agent_backend.get(agent_id)
        if existing is not None:
            return f"Agent '{agent_id}' already exists. Use update_agent to modify it."
        from app.domain.models.agent_definition import AgentDefinition
        new_agent = AgentDefinition(
            id=agent_id,
            name=name,
            description=description,
            default_runtime=default_runtime,
            agent_input=agent_input,
        )
        await agent_backend.create(new_agent)
        return f"Agent '{agent_id}' created."

    @tool
    async def update_agent(agent_id: str, name: str = None, description: str = None, default_runtime: str = None, agent_input_json: str = None) -> str:
        """Update an existing agent definition. Only provided fields are changed; others preserved."""
        if agent_backend is None:
            return "Agent backend not configured."
        resolved, err = await _resolve_agent_id(agent_id)
        if err:
            return err
        existing = await agent_backend.get(resolved)
        if existing is None:
            return f"Agent '{resolved}' not found."
        import json as _json
        # Partial update — only mutate provided fields
        updated = existing.model_copy()
        if name is not None:
            updated.name = name
        if description is not None:
            updated.description = description
        if default_runtime is not None:
            if default_runtime not in ("local", "docker", "k8s"):
                return f"Invalid default_runtime '{default_runtime}'. Must be one of: local, docker, k8s."
            updated.default_runtime = default_runtime
        if agent_input_json is not None:
            try:
                agent_input = _json.loads(agent_input_json)
                if not isinstance(agent_input, dict):
                    return "agent_input_json must be a JSON object."
                updated.agent_input = agent_input
            except Exception as e:
                return f"Invalid agent_input_json: {e}"
        await agent_backend.update(resolved, updated)
        return f"Agent '{resolved}' updated."

    @tool
    async def delete_agent(agent_id: str) -> str:
        """Delete an agent definition by exact id."""
        if agent_backend is None:
            return "Agent backend not configured."
        existing = await agent_backend.get(agent_id)
        if existing is None:
            return f"Agent '{agent_id}' not found. Use list_agents to see available agents."
        await agent_backend.delete(agent_id)
        return f"Agent '{agent_id}' deleted."

    tools = [list_workflows, run_workflow, list_runs, get_run, ask_user,
             create_workflow, update_workflow, delete_workflow,
             list_agents, get_agent, create_agent, update_agent, delete_agent]
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
