"""
Default CopilotKit agent — the entry point for all chat interactions.

On each user message the agent runs a structured LLM call that decides whether
to reply directly or spawn one of the registered YAML workflows.  All available
workflows are injected into the system prompt so the model can make an informed
routing decision.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from copilotkit import CopilotKitState
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RunnableConfig
from pydantic import BaseModel, Field

from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.yaml_graph import stream_graph_to_pause

if TYPE_CHECKING:
    from app.infrastructure.config.graph_loader import YamlGraphRegistry
    from app.infrastructure.persistence.mongo import MongoGraphRunRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema for the routing decision
# ---------------------------------------------------------------------------

class RouterDecision(BaseModel):
    action: Literal["reply", "run_workflow"] = Field(
        description=(
            "Choose 'reply' to answer the user directly. "
            "Choose 'run_workflow' to start a workflow that handles the request."
        )
    )
    reply_text: str | None = Field(
        None,
        description="The response to send to the user (required when action='reply').",
    )
    workflow_id: str | None = Field(
        None,
        description="ID of the workflow to start (required when action='run_workflow').",
    )
    workflow_request: str | None = Field(
        None,
        description=(
            "Detailed task description to pass to the workflow "
            "(required when action='run_workflow')."
        ),
    )


# ---------------------------------------------------------------------------
# Extended state (adds routing decision on top of CopilotKit messages)
# ---------------------------------------------------------------------------

from typing import TypedDict  # noqa: E402


class DefaultWorkflowState(CopilotKitState, total=False):  # type: ignore[misc]
    decision: Any  # RouterDecision, typed loosely to satisfy TypedDict constraints


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_default_workflow(
    llm: BaseChatModel,
    registry: YamlGraphRegistry,
    run_repository: MongoGraphRunRepository,
):
    """Build and compile the default CopilotKit LangGraph agent."""

    # Build the workflows section of the system prompt once at startup.
    definitions = registry.list_definitions()
    if definitions:
        workflow_lines = "\n".join(
            f"- **{d['id']}**: {(d.get('description') or '').strip()}"
            for d in definitions
        )
        workflows_section = f"Available workflows:\n{workflow_lines}"
    else:
        workflows_section = "No workflows are currently configured."

    SYSTEM_PROMPT = f"""\
You are an intelligent assistant for a software-engineering workflow automation platform.

{workflows_section}

For each user message, decide ONE of:
1. **reply** — answer the question, explain a workflow, or have a general conversation.
2. **run_workflow** — the user wants to start an automated task; pick the best matching \
workflow and craft a detailed `workflow_request`.

Rules:
- Use `run_workflow` only when the user clearly wants work to be done (e.g. "implement \
ticket X", "develop feature Y").
- For questions, greetings, status checks, or anything that doesn't require running a \
workflow, use `reply`.
- `workflow_request` must be self-contained — include all details the workflow needs.
"""

    # ── nodes ────────────────────────────────────────────────────────────────

    async def decide(state: DefaultWorkflowState, config: RunnableConfig) -> dict:
        """Run a structured LLM call to decide how to handle the user's message."""
        messages = list(state.get("messages", []))
        structured_llm = llm.with_structured_output(RouterDecision)
        decision: RouterDecision = await structured_llm.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + messages
        )
        logger.info(
            "default_workflow: decision action=%s workflow_id=%s",
            decision.action, decision.workflow_id,
        )
        return {"decision": decision}

    async def reply(state: DefaultWorkflowState, config: RunnableConfig) -> dict:
        """Return the LLM's direct reply to the user."""
        decision: RouterDecision = state["decision"]
        text = decision.reply_text or ""
        if not text:
            # Fallback: ask the LLM again without structured output so it can
            # produce a natural conversational response.
            ck_config = copilotkit_customize_config(config, emit_messages=True)
            response = await llm.ainvoke(
                [SystemMessage(content=SYSTEM_PROMPT)] + list(state.get("messages", [])),
                config=ck_config,
            )
            return {"messages": [response]}
        return {"messages": [AIMessage(content=text)]}

    async def spawn_workflow(state: DefaultWorkflowState, config: RunnableConfig) -> dict:
        """Spawn the chosen workflow as an independent background run."""
        decision: RouterDecision = state["decision"]
        workflow_id = decision.workflow_id or ""
        workflow_request = decision.workflow_request or ""

        runner = registry.get(workflow_id)
        if runner is None:
            logger.error("default_workflow: workflow '%s' not found in registry", workflow_id)
            return {
                "messages": [AIMessage(
                    content=f"Sorry, I couldn't find the workflow **{workflow_id}**. "
                            f"Available workflows: {', '.join(registry.list_ids()) or 'none'}."
                )]
            }

        child_run_id = str(uuid4())
        child_run = GraphRun(
            id=child_run_id,
            graph_id=workflow_id,
            user_request=workflow_request,
            status="running",
            step_statuses={s["id"]: "pending" for s in runner.steps},
        )
        await run_repository.create(child_run)

        asyncio.create_task(
            stream_graph_to_pause(runner, child_run, run_repository, {"request": workflow_request})
        )

        logger.info(
            "default_workflow: spawned '%s' as run %s", workflow_id, child_run_id
        )
        return {
            "messages": [AIMessage(
                content=(
                    f"I've started the **{runner.name}** workflow for you.\n\n"
                    f"Run ID: `{child_run_id}`\n\n"
                    f"You can track its progress in the workflow panel."
                )
            )]
        }

    # ── routing ──────────────────────────────────────────────────────────────

    def route(state: DefaultWorkflowState) -> str:
        decision: RouterDecision | None = state.get("decision")  # type: ignore[assignment]
        if (
            decision is not None
            and getattr(decision, "action", None) == "run_workflow"
            and getattr(decision, "workflow_id", None)
        ):
            return "spawn_workflow"
        return "reply"

    # ── graph ────────────────────────────────────────────────────────────────

    sg: StateGraph = StateGraph(DefaultWorkflowState)
    sg.add_node("decide", decide)
    sg.add_node("reply", reply)
    sg.add_node("spawn_workflow", spawn_workflow)

    sg.add_edge(START, "decide")
    sg.add_conditional_edges("decide", route, {"reply": "reply", "spawn_workflow": "spawn_workflow"})
    sg.add_edge("reply", END)
    sg.add_edge("spawn_workflow", END)

    return sg.compile()
