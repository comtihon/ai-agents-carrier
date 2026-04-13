from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.application.services.classifier_service import ClassifierService
from app.application.services.llm_agent_service import LlmAgentService
from app.application.services.planning_service import PlanningService
from app.domain.interfaces.openhands import OpenHandsPort
from app.domain.interfaces.repositories import WorkflowRunRepository
from app.domain.models.runtime import ActionStepResult, ApprovalGate, ExecutionStepResult, LlmAgentStepResult, LlmStepResult, PlanResult, ToolCallResult, WorkflowRequest, WorkflowRun
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.actions.http_executor import HttpStepExecutor
from app.infrastructure.actions.registry import ActionRegistry
from app.infrastructure.tools.mcp_client import McpToolsProvider


class WorkflowGraphState(TypedDict):
    workflow_run: WorkflowRun
    workflow_definition: WorkflowDefinition
    plan: PlanResult | None
    execution_results: list[ExecutionStepResult]
    selected_fetcher_ids: list[str] | None


class WorkflowGraphRunner:
    def __init__(
        self,
        planning_service: PlanningService,
        openhands_port: OpenHandsPort,
        workflow_run_repository: WorkflowRunRepository,
        mcp_tools_provider: McpToolsProvider,
        http_executor: HttpStepExecutor,
        action_registry: ActionRegistry,
        llm_agent_service: LlmAgentService,
        classifier_service: ClassifierService,
    ) -> None:
        self._planning_service = planning_service
        self._openhands_port = openhands_port
        self._workflow_run_repository = workflow_run_repository
        self._mcp_tools_provider = mcp_tools_provider
        self._http_executor = http_executor
        self._action_registry = action_registry
        self._llm_agent_service = llm_agent_service
        self._classifier_service = classifier_service
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowGraphState)
        graph.add_node("request", self._request_node)
        graph.add_node("classifier", self._classifier_node)
        graph.add_node("fetch_context", self._fetch_context_node)
        graph.add_node("llm_agent", self._llm_agent_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("approval", self._approval_node)
        graph.add_node("run_actions", self._run_actions_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("result", self._result_node)
        graph.set_entry_point("request")
        graph.add_edge("request", "classifier")
        graph.add_edge("classifier", "fetch_context")
        graph.add_edge("fetch_context", "llm_agent")
        graph.add_edge("llm_agent", "plan")
        graph.add_edge("plan", "approval")
        graph.add_conditional_edges("approval", self._route_after_approval)
        graph.add_edge("run_actions", "execute")
        graph.add_edge("execute", "result")
        graph.add_edge("result", END)
        return graph.compile()

    def _route_after_approval(self, state: WorkflowGraphState) -> str:
        workflow_run = state["workflow_run"]
        if workflow_run.status in ("waiting_approval", "failed"):
            return END
        return "run_actions"

    async def run(self, workflow_run: WorkflowRun, workflow_definition: WorkflowDefinition) -> WorkflowRun:
        result: dict[str, Any] = await self._graph.ainvoke(
            {
                "workflow_run": workflow_run,
                "workflow_definition": workflow_definition,
                "plan": workflow_run.plan,
                "execution_results": workflow_run.execution_results,
                "selected_fetcher_ids": None,
            }
        )
        return result["workflow_run"]

    async def _request_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_run.status = "running"
        workflow_run.current_step = "request"
        await self._workflow_run_repository.update(workflow_run)
        return state

    async def _classifier_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]

        if workflow_run.status == "failed":
            return state

        if not workflow_definition.metadata.get("use_classifier", False):
            return state

        fetch_steps = [s for s in workflow_definition.steps if s.type == "fetch"]
        if not fetch_steps:
            return state

        workflow_run.current_step = "classifier"

        fetch_steps_info = [
            {
                "id": s.id,
                "tool": s.tool,
                "description": self._mcp_tools_provider.get_tool(s.tool).description  # type: ignore[union-attr]
                if self._mcp_tools_provider.get_tool(s.tool) is not None
                else s.tool,
            }
            for s in fetch_steps
        ]

        prompt_override: str | None = workflow_definition.metadata.get("classifier_prompt")
        selected_ids = await self._classifier_service.classify(
            user_request=workflow_run.user_request,
            fetch_steps=fetch_steps_info,
            prompt_override=prompt_override,
        )

        state["selected_fetcher_ids"] = selected_ids
        return state

    async def _fetch_context_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]

        selected_ids = state.get("selected_fetcher_ids")
        fetch_steps = [
            s for s in workflow_definition.steps
            if s.type == "fetch" and (selected_ids is None or s.id in selected_ids)
        ]
        if not fetch_steps:
            return state

        # Skip if already completed — handles resume after approval pause
        if workflow_run.tool_call_results:
            completed_ids = {r.step_id for r in workflow_run.tool_call_results}
            if {s.id for s in fetch_steps} <= completed_ids:
                return state

        workflow_run.current_step = "fetch_context"

        # Run all fetch steps in parallel
        raw_results = await asyncio.gather(
            *[self._execute_fetch_step(step) for step in fetch_steps],
            return_exceptions=True,
        )

        tool_call_results: list[ToolCallResult] = list(workflow_run.tool_call_results)
        for step, item in zip(fetch_steps, raw_results):
            if isinstance(item, Exception):
                workflow_run.status = "failed"
                workflow_run.error = f"Fetch step '{step.id}' failed: {item}"
                tool_call_results.append(
                    ToolCallResult(step_id=step.id, tool=step.tool or "", status="failed", error=str(item))
                )
                workflow_run.tool_call_results = tool_call_results
                await self._workflow_run_repository.update(workflow_run)
                state["workflow_run"] = workflow_run
                return state

            output: dict[str, Any] = item  # type: ignore[assignment]
            output_key = step.output_key or step.id
            workflow_run.intermediate_outputs[output_key] = output
            tool_call_results.append(
                ToolCallResult(step_id=step.id, tool=step.tool or "", status="success", output=output)
            )

        workflow_run.tool_call_results = tool_call_results
        await self._workflow_run_repository.update(workflow_run)
        state["workflow_run"] = workflow_run
        return state

    async def _execute_fetch_step(self, step: Any) -> dict[str, Any]:
        """Run a single fetch step; raises on tool-not-found or invocation error."""
        tool_name: str = step.tool  # validated non-None at definition load
        tool = self._mcp_tools_provider.get_tool(tool_name)
        if tool is None:
            raise RuntimeError(
                f"Fetch step '{step.id}' requires MCP tool '{tool_name}', "
                "but it is not available. Ensure the integration is enabled and configured."
            )
        raw = await tool.ainvoke(step.tool_input)
        return raw if isinstance(raw, dict) else {"result": raw}

    async def _llm_agent_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]

        if workflow_run.status == "failed":
            return state

        llm_steps = [s for s in workflow_definition.steps if s.type == "llm"]
        if not llm_steps:
            return state

        # Skip if already completed — handles resume after approval pause
        if workflow_run.llm_agent_results:
            completed_ids = {r.step_id for r in workflow_run.llm_agent_results if r.status == "success"}
            if {s.id for s in llm_steps} <= completed_ids:
                return state

        workflow_run.current_step = "llm_agent"

        # Run all LLM steps in parallel
        raw_results = await asyncio.gather(
            *[self._execute_llm_step(step, workflow_run.user_request) for step in llm_steps],
            return_exceptions=True,
        )

        llm_results: list[LlmAgentStepResult] = list(workflow_run.llm_agent_results)
        for step, item in zip(llm_steps, raw_results):
            if isinstance(item, Exception):
                workflow_run.status = "failed"
                workflow_run.error = f"LLM agent step '{step.id}' failed: {item}"
                llm_results.append(LlmAgentStepResult(step_id=step.id, status="failed", error=str(item)))
                workflow_run.llm_agent_results = llm_results
                await self._workflow_run_repository.update(workflow_run)
                state["workflow_run"] = workflow_run
                return state

            result: LlmStepResult = item  # type: ignore[assignment]
            output_key = step.output_key or step.id
            workflow_run.intermediate_outputs[output_key] = result.model_dump(mode="python")
            llm_results.append(
                LlmAgentStepResult(
                    step_id=step.id,
                    status="success",
                    response=result.response,
                    tool_calls_made=result.tool_calls_made,
                )
            )

        workflow_run.llm_agent_results = llm_results
        await self._workflow_run_repository.update(workflow_run)
        state["workflow_run"] = workflow_run
        return state

    async def _execute_llm_step(self, step: Any, user_request: str) -> LlmStepResult:
        """Run a single LLM step; raises on failure."""
        return await self._llm_agent_service.run(user_request)

    async def _plan_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]

        if workflow_run.status == "failed":
            return state

        # Skip if plan already exists — handles resume after approval pause
        if workflow_run.plan is not None:
            state["plan"] = workflow_run.plan
            return state

        workflow_run.current_step = "plan"
        plan = await self._planning_service.create_plan(
            WorkflowRequest(
                workflow_id=workflow_run.workflow_id,
                user_request=workflow_run.user_request,
                session_id=workflow_run.session_id,
                user_id=workflow_run.user_id,
                context=workflow_run.metadata.get("request_context", {}),
            ),
            workflow_definition,
        )
        workflow_run.plan = plan
        workflow_run.intermediate_outputs["plan_summary"] = plan.summary
        await self._workflow_run_repository.update(workflow_run)
        state["plan"] = plan
        state["workflow_run"] = workflow_run
        return state

    async def _approval_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]

        if workflow_run.status == "failed":
            return state

        # No approval steps defined in this workflow — pass through
        approval_steps = [s for s in workflow_definition.steps if s.type == "approval"]
        if not approval_steps:
            return state

        # Already approved — pass through (idempotent resume)
        if workflow_run.approval_status == "approved":
            return state

        # Rejected — mark as failed
        if workflow_run.approval_status == "rejected":
            workflow_run.status = "failed"
            workflow_run.error = (
                workflow_run.metadata.get("rejection_reason")
                or "Workflow run rejected during approval review."
            )
            await self._workflow_run_repository.update(workflow_run)
            state["workflow_run"] = workflow_run
            return state

        # First encounter — initialize approval gates if not already done
        if not workflow_run.approval_gates:
            workflow_run.approval_gates = [
                ApprovalGate(gate_id=s.id, name=s.name, status="pending")
                for s in approval_steps
            ]

        workflow_run.status = "waiting_approval"
        workflow_run.approval_status = "pending"
        workflow_run.current_step = "approval"
        workflow_run.intermediate_outputs["approval_steps"] = [
            {"id": s.id, "name": s.name, "metadata": s.metadata} for s in approval_steps
        ]
        await self._workflow_run_repository.update(workflow_run)
        state["workflow_run"] = workflow_run
        return state

    async def _run_actions_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]

        if workflow_run.status == "failed":
            return state

        action_steps = [s for s in workflow_definition.steps if s.type in ("http", "action")]
        if not action_steps:
            return state

        workflow_run.current_step = "run_actions"
        action_results: list[ActionStepResult] = list(workflow_run.action_results)

        for step in action_steps:
            try:
                if step.type == "http":
                    output = await self._http_executor.execute(step, workflow_run)
                else:
                    output = await self._action_registry.execute(
                        step.handler,  # type: ignore[arg-type]  -- validated non-None for action steps
                        step.handler_input,
                        workflow_run,
                    )
            except Exception as exc:
                workflow_run.status = "failed"
                workflow_run.error = f"Action step '{step.id}' failed: {exc}"
                action_results.append(
                    ActionStepResult(step_id=step.id, type=step.type, status="failed", error=str(exc))
                )
                workflow_run.action_results = action_results
                await self._workflow_run_repository.update(workflow_run)
                state["workflow_run"] = workflow_run
                return state

            output_key = step.output_key or step.id
            workflow_run.intermediate_outputs[output_key] = output
            action_results.append(
                ActionStepResult(step_id=step.id, type=step.type, status="success", output=output)
            )

        workflow_run.action_results = action_results
        await self._workflow_run_repository.update(workflow_run)
        state["workflow_run"] = workflow_run
        return state

    async def _execute_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        plan = state["plan"]

        if workflow_run.status == "failed":
            return state

        if plan is None:
            raise ValueError("Execution requires a plan.")

        workflow_run.current_step = "execute"
        execution_results: list[ExecutionStepResult] = []
        for task in plan.tasks:
            result = await self._openhands_port.execute_task(workflow_run, task)
            execution_step = ExecutionStepResult(
                step_id=task.step_id or task.repo,
                repo=task.repo,
                status=result.status,
                openhands_result=result,
            )
            execution_results.append(execution_step)
            workflow_run.execution_results = execution_results
            workflow_run.intermediate_outputs[task.repo] = result.model_dump(mode="python")
            if result.status == "failed":
                workflow_run.status = "failed"
                workflow_run.error = f"Execution failed for repository '{task.repo}'."
                await self._workflow_run_repository.update(workflow_run)
                state["execution_results"] = execution_results
                state["workflow_run"] = workflow_run
                return state

        await self._workflow_run_repository.update(workflow_run)
        state["execution_results"] = execution_results
        state["workflow_run"] = workflow_run
        return state

    async def _result_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        if workflow_run.status != "failed":
            workflow_run.status = "completed"
        workflow_run.current_step = "result"
        await self._workflow_run_repository.update(workflow_run)
        state["workflow_run"] = workflow_run
        return state
