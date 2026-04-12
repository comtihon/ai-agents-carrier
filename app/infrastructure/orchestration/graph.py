from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.application.services.planning_service import PlanningService
from app.domain.interfaces.openhands import OpenHandsPort
from app.domain.interfaces.repositories import WorkflowRunRepository
from app.domain.models.runtime import ActionStepResult, ExecutionStepResult, PlanResult, ToolCallResult, WorkflowRequest, WorkflowRun
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.actions.http_executor import HttpStepExecutor
from app.infrastructure.actions.registry import ActionRegistry
from app.infrastructure.tools.mcp_client import McpToolsProvider


class WorkflowGraphState(TypedDict):
    workflow_run: WorkflowRun
    workflow_definition: WorkflowDefinition
    plan: PlanResult | None
    execution_results: list[ExecutionStepResult]


class WorkflowGraphRunner:
    def __init__(
        self,
        planning_service: PlanningService,
        openhands_port: OpenHandsPort,
        workflow_run_repository: WorkflowRunRepository,
        mcp_tools_provider: McpToolsProvider,
        http_executor: HttpStepExecutor,
        action_registry: ActionRegistry,
    ) -> None:
        self._planning_service = planning_service
        self._openhands_port = openhands_port
        self._workflow_run_repository = workflow_run_repository
        self._mcp_tools_provider = mcp_tools_provider
        self._http_executor = http_executor
        self._action_registry = action_registry
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowGraphState)
        graph.add_node("request", self._request_node)
        graph.add_node("fetch_context", self._fetch_context_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("approval_check", self._approval_check_node)
        graph.add_node("run_actions", self._run_actions_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("result", self._result_node)
        graph.set_entry_point("request")
        graph.add_edge("request", "fetch_context")
        graph.add_edge("fetch_context", "plan")
        graph.add_edge("plan", "approval_check")
        graph.add_conditional_edges(
            "approval_check",
            self._route_after_approval,
            {
                "run_actions": "run_actions",
                "result": "result",
            }
        )
        graph.add_edge("run_actions", "execute")
        graph.add_edge("execute", "result")
        graph.add_edge("result", END)
        return graph.compile()

    async def run(self, workflow_run: WorkflowRun, workflow_definition: WorkflowDefinition) -> WorkflowRun:
        result: dict[str, Any] = await self._graph.ainvoke(
            {
                "workflow_run": workflow_run,
                "workflow_definition": workflow_definition,
                "plan": workflow_run.plan,
                "execution_results": workflow_run.execution_results,
            }
        )
        return result["workflow_run"]

    async def _request_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_run.status = "running"
        workflow_run.current_step = "request"
        await self._workflow_run_repository.update(workflow_run)
        return state

    async def _fetch_context_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]

        fetch_steps = [s for s in workflow_definition.steps if s.type == "fetch"]
        if not fetch_steps:
            return state

        workflow_run.current_step = "fetch_context"
        tool_call_results: list[ToolCallResult] = list(workflow_run.tool_call_results)

        for step in fetch_steps:
            tool_name = step.tool  # always set — validated at definition load time
            tool = self._mcp_tools_provider.get_tool(tool_name)  # type: ignore[arg-type]
            if tool is None:
                workflow_run.status = "failed"
                workflow_run.error = (
                    f"Fetch step '{step.id}' requires MCP tool '{tool_name}', "
                    "but it is not available. Ensure the integration is enabled and configured."
                )
                tool_call_results.append(
                    ToolCallResult(step_id=step.id, tool=tool_name, status="failed", error=workflow_run.error)
                )
                workflow_run.tool_call_results = tool_call_results
                await self._workflow_run_repository.update(workflow_run)
                state["workflow_run"] = workflow_run
                return state

            try:
                raw_output = await tool.ainvoke(step.tool_input)
            except Exception as exc:
                workflow_run.status = "failed"
                workflow_run.error = f"Fetch step '{step.id}' failed calling tool '{tool_name}': {exc}"
                tool_call_results.append(
                    ToolCallResult(step_id=step.id, tool=tool_name, status="failed", error=str(exc))
                )
                workflow_run.tool_call_results = tool_call_results
                await self._workflow_run_repository.update(workflow_run)
                state["workflow_run"] = workflow_run
                return state

            output: dict[str, Any] = raw_output if isinstance(raw_output, dict) else {"result": raw_output}
            output_key = step.output_key or step.id
            workflow_run.intermediate_outputs[output_key] = output
            tool_call_results.append(
                ToolCallResult(step_id=step.id, tool=tool_name, status="success", output=output)
            )

        workflow_run.tool_call_results = tool_call_results
        await self._workflow_run_repository.update(workflow_run)
        state["workflow_run"] = workflow_run
        return state

    async def _plan_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]

        if workflow_run.status == "failed":
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

    async def _approval_check_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]

        if workflow_run.status == "failed":
            return state

        # Check if workflow definition has an approval step
        has_approval_step = any(step.type == "approval" for step in workflow_definition.steps)
        
        if not has_approval_step:
            # No approval required, continue to next phase
            return state

        workflow_run.current_step = "approval_check"
        
        # If approval is already approved, continue
        if workflow_run.approval_status == "approved":
            await self._workflow_run_repository.update(workflow_run)
            return state
        
        # If approval is rejected, fail the workflow
        if workflow_run.approval_status == "rejected":
            workflow_run.status = "failed"
            workflow_run.error = "Workflow execution rejected by user during approval step."
            await self._workflow_run_repository.update(workflow_run)
            return state
        
        # Otherwise, require approval - pause execution
        workflow_run.status = "waiting_approval"
        workflow_run.approval_status = "pending"
        workflow_run.intermediate_outputs["approval_required"] = True
        workflow_run.intermediate_outputs["approval_message"] = (
            f"Please review the plan and approve to continue execution:\n{workflow_run.plan.summary if workflow_run.plan else 'No plan available'}"
        )
        await self._workflow_run_repository.update(workflow_run)
        state["workflow_run"] = workflow_run
        
        # Raise interrupt to pause the graph execution
        raise ValueError(
            "Workflow execution paused for approval. "
            f"Use POST /workflows/runs/{workflow_run.id}/approve or /reject to continue."
        )

    def _route_after_approval(self, state: WorkflowGraphState) -> str:
        workflow_run = state["workflow_run"]
        
        # If rejected or failed, go to result
        if workflow_run.approval_status == "rejected" or workflow_run.status == "failed":
            return "result"
        
        # Otherwise continue to run_actions
        return "run_actions"

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
