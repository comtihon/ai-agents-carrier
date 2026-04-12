from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.application.services.planning_service import PlanningService
from app.domain.interfaces.openhands import OpenHandsPort
from app.domain.interfaces.repositories import WorkflowRunRepository
from app.domain.models.runtime import ExecutionStepResult, PlanResult, WorkflowRequest, WorkflowRun
from app.domain.models.workflow_definition import WorkflowDefinition


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
    ) -> None:
        self._planning_service = planning_service
        self._openhands_port = openhands_port
        self._workflow_run_repository = workflow_run_repository
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowGraphState)
        graph.add_node("request", self._request_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("result", self._result_node)
        graph.set_entry_point("request")
        graph.add_edge("request", "plan")
        graph.add_edge("plan", "execute")
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

    async def _plan_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        workflow_definition = state["workflow_definition"]
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

    async def _execute_node(self, state: WorkflowGraphState) -> WorkflowGraphState:
        workflow_run = state["workflow_run"]
        plan = state["plan"]
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
