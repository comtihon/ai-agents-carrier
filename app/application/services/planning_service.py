from __future__ import annotations

from app.domain.models.runtime import PlanResult, RepositoryTask, WorkflowRequest
from app.domain.models.workflow_definition import WorkflowDefinition


class PlanningService:
    async def create_plan(self, request: WorkflowRequest, workflow: WorkflowDefinition) -> PlanResult:
        executable_steps = [step for step in workflow.steps if step.type == "execute"]
        tasks = [
            RepositoryTask(
                repo=step.repo or workflow.metadata.get("default_repo", "unknown"),
                instructions=step.instructions or request.user_request,
                order=index,
                depends_on=step.requires,
                step_id=step.id,
            )
            for index, step in enumerate(executable_steps, start=1)
        ]
        if not tasks:
            tasks = [
                RepositoryTask(
                    repo=workflow.metadata.get("default_repo", "default"),
                    instructions=request.user_request,
                    order=1,
                    step_id="default-execution",
                )
            ]
        return PlanResult(
            summary=f"Execution plan for workflow '{workflow.name}' with {len(tasks)} repository task(s).",
            tasks=tasks,
            execution_order=[task.repo for task in tasks],
            outputs_required=list(workflow.metadata.get("outputs_required", [])),
        )
