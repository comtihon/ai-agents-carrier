from __future__ import annotations

from fastapi import HTTPException, status

from app.domain.interfaces.repositories import WorkflowRunRepository
from app.domain.interfaces.workflow_registry import WorkflowDefinitionRegistry
from app.domain.models.runtime import WorkflowRequest, WorkflowRun
from app.infrastructure.orchestration.graph import WorkflowGraphRunner


class OrchestrationService:
    def __init__(
        self,
        workflow_registry: WorkflowDefinitionRegistry,
        workflow_run_repository: WorkflowRunRepository,
        graph_runner: WorkflowGraphRunner,
    ) -> None:
        self._workflow_registry = workflow_registry
        self._workflow_run_repository = workflow_run_repository
        self._graph_runner = graph_runner

    async def submit(self, request: WorkflowRequest) -> WorkflowRun:
        try:
            workflow = self._workflow_registry.get_definition(request.workflow_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        workflow_run = WorkflowRun(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            user_request=request.user_request,
            session_id=request.session_id,
            user_id=request.user_id,
            metadata={"request_context": request.context},
        )
        await self._workflow_run_repository.create(workflow_run)
        return await self._graph_runner.run(workflow_run=workflow_run, workflow_definition=workflow)

    async def get_run(self, run_id: str) -> WorkflowRun:
        workflow_run = await self._workflow_run_repository.get_by_id(run_id)
        if workflow_run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow run not found.")
        return workflow_run

    async def resume(self, run_id: str) -> WorkflowRun:
        workflow_run = await self.get_run(run_id)
        try:
            workflow_definition = self._workflow_registry.get_definition(workflow_run.workflow_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return await self._graph_runner.run(workflow_run=workflow_run, workflow_definition=workflow_definition)
