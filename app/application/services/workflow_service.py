from __future__ import annotations

from fastapi import HTTPException, status

from app.domain.interfaces.repositories import WorkflowRunRepository
from app.domain.interfaces.workflow_registry import WorkflowDefinitionRegistry
from app.domain.models.runtime import WorkflowRequest, WorkflowRun


class WorkflowService:
    def __init__(
        self,
        workflow_registry: WorkflowDefinitionRegistry,
        workflow_run_repository: WorkflowRunRepository,
    ) -> None:
        self._workflow_registry = workflow_registry
        self._workflow_run_repository = workflow_run_repository

    async def initialize_run(self, request: WorkflowRequest) -> WorkflowRun:
        try:
            workflow = self._workflow_registry.get_definition(request.workflow_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        run = WorkflowRun(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            user_request=request.user_request,
            session_id=request.session_id,
            user_id=request.user_id,
            metadata={"request_context": request.context},
        )
        return await self._workflow_run_repository.create(run)

    async def get_run(self, run_id: str) -> WorkflowRun | None:
        return await self._workflow_run_repository.get_by_id(run_id)
