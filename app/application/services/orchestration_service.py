from __future__ import annotations

from app.domain.exceptions import NotFoundError
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
            raise NotFoundError(str(exc)) from exc
        workflow_run = WorkflowRun(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            user_request=request.user_request,
            session_id=request.session_id,
            user_id=request.user_id,
            metadata={"request_context": request.context},
        )
        await self._workflow_run_repository.create(workflow_run)
        try:
            return await self._graph_runner.run(workflow_run=workflow_run, workflow_definition=workflow)
        except Exception as exc:
            workflow_run.status = "failed"
            workflow_run.error = str(exc)
            await self._workflow_run_repository.update(workflow_run)
            raise

    async def get_run(self, run_id: str) -> WorkflowRun:
        workflow_run = await self._workflow_run_repository.get_by_id(run_id)
        if workflow_run is None:
            raise NotFoundError(f"Workflow run '{run_id}' not found.")
        return workflow_run

    async def resume(self, run_id: str) -> WorkflowRun:
        workflow_run = await self.get_run(run_id)
        try:
            workflow_definition = self._workflow_registry.get_definition(workflow_run.workflow_id)
        except KeyError as exc:
            raise NotFoundError(str(exc)) from exc
        return await self._graph_runner.run(workflow_run=workflow_run, workflow_definition=workflow_definition)

    async def approve(self, run_id: str) -> WorkflowRun:
        workflow_run = await self.get_run(run_id)
        if workflow_run.status != "waiting_approval":
            raise ValueError(f"Workflow run '{run_id}' is not in 'waiting_approval' status.")
        workflow_run.approval_status = "approved"
        workflow_run.status = "running"
        workflow_run.metadata["approved_at"] = workflow_run.updated_at.isoformat()
        await self._workflow_run_repository.update(workflow_run)
        try:
            workflow_definition = self._workflow_registry.get_definition(workflow_run.workflow_id)
        except KeyError as exc:
            raise NotFoundError(str(exc)) from exc
        return await self._graph_runner.run(workflow_run=workflow_run, workflow_definition=workflow_definition)

    async def reject(self, run_id: str, reason: str | None = None) -> WorkflowRun:
        workflow_run = await self.get_run(run_id)
        if workflow_run.status != "waiting_approval":
            raise ValueError(f"Workflow run '{run_id}' is not in 'waiting_approval' status.")
        workflow_run.approval_status = "rejected"
        workflow_run.status = "failed"
        workflow_run.error = reason or "Workflow execution rejected by user."
        workflow_run.metadata["rejected_at"] = workflow_run.updated_at.isoformat()
        if reason:
            workflow_run.metadata["rejection_reason"] = reason
        await self._workflow_run_repository.update(workflow_run)
        return workflow_run
