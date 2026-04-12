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

    async def approve(self, run_id: str, feedback: str | None = None) -> WorkflowRun:
        """Approve all pending gates at once and resume the run.

        For single-gate workflows this is identical to the previous behaviour.
        For multi-gate workflows it approves every pending gate in one call —
        use approve_gate() to approve gates individually.
        """
        workflow_run = await self.get_run(run_id)
        if workflow_run.status != "waiting_approval":
            raise ValueError(
                f"Workflow run '{run_id}' is not awaiting approval (current status: '{workflow_run.status}')."
            )
        try:
            workflow_definition = self._workflow_registry.get_definition(workflow_run.workflow_id)
        except KeyError as exc:
            raise NotFoundError(str(exc)) from exc

        # Approve all pending gates
        for gate in workflow_run.approval_gates:
            if gate.status == "pending":
                gate.status = "approved"
                if feedback:
                    gate.feedback = feedback

        workflow_run.approval_status = "approved"
        if feedback:
            workflow_run.metadata["approval_feedback"] = feedback
        await self._workflow_run_repository.update(workflow_run)
        try:
            return await self._graph_runner.run(workflow_run=workflow_run, workflow_definition=workflow_definition)
        except Exception as exc:
            workflow_run.status = "failed"
            workflow_run.error = str(exc)
            await self._workflow_run_repository.update(workflow_run)
            raise

    async def approve_gate(self, run_id: str, gate_id: str, feedback: str | None = None) -> WorkflowRun:
        """Approve a single named approval gate.

        Resumes graph execution only once every gate in the run is approved.
        """
        workflow_run = await self.get_run(run_id)
        if workflow_run.status != "waiting_approval":
            raise ValueError(
                f"Workflow run '{run_id}' is not awaiting approval (current status: '{workflow_run.status}')."
            )

        gate = next((g for g in workflow_run.approval_gates if g.gate_id == gate_id), None)
        if gate is None:
            raise NotFoundError(f"Approval gate '{gate_id}' not found in run '{run_id}'.")
        if gate.status != "pending":
            raise ValueError(f"Gate '{gate_id}' is already '{gate.status}'.")

        gate.status = "approved"
        if feedback:
            gate.feedback = feedback

        all_approved = all(g.status == "approved" for g in workflow_run.approval_gates)
        if all_approved:
            workflow_run.approval_status = "approved"
            await self._workflow_run_repository.update(workflow_run)
            try:
                workflow_definition = self._workflow_registry.get_definition(workflow_run.workflow_id)
            except KeyError as exc:
                raise NotFoundError(str(exc)) from exc
            try:
                return await self._graph_runner.run(workflow_run=workflow_run, workflow_definition=workflow_definition)
            except Exception as exc:
                workflow_run.status = "failed"
                workflow_run.error = str(exc)
                await self._workflow_run_repository.update(workflow_run)
                raise

        # Still waiting for other gates — persist and return current state
        await self._workflow_run_repository.update(workflow_run)
        return workflow_run

    async def reject_gate(self, run_id: str, gate_id: str, reason: str | None = None) -> WorkflowRun:
        """Reject a single named approval gate, immediately failing the run."""
        workflow_run = await self.get_run(run_id)
        if workflow_run.status != "waiting_approval":
            raise ValueError(
                f"Workflow run '{run_id}' is not awaiting approval (current status: '{workflow_run.status}')."
            )

        gate = next((g for g in workflow_run.approval_gates if g.gate_id == gate_id), None)
        if gate is None:
            raise NotFoundError(f"Approval gate '{gate_id}' not found in run '{run_id}'.")
        if gate.status != "pending":
            raise ValueError(f"Gate '{gate_id}' is already '{gate.status}'.")

        gate.status = "rejected"
        if reason:
            gate.feedback = reason

        workflow_run.approval_status = "rejected"
        workflow_run.status = "failed"
        workflow_run.error = reason or f"Approval gate '{gate_id}' was rejected."
        if reason:
            workflow_run.metadata["rejection_reason"] = reason
        await self._workflow_run_repository.update(workflow_run)
        return workflow_run

    async def reject(self, run_id: str, reason: str | None = None) -> WorkflowRun:
        workflow_run = await self.get_run(run_id)
        if workflow_run.status != "waiting_approval":
            raise ValueError(
                f"Workflow run '{run_id}' is not awaiting approval (current status: '{workflow_run.status}')."
            )
        workflow_run.approval_status = "rejected"
        workflow_run.status = "failed"
        workflow_run.error = reason or "Workflow run rejected during approval review."
        if reason:
            workflow_run.metadata["rejection_reason"] = reason
        await self._workflow_run_repository.update(workflow_run)
        return workflow_run
