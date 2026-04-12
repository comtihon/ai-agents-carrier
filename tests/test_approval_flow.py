from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.orchestration_service import OrchestrationService
from app.domain.exceptions import NotFoundError
from app.domain.models.runtime import PlanResult, RepositoryTask, WorkflowRun
from app.domain.models.workflow_definition import WorkflowDefinition, WorkflowStepDefinition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workflow_def(with_approval: bool = True) -> WorkflowDefinition:
    steps: list[WorkflowStepDefinition] = [
        WorkflowStepDefinition(id="plan", name="Plan", type="plan"),
    ]
    if with_approval:
        steps.append(WorkflowStepDefinition(id="review", name="Review Plan", type="approval"))
    steps.append(WorkflowStepDefinition(id="result", name="Result", type="result"))
    return WorkflowDefinition(
        id="test_wf",
        name="Test Workflow",
        description="A test workflow",
        entrypoint="plan",
        steps=steps,
    )


def _make_run(**kwargs: Any) -> WorkflowRun:
    return WorkflowRun(
        workflow_id=kwargs.get("workflow_id", "test_wf"),
        workflow_name=kwargs.get("workflow_name", "Test Workflow"),
        user_request=kwargs.get("user_request", "Do something"),
        status=kwargs.get("status", "pending"),
        approval_status=kwargs.get("approval_status", "not_required"),
    )


def _make_plan() -> PlanResult:
    return PlanResult(
        summary="Build the feature",
        tasks=[RepositoryTask(repo="org/app", instructions="Do it", order=1)],
        execution_order=["org/app"],
    )


def _make_orchestration_service(
    run: WorkflowRun | None = None,
    workflow_def: WorkflowDefinition | None = None,
) -> tuple[OrchestrationService, MagicMock, MagicMock]:
    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=run)
    repo.update = AsyncMock()

    registry = MagicMock()
    registry.get_definition = MagicMock(return_value=workflow_def or _make_workflow_def())

    graph_runner = MagicMock()
    graph_runner.run = AsyncMock(return_value=run)

    service = OrchestrationService(
        workflow_registry=registry,
        workflow_run_repository=repo,
        graph_runner=graph_runner,
    )
    return service, repo, graph_runner


# ---------------------------------------------------------------------------
# _approval_node tests (via graph state)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approval_node_pauses_when_pending() -> None:
    """First time the approval node is hit, it should pause the run."""
    from app.infrastructure.orchestration.graph import WorkflowGraphRunner

    runner = _make_graph_runner()
    workflow_def = _make_workflow_def(with_approval=True)
    run = _make_run(status="running", approval_status="not_required")

    state = {
        "workflow_run": run,
        "workflow_definition": workflow_def,
        "plan": None,
        "execution_results": [],
    }
    result_state = await runner._approval_node(state)

    assert result_state["workflow_run"].status == "waiting_approval"
    assert result_state["workflow_run"].approval_status == "pending"
    assert result_state["workflow_run"].current_step == "approval"
    runner._workflow_run_repository.update.assert_called_once()


@pytest.mark.asyncio
async def test_approval_node_passes_through_when_approved() -> None:
    """If approval_status is already 'approved', the node is a no-op."""
    runner = _make_graph_runner()
    workflow_def = _make_workflow_def(with_approval=True)
    run = _make_run(status="running", approval_status="approved")

    state = {
        "workflow_run": run,
        "workflow_definition": workflow_def,
        "plan": None,
        "execution_results": [],
    }
    result_state = await runner._approval_node(state)

    assert result_state["workflow_run"].status == "running"
    runner._workflow_run_repository.update.assert_not_called()


@pytest.mark.asyncio
async def test_approval_node_fails_when_rejected() -> None:
    """If approval_status is 'rejected', the node marks the run as failed."""
    runner = _make_graph_runner()
    workflow_def = _make_workflow_def(with_approval=True)
    run = _make_run(status="running", approval_status="rejected")

    state = {
        "workflow_run": run,
        "workflow_definition": workflow_def,
        "plan": None,
        "execution_results": [],
    }
    result_state = await runner._approval_node(state)

    assert result_state["workflow_run"].status == "failed"
    assert result_state["workflow_run"].error is not None
    runner._workflow_run_repository.update.assert_called_once()


@pytest.mark.asyncio
async def test_approval_node_skipped_when_no_approval_steps() -> None:
    """Workflows without approval steps pass through the node untouched."""
    runner = _make_graph_runner()
    workflow_def = _make_workflow_def(with_approval=False)
    run = _make_run(status="running", approval_status="not_required")

    state = {
        "workflow_run": run,
        "workflow_definition": workflow_def,
        "plan": None,
        "execution_results": [],
    }
    result_state = await runner._approval_node(state)

    assert result_state["workflow_run"].status == "running"
    runner._workflow_run_repository.update.assert_not_called()


@pytest.mark.asyncio
async def test_approval_node_stores_step_details_in_outputs() -> None:
    """When pausing, approval step metadata should be stored in intermediate_outputs."""
    runner = _make_graph_runner()
    workflow_def = _make_workflow_def(with_approval=True)
    run = _make_run(status="running", approval_status="not_required")

    state = {
        "workflow_run": run,
        "workflow_definition": workflow_def,
        "plan": None,
        "execution_results": [],
    }
    result_state = await runner._approval_node(state)

    assert "approval_steps" in result_state["workflow_run"].intermediate_outputs
    steps = result_state["workflow_run"].intermediate_outputs["approval_steps"]
    assert len(steps) == 1
    assert steps[0]["id"] == "review"


# ---------------------------------------------------------------------------
# _route_after_approval tests
# ---------------------------------------------------------------------------

def test_route_after_approval_routes_to_run_actions_when_approved() -> None:
    runner = _make_graph_runner()
    run = _make_run(status="running")
    state = {"workflow_run": run, "workflow_definition": _make_workflow_def(), "plan": None, "execution_results": []}
    assert runner._route_after_approval(state) == "run_actions"


def test_route_after_approval_routes_to_end_when_waiting() -> None:
    from langgraph.graph import END
    runner = _make_graph_runner()
    run = _make_run(status="waiting_approval")
    state = {"workflow_run": run, "workflow_definition": _make_workflow_def(), "plan": None, "execution_results": []}
    assert runner._route_after_approval(state) == END


def test_route_after_approval_routes_to_end_when_failed() -> None:
    from langgraph.graph import END
    runner = _make_graph_runner()
    run = _make_run(status="failed")
    state = {"workflow_run": run, "workflow_definition": _make_workflow_def(), "plan": None, "execution_results": []}
    assert runner._route_after_approval(state) == END


# ---------------------------------------------------------------------------
# Plan node skip logic tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plan_node_skips_if_plan_already_set() -> None:
    """On resume, if the plan was already created, don't re-plan."""
    runner = _make_graph_runner()
    existing_plan = _make_plan()
    run = _make_run(status="running")
    run.plan = existing_plan

    state = {
        "workflow_run": run,
        "workflow_definition": _make_workflow_def(),
        "plan": None,
        "execution_results": [],
    }
    result_state = await runner._plan_node(state)

    assert result_state["plan"] == existing_plan
    runner._planning_service.create_plan.assert_not_called()


# ---------------------------------------------------------------------------
# OrchestrationService.approve / reject tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_sets_approval_status_and_reruns_graph() -> None:
    run = _make_run(status="waiting_approval", approval_status="pending")
    service, repo, graph_runner = _make_orchestration_service(run=run)

    approved_run = _make_run(status="completed", approval_status="approved")
    graph_runner.run = AsyncMock(return_value=approved_run)

    result = await service.approve(run.id)

    assert run.approval_status == "approved"
    repo.update.assert_called()
    graph_runner.run.assert_called_once()
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_approve_stores_feedback_in_metadata() -> None:
    run = _make_run(status="waiting_approval", approval_status="pending")
    service, repo, graph_runner = _make_orchestration_service(run=run)
    graph_runner.run = AsyncMock(return_value=run)

    await service.approve(run.id, feedback="Looks good to me")

    assert run.metadata.get("approval_feedback") == "Looks good to me"


@pytest.mark.asyncio
async def test_approve_raises_conflict_when_not_waiting() -> None:
    run = _make_run(status="completed", approval_status="approved")
    service, _, _ = _make_orchestration_service(run=run)

    with pytest.raises(ValueError, match="not awaiting approval"):
        await service.approve(run.id)


@pytest.mark.asyncio
async def test_reject_marks_run_failed_and_does_not_rerun_graph() -> None:
    run = _make_run(status="waiting_approval", approval_status="pending")
    service, repo, graph_runner = _make_orchestration_service(run=run)

    result = await service.reject(run.id, reason="Not what we need")

    assert result.status == "failed"
    assert result.approval_status == "rejected"
    assert result.error == "Not what we need"
    assert result.metadata.get("rejection_reason") == "Not what we need"
    graph_runner.run.assert_not_called()


@pytest.mark.asyncio
async def test_reject_uses_default_error_when_no_reason_given() -> None:
    run = _make_run(status="waiting_approval", approval_status="pending")
    service, _, _ = _make_orchestration_service(run=run)

    result = await service.reject(run.id)

    assert result.status == "failed"
    assert result.error == "Workflow run rejected during approval review."


@pytest.mark.asyncio
async def test_reject_raises_conflict_when_not_waiting() -> None:
    run = _make_run(status="running", approval_status="pending")
    service, _, _ = _make_orchestration_service(run=run)

    with pytest.raises(ValueError, match="not awaiting approval"):
        await service.reject(run.id)


@pytest.mark.asyncio
async def test_approve_raises_not_found_when_run_missing() -> None:
    service, repo, _ = _make_orchestration_service(run=None)
    repo.get_by_id = AsyncMock(return_value=None)

    with pytest.raises(NotFoundError):
        await service.approve("nonexistent-id")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _make_graph_runner():
    from app.infrastructure.orchestration.graph import WorkflowGraphRunner

    runner = WorkflowGraphRunner.__new__(WorkflowGraphRunner)
    runner._planning_service = MagicMock()
    runner._planning_service.create_plan = AsyncMock()
    runner._openhands_port = MagicMock()
    runner._workflow_run_repository = MagicMock()
    runner._workflow_run_repository.update = AsyncMock()
    runner._mcp_tools_provider = MagicMock()
    runner._http_executor = MagicMock()
    runner._action_registry = MagicMock()
    return runner
