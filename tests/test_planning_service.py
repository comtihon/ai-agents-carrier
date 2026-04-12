import pytest

from app.application.services.planning_service import PlanningService
from app.domain.models.runtime import WorkflowRequest
from app.infrastructure.config.workflow_loader import WorkflowDefinitionLoader


@pytest.mark.asyncio
async def test_planning_service_creates_repo_tasks() -> None:
    workflow = WorkflowDefinitionLoader("workflows").load().get_definition("multi_repo_delivery")
    plan = await PlanningService().create_plan(
        WorkflowRequest(workflow_id=workflow.id, user_request="Ship coordinated backend and frontend changes."),
        workflow,
    )
    assert len(plan.tasks) == 2
    assert plan.tasks[0].repo == "acme/backend"
    assert plan.tasks[1].depends_on == ["execute_backend"]
