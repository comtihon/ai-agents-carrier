from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_orchestration_service
from app.application.services.orchestration_service import OrchestrationService
from app.domain.models.runtime import WorkflowRequest, WorkflowRunResponse

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("/runs", response_model=WorkflowRunResponse, status_code=status.HTTP_201_CREATED)
async def submit_workflow(
    request: WorkflowRequest,
    orchestration_service: OrchestrationService = Depends(get_orchestration_service),
) -> WorkflowRunResponse:
    return WorkflowRunResponse(run=await orchestration_service.submit(request))


@router.get("/runs/{run_id}", response_model=WorkflowRunResponse)
async def get_workflow_run(
    run_id: str,
    orchestration_service: OrchestrationService = Depends(get_orchestration_service),
) -> WorkflowRunResponse:
    return WorkflowRunResponse(run=await orchestration_service.get_run(run_id))


@router.post("/runs/{run_id}/resume", response_model=WorkflowRunResponse)
async def resume_workflow_run(
    run_id: str,
    orchestration_service: OrchestrationService = Depends(get_orchestration_service),
) -> WorkflowRunResponse:
    return WorkflowRunResponse(run=await orchestration_service.resume(run_id))
