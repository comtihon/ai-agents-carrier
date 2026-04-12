from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.api.dependencies import get_orchestration_service
from app.application.services.orchestration_service import OrchestrationService
from app.domain.exceptions import NotFoundError
from app.domain.models.runtime import WorkflowRequest, WorkflowRunResponse
from pydantic import BaseModel

router = APIRouter(prefix="/workflows", tags=["workflows"])


class ApprovalRequest(BaseModel):
    reason: str | None = None


@router.post("/runs", response_model=WorkflowRunResponse, status_code=status.HTTP_201_CREATED)
async def submit_workflow(
    request: WorkflowRequest,
    orchestration_service: OrchestrationService = Depends(get_orchestration_service),
) -> WorkflowRunResponse:
    try:
        return WorkflowRunResponse(run=await orchestration_service.submit(request))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/runs/{run_id}", response_model=WorkflowRunResponse)
async def get_workflow_run(
    run_id: str,
    orchestration_service: OrchestrationService = Depends(get_orchestration_service),
) -> WorkflowRunResponse:
    try:
        return WorkflowRunResponse(run=await orchestration_service.get_run(run_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/runs/{run_id}/resume", response_model=WorkflowRunResponse)
async def resume_workflow_run(
    run_id: str,
    orchestration_service: OrchestrationService = Depends(get_orchestration_service),
) -> WorkflowRunResponse:
    try:
        return WorkflowRunResponse(run=await orchestration_service.resume(run_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/runs/{run_id}/approve", response_model=WorkflowRunResponse)
async def approve_workflow_run(
    run_id: str,
    orchestration_service: OrchestrationService = Depends(get_orchestration_service),
) -> WorkflowRunResponse:
    try:
        return WorkflowRunResponse(run=await orchestration_service.approve(run_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/runs/{run_id}/reject", response_model=WorkflowRunResponse)
async def reject_workflow_run(
    run_id: str,
    request: ApprovalRequest = Body(default=ApprovalRequest()),
    orchestration_service: OrchestrationService = Depends(get_orchestration_service),
) -> WorkflowRunResponse:
    try:
        return WorkflowRunResponse(run=await orchestration_service.reject(run_id, request.reason))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
