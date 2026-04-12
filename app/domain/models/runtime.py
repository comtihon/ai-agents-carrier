from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowRequest(BaseModel):
    workflow_id: str
    user_request: str
    session_id: str | None = None
    user_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class RepositoryTask(BaseModel):
    repo: str
    instructions: str
    order: int
    depends_on: list[str] = Field(default_factory=list)
    step_id: str | None = None


class PlanResult(BaseModel):
    summary: str
    tasks: list[RepositoryTask]
    execution_order: list[str]
    outputs_required: list[str] = Field(default_factory=list)


class OpenHandsExecutionResult(BaseModel):
    branch: str | None = None
    summary: str
    pr_url: str | None = None
    status: Literal["success", "failed"]
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionStepResult(BaseModel):
    step_id: str
    repo: str
    status: Literal["success", "failed"]
    openhands_result: OpenHandsExecutionResult


class ToolCallResult(BaseModel):
    step_id: str
    tool: str
    status: Literal["success", "failed"]
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class WorkflowRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    workflow_name: str
    user_request: str
    session_id: str | None = None
    user_id: str | None = None
    status: Literal["pending", "running", "waiting_approval", "completed", "failed"] = "pending"
    current_step: str | None = None
    approval_status: Literal["not_required", "pending", "approved", "rejected"] = "not_required"
    plan: PlanResult | None = None
    tool_call_results: list[ToolCallResult] = Field(default_factory=list)
    execution_results: list[ExecutionStepResult] = Field(default_factory=list)
    intermediate_outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def touch(self) -> None:
        self.updated_at = utcnow()


class WorkflowRunResponse(BaseModel):
    run: WorkflowRun
