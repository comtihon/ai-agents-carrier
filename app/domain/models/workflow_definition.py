from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class WorkflowStepDefinition(BaseModel):
    id: str
    name: str
    type: Literal["plan", "execute", "approval", "result"]
    repo: str | None = None
    instructions: str | None = None
    requires: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowDefinition(BaseModel):
    id: str
    name: str
    description: str
    entrypoint: str = "plan"
    steps: list[WorkflowStepDefinition]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_structure(self) -> "WorkflowDefinition":
        if not self.steps:
            raise ValueError("Workflow must define at least one step.")
        step_ids = {step.id for step in self.steps}
        if len(step_ids) != len(self.steps):
            raise ValueError("Workflow step ids must be unique.")
        if self.entrypoint not in step_ids:
            raise ValueError("Workflow entrypoint must refer to an existing step id.")
        for step in self.steps:
            missing = set(step.requires) - step_ids
            if missing:
                raise ValueError(f"Step '{step.id}' references unknown dependencies: {sorted(missing)}")
        return self
