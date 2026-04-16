from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class WorkflowDefinition(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    steps: list[dict[str, Any]] = []
    readonly: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def to_raw_dict(self) -> dict[str, Any]:
        """Return the raw definition dict as expected by YamlGraphRunner."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
        }
