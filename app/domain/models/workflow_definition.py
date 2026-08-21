from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class WorkflowDefinition(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    steps: list[dict[str, Any]] = []
    ui: dict[str, Any] = Field(default_factory=dict)
    readonly: bool = False
    # Disabling a workflow stops every trigger and every manual start (see
    # app.application.run_control.ensure_workflow_enabled). Defaults to True so
    # documents stored before the field existed keep running as they did.
    enabled: bool = True
    use_meta_llm: bool = True
    # Opt-in per-workflow key/value storage (see
    # app.infrastructure.persistence.workflow_storage). Off by default: a
    # workflow that never asked for state should not silently accumulate any,
    # and `storage` steps fail loudly rather than no-op when it is off.
    use_storage: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def to_raw_dict(self) -> dict[str, Any]:
        """Return the raw definition dict as expected by YamlGraphRunner."""
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
        }
        if self.ui:
            d["ui"] = self.ui
        d["use_meta_llm"] = self.use_meta_llm
        d["use_storage"] = self.use_storage
        d["enabled"] = self.enabled
        return d
