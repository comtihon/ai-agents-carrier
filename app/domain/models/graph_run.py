from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class GraphRun(BaseModel):
    id: str                          # == LangGraph thread_id
    graph_id: str
    user_request: str = ""
    status: Literal["running", "waiting_approval", "completed", "failed", "cancelled"]
    state: dict[str, Any] = {}       # latest graph state snapshot
    current_step: str | None = None  # id of the node currently active / paused
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
