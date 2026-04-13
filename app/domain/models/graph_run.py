from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class GraphRun(BaseModel):
    id: str                          # == LangGraph thread_id
    graph_id: str
    status: Literal["running", "waiting_approval", "completed", "failed"]
    state: dict[str, Any] = {}       # latest graph state snapshot
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
