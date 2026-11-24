"""Runtime pools: a set of runner agents that team workflows run on.

A pool is a runtime of its own. An agent with ``default_runtime: "pool"``
names a pool (``pool_id``) instead of one machine; each run of a workflow
takes the next online member of the pool (round robin) for its first step on
the pool and keeps it for every later step, so all of a run's steps share one
machine and its workspace. The run moves to the next member only when its
member goes offline (app.application.runtime_pools).

Members are runner agents of any visibility, private ones included: a private
agent joins a pool only by its owner's choice, which is what lets a team
workflow run on it without the agent being shared with the team.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models.ownership import OwnedResource


class PoolMember(BaseModel):
    agent_id: str
    # Who added it (always the agent's owner) and when.
    added_by: str | None = None
    added_by_name: str | None = None
    added_at: datetime | None = None


class RuntimePool(OwnedResource):
    id: str
    name: str = ""
    description: str | None = None
    # Only the membership routes change these: every save through the scoped
    # backend keeps the stored list (see app.application.runtime_pools).
    members: list[PoolMember] = Field(default_factory=list)
    # Round-robin position; advanced atomically by the backend.
    cursor: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def member_ids(self) -> list[str]:
        return [m.agent_id for m in self.members]

    def touch(self) -> None:
        from datetime import timezone
        self.updated_at = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = self.updated_at


class PoolAssignment(BaseModel):
    """Which member of a pool one workflow run uses."""

    pool_id: str
    run_id: str
    agent_id: str
    assigned_at: datetime
    # Members this run moved away from because they went offline.
    previous: list[str] = Field(default_factory=list)
