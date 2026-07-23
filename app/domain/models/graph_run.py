from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class RoutingEvent(BaseModel):
    """Single routing or node-start event persisted for restart resilience."""
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event: Literal["route", "node_start"]
    node: str            # source node for "route", executing node for "node_start"
    target: str = ""     # destination for "route" events
    condition: str = ""  # matched when-condition (empty = default route)
    iteration: int = 0   # _visit_counts for the source node at decision time


class WaitingTransition(BaseModel):
    """Active inter-node `wait_seconds` delay (set while the router sleeps)."""
    source: str
    target: str
    wait_seconds: float
    started_at: datetime


class GraphRun(BaseModel):
    id: str                          # == LangGraph thread_id
    graph_id: str
    user_request: str = ""
    status: Literal["running", "waiting_approval", "waiting_agent", "completed", "failed", "cancelled", "rejected"]
    parent_run_id: str | None = None  # set when this run was spawned by a workflow step
    agent_url: str | None = None  # URL of the running agent HTTP server (set while waiting_agent)
    # Snapshot of the workflow definition at the time the run was started.
    # Stored so that approval-resume uses the exact same definition even after
    # the workflow is updated or the registry is refreshed.
    workflow_definition: dict[str, Any] | None = None
    state: dict[str, Any] = {}       # latest graph state snapshot
    current_step: str | None = None  # id of the node currently active / paused
    step_statuses: dict[str, str] = {}   # step_id → pending/running/finished/skipped/failed/waiting_clarification/waiting_approval
    step_inputs: dict[str, Any] = {}    # step_id → state snapshot passed into the node
    step_outputs: dict[str, Any] = {}   # step_id → raw node output dict (captured during streaming)
    waiting_transition: WaitingTransition | None = None
    # Routing + node-start journal — used to recover from backend restarts mid-run.
    # Appended on every routing decision and every node execution start so that on
    # resume we know the exact position in the graph and can clear stale sentinels
    # (e.g. __failed_step__ left over from a previous loop iteration).
    routing_log: list[RoutingEvent] = Field(default_factory=list)
    # LangSmith / local tracing
    langsmith_run_id: str | None = None
    trace_data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
