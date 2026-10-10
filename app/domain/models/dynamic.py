"""Job workflows: configuration and run-time DAG models.

A ``type: dynamic`` step runs a DAG of jobs. Each job is one agent working on
one part of the work: planning, execution, validation or integration. The DAG
comes from one of two places:

- *dynamic*: the author declares which agents may take part (the agent pool)
  and how many jobs of each category the run may hold; at run time the
  dispatcher — the meta-LLM — reads the request and the pool's descriptions and
  fills job slots with agents, splitting a big job into parts across several
  agents of the same category;
- *static*: the author writes the jobs into the step (``plan``); the
  dispatcher is only consulted when a job delegates.

Every category is optional: a run may have no executor (research only) or no
validator. Jobs may hand work back (a validator returning a part to the coder
that produced it — the next iteration) or delegate (a planner asking for a
researcher), so the logical flow loops while the persisted record stays an
append-only list of job attempts. Only the jobs that failed or were handed
back run again; finished parts keep their results.

The config is stored as an ordinary workflow step (``type: dynamic``), so a
workflow can be entirely dynamic (the designer shows only this config) or embed
a dynamic stage between static steps.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

JobCategory = Literal["planning", "execution", "validation", "integration"]
JOB_CATEGORIES: tuple[JobCategory, ...] = ("planning", "execution", "validation", "integration")

# Mirrors Claude Code's permission modes: from "ask about everything" down to
# "never ask". Escalations (a job that keeps failing, an exhausted budget) are
# the one thing every mode but bypass still brings to a human.
AutomationMode = Literal["ask", "plan", "auto", "bypass"]

DecisionKind = Literal["plan", "handback", "delegate", "escalation", "question"]

# Which decisions need a human under each mode. Anything not listed is applied
# straight away. ``question`` (an agent asked something it cannot go on
# without) is listed everywhere but bypass: nobody else can answer it.
APPROVAL_POLICY: dict[str, frozenset[str]] = {
    "ask": frozenset({"plan", "handback", "delegate", "escalation", "question"}),
    "plan": frozenset({"plan", "escalation", "question"}),
    "auto": frozenset({"escalation", "question"}),
    "bypass": frozenset(),
}


class PoolAgent(BaseModel):
    """One agent the dispatcher may place into jobs."""

    agent_id: str
    max_instances: int = Field(default=1, ge=1, le=32)
    # Restricts the categories this agent may fill. Empty = any category.
    categories: list[JobCategory] = Field(default_factory=list)


class JobSlot(BaseModel):
    min: int = Field(default=0, ge=0)
    max: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def _min_le_max(self) -> "JobSlot":
        if self.min > self.max:
            raise ValueError(f"job slot min ({self.min}) exceeds max ({self.max})")
        return self


def _default_slots() -> dict[str, JobSlot]:
    return {
        "planning": JobSlot(min=0, max=3),
        "execution": JobSlot(min=0, max=8),
        "validation": JobSlot(min=0, max=3),
        "integration": JobSlot(min=0, max=1),
    }


class DynamicLimits(BaseModel):
    max_total_jobs: int = Field(default=30, ge=1, le=500)
    # Attempts per job beyond the first, whatever caused them: a validator's
    # handback, a meta-LLM rejection, an agent crash.
    max_handbacks: int = Field(default=3, ge=0, le=20)
    max_delegations: int = Field(default=5, ge=0, le=50)
    # How many times a rejected plan is re-planned before the run fails.
    max_replans: int = Field(default=2, ge=0, le=10)
    max_parallel: int = Field(default=4, ge=1, le=32)


class SharedVolume(BaseModel):
    """A run-wide volume every job mounts (PVC; k8s runtime only)."""

    mount_point: str = "/shared"
    ttl: str = "6h"


class RepoConfig(BaseModel):
    """Git conventions handed to coding jobs."""

    url: str
    base_branch: str = "main"
    # Commands the integration job and validators must run on the integrated head.
    verify: list[str] = Field(default_factory=list)


class JobSpec(BaseModel):
    """One job as planned: by the dispatcher, a planner's expansion, or the author."""

    id: str
    category: JobCategory
    agent_id: str
    title: str = ""
    prompt: str
    depends_on: list[str] = Field(default_factory=list)
    # What the job may change: file globs, or named parts of an artifact.
    owns: list[str] = Field(default_factory=list)


class DynamicConfig(BaseModel):
    """The ``type: dynamic`` step's own fields (everything but id/type/next)."""

    agent_pool: list[PoolAgent] = Field(default_factory=list)
    jobs: dict[JobCategory, JobSlot] = Field(default_factory=_default_slots)
    # An authored job DAG. When set the dispatcher does not plan the run.
    plan: list[JobSpec] | None = None
    automation: AutomationMode = "plan"
    limits: DynamicLimits = Field(default_factory=DynamicLimits)
    # Dispatcher (meta-LLM) overrides; fall back to META_LLM_* settings.
    dispatcher_provider: str | None = None
    dispatcher_model: str | None = None
    dispatcher_instructions: str = ""
    shared_volume: SharedVolume | None = None
    repo: RepoConfig | None = None
    # Where the final summary lands in workflow state.
    output_key: str | None = None

    @field_validator("jobs", mode="before")
    @classmethod
    def _fill_slots(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        merged: dict[str, Any] = {k: s.model_dump() for k, s in _default_slots().items()}
        merged.update(v)
        return merged

    @model_validator(mode="after")
    def _check_pool(self) -> "DynamicConfig":
        ids = [a.agent_id for a in self.agent_pool]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"agent_pool lists {sorted(dup)} more than once")
        if self.plan is not None and not self.plan:
            raise ValueError("plan is empty: list the jobs, or leave plan out to let the dispatcher plan")
        return self

    @classmethod
    def from_step(cls, step: dict[str, Any]) -> "DynamicConfig":
        body = {k: v for k, v in step.items() if k not in ("id", "type", "next", "routes", "when", "name", "label")}
        return cls.model_validate(body)

    def pool_entry(self, agent_id: str) -> PoolAgent | None:
        return next((a for a in self.agent_pool if a.agent_id == agent_id), None)

    def needs_approval(self, kind: str) -> bool:
        return kind in APPROVAL_POLICY[self.automation]


# ─── Run-time record ─────────────────────────────────────────────────────────

JobStatus = Literal["pending", "running", "finished", "failed", "waiting", "skipped", "cancelled"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobAttempt(BaseModel):
    n: int
    child_run_id: str | None = None
    status: Literal["running", "finished", "failed", "cancelled", "needs_input"] = "running"
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    # Why this attempt exists: initial | handback | retry | delegate_return | answer
    reason: str = "initial"
    feedback: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
    # What the agent asked when the attempt ended in needs_input.
    questions: list[str] = Field(default_factory=list)


class Job(BaseModel):
    id: str
    category: JobCategory
    agent_id: str
    title: str = ""
    prompt: str
    depends_on: list[str] = Field(default_factory=list)
    # What the job may change: file globs, or named parts of an artifact
    # (e.g. a Blender collection). Advisory for the agent, checked by validators.
    owns: list[str] = Field(default_factory=list)
    status: JobStatus = "pending"
    attempts: list[JobAttempt] = Field(default_factory=list)
    # A job created by another job's delegate request; its result returns there.
    delegated_by: str | None = None
    # Jobs this one is waiting for beyond depends_on (open delegations).
    awaiting: list[str] = Field(default_factory=list)
    # Feedback / answers the next attempt must see.
    pending_feedback: str | None = None
    pending_reason: str | None = None
    pending_answers: dict[str, Any] | None = None
    created_by: Literal["dispatcher", "planner", "delegation", "human", "author"] = "dispatcher"

    @property
    def last(self) -> JobAttempt | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def output(self) -> dict[str, Any] | None:
        for a in reversed(self.attempts):
            if a.status == "finished" and a.output is not None:
                return a.output
        return None


class Decision(BaseModel):
    """Something the engine wants to do that the automation mode gates."""

    id: str
    kind: DecisionKind
    summary: str
    # kind-specific: plan → {"jobs": [...]}, handback → {"job_id", "feedback", "by"},
    # delegate → {"job_id", "request", "job": {...}}, escalation → {"job_id", "error"},
    # question → {"job_id", "questions": [...]}
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
    resolved: bool = False
    approved: bool | None = None
    reason: str | None = None


class DagEvent(BaseModel):
    ts: datetime = Field(default_factory=_now)
    kind: str
    job_id: str | None = None
    detail: str = ""


class DynamicRunState(BaseModel):
    """Everything a dynamic step knows about its run. Persisted on GraphRun.dynamic[step_id]."""

    step_id: str
    status: Literal["planning", "running", "waiting_approval", "completed", "failed"] = "planning"
    jobs: list[Job] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    events: list[DagEvent] = Field(default_factory=list)
    replans: int = 0
    delegations: int = 0
    expanded: bool = False  # planner output already turned into jobs
    summary: str | None = None
    error: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)

    def job(self, job_id: str) -> Job | None:
        return next((j for j in self.jobs if j.id == job_id), None)

    def open_decisions(self) -> list[Decision]:
        return [d for d in self.decisions if not d.resolved]

    def log(self, kind: str, job_id: str | None = None, detail: str = "") -> None:
        self.events.append(DagEvent(kind=kind, job_id=job_id, detail=detail[:2000]))
        # Bounded: the DAG lives on the run document.
        if len(self.events) > 500:
            self.events = self.events[-500:]
