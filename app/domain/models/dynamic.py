"""Job workflows: configuration and run-time DAG models.

A ``type: dynamic`` step runs a DAG of jobs. Each job is one agent working on
one part of the work: planning (information gathering, research, design),
execution, validation or integration. A meta-agent orchestrates it:

- given the request it either plans the work, or — when it does not understand
  the request well enough — first spawns information-gathering jobs and plans
  the next phase once they report;
- for each job it picks the best agent from the pool and the data sources
  (from those enabled for the step) the job needs;
- whenever something happens that the rules cannot settle — an agent asks a
  question, a validator requests changes, a job asks for help or keeps
  failing, a phase ends — it sees the whole DAG and decides: answer the agent,
  message or stop a running agent, rewind the DAG to a job (everything after it
  runs again) and rearrange what follows, add jobs, finish, or ask a human.

Every category is optional. The DAG may instead be authored (``plan``); the
meta-agent then only steps in on such events. The persisted record is
append-only: removed jobs and superseded attempts stay, with their reasons.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

JobCategory = Literal["planning", "execution", "validation", "integration"]
JOB_CATEGORIES: tuple[JobCategory, ...] = ("planning", "execution", "validation", "integration")

# Mirrors Claude Code's permission modes: from "ask about everything" down to
# "never ask".
AutomationMode = Literal["ask", "plan", "auto", "bypass"]

# plan       — the meta-agent's plan for a phase (initial or after a phase ended)
# rearrange  — a rewind, removal, change or addition of jobs in reaction to an event
# question   — the meta-agent asks a human (it cannot settle something itself)
# escalation — a budget ran out; a human decides whether the run goes on
# answer     — answers and messages to agents only (never waits for a human)
# handback, delegate — kinds of records written before the meta-agent decided rewinds
DecisionKind = Literal["plan", "rearrange", "answer", "question", "escalation", "handback", "delegate"]

# Which meta-agent decisions wait for a human under each mode. Anything not
# listed is applied straight away. Answers and messages to agents never wait.
APPROVAL_POLICY: dict[str, frozenset[str]] = {
    "ask": frozenset({"plan", "rearrange", "question", "escalation"}),
    "plan": frozenset({"plan", "question", "escalation"}),
    "auto": frozenset({"question", "escalation"}),
    "bypass": frozenset(),
}


class PoolAgent(BaseModel):
    """One agent the meta-agent may place into jobs."""

    agent_id: str
    max_instances: int = Field(default=1, ge=1, le=32)
    # Restricts the categories this agent may fill. Empty = any category.
    categories: list[JobCategory] = Field(default_factory=list)


class StepDatasource(BaseModel):
    """A data source enabled for the step: the meta-agent may hand it to jobs."""

    source_id: str
    operations: list[str] = Field(default_factory=list)
    description: str = ""


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
        "planning": JobSlot(min=0, max=5),
        "execution": JobSlot(min=0, max=8),
        "validation": JobSlot(min=0, max=4),
        "integration": JobSlot(min=0, max=1),
    }


class DynamicLimits(BaseModel):
    max_total_jobs: int = Field(default=30, ge=1, le=500)
    # Automatic retries of a job whose attempt failed (agent crash, quality gate).
    max_retries: int = Field(default=2, ge=0, le=20, validation_alias=AliasChoices("max_retries", "max_handbacks"))
    # Meta-agent decisions that change the DAG in reaction to an event:
    # rewinds, removals, changed or added jobs.
    max_rearrangements: int = Field(default=10, ge=0, le=100)
    # How many times a human may reject the meta-agent's decision before the run fails.
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
    """One job as planned: by the meta-agent or by the author."""

    id: str
    category: JobCategory
    agent_id: str
    title: str = ""
    prompt: str
    depends_on: list[str] = Field(default_factory=list)
    # What the job may change: file globs, or named parts of an artifact.
    owns: list[str] = Field(default_factory=list)
    # Data sources (ids from the step's ``datasources``) granted to this job.
    datasources: list[str] = Field(default_factory=list)


class DynamicConfig(BaseModel):
    """The ``type: dynamic`` step's own fields (everything but id/type/next)."""

    agent_pool: list[PoolAgent] = Field(default_factory=list)
    jobs: dict[JobCategory, JobSlot] = Field(default_factory=_default_slots)
    # An authored job DAG. When set the meta-agent does not plan the run.
    plan: list[JobSpec] | None = None
    # Data sources the meta-agent may grant to jobs, on top of the agents' own addons.
    datasources: list[StepDatasource] = Field(default_factory=list)
    automation: AutomationMode = "plan"
    limits: DynamicLimits = Field(default_factory=DynamicLimits)
    # Meta-agent (meta-LLM) overrides; fall back to META_LLM_* settings.
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
        sources = [d.source_id for d in self.datasources]
        dup = {s for s in sources if sources.count(s) > 1}
        if dup:
            raise ValueError(f"datasources lists {sorted(dup)} more than once")
        if self.plan is not None and not self.plan:
            raise ValueError("plan is empty: list the jobs, or leave plan out to let the meta-agent plan")
        return self

    @classmethod
    def from_step(cls, step: dict[str, Any]) -> "DynamicConfig":
        body = {k: v for k, v in step.items() if k not in ("id", "type", "next", "routes", "when", "name", "label")}
        return cls.model_validate(body)

    def pool_entry(self, agent_id: str) -> PoolAgent | None:
        return next((a for a in self.agent_pool if a.agent_id == agent_id), None)

    def datasource(self, source_id: str) -> StepDatasource | None:
        return next((d for d in self.datasources if d.source_id == source_id), None)

    def needs_approval(self, kind: str) -> bool:
        return kind in APPROVAL_POLICY[self.automation]


# ─── Run-time record ─────────────────────────────────────────────────────────

# waiting: the job stopped and needs the meta-agent (a question, a request for
#   help, failures beyond its retries) or a human.
# removed: the meta-agent took the job out of the DAG.
JobStatus = Literal["pending", "running", "finished", "failed", "waiting", "skipped", "cancelled", "removed"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobAttempt(BaseModel):
    n: int
    child_run_id: str | None = None
    status: Literal["running", "finished", "failed", "cancelled", "needs_input"] = "running"
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    # Why this attempt exists: initial | retry | rewind | upstream_changed | answer | info_ready | restart
    reason: str = "initial"
    feedback: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
    # What the agent asked when the attempt ended in needs_input.
    questions: list[str] = Field(default_factory=list)


class JobNote(BaseModel):
    """Something a job must know: a validator's critique, an answer, a message.

    Notes accumulate, so a job sent back twice sees both reasons.
    """

    ts: datetime = Field(default_factory=_now)
    # validator:<job id> | meta | human | system
    source: str
    text: str
    # Attempts the job had when the note was added: the next attempt is the first to see it.
    after_attempt: int = 0


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
    datasources: list[str] = Field(default_factory=list)
    status: JobStatus = "pending"
    attempts: list[JobAttempt] = Field(default_factory=list)
    notes: list[JobNote] = Field(default_factory=list)
    # Why the next attempt runs (initial when absent).
    pending_reason: str | None = None
    # Answers the next attempt receives as clarification_context.
    pending_answers: dict[str, Any] | None = None
    created_by: Literal["dispatcher", "planner", "delegation", "human", "author", "meta"] = "meta"
    # Legacy records (before meta-agent rewinds) — read, never written.
    delegated_by: str | None = None
    awaiting: list[str] = Field(default_factory=list)
    pending_feedback: str | None = None

    @property
    def last(self) -> JobAttempt | None:
        return self.attempts[-1] if self.attempts else None

    @property
    def output(self) -> dict[str, Any] | None:
        for a in reversed(self.attempts):
            if a.status == "finished" and a.output is not None:
                return a.output
        return None

    @property
    def active(self) -> bool:
        return self.status not in ("removed", "cancelled")

    def note(self, source: str, text: str) -> None:
        self.notes.append(JobNote(source=source, text=text[:4000], after_attempt=len(self.attempts)))
        if len(self.notes) > 50:
            self.notes = self.notes[-50:]

    def new_notes(self) -> list[JobNote]:
        """Notes the next attempt has not seen yet."""
        return [n for n in self.notes if n.after_attempt >= len(self.attempts)]


SituationKind = Literal[
    "start",            # plan the request
    "phase_done",       # nothing is running or ready: plan the next phase, or finish
    "change_request",   # a validator failed its check
    "question",         # an attempt ended asking questions
    "live_question",    # a running agent asks and waits for the answer
    "needs_help",       # a job asked for information or help from another agent
    "failure",          # a job failed beyond its retries
    "rejected",         # a human rejected the meta-agent's last decision
    "human_answer",     # a human answered the meta-agent's question
]


class Situation(BaseModel):
    """Something the meta-agent has to decide about."""

    kind: SituationKind
    job_id: str | None = None
    attempt: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=_now)


class Decision(BaseModel):
    """A meta-agent decision (or escalation) recorded on the DAG.

    Applied ones are recorded resolved; ones the automation mode gates wait
    for a human (``resolved=False``) while the run is paused.
    """

    id: str
    kind: DecisionKind
    summary: str
    # meta decisions: {"actions": [...], "situations": [...]}; question: {"question", "job_id"}
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
    # Waiting for the meta-agent.
    situations: list[Situation] = Field(default_factory=list)
    replans: int = 0
    rearrangements: int = 0
    # The meta-agent declared the work done.
    finished: bool = False
    summary: str | None = None
    error: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    # Legacy counters (before meta-agent rewinds).
    delegations: int = 0
    expanded: bool = False

    def job(self, job_id: str) -> Job | None:
        return next((j for j in self.jobs if j.id == job_id), None)

    def active_jobs(self) -> list[Job]:
        return [j for j in self.jobs if j.active]

    def running(self) -> list[Job]:
        return [j for j in self.jobs if j.status == "running"]

    def open_decisions(self) -> list[Decision]:
        return [d for d in self.decisions if not d.resolved]

    def log(self, kind: str, job_id: str | None = None, detail: str = "") -> None:
        self.events.append(DagEvent(kind=kind, job_id=job_id, detail=detail[:2000]))
        # Bounded: the DAG lives on the run document.
        if len(self.events) > 500:
            self.events = self.events[-500:]
