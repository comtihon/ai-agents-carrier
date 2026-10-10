"""The job DAG's rules: events in, meta-agent decisions, launches and stops out.

``JobDag`` holds no tasks and no loop of its own. The LangGraph job graph
(``graph.py``) calls ``JobDag.schedule()`` every time something happens — an
attempt finished, a running agent asked a question, a human decided — and
carries out the ``Step`` it returns: attempts to launch, attempts to stop,
answers to deliver to waiting agents, and whether to wait for the next event,
pause for a human, or end.

Fixed rules handle the routine: a finished job unblocks its dependents, a
failed attempt is retried while it has retries left. Everything else becomes a
``Situation`` for the meta-agent, who sees the whole DAG and answers with
actions (``meta_agent.MetaDecision``). Every decision is validated against the
DAG before it is applied — on a copy first — and the automation mode decides
which ones wait for a human.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.models.dynamic import (
    Decision,
    DynamicConfig,
    DynamicRunState,
    Job,
    JobAttempt,
    Situation,
)
from app.infrastructure.orchestration.dynamic.dispatcher import (
    DispatcherError,
    check_assignment,
    check_minimums,
    has_cycle,
    validate_jobs,
)
from app.infrastructure.orchestration.dynamic.meta_agent import MetaAction, MetaAgent, MetaDecision, MetaView

logger = logging.getLogger(__name__)

StepStatus = Literal["run", "gate", "completed", "failed"]

# Situations about a job that stopped and waits: the decision must settle that job.
_BLOCKING = ("question", "live_question", "needs_help", "failure")
# Situations that ask for the next phase of work.
_PHASE = ("start", "phase_done")
_REARRANGING = ("rewind", "remove", "update_job", "add_jobs")
_UPDATABLE = ("prompt", "title", "agent_id", "depends_on", "owns", "datasources")


class JobOutcome(BaseModel):
    status: Literal["finished", "failed", "needs_input"]
    output: dict[str, Any] | None = None
    error: str | None = None
    questions: list[str] = Field(default_factory=list)


class AttemptResult(JobOutcome):
    """A job attempt's outcome, addressed to the attempt that produced it."""

    type: Literal["result"] = "result"
    job_id: str
    attempt: int
    child_run_id: str | None = None


class LiveQuestion(BaseModel):
    """A running agent asked a question and waits for the answer."""

    type: Literal["live_question"] = "live_question"
    job_id: str
    attempt: int
    child_run_id: str | None = None
    question: str


class LostAttempt(BaseModel):
    """An attempt the graph believes running has no live task (restart)."""

    type: Literal["lost"] = "lost"
    job_id: str
    attempt: int


Event = AttemptResult | LiveQuestion | LostAttempt


def parse_event(raw: dict[str, Any]) -> Event:
    kind = raw.get("type", "result")
    if kind == "live_question":
        return LiveQuestion.model_validate(raw)
    if kind == "lost":
        return LostAttempt.model_validate(raw)
    return AttemptResult.model_validate(raw)


class Launch(BaseModel):
    """One job attempt to start."""

    job: Job
    attempt: JobAttempt
    payload: dict[str, Any]


class AttemptRef(BaseModel):
    job_id: str
    attempt: int


class Delivery(BaseModel):
    """An answer for a running agent that is waiting on its question."""

    job_id: str
    attempt: int
    child_run_id: str | None
    text: str


class Step(BaseModel):
    status: StepStatus
    launches: list[Launch] = Field(default_factory=list)
    cancels: list[AttemptRef] = Field(default_factory=list)
    deliveries: list[Delivery] = Field(default_factory=list)


JobRunner = Callable[[Job, JobAttempt, dict[str, Any], Callable[[str], Awaitable[None]]], Awaitable[JobOutcome]]


# Fields each category's agent is asked to return. execute_agent_step needs at
# least one of them present; the rest are optional.
OUTPUT_FIELDS: dict[str, list[str]] = {
    "planning": ["summary", "plan", "delegate"],
    "execution": ["summary", "branch", "artifacts", "delegate"],
    "validation": ["verdict", "feedback", "handback_to", "summary"],
    "integration": ["summary", "branch", "conflicts", "delegate"],
}

_INSTRUCTIONS: dict[str, str] = {
    "planning": (
        "You are a planning job in a multi-agent workflow: gather information, research or design; do not "
        "implement. Return `summary` and `plan` (your findings, or a concrete breakdown: parts of the work, what "
        "each part may change, dependencies between parts, how to verify). If you need information another "
        "agent should gather, return `delegate` with a precise request instead of guessing."
    ),
    "execution": (
        "You are an execution job in a multi-agent workflow. Do the work in your brief and only that: "
        "change only what your `owns` list names; other jobs work on the rest in parallel. "
        "Return `summary` (what you changed and how you checked it), plus `branch` if you pushed code and "
        "`artifacts` (paths/URLs of files you produced). If you are blocked on missing information, "
        "return `delegate` with a precise request."
    ),
    "validation": (
        "You are a validation job in a multi-agent workflow. Check the upstream jobs' results against the "
        "request. Return `verdict`: \"pass\" or \"fail\". On fail return `feedback` (exactly what is wrong and "
        "what must change) and `handback_to` (the job ids you believe must redo their part)."
    ),
    "integration": (
        "You are the integration job in a multi-agent workflow. Combine the upstream jobs' results into one "
        "(e.g. merge their branches into the integration branch, or assemble their files). Do not rewrite "
        "their work. Return `summary`, the integrated `branch` or artifacts, and `conflicts` listing anything "
        "you could not combine and which job owns it."
    ),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _truncate(value: Any, cap: int = 12000) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) <= cap:
        return value
    return text[:cap] + f"... [truncated {len(text) - cap} chars]"


def _verdict_failed(output: dict[str, Any] | None) -> bool:
    if not isinstance(output, dict):
        return False
    verdict = str(output.get("verdict", "pass")).strip().lower()
    return verdict in ("fail", "failed", "reject", "rejected", "no")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.replace(";", ",").split(",") if v.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _help_request(output: dict[str, Any] | None) -> str | None:
    if not isinstance(output, dict):
        return None
    raw = output.get("delegate")
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw.get("request") or json.dumps(raw, ensure_ascii=False)
    return str(raw)


class JobDag:
    def __init__(
        self,
        *,
        config: DynamicConfig,
        state: DynamicRunState,
        meta: MetaAgent,
        roster: str,
        request: str,
        run_id: str,
        workspace: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.state = state
        self.meta = meta
        self.roster = roster
        self.request = request
        self.run_id = run_id
        self.workspace = workspace or {}
        self._step = Step(status="run")
        # Live questions not yet answered, by job id: (attempt, child run id).
        self._live: dict[str, tuple[int, str | None]] = {}

    # ── entry point ─────────────────────────────────────────────────────────

    async def schedule(
        self,
        *,
        events: Iterable[Event] = (),
        resolution: dict[str, Any] | None = None,
        stopped: bool = False,
    ) -> Step:
        """Fold in what happened; decide; return what the graph does next."""
        self._step = Step(status="run")
        try:
            status = await self._schedule(list(events), resolution, stopped)
        finally:
            self._collect_usage()
        self._step.status = status
        return self._step

    async def _schedule(self, events: list[Event], resolution: dict[str, Any] | None, stopped: bool) -> StepStatus:
        if self.state.status in ("completed", "failed"):
            return self.state.status  # type: ignore[return-value]
        self._load_live()
        for event in events:
            self._ingest(event)

        if stopped:
            self._stop_all()
            self._fail("run was stopped")
            return "failed"

        if resolution is not None and self.state.open_decisions():
            await self._resolve(resolution)

        if not self.state.jobs and not self.state.situations and self.state.status == "planning":
            if self.config.plan is not None:
                self._start_authored()
            else:
                self._situation("start")

        # Each round settles the open situations; a phase that ends during it
        # opens the next round. Bounded: a meta-agent that adds nothing to do
        # cannot spin here.
        for _ in range(4):
            if self.state.status == "failed" or self.state.open_decisions():
                break
            if self.state.situations:
                await self._consult()
                continue
            launches = self._launch_ready()
            if launches or self.state.running() or self._phase_over():
                break
            self._situation("phase_done")
        else:
            if self.state.status != "failed" and not self.state.open_decisions() and not self.state.situations:
                self._launch_ready()

        if self.state.status == "failed":
            self._stop_all()
            return "failed"
        if self.state.open_decisions():
            self.state.status = "waiting_approval"
            return "gate"
        if self.state.running():
            self.state.status = "running"
            return "run"
        if self.state.situations:
            self._fail("the meta-agent did not settle: " + ", ".join(s.kind for s in self.state.situations))
            return "failed"
        return self._finish_or_fail()

    def _collect_usage(self) -> None:
        for k, v in self.meta.usage.items():
            self.state.usage[k] = self.state.usage.get(k, 0) + v
        self.meta.usage = {}

    def _load_live(self) -> None:
        """Live questions still waiting for an answer, from situations and gated decisions."""
        pending = [s.model_dump(mode="json") for s in self.state.situations]
        for d in self.state.open_decisions():
            pending += d.payload.get("situations") or []
        for s in pending:
            if s.get("kind") == "live_question" and s.get("job_id"):
                self._live[s["job_id"]] = (s.get("attempt") or 0, (s.get("detail") or {}).get("child_run_id"))

    # ── events ──────────────────────────────────────────────────────────────

    def _ingest(self, event: Event) -> None:
        job = self.state.job(event.job_id)
        attempt = next((a for a in job.attempts if a.n == event.attempt), None) if job is not None else None
        if job is None or attempt is None:
            logger.warning("dynamic: event for unknown attempt %s #%s ignored", event.job_id, event.attempt)
            return
        if isinstance(event, LiveQuestion):
            if job.status == "running" and attempt.status == "running":
                self._situation("live_question", job, attempt.n, {"question": event.question, "child_run_id": event.child_run_id})
                self._live[job.id] = (attempt.n, event.child_run_id)
            return
        if attempt.status != "running":
            return  # already settled (stopped, or reported twice)
        # Cancelled from outside (shutdown, the workflow task stopped): not the
        # agent's failure, so it costs no retry. Stops decided here never get
        # this far — their attempt is marked cancelled when they are decided.
        lost = isinstance(event, LostAttempt) or (event.status == "failed" and event.error == "cancelled")
        if lost:
            attempt.status = "cancelled"
            attempt.finished_at = _now()
            if job.status == "running":
                job.status = "pending"
                job.pending_reason = "restart"
            self.state.log("attempt_lost", job.id, "attempt was in flight when the run restarted")
            return
        self._record(job, attempt, event)

    def _record(self, job: Job, attempt: JobAttempt, result: AttemptResult) -> None:
        attempt.finished_at = _now()
        attempt.child_run_id = result.child_run_id or attempt.child_run_id
        attempt.output = result.output
        attempt.error = result.error
        attempt.questions = result.questions
        attempt.status = result.status  # type: ignore[assignment]
        self._live.pop(job.id, None)
        self.state.situations = [s for s in self.state.situations if not (s.kind == "live_question" and s.job_id == job.id)]
        if job.status != "running":
            return  # stopped while it ran: the result is kept, nothing follows from it

        if result.status == "needs_input":
            job.status = "waiting"
            self._situation("question", job, attempt.n, {"questions": result.questions})
            return

        if result.status == "failed":
            self.state.log("attempt_failed", job.id, result.error or "")
            if self._retries_left(job):
                job.status = "pending"
                job.pending_reason = "retry"
                job.note("system", f"Your previous attempt failed: {result.error}")
                return
            job.status = "waiting"
            self._situation("failure", job, attempt.n, {"error": result.error, "attempts": len(job.attempts)})
            return

        job.status = "finished"
        self.state.log("attempt_finished", job.id, str((result.output or {}).get("summary", ""))[:300])
        ask = _help_request(result.output)
        if ask:
            job.status = "waiting"
            self._situation("needs_help", job, attempt.n, {"request": ask, "output": _truncate(result.output, 4000)})
            return
        if job.category == "validation" and _verdict_failed(result.output):
            output = result.output or {}
            self._situation("change_request", job, attempt.n, {
                "feedback": output.get("feedback") or output.get("summary") or "validation failed",
                "suggested_targets": _as_list(output.get("handback_to")),
            })

    def _retries_left(self, job: Job) -> bool:
        """Failed attempts since the job last ran for another reason, against the budget."""
        retries = 0
        for a in reversed(job.attempts[1:]):
            if a.reason != "retry":
                break
            retries += 1
        granted = sum(
            1 for d in self.state.decisions
            if d.kind == "escalation" and d.approved and d.payload.get("job_id") == job.id
        )
        return retries < self.config.limits.max_retries + granted

    def _situation(self, kind: str, job: Job | None = None, attempt: int | None = None, detail: dict[str, Any] | None = None) -> None:
        self.state.situations.append(Situation(kind=kind, job_id=job.id if job else None, attempt=attempt, detail=detail or {}))  # type: ignore[arg-type]
        self.state.log(f"situation:{kind}", job.id if job else None, json.dumps(detail or {}, default=str)[:500])

    # ── the meta-agent ──────────────────────────────────────────────────────

    async def _consult(self, rejection: dict[str, Any] | None = None) -> None:
        situations = list(self.state.situations)
        if rejection is not None:
            situations.append(Situation(kind="rejected", detail=rejection))
        view = MetaView(request=self.request, config=self.config, roster=self.roster, dag=self.state, situations=situations)
        try:
            decision = await self.meta.decide(view, lambda d: self.validate(d, situations))
        except DispatcherError as exc:
            self._fail(str(exc))
            return
        self.state.situations = []
        kind = self._kind_of(decision, situations)
        budget = self.config.limits.max_rearrangements + self._granted_rearrangements()
        if kind == "rearrange" and self.state.rearrangements >= budget:
            if self.config.automation == "bypass":
                self._fail(f"rearrangement budget ({self.config.limits.max_rearrangements}) exhausted: {decision.summary}")
                return
            self._record_decision("escalation", f"rearrangement budget exhausted; the meta-agent wants to: {decision.summary}", decision, situations)
            return
        if kind is not None and self.config.needs_approval(kind):
            self._record_decision(kind, decision.summary or kind, decision, situations)
            return
        record = self._record_decision(kind or "answer", decision.summary or "meta-agent decision", decision, situations)
        record.resolved, record.approved, record.reason = True, True, f"auto ({self.config.automation})"
        self._apply(decision, situations)

    def _kind_of(self, decision: MetaDecision, situations: list[Situation]) -> str | None:
        """The approval category of a decision; None for answers and messages only."""
        kinds = {a.kind for a in decision.actions}
        drops_work = "finish" in kinds and any(j.status != "finished" for j in self.state.active_jobs())
        if kinds & set(_REARRANGING) or drops_work:
            # A human's rejection or answer keeps the category of what it was about.
            about = [s for s in situations if s.kind not in ("rejected", "human_answer")]
            phase = all(s.kind in _PHASE for s in about) and kinds <= {"add_jobs", "finish"}
            return "plan" if phase else "rearrange"
        return None

    def _granted_rearrangements(self) -> int:
        return sum(
            self.config.limits.max_rearrangements or 1
            for d in self.state.decisions
            if d.kind == "escalation" and d.approved and "job_id" not in d.payload
        )

    def _record_decision(self, kind: str, summary: str, decision: MetaDecision, situations: list[Situation]) -> Decision:
        record = Decision(
            id=uuid.uuid4().hex[:10],
            kind=kind,  # type: ignore[arg-type]
            summary=summary[:2000],
            payload={
                "summary": decision.summary,
                "actions": [a.model_dump(mode="json") for a in decision.actions],
                "situations": [s.model_dump(mode="json") for s in situations],
            },
        )
        self.state.decisions.append(record)
        self.state.log(f"decision:{kind}", None, summary)
        return record

    async def _resolve(self, resolution: dict[str, Any]) -> None:
        approved = bool(resolution.get("approved", False))
        reason = resolution.get("reason")
        corrections = resolution.get("corrections") or {}
        for record in self.state.open_decisions():
            record.resolved, record.approved, record.reason = True, approved, reason
            situations = [Situation.model_validate(s) for s in record.payload.get("situations") or []]
            decision = MetaDecision(
                summary=record.payload.get("summary") or "",
                actions=[MetaAction.model_validate(a) for a in record.payload.get("actions") or []],
            )
            if record.kind == "question":
                self._human_answered(record, approved, reason, corrections)
            elif record.kind == "escalation":
                if not approved:
                    self._fail(f"escalation rejected: {reason or record.summary}")
                elif record.payload.get("job_id"):
                    job = self.state.job(record.payload["job_id"])
                    if job is not None:
                        job.status, job.pending_reason = "pending", "retry"
                else:
                    self._apply(decision, situations)
            elif approved:
                if corrections.get("jobs"):
                    decision = self._with_corrected_jobs(decision, corrections["jobs"])
                    errors = self.validate(decision, situations)
                    if errors:
                        await self._reject(record, "the edited plan breaks the rules: " + "; ".join(errors), situations)
                        continue
                self._apply(decision, situations, created_by="human" if corrections.get("jobs") else "meta")
            else:
                await self._reject(record, reason or "rejected without a reason", situations)

    async def _reject(self, record: Decision, reason: str, situations: list[Situation]) -> None:
        if self.state.replans >= self.config.limits.max_replans:
            self._fail(f"decision rejected {self.state.replans + 1} times; last reason: {reason}")
            return
        self.state.replans += 1
        self.state.situations = situations
        await self._consult({"rejected_decision": record.payload.get("summary") or record.summary,
                             "actions": record.payload.get("actions"), "reason": reason})

    def _with_corrected_jobs(self, decision: MetaDecision, jobs: list[dict[str, Any]]) -> MetaDecision:
        others = [a for a in decision.actions if a.kind != "add_jobs"]
        return MetaDecision(summary=decision.summary, actions=[MetaAction.model_validate({"kind": "add_jobs", "jobs": jobs}), *others])

    def _human_answered(self, record: Decision, approved: bool, reason: str | None, corrections: dict[str, Any]) -> None:
        answers = corrections.get("answers") or {}
        text = "; ".join(f"{q}: {a}" for q, a in answers.items()) if answers else (reason or "")
        if not approved and not text:
            text = "Nobody can answer this. Proceed with your best judgement and state your assumptions."
        job_id = record.payload.get("job_id")
        job = self.state.job(job_id) if job_id else None
        if job is not None and job.active:
            self._answer(job, text, source="human", answers=answers or None)
        else:
            self._situation("human_answer", None, None, {"question": record.payload.get("question"), "answer": text})

    # ── validating and applying decisions ───────────────────────────────────

    def validate(self, decision: MetaDecision, situations: list[Situation]) -> list[str]:
        """Every problem with *decision*; it is applied to a copy of the DAG to check the result."""
        if not decision.actions:
            return ["the decision has no actions"]
        errors: list[str] = []
        for action in decision.actions:
            errors += self._check_action(action)
        if errors:
            return errors
        trial = JobDag(config=self.config, state=self.state.model_copy(deep=True), meta=self.meta,
                       roster=self.roster, request=self.request, run_id=self.run_id, workspace=self.workspace)
        trial._live = dict(self._live)
        try:
            trial._apply(decision, situations, dry_run=True)
        except ValueError as exc:
            return [str(exc)]
        errors += trial._structure_errors()
        kinds = {a.kind for a in decision.actions}
        if kinds & {"finish", "fail"}:
            return errors
        settled = {jid for a in decision.actions for jid in ([a.job_id] if a.job_id else []) + list(a.job_ids)}
        for s in situations:
            if s.kind in _BLOCKING and s.job_id and s.job_id not in settled:
                job = self.state.job(s.job_id)
                if job is not None and job.active:
                    errors.append(
                        f"job {s.job_id!r} waits on its {s.kind.replace('_', ' ')}: "
                        "answer, rewind, message, remove it or ask a human"
                    )
        if any(s.kind in _PHASE for s in situations) and not kinds & {"add_jobs", "ask_human", "rewind", "update_job"}:
            errors.append("plan the next jobs (add_jobs), finish, fail, or ask a human")
        return errors

    def _check_action(self, a: MetaAction) -> list[str]:
        name = a.kind
        if a.kind in ("update_job", "answer", "message") and not a.job_id:
            return [f"{name}: job_id is required"]
        if a.kind in ("rewind", "remove") and not a.job_ids:
            return [f"{name}: job_ids is required"]
        if a.kind == "add_jobs" and not a.jobs:
            return ["add_jobs: jobs is empty"]
        if a.kind in ("answer", "message", "ask_human", "fail") and not a.text.strip():
            return [f"{name}: text is required"]
        if a.kind == "ask_human" and self.config.automation == "bypass":
            return ["ask_human: nobody answers in bypass mode; decide yourself"]
        errors = []
        for jid in ([a.job_id] if a.job_id else []) + list(a.job_ids):
            job = self.state.job(jid)
            if job is None or not job.active:
                errors.append(f"{name}: job {jid!r} does not exist or was removed")
        if a.kind == "update_job":
            job = self.state.job(a.job_id or "")
            unknown = set(a.changes) - set(_UPDATABLE)
            if unknown:
                errors.append(f"update_job: cannot change {sorted(unknown)} (allowed: {list(_UPDATABLE)})")
            if job is not None and job.status == "running":
                errors.append(f"update_job: {job.id!r} is running; rewind or remove it first")
        return errors

    def _structure_errors(self) -> list[str]:
        errors: list[str] = []
        active = self.state.active_jobs()
        ids = {j.id for j in active}
        for job in active:
            for dep in job.depends_on:
                if dep not in ids:
                    errors.append(f"job {job.id!r} depends on {dep!r}, which is removed or unknown; rewire it with update_job")
            errors += check_assignment(job.id, job.category, job.agent_id, job.datasources, self.config)
        if has_cycle({j.id: list(j.depends_on) for j in active}):
            errors.append("depends_on contains a cycle; loop by rewinding, not with cycles")
        for cat, slot in self.config.jobs.items():
            count = sum(1 for j in active if j.category == cat)
            if count > slot.max:
                errors.append(f"too many {cat} jobs: {count} > max {slot.max}")
        if len(self.state.jobs) > self.config.limits.max_total_jobs:
            errors.append(f"too many jobs in total: {len(self.state.jobs)} > {self.config.limits.max_total_jobs}")
        return errors

    def _apply(self, decision: MetaDecision, situations: list[Situation], *, dry_run: bool = False, created_by: str = "meta") -> None:
        if not dry_run and self._kind_of(decision, situations) == "rearrange":
            self.state.rearrangements += 1
        for action in decision.actions:
            getattr(self, f"_do_{action.kind}")(action, created_by)
        if self.state.status == "planning" and self.state.jobs:
            self.state.status = "running"

    def _do_add_jobs(self, a: MetaAction, created_by: str) -> None:
        errors = validate_jobs(a.jobs, self.config, self.state.jobs)
        if errors:
            raise ValueError("add_jobs: " + "; ".join(errors))
        for spec in a.jobs:
            self.state.jobs.append(Job(**spec.model_dump(), created_by=created_by))  # type: ignore[arg-type]
            self.state.log("job_added", spec.id, f"{spec.category} → {spec.agent_id}")

    def _do_update_job(self, a: MetaAction, created_by: str) -> None:
        job = self._get(a.job_id)
        updated = Job.model_validate({**job.model_dump(), **a.changes})  # raises on malformed changes
        for key in a.changes:
            setattr(job, key, getattr(updated, key))
        if job.status in ("finished", "failed", "skipped", "waiting"):
            job.status, job.pending_reason = "pending", job.pending_reason or "rewind"
        self.state.log("job_updated", job.id, ", ".join(a.changes))

    def _do_rewind(self, a: MetaAction, created_by: str) -> None:
        for jid in a.job_ids:
            target = self._get(jid)
            self._reset(target, "rewind")
            target.note("meta", a.text or "Do this part again.")
            for dep in self._dependents(jid):
                if dep.id in a.job_ids:
                    continue
                if dep.status in ("finished", "failed", "skipped", "waiting", "running"):
                    self._reset(dep, "upstream_changed")
                    dep.note("system", f"Upstream job {jid} is being redone: {a.text}")
        self.state.log("rewind", ",".join(a.job_ids), a.text)

    def _do_remove(self, a: MetaAction, created_by: str) -> None:
        for jid in a.job_ids:
            job = self._get(jid)
            self._stop(job)
            job.status = "removed"
            job.note("meta", f"Removed: {a.text}")
        self.state.log("removed", ",".join(a.job_ids), a.text)

    def _do_answer(self, a: MetaAction, created_by: str) -> None:
        self._answer(self._get(a.job_id), a.text, source="meta")

    def _do_message(self, a: MetaAction, created_by: str) -> None:
        job = self._get(a.job_id)
        job.note("meta", a.text)
        if job.status == "running" and a.restart:
            self._reset(job, "restart")
        elif job.status == "waiting":
            job.status, job.pending_reason = "pending", "message"
        elif job.status == "running" and job.id in self._live:
            # The agent is waiting on a question right now: the message is its answer.
            self._deliver(job, a.text)
        self.state.log("message", job.id, a.text)

    def _do_ask_human(self, a: MetaAction, created_by: str) -> None:
        self.state.decisions.append(Decision(
            id=uuid.uuid4().hex[:10], kind="question", summary=a.text[:2000],
            payload={"question": a.text, "job_id": a.job_id},
        ))
        self.state.log("ask_human", a.job_id, a.text)

    def _do_finish(self, a: MetaAction, created_by: str) -> None:
        for job in self.state.active_jobs():
            if job.status != "finished":
                self._stop(job)
                job.status = "removed"
                job.note("meta", "Dropped: the work finished without it.")
        self.state.finished = True
        self.state.summary = a.text or None
        self.state.log("finish", None, a.text)

    def _do_fail(self, a: MetaAction, created_by: str) -> None:
        self._fail(a.text)

    # ── job state helpers ───────────────────────────────────────────────────

    def _get(self, job_id: str | None) -> Job:
        job = self.state.job(job_id or "")
        if job is None:
            raise ValueError(f"job {job_id!r} does not exist")
        return job

    def _answer(self, job: Job, text: str, *, source: str, answers: dict[str, Any] | None = None) -> None:
        job.note(source, f"Answer: {text}")
        if job.status == "running" and job.id in self._live:
            self._deliver(job, text)
        elif job.status in ("waiting", "pending"):
            questions = job.last.questions if job.last is not None else []
            job.pending_answers = answers or ({q: text for q in questions} if questions else {"answer": text})
            job.status, job.pending_reason = "pending", "answer"

    def _deliver(self, job: Job, text: str) -> None:
        attempt, child = self._live.pop(job.id)
        self._step.deliveries.append(Delivery(job_id=job.id, attempt=attempt, child_run_id=child, text=text))
        self.state.situations = [s for s in self.state.situations if not (s.kind == "live_question" and s.job_id == job.id)]

    def _reset(self, job: Job, reason: str) -> None:
        self._stop(job)
        job.status = "pending"
        job.pending_reason = reason
        self.state.situations = [s for s in self.state.situations if s.job_id != job.id]

    def _stop(self, job: Job) -> None:
        last = job.last
        if job.status == "running" and last is not None and last.status == "running":
            last.status = "cancelled"
            last.finished_at = _now()
            self._step.cancels.append(AttemptRef(job_id=job.id, attempt=last.n))
        self._live.pop(job.id, None)

    def _stop_all(self) -> None:
        for job in self.state.jobs:
            if job.status == "running":
                self._stop(job)
                job.status = "cancelled"
            elif job.status in ("pending", "waiting"):
                job.status = "cancelled"

    def _dependents(self, job_id: str) -> list[Job]:
        """Every active job downstream of *job_id*, transitively."""
        out: list[Job] = []
        frontier = [job_id]
        seen = {job_id}
        while frontier:
            current = frontier.pop()
            for j in self.state.jobs:
                if j.active and current in j.depends_on and j.id not in seen:
                    seen.add(j.id)
                    out.append(j)
                    frontier.append(j.id)
        return out

    def _fail(self, error: str) -> None:
        self.state.status = "failed"
        self.state.error = error
        self.state.log("failed", None, error)

    def _start_authored(self) -> None:
        """The author wrote the DAG: it is checked, not approved."""
        jobs = list(self.config.plan or [])
        errors = validate_jobs(jobs, self.config, []) + check_minimums(jobs, self.config)
        if errors:
            self._fail("the authored plan is invalid: " + "; ".join(errors))
            return
        for spec in jobs:
            self.state.jobs.append(Job(**spec.model_dump(), created_by="author"))
        self.state.status = "running"
        self.state.log("plan_authored", None, f"{len(jobs)} job(s)")

    def _phase_over(self) -> bool:
        """No further phase: the meta-agent finished, or the authored DAG is done."""
        if self.state.finished:
            return True
        return self.config.plan is not None and all(j.status in ("finished", "skipped") for j in self.state.active_jobs())

    # ── launching ───────────────────────────────────────────────────────────

    def _ready(self) -> list[Job]:
        ready = []
        for job in self.state.jobs:
            if job.status != "pending":
                continue
            deps = [self.state.job(d) for d in job.depends_on]
            if all(d is not None and d.status in ("finished", "skipped") for d in deps):
                ready.append(job)
        return ready

    def _launch_ready(self) -> list[Launch]:
        running = self.state.running()
        per_agent: dict[str, int] = {}
        for job in running:
            per_agent[job.agent_id] = per_agent.get(job.agent_id, 0) + 1
        launches: list[Launch] = []
        for job in self._ready():
            if len(running) + len(launches) >= self.config.limits.max_parallel:
                break
            entry = self.config.pool_entry(job.agent_id)
            cap = entry.max_instances if entry else 1
            if per_agent.get(job.agent_id, 0) >= cap:
                continue
            per_agent[job.agent_id] = per_agent.get(job.agent_id, 0) + 1
            fresh = job.new_notes()
            attempt = JobAttempt(
                n=len(job.attempts) + 1,
                reason=job.pending_reason or "initial",
                feedback="\n\n".join(f"[{n.source}] {n.text}" for n in fresh) or None,
            )
            payload = self._job_input(job, attempt)
            job.attempts.append(attempt)
            job.status = "running"
            if job.pending_answers:
                payload["clarification_context"] = job.pending_answers
            job.pending_reason = None
            job.pending_answers = None
            self.state.status = "running"
            self.state.log("attempt_started", job.id, f"attempt {attempt.n} ({attempt.reason})")
            launch = Launch(job=job.model_copy(deep=True), attempt=attempt.model_copy(), payload=payload)
            launches.append(launch)
            self._step.launches.append(launch)
        return launches

    def _job_input(self, job: Job, attempt: JobAttempt) -> dict[str, Any]:
        upstream = {}
        for dep_id in job.depends_on:
            dep = self.state.job(dep_id)
            if dep is not None and dep.output is not None:
                upstream[dep_id] = {"agent": dep.agent_id, "category": dep.category, "title": dep.title, "output": _truncate(dep.output)}
        payload: dict[str, Any] = {
            "request": self.request,
            "task": job.prompt,
            "job": {"id": job.id, "category": job.category, "title": job.title, "attempt": attempt.n, "owns": job.owns},
            "instructions": _INSTRUCTIONS[job.category],
        }
        if upstream:
            payload["upstream"] = upstream
        if attempt.feedback:
            payload["feedback"] = attempt.feedback
        if job.notes:
            # Everything the job was told so far, oldest first: a second rewind
            # does not hide the first one's reason.
            payload["notes"] = [f"[{n.source}] {n.text}" for n in job.notes[-20:]]
        previous = next((a for a in reversed(job.attempts) if a.output or a.error), None)
        if previous is not None:
            payload["previous_attempt"] = _truncate({"output": previous.output, "error": previous.error}, 6000)
        if job.datasources:
            payload["datasources"] = [
                {"source_id": d.source_id, "operations": d.operations, "description": d.description}
                for sid in job.datasources if (d := self.config.datasource(sid)) is not None
            ]
        ws = dict(self.workspace)
        if self.config.repo is not None:
            ws["repo"] = self.config.repo.url
            ws["integration_branch"] = f"carrier/{self.run_id[:8]}"
            ws["base_branch"] = self.config.repo.base_branch
            if job.category == "execution":
                ws["job_branch"] = f"carrier/{self.run_id[:8]}/{job.id}"
                ws["git"] = (
                    f"Branch {ws['job_branch']} from {ws['integration_branch']} if it exists, else from "
                    f"{ws['base_branch']}. Commit and push your branch; report it as `branch`."
                )
            if self.config.repo.verify:
                ws["verify"] = self.config.repo.verify
        if self.config.shared_volume is not None:
            mount = self.config.shared_volume.mount_point.rstrip("/")
            ws["shared_volume"] = mount
            ws["output_dir"] = f"{mount}/jobs/{job.id}/out"
            ws["volume_rules"] = (
                f"Read inputs anywhere under {mount}; write only under {mount}/jobs/{job.id}/. "
                "Report produced files as `artifacts`."
            )
        if ws:
            payload["workspace"] = ws
        return payload

    # ── the end ─────────────────────────────────────────────────────────────

    def _finish_or_fail(self) -> StepStatus:
        for job in self.state.jobs:
            if job.status != "pending":
                continue
            if any((d := self.state.job(dep)) is None or d.status in ("failed", "cancelled", "removed") for dep in job.depends_on):
                job.status = "skipped"
                self.state.log("skipped", job.id, "a dependency failed or was removed")
        active = self.state.active_jobs()
        unfinished = [j for j in active if j.status not in ("finished", "skipped")]
        if unfinished:
            self._fail("unfinished jobs: " + ", ".join(f"{j.id} ({j.status})" for j in unfinished))
            return "failed"
        missing = check_minimums(active, self.config)
        if missing:
            self._fail("; ".join(missing))
            return "failed"
        self.state.status = "completed"
        lines = [f"- {j.id} ({j.category}, {j.agent_id}): {str((j.output or {}).get('summary', '')).strip()[:500]}" for j in active]
        jobs_summary = "\n".join(lines)
        self.state.summary = f"{self.state.summary}\n\n{jobs_summary}" if self.state.summary else jobs_summary
        self.state.log("completed")
        return "completed"


def result_of(dag: DynamicRunState) -> dict[str, Any]:
    """What the dynamic step writes into workflow state."""
    return {
        "status": dag.status,
        "summary": dag.summary,
        "error": dag.error,
        "jobs": {
            j.id: {"category": j.category, "agent_id": j.agent_id, "status": j.status, "output": j.output}
            for j in dag.jobs if j.status != "removed"
        },
    }


def reset_for_retry(raw: dict[str, Any]) -> dict[str, Any]:
    """Make a failed dynamic step's record runnable again, keeping finished work.

    Failed, skipped, cancelled and waiting jobs go back to pending with one
    extra retry granted (recorded as an approved escalation, which is what
    ``_retries_left`` counts). Jobs the meta-agent removed stay removed.
    """
    state = DynamicRunState.model_validate(raw)
    for d in state.open_decisions():
        d.resolved, d.approved, d.reason = True, False, "superseded by a manual retry"
    for job in state.jobs:
        if job.status in ("failed", "skipped", "cancelled", "waiting", "running"):
            if job.last is not None and job.last.status == "running":
                job.last.status = "cancelled"
            job.status = "pending"
            job.awaiting = []
            job.pending_reason = "retry"
            state.decisions.append(Decision(
                id=uuid.uuid4().hex[:10], kind="escalation", summary=f"manual retry of {job.id}",
                payload={"job_id": job.id}, resolved=True, approved=True, reason="manual retry",
            ))
    state.situations = [s for s in state.situations if s.kind != "live_question"]
    state.status = "running" if state.jobs else "planning"
    state.finished = False
    state.error = None
    state.log("manual_retry")
    return state.model_dump(mode="json")
