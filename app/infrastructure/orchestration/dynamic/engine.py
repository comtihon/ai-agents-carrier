"""The dynamic engine: schedules jobs, applies handbacks/delegations, gates on approvals.

The engine is deliberately free of LangGraph and Mongo. It is handed:

- a ``Dispatcher`` (the meta-LLM),
- a ``job_runner`` that executes one job attempt and returns its outcome,
- a ``persist`` callback that stores ``DynamicRunState`` wherever the caller keeps it.

``advance()`` runs until the work is done, failed, or needs a human. In the
last case it stops dispatching, lets in-flight jobs finish (their results may
add decisions of their own), persists, and returns ``"gate"``. The caller then
pauses the run; when a human decides, ``advance(resolution)`` is called again
on a freshly loaded state — every pass starts from the persisted record, which
is why a backend restart mid-run loses nothing but the attempts in flight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.models.dynamic import (
    Decision,
    DynamicConfig,
    DynamicRunState,
    Job,
    JobAttempt,
)
from app.infrastructure.orchestration.dynamic.dispatcher import (
    Dispatcher,
    DispatcherError,
    ProposedJob,
    check_minimums,
    validate_jobs,
)

logger = logging.getLogger(__name__)

AdvanceResult = Literal["completed", "gate", "failed"]


class JobOutcome(BaseModel):
    status: Literal["finished", "failed", "needs_input"]
    output: dict[str, Any] | None = None
    error: str | None = None
    questions: list[str] = Field(default_factory=list)


JobRunner = Callable[[Job, JobAttempt, dict[str, Any], Callable[[str], Awaitable[None]]], Awaitable[JobOutcome]]
Persist = Callable[[DynamicRunState], Awaitable[None]]


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
        "You are a planning job in a multi-agent workflow. Research and/or design; do not implement. "
        "Return `summary` and `plan` (a concrete breakdown: parts of the work, what each part may change, "
        "dependencies between parts, how to verify). If you need information another agent should gather, "
        "return `delegate` with a precise request instead of guessing."
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
        "what must change) and `handback_to` (the list of upstream job ids that must redo their part)."
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


def _delegate_request(output: dict[str, Any] | None) -> str | None:
    if not isinstance(output, dict):
        return None
    raw = output.get("delegate")
    if raw in (None, "", [], {}, False):
        return None
    if isinstance(raw, dict):
        return raw.get("request") or json.dumps(raw, ensure_ascii=False)
    return str(raw)


class DynamicEngine:
    def __init__(
        self,
        *,
        config: DynamicConfig,
        state: DynamicRunState,
        dispatcher: Dispatcher,
        roster: str,
        request: str,
        job_runner: JobRunner,
        persist: Persist,
        run_id: str,
        workspace: dict[str, Any] | None = None,
        stop_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self.config = config
        self.state = state
        self.dispatcher = dispatcher
        self.roster = roster
        self.request = request
        self.job_runner = job_runner
        self._persist = persist
        self.run_id = run_id
        self.workspace = workspace or {}
        # Asked between scheduling rounds: True when the run was stopped from
        # outside (terminate), so no further job is dispatched.
        self._stop_check = stop_check
        self._tasks: dict[asyncio.Task, str] = {}

    # ── persistence ─────────────────────────────────────────────────────────

    async def persist(self) -> None:
        for k, v in self.dispatcher.usage.items():
            self.state.usage[k] = v
        await self._persist(self.state)

    # ── decisions ───────────────────────────────────────────────────────────

    def _decide(self, kind: str, summary: str, payload: dict[str, Any]) -> Decision:
        decision = Decision(id=uuid.uuid4().hex[:10], kind=kind, summary=summary, payload=payload)  # type: ignore[arg-type]
        self.state.decisions.append(decision)
        self.state.log(f"decision:{kind}", payload.get("job_id"), summary)
        return decision

    async def _propose(self, kind: str, summary: str, payload: dict[str, Any]) -> None:
        """Record a decision and apply it unless the automation mode gates it."""
        decision = self._decide(kind, summary, payload)
        if not self.config.needs_approval(kind):
            decision.resolved = True
            decision.approved = True
            decision.reason = f"auto ({self.config.automation})"
            await self._apply(decision, approved=True, reason=None, corrections={})

    async def _resolve(self, resolution: dict[str, Any]) -> None:
        approved = bool(resolution.get("approved", False))
        reason = resolution.get("reason")
        corrections = resolution.get("corrections") or {}
        for decision in self.state.open_decisions():
            decision.resolved = True
            decision.approved = approved
            decision.reason = reason
            await self._apply(decision, approved=approved, reason=reason, corrections=corrections)

    async def _apply(self, d: Decision, *, approved: bool, reason: str | None, corrections: dict[str, Any]) -> None:
        p = d.payload
        if d.kind == "plan":
            if approved:
                raw_jobs = corrections.get("jobs") or p.get("jobs") or []
                proposed = [ProposedJob.model_validate(j) for j in raw_jobs]
                existing = [j for j in self.state.jobs]
                errors = validate_jobs(proposed, self.config, existing)
                if errors:
                    # An edited plan that breaks the rules is treated as a rejection with the reasons.
                    await self._replan(p, "; ".join(errors))
                    return
                self._add_jobs(proposed, created_by="human" if corrections.get("jobs") else ("planner" if p.get("expansion") else "dispatcher"))
                if p.get("expansion"):
                    self.state.expanded = True
                self.state.status = "running"
            else:
                await self._replan(p, reason or "rejected without a reason")
        elif d.kind == "handback":
            if approved:
                feedback = corrections.get("feedback") or p.get("feedback") or ""
                self._handback(p["targets"], feedback, by=p["by"])
            else:
                self.state.log("handback_declined", p.get("by"), reason or "")
        elif d.kind == "delegate":
            requester = self.state.job(p["job_id"])
            if requester is None:
                return
            if approved:
                job = Job(
                    id=self._unique_id(f"{requester.id}-help"),
                    category=p["category"],
                    agent_id=p["agent_id"],
                    title=p.get("title") or f"help for {requester.id}",
                    prompt=corrections.get("prompt") or p["prompt"],
                    delegated_by=requester.id,
                    created_by="delegation",
                )
                self.state.jobs.append(job)
                self.state.delegations += 1
                requester.status = "waiting"
                requester.awaiting = [job.id]
            else:
                requester.status = "pending"
                requester.pending_reason = "delegate_declined"
                requester.pending_feedback = (
                    f"Your delegation request was declined{': ' + reason if reason else ''}. "
                    "Continue with what you have."
                )
        elif d.kind == "escalation":
            job = self.state.job(p["job_id"])
            if job is None:
                return
            if approved and p.get("targets"):
                # A validator that keeps rejecting: one more handback round for its targets.
                self._handback(p["targets"], corrections.get("feedback") or p.get("error") or "", by=job.id)
            elif approved:
                # One more attempt beyond the budget (counted in _attempts_left).
                job.status = "pending"
                job.pending_reason = "retry"
                job.pending_feedback = corrections.get("feedback") or p.get("error")
            else:
                job.status = "failed"
                self._fail(f"job {job.id} failed and the escalation was rejected: {reason or p.get('error')}")
        elif d.kind == "question":
            job = self.state.job(p["job_id"])
            if job is None:
                return
            job.status = "pending"
            job.pending_reason = "answer"
            if approved:
                answers = corrections.get("answers") or {}
                if not answers and reason:
                    answers = {"answer": reason}
                job.pending_answers = answers or {"answer": "proceed with your best judgement"}
            else:
                job.pending_feedback = "Nobody can answer your questions. Proceed with your best judgement and say what you assumed."

    async def _replan(self, payload: dict[str, Any], feedback: str) -> None:
        if self.state.replans >= self.config.limits.max_replans:
            self._fail(f"plan rejected {self.state.replans + 1} times; last reason: {feedback}")
            return
        self.state.replans += 1
        previous = [Job(**{**j, "status": "pending"}) for j in payload.get("jobs") or [] if isinstance(j, dict) and "prompt" in j]
        try:
            if payload.get("expansion"):
                plan = await self.dispatcher.expand(
                    self.request + f"\n\nA human rejected the previous breakdown: {feedback}",
                    self.config, self.roster, self.state.jobs,
                )
            else:
                plan = await self.dispatcher.plan(self.request, self.config, self.roster, feedback=feedback, previous=previous)
        except DispatcherError as exc:
            self._fail(str(exc))
            return
        await self._propose(
            "plan",
            plan.summary or "revised plan",
            {"jobs": [j.model_dump() for j in plan.jobs], "summary": plan.summary, "expansion": bool(payload.get("expansion"))},
        )

    # ── job graph helpers ───────────────────────────────────────────────────

    def _unique_id(self, base: str) -> str:
        base = base[:40]
        ids = {j.id for j in self.state.jobs}
        if base not in ids:
            return base
        n = 2
        while f"{base}-{n}" in ids:
            n += 1
        return f"{base}-{n}"

    def _add_jobs(self, proposed: list[ProposedJob], created_by: str) -> None:
        for pj in proposed:
            self.state.jobs.append(Job(**pj.model_dump(), created_by=created_by))  # type: ignore[arg-type]
            self.state.log("job_added", pj.id, f"{pj.category} → {pj.agent_id}")

    def _dependents(self, job_id: str) -> list[Job]:
        """Every job downstream of *job_id*, transitively."""
        out: list[Job] = []
        frontier = [job_id]
        seen = {job_id}
        while frontier:
            current = frontier.pop()
            for j in self.state.jobs:
                if current in j.depends_on and j.id not in seen:
                    seen.add(j.id)
                    out.append(j)
                    frontier.append(j.id)
        return out

    def _handback(self, targets: list[str], feedback: str, by: str) -> None:
        for tid in targets:
            target = self.state.job(tid)
            if target is None:
                continue
            target.status = "pending"
            target.pending_reason = "handback"
            target.pending_feedback = f"Validator {by} handed this back:\n{feedback}"
            for dep in self._dependents(tid):
                if dep.status in ("finished", "failed", "skipped"):
                    dep.status = "pending"
                    dep.pending_reason = dep.pending_reason or "upstream_changed"
        validator = self.state.job(by)
        if validator is not None and validator.status != "pending":
            validator.status = "pending"
            validator.pending_reason = "upstream_changed"

    def _attempts_left(self, job: Job) -> bool:
        used = max(len(job.attempts) - 1, 0)
        granted = sum(
            1 for d in self.state.decisions
            if d.kind == "escalation" and d.approved and (
                job.id in (d.payload.get("targets") or [])
                or (d.payload.get("job_id") == job.id and not d.payload.get("targets"))
            )
        )
        return used < self.config.limits.max_handbacks + granted

    def _fail(self, error: str) -> None:
        self.state.status = "failed"
        self.state.error = error
        self.state.log("failed", None, error)

    def _ready(self) -> list[Job]:
        ready = []
        for job in self.state.jobs:
            if job.status != "pending" or job.awaiting:
                continue
            deps = [self.state.job(d) for d in job.depends_on]
            if all(d is not None and d.status in ("finished", "skipped") for d in deps):
                ready.append(job)
        return ready

    def _running_count(self, agent_id: str | None = None) -> int:
        return sum(
            1 for job_id in self._tasks.values()
            if agent_id is None or (self.state.job(job_id) and self.state.job(job_id).agent_id == agent_id)  # type: ignore[union-attr]
        )

    def _blocked(self) -> list[Job]:
        """Pending jobs that can never become ready because a dependency failed."""
        blocked = []
        for job in self.state.jobs:
            if job.status != "pending":
                continue
            for dep_id in job.depends_on:
                dep = self.state.job(dep_id)
                if dep is None or dep.status in ("failed", "cancelled"):
                    blocked.append(job)
                    break
        return blocked

    # ── job input ───────────────────────────────────────────────────────────

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
        if job.delegated_by:
            payload["delegated_by"] = job.delegated_by
        # Results of delegations this job asked for.
        helpers = [j for j in self.state.jobs if j.delegated_by == job.id and j.output is not None]
        if helpers:
            payload["delegation_results"] = {h.id: _truncate(h.output) for h in helpers}
        if attempt.feedback:
            payload["feedback"] = attempt.feedback
        previous = next((a for a in reversed(job.attempts[:-1]) if a.output or a.error), None)
        if previous is not None:
            payload["previous_attempt"] = _truncate({"output": previous.output, "error": previous.error}, 6000)
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

    # ── main loop ───────────────────────────────────────────────────────────

    async def advance(self, resolution: dict[str, Any] | None = None) -> AdvanceResult:
        if self.state.status in ("completed", "failed"):
            return self.state.status  # type: ignore[return-value]

        # Attempts that were in flight when the previous pass ended (restart,
        # cancellation) never reported back: retry them.
        for job in self.state.jobs:
            if job.status == "running":
                if job.last is not None and job.last.status == "running":
                    job.last.status = "cancelled"
                    job.last.finished_at = _now()
                job.status = "pending"
                job.pending_reason = job.pending_reason or "retry"
                self.state.log("attempt_lost", job.id, "attempt was in flight when the run paused or restarted")

        if resolution is not None and self.state.open_decisions():
            await self._resolve(resolution)
            await self.persist()

        if not self.state.jobs and not self.state.open_decisions() and self.state.status == "planning":
            try:
                plan = await self.dispatcher.plan(self.request, self.config, self.roster)
            except DispatcherError as exc:
                self._fail(str(exc))
                await self.persist()
                return "failed"
            await self._propose(
                "plan",
                plan.summary or "initial plan",
                {"jobs": [j.model_dump() for j in plan.jobs], "summary": plan.summary, "expansion": False},
            )
            await self.persist()

        try:
            return await self._loop()
        except asyncio.CancelledError:
            for task in list(self._tasks):
                task.cancel()
            raise

    async def _loop(self) -> AdvanceResult:
        while True:
            if self._stop_check is not None and await self._stop_check():
                for task in list(self._tasks):
                    task.cancel()
                if self._tasks:
                    await asyncio.wait(list(self._tasks))
                    self._tasks.clear()
                for job in self.state.jobs:
                    if job.status == "running":
                        job.status = "cancelled"
                        if job.last is not None:
                            job.last.status = "cancelled"
                self._fail("run was stopped")
                await self.persist()
                return "failed"
            if self.state.status == "failed":
                await self._drain()
                await self.persist()
                return "failed"

            gated = bool(self.state.open_decisions())

            if not gated:
                await self._maybe_expand()
                gated = bool(self.state.open_decisions())
                if self.state.status == "failed":
                    continue

            if not gated:
                self._dispatch_ready()

            if not self._tasks:
                if gated:
                    self.state.status = "waiting_approval"
                    await self.persist()
                    return "gate"
                return await self._finish_or_fail()

            done, _ = await asyncio.wait(list(self._tasks), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                job_id = self._tasks.pop(task)
                await self._on_done(job_id, task)
            await self.persist()

    async def _drain(self) -> None:
        if not self._tasks:
            return
        done, _ = await asyncio.wait(list(self._tasks))
        for task in done:
            job_id = self._tasks.pop(task)
            await self._on_done(job_id, task)

    async def _maybe_expand(self) -> None:
        if self.state.expanded:
            return
        planning = [j for j in self.state.jobs if j.category == "planning" and not j.delegated_by]
        non_planning = [j for j in self.state.jobs if j.category != "planning" and not j.delegated_by]
        if not planning or non_planning:
            # No planner (dispatcher planned everything), or work already exists.
            if self.state.jobs:
                self.state.expanded = True
            return
        if not all(j.status == "finished" for j in planning):
            return
        try:
            plan = await self.dispatcher.expand(self.request, self.config, self.roster, self.state.jobs)
        except DispatcherError as exc:
            self._fail(str(exc))
            return
        await self._propose(
            "plan",
            plan.summary or "jobs from the planners' output",
            {"jobs": [j.model_dump() for j in plan.jobs], "summary": plan.summary, "expansion": True},
        )

    def _dispatch_ready(self) -> None:
        for job in self._ready():
            if self._running_count() >= self.config.limits.max_parallel:
                break
            entry = self.config.pool_entry(job.agent_id)
            cap = entry.max_instances if entry else 1
            if self._running_count(job.agent_id) >= cap:
                continue
            attempt = JobAttempt(
                n=len(job.attempts) + 1,
                reason=job.pending_reason or "initial",
                feedback=job.pending_feedback,
            )
            job.attempts.append(attempt)
            job.status = "running"
            payload = self._job_input(job, attempt)
            if job.pending_answers:
                payload["clarification_context"] = job.pending_answers
            job.pending_reason = None
            job.pending_feedback = None
            job.pending_answers = None
            self.state.status = "running"
            self.state.log("attempt_started", job.id, f"attempt {attempt.n} ({attempt.reason})")

            async def _started(child_run_id: str, _a: JobAttempt = attempt) -> None:
                _a.child_run_id = child_run_id
                await self.persist()

            task = asyncio.create_task(self._run_job(job, attempt, payload, _started))
            self._tasks[task] = job.id

    async def _run_job(self, job: Job, attempt: JobAttempt, payload: dict[str, Any], started) -> JobOutcome:
        try:
            return await self.job_runner(job, attempt, payload, started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the runner should not raise, but a crash is just a failed attempt
            logger.exception("dynamic job %s crashed", job.id)
            return JobOutcome(status="failed", error=f"{type(exc).__name__}: {exc}")

    async def _on_done(self, job_id: str, task: asyncio.Task) -> None:
        job = self.state.job(job_id)
        if job is None:
            return
        attempt = job.last
        try:
            outcome: JobOutcome = task.result()
        except asyncio.CancelledError:
            outcome = JobOutcome(status="failed", error="cancelled")
        if attempt is not None:
            attempt.finished_at = _now()
            attempt.output = outcome.output
            attempt.error = outcome.error
            attempt.status = {"finished": "finished", "failed": "failed", "needs_input": "needs_input"}[outcome.status]  # type: ignore[assignment]

        if outcome.status == "needs_input":
            job.status = "waiting"
            if self.config.needs_approval("question"):
                self._decide("question", f"{job.id} asks: " + " / ".join(outcome.questions)[:500], {"job_id": job.id, "questions": outcome.questions})
            else:
                job.status = "pending"
                job.pending_reason = "answer"
                job.pending_feedback = "Nobody will answer questions in this run. Proceed with your best judgement and state your assumptions."
            return

        if outcome.status == "failed":
            self.state.log("attempt_failed", job.id, outcome.error or "")
            if self._attempts_left(job):
                job.status = "pending"
                job.pending_reason = "retry"
                job.pending_feedback = f"Your previous attempt failed: {outcome.error}"
                return
            job.status = "failed"
            if self.config.automation == "bypass":
                self._fail(f"job {job.id} failed after {len(job.attempts)} attempts: {outcome.error}")
                return
            job.status = "waiting"
            await self._propose("escalation", f"{job.id} keeps failing: {(outcome.error or '')[:300]}", {"job_id": job.id, "error": outcome.error})
            return

        # finished
        job.status = "finished"
        self.state.log("attempt_finished", job.id, str((outcome.output or {}).get("summary", ""))[:300])

        # A delegated job returns its result to whoever asked.
        if job.delegated_by:
            requester = self.state.job(job.delegated_by)
            if requester is not None and job.id in requester.awaiting:
                requester.awaiting.remove(job.id)
                if not requester.awaiting:
                    requester.status = "pending"
                    requester.pending_reason = "delegate_return"
                    requester.pending_feedback = f"The help you asked for is in `delegation_results` ({job.id})."

        ask = _delegate_request(outcome.output)
        if ask:
            if self.state.delegations >= self.config.limits.max_delegations:
                self.state.log("delegate_refused", job.id, "delegation budget exhausted")
            else:
                await self._delegate(job, ask)
                return

        if job.category == "validation" and _verdict_failed(outcome.output):
            output = outcome.output or {}
            targets = [t for t in _as_list(output.get("handback_to")) if self.state.job(t) is not None and t != job.id]
            if not targets:
                targets = [d for d in job.depends_on if (dj := self.state.job(d)) is not None and dj.category in ("execution", "integration")]
            if not targets:
                self.state.log("verdict_fail_no_target", job.id, str(output.get("feedback", ""))[:300])
                return
            exhausted = [t for t in targets if not self._attempts_left(self.state.job(t))]  # type: ignore[arg-type]
            feedback = str(output.get("feedback") or output.get("summary") or "validation failed")
            if exhausted:
                if self.config.automation == "bypass":
                    self._fail(f"validator {job.id} still failing {exhausted} after {self.config.limits.max_handbacks} handbacks: {feedback[:500]}")
                    return
                job.status = "waiting"
                await self._propose(
                    "escalation",
                    f"{job.id} still rejects {', '.join(exhausted)} after the handback budget",
                    {"job_id": job.id, "error": feedback, "targets": targets},
                )
                return
            await self._propose(
                "handback",
                f"{job.id} hands back to {', '.join(targets)}: {feedback[:300]}",
                {"by": job.id, "targets": targets, "feedback": feedback},
            )

    async def _delegate(self, job: Job, ask: str) -> None:
        try:
            proposal = await self.dispatcher.assign_delegate(self.request, self.config, self.roster, job, ask, self.state.jobs)
        except DispatcherError as exc:
            self.state.log("delegate_failed", job.id, str(exc))
            return
        entry = self.config.pool_entry(proposal.agent_id)
        category = "planning"
        if entry is not None and entry.categories and "planning" not in entry.categories:
            category = entry.categories[0]
        job.status = "waiting"
        await self._propose(
            "delegate",
            f"{job.id} asks {proposal.agent_id}: {ask[:300]}",
            {"job_id": job.id, "request": ask, "agent_id": proposal.agent_id, "title": proposal.title, "prompt": proposal.prompt, "category": category},
        )

    async def _finish_or_fail(self) -> AdvanceResult:
        blocked = self._blocked()
        for job in blocked:
            job.status = "skipped"
            self.state.log("skipped", job.id, "a dependency failed")
        failed = [j for j in self.state.jobs if j.status == "failed"]
        stuck = [j for j in self.state.jobs if j.status in ("pending", "waiting")]
        if failed or stuck:
            self._fail(
                "unfinished jobs: "
                + ", ".join(f"{j.id} ({j.status})" for j in failed + stuck)
            )
            await self.persist()
            return "failed"
        missing = check_minimums(self.state.jobs, self.config)
        if missing:
            self._fail("; ".join(missing))
            await self.persist()
            return "failed"
        self.state.status = "completed"
        lines = [f"- {j.id} ({j.category}, {j.agent_id}): {str((j.output or {}).get('summary', '')).strip()[:500]}" for j in self.state.jobs]
        self.state.summary = "\n".join(lines)
        self.state.log("completed")
        await self.persist()
        return "completed"

    def result(self) -> dict[str, Any]:
        """What the dynamic step writes into workflow state."""
        return {
            "status": self.state.status,
            "summary": self.state.summary,
            "error": self.state.error,
            "jobs": {j.id: {"category": j.category, "agent_id": j.agent_id, "status": j.status, "output": j.output} for j in self.state.jobs},
        }


def reset_for_retry(raw: dict[str, Any]) -> dict[str, Any]:
    """Make a failed dynamic step's record runnable again, keeping finished work.

    Failed, skipped and cancelled jobs go back to pending with one extra
    attempt granted (recorded as an approved escalation, which is what
    ``_attempts_left`` counts), so a job that had used up its budget gets a
    real retry instead of failing again on the spot.
    """
    state = DynamicRunState.model_validate(raw)
    for d in state.open_decisions():
        d.resolved, d.approved, d.reason = True, False, "superseded by a manual retry"
    for job in state.jobs:
        if job.status in ("failed", "skipped", "cancelled", "waiting", "running"):
            job.status = "pending"
            job.awaiting = []
            job.pending_reason = "retry"
            state.decisions.append(Decision(
                id=uuid.uuid4().hex[:10], kind="escalation", summary=f"manual retry of {job.id}",
                payload={"job_id": job.id}, resolved=True, approved=True, reason="manual retry",
            ))
    state.status = "running" if state.jobs else "planning"
    state.error = None
    state.log("manual_retry")
    return state.model_dump(mode="json")
