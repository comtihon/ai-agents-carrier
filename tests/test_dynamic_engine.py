"""The job graph: a meta-agent orchestrating jobs on LangGraph, driven by events.

Every test runs the real LangGraph job graph (``build_job_graph``) with a
MemorySaver, a ``JobHub`` running the attempts in the background, a scripted
meta-agent and a scripted job runner.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.domain.models.dynamic import DynamicConfig, DynamicRunState, Job, JobAttempt
from app.infrastructure.orchestration.dynamic.dispatcher import ProposedJob, validate_jobs
from app.infrastructure.orchestration.dynamic.engine import AttemptResult, JobOutcome
from app.infrastructure.orchestration.dynamic.graph import build_job_graph, initial_input
from app.infrastructure.orchestration.dynamic.hub import JobHub
from app.infrastructure.orchestration.dynamic.meta_agent import LlmMetaAgent, MetaDecision, MetaView


def _cfg(**over: Any) -> DynamicConfig:
    base = {
        "agent_pool": [
            {"agent_id": "planner", "categories": ["planning"]},
            {"agent_id": "researcher", "max_instances": 2, "categories": ["planning"]},
            {"agent_id": "coder", "max_instances": 3, "categories": ["execution", "integration"]},
            {"agent_id": "tester", "max_instances": 2, "categories": ["validation"]},
        ],
        "datasources": [{"source_id": "crm", "operations": ["get_account"], "description": "customer accounts"}],
        "automation": "auto",
    }
    base.update(over)
    return DynamicConfig.model_validate(base)


def _job(id: str, category: str, agent: str, deps: list[str] | None = None, **extra: Any) -> dict:
    return {"id": id, "category": category, "agent_id": agent, "title": id, "prompt": f"do {id}", "depends_on": deps or [], **extra}


def add(*jobs: dict) -> dict:
    return {"kind": "add_jobs", "jobs": list(jobs)}


def finish(text: str = "done") -> dict:
    return {"kind": "finish", "text": text}


Script = dict | list | Callable[[MetaView], Any]


class ScriptedMeta:
    """A meta-agent answering from a script; records what it was shown."""

    def __init__(self, *answers: Script) -> None:
        self.answers = list(answers)
        self.views: list[MetaView] = []
        self.usage: dict[str, int] = {}

    async def decide(self, view: MetaView, validate) -> MetaDecision:
        self.views.append(view.situations and MetaView(view.request, view.config, view.roster, view.dag.model_copy(deep=True), list(view.situations)))
        if not self.answers:
            raise AssertionError(f"meta-agent asked more than scripted: {[s.kind for s in view.situations]}")
        answer = self.answers.pop(0)
        data = answer(view) if callable(answer) else answer
        if not (isinstance(data, dict) and "actions" in data):
            data = {"summary": "scripted", "actions": data if isinstance(data, list) else [data]}
        decision = MetaDecision.model_validate(data)
        errors = validate(decision)
        assert not errors, errors
        self.usage = {"total_tokens": self.usage.get("total_tokens", 0) + 10}
        return decision

    def kinds(self) -> list[list[str]]:
        return [[s.kind for s in v.situations] for v in self.views]


class FakeRunner:
    """Scripted outcomes per job id (consumed per attempt); ``hold`` keeps a job running until released."""

    def __init__(self, script: dict[str, list[JobOutcome]] | None = None, delay: float = 0.0) -> None:
        self.script = script or {}
        self.delay = delay
        self.calls: list[tuple[str, int, dict]] = []
        self.cancelled: list[tuple[str, int]] = []
        self.holds: dict[str, asyncio.Event] = {}
        self.started: dict[str, asyncio.Event] = {}
        self.concurrent = 0
        self.max_concurrent = 0

    def hold(self, job_id: str) -> asyncio.Event:
        self.holds[job_id] = asyncio.Event()
        self.started[job_id] = asyncio.Event()
        return self.holds[job_id]

    async def __call__(self, job: Job, attempt: JobAttempt, payload: dict, started) -> JobOutcome:
        await started(f"child-{job.id}-{attempt.n}")
        self.calls.append((job.id, attempt.n, payload))
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if job.id in self.started:
                self.started[job.id].set()
            if job.id in self.holds:
                await self.holds[job.id].wait()
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled.append((job.id, attempt.n))
            raise
        finally:
            self.concurrent -= 1
        queue = self.script.get(job.id)
        if queue:
            return queue.pop(0)
        return JobOutcome(status="finished", output={"summary": f"{job.id} done"})

    def ran(self, job_id: str) -> int:
        return sum(1 for c in self.calls if c[0] == job_id)

    def payload(self, job_id: str, n: int) -> dict:
        return next(c[2] for c in self.calls if c[0] == job_id and c[1] == n)


def F(**output: Any) -> JobOutcome:
    return JobOutcome(status="finished", output=output)


class Harness:
    """One dynamic step's job graph on its own thread; ``advance`` runs it to the next pause."""

    def __init__(self, cfg: DynamicConfig, meta: Any, runner: FakeRunner, state: DynamicRunState | None = None, **kw: Any) -> None:
        self.persisted: list[dict] = []
        self.attempt_events: list[tuple[str, int, str | None, str | None]] = []
        self.delivered: list[tuple[str, str]] = []
        self.hub = JobHub()

        async def on_attempt(job_id: str, n: int, child: str | None, result: AttemptResult | None) -> None:
            self.attempt_events.append((job_id, n, child, result.status if result else None))

        async def deliver(child_run_id: str, text: str) -> None:
            self.delivered.append((child_run_id, text))

        self.hub.configure(job_runner=runner, on_attempt=on_attempt, recorded=kw.pop("recorded", None), deliver_answer=deliver)

        async def persist(s: DynamicRunState) -> None:
            self.persisted.append(s.model_dump(mode="json"))

        self.graph = build_job_graph(
            config=cfg, meta=meta, roster="(roster)", request="build features A and B", run_id="12345678-run",
            hub=self.hub, persist=persist, checkpointer=MemorySaver(), wake_seconds=0.05, **kw,
        )
        self.config = {"configurable": {"thread_id": "t"}, "recursion_limit": 10_000}
        self.initial = state or DynamicRunState(step_id="orchestrator")
        self.started = False
        self.state = self.initial
        self.approvals: list[dict] = []

    async def advance(self, resolution: dict[str, Any] | None = None) -> str:
        if not self.started:
            self.started = True
            out = await asyncio.wait_for(self.graph.ainvoke(initial_input(self.initial), self.config), 10)
        else:
            out = await asyncio.wait_for(self.graph.ainvoke(Command(resume=resolution), self.config), 10)
        self.state = DynamicRunState.model_validate(out["dag"])
        self.approvals = out.get("approvals") or []
        if out.get("__interrupt__"):
            return "gate"
        return self.state.status

    def job(self, job_id: str) -> Job:
        job = self.state.job(job_id)
        assert job is not None, job_id
        return job


# ─── config and validation ───────────────────────────────────────────────────


def test_config_defaults_and_rules():
    cfg = DynamicConfig.from_step({"id": "x", "type": "dynamic", "next": "END", "agent_pool": [{"agent_id": "a"}], "jobs": {"validation": {"min": 1, "max": 2}}})
    assert cfg.jobs["execution"].min == 0 and cfg.jobs["validation"].min == 1
    assert DynamicConfig.model_validate({"limits": {"max_handbacks": 4}}).limits.max_retries == 4  # old name still read
    with pytest.raises(ValueError):
        DynamicConfig.model_validate({"agent_pool": [{"agent_id": "a"}, {"agent_id": "a"}]})
    with pytest.raises(ValueError):
        DynamicConfig.model_validate({"plan": []})


def test_approval_policy_by_mode():
    assert _cfg(automation="ask").needs_approval("rearrange")
    assert _cfg(automation="plan").needs_approval("plan")
    assert not _cfg(automation="plan").needs_approval("rearrange")
    assert _cfg(automation="auto").needs_approval("question")
    assert not _cfg(automation="bypass").needs_approval("escalation")


def test_validate_jobs_catches_pool_category_cycle_datasource_and_slot_violations():
    cfg = _cfg(jobs={"execution": {"min": 1, "max": 2}})
    jobs = [
        ProposedJob(**_job("a", "execution", "coder", ["b"])),
        ProposedJob(**_job("b", "execution", "coder", ["a"])),
        ProposedJob(**_job("c", "execution", "coder", datasources=["billing"])),
        ProposedJob(**_job("d", "validation", "coder")),
        ProposedJob(**_job("e", "execution", "ghost")),
    ]
    errors = " | ".join(validate_jobs(jobs, cfg, []))
    assert "cycle" in errors
    assert "may only fill" in errors
    assert "not in the pool" in errors
    assert "data source 'billing' is not enabled" in errors
    assert "too many execution jobs" in errors


# ─── orchestration ───────────────────────────────────────────────────────────


async def test_understood_request_is_planned_in_full_and_finished_by_the_meta_agent():
    meta = ScriptedMeta(
        add(_job("code-a", "execution", "coder", owns=["a/**"]), _job("code-b", "execution", "coder", owns=["b/**"]),
            _job("test", "validation", "tester", ["code-a", "code-b"])),
        finish("A and B are built and tested"),
    )
    runner = FakeRunner(delay=0.01)
    h = Harness(_cfg(), meta, runner)
    assert await h.advance() == "completed"
    assert meta.kinds() == [["start"], ["phase_done"]]
    assert [c[0] for c in runner.calls][2] == "test" and runner.max_concurrent == 2
    assert set(runner.payload("test", 1)["upstream"]) == {"code-a", "code-b"}
    assert h.state.summary.startswith("A and B are built and tested")
    assert h.state.usage["total_tokens"] == 20


async def test_unclear_request_gathers_information_first_then_plans_the_next_phase():
    def plan_from_findings(view: MetaView) -> dict:
        assert view.dag.job("gather").output == {"summary": "gather done"}
        return add(
            _job("research", "planning", "researcher", ["gather"]),
            _job("code", "execution", "coder", ["research"], datasources=["crm"]),
            _job("review", "validation", "tester", ["code"]),
        )

    meta = ScriptedMeta(add(_job("gather", "planning", "researcher", datasources=["crm"])), plan_from_findings, finish())
    runner = FakeRunner()
    h = Harness(_cfg(), meta, runner)
    assert await h.advance() == "completed"
    assert meta.kinds() == [["start"], ["phase_done"], ["phase_done"]]
    assert [c[0] for c in runner.calls] == ["gather", "research", "code", "review"]
    assert runner.payload("code", 1)["datasources"] == [{"source_id": "crm", "operations": ["get_account"], "description": "customer accounts"}]


async def test_a_failed_part_is_retried_alone_without_the_meta_agent():
    meta = ScriptedMeta(add(*[_job(f"code-{p}", "execution", "coder", owns=[f"{p}/**"]) for p in "abc"]), finish())
    runner = FakeRunner({"code-b": [JobOutcome(status="failed", error="pod crashed")]})
    h = Harness(_cfg(), meta, runner)
    assert await h.advance() == "completed"
    assert (runner.ran("code-a"), runner.ran("code-b"), runner.ran("code-c")) == (1, 2, 1)
    assert "pod crashed" in runner.payload("code-b", 2)["feedback"]
    assert meta.kinds() == [["start"], ["phase_done"]]


async def test_failures_beyond_the_retries_go_to_the_meta_agent():
    def give_up(view: MetaView) -> dict:
        assert view.situations[0].kind == "failure" and view.situations[0].detail["error"] == "boom"
        return {"kind": "fail", "text": "the coder cannot build it"}

    meta = ScriptedMeta(add(_job("code", "execution", "coder")), give_up)
    runner = FakeRunner({"code": [JobOutcome(status="failed", error="boom")] * 3})
    h = Harness(_cfg(limits={"max_retries": 2}), meta, runner)
    assert await h.advance() == "failed"
    assert runner.ran("code") == 3 and h.state.error == "the coder cannot build it"


# ─── going back in the graph ─────────────────────────────────────────────────


async def test_change_request_rewinds_back_to_the_researcher_and_everything_after_reruns():
    def rewind_to_research(view: MetaView) -> dict:
        s = view.situations[0]
        assert s.kind == "change_request" and s.job_id == "review"
        assert s.detail["feedback"] == "wrong approach, use streaming" and s.detail["suggested_targets"] == ["code"]
        return {"kind": "rewind", "job_ids": ["research"], "text": "Find out how X streams; the validator rejected the batch approach."}

    meta = ScriptedMeta(
        add(_job("research", "planning", "researcher"), _job("code", "execution", "coder", ["research"]),
            _job("review", "validation", "tester", ["code"])),
        rewind_to_research,
        finish(),
    )
    runner = FakeRunner({"review": [F(verdict="fail", feedback="wrong approach, use streaming", handback_to=["code"]), F(verdict="pass")]})
    h = Harness(_cfg(), meta, runner)
    assert await h.advance() == "completed"
    assert [c[0] for c in runner.calls] == ["research", "code", "review", "research", "code", "review"]
    assert "how X streams" in runner.payload("research", 2)["feedback"]
    assert any("Upstream job research is being redone" in n for n in runner.payload("code", 2)["notes"])
    assert h.job("code").attempts[1].reason == "upstream_changed"
    assert h.state.rearrangements == 1


async def test_coder_needs_info_meta_adds_a_research_job_and_the_critique_survives():
    """Validator rejects -> coder rewound -> coder asks for info -> research added -> coder sees both notes."""
    meta = ScriptedMeta(
        add(_job("code", "execution", "coder"), _job("review", "validation", "tester", ["code"])),
        {"kind": "rewind", "job_ids": ["code"], "text": "wrong approach, use streaming"},
        [
            add(_job("dig", "planning", "researcher")),
            {"kind": "update_job", "job_id": "code", "changes": {"depends_on": ["dig"]}},
            {"kind": "rewind", "job_ids": ["code"], "text": "Wait for dig's findings on X streaming, then redo."},
        ],
        finish(),
    )
    runner = FakeRunner({
        "review": [F(verdict="fail", feedback="wrong approach"), F(verdict="pass")],
        "code": [F(summary="v1"), F(summary="blocked", delegate="how does X stream?"), F(summary="v2")],
    })
    h = Harness(_cfg(), meta, runner)
    assert await h.advance() == "completed"
    assert meta.kinds()[2] == ["needs_help"]
    assert [c[0] for c in runner.calls] == ["code", "review", "code", "dig", "code", "review"]
    third = runner.payload("code", 3)
    assert third["upstream"]["dig"]["output"] == {"summary": "dig done"}
    assert any("use streaming" in n for n in third["notes"]) and any("dig's findings" in n for n in third["notes"])


async def test_agent_question_is_answered_by_the_meta_agent():
    meta = ScriptedMeta(
        add(_job("code", "execution", "coder")),
        {"kind": "answer", "job_id": "code", "text": "postgres, as the research said"},
        finish(),
    )
    runner = FakeRunner({"code": [JobOutcome(status="needs_input", questions=["which db?"])]})
    h = Harness(_cfg(), meta, runner)
    assert await h.advance() == "completed"
    assert runner.payload("code", 2)["clarification_context"] == {"which db?": "postgres, as the research said"}
    assert h.job("code").attempts[1].reason == "answer"


async def test_live_question_from_a_running_agent_is_answered_while_others_keep_working():
    runner = FakeRunner()
    release = runner.hold("code")

    def answer_live(view: MetaView) -> dict:
        s = view.situations[0]
        assert s.kind == "live_question" and s.detail["question"] == "tabs or spaces?"
        return {"kind": "answer", "job_id": "code", "text": "spaces"}

    meta = ScriptedMeta(add(_job("code", "execution", "coder"), _job("other", "execution", "coder")), answer_live, finish())
    h = Harness(_cfg(), meta, runner)
    task = asyncio.create_task(h.advance())
    await asyncio.wait_for(runner.started["code"].wait(), 5)
    assert h.hub.ask("child-code-1", "tabs or spaces?")
    for _ in range(200):
        if h.delivered:
            break
        await asyncio.sleep(0.01)
    assert h.delivered == [("child-code-1", "spaces")]
    release.set()
    assert await task == "completed"
    assert runner.ran("code") == 1  # answered live, not restarted
    assert any("spaces" in n.text for n in h.job("code").notes)


async def test_meta_agent_stops_a_running_agent_it_no_longer_needs():
    runner = FakeRunner({"check-a": [F(verdict="fail", feedback="the whole approach is wrong")]})
    runner.hold("slow")  # never released: only stopping it ends it

    meta = ScriptedMeta(
        add(_job("code-a", "execution", "coder"), _job("slow", "execution", "coder"),
            _job("check-a", "validation", "tester", ["code-a"])),
        [
            {"kind": "remove", "job_ids": ["slow"], "text": "approach abandoned"},
            {"kind": "rewind", "job_ids": ["code-a"], "text": "use the other approach"},
        ],
        finish(),
    )
    runner.script["check-a"].append(F(verdict="pass"))
    h = Harness(_cfg(), meta, runner)
    assert await h.advance() == "completed"
    assert ("slow", 1) in runner.cancelled
    assert h.job("slow").status == "removed" and h.job("slow").attempts[0].status == "cancelled"
    assert runner.ran("code-a") == 2


async def test_message_with_restart_stops_the_agent_and_starts_it_over():
    runner = FakeRunner()
    runner.hold("code")

    def restart(view: MetaView) -> dict:
        assert view.situations[0].kind == "live_question"
        return {"kind": "message", "job_id": "code", "text": "requirements changed: add B too", "restart": True}

    meta = ScriptedMeta(add(_job("code", "execution", "coder")), restart, finish())
    h = Harness(_cfg(), meta, runner)
    task = asyncio.create_task(h.advance())
    await asyncio.wait_for(runner.started["code"].wait(), 5)
    runner.started["code"].clear()
    h.hub.ask("child-code-1", "anything else?")
    await asyncio.wait_for(runner.started["code"].wait(), 5)  # attempt 2 started
    runner.holds["code"].set()
    assert await task == "completed"
    assert ("code", 1) in runner.cancelled
    assert "add B too" in runner.payload("code", 2)["feedback"]
    assert h.job("code").attempts[1].reason == "restart"


# ─── humans ──────────────────────────────────────────────────────────────────


async def test_plan_mode_gates_the_plan_and_a_rejection_goes_back_to_the_meta_agent():
    def replan(view: MetaView) -> dict:
        assert [s.kind for s in view.situations] == ["start", "rejected"]
        assert view.situations[1].detail["reason"] == "split it"
        return add(_job("code-a", "execution", "coder"), _job("code-b", "execution", "coder"))

    meta = ScriptedMeta(add(_job("code", "execution", "coder")), replan, finish())
    runner = FakeRunner()
    h = Harness(_cfg(automation="plan"), meta, runner)
    assert await h.advance() == "gate"
    assert runner.calls == [] and h.state.status == "waiting_approval"
    assert await h.advance({"approved": False, "reason": "split it"}) == "gate"
    assert await h.advance({"approved": True, "approver_name": "ann"}) == "completed"  # finishing drops nothing: no gate
    assert sorted(c[0] for c in runner.calls) == ["code-a", "code-b"]
    assert h.approvals[0]["approved"] is False and h.approvals[1]["approver_name"] == "ann"


async def test_edited_plan_from_a_human_is_honoured():
    meta = ScriptedMeta(add(_job("code", "execution", "coder")), finish())
    runner = FakeRunner()
    h = Harness(_cfg(automation="plan"), meta, runner)
    assert await h.advance() == "gate"
    assert await h.advance({"approved": True, "corrections": {"jobs": [_job("mine", "execution", "coder")]}}) == "completed"
    assert [c[0] for c in runner.calls] == ["mine"] and h.job("mine").created_by == "human"


async def test_meta_agent_asks_a_human_and_the_answer_reaches_the_job():
    meta = ScriptedMeta(
        add(_job("code", "execution", "coder")),
        {"kind": "ask_human", "job_id": "code", "text": "Which customer is this for?"},
        finish(),
    )
    runner = FakeRunner({"code": [JobOutcome(status="needs_input", questions=["which customer?"])]})
    h = Harness(_cfg(), meta, runner)
    assert await h.advance() == "gate"
    assert h.state.open_decisions()[0].kind == "question"
    assert await h.advance({"approved": True, "corrections": {"answers": {"customer": "ACME"}}}) == "completed"
    assert runner.payload("code", 2)["clarification_context"] == {"customer": "ACME"}


async def test_ask_mode_gates_plans_and_rearrangements():
    meta = ScriptedMeta(
        add(_job("code", "execution", "coder"), _job("review", "validation", "tester", ["code"])),
        {"kind": "rewind", "job_ids": ["code"], "text": "fix it"},
        finish(),
    )
    runner = FakeRunner({"review": [F(verdict="fail", feedback="x"), F(verdict="pass")]})
    h = Harness(_cfg(automation="ask"), meta, runner)
    assert await h.advance() == "gate"                       # plan
    assert await h.advance({"approved": True}) == "gate"     # rewind
    assert h.state.open_decisions()[0].kind == "rearrange"
    assert await h.advance({"approved": True}) == "completed"
    assert [c[0] for c in runner.calls] == ["code", "review", "code", "review"]


async def test_rearrangement_budget_exhaustion_escalates_and_bypass_fails():
    plan = add(_job("code", "execution", "coder"), _job("review", "validation", "tester", ["code"]))
    rewind = {"kind": "rewind", "job_ids": ["code"], "text": "again"}
    fail = F(verdict="fail", feedback="still red")

    h = Harness(_cfg(limits={"max_rearrangements": 1}), ScriptedMeta(plan, rewind, rewind), FakeRunner({"review": [fail, fail]}))
    assert await h.advance() == "gate"
    assert h.state.open_decisions()[0].kind == "escalation"

    h = Harness(_cfg(automation="bypass", limits={"max_rearrangements": 1}), ScriptedMeta(plan, rewind, rewind), FakeRunner({"review": [fail, fail]}))
    assert await h.advance() == "failed"
    assert "rearrangement budget" in h.state.error


# ─── authored DAGs ───────────────────────────────────────────────────────────


async def test_authored_plan_runs_without_the_meta_agent_until_something_needs_it():
    cfg = _cfg(automation="plan", plan=[_job("code", "execution", "coder"), _job("review", "validation", "tester", ["code"])])
    meta = ScriptedMeta({"kind": "rewind", "job_ids": ["code"], "text": "rename it"})
    runner = FakeRunner({"review": [F(verdict="fail", feedback="rename it"), F(verdict="pass")]})
    h = Harness(cfg, meta, runner)
    assert await h.advance() == "completed"
    assert [c[0] for c in runner.calls] == ["code", "review", "code", "review"]
    assert {j.created_by for j in h.state.jobs} == {"author"}
    assert meta.kinds() == [["change_request"]]


async def test_invalid_authored_plan_fails_the_run():
    h = Harness(_cfg(plan=[_job("code", "execution", "tester")]), ScriptedMeta(), FakeRunner())
    assert await h.advance() == "failed"
    assert "authored plan is invalid" in h.state.error and "may only fill" in h.state.error


# ─── restarts, retries, stops ────────────────────────────────────────────────


async def test_attempts_without_a_live_task_are_adopted_or_rerun_after_a_restart():
    state = DynamicRunState(step_id="orchestrator", status="running", jobs=[
        Job(id="a", category="execution", agent_id="coder", prompt="p", status="running", attempts=[JobAttempt(n=1, child_run_id="c-a")]),
        Job(id="b", category="execution", agent_id="coder", prompt="p", status="running", attempts=[JobAttempt(n=1, child_run_id="c-b")]),
        Job(id="check", category="validation", agent_id="tester", prompt="p", depends_on=["a", "b"]),
    ])
    done = AttemptResult(status="finished", output={"summary": "a ok"}, job_id="a", attempt=1, child_run_id="c-a")
    runner = FakeRunner()
    h = Harness(_cfg(), ScriptedMeta(finish()), runner, state=state, recorded=lambda j, n: done if (j, n) == ("a", 1) else None)
    assert await h.advance() == "completed"
    assert [(c[0], c[1]) for c in runner.calls] == [("b", 2), ("check", 1)]
    assert h.job("a").output == {"summary": "a ok"} and len(h.job("a").attempts) == 1
    assert h.job("b").attempts[0].status == "cancelled" and h.job("b").attempts[1].reason == "restart"


async def test_reset_for_retry_keeps_finished_jobs():
    from app.infrastructure.orchestration.dynamic.engine import reset_for_retry

    meta = ScriptedMeta(add(_job("a", "execution", "coder"), _job("b", "execution", "coder")), {"kind": "fail", "text": "b broke"})
    runner = FakeRunner({"b": [JobOutcome(status="failed", error="boom")]})
    h = Harness(_cfg(limits={"max_retries": 0}), meta, runner)
    assert await h.advance() == "failed"
    state = DynamicRunState.model_validate(reset_for_retry(h.persisted[-1]))
    assert state.job("a").status == "finished" and state.job("b").status == "pending"
    h2 = Harness(_cfg(limits={"max_retries": 0}), ScriptedMeta(finish()), runner, state=state)
    assert await h2.advance() == "completed"
    assert runner.ran("a") == 1


async def test_stop_check_cancels_running_attempts_and_fails():
    runner = FakeRunner()
    runner.hold("a")
    calls = {"n": 0}

    async def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    h = Harness(_cfg(), ScriptedMeta(add(_job("a", "execution", "coder"), _job("b", "execution", "coder", ["a"]))), runner, stop_check=stop)
    assert await h.advance() == "failed"
    assert h.state.error == "run was stopped"
    await asyncio.sleep(0.05)
    assert runner.cancelled == [("a", 1)] and h.job("b").status == "cancelled"


async def test_job_input_carries_git_volume_conventions_and_attempt_reports():
    cfg = _cfg(repo={"url": "https://git/x.git", "verify": ["make test"]}, shared_volume={"mount_point": "/shared"})
    runner = FakeRunner()
    h = Harness(cfg, ScriptedMeta(add(_job("code", "execution", "coder", owns=["src/**"])), finish()), runner)
    assert await h.advance() == "completed"
    ws = runner.payload("code", 1)["workspace"]
    assert ws["job_branch"] == "carrier/12345678/code"
    assert ws["integration_branch"] == "carrier/12345678"
    assert ws["output_dir"] == "/shared/jobs/code/out"
    assert runner.payload("code", 1)["job"]["owns"] == ["src/**"]
    assert h.attempt_events == [("code", 1, "child-code-1", None), ("code", 1, "child-code-1", "finished")]


# ─── the meta-LLM ────────────────────────────────────────────────────────────


def _fence(obj: Any) -> str:
    return "```json\n" + json.dumps(obj) + "\n```"


class ScriptedLLM:
    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def __call__(self, system: str, user: str):
        self.prompts.append(user)
        ans = self.answers.pop(0)
        return (ans if isinstance(ans, str) else _fence(ans)), {"total_tokens": 15}


async def test_llm_meta_agent_sees_the_dag_and_repairs_an_invalid_answer_once():
    bad = {"summary": "s", "actions": [add(_job("code", "execution", "coder", datasources=["billing"]))]}
    good = {"summary": "s", "actions": [add(_job("code", "execution", "coder", datasources=["crm"]))]}
    llm = ScriptedLLM(bad, good, {"summary": "done", "actions": [finish()]})
    h = Harness(_cfg(), LlmMetaAgent(llm), FakeRunner())
    assert await h.advance() == "completed"
    assert "Data sources you may grant:\n- crm: operations get_account" in llm.prompts[0]
    assert "data source 'billing' is not enabled" in llm.prompts[1]
    assert "- code [execution] agent=coder status=finished" in llm.prompts[2]
    assert h.state.usage["total_tokens"] == 45

    h = Harness(_cfg(), LlmMetaAgent(ScriptedLLM(bad, bad)), FakeRunner())
    assert await h.advance() == "failed"
    assert "still invalid" in h.state.error


async def test_a_decision_that_leaves_a_waiting_job_unsettled_is_rejected():
    llm = ScriptedLLM(
        {"actions": [add(_job("code", "execution", "coder"))]},
        {"actions": [add(_job("extra", "planning", "researcher"))]},  # ignores the question
        {"actions": [{"kind": "answer", "job_id": "code", "text": "yes"}]},
        {"actions": [finish()]},
    )
    runner = FakeRunner({"code": [JobOutcome(status="needs_input", questions=["ok?"])]})
    h = Harness(_cfg(), LlmMetaAgent(llm), runner)
    assert await h.advance() == "completed"
    assert "job 'code' waits on its question" in llm.prompts[2]


async def test_jobs_planned_in_the_last_round_of_a_schedule_still_start():
    """A change request settled without rework, then a new phase: its jobs must launch."""
    meta = ScriptedMeta(
        add(_job("code", "execution", "coder"), _job("review", "validation", "tester", ["code"])),
        {"kind": "message", "job_id": "review", "text": "noted; the style nit is out of scope"},
        add(_job("polish", "execution", "coder", ["review"])),
        finish(),
    )
    runner = FakeRunner({"review": [F(verdict="fail", feedback="style nit")]})
    h = Harness(_cfg(), meta, runner)
    assert await h.advance() == "completed"
    assert [c[0] for c in runner.calls] == ["code", "review", "polish"]
    assert meta.kinds() == [["start"], ["change_request"], ["phase_done"], ["phase_done"]]


def test_a_malformed_update_is_rejected_by_validation():
    from app.infrastructure.orchestration.dynamic.engine import JobDag
    from app.infrastructure.orchestration.dynamic.meta_agent import MetaAction

    state = DynamicRunState(step_id="s", jobs=[Job(id="code", category="execution", agent_id="coder", prompt="p", status="finished")])
    dag = JobDag(config=_cfg(), state=state, meta=ScriptedMeta(), roster="", request="", run_id="r")
    errors = dag.validate(MetaDecision(actions=[MetaAction(kind="update_job", job_id="code", changes={"depends_on": 5})]), [])
    assert errors and "depends_on" in errors[0]
