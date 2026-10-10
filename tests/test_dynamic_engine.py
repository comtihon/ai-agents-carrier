"""The job graph: planning, waves of parallel parts, handbacks, delegation, approval gates.

Every test drives the real LangGraph job graph (``build_job_graph``) with a
MemorySaver, a scripted dispatcher and a scripted job runner.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.domain.models.dynamic import DynamicConfig, DynamicRunState, Job, JobAttempt
from app.infrastructure.orchestration.dynamic.dispatcher import (
    Dispatcher,
    ProposedJob,
    validate_jobs,
)
from app.infrastructure.orchestration.dynamic.engine import AttemptResult, JobOutcome
from app.infrastructure.orchestration.dynamic.graph import build_job_graph, initial_input


def _cfg(**over: Any) -> DynamicConfig:
    base = {
        "agent_pool": [
            {"agent_id": "planner", "categories": ["planning"]},
            {"agent_id": "researcher", "max_instances": 2, "categories": ["planning"]},
            {"agent_id": "coder", "max_instances": 3, "categories": ["execution", "integration"]},
            {"agent_id": "tester", "max_instances": 2, "categories": ["validation"]},
        ],
        "automation": "auto",
    }
    base.update(over)
    return DynamicConfig.model_validate(base)


def _fence(obj: Any) -> str:
    return "```json\n" + json.dumps(obj) + "\n```"


class ScriptedLLM:
    """Returns queued answers in order; records every prompt."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def __call__(self, system: str, user: str):
        self.prompts.append(user)
        if not self.answers:
            raise AssertionError("dispatcher asked more questions than scripted:\n" + user[:500])
        ans = self.answers.pop(0)
        return (ans if isinstance(ans, str) else _fence(ans)), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def _job(id: str, category: str, agent: str, deps: list[str] | None = None, owns: list[str] | None = None) -> dict:
    return {"id": id, "category": category, "agent_id": agent, "title": id, "prompt": f"do {id}", "depends_on": deps or [], "owns": owns or []}


class FakeRunner:
    """Scripted job outcomes per job id (a list consumed per attempt)."""

    def __init__(self, script: dict[str, list[JobOutcome]] | None = None, delay: float = 0.0) -> None:
        self.script = script or {}
        self.delay = delay
        self.calls: list[tuple[str, int, dict]] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def __call__(self, job: Job, attempt: JobAttempt, payload: dict, started) -> JobOutcome:
        await started(f"child-{job.id}-{attempt.n}")
        self.calls.append((job.id, attempt.n, payload))
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.concurrent -= 1
        queue = self.script.get(job.id)
        if queue:
            return queue.pop(0)
        return JobOutcome(status="finished", output={"summary": f"{job.id} done"})

    def ran(self, job_id: str) -> int:
        return sum(1 for c in self.calls if c[0] == job_id)


class Harness:
    """One dynamic step's job graph on its own thread; ``advance`` runs it to the next pause."""

    def __init__(self, cfg: DynamicConfig, llm: ScriptedLLM, runner: FakeRunner, state: DynamicRunState | None = None, **kw: Any) -> None:
        self.persisted: list[dict] = []
        self.attempt_events: list[tuple[str, int, str | None, str | None]] = []

        async def persist(s: DynamicRunState) -> None:
            self.persisted.append(s.model_dump(mode="json"))

        async def on_attempt(job_id: str, n: int, child: str | None, result: AttemptResult | None) -> None:
            self.attempt_events.append((job_id, n, child, result.status if result else None))

        self.graph = build_job_graph(
            config=cfg,
            dispatcher=Dispatcher(llm),
            roster="(roster)",
            request="build features A and B",
            run_id="12345678-run",
            job_runner=runner,
            persist=persist,
            on_attempt=on_attempt,
            checkpointer=MemorySaver(),
            **kw,
        )
        self.config = {"configurable": {"thread_id": "t"}, "recursion_limit": 1000}
        self.initial = state or DynamicRunState(step_id="orchestrator")
        self.started = False
        self.state = self.initial
        self.approvals: list[dict] = []

    async def advance(self, resolution: dict[str, Any] | None = None) -> str:
        if not self.started:
            self.started = True
            out = await self.graph.ainvoke(initial_input(self.initial), self.config)
        else:
            out = await self.graph.ainvoke(Command(resume=resolution), self.config)
        self.state = DynamicRunState.model_validate(out["dag"])
        self.approvals = out.get("approvals") or []
        if out.get("__interrupt__"):
            return "gate"
        return self.state.status


def _engine(cfg: DynamicConfig, llm: ScriptedLLM, runner: FakeRunner, state: DynamicRunState | None = None, **kw: Any):
    h = Harness(cfg, llm, runner, state, **kw)
    return h, h.persisted


# ─── config ──────────────────────────────────────────────────────────────────


def test_config_from_step_fills_default_slots_and_rejects_duplicates():
    cfg = DynamicConfig.from_step({"id": "x", "type": "dynamic", "next": "END", "agent_pool": [{"agent_id": "a"}], "jobs": {"validation": {"min": 1, "max": 2}}})
    assert cfg.jobs["execution"].min == 0 and cfg.jobs["execution"].max == 8
    assert cfg.jobs["validation"].min == 1
    with pytest.raises(ValueError):
        DynamicConfig.model_validate({"agent_pool": [{"agent_id": "a"}, {"agent_id": "a"}]})
    # Every category is optional, execution included.
    assert DynamicConfig.model_validate({"jobs": {"execution": {"min": 0, "max": 3}}}).jobs["execution"].min == 0
    with pytest.raises(ValueError):
        DynamicConfig.model_validate({"plan": []})


def test_approval_policy_by_mode():
    assert _cfg(automation="ask").needs_approval("handback")
    assert _cfg(automation="plan").needs_approval("plan")
    assert not _cfg(automation="plan").needs_approval("handback")
    assert _cfg(automation="auto").needs_approval("escalation")
    assert not _cfg(automation="bypass").needs_approval("escalation")


def test_validate_jobs_catches_pool_category_cycle_and_slot_violations():
    cfg = _cfg(jobs={"execution": {"min": 1, "max": 2}})
    jobs = [
        ProposedJob(**_job("a", "execution", "coder", ["b"])),
        ProposedJob(**_job("b", "execution", "coder", ["a"])),
        ProposedJob(**_job("c", "execution", "coder")),
        ProposedJob(**_job("d", "validation", "coder")),
        ProposedJob(**_job("e", "execution", "ghost")),
    ]
    errors = " | ".join(validate_jobs(jobs, cfg, []))
    assert "cycle" in errors
    assert "may only fill" in errors
    assert "not in the pool" in errors
    assert "too many execution jobs" in errors


# ─── happy paths ─────────────────────────────────────────────────────────────


async def test_full_plan_runs_parallel_coders_then_validator():
    llm = ScriptedLLM({
        "summary": "two coders, one tester",
        "jobs": [
            _job("code-a", "execution", "coder", owns=["src/a/**"]),
            _job("code-b", "execution", "coder", owns=["src/b/**"]),
            _job("test", "validation", "tester", ["code-a", "code-b"]),
        ],
    })
    runner = FakeRunner(delay=0.01)
    eng, persisted = _engine(_cfg(), llm, runner)
    assert await eng.advance() == "completed"
    order = [c[0] for c in runner.calls]
    assert set(order[:2]) == {"code-a", "code-b"} and order[2] == "test"
    assert runner.max_concurrent == 2
    # The validator sees both upstream outputs.
    assert set(runner.calls[2][2]["upstream"]) == {"code-a", "code-b"}
    assert eng.state.status == "completed" and "code-a" in eng.state.summary
    assert persisted and eng.state.usage["total_tokens"] == 15


async def test_planner_output_is_expanded_into_execution_jobs():
    llm = ScriptedLLM(
        {"summary": "research then plan", "jobs": [_job("research", "planning", "researcher"), _job("plan", "planning", "planner", ["research"])]},
        {"summary": "from plan", "jobs": [_job("code-a", "execution", "coder", ["plan"]), _job("test", "validation", "tester", ["code-a"])]},
    )
    runner = FakeRunner({"plan": [JobOutcome(status="finished", output={"summary": "plan", "plan": "1. A"})]})
    eng, _ = _engine(_cfg(), llm, runner)
    assert await eng.advance() == "completed"
    assert [c[0] for c in runner.calls] == ["research", "plan", "code-a", "test"]
    assert "1. A" in llm.prompts[1]  # the expansion prompt carries the planner's output
    assert eng.state.expanded


async def test_max_instances_caps_parallel_jobs_of_one_agent():
    llm = ScriptedLLM({"jobs": [_job(f"c{i}", "execution", "coder") for i in range(5)]})
    runner = FakeRunner(delay=0.01)
    cfg = _cfg(limits={"max_parallel": 8})
    eng, _ = _engine(cfg, llm, runner)
    assert await eng.advance() == "completed"
    assert runner.max_concurrent == 3  # coder.max_instances


# ─── loops ───────────────────────────────────────────────────────────────────


async def test_validator_handback_reruns_target_with_feedback_then_passes():
    llm = ScriptedLLM({"jobs": [_job("code", "execution", "coder"), _job("test", "validation", "tester", ["code"])]})
    runner = FakeRunner({
        "test": [
            JobOutcome(status="finished", output={"verdict": "fail", "feedback": "tests red", "handback_to": ["code"]}),
            JobOutcome(status="finished", output={"verdict": "pass", "summary": "green"}),
        ]
    })
    eng, _ = _engine(_cfg(), llm, runner)
    assert await eng.advance() == "completed"
    assert [c[0] for c in runner.calls] == ["code", "test", "code", "test"]
    second_code = runner.calls[2][2]
    assert "tests red" in second_code["feedback"]
    assert eng.state.job("code").attempts[1].reason == "handback"


async def test_handback_budget_exhaustion_escalates_and_bypass_fails():
    fail = JobOutcome(status="finished", output={"verdict": "fail", "feedback": "still red"})
    plan = {"jobs": [_job("code", "execution", "coder"), _job("test", "validation", "tester", ["code"])]}

    eng, _ = _engine(_cfg(limits={"max_handbacks": 1}), ScriptedLLM(plan), FakeRunner({"test": [fail, fail, fail]}))
    assert await eng.advance() == "gate"
    assert eng.state.open_decisions()[0].kind == "escalation"

    eng, _ = _engine(_cfg(automation="bypass", limits={"max_handbacks": 1}), ScriptedLLM(plan), FakeRunner({"test": [fail, fail, fail]}))
    assert await eng.advance() == "failed"
    assert "still failing" in eng.state.error


async def test_failed_attempt_is_retried_then_succeeds():
    llm = ScriptedLLM({"jobs": [_job("code", "execution", "coder")]})
    runner = FakeRunner({"code": [JobOutcome(status="failed", error="pod crashed")]})
    eng, _ = _engine(_cfg(), llm, runner)
    assert await eng.advance() == "completed"
    assert "pod crashed" in runner.calls[1][2]["feedback"]


async def test_delegation_spawns_helper_and_returns_result_to_requester():
    llm = ScriptedLLM(
        {"jobs": [_job("plan", "planning", "planner")]},
        {"agent_id": "researcher", "title": "find api", "prompt": "find the billing API"},
        {"jobs": [_job("code", "execution", "coder", ["plan"])]},
    )
    runner = FakeRunner({
        "plan": [
            JobOutcome(status="finished", output={"summary": "need info", "delegate": "which billing API?"}),
            JobOutcome(status="finished", output={"summary": "plan", "plan": "use v2"}),
        ],
        "plan-help": [JobOutcome(status="finished", output={"summary": "billing API is v2"})],
    })
    eng, _ = _engine(_cfg(), llm, runner)
    assert await eng.advance() == "completed"
    assert [c[0] for c in runner.calls] == ["plan", "plan-help", "plan", "code"]
    second_plan = runner.calls[2][2]
    assert second_plan["delegation_results"]["plan-help"]["summary"] == "billing API is v2"
    assert eng.state.job("plan-help").delegated_by == "plan"


# ─── gates ───────────────────────────────────────────────────────────────────


async def test_plan_mode_gates_initial_plan_and_resumes_after_approval():
    plan = {"summary": "s", "jobs": [_job("code", "execution", "coder")]}
    cfg = _cfg(automation="plan")
    runner = FakeRunner()
    eng, persisted = _engine(cfg, ScriptedLLM(plan), runner)
    assert await eng.advance() == "gate"
    assert runner.calls == []
    assert eng.state.status == "waiting_approval"
    assert persisted[-1]["status"] == "waiting_approval"

    # The gate's interrupt resumes the graph from its checkpoint.
    assert await eng.advance({"approved": True, "approver_name": "ann"}) == "completed"
    assert [c[0] for c in runner.calls] == ["code"]
    assert eng.approvals[0]["approved"] is True and eng.approvals[0]["approver_name"] == "ann"


async def test_rejected_plan_is_replanned_with_reason_and_edited_plan_is_honoured():
    cfg = _cfg(automation="plan")
    llm = ScriptedLLM(
        {"jobs": [_job("code", "execution", "coder")]},
        {"jobs": [_job("code-v2", "execution", "coder")]},
    )
    runner = FakeRunner()
    eng, _ = _engine(cfg, llm, runner)
    assert await eng.advance() == "gate"
    assert await eng.advance({"approved": False, "reason": "split it differently"}) == "gate"
    assert "split it differently" in llm.prompts[1]
    edited = [_job("mine", "execution", "coder")]
    assert await eng.advance({"approved": True, "corrections": {"jobs": edited}}) == "completed"
    assert [c[0] for c in runner.calls] == ["mine"]
    assert eng.state.job("mine").created_by == "human"


async def test_replan_budget_exhausted_fails():
    cfg = _cfg(automation="plan", limits={"max_replans": 0})
    eng, _ = _engine(cfg, ScriptedLLM({"jobs": [_job("code", "execution", "coder")]}), FakeRunner())
    assert await eng.advance() == "gate"
    assert await eng.advance({"approved": False, "reason": "no"}) == "failed"


async def test_ask_mode_gates_handback_and_lets_in_flight_jobs_finish():
    cfg = _cfg(automation="ask")
    plan = {"jobs": [_job("code", "execution", "coder"), _job("slow", "execution", "coder"), _job("test", "validation", "tester", ["code"])]}
    runner = FakeRunner({"test": [JobOutcome(status="finished", output={"verdict": "fail", "feedback": "x"})]})
    eng, _ = _engine(cfg, ScriptedLLM(plan), runner)
    assert await eng.advance() == "gate"                     # plan
    assert await eng.advance({"approved": True}) == "gate"   # handback proposed
    assert eng.state.open_decisions()[0].kind == "handback"
    assert eng.state.job("slow").status == "finished"        # drained, not killed
    assert await eng.advance({"approved": True}) == "completed"
    assert [c[0] for c in runner.calls if c[0] in ("code", "test")] == ["code", "test", "code", "test"]


async def test_question_from_agent_is_gated_and_answer_reaches_next_attempt():
    cfg = _cfg(automation="auto")
    runner = FakeRunner({"code": [JobOutcome(status="needs_input", questions=["which db?"])]})
    eng, _ = _engine(cfg, ScriptedLLM({"jobs": [_job("code", "execution", "coder")]}), runner)
    assert await eng.advance() == "gate"
    assert eng.state.open_decisions()[0].kind == "question"
    assert await eng.advance({"approved": True, "corrections": {"answers": {"which db?": "postgres"}}}) == "completed"
    assert runner.calls[1][2]["clarification_context"] == {"which db?": "postgres"}


async def test_in_flight_attempt_from_previous_pass_is_retried():
    state = DynamicRunState(step_id="orchestrator", status="running", expanded=True, jobs=[
        Job(id="code", category="execution", agent_id="coder", prompt="p", status="running",
            attempts=[JobAttempt(n=1, child_run_id="dead")]),
    ])
    runner = FakeRunner()
    eng, _ = _engine(_cfg(), ScriptedLLM(), runner, state=state)
    assert await eng.advance() == "completed"
    assert eng.state.job("code").attempts[0].status == "cancelled"
    assert runner.calls[0][1] == 2


async def test_invalid_dispatcher_answer_is_repaired_once_then_fails():
    bad = {"jobs": [_job("code", "execution", "ghost")]}
    good = {"jobs": [_job("code", "execution", "coder")]}
    llm = ScriptedLLM(bad, good)
    eng, _ = _engine(_cfg(), llm, FakeRunner())
    assert await eng.advance() == "completed"
    assert "not in the pool" in llm.prompts[1]

    eng, _ = _engine(_cfg(), ScriptedLLM(bad, bad), FakeRunner())
    assert await eng.advance() == "failed"
    assert "still invalid" in eng.state.error


async def test_job_input_carries_git_and_volume_conventions():
    cfg = _cfg(repo={"url": "https://git/x.git", "verify": ["make test"]}, shared_volume={"mount_point": "/shared"})
    runner = FakeRunner()
    eng, _ = _engine(cfg, ScriptedLLM({"jobs": [_job("code", "execution", "coder", owns=["src/**"])]}), runner)
    assert await eng.advance() == "completed"
    ws = runner.calls[0][2]["workspace"]
    assert ws["job_branch"] == "carrier/12345678/code"
    assert ws["integration_branch"] == "carrier/12345678"
    assert ws["output_dir"] == "/shared/jobs/code/out"
    assert runner.calls[0][2]["job"]["owns"] == ["src/**"]


async def test_reset_for_retry_keeps_finished_jobs_and_grants_an_attempt():
    from app.infrastructure.orchestration.dynamic.engine import reset_for_retry

    plan = {"jobs": [_job("a", "execution", "coder"), _job("b", "execution", "coder")]}
    cfg = _cfg(automation="bypass", limits={"max_handbacks": 0})
    runner = FakeRunner({"b": [JobOutcome(status="failed", error="boom")]})
    eng, persisted = _engine(cfg, ScriptedLLM(plan), runner)
    assert await eng.advance() == "failed"

    state = DynamicRunState.model_validate(reset_for_retry(persisted[-1]))
    assert state.job("a").status == "finished" and state.job("b").status == "pending"
    eng2, _ = _engine(cfg, ScriptedLLM(), runner, state=state)
    assert await eng2.advance() == "completed"
    assert [c[0] for c in runner.calls] .count("a") == 1


async def test_stop_check_ends_the_run_before_the_next_wave():
    plan = {"jobs": [_job("a", "execution", "coder"), _job("b", "execution", "coder", ["a"])]}
    runner = FakeRunner()
    calls = {"n": 0}

    async def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # stop after the first wave

    eng, _ = _engine(_cfg(), ScriptedLLM(plan), runner, stop_check=stop)
    assert await eng.advance() == "failed"
    assert eng.state.error == "run was stopped"
    assert [c[0] for c in runner.calls] == ["a"]
    assert eng.state.job("b").status == "cancelled"


# ─── parts, optional categories, authored DAGs ──────────────────────────────


async def test_only_the_failed_part_runs_again():
    plan = {"jobs": [_job(f"code-{p}", "execution", "coder", owns=[f"src/{p}/**"]) for p in "abc"]}
    runner = FakeRunner({"code-b": [JobOutcome(status="failed", error="pod crashed")]})
    eng, _ = _engine(_cfg(), ScriptedLLM(plan), runner)
    assert await eng.advance() == "completed"
    assert (runner.ran("code-a"), runner.ran("code-b"), runner.ran("code-c")) == (1, 2, 1)
    assert runner.max_concurrent == 3
    assert [a.status for a in eng.state.job("code-b").attempts] == ["failed", "finished"]


async def test_validator_per_part_hands_back_only_its_part():
    plan = {"jobs": [
        _job("code-a", "execution", "coder", owns=["a/**"]),
        _job("code-b", "execution", "coder", owns=["b/**"]),
        _job("check-a", "validation", "tester", ["code-a"]),
        _job("check-b", "validation", "tester", ["code-b"]),
        _job("merge", "integration", "coder", ["check-a", "check-b"]),
    ]}
    runner = FakeRunner({"check-b": [
        JobOutcome(status="finished", output={"verdict": "fail", "feedback": "b is wrong"}),
        JobOutcome(status="finished", output={"verdict": "pass"}),
    ]})
    eng, _ = _engine(_cfg(), ScriptedLLM(plan), runner)
    assert await eng.advance() == "completed"
    assert {j: runner.ran(j) for j in ("code-a", "check-a", "code-b", "check-b", "merge")} == {
        "code-a": 1, "check-a": 1, "code-b": 2, "check-b": 2, "merge": 1,
    }
    second_b = [c for c in runner.calls if c[0] == "code-b"][1][2]
    assert "b is wrong" in second_b["feedback"]


async def test_several_researchers_split_the_research_and_no_executor_is_needed():
    llm = ScriptedLLM(
        {"jobs": [_job("q1", "planning", "researcher"), _job("q2", "planning", "researcher"), _job("sum", "planning", "planner", ["q1", "q2"])]},
        {"summary": "the research answers the request", "jobs": []},
    )
    runner = FakeRunner(delay=0.01)
    eng, _ = _engine(_cfg(), llm, runner)
    assert await eng.advance() == "completed"
    assert [c[0] for c in runner.calls][2] == "sum" and runner.max_concurrent == 2
    assert all(j.category == "planning" for j in eng.state.jobs)
    assert eng.state.expanded


async def test_authored_plan_runs_without_asking_the_dispatcher():
    cfg = _cfg(automation="plan", plan=[
        _job("code", "execution", "coder"),
        _job("review", "validation", "tester", ["code"]),
    ])
    runner = FakeRunner({"review": [
        JobOutcome(status="finished", output={"verdict": "fail", "feedback": "rename it"}),
        JobOutcome(status="finished", output={"verdict": "pass"}),
    ]})
    llm = ScriptedLLM()  # any dispatcher call would fail the test
    eng, _ = _engine(cfg, llm, runner)
    assert await eng.advance() == "completed"  # the authored DAG needs no plan approval
    assert [c[0] for c in runner.calls] == ["code", "review", "code", "review"]
    assert {j.created_by for j in eng.state.jobs} == {"author"}
    assert llm.prompts == []


async def test_invalid_authored_plan_fails_the_run():
    cfg = _cfg(plan=[_job("code", "execution", "tester")])
    eng, _ = _engine(cfg, ScriptedLLM(), FakeRunner())
    assert await eng.advance() == "failed"
    assert "authored plan is invalid" in eng.state.error and "may only fill" in eng.state.error


async def test_a_finished_attempt_is_not_run_again_when_a_wave_is_replayed():
    plan = {"jobs": [_job("a", "execution", "coder"), _job("b", "execution", "coder")]}
    done = AttemptResult(status="finished", output={"summary": "a from before the restart"}, job_id="a", attempt=1, child_run_id="old")

    def recorded(job_id: str, n: int) -> AttemptResult | None:
        return done if (job_id, n) == ("a", 1) else None

    runner = FakeRunner()
    eng, _ = _engine(_cfg(), ScriptedLLM(plan), runner, recorded=recorded)
    assert await eng.advance() == "completed"
    assert [c[0] for c in runner.calls] == ["b"]
    assert eng.state.job("a").output == {"summary": "a from before the restart"}
    assert eng.state.job("a").attempts[0].child_run_id == "old"


async def test_attempts_are_reported_as_they_start_and_finish():
    eng, _ = _engine(_cfg(), ScriptedLLM({"jobs": [_job("code", "execution", "coder")]}), FakeRunner())
    assert await eng.advance() == "completed"
    assert eng.attempt_events == [("code", 1, "child-code-1", None), ("code", 1, "child-code-1", "finished")]
    assert eng.state.job("code").attempts[0].child_run_id == "child-code-1"


async def test_a_pass_from_the_mirrored_dag_keeps_attempts_that_reported_before_a_restart():
    """Recovery re-seeds the workflow, so the step starts from the mirror the attempts wrote to."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    state = DynamicRunState(step_id="orchestrator", status="running", expanded=True, jobs=[
        Job(id="a", category="execution", agent_id="coder", prompt="p", status="running",
            attempts=[JobAttempt(n=1, child_run_id="c-a", status="finished", finished_at=now, output={"summary": "a ok"})]),
        Job(id="b", category="execution", agent_id="coder", prompt="p", status="running",
            attempts=[JobAttempt(n=1, child_run_id="c-b")]),
        Job(id="check", category="validation", agent_id="tester", prompt="p", depends_on=["a", "b"]),
    ])
    runner = FakeRunner()
    eng, _ = _engine(_cfg(), ScriptedLLM(), runner, state=state)
    assert await eng.advance() == "completed"
    assert [(c[0], c[1]) for c in runner.calls] == [("b", 2), ("check", 1)]
    assert eng.state.job("a").output == {"summary": "a ok"} and len(eng.state.job("a").attempts) == 1
    assert eng.state.job("b").attempts[0].status == "cancelled"
