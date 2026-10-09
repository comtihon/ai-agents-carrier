"""Dynamic workflow engine: planning, scheduling, handbacks, delegation, approval gates."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.domain.models.dynamic import DynamicConfig, DynamicRunState, Job, JobAttempt
from app.infrastructure.orchestration.dynamic.dispatcher import (
    Dispatcher,
    ProposedJob,
    validate_jobs,
)
from app.infrastructure.orchestration.dynamic.engine import DynamicEngine, JobOutcome


def _cfg(**over: Any) -> DynamicConfig:
    base = {
        "agent_pool": [
            {"agent_id": "planner", "categories": ["planning"]},
            {"agent_id": "researcher", "categories": ["planning"]},
            {"agent_id": "coder", "max_instances": 3, "categories": ["execution", "integration"]},
            {"agent_id": "tester", "categories": ["validation"]},
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


def _engine(cfg: DynamicConfig, llm: ScriptedLLM, runner: FakeRunner, state: DynamicRunState | None = None):
    persisted: list[dict] = []

    async def persist(s: DynamicRunState) -> None:
        persisted.append(s.model_dump(mode="json"))

    eng = DynamicEngine(
        config=cfg,
        state=state or DynamicRunState(step_id="orchestrator"),
        dispatcher=Dispatcher(llm),
        roster="(roster)",
        request="build features A and B",
        job_runner=runner,
        persist=persist,
        run_id="12345678-run",
    )
    return eng, persisted


# ─── config ──────────────────────────────────────────────────────────────────


def test_config_from_step_fills_default_slots_and_rejects_duplicates():
    cfg = DynamicConfig.from_step({"id": "x", "type": "dynamic", "next": "END", "agent_pool": [{"agent_id": "a"}], "jobs": {"validation": {"min": 1, "max": 2}}})
    assert cfg.jobs["execution"].min == 1
    assert cfg.jobs["validation"].min == 1
    with pytest.raises(ValueError):
        DynamicConfig.model_validate({"agent_pool": [{"agent_id": "a"}, {"agent_id": "a"}]})
    with pytest.raises(ValueError):
        DynamicConfig.model_validate({"jobs": {"execution": {"min": 0, "max": 3}}})


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

    # Resume from the persisted record, as the LangGraph node does after the gate.
    state = DynamicRunState.model_validate(persisted[-1])
    eng2, _ = _engine(cfg, ScriptedLLM(), runner, state=state)
    assert await eng2.advance({"approved": True}) == "completed"
    assert [c[0] for c in runner.calls] == ["code"]


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


async def test_stop_check_cancels_in_flight_jobs_and_fails():
    plan = {"jobs": [_job("a", "execution", "coder"), _job("b", "execution", "coder", ["a"])]}
    runner = FakeRunner(delay=0.05)
    calls = {"n": 0}

    async def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1  # stop after the first scheduling round

    eng, _ = _engine(_cfg(), ScriptedLLM(plan), runner)
    eng._stop_check = stop
    assert await eng.advance() == "failed"
    assert eng.state.error == "run was stopped"
    assert [c[0] for c in runner.calls] == ["a"]
