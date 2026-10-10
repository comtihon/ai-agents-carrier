"""A dynamic step inside a real LangGraph run: job subgraph, gate, resume, child job runs."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
from app.infrastructure.tools.mcp_client import McpToolsProvider


class _Repo:
    """In-memory run repository with the dynamic-specific write."""

    def __init__(self) -> None:
        self.runs: dict[str, GraphRun] = {}

    async def create(self, run: GraphRun) -> None:
        self.runs[run.id] = run.model_copy(deep=True)

    async def update(self, run: GraphRun) -> None:
        self.runs[run.id] = run.model_copy(deep=True)

    async def get(self, run_id: str) -> GraphRun | None:
        r = self.runs.get(run_id)
        return r.model_copy(deep=True) if r else None

    async def set_dynamic(self, run_id: str, step_id: str, data: dict) -> None:
        r = self.runs[run_id]
        r.dynamic = {**r.dynamic, step_id: data}


class _Agents:
    async def get(self, agent_id: str):
        return SimpleNamespace(
            id=agent_id, name=agent_id.title(), description=f"the {agent_id}",
            mcp_addon=None, tools_addon=None, datasource_addons=[], s3_addon=None,
        )


STEPS = [
    {
        "id": "orchestrator",
        "type": "dynamic",
        "automation": "plan",
        "agent_pool": [
            {"agent_id": "coder", "max_instances": 2, "categories": ["execution"]},
            {"agent_id": "tester", "categories": ["validation"]},
        ],
        "output_key": "outcome",
    },
    {"id": "after", "type": "llm", "output_key": "after_out"},
]


def _plan_llm(*answers):
    queue = list(answers)

    async def call(system, user):
        return "```json\n" + json.dumps(queue.pop(0)) + "\n```", None

    return lambda config, settings: call


def _runner(steps: list[dict] = STEPS) -> YamlGraphRunner:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="after done")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner({"id": "dyn", "steps": steps}, llm=llm, mcp_tools_provider=mcp)
    runner._agent_backend = _Agents()
    return runner


def _run() -> GraphRun:
    now = datetime.now(tz=timezone.utc)
    return GraphRun(id="parent-run-0001", graph_id="dyn", user_request="add A and B", status="running",
                    step_statuses={}, created_at=now, updated_at=now)


def test_dynamic_step_is_a_single_workflow_node():
    runner = _runner()
    assert [s["id"] for s in runner.steps] == ["orchestrator", "after"]


@pytest.mark.asyncio
async def test_dynamic_run_pauses_for_plan_then_runs_jobs_as_child_runs():
    plan = {"summary": "split", "jobs": [
        {"id": "code-a", "category": "execution", "agent_id": "coder", "prompt": "A", "owns": ["a/**"]},
        {"id": "code-b", "category": "execution", "agent_id": "coder", "prompt": "B", "owns": ["b/**"]},
        {"id": "test", "category": "validation", "agent_id": "tester", "prompt": "test", "depends_on": ["code-a", "code-b"]},
    ]}
    executed: list[tuple[str, str]] = []

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        executed.append((step["id"], run_id))
        assert step["output_mapping"]
        assert state["task"] in ("A", "B", "test")
        if step["id"] == "test":
            return {"verdict": "pass", "summary": "green"}
        return {"summary": f"{step['id']} done", "branch": f"b-{step['id']}"}

    runner = _runner()
    run = _run()
    repo = _Repo()
    await repo.create(run)

    with patch("app.infrastructure.orchestration.dynamic.node.build_llm_call", _plan_llm(plan)), \
         patch("app.steps.agent_executor.execute_agent_step", new=fake_execute), \
         patch("app.services.agent_cleanup.cleanup_run_agents", new=_noop):
        await stream_graph_to_pause(runner, run, repo, {"request": "add A and B"})
        assert run.status == "waiting_approval"
        assert run.current_step == "orchestrator"
        dag = repo.runs[run.id].dynamic["orchestrator"]
        assert dag["status"] == "waiting_approval"
        assert [d["kind"] for d in dag["decisions"]] == ["plan"]
        assert executed == []

        await stream_graph_to_pause(runner, run, repo, Command(resume={"approved": True}))

    assert run.status == "completed", run.state.get("error")
    assert sorted(e[0] for e in executed) == ["code-a", "code-b", "test"]
    # Every attempt ran under its own child run id, recorded as a job run.
    child_ids = {e[1] for e in executed}
    assert len(child_ids) == 3 and run.id not in child_ids
    for cid in child_ids:
        child = repo.runs[cid]
        assert child.kind == "job" and child.parent_run_id == run.id and child.status == "completed"
    dag = repo.runs[run.id].dynamic["orchestrator"]
    assert dag["status"] == "completed"
    assert run.state["outcome"]["jobs"]["code-a"]["output"]["branch"] == "b-code-a"
    assert run.state["after_out"] == "after done"
    assert run.state["approval_history"][0]["approved"] is True


@pytest.mark.asyncio
async def test_rejected_plan_replans_without_ending_the_run():
    first = {"jobs": [{"id": "code", "category": "execution", "agent_id": "coder", "prompt": "x"}]}
    second = {"jobs": [{"id": "code2", "category": "execution", "agent_id": "coder", "prompt": "y"}]}

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        return {"summary": "ok"}

    runner = _runner()
    run = _run()
    repo = _Repo()
    await repo.create(run)
    with patch("app.infrastructure.orchestration.dynamic.node.build_llm_call", _plan_llm(first, second)), \
         patch("app.steps.agent_executor.execute_agent_step", new=fake_execute), \
         patch("app.services.agent_cleanup.cleanup_run_agents", new=_noop):
        await stream_graph_to_pause(runner, run, repo, {"request": "r"})
        await stream_graph_to_pause(runner, run, repo, Command(resume={"approved": False, "reason": "smaller"}))
        assert run.status == "waiting_approval"
        dag = repo.runs[run.id].dynamic["orchestrator"]
        assert dag["replans"] == 1
        await stream_graph_to_pause(runner, run, repo, Command(resume={"approved": True}))
    assert run.status == "completed"
    assert list(run.state["outcome"]["jobs"]) == ["code2"]


async def _noop(*a, **k):
    return None


def test_validate_dynamic_steps_reports_config_problems():
    from app.infrastructure.orchestration.dynamic.node import validate_dynamic_steps

    assert validate_dynamic_steps(STEPS) == []
    errors = validate_dynamic_steps([
        {"id": "a", "type": "dynamic", "agent_pool": []},
        {"id": "b", "type": "dynamic", "agent_pool": [{"agent_id": "t", "categories": ["validation"]}],
         "jobs": {"execution": {"min": 1, "max": 2}}},
        {"id": "c", "type": "dynamic", "automation": "sometimes", "agent_pool": [{"agent_id": "x"}]},
        {"id": "d", "type": "dynamic", "agent_pool": [{"agent_id": "t", "categories": ["validation"]}],
         "plan": [{"id": "code", "category": "execution", "agent_id": "t", "prompt": "p"},
                  {"id": "review", "category": "validation", "agent_id": "t", "prompt": "p", "depends_on": ["ghost"]}]},
    ])
    text = " | ".join(errors)
    assert "agent_pool is empty" in text
    assert "no agent in the pool may fill execution jobs (jobs.execution.min is 1)" in text
    assert "step 'c': automation" in text
    assert "step 'd': plan: job 'code': agent 't' may only fill" in text
    assert "depends on unknown job 'ghost'" in text
    # Every category is optional: a validator-only pool is fine without a minimum.
    assert validate_dynamic_steps([{"id": "e", "type": "dynamic", "agent_pool": [{"agent_id": "t", "categories": ["validation"]}]}]) == []


def test_roster_carries_description_and_addons():
    from app.domain.models.agent_addon import DatasourceAddon, MCPAddon, ToolsAddon
    from app.domain.models.agent_definition import AgentDefinition
    from app.domain.models.dynamic import DynamicConfig
    from app.infrastructure.orchestration.dynamic.dispatcher import build_roster

    agent = AgentDefinition(
        id="validator", name="Validator", description="Checks roof geometry against the survey",
        addons=[
            MCPAddon(servers={"blender": True, "jira": False}),
            ToolsAddon(tools={"gh": True}),
            DatasourceAddon(source_id="survey", allowed_operations=["get_points"]),
        ],
    )
    cfg = DynamicConfig.model_validate({"agent_pool": [{"agent_id": "validator", "categories": ["validation", "execution"]}]})
    roster = build_roster(cfg, {"validator": agent})
    assert "Checks roof geometry" in roster
    assert "MCP: blender" in roster and "jira" not in roster
    assert "tools: gh" in roster
    assert "datasource survey (get_points)" in roster


@pytest.mark.asyncio
async def test_authored_job_chain_longer_than_the_default_recursion_limit():
    """15 sequential jobs are 30+ supersteps of the job subgraph: more than LangGraph's default 25."""
    chain = [
        {"id": f"part-{i}", "category": "execution", "agent_id": "coder", "prompt": f"part {i}",
         "depends_on": [f"part-{i - 1}"] if i else []}
        for i in range(15)
    ]
    steps = [{**STEPS[0], "automation": "auto", "plan": chain, "jobs": {"execution": {"max": 15}}, "next": "END"}]
    executed: list[str] = []

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        executed.append(step["id"])
        return {"summary": f"{step['id']} done"}

    runner = _runner(steps)
    run = _run()
    repo = _Repo()
    await repo.create(run)
    with patch("app.steps.agent_executor.execute_agent_step", new=fake_execute), \
         patch("app.services.agent_cleanup.cleanup_run_agents", new=_noop):
        await stream_graph_to_pause(runner, run, repo, {"request": "build it"})
    assert run.status == "completed", run.state.get("error")
    assert executed == [f"part-{i}" for i in range(15)]
    dag = repo.runs[run.id].dynamic["orchestrator"]
    assert dag["status"] == "completed" and all(j["created_by"] == "author" for j in dag["jobs"])
    # The mirrored DAG carries every attempt's child run and result for the UI.
    first = dag["jobs"][0]["attempts"][0]
    assert first["child_run_id"] and first["status"] == "finished"


@pytest.mark.asyncio
async def test_question_gate_pauses_the_workflow_at_the_dynamic_step():
    from langgraph.errors import GraphInterrupt
    from langgraph.types import Interrupt

    plan = {"jobs": [{"id": "code", "category": "execution", "agent_id": "coder", "prompt": "x"}]}
    asked = {"n": 0}

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        asked["n"] += 1
        if asked["n"] == 1:
            raise GraphInterrupt([Interrupt(value={"questions": ["which db?"]})])
        assert state["_clarification_answers"] == {"which db?": "postgres"}
        return {"summary": "done"}

    runner = _runner()
    run = _run()
    repo = _Repo()
    await repo.create(run)
    with patch("app.infrastructure.orchestration.dynamic.node.build_llm_call", _plan_llm(plan)), \
         patch("app.steps.agent_executor.execute_agent_step", new=fake_execute), \
         patch("app.services.agent_cleanup.cleanup_run_agents", new=_noop):
        await stream_graph_to_pause(runner, run, repo, {"request": "r"})
        assert run.status == "waiting_approval" and run.current_step == "orchestrator"
        await stream_graph_to_pause(runner, run, repo, Command(resume={"approved": True}))  # the plan
        assert run.status == "waiting_approval"
        dag = repo.runs[run.id].dynamic["orchestrator"]
        assert [d["kind"] for d in dag["decisions"] if not d["resolved"]] == ["question"]
        await stream_graph_to_pause(
            runner, run, repo,
            Command(resume={"approved": True, "corrections": {"answers": {"which db?": "postgres"}}}),
        )
    assert run.status == "completed", run.state.get("error")
    assert [h["approved"] for h in run.state["approval_history"]] == [True, True]


class _Crash(BaseException):
    """Stands in for the process dying mid-wave: nothing in the job runner catches it."""


@pytest.mark.asyncio
async def test_crash_mid_wave_reruns_only_the_part_that_had_not_finished():
    plan = {"jobs": [
        {"id": "code-a", "category": "execution", "agent_id": "coder", "prompt": "A", "owns": ["a/**"]},
        {"id": "code-b", "category": "execution", "agent_id": "coder", "prompt": "B", "owns": ["b/**"]},
    ]}
    executed: list[str] = []
    crash = {"armed": True}

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        executed.append(step["id"])
        if step["id"] == "code-b":
            await asyncio.sleep(0.02)  # code-a reports first
            if crash["armed"]:
                crash["armed"] = False
                raise _Crash()
        return {"summary": f"{step['id']} done"}

    runner = _runner([{**STEPS[0], "automation": "auto"}, STEPS[1]])
    run = _run()
    repo = _Repo()
    await repo.create(run)
    with patch("app.infrastructure.orchestration.dynamic.node.build_llm_call", _plan_llm(plan)), \
         patch("app.steps.agent_executor.execute_agent_step", new=fake_execute), \
         patch("app.services.agent_cleanup.cleanup_run_agents", new=_noop):
        with pytest.raises(_Crash):
            await stream_graph_to_pause(runner, run, repo, {"request": "add A and B"})
        mirrored = repo.runs[run.id].dynamic["orchestrator"]
        a = next(j for j in mirrored["jobs"] if j["id"] == "code-a")
        assert a["attempts"][0]["status"] == "finished"  # reported before the crash
        # Resume the workflow from its checkpoint, as a restart does.
        await stream_graph_to_pause(runner, run, repo, None)
    assert run.status == "completed", run.state.get("error")
    assert sorted(executed) == ["code-a", "code-b", "code-b"]
