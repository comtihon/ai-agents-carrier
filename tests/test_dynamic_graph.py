"""A dynamic step inside a real workflow run: meta-agent, job subgraph, gates, child job runs."""

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
            {"agent_id": "researcher", "categories": ["planning"]},
            {"agent_id": "coder", "max_instances": 2, "categories": ["execution"]},
            {"agent_id": "tester", "categories": ["validation"]},
        ],
        "datasources": [{"source_id": "crm", "operations": ["get_account", "list_accounts"]}],
        "output_key": "outcome",
    },
    {"id": "after", "type": "llm", "output_key": "after_out"},
]


def _job(id: str, category: str, agent: str, deps: list[str] | None = None, **extra) -> dict:
    return {"id": id, "category": category, "agent_id": agent, "prompt": id, "depends_on": deps or [], **extra}


def _meta_llm(*decisions):
    """Patch target for build_llm_call: a meta-LLM answering scripted decisions."""
    queue = list(decisions)
    prompts: list[str] = []

    async def call(system, user):
        prompts.append(user)
        decision = queue.pop(0)
        if "actions" not in decision:
            decision = {"summary": "scripted", "actions": [decision]}
        return "```json\n" + json.dumps(decision) + "\n```", None

    factory = lambda config, settings: call  # noqa: E731
    factory.prompts = prompts  # type: ignore[attr-defined]
    return factory


def add(*jobs):
    return {"kind": "add_jobs", "jobs": list(jobs)}


FINISH = {"kind": "finish", "text": "done"}


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


async def _noop(*a, **k):
    return None


def _patched(meta, execute):
    return (
        patch("app.infrastructure.orchestration.dynamic.node.build_llm_call", meta),
        patch("app.steps.agent_executor.execute_agent_step", new=execute),
        patch("app.services.agent_cleanup.cleanup_run_agents", new=_noop),
    )


async def _setup(steps=STEPS):
    runner = _runner(steps)
    run = _run()
    repo = _Repo()
    await repo.create(run)
    return runner, run, repo


def test_dynamic_step_is_a_single_workflow_node():
    assert [s["id"] for s in _runner().steps] == ["orchestrator", "after"]


@pytest.mark.asyncio
async def test_plan_gate_then_jobs_run_as_child_runs_with_granted_data_sources():
    meta = _meta_llm(
        add(_job("code-a", "execution", "coder", owns=["a/**"], datasources=["crm"]),
            _job("code-b", "execution", "coder", owns=["b/**"]),
            _job("test", "validation", "tester", ["code-a", "code-b"])),
        FINISH,
    )
    executed: list[tuple[str, str, dict | None]] = []

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        executed.append((step["id"], run_id, kw.get("datasource_grants")))
        assert step["output_mapping"] and step["questions_mode"] == "return"
        if step["id"] == "test":
            return {"verdict": "pass", "summary": "green"}
        return {"summary": f"{step['id']} done", "branch": f"b-{step['id']}"}

    runner, run, repo = await _setup()
    p1, p2, p3 = _patched(meta, fake_execute)
    with p1, p2, p3:
        await stream_graph_to_pause(runner, run, repo, {"request": "add A and B"})
        assert run.status == "waiting_approval" and run.current_step == "orchestrator"
        dag = repo.runs[run.id].dynamic["orchestrator"]
        assert dag["status"] == "waiting_approval" and [d["kind"] for d in dag["decisions"]] == ["plan"]
        assert executed == []
        await stream_graph_to_pause(runner, run, repo, Command(resume={"approved": True}))

    assert run.status == "completed", run.state.get("error")
    assert sorted(e[0] for e in executed) == ["code-a", "code-b", "test"]
    grants = {e[0]: e[2] for e in executed}
    assert grants == {"code-a": {"crm": ["get_account", "list_accounts"]}, "code-b": None, "test": None}
    child_ids = {e[1] for e in executed}
    assert len(child_ids) == 3 and run.id not in child_ids
    for cid in child_ids:
        child = repo.runs[cid]
        assert child.kind == "job" and child.parent_run_id == run.id and child.status == "completed"
    assert repo.runs[run.id].dynamic["orchestrator"]["status"] == "completed"
    assert run.state["outcome"]["jobs"]["code-a"]["output"]["branch"] == "b-code-a"
    assert run.state["after_out"] == "after done"
    assert run.state["approval_history"][0]["approved"] is True


@pytest.mark.asyncio
async def test_rejected_plan_goes_back_to_the_meta_agent():
    meta = _meta_llm(add(_job("code", "execution", "coder")), add(_job("code2", "execution", "coder")), FINISH)

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        return {"summary": "ok"}

    runner, run, repo = await _setup()
    p1, p2, p3 = _patched(meta, fake_execute)
    with p1, p2, p3:
        await stream_graph_to_pause(runner, run, repo, {"request": "r"})
        await stream_graph_to_pause(runner, run, repo, Command(resume={"approved": False, "reason": "smaller"}))
        assert run.status == "waiting_approval"
        assert repo.runs[run.id].dynamic["orchestrator"]["replans"] == 1
        assert "smaller" in meta.prompts[1]
        await stream_graph_to_pause(runner, run, repo, Command(resume={"approved": True}))
    assert run.status == "completed"
    assert list(run.state["outcome"]["jobs"]) == ["code2"]


@pytest.mark.asyncio
async def test_running_agent_asks_and_the_meta_agent_answers_it_live():
    """agent -> /agent/question -> meta-agent -> answer channel -> the same running agent."""
    from app.infrastructure.orchestration.dynamic.hub import route_live_question
    from app.services import agent_inbox

    meta = _meta_llm(add(_job("code", "execution", "coder")), {"kind": "answer", "job_id": "code", "text": "spaces"}, FINISH)

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        assert route_live_question("parent-run-0001", run_id, "tabs or spaces?")
        await asyncio.wait_for(agent_inbox.event_for(run_id).wait(), 5)
        return {"summary": f"used {agent_inbox.answers.pop(run_id)}"}

    runner, run, repo = await _setup([{**STEPS[0], "automation": "auto"}, STEPS[1]])
    p1, p2, p3 = _patched(meta, fake_execute)
    with p1, p2, p3:
        await stream_graph_to_pause(runner, run, repo, {"request": "r"})
    assert run.status == "completed", run.state.get("error")
    assert run.state["outcome"]["jobs"]["code"]["output"]["summary"] == "used spaces"
    assert "tabs or spaces?" in meta.prompts[1]
    child_id = repo.runs[run.id].dynamic["orchestrator"]["jobs"][0]["attempts"][0]["child_run_id"]
    assert repo.runs[child_id].state["_pending_answer"] == "spaces"


@pytest.mark.asyncio
async def test_agent_questions_in_its_result_go_to_the_meta_agent_not_a_human():
    from app.steps.agent_executor import AgentNeedsInput

    meta = _meta_llm(add(_job("code", "execution", "coder")), {"kind": "answer", "job_id": "code", "text": "postgres"}, FINISH)
    asked = {"n": 0}

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        asked["n"] += 1
        if asked["n"] == 1:
            raise AgentNeedsInput(["which db?"])
        assert state["_clarification_answers"] == {"which db?": "postgres"}
        return {"summary": "done"}

    runner, run, repo = await _setup([{**STEPS[0], "automation": "auto"}, STEPS[1]])
    p1, p2, p3 = _patched(meta, fake_execute)
    with p1, p2, p3:
        await stream_graph_to_pause(runner, run, repo, {"request": "r"})
    assert run.status == "completed", run.state.get("error")
    assert "which db?" in meta.prompts[1]


@pytest.mark.asyncio
async def test_validator_change_request_rewinds_to_the_researcher_across_child_runs():
    meta = _meta_llm(
        add(_job("research", "planning", "researcher"), _job("code", "execution", "coder", ["research"]),
            _job("review", "validation", "tester", ["code"])),
        {"kind": "rewind", "job_ids": ["research"], "text": "find the streaming API"},
        FINISH,
    )
    seen: list[tuple[str, int]] = []
    reviews = iter([{"verdict": "fail", "feedback": "batch is wrong"}, {"verdict": "pass"}])

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        seen.append((step["id"], state["job"]["attempt"]))
        return next(reviews) if step["id"] == "review" else {"summary": f"{step['id']} done"}

    runner, run, repo = await _setup([{**STEPS[0], "automation": "auto"}, STEPS[1]])
    p1, p2, p3 = _patched(meta, fake_execute)
    with p1, p2, p3:
        await stream_graph_to_pause(runner, run, repo, {"request": "r"})
    assert run.status == "completed", run.state.get("error")
    assert seen == [("research", 1), ("code", 1), ("review", 1), ("research", 2), ("code", 2), ("review", 2)]
    assert "batch is wrong" in meta.prompts[1]


@pytest.mark.asyncio
async def test_authored_job_chain_longer_than_the_default_recursion_limit():
    """15 sequential jobs are 30+ supersteps of the job subgraph: more than LangGraph's default 25."""
    chain = [
        _job(f"part-{i}", "execution", "coder", [f"part-{i - 1}"] if i else [])
        for i in range(15)
    ]
    steps = [{**STEPS[0], "automation": "auto", "plan": chain, "jobs": {"execution": {"max": 15}}, "next": "END"}]
    executed: list[str] = []

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        executed.append(step["id"])
        return {"summary": f"{step['id']} done"}

    runner, run, repo = await _setup(steps)
    p1, p2, p3 = _patched(_meta_llm(), fake_execute)
    with p1, p2, p3:
        await stream_graph_to_pause(runner, run, repo, {"request": "build it"})
    assert run.status == "completed", run.state.get("error")
    assert executed == [f"part-{i}" for i in range(15)]
    dag = repo.runs[run.id].dynamic["orchestrator"]
    assert dag["status"] == "completed" and all(j["created_by"] == "author" for j in dag["jobs"])
    first = dag["jobs"][0]["attempts"][0]
    assert first["child_run_id"] and first["status"] == "finished"


@pytest.mark.asyncio
async def test_interrupted_run_resumes_without_rerunning_finished_jobs():
    """The workflow task dies mid-run; resuming from the checkpoint reruns only the unfinished job."""
    from app.infrastructure.orchestration.dynamic import hub as hubs

    meta = _meta_llm(add(_job("code-a", "execution", "coder"), _job("code-b", "execution", "coder")), FINISH)
    executed: list[str] = []
    b_started = asyncio.Event()
    first_b = {"held": True}

    async def fake_execute(step, state, backend, run_id, cb, **kw):
        executed.append(step["id"])
        if step["id"] == "code-b" and first_b["held"]:
            first_b["held"] = False
            b_started.set()
            await asyncio.sleep(3600)
        return {"summary": f"{step['id']} done"}

    runner, run, repo = await _setup([{**STEPS[0], "automation": "auto"}, STEPS[1]])
    p1, p2, p3 = _patched(meta, fake_execute)
    with p1, p2, p3:
        task = asyncio.create_task(stream_graph_to_pause(runner, run, repo, {"request": "r"}))
        await asyncio.wait_for(b_started.wait(), 5)
        for _ in range(100):  # code-a reported
            jobs = (repo.runs[run.id].dynamic.get("orchestrator") or {}).get("jobs") or []
            if any(j["id"] == "code-a" and j["status"] == "finished" for j in jobs):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not hubs._HUBS  # the step's attempts were stopped with it
        await stream_graph_to_pause(runner, run, repo, None)
    assert run.status == "completed", run.state.get("error")
    assert sorted(executed) == ["code-a", "code-b", "code-b"]
    b = next(j for j in repo.runs[run.id].dynamic["orchestrator"]["jobs"] if j["id"] == "code-b")
    assert [a["reason"] for a in b["attempts"]] == ["initial", "restart"]


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
                  {"id": "review", "category": "validation", "agent_id": "t", "prompt": "p",
                   "depends_on": ["ghost"], "datasources": ["billing"]}]},
    ])
    text = " | ".join(errors)
    assert "agent_pool is empty" in text
    assert "no agent in the pool may fill execution jobs (jobs.execution.min is 1)" in text
    assert "step 'c': automation" in text
    assert "step 'd': plan: job 'code': agent 't' may only fill" in text
    assert "depends on unknown or removed job 'ghost'" in text
    assert "data source 'billing' is not enabled" in text
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
