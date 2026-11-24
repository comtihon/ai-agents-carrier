"""How a workflow uses an agent: runtime and addons per use (agent_usage)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.application.visibility import (
    InvalidReference, ReferenceGuard, RunnerLookup, VisibilityScopedBackend, datasource_operations, references,
)
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.agent_usage import RuntimeSelection, apply_usage, merge_addons, parse_addons, parse_runtime
from app.domain.models.dynamic import DynamicConfig
from app.domain.models.runner import Runner
from app.infrastructure.auth.authorization import Subject, reset_current_subject, set_current_subject

ROLE = AgentDefinition(
    id="coder", default_runtime="k8s", protocol="acp", agent_input={"system_prompt": "code"},
    addons=parse_addons([{"type": "s3", "bucket": "b", "path": "p"},
                         {"type": "datasource", "source_id": "jira", "allowed_operations": ["get"]},
                         {"type": "datasource", "source_id": "crm", "allowed_operations": ["list"]}]),
)


def test_default_or_missing_runtime_keeps_the_agent():
    assert apply_usage(ROLE) is ROLE
    assert apply_usage(ROLE, {"kind": "default"}) is ROLE


def test_runner_runtime_moves_the_agent_onto_a_machine():
    a = apply_usage(ROLE, {"kind": "runner", "runner_id": "rnr-1", "acp_agent": "codex"})
    assert (a.default_runtime, a.runner_id, a.acp_agent, a.protocol, a.pool_id) == ("runner", "rnr-1", "codex", "acp", None)
    assert a.agent_input == ROLE.agent_input and ROLE.default_runtime == "k8s"


def test_pool_runtime():
    a = apply_usage(ROLE, {"kind": "pool", "pool_id": "eng"})
    assert (a.default_runtime, a.pool_id, a.runner_id, a.protocol) == ("pool", "eng", None, "acp")


def test_plain_runtime_and_legacy_override():
    assert apply_usage(ROLE, {"kind": "docker"}).default_runtime == "docker"
    assert apply_usage(ROLE, None, None, "local").default_runtime == "local"
    machine = AgentDefinition(id="m", default_runtime="runner", runner_id="r", protocol="acp")
    # A legacy override never moved an agent off its machine; an explicit runtime does.
    assert apply_usage(machine, None, None, "k8s") is machine
    assert apply_usage(machine, {"kind": "pool", "pool_id": "p"}).default_runtime == "pool"


def test_runtime_selection_needs_its_target():
    with pytest.raises(ValueError):
        RuntimeSelection(kind="runner")
    with pytest.raises(ValueError):
        RuntimeSelection(kind="pool")
    assert parse_runtime({"kind": "default"}) is None


def test_addons_override_same_type_and_merge_data_sources_by_source():
    a = apply_usage(ROLE, None, [{"type": "s3", "bucket": "other", "path": "q"},
                                 {"type": "datasource", "source_id": "jira", "allowed_operations": ["get", "move"]},
                                 {"type": "mcp", "servers": {"x": True}}])
    by = {(x.type, getattr(x, "source_id", None)): x for x in a.addons}
    assert by[("s3", None)].bucket == "other"
    assert by[("datasource", "jira")].allowed_operations == ["get", "move"]
    assert by[("datasource", "crm")].allowed_operations == ["list"]
    assert ("mcp", None) in by and len(a.addons) == 4
    assert len(ROLE.addons) == 3  # the definition is untouched
    assert merge_addons(ROLE.addons, []) == ROLE.addons


def test_pool_entry_carries_runtime_and_addons_and_validates_them():
    cfg = DynamicConfig.from_step({"id": "d", "type": "dynamic", "agent_pool": [
        {"agent_id": "coder", "runtime": {"kind": "pool", "pool_id": "eng"},
         "addons": [{"type": "datasource", "source_id": "jira", "allowed_operations": ["get"]}]}]})
    assert cfg.agent_pool[0].runtime == {"kind": "pool", "pool_id": "eng"}
    with pytest.raises(ValueError):
        DynamicConfig.from_step({"id": "d", "type": "dynamic", "agent_pool": [{"agent_id": "c", "runtime": {"kind": "runner"}}]})


def _wf(steps):
    return SimpleNamespace(id="wf", steps=steps, visibility="team", team_id="t", owner_id="lead")


def test_workflow_references_runtimes_and_addon_sources():
    wf = _wf([
        {"id": "a", "type": "langgraph-agent", "agent_id": "coder", "runtime": {"kind": "runner", "runner_id": "rnr-1"},
         "addons": [{"type": "datasource", "source_id": "jira", "allowed_operations": ["move"]}]},
        {"id": "d", "type": "dynamic", "agent_pool": [{"agent_id": "rev", "runtime": {"kind": "pool", "pool_id": "eng"}}]},
    ])
    refs = set(references("workflow", wf))
    assert {("agent", "coder"), ("runner", "rnr-1"), ("datasource", "jira"), ("agent", "rev"), ("pool", "eng")} <= refs
    assert datasource_operations("workflow", wf) == {"jira": {"move"}}


class _Runners:
    def __init__(self, runners):
        self.runners = {r.id: r for r in runners}

    async def get_runner(self, rid):
        return self.runners.get(rid)

    async def list_runners(self):
        return list(self.runners.values())


class _Mem:
    def __init__(self):
        self.docs = {}

    async def list(self):
        return list(self.docs.values())

    async def get(self, i):
        return self.docs.get(i)

    async def create(self, d):
        self.docs[d.id] = d
        return d

    async def update(self, i, d):
        self.docs[i] = d
        return d

    async def delete(self, i):
        self.docs.pop(i, None)


@pytest.mark.asyncio
async def test_a_team_workflow_cannot_run_an_agent_on_a_private_machine():
    from app.domain.models.workflow_definition import WorkflowDefinition
    runners = _Runners([
        Runner(id="mine", name="m", owner_id="lead", visibility="private"),
        Runner(id="shared", name="s", owner_id="lead", visibility="team", team_id="t-eng"),
    ])
    workflows = _Mem()
    guard = ReferenceGuard(lambda kind: RunnerLookup(runners) if kind == "runner" else None)
    scoped = VisibilityScopedBackend(workflows, "workflow", guard)
    token = set_current_subject(Subject(id="lead", name="Lead", team_id="t-eng"))
    try:
        def wf(runner_id):
            return WorkflowDefinition(id=f"wf-{runner_id}", name="w", visibility="team", steps=[
                {"id": "a", "type": "langgraph-agent", "agent_id": "coder",
                 "runtime": {"kind": "runner", "runner_id": runner_id}}])
        with pytest.raises(InvalidReference):
            await scoped.create(wf("mine"))
        await scoped.create(wf("shared"))
    finally:
        reset_current_subject(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime,expect", [
    ({"kind": "pool", "pool_id": "eng"}, ("pool", "pool", "eng", None)),
    ({"kind": "runner", "runner_id": "rnr-1", "acp_agent": "codex"}, ("acp", "runner", None, "rnr-1")),
])
async def test_execute_agent_step_runs_the_use_where_the_step_says(monkeypatch, runtime, expect):
    from app.steps import agent_executor

    seen = {}

    async def fake_pool(**kw):
        seen["via"] = "pool"
        seen["agent"] = kw["agent_def"]
        return {"result": '{"summary": "ok"}'}

    async def fake_acp(**kw):
        seen["via"] = "acp"
        seen["agent"] = kw["agent_def"]
        return {"result": '{"summary": "ok"}'}

    monkeypatch.setattr("app.steps.acp_pool_executor.run_pool_agent", fake_pool)
    monkeypatch.setattr("app.steps.acp_executor.run_acp_agent", fake_acp)

    class Backend:
        async def get(self, _id):
            return ROLE

    out = await agent_executor.execute_agent_step(
        {"id": "code", "agent_id": "coder", "output_mapping": {"summary": "summary"}, "runtime": runtime,
         "addons": [{"type": "s3", "bucket": "wf", "path": "x"}]},
        {"request": "go"}, Backend(), "run-9", "http://carrier", settings=agent_executor.get_settings(), use_meta_llm=False,
    )
    assert out["summary"] == "ok"
    a = seen["agent"]
    assert (seen["via"], a.default_runtime, a.pool_id, a.runner_id) == expect
    assert any(x.type == "s3" and x.bucket == "wf" for x in a.addons)
