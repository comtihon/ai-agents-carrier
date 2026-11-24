"""Runtime pools: membership rules, the visibility of pool agents, member
assignment (round robin, sticky per run, failover), the pool step executor,
and the REST routes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import app.application.runtime_pools as rp
from app.application.runners import RunnerHub
from app.application.runtime_pools import NoMemberAvailable, add_member, assign_member, present, remove_member
from app.application.visibility import (
    InvalidReference,
    NotOwner,
    ReferenceGuard,
    StillReferenced,
    VisibilityScopedBackend,
)
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.runner import AgentLimit, Runner
from app.domain.models.runtime_pool import RuntimePool
from app.infrastructure.auth.authorization import Subject, reset_current_subject, set_current_subject
from app.infrastructure.persistence.runner_backend import InMemoryRunnerBackend
from app.infrastructure.persistence.runtime_pool_backend import InMemoryRuntimePoolBackend


class MemoryAgents:
    def __init__(self) -> None:
        self.docs: dict[str, AgentDefinition] = {}

    async def list(self):
        return list(self.docs.values())

    async def get(self, agent_id):
        return self.docs.get(agent_id)

    async def create(self, d):
        d.touch()
        self.docs[d.id] = d
        return d

    async def update(self, agent_id, d):
        d.id = agent_id
        return await self.create(d)

    async def delete(self, agent_id):
        self.docs.pop(agent_id, None)


class Session:
    closed = False


class World:
    """Raw and scoped backends, a hub with machines that can go on and offline."""

    def __init__(self) -> None:
        self.agents_raw = MemoryAgents()
        self.pools_raw = InMemoryRuntimePoolBackend()
        raw = {"agent": self.agents_raw, "pool": self.pools_raw}
        guard = ReferenceGuard(lambda kind: raw.get(kind))
        self.agents = VisibilityScopedBackend(self.agents_raw, "agent", guard)
        self.pools = VisibilityScopedBackend(self.pools_raw, "pool", guard)
        self.hub = RunnerHub()
        self.hub.backend = InMemoryRunnerBackend()
        self.hub.pool_backend = self.pools_raw

    async def machine(self, owner: str, name: str, *, online: bool = True, visibility: str = "private") -> AgentDefinition:
        runner = Runner(id=f"rnr-{name}", name=name, agents=["claude"], default_agent="claude",
                        owner_id=owner, owner_name=owner.title(), visibility=visibility)
        await self.hub.backend.save_runner(runner)
        if online:
            self.online(runner.id)
        agent = AgentDefinition(id=f"{name}-claude", name=f"claude on {name}", default_runtime="runner",
                                runner_id=runner.id, protocol="acp", acp_agent="claude",
                                owner_id=owner, owner_name=owner.title(), visibility="private")
        self.agents_raw.docs[agent.id] = agent
        return agent

    def online(self, runner_id: str) -> None:
        self.hub.sessions[runner_id] = Session()

    def offline(self, runner_id: str) -> None:
        self.hub.sessions.pop(runner_id, None)

    async def team_pool(self, pool_id: str = "eng") -> RuntimePool:
        pool = RuntimePool(id=pool_id, name="Engineering", visibility="team", team_id="t-eng",
                           owner_id="lead", owner_name="Lead")
        return await self.pools_raw.create(pool)


def as_user(user: str | None, team: str | None = "t-eng", role: str | None = None, admin: bool = False):
    subject = None if user is None else Subject(
        id=user, name=user.title(), team_id=team, team_role=role, is_super_admin=admin,
    )
    return set_current_subject(subject)


@pytest.fixture
def world():
    return World()


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(rp, "_POLL_S", 0.01)


# ─── membership ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_owner_adds_own_private_agent(world):
    a = await world.machine("ann", "ann-laptop")
    pool = await world.team_pool()
    token = as_user("ann")
    try:
        pool = await add_member(world.pools_raw, pool, a)
    finally:
        reset_current_subject(token)
    stored = await world.pools_raw.get("eng")
    assert stored.member_ids() == [a.id]
    assert stored.members[0].added_by == "ann"


@pytest.mark.asyncio
async def test_nobody_else_may_add_a_private_agent_not_even_a_super_admin(world):
    a = await world.machine("ann", "ann-laptop")
    pool = await world.team_pool()
    for user, kwargs in (("bob", {}), ("lead", {"role": "admin"}), ("root", {"admin": True})):
        token = as_user(user, **kwargs)
        try:
            with pytest.raises(NotOwner):
                await add_member(world.pools_raw, pool, a)
        finally:
            reset_current_subject(token)
    assert (await world.pools_raw.get("eng")).members == []


@pytest.mark.asyncio
async def test_only_runner_agents_may_join(world):
    pool = await world.team_pool()
    plain = AgentDefinition(id="pod", default_runtime="k8s", owner_id="ann", visibility="private")
    token = as_user("ann")
    try:
        with pytest.raises(rp.PoolError):
            await add_member(world.pools_raw, pool, plain)
    finally:
        reset_current_subject(token)


@pytest.mark.asyncio
async def test_pool_outside_callers_scope_is_not_found(world):
    a = await world.machine("ann", "ann-laptop")
    pool = await world.team_pool()
    token = as_user("ann", team="t-other")
    try:
        with pytest.raises(rp.ResourceNotFound):
            await add_member(world.pools_raw, pool, a)
    finally:
        reset_current_subject(token)


@pytest.mark.asyncio
async def test_member_removed_by_owner_or_team_admin_not_by_other_member(world):
    a = await world.machine("ann", "ann-laptop")
    b = await world.machine("bob", "bob-laptop")
    pool = await world.team_pool()
    for user, agent in (("ann", a), ("bob", b)):
        token = as_user(user)
        try:
            pool = await add_member(world.pools_raw, pool, agent)
        finally:
            reset_current_subject(token)
    token = as_user("bob")
    try:
        with pytest.raises(NotOwner):
            await remove_member(world.pools_raw, pool, a.id, a)
        pool = await remove_member(world.pools_raw, pool, b.id, b)
    finally:
        reset_current_subject(token)
    token = as_user("lead2", role="admin")
    try:
        pool = await remove_member(world.pools_raw, pool, a.id, a)
    finally:
        reset_current_subject(token)
    assert (await world.pools_raw.get("eng")).members == []


@pytest.mark.asyncio
async def test_scoped_save_cannot_add_members_or_rewind_cursor(world):
    a = await world.machine("ann", "ann-laptop")
    pool = await world.team_pool()
    await world.pools_raw.set_members("eng", [rp.PoolMember(agent_id=a.id)])
    await world.pools_raw.next_cursor("eng")
    token = as_user("lead")
    try:
        forged = RuntimePool(id="eng", name="renamed", visibility="team",
                             members=[rp.PoolMember(agent_id="someone-elses")], cursor=0)
        await world.pools.update("eng", forged)
    finally:
        reset_current_subject(token)
    stored = await world.pools_raw.get("eng")
    assert stored.name == "renamed"
    assert stored.member_ids() == [a.id]
    assert stored.cursor == 1


# ─── visibility of pool agents ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_team_agent_may_use_its_teams_pool(world):
    await world.team_pool()
    token = as_user("lead")
    try:
        await world.agents.create(AgentDefinition(
            id="coder", default_runtime="pool", pool_id="eng", protocol="acp", visibility="team",
        ))
    finally:
        reset_current_subject(token)
    assert world.agents_raw.docs["coder"].team_id == "t-eng"


@pytest.mark.asyncio
async def test_team_agent_may_not_use_a_private_pool(world):
    await world.pools_raw.create(RuntimePool(id="mine", visibility="private", owner_id="lead"))
    token = as_user("lead")
    try:
        with pytest.raises(InvalidReference):
            await world.agents.create(AgentDefinition(
                id="coder", default_runtime="pool", pool_id="mine", protocol="acp", visibility="team",
            ))
    finally:
        reset_current_subject(token)


@pytest.mark.asyncio
async def test_pool_used_by_team_agent_cannot_be_narrowed(world):
    await world.team_pool()
    token = as_user("lead")
    try:
        await world.agents.create(AgentDefinition(
            id="coder", default_runtime="pool", pool_id="eng", protocol="acp", visibility="team",
        ))
        pool = await world.pools.get("eng")
        pool.visibility = "private"
        with pytest.raises(StillReferenced):
            await world.pools.update("eng", pool)
    finally:
        reset_current_subject(token)


# ─── assignment ──────────────────────────────────────────────────────────────

async def _pool_of(world, *names, online=True):
    agents = [await world.machine(n, f"{n}-laptop", online=online) for n in names]
    await world.team_pool()
    await world.pools_raw.set_members("eng", [rp.PoolMember(agent_id=a.id) for a in agents])
    return agents


async def _assign(world, run_id, **kw):
    kw.setdefault("grace_s", 0.01)
    kw.setdefault("wait_s", 0.05)
    member, assignment = await assign_member(
        "eng", run_id, pool_backend=world.pools_raw, agent_backend=world.agents_raw, hub=world.hub, **kw,
    )
    return member, assignment


@pytest.mark.asyncio
async def test_runs_take_members_round_robin_and_keep_them(world):
    a, b, c = await _pool_of(world, "ann", "bob", "cid")
    picks = [(await _assign(world, f"run-{i}"))[0].agent_id for i in range(4)]
    assert picks == [a.id, b.id, c.id, a.id]
    # Every later step of run-1 stays on its member.
    for _ in range(3):
        assert (await _assign(world, "run-1"))[0].agent_id == b.id


@pytest.mark.asyncio
async def test_offline_members_are_skipped(world):
    a, b = await _pool_of(world, "ann", "bob")
    world.offline(a.runner_id)
    picks = {(await _assign(world, f"run-{i}"))[0].agent_id for i in range(3)}
    assert picks == {b.id}


@pytest.mark.asyncio
async def test_member_out_of_quota_is_only_a_fallback(world):
    a, b = await _pool_of(world, "ann", "bob")
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    await world.hub.backend.set_limit(a.runner_id, "claude", AgentLimit(until=later))
    assert {(await _assign(world, f"run-{i}"))[0].agent_id for i in range(3)} == {b.id}
    world.offline(b.runner_id)
    assert (await _assign(world, "run-x"))[0].agent_id == a.id


@pytest.mark.asyncio
async def test_run_moves_on_when_its_member_goes_offline(world):
    a, b = await _pool_of(world, "ann", "bob")
    assert (await _assign(world, "run-1"))[0].agent_id == a.id
    world.offline(a.runner_id)
    member, assignment = await _assign(world, "run-1")
    assert member.agent_id == b.id
    assert assignment.previous == [a.id]
    # Back online, the run does not move back: it moves only when it must.
    world.online(a.runner_id)
    assert (await _assign(world, "run-1"))[0].agent_id == b.id


@pytest.mark.asyncio
async def test_failed_member_is_left_even_if_it_reconnected(world):
    a, b = await _pool_of(world, "ann", "bob")
    await _assign(world, "run-1")
    member, _ = await _assign(world, "run-1", failed=a.id)
    assert member.agent_id == b.id


@pytest.mark.asyncio
async def test_parallel_steps_agree_on_one_member(world):
    import asyncio
    await _pool_of(world, "ann", "bob", "cid")
    picks = await asyncio.gather(*[_assign(world, "run-1") for _ in range(5)])
    assert len({m.agent_id for m, _ in picks}) == 1


@pytest.mark.asyncio
async def test_no_member_online_fails_after_waiting(world):
    await _pool_of(world, "ann", "bob", online=False)
    with pytest.raises(NoMemberAvailable):
        await _assign(world, "run-1")


@pytest.mark.asyncio
async def test_a_member_coming_online_while_waiting_is_taken(world):
    import asyncio
    (a,) = await _pool_of(world, "ann", online=False)

    async def later():
        await asyncio.sleep(0.03)
        world.online(a.runner_id)

    waited = []

    async def on_wait(pool):
        waited.append(pool.id)

    task = asyncio.create_task(later())
    member, _ = await _assign(world, "run-1", wait_s=2, on_wait=on_wait)
    await task
    assert member.agent_id == a.id
    assert waited == ["eng"]


@pytest.mark.asyncio
async def test_present_counts_online_members_and_shows_holders(world):
    a, b = await _pool_of(world, "ann", "bob")
    world.offline(b.runner_id)
    await _assign(world, "run-1")
    pool = await world.pools_raw.get("eng")
    data = await present(pool, agent_backend=world.agents_raw, hub=world.hub, pool_backend=world.pools_raw)
    assert (data["member_count"], data["online_count"], data["available_count"]) == (2, 1, 1)
    by_id = {m["agent_id"]: m for m in data["members"]}
    assert by_id[a.id]["online"] and not by_id[b.id]["online"]
    assert [r["run_id"] for r in by_id[a.id]["runs"]] == ["run-1"]
    # A private member shows its name and owner, never its definition.
    assert "agent_input" not in by_id[a.id]


# ─── pool step executor ──────────────────────────────────────────────────────

class Tasks:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    async def update_task(self, key, fields):
        self.updates.append(fields)


def _pool_agent() -> AgentDefinition:
    return AgentDefinition(id="coder", default_runtime="pool", pool_id="eng", protocol="acp",
                           agent_input={"system_prompt": "write code"})


@pytest.mark.asyncio
async def test_pool_step_runs_as_runner_step_on_the_member(world, monkeypatch):
    from app.steps import acp_executor
    from app.steps.acp_pool_executor import run_pool_agent
    a, _ = await _pool_of(world, "ann", "bob")
    seen: list[Any] = []

    async def fake_run(**kw):
        seen.append(kw["agent_def"])
        return {"result": "done"}

    monkeypatch.setattr(acp_executor, "run_acp_agent", fake_run)
    out = await run_pool_agent(agent_def=_pool_agent(), run_id="run-1", task_key="t", step={"id": "code"},
                               settings=object(), runner_hub=world.hub, agent_backend=world.agents_raw)
    assert out == {"result": "done"}
    (d,) = seen
    assert (d.default_runtime, d.runner_id, d.acp_agent) == ("runner", a.runner_id, "claude")
    assert d.agent_input["system_prompt"] == "write code"
    assert d.agent_input["runner_wait_s"] == rp.DEFAULT_GRACE_S


@pytest.mark.asyncio
async def test_pool_step_restarts_on_next_member_when_machine_goes_offline(world, monkeypatch):
    from app.steps import acp_executor
    from app.steps.acp_pool_executor import run_pool_agent
    a, b = await _pool_of(world, "ann", "bob")
    tasks = Tasks()
    runners: list[str] = []

    async def fake_run(**kw):
        runners.append(kw["agent_def"].runner_id)
        if len(runners) == 1:
            world.offline(a.runner_id)
            raise RuntimeError("machine 'ann-laptop' is offline")
        return {"result": "done"}

    monkeypatch.setattr(acp_executor, "run_acp_agent", fake_run)
    agent = _pool_agent()
    agent.agent_input["pool_grace_s"] = 0.01
    out = await run_pool_agent(agent_def=agent, run_id="run-1", task_key="t", step={"id": "code"},
                               settings=object(), runner_hub=world.hub, agent_backend=world.agents_raw,
                               agent_task_repository=tasks)
    assert out == {"result": "done"}
    assert runners == [a.runner_id, b.runner_id]
    assert {"status": "working", "acp": None} in tasks.updates
    assert (await world.pools_raw.get_assignment("eng", "run-1")).agent_id == b.id


@pytest.mark.asyncio
async def test_pool_step_error_on_online_machine_is_not_a_failover(world, monkeypatch):
    from app.steps import acp_executor
    from app.steps.acp_pool_executor import run_pool_agent
    await _pool_of(world, "ann", "bob")

    async def fake_run(**kw):
        raise ValueError("agent said no")

    monkeypatch.setattr(acp_executor, "run_acp_agent", fake_run)
    with pytest.raises(ValueError):
        await run_pool_agent(agent_def=_pool_agent(), run_id="run-1", task_key="t", step={"id": "code"},
                             settings=object(), runner_hub=world.hub, agent_backend=world.agents_raw)


class Runs:
    def __init__(self, parents: dict[str, str | None]) -> None:
        self.parents = parents

    async def get(self, run_id):
        from types import SimpleNamespace
        if run_id not in self.parents:
            return None
        return SimpleNamespace(id=run_id, parent_run_id=self.parents[run_id])


@pytest.mark.asyncio
async def test_jobs_of_one_workflow_run_share_its_member(world, monkeypatch):
    from app.steps import acp_executor
    from app.steps.acp_pool_executor import root_run_id, run_pool_agent
    a, b, c = await _pool_of(world, "ann", "bob", "cid")
    runs = Runs({"wf-1": None, "job-1": "wf-1", "job-2": "wf-1", "sub-1": "wf-1", "job-3": "sub-1",
                 "wf-2": None, "job-4": "wf-2"})
    assert await root_run_id(runs, "job-3") == "wf-1"
    assert await root_run_id(runs, "unknown") == "unknown"
    seen: dict[str, str] = {}

    async def fake_run(**kw):
        seen[kw["run_id"]] = kw["agent_def"].runner_id
        return {"result": "ok"}

    monkeypatch.setattr(acp_executor, "run_acp_agent", fake_run)
    for child in ("job-1", "job-2", "job-3", "job-4"):
        await run_pool_agent(agent_def=_pool_agent(), run_id=child, task_key=child, step={"id": child},
                             settings=object(), runner_hub=world.hub, agent_backend=world.agents_raw,
                             run_repository=runs)
    # The child run id still reaches the step (its own session and workspace).
    assert set(seen) == {"job-1", "job-2", "job-3", "job-4"}
    assert seen["job-1"] == seen["job-2"] == seen["job-3"] == a.runner_id
    assert seen["job-4"] == b.runner_id
    assert (await world.pools_raw.get_assignment("eng", "wf-1")).agent_id == a.id


@pytest.mark.asyncio
async def test_root_run_id_survives_a_parent_cycle():
    from app.steps.acp_pool_executor import root_run_id
    assert await root_run_id(Runs({"x": "y", "y": "x"}), "x") in ("x", "y")
