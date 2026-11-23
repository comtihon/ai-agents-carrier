"""Global concurrency limits on agent runtimes (app.application.concurrency)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.application import management_tools as core
from app.application.concurrency import (
    ConcurrencySlot,
    agent_key,
    configured_limits,
    describe,
    merge_agent_concurrency,
    parse_key,
    runner_key,
    runtime_for,
)
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.runner import Runner
from app.infrastructure.persistence.concurrency_backend import InMemoryConcurrencyBackend
from app.infrastructure.persistence.runner_backend import InMemoryRunnerBackend
from tests.test_acp_executor import AGENT_CONFIG, FakeProxy, FakeRuntime, Repo
from tests.test_agents_api import InMemoryAgentBackend, _build_container
from tests.test_runners import _with_test_subject

T0 = datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc)


def _entry(i: str, at: datetime = T0) -> dict:
    return {"id": i, "run_id": f"run-{i}", "step_id": "s", "agent_id": "a", "workflow_id": "wf", "enqueued_at": at}


# ─── the lease store ─────────────────────────────────────────────────────────


async def test_limit_one_admits_one_and_queues_the_rest_fifo():
    b = InMemoryConcurrencyBackend()
    first = await b.try_acquire("k", _entry("a", T0), 1, T0, 60)
    # "c" arrives at Mongo before "b" but was enqueued later: order is by enqueue time.
    c = await b.try_acquire("k", _entry("c", T0 + timedelta(seconds=2)), 1, T0, 60)
    bb = await b.try_acquire("k", _entry("b", T0 + timedelta(seconds=1)), 1, T0, 60)
    assert first.acquired and not c.acquired and not bb.acquired
    assert bb.position == 1
    assert (await b.try_acquire("k", _entry("c"), 1, T0, 60)).position == 2
    # "c" polls first after the slot frees, but "b" is ahead of it.
    await b.release("k", "a")
    assert not (await b.try_acquire("k", _entry("c"), 1, T0, 60)).acquired
    assert (await b.try_acquire("k", _entry("b"), 1, T0, 60)).acquired
    assert (await b.try_acquire("k", _entry("c"), 1, T0, 60)).position == 1


async def test_limit_two_holds_two_at_once():
    b = InMemoryConcurrencyBackend()
    got = [await b.try_acquire("k", _entry(i, T0 + timedelta(seconds=n)), 2, T0, 60) for n, i in enumerate("abc")]
    assert [g.acquired for g in got] == [True, True, False]
    assert got[2].holders == 2 and got[2].position == 1
    [row] = await b.list_keys(T0)
    assert [h["id"] for h in row["holders"]] == ["a", "b"] and [q["id"] for q in row["queue"]] == ["c"]


async def test_reacquire_with_the_same_id_keeps_the_slot():
    b = InMemoryConcurrencyBackend()
    assert (await b.try_acquire("k", _entry("a"), 1, T0, 60)).acquired
    assert (await b.try_acquire("k", _entry("a"), 1, T0, 60)).acquired
    assert not (await b.try_acquire("k", _entry("b"), 1, T0, 60)).acquired


async def test_expired_holder_frees_its_slot():
    b = InMemoryConcurrencyBackend()
    await b.try_acquire("k", _entry("dead"), 1, T0, 30)
    assert not (await b.try_acquire("k", _entry("b"), 1, T0 + timedelta(seconds=10), 30)).acquired
    # The crashed replica stopped heartbeating: past its TTL the waiter gets in.
    assert (await b.try_acquire("k", _entry("b"), 1, T0 + timedelta(seconds=31), 30)).acquired
    assert not await b.heartbeat("k", "dead", T0 + timedelta(seconds=31), 30)


async def test_heartbeat_keeps_a_holder_alive():
    b = InMemoryConcurrencyBackend()
    await b.try_acquire("k", _entry("a"), 1, T0, 30)
    assert await b.heartbeat("k", "a", T0 + timedelta(seconds=20), 30)
    assert not (await b.try_acquire("k", _entry("b"), 1, T0 + timedelta(seconds=40), 30)).acquired


async def test_force_takes_a_slot_past_the_limit():
    b = InMemoryConcurrencyBackend()
    await b.try_acquire("k", _entry("a"), 1, T0, 60)
    assert (await b.try_acquire("k", _entry("re"), 1, T0, 60, force=True)).acquired
    [row] = await b.list_keys(T0)
    assert len(row["holders"]) == 2


# ─── the slot ────────────────────────────────────────────────────────────────


def _slot(backend, i: str, limit: int | None = 1, **kw) -> ConcurrencySlot:
    return ConcurrencySlot("agent:x", limit, {"id": i, "run_id": i}, backend=backend, poll_s=0.01, ttl_s=5, **kw)


async def test_second_slot_waits_until_the_first_is_released():
    b = InMemoryConcurrencyBackend()
    waits: list[tuple[int, int]] = []

    async def on_wait(position, holders):
        waits.append((position, holders))

    first = _slot(b, "a")
    await first.acquire()
    second = _slot(b, "b", on_wait=on_wait)
    task = asyncio.ensure_future(second.acquire())
    await asyncio.sleep(0.05)
    assert not task.done() and waits == [(1, 1)]
    await first.release()
    await asyncio.wait_for(task, 1)
    assert second.held
    await second.release()
    assert await b.list_keys(datetime.now(timezone.utc)) == []


async def test_cancelled_waiter_leaves_the_queue():
    b = InMemoryConcurrencyBackend()
    first = _slot(b, "a")
    await first.acquire()
    task = asyncio.ensure_future(_slot(b, "b").acquire())
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [row] = await b.list_keys(datetime.now(timezone.utc))
    assert row["queue"] == [] and len(row["holders"]) == 1
    await first.release()


async def test_unlimited_slot_never_touches_the_store():
    b = InMemoryConcurrencyBackend()
    slot = _slot(b, "a", limit=None)
    async with slot:
        assert not slot.held
    assert b.docs == {}


async def test_release_on_error_inside_the_slot():
    b = InMemoryConcurrencyBackend()
    with pytest.raises(RuntimeError):
        async with _slot(b, "a"):
            raise RuntimeError("agent failed")
    assert await b.list_keys(datetime.now(timezone.utc)) == []


# ─── keys and limits ─────────────────────────────────────────────────────────


def _runner(**kw) -> Runner:
    return Runner(id="r1", name="mbp", agents=["claude", "codex"], default_agent="claude", **kw)


async def test_runtime_key_and_effective_limit():
    runner = _runner(agent_concurrency={"claude": 3})
    agents = InMemoryAgentBackend()
    mine = AgentDefinition(id="mine", default_runtime="runner", runner_id="r1", protocol="acp")
    other = AgentDefinition(id="other", default_runtime="runner", runner_id="r1", protocol="acp",
                            acp_agent="claude", max_concurrency=2)
    codex = AgentDefinition(id="codex", default_runtime="runner", runner_id="r1", protocol="acp",
                            acp_agent="codex", max_concurrency=1)
    for d in (mine, other, codex):
        await agents.create(d)
    # The default agent is claude; every definition using it counts.
    assert await runtime_for(mine, runner, agents) == ("runner:r1:claude", 2)
    assert await runtime_for(codex, runner, agents) == ("runner:r1:codex", 1)
    pod = AgentDefinition(id="pod", max_concurrency=4)
    assert await runtime_for(pod) == ("agent:pod", 4)
    assert await runtime_for(AgentDefinition(id="free")) == ("agent:free", None)
    assert await runtime_for(mine, _runner(), None) == ("runner:r1:claude", None)

    runners = InMemoryRunnerBackend()
    await runners.save_runner(runner)
    await agents.create(pod)
    assert await configured_limits(agents, runners) == {
        "runner:r1:claude": 2, "runner:r1:codex": 1, "agent:pod": 4,
    }
    assert parse_key(runner_key("r1", "claude")) == {"kind": "runner", "runner_id": "r1", "acp_agent": "claude", "agent_id": None}
    assert parse_key(agent_key("pod"))["agent_id"] == "pod"


def test_merge_agent_concurrency():
    assert merge_agent_concurrency({"a": 1}, {"b": 2, "a": None}) == {"b": 2}
    assert merge_agent_concurrency({"a": 1}, {"a": 0}) == {}
    with pytest.raises(ValueError):
        merge_agent_concurrency({}, {"a": -1})
    with pytest.raises(ValueError):
        merge_agent_concurrency({}, {" ": 1})


def test_max_concurrency_must_be_positive():
    with pytest.raises(ValueError):
        AgentDefinition(id="x", max_concurrency=0)


async def test_describe_hides_runs_the_caller_cannot_see():
    b = InMemoryConcurrencyBackend()
    now = datetime.now(timezone.utc)
    await b.try_acquire("agent:pod", {**_entry("a", now), "workflow_id": "secret"}, 1, now, 60)
    await b.try_acquire("agent:pod", {**_entry("b", now), "workflow_id": "open"}, 1, now, 60)
    agents = InMemoryAgentBackend()
    await agents.create(AgentDefinition(id="pod", max_concurrency=1))

    async def wf_visible(wf):
        return wf != "secret"

    [row] = await describe(agent_backend=agents, runner_backend=None, backend=b, workflow_visible=wf_visible)
    assert row["key"] == "agent:pod" and row["kind"] == "agent" and row["limit"] == 1
    assert row["holders"][0]["hidden"] and row["holders"][0]["run_id"] is None
    assert row["queue"][0] | {} == row["queue"][0] and row["queue"][0]["run_id"] == "run-b"
    assert row["queue"][0]["position"] == 1


# ─── executor gating ─────────────────────────────────────────────────────────


class GatedProxy(FakeProxy):
    """A proxy whose agent answers its first prompt only once released."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.prompted = asyncio.Event()

    async def _on_client(self, msg: dict) -> None:
        if msg.get("method") == "session/prompt" and self.prompts == 0:
            self.prompted.set()
            await self.release.wait()
        await super()._on_client(msg)


async def test_acp_step_waits_for_the_runtime_slot():
    from app.steps.acp_executor import run_acp_agent

    backend = InMemoryConcurrencyBackend()
    agent = AgentDefinition(id="pi", default_runtime="k8s", protocol="acp", acp_agent="pi", max_concurrency=1)
    settings = SimpleNamespace(agent_concurrency_poll_s=0.01, agent_concurrency_ttl_s=30)
    repo = Repo()
    proxies, runtimes = [], []

    async def run(n: int):
        proxy = GatedProxy()
        await proxy.start()
        proxies.append(proxy)
        runtime = FakeRuntime(proxy)
        runtimes.append(runtime)
        repo.runs[f"run-{n}"] = SimpleNamespace(id=f"run-{n}", state={}, touch=lambda: None)
        try:
            return await run_acp_agent(
                step={"id": "code"}, agent_def=agent, input_data={"task": "x"}, agent_config=AGENT_CONFIG,
                runtime=runtime, run_id=f"run-{n}", task_key=f"run-{n}_code_0", callback_base_url="http://carrier",
                settings=settings, run_repository=repo, concurrency_backend=backend, workflow_id="wf",
            )
        finally:
            await proxy.stop()

    first = asyncio.ensure_future(run(1))
    while not proxies:
        await asyncio.sleep(0.01)
    await asyncio.wait_for(proxies[0].prompted.wait(), 5)
    second = asyncio.ensure_future(run(2))
    await asyncio.sleep(0.2)
    # The second run queues: nothing spawned, its state says where it stands.
    assert len(runtimes) == 2 and runtimes[1].spawned == []
    wait = repo.runs["run-2"].state["_queue_wait_code"]
    assert wait["key"] == "agent:pi" and wait["position"] == 1 and wait["limit"] == 1 and wait["holders"] == 1
    assert any("queued for agent:pi" in line for line in repo.runs["run-2"].state["_agent_progress_code"])

    proxies[0].release.set()
    assert (await asyncio.wait_for(first, 5))["result"]
    await asyncio.wait_for(proxies[1].prompted.wait(), 5)
    assert len(runtimes[1].spawned) == 1
    assert "_queue_wait_code" not in repo.runs["run-2"].state
    proxies[1].release.set()
    await asyncio.wait_for(second, 5)
    assert await backend.list_keys(datetime.now(timezone.utc)) == []


async def test_http_poll_step_waits_for_the_runtime_slot():
    from app.steps.agent_executor import execute_agent_step

    agent = AgentDefinition(id="pod", default_runtime="k8s", max_concurrency=1)
    agents = InMemoryAgentBackend()
    await agents.create(agent)
    released = {"run-1": asyncio.Event(), "run-2": asyncio.Event()}
    spawned: list[str] = []

    def response(status: str) -> MagicMock:
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {"status": status, "outputs": [{"type": "final", "content": {"result": "ok"}}] if status == "finished" else []}
        return r

    current: dict[str, str] = {}

    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, timeout=None):
            current[url] = json["run_id"]
            return response("working")

        async def get(self, url, timeout=None):
            run_id = current[url.rsplit("/", 1)[0] + "/start"]
            return response("finished" if released[run_id].is_set() else "working")

    async def spawn(agent_def, step, run_id, cb, extra_env=None):
        spawned.append(run_id)
        return f"http://{run_id}"

    runtime = MagicMock()
    runtime.has_container_for_run = AsyncMock(return_value=False)
    runtime.terminate_by_run_id = AsyncMock()
    runtime.rewrite_callback_url = MagicMock(return_value="http://localhost")
    runtime.spawn = spawn
    poll_settings = MagicMock(agent_poll_interval_seconds=0.01, agent_max_loops=3)
    settings = MagicMock(agent_concurrency_poll_s=0.01, agent_concurrency_ttl_s=30)
    settings.get_mcp_integrations.return_value = []
    settings.get_llm_integrations.return_value = []
    settings.get_forwardable_config.return_value = {}

    async def run(run_id: str):
        return await execute_agent_step(
            step={"id": "s", "agent_id": "pod"}, state={"request": "go"}, agent_backend=agents,
            run_id=run_id, callback_base_url="http://localhost", settings=settings, use_meta_llm=False,
        )

    with (
        patch("app.runtime.factory.get_runtime", return_value=runtime),
        patch("httpx.AsyncClient", Client),
        patch("app.steps.agent_executor.get_settings", return_value=poll_settings),
    ):
        first = asyncio.ensure_future(run("run-1"))
        await asyncio.sleep(0.1)
        second = asyncio.ensure_future(run("run-2"))
        await asyncio.sleep(0.1)
        assert spawned == ["run-1"]
        released["run-1"].set()
        await asyncio.wait_for(first, 5)
        await asyncio.sleep(0.1)
        assert spawned == ["run-1", "run-2"]
        released["run-2"].set()
        await asyncio.wait_for(second, 5)


# ─── API / MCP round trips ───────────────────────────────────────────────────


@pytest.fixture
async def api():
    agents = InMemoryAgentBackend()
    runners = InMemoryRunnerBackend()
    await runners.save_runner(_runner(owner_id="alice", visibility="private"))
    container = _build_container(agents)
    container.runner_backend = runners
    from app.api.app import create_app
    app = create_app()
    app.state.container = container
    transport = ASGITransport(app=_with_test_subject(app))
    async with AsyncClient(transport=transport, base_url="http://t/api/v1", headers={"x-test-user": "alice"}) as c:
        yield c, agents, runners


async def test_agent_api_round_trips_max_concurrency(api):
    c, agents, _ = api
    r = await c.post("/agents", json={"id": "pod", "name": "Pod", "max_concurrency": 2})
    assert r.status_code == 201, r.text
    assert r.json()["max_concurrency"] == 2
    # Omitted on PUT: kept.
    r = await c.put("/agents/pod", json={"name": "Pod 2"})
    assert r.status_code == 200, r.text
    assert r.json()["max_concurrency"] == 2
    r = await c.put("/agents/pod", json={"name": "Pod 2", "max_concurrency": None})
    assert r.json()["max_concurrency"] is None
    assert (await c.put("/agents/pod", json={"name": "x", "max_concurrency": 0})).status_code == 422
    assert (await c.get("/agents/pod")).json()["max_concurrency"] is None
    await c.put("/agents/pod", json={"name": "x", "max_concurrency": 5})
    [summary] = (await c.get("/agents", params={"view": "summary"})).json()
    assert summary["max_concurrency"] == 5


async def test_runner_api_sets_agent_concurrency_and_live_view(api):
    c, agents, runners = api
    r = await c.patch("/acp/runners/r1", json={"agent_concurrency": {"claude": 1, "codex": 2}})
    assert r.status_code == 200, r.text
    assert r.json()["agent_concurrency"] == {"claude": 1, "codex": 2}
    r = await c.patch("/acp/runners/r1", json={"agent_concurrency": {"codex": None}})
    assert r.json()["agent_concurrency"] == {"claude": 1}
    assert (await c.patch("/acp/runners/r1", json={"agent_concurrency": {"claude": -1}})).status_code == 400
    # Someone who cannot see the runner gets 404, and does not see its key.
    bob = {"x-test-user": "bob"}
    assert (await c.patch("/acp/runners/r1", json={"agent_concurrency": {"claude": 5}}, headers=bob)).status_code == 404

    r = await c.get("/agent-runtimes/concurrency")
    assert r.status_code == 200, r.text
    assert r.json() == [{
        "key": "runner:r1:claude", "kind": "runner", "runner_id": "r1", "acp_agent": "claude",
        "agent_id": None, "limit": 1, "holders": [], "queue": [],
    }]
    assert (await c.get("/agent-runtimes/concurrency", headers=bob)).json() == []


async def test_mcp_tools_set_and_show_concurrency():
    agents = InMemoryAgentBackend()
    runners = InMemoryRunnerBackend()
    await runners.save_runner(_runner())
    deps = core.ManagementDeps(registry=MagicMock(), run_repository=AsyncMock(), agent_backend=agents, runner_backend=runners)
    assert "created" in await core.create_agent(deps, "pod", "Pod", max_concurrency=3)
    assert (await agents.get("pod")).max_concurrency == 3
    assert "updated" in await core.update_agent(deps, "pod", name="Pod")
    assert (await agents.get("pod")).max_concurrency == 3
    await core.update_agent(deps, "pod", max_concurrency=0)
    assert (await agents.get("pod")).max_concurrency is None
    assert "positive" in await core.update_agent(deps, "pod", max_concurrency=-1)

    out = await core.update_runner_concurrency(deps, "r1", "claude", 1)
    assert "at most 1" in out
    assert (await runners.get_runner("r1")).agent_concurrency == {"claude": 1}
    listed = json.loads(await core.list_agent_concurrency(deps))
    assert listed[0]["key"] == "runner:r1:claude" and listed[0]["limit"] == 1
    assert "no concurrency cap" in await core.update_runner_concurrency(deps, "r1", "claude", 0)
