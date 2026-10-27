"""Startup recovery crash-loop guard.

A run that is resumed on startup and takes the server down with it (prod
incident: a run document grew past 16 MB, every resume raised
DocumentTooLarge / OOMKilled the pod -> CrashLoopBackOff) must not be resumed
forever. Each recovery is counted on the run document *before* it executes;
after ``run_recovery_max_attempts`` consecutive recoveries without progress
the next startup fails the run and disables its workflow instead.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
from app.infrastructure.persistence.mongo import INCOMPLETE_STATUSES, RecoveryCandidate
from app.infrastructure.tools.mcp_client import McpToolsProvider


class _FakeRunRepository:
    """In-memory stand-in for the recovery-related MongoGraphRunRepository API."""

    def __init__(self, runs: list[GraphRun]) -> None:
        self.docs: dict[str, GraphRun] = {r.id: r.model_copy(deep=True) for r in runs}
        self.events: list[tuple[str, str]] = []

    async def list_recovery_candidates(self) -> list[RecoveryCandidate]:
        return [
            RecoveryCandidate(
                id=r.id, graph_id=r.graph_id, status=r.status, kind=r.kind,
                recovery_attempts=r.recovery_attempts,
            )
            for r in self.docs.values()
            if r.status in INCOMPLETE_STATUSES
        ]

    async def begin_recovery(self, run_id: str) -> GraphRun | None:
        r = self.docs.get(run_id)
        if r is None or r.status not in INCOMPLETE_STATUSES:
            return None
        r.recovery_attempts += 1
        r.last_recovery_at = datetime.now(timezone.utc)
        self.events.append(("begin_recovery", run_id))
        return r.model_copy(deep=True)

    async def reset_recovery_attempts(self, run_id: str) -> None:
        self.docs[run_id].recovery_attempts = 0

    async def fail_incomplete(self, run_id: str, error: str) -> bool:
        r = self.docs[run_id]
        if r.status not in INCOMPLETE_STATUSES:
            return False
        r.status = "failed"
        r.state = {**r.state, "error": error}
        self.events.append(("fail_incomplete", run_id))
        return True

    async def get(self, run_id: str) -> GraphRun | None:
        r = self.docs.get(run_id)
        return r.model_copy(deep=True) if r else None

    async def update(self, run: GraphRun) -> None:
        self.docs[run.id] = run.model_copy(deep=True)


def _container(repo: _FakeRunRepository, *, max_attempts: int = 3, workflows: dict | None = None) -> ApplicationContainer:
    registry = MagicMock()
    registry.get = MagicMock(return_value=None)
    backend = None
    if workflows is not None:
        backend = AsyncMock()
        backend.get = AsyncMock(side_effect=lambda wid: workflows.get(wid))

        async def _update(wid, defn):
            workflows[wid] = defn
            return defn

        backend.update = AsyncMock(side_effect=_update)
    container = ApplicationContainer(
        settings=Settings(mcp_datasources_enabled=False, run_recovery_max_attempts=max_attempts),
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=MagicMock(spec=McpToolsProvider),
        yaml_graph_registry=registry,
        mongo_provider=MagicMock(),
        run_repository=repo,
        openhands=MagicMock(spec=OpenHandsAdapter),
        workflow_backend=backend,
        data_source_backend=None,
    )
    container.refresh_runner = AsyncMock()
    return container


def _run(run_id: str = "r1", status: str = "running", attempts: int = 0, graph_id: str = "wf") -> GraphRun:
    return GraphRun(id=run_id, graph_id=graph_id, status=status, recovery_attempts=attempts)


@pytest.mark.asyncio
async def test_counter_is_persisted_before_the_run_is_resumed():
    repo = _FakeRunRepository([_run()])
    container = _container(repo)
    seen: list[int] = []

    async def _recover(run: GraphRun) -> None:
        # The "crash" would happen in here: the increment must already be in the DB.
        seen.append(repo.docs[run.id].recovery_attempts)
        repo.events.append(("recover", run.id))

    container._recover_run = AsyncMock(side_effect=_recover)
    await container._recover_incomplete_runs()

    assert seen == [1]
    assert repo.events == [("begin_recovery", "r1"), ("recover", "r1")]
    assert repo.docs["r1"].last_recovery_at is not None


@pytest.mark.asyncio
async def test_three_crashing_recoveries_then_fourth_startup_fails_and_disables():
    workflows = {"wf": WorkflowDefinition(id="wf", name="wf", steps=[], enabled=True)}
    repo = _FakeRunRepository([_run()])
    container = _container(repo, workflows=workflows)
    # Simulated crash: the resume never gets to complete a step, so nothing
    # resets the counter before the process dies.
    container._recover_run = AsyncMock()

    for _ in range(3):
        await container._recover_incomplete_runs()
    assert container._recover_run.await_count == 3
    assert repo.docs["r1"].recovery_attempts == 3
    assert repo.docs["r1"].status == "running"

    # 4th startup: not resumed.
    await container._recover_incomplete_runs()
    assert container._recover_run.await_count == 3
    doc = repo.docs["r1"]
    assert doc.status == "failed"
    assert "crashed the server on 3 consecutive recoveries" in doc.state["error"]
    assert workflows["wf"].enabled is False
    container.refresh_runner.assert_awaited_with("wf")

    # And it is no longer picked up at all.
    await container._recover_incomplete_runs()
    assert container._recover_run.await_count == 3


@pytest.mark.asyncio
async def test_quarantine_disables_in_memory_runner_too():
    repo = _FakeRunRepository([_run(attempts=3)])
    container = _container(repo)  # no workflow backend
    runner = MagicMock()
    runner.enabled = True
    container.yaml_graph_registry.get = MagicMock(return_value=runner)
    container._recover_run = AsyncMock()

    await container._recover_incomplete_runs()

    container._recover_run.assert_not_awaited()
    assert runner.enabled is False
    assert repo.docs["r1"].status == "failed"


@pytest.mark.asyncio
async def test_threshold_is_configurable_and_zero_disables_the_guard():
    repo = _FakeRunRepository([_run(attempts=1)])
    container = _container(repo, max_attempts=1)
    container._recover_run = AsyncMock()
    await container._recover_incomplete_runs()
    container._recover_run.assert_not_awaited()
    assert repo.docs["r1"].status == "failed"

    repo = _FakeRunRepository([_run(attempts=50)])
    container = _container(repo, max_attempts=0)
    container._recover_run = AsyncMock()
    await container._recover_incomplete_runs()
    container._recover_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_reparked_waiting_run_resets_counter():
    """running -> waiting_approval -> running across restarts is not a crash loop."""
    repo = _FakeRunRepository([_run(status="waiting_approval", attempts=2)])
    container = _container(repo)
    container._recover_run = AsyncMock()  # re-armed from checkpoint, status unchanged

    for _ in range(5):
        await container._recover_incomplete_runs()

    assert container._recover_run.await_count == 5
    assert repo.docs["r1"].recovery_attempts == 0
    assert repo.docs["r1"].status == "waiting_approval"


@pytest.mark.asyncio
async def test_one_failing_run_does_not_block_the_others():
    repo = _FakeRunRepository([_run("bad"), _run("doomed", attempts=3, graph_id="other"), _run("good")])
    container = _container(repo)
    recovered: list[str] = []

    async def _recover(run: GraphRun) -> None:
        if run.id == "bad":
            raise RuntimeError("boom")
        recovered.append(run.id)

    container._recover_run = AsyncMock(side_effect=_recover)
    container.run_repository.fail_incomplete = AsyncMock(side_effect=RuntimeError("mongo down"))

    await container._recover_incomplete_runs()  # must not raise

    assert recovered == ["good"]


@pytest.mark.asyncio
async def test_candidate_query_failure_does_not_raise():
    repo = _FakeRunRepository([])
    repo.list_recovery_candidates = AsyncMock(side_effect=RuntimeError("mongo down"))
    container = _container(repo)
    await container._recover_incomplete_runs()


@pytest.mark.asyncio
async def test_jobs_are_closed_without_counting():
    job = GraphRun(id="j1", graph_id="wf", status="running", kind="job", recovery_attempts=99)
    repo = _FakeRunRepository([job])
    container = _container(repo)
    container._recover_run = AsyncMock()
    await container._recover_incomplete_runs()
    container._recover_run.assert_awaited_once()
    assert repo.docs["j1"].recovery_attempts == 99
    assert ("fail_incomplete", "j1") not in repo.events


@pytest.mark.asyncio
async def test_completing_a_step_resets_the_counter():
    graph = {
        "id": "wf",
        "steps": [{
            "id": "step1", "type": "llm", "output_key": "answer",
            "system_prompt": "Answer.", "user_template": "{request}",
        }],
    }
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])
    runner = YamlGraphRunner(
        graph, llm=FakeMessagesListChatModel(responses=[AIMessage(content="42")]), mcp_tools_provider=mcp,
    )
    run = GraphRun(id="t1", graph_id="wf", status="running", recovery_attempts=2)
    run.step_statuses = {"step1": "pending"}
    repo = AsyncMock()
    persisted: list[tuple[str | None, int]] = []
    repo.update = AsyncMock(
        side_effect=lambda r: persisted.append((r.step_statuses.get("step1"), r.recovery_attempts)),
    )

    await stream_graph_to_pause(runner, run, repo, {"request": "q"})

    assert run.recovery_attempts == 0
    # Before the step finishes the counter is untouched ("running" marker);
    # the write that records the finished step carries the reset.
    assert persisted[0] == ("running", 2)
    assert ("finished", 0) in persisted
