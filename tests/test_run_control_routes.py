"""Regression tests for the REST run-control routes after their logic moved
into ``app.application.run_control``.

Covers one happy path (terminate) plus the refusal branches of every route, so
the delegation keeps the previous status codes and side effects.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.domain.models.graph_run import GraphRun
from tests.test_graphs_api import _build_container, _build_registry

_RUNS = "/api/v1/workflows/runs"


@pytest.fixture
async def client():
    container = _build_container(_build_registry())
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, container


def _run(status: str, run_id: str = "tid1") -> GraphRun:
    return GraphRun(id=run_id, graph_id="simple", user_request="hello", status=status)


async def test_terminate_marks_the_run_failed(client, monkeypatch):
    c, container = client
    monkeypatch.setattr("app.services.agent_cleanup.cleanup_run_agents", AsyncMock())
    container.run_repository.get = AsyncMock(return_value=_run("running"))

    resp = await c.post(f"{_RUNS}/tid1/terminate")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "tid1"
    assert body["status"] == "failed"
    assert body["error"] == "Terminated by user"


async def test_terminate_404_when_run_missing(client):
    c, container = client
    container.run_repository.get = AsyncMock(return_value=None)
    resp = await c.post(f"{_RUNS}/nope/terminate")
    assert resp.status_code == 404


async def test_terminate_409_when_run_not_active(client):
    c, container = client
    container.run_repository.get = AsyncMock(return_value=_run("completed"))
    resp = await c.post(f"{_RUNS}/tid1/terminate")
    assert resp.status_code == 409


async def test_retry_409_when_run_not_failed(client):
    c, container = client
    container.run_repository.get = AsyncMock(return_value=_run("running"))
    resp = await c.post(f"{_RUNS}/tid1/retry")
    assert resp.status_code == 409


async def test_restart_from_step_409_while_running(client):
    c, container = client
    container.run_repository.get = AsyncMock(return_value=_run("running"))
    resp = await c.post(f"{_RUNS}/tid1/restart-from-step", json={"step_id": "step1"})
    assert resp.status_code == 409


async def test_approve_404_when_run_missing(client):
    c, container = client
    container.run_repository.claim_for_resume = AsyncMock(return_value=None)
    container.run_repository.get = AsyncMock(return_value=None)
    resp = await c.post(f"{_RUNS}/nope/approve")
    assert resp.status_code == 404


async def test_approve_409_when_not_awaiting_approval(client):
    c, container = client
    container.run_repository.claim_for_resume = AsyncMock(return_value=None)
    container.run_repository.get = AsyncMock(return_value=_run("running"))
    resp = await c.post(f"{_RUNS}/tid1/approve")
    assert resp.status_code == 409
    assert "not awaiting approval" in resp.json()["detail"]


async def test_reject_409_when_not_awaiting_approval(client):
    c, container = client
    container.run_repository.claim_for_resume = AsyncMock(return_value=None)
    container.run_repository.get = AsyncMock(return_value=_run("completed"))
    resp = await c.post(f"{_RUNS}/tid1/reject")
    assert resp.status_code == 409


async def test_approve_schedules_resume_and_flips_the_step(client, monkeypatch):
    c, container = client
    run = _run("waiting_approval")
    run.current_step = "step1"
    run.step_statuses = {"step1": "waiting_approval"}
    container.run_repository.claim_for_resume = AsyncMock(return_value=run)
    container.live_runners[run.id] = MagicMock()

    seen: dict = {}

    async def fake_resume(runner, r, cont, corrections, approver_name=None,
                          approver_id=None, approver_source="ui"):
        seen.update({"approver_source": approver_source, "corrections": corrections})

    monkeypatch.setattr("app.application.run_control._resume_approved", fake_resume)

    resp = await c.post(f"{_RUNS}/tid1/approve", json={"corrections": {"a": 1}})

    assert resp.status_code == 200
    assert run.step_statuses["step1"] == "finished"
    # BackgroundTask has run by the time the response is delivered.
    assert seen == {"approver_source": "ui", "corrections": {"a": 1}}


# ---------------------------------------------------------------------------
# retry / restart-from-step happy paths
#
# These are the riskiest transitions in the run-control layer: they seed the
# LangGraph checkpoint (aupdate_state as_node=<last completed step>), rebuild
# accumulated state from persisted step outputs, and — for restart — clean up
# the Docker runtime.  The runner/checkpointer is faked the same way the other
# tests here fake their collaborators.
# ---------------------------------------------------------------------------

_STEPS = [{"id": "step1", "type": "llm"}, {"id": "step2", "type": "llm"},
          {"id": "step3", "type": "llm"}]


class _FakeRunner:
    """Only what run_control and the run response actually touch."""

    def __init__(self, steps=None) -> None:
        self.name = "Simple"
        self.steps = steps or _STEPS
        self.graph = MagicMock()
        self.graph.aupdate_state = AsyncMock()


@pytest.fixture
def scheduled(monkeypatch):
    """Capture the continuation the route schedules instead of streaming."""
    calls: list = []

    async def fake_retry_graph(runner, run, cont, resume_input):
        calls.append({"runner": runner, "run": run, "resume_input": resume_input})

    monkeypatch.setattr("app.application.run_control._retry_graph", fake_retry_graph)
    return calls


async def test_retry_seeds_the_checkpoint_and_schedules_the_continuation(
    client, monkeypatch, scheduled
):
    c, container = client
    run = _run("failed")
    run.step_statuses = {"step1": "finished", "step2": "failed", "step3": "pending"}
    run.step_outputs = {"step1": {"answer": "a1"}}
    run.state = {"_conv_id": "c1", "_visit_counts": {"step2": 3}}
    container.run_repository.get = AsyncMock(return_value=run)
    runner = _FakeRunner()
    monkeypatch.setattr(container, "_build_runner_for_recovery", lambda r: runner)

    resp = await c.post(f"{_RUNS}/tid1/retry")

    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    # Finished steps kept; the failed step and everything after it reset.
    assert run.step_statuses == {
        "step1": "finished", "step2": "pending", "step3": "pending"
    }
    # Accumulated state = request + outputs of finished steps + internal keys,
    # but never _visit_counts (a retry starts with a fresh loop counter).
    assert run.state == {"request": "hello", "answer": "a1", "_conv_id": "c1"}
    # Checkpoint seeded at the last completed step, so the graph resumes there.
    runner.graph.aupdate_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "tid1"}}, run.state, as_node="step1"
    )
    assert container.live_runners["tid1"] is runner
    # Continuation scheduled with resume_input=None → resume from the checkpoint.
    assert len(scheduled) == 1
    assert scheduled[0]["runner"] is runner
    assert scheduled[0]["resume_input"] is None


async def test_retry_without_any_completed_step_starts_fresh(
    client, monkeypatch, scheduled
):
    c, container = client
    run = _run("failed")
    run.step_statuses = {"step1": "failed", "step2": "pending", "step3": "pending"}
    container.run_repository.get = AsyncMock(return_value=run)
    runner = _FakeRunner()
    monkeypatch.setattr(container, "_build_runner_for_recovery", lambda r: runner)

    resp = await c.post(f"{_RUNS}/tid1/retry")

    assert resp.status_code == 200
    runner.graph.aupdate_state.assert_not_awaited()
    assert scheduled[0]["resume_input"] == {"request": "hello"}


async def test_restart_from_step_discards_that_step_onwards(
    client, monkeypatch, scheduled
):
    c, container = client
    run = _run("completed")
    run.step_statuses = {"step1": "finished", "step2": "finished", "step3": "finished"}
    run.step_inputs = {"step1": {"i": 1}, "step2": {"i": 2}, "step3": {"i": 3}}
    run.step_outputs = {"step1": {"answer": "a1"}, "step2": {"b": 2}, "step3": {"c": 3}}
    run.state = {"_conv_id": "c1", "_visit_counts": {"step2": 3},
                 "_pending_question": "q?"}
    container.run_repository.get = AsyncMock(return_value=run)
    runner = _FakeRunner()
    monkeypatch.setattr(container, "_build_runner_for_recovery", lambda r: runner)
    container.live_runners["tid1"] = MagicMock()

    docker = MagicMock()
    docker.return_value.terminate_by_run_id = AsyncMock()
    monkeypatch.setattr("app.runtime.docker.DockerRuntime", docker)

    resp = await c.post(f"{_RUNS}/tid1/restart-from-step", json={"step_id": "step2"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    # Any live agent container for the run is torn down first.
    docker.return_value.terminate_by_run_id.assert_awaited_once_with(None, "tid1")
    # step2 onwards reset and their recorded inputs/outputs discarded.
    assert run.step_statuses == {
        "step1": "finished", "step2": "pending", "step3": "pending"
    }
    assert set(run.step_inputs) == {"step1"}
    assert set(run.step_outputs) == {"step1"}
    # Only steps BEFORE step2 contribute state; _visit_counts and the transient
    # ask-context keys are dropped.
    assert run.state == {"request": "hello", "answer": "a1", "_conv_id": "c1"}
    assert run.current_step is None
    runner.graph.aupdate_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "tid1"}}, run.state, as_node="step1"
    )
    assert container.live_runners["tid1"] is runner
    assert len(scheduled) == 1
    assert scheduled[0]["resume_input"] is None


async def test_restart_from_the_first_step_starts_fresh(client, monkeypatch, scheduled):
    c, container = client
    run = _run("failed")
    run.step_statuses = {"step1": "finished", "step2": "failed", "step3": "pending"}
    run.step_outputs = {"step1": {"answer": "a1"}}
    container.run_repository.get = AsyncMock(return_value=run)
    runner = _FakeRunner()
    monkeypatch.setattr(container, "_build_runner_for_recovery", lambda r: runner)
    monkeypatch.setattr("app.runtime.docker.DockerRuntime", MagicMock())

    resp = await c.post(f"{_RUNS}/tid1/restart-from-step", json={"step_id": "step1"})

    assert resp.status_code == 200
    runner.graph.aupdate_state.assert_not_awaited()
    assert scheduled[0]["resume_input"] == {"request": "hello"}
    assert run.step_outputs == {}


async def test_restart_from_unknown_step_is_refused(client, monkeypatch):
    c, container = client
    run = _run("failed")
    container.run_repository.get = AsyncMock(return_value=run)
    monkeypatch.setattr(container, "_build_runner_for_recovery", lambda r: _FakeRunner())
    resp = await c.post(f"{_RUNS}/tid1/restart-from-step", json={"step_id": "nope"})
    assert resp.status_code == 409
    assert "Unknown step_id" in resp.json()["detail"]
