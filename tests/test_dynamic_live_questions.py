"""A dynamic job's agent asking through /agent/question reaches its meta-agent, not Slack."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.dynamic import hub as hubs
from app.services import agent_inbox
from tests.test_agent_callbacks_tools import _build_container


def _job_run() -> GraphRun:
    return GraphRun(id="child-1", graph_id="g", user_request="[code #1] x", status="running",
                    kind="job", parent_run_id="parent-1", current_step="code", state={})


async def _ask(run: GraphRun):
    container = _build_container(run)
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post(f"/api/v1/runs/{run.id}/agent/question", json={"question": "tabs or spaces?"})
    return resp, container


@pytest.mark.asyncio
async def test_question_from_a_dynamic_job_goes_to_its_step_hub():
    hub = hubs.hub_for("parent-1", "orchestrator")
    hub._children["child-1"] = ("code", 1)
    try:
        with patch("app.infrastructure.notifications.webhook_notifier.post_slack_ask_context", new=AsyncMock()) as slack:
            resp, container = await _ask(_job_run())
        assert resp.status_code == 202
        events = await hub.next_events(timeout=1)
        assert events == [{"type": "live_question", "job_id": "code", "attempt": 1, "child_run_id": "child-1", "question": "tabs or spaces?"}]
        slack.assert_not_called()
    finally:
        hubs.drop_hub("parent-1", "orchestrator")


@pytest.mark.asyncio
async def test_question_from_a_job_without_a_live_hub_is_handled_as_before():
    resp, container = await _ask(_job_run())
    assert resp.status_code == 202
    saved = container.run_repository.update.await_args.args[0]
    assert saved.state["_pending_question"]["question"] == "tabs or spaces?"


@pytest.mark.asyncio
async def test_deliver_answer_wakes_the_waiting_agent_and_persists_the_answer():
    run = _job_run()
    run.state = {"_pending_question": {"question": "q"}}
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=run)
    await agent_inbox.deliver_answer(repo, "child-1", "spaces")
    assert agent_inbox.event_for("child-1").is_set()
    assert agent_inbox.answers.pop("child-1") == "spaces"
    saved = repo.update.await_args.args[0]
    assert saved.state == {"_pending_answer": "spaces"}
    agent_inbox.answer_events.pop("child-1", None)
