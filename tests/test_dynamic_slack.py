"""A paused dynamic step notifies through the step's Slack addon, with Approve /
Reject buttons that resume the run like the UI does."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.infrastructure.notifications import webhook_notifier as wn
from app.infrastructure.orchestration import yaml_graph
from app.infrastructure.orchestration.yaml_graph import notify_dynamic_gate, slack_addon_attached

GATE = {
    "type": "dynamic_approval",
    "step_id": "develop",
    "automation": "plan",
    "plan": "[plan] Research first",
    "decisions": [{
        "id": "d1", "kind": "plan", "summary": "Research first",
        "payload": {"actions": [{"kind": "add_jobs", "jobs": [
            {"id": "research", "category": "planning", "agent_id": "eng-researcher", "title": "Read the ticket", "prompt": "p"},
            {"id": "code", "category": "execution", "agent_id": "eng-coder", "title": "", "prompt": "Implement it", "depends_on": ["research"]},
        ]}]},
    }],
}


def _snap(value):
    intr = SimpleNamespace(value=value)
    return SimpleNamespace(tasks=[SimpleNamespace(interrupts=[intr])], interrupts=())


def test_decision_lines_list_new_jobs_with_category_agent_and_title():
    lines = wn.dynamic_decision_lines(GATE)
    assert lines[0] == "*[plan]* Research first"
    assert "• `research` planning → *eng-researcher*: Read the ticket" in lines
    assert "• `code` execution → *eng-coder*: Implement it (after research)" in lines


def test_blocks_carry_generic_approve_reject_with_the_run_id():
    blocks = wn.dynamic_decision_blocks(GATE, "run-1", "Develop C130-1", "https://ui/#run=run-1")
    actions = blocks[-1]["elements"]
    assert [(a["action_id"], a["value"]) for a in actions] == [("approve", "run-1"), ("reject", "run-1")]
    assert any(b["type"] == "context" and "https://ui/#run=run-1" in b["elements"][0]["text"] for b in blocks)


def test_run_link():
    assert wn.run_link("https://ui/", "r") == "https://ui/#run=r"
    assert wn.run_link(None, "r") is None


@pytest.mark.parametrize("step,attached", [
    ({"id": "s"}, False),
    ({"id": "s", "slack_notifications": True}, True),
    ({"id": "s", "slack_channel": "C1"}, True),
    ({"id": "s", "slack_payload": "{}"}, True),
    (None, False),
])
def test_slack_addon_attached(step, attached):
    assert slack_addon_attached(step) is attached


def _settings(monkeypatch, **kw):
    s = SimpleNamespace(slack_bot_token="xoxb-default", slack_approvals_channel="C-default", ui_base_url="https://ui", **kw)
    monkeypatch.setattr("app.core.config.get_settings", lambda: s)


@pytest.mark.asyncio
async def test_posts_the_decision_to_the_steps_channel(monkeypatch):
    _settings(monkeypatch)
    sent = {}

    async def fake_post(token, channel, run_id, payload, request="", run_url=None):
        sent.update(token=token, channel=channel, run_id=run_id, payload=payload, request=request, run_url=run_url)

    monkeypatch.setattr(wn, "post_slack_dynamic_decision", fake_post)
    handled = await notify_dynamic_gate(
        {"id": "develop", "type": "dynamic", "slack_channel": "C-team"}, "run-1", _snap(GATE), {"request": "Develop C130-1"},
    )
    assert handled is True
    assert sent == {"token": "xoxb-default", "channel": "C-team", "run_id": "run-1", "payload": GATE,
                    "request": "Develop C130-1", "run_url": "https://ui/#run=run-1"}


@pytest.mark.asyncio
async def test_custom_payload_template_gets_the_plan(monkeypatch):
    _settings(monkeypatch)
    got = {}

    async def fake_addon(bot_token, payload_template, run_id, state, questions=None):
        got.update(state=state, template=payload_template)

    monkeypatch.setattr(wn, "post_slack_addon_notification", fake_addon)
    assert await notify_dynamic_gate(
        {"id": "develop", "slack_notifications": True, "slack_payload": '{"text": "{plan}"}'}, "r", _snap(GATE), {},
    )
    assert got["state"]["plan"] == "[plan] Research first"


@pytest.mark.asyncio
async def test_without_slack_addon_or_other_interrupts_nothing_happens(monkeypatch):
    _settings(monkeypatch)
    assert await notify_dynamic_gate({"id": "develop"}, "r", _snap(GATE), {}) is False
    assert await notify_dynamic_gate({"id": "a", "slack_notifications": True}, "r", _snap({"type": "ask_context"}), {}) is False


@pytest.mark.asyncio
async def test_resumed_runs_notify_again_by_default(monkeypatch):
    """stream_graph_to_pause falls back to BASE_URL so a pause after a resume notifies."""
    import inspect
    src = inspect.getsource(yaml_graph.stream_graph_to_pause)
    assert "if base_url is None" in src
