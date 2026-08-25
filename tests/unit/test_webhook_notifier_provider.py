"""The notification helpers now go through the messaging provider.

There used to be two Slack clients in the backend: four hand-rolled
``httpx.post`` calls in the webhook notifier, and nothing else.  These tests pin
that the *wire* behaviour did not change when they moved onto
``SlackProvider`` — same endpoint, same bearer token, same JSON body — because
the approval and ask_context paths are load-bearing and are what the run's
``_slack_thread_ts`` threading is built on.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.infrastructure.notifications.webhook_notifier import (
    post_slack_addon_notification,
    post_slack_ask_context,
    post_slack_thread_questions,
    post_slack_thread_reply,
    send_approval_notification,
)

TOKEN = "xoxb-notifier-token-0001"  # noqa: S105


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.reply)


@pytest.fixture
def slack(monkeypatch):
    def _install(reply: dict | None = None) -> _Transport:
        transport = _Transport(reply or {"ok": True, "ts": "1.1", "channel": "C9"})
        real = httpx.AsyncClient
        monkeypatch.setattr(
            "app.infrastructure.messaging.slack.httpx.AsyncClient",
            lambda *a, **kw: real(*a, transport=transport, **kw),
        )
        return transport

    return _install


def _body(transport: _Transport) -> dict:
    return json.loads(transport.requests[0].content)


@pytest.mark.asyncio
async def test_thread_reply_hits_chat_postmessage_with_the_thread_ts(slack):
    t = slack()
    await post_slack_thread_reply(TOKEN, "C9", "170.1", "done")

    request = t.requests[0]
    assert str(request.url) == "https://slack.com/api/chat.postMessage"
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert _body(t) == {"channel": "C9", "text": "done", "thread_ts": "170.1"}


@pytest.mark.asyncio
async def test_thread_questions_keep_their_numbering(slack):
    t = slack()
    await post_slack_thread_questions(TOKEN, "C9", "170.1", ["Which repo?", "2) Branch?"])

    text = _body(t)["text"]
    assert text.startswith("I need a bit more information to proceed:")
    assert "1. Which repo?" in text
    assert "2) Branch?" in text  # already numbered, not renumbered


@pytest.mark.asyncio
async def test_thread_questions_with_none_posts_nothing(slack):
    t = slack()
    await post_slack_thread_questions(TOKEN, "C9", "170.1", [])
    assert t.requests == []


@pytest.mark.asyncio
async def test_ask_context_returns_the_slack_response_the_runner_threads_on(slack):
    """stream_graph_to_pause reads ok/ts/channel off this to set
    ``_slack_ask_context_ts`` — the return shape is the contract."""
    t = slack({"ok": True, "ts": "555.1", "channel": "C9"})

    result = await post_slack_ask_context(
        TOKEN, "C9", ["Which repo?"], "run-1", {"ticket_id": "ABC-1"}
    )

    assert result == {"ok": True, "ts": "555.1", "channel": "C9"}
    body = _body(t)
    assert "thread_ts" not in body  # a new root message, not a reply
    assert "Context needed for `ABC-1`" in body["text"]
    assert "Reply in this thread with your answer." in body["text"]


@pytest.mark.asyncio
async def test_ask_context_returns_none_when_slack_refuses(slack):
    slack({"ok": False, "error": "not_in_channel"})
    assert await post_slack_ask_context(TOKEN, "C9", ["q"], "run-1", {}) is None


@pytest.mark.asyncio
async def test_ask_context_multi_question_hint(slack):
    t = slack()
    await post_slack_ask_context(TOKEN, "C9", ["a", "b"], "run-1", {})
    assert "2 numbered answers" in _body(t)["text"]


@pytest.mark.asyncio
async def test_addon_notification_posts_the_rendered_template(slack):
    t = slack()
    settings = MagicMock()
    settings.slack_bot_token = TOKEN
    settings.slack_approvals_channel = "C0ATMLKBJ02"
    settings.meta_llm_provider = "openrouter"
    settings.meta_llm_model = "m"

    with patch("app.core.config.get_settings", return_value=settings):
        await post_slack_addon_notification(
            TOKEN,
            '{"channel": "{slack_channel}", "text": "run {run_id}: {request}"}',
            "run-7",
            {"slack_channel": "C9", "request": "ship it"},
        )

    assert _body(t) == {"channel": "C9", "text": "run run-7: ship it"}


@pytest.mark.asyncio
async def test_addon_notification_with_broken_json_posts_nothing(slack):
    t = slack()
    settings = MagicMock()
    settings.slack_bot_token = TOKEN
    settings.slack_approvals_channel = ""
    with patch("app.core.config.get_settings", return_value=settings):
        await post_slack_addon_notification(TOKEN, "{not json", "run-7", {})
    assert t.requests == []


# ---------------------------------------------------------------------------
# The approval notify path: still the generic webhook sender, Slack payload
# surgery delegated to the provider module.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approval_notification_threads_and_tags_on_a_second_approval():
    settings = MagicMock()
    settings.slack_bot_token = TOKEN
    settings.slack_approvals_channel = "C0ATMLKBJ02"
    settings.meta_llm_provider = "openrouter"
    settings.meta_llm_model = "m"

    captured: dict = {}

    class _Client:
        def __init__(self, *a, **kw) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, method, url, json=None, headers=None, auth=None):
            captured.update(method=method, url=url, json=json, headers=headers)
            return httpx.Response(
                200,
                json={"ok": True, "ts": "9.9", "channel": "C1"},
                request=httpx.Request(method, url),
            )

    notify = {
        "url": "https://slack.com/api/chat.postMessage",
        "headers": {"Authorization": "Bearer {slack_bot_token}"},
        "payload": {
            "channel": "{slack_approvals_channel}",
            "text": "approve {run_id}?",
            "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "*plan*"}}],
        },
    }
    state = {"_slack_thread_ts": "170.1", "_slack_approver_id": "U7"}

    with patch("app.core.config.get_settings", return_value=settings), \
            patch("app.infrastructure.notifications.webhook_notifier.httpx.AsyncClient",
                  _Client):
        result = await send_approval_notification(notify, "run-1", state, "https://api")

    assert result == {"ok": True, "ts": "9.9", "channel": "C1"}
    assert captured["headers"]["Authorization"] == f"Bearer {TOKEN}"
    payload = captured["json"]
    assert payload["channel"] == "C0ATMLKBJ02"
    assert payload["thread_ts"] == "170.1"
    assert payload["text"] == "<@U7> approve run-1?"
    assert payload["blocks"][0]["text"]["text"] == "<@U7> *plan*"


@pytest.mark.asyncio
async def test_a_non_slack_notify_url_is_untouched():
    settings = MagicMock()
    settings.slack_bot_token = TOKEN
    settings.slack_approvals_channel = "C0ATMLKBJ02"
    settings.meta_llm_provider = "openrouter"
    settings.meta_llm_model = "m"
    captured: dict = {}

    class _Client:
        def __init__(self, *a, **kw) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, method, url, json=None, headers=None, auth=None):
            captured.update(json=json)
            return httpx.Response(204, request=httpx.Request(method, url))

    with patch("app.core.config.get_settings", return_value=settings), \
            patch("app.infrastructure.notifications.webhook_notifier.httpx.AsyncClient",
                  _Client):
        await send_approval_notification(
            {"url": "https://example.test/hook", "payload": {"text": "hi"}},
            "run-1",
            {"_slack_thread_ts": "170.1", "_slack_approver_id": "U7"},
            "https://api",
        )

    assert captured["json"] == {"text": "hi"}
