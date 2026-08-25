"""The ``slack`` workflow step.

The step is provider-agnostic on purpose: it resolves a provider by name and
then only speaks the abstraction's vocabulary.  These tests therefore drive it
with a fake provider registered under its own name — if the step ever reached
for Slack directly, every one of them would fail.
"""
from __future__ import annotations

import pytest

from app.infrastructure.messaging import (
    Message,
    MessagingError,
    MessagingProvider,
    PostedMessage,
    register_provider,
    reset_providers,
)
from app.infrastructure.messaging import registry as provider_registry
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner

FAKE = "fake-chat"


class FakeProvider(MessagingProvider):
    """Records calls; scripted history/thread contents."""

    name = FAKE

    def __init__(self) -> None:
        self.posted: list[dict] = []
        self.opened: list[str] = []
        self.deleted: list[tuple[str, str]] = []
        self.history: list[Message] = []
        self.threads: dict[str, list[Message]] = {}
        self.thread_reads: list[str] = []
        self.fail_with: MessagingError | None = None
        self._seq = 0

    def _next_ts(self) -> str:
        self._seq += 1
        return f"99.{self._seq}"

    async def post_message(self, channel, text="", *, thread_id=None, blocks=None,
                           extra=None):
        if self.fail_with:
            raise self.fail_with
        self.posted.append({"channel": channel, "text": text, "thread_id": thread_id})
        return PostedMessage(id=self._next_ts(), channel=channel)

    async def reply_in_thread(self, channel, thread_id, text):
        return await self.post_message(channel, text, thread_id=thread_id)

    async def read_history(self, channel, oldest=None, limit=None):
        if self.fail_with:
            raise self.fail_with
        self.last_history = {"channel": channel, "oldest": oldest, "limit": limit}
        return list(self.history)

    async def read_thread(self, channel, thread_id):
        if self.fail_with:
            raise self.fail_with
        self.thread_reads.append(thread_id)
        return list(self.threads.get(thread_id, []))

    async def open_dm(self, user_id):
        if self.fail_with:
            raise self.fail_with
        self.opened.append(user_id)
        return f"D-{user_id}"

    async def delete_message(self, channel, message_id):
        if self.fail_with:
            raise self.fail_with
        self.deleted.append((channel, message_id))


@pytest.fixture
def provider():
    register_provider(FakeProvider)
    reset_providers()
    instance = provider_registry.get_provider(FAKE)
    yield instance
    provider_registry._PROVIDERS.pop(FAKE, None)
    reset_providers()


class _FakeLLM:
    """No messaging action ever reaches the chat model."""


def _node(step: dict):
    step = {"type": "slack", "provider": FAKE, **step}
    runner = YamlGraphRunner(
        {"id": "wf", "steps": [step]}, _FakeLLM(), lambda *a, **k: []
    )
    return runner._messaging_node(step)


# ---------------------------------------------------------------------------
# post / reply / dm / delete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_templates_channel_and_text(provider):
    node = _node({"id": "digest", "action": "post",
                  "channel": "{target}", "text": "Board:\n{digest}"})

    out = await node({"target": "C0BLDDSEB1D", "digest": "3 alerts"})

    assert provider.posted == [
        {"channel": "C0BLDDSEB1D", "text": "Board:\n3 alerts", "thread_id": None}
    ]
    assert out["digest"] == {"message_id": "99.1", "channel": "C0BLDDSEB1D"}


@pytest.mark.asyncio
async def test_post_writes_to_output_key_not_the_step_id(provider):
    node = _node({"id": "s", "action": "post", "channel": "C1", "text": "x",
                  "output_key": "digest_post"})
    out = await node({})
    assert set(out) == {"digest_post"}


@pytest.mark.asyncio
async def test_reply_needs_a_thread_id(provider):
    node = _node({"id": "s", "action": "reply", "channel": "C1", "text": "x"})
    out = await node({})
    assert "needs a non-empty 'thread_id'" in out["s"]["error"]
    assert out["__failed_step__"] == "s"
    assert provider.posted == []


@pytest.mark.asyncio
async def test_reply_posts_in_the_named_thread(provider):
    node = _node({"id": "s", "action": "reply", "channel": "C1",
                  "thread_id": "{ts}", "text": "ack"})
    out = await node({"ts": "170.1"})
    assert provider.posted == [{"channel": "C1", "text": "ack", "thread_id": "170.1"}]
    assert out["s"]["thread_id"] == "170.1"


@pytest.mark.asyncio
async def test_dm_opens_a_channel_then_posts_to_it(provider):
    """The failure path: DM instead of the channel."""
    node = _node({"id": "warn", "action": "dm", "user_id": "{owner}",
                  "text": "no usable export"})

    out = await node({"owner": "U0ATR9G06MA"})

    assert provider.opened == ["U0ATR9G06MA"]
    assert provider.posted == [
        {"channel": "D-U0ATR9G06MA", "text": "no usable export", "thread_id": None}
    ]
    assert out["warn"] == {"message_id": "99.1", "channel": "D-U0ATR9G06MA",
                           "user_id": "U0ATR9G06MA"}


@pytest.mark.asyncio
async def test_dm_needs_a_user_id(provider):
    node = _node({"id": "s", "action": "dm", "text": "x"})
    out = await node({})
    assert "needs a non-empty 'user_id'" in out["s"]["error"]
    assert provider.opened == []


@pytest.mark.asyncio
async def test_dm_failure_does_not_fall_back_to_the_channel(provider):
    """A DM that cannot be delivered must not become a channel post — that is
    precisely the false all-clear the failure path exists to prevent."""
    provider.fail_with = MessagingError("no im:write", code="missing_scope")
    node = _node({"id": "s", "action": "dm", "user_id": "U1", "text": "x"})

    out = await node({})

    assert provider.posted == []
    assert out["__failed_step__"] == "s"
    assert "missing_scope" not in out["s"]["error"]  # message, not code
    assert "no im:write" in out["s"]["error"]


@pytest.mark.asyncio
async def test_delete(provider):
    node = _node({"id": "s", "action": "delete", "channel": "C1",
                  "message_id": "{ts}"})
    out = await node({"ts": "1.1"})
    assert provider.deleted == [("C1", "1.1")]
    assert out["s"]["deleted"] is True


@pytest.mark.asyncio
async def test_unknown_action_is_refused(provider):
    node = _node({"id": "s", "action": "react", "channel": "C1"})
    out = await node({})
    assert "unknown messaging action 'react'" in out["s"]["error"]


@pytest.mark.asyncio
async def test_unknown_provider_is_refused():
    runner = YamlGraphRunner(
        {"id": "wf", "steps": []}, _FakeLLM(), lambda *a, **k: []
    )
    step = {"id": "s", "type": "slack", "action": "post", "provider": "telepathy",
            "channel": "C1", "text": "x"}
    out = await runner._messaging_node(step)({})
    assert "Unknown messaging provider" in out["s"]["error"]


# ---------------------------------------------------------------------------
# history / thread
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_history_output_feeds_the_override_parser_verbatim(provider):
    """csm-deadline-parse-overrides reads state["slack_messages"] and pulls
    ``text``/``ts``/``user`` off each item, so the step's list shape has to carry
    them — that is the whole wiring between the step and the library script."""
    provider.history = [
        Message(id="9.1", channel="C1", text="FRIST 128498: 3", author="U9"),
    ]
    node = _node({"id": "read", "action": "history", "channel": "C1",
                  "oldest": "{window}", "limit": 200,
                  "output_key": "slack_messages"})

    out = await node({"window": "1787600000"})

    assert provider.last_history == {"channel": "C1", "oldest": "1787600000",
                                     "limit": 200}
    assert out["slack_messages"] == [
        {"id": "9.1", "channel": "C1", "text": "FRIST 128498: 3", "author": "U9",
         "thread_id": None}
    ]


@pytest.mark.asyncio
async def test_history_ignores_a_non_numeric_limit(provider):
    node = _node({"id": "s", "action": "history", "channel": "C1", "limit": "{nope}"})
    await node({})
    assert provider.last_history["limit"] is None


@pytest.mark.asyncio
async def test_thread_reads_the_whole_thread(provider):
    provider.threads["1.1"] = [
        Message(id="1.1", channel="C1", text="root", author="U1"),
        Message(id="1.2", channel="C1", text="reply", author="U2", thread_id="1.1"),
    ]
    node = _node({"id": "s", "action": "thread", "channel": "C1",
                  "thread_id": "1.1"})
    out = await node({})
    assert [m["text"] for m in out["s"]] == ["root", "reply"]


# ---------------------------------------------------------------------------
# Batch replies and the idempotency guard
# ---------------------------------------------------------------------------

def _confirmations_state() -> dict:
    return {
        "parsed": {
            "confirmations": [
                {"thread_ts": "1.1", "text": "Frist fuer Projekt 128498 uebernommen: 3 Arbeitstage"},
                {"thread_ts": "2.1", "text": "Frist fuer Projekt 133170 uebernommen: fix 2026-08-15"},
            ]
        }
    }


@pytest.mark.asyncio
async def test_items_posts_one_reply_per_entry(provider):
    node = _node({"id": "confirm", "action": "reply", "channel": "C1",
                  "items": "parsed.confirmations"})

    out = await node(_confirmations_state())

    assert [p["thread_id"] for p in provider.posted] == ["1.1", "2.1"]
    assert out["confirm"]["posted_count"] == 2
    assert out["confirm"]["skipped_count"] == 0


@pytest.mark.asyncio
async def test_a_thread_that_already_has_the_confirmation_is_not_confirmed_twice(provider):
    """The idempotency guard.  The watcher re-reads an overlapping 26-hour
    window every morning, so the same accepted override comes round again."""
    provider.threads["1.1"] = [
        Message(id="1.1", channel="C1", text="FRIST 128498: 3", author="U9"),
        Message(id="1.5", channel="C1", author="Ubot", thread_id="1.1",
                text="Frist fuer Projekt 128498 uebernommen: 3 Arbeitstage"),
    ]
    node = _node({"id": "confirm", "action": "reply", "channel": "C1",
                  "items": "parsed.confirmations", "skip_if_replied": True})

    out = await node(_confirmations_state())

    assert [p["thread_id"] for p in provider.posted] == ["2.1"]
    assert out["confirm"]["posted_count"] == 1
    assert out["confirm"]["skipped"] == [
        {"thread_id": "1.1", "reason": "already_replied"}
    ]


@pytest.mark.asyncio
async def test_the_guard_only_looks_at_replies_not_the_root_message(provider):
    """A root message whose text happens to equal the confirmation must not
    suppress the confirmation — otherwise a CSM quoting the bot silences it."""
    text = "Frist fuer Projekt 128498 uebernommen: 3 Arbeitstage"
    provider.threads["1.1"] = [Message(id="1.1", channel="C1", text=text, author="U9")]
    node = _node({"id": "confirm", "action": "reply", "channel": "C1",
                  "items": [{"thread_id": "1.1", "text": text}],
                  "skip_if_replied": True})

    out = await node({})

    assert out["confirm"]["posted_count"] == 1


@pytest.mark.asyncio
async def test_without_the_flag_no_thread_is_read_at_all(provider):
    node = _node({"id": "confirm", "action": "reply", "channel": "C1",
                  "items": "parsed.confirmations"})
    await node(_confirmations_state())
    assert provider.thread_reads == []


@pytest.mark.asyncio
async def test_a_single_reply_is_skipped_when_it_is_already_there(provider):
    provider.threads["1.1"] = [
        Message(id="1.1", channel="C1", text="root", author="U9"),
        Message(id="1.6", channel="C1", text="ack", author="Ubot", thread_id="1.1"),
    ]
    node = _node({"id": "s", "action": "reply", "channel": "C1",
                  "thread_id": "1.1", "text": "ack", "skip_if_replied": True})

    out = await node({})

    assert provider.posted == []
    assert out["s"] == {"skipped": True, "reason": "already_replied",
                        "thread_id": "1.1"}


@pytest.mark.asyncio
async def test_an_empty_items_list_posts_nothing_and_does_not_fail(provider):
    node = _node({"id": "confirm", "action": "reply", "channel": "C1",
                  "items": "parsed.confirmations"})
    out = await node({"parsed": {"confirmations": []}})
    assert provider.posted == []
    assert out["confirm"] == {"posted": [], "skipped": [], "posted_count": 0,
                              "skipped_count": 0}


@pytest.mark.asyncio
async def test_a_missing_items_path_falls_back_to_the_single_reply_fields(provider):
    node = _node({"id": "s", "action": "reply", "channel": "C1",
                  "items": "parsed.confirmations", "thread_id": "1.1",
                  "text": "ack"})
    out = await node({})
    assert provider.posted == [{"channel": "C1", "text": "ack", "thread_id": "1.1"}]
    assert out["s"]["thread_id"] == "1.1"


@pytest.mark.asyncio
async def test_an_items_path_that_is_not_a_list_is_refused(provider):
    node = _node({"id": "s", "action": "reply", "channel": "C1",
                  "items": "parsed.confirmations"})
    out = await node({"parsed": {"confirmations": {"thread_id": "1.1"}}})
    assert "is not a list" in out["s"]["error"]


@pytest.mark.asyncio
async def test_incomplete_items_are_skipped_not_posted(provider):
    node = _node({"id": "s", "action": "reply", "channel": "C1",
                  "items": [{"thread_id": "1.1"}, {"text": "orphan"}]})
    out = await node({})
    assert provider.posted == []
    assert out["s"]["skipped_count"] == 2


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_provider_error_fails_the_run_by_default(provider):
    provider.fail_with = MessagingError("not_in_channel", code="not_in_channel")
    node = _node({"id": "s", "action": "post", "channel": "C1", "text": "x"})
    out = await node({})
    assert out["__failed_step__"] == "s"


@pytest.mark.asyncio
async def test_ignore_errors_keeps_a_read_leg_from_taking_the_run_down(provider):
    """The CSM watcher must still compute deadlines when Slack is unreachable."""
    provider.fail_with = MessagingError("not_in_channel", code="not_in_channel")
    node = _node({"id": "read", "action": "history", "channel": "C1",
                  "ignore_errors": True, "output_key": "slack_messages"})

    out = await node({})

    assert out == {"slack_messages": []}  # a list, so the parser still iterates
    assert "__failed_step__" not in out


@pytest.mark.asyncio
async def test_ignore_errors_on_a_write_leg_reports_the_error_in_place(provider):
    provider.fail_with = MessagingError("channel_not_found", code="channel_not_found")
    node = _node({"id": "s", "action": "post", "channel": "C1", "text": "x",
                  "ignore_errors": True})
    out = await node({})
    assert "channel_not_found" in out["s"]["error"]
    assert "__failed_step__" not in out


# ---------------------------------------------------------------------------
# The credential never reaches run state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_token_reaches_step_output_or_state(monkeypatch):
    """Not "the step does not print the token" but "there is nowhere for it to
    go": the step never sees it, and a provider error is scrubbed at the source.
    """
    token = "xoxb-secret-999888777"  # noqa: S105
    monkeypatch.setattr(
        "app.infrastructure.messaging.slack.bot_token", lambda: token
    )

    def _explode(*args, **kwargs):
        raise RuntimeError(f"boom Authorization: Bearer {token}")

    monkeypatch.setattr(
        "app.infrastructure.messaging.slack.httpx.AsyncClient", _explode
    )
    reset_providers()
    runner = YamlGraphRunner(
        {"id": "wf", "steps": []}, _FakeLLM(), lambda *a, **k: []
    )
    step = {"id": "s", "type": "slack", "action": "post", "channel": "C1",
            "text": "x", "ignore_errors": True}

    out = await runner._messaging_node(step)({"secret_hint": "none"})

    rendered = repr(out)
    assert token not in rendered
    assert "***" in out["s"]["error"]
    assert token not in repr(step)  # the config never carried it either
    reset_providers()
