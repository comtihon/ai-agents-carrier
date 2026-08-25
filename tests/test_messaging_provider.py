"""The messaging abstraction: one interface, one Slack client, no leaked token.

These tests drive the provider against a stubbed httpx transport rather than a
mock of our own wrapper, so the request that would actually go to Slack — url,
body encoding, bearer header — is what is asserted.  ``conversations.open`` is
covered here because it is the failure path (DM instead of channel) and the one
call the live bot may not be scoped for.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.infrastructure.messaging import (
    Message,
    MessagingError,
    MessagingProvider,
    available_providers,
    get_provider,
    register_provider,
    reset_providers,
)
from app.infrastructure.messaging.slack import (
    SlackMessage,
    SlackProvider,
    apply_thread_context,
    scrub,
)

TOKEN = "xoxb-test-token-000111222"  # noqa: S105 — fake, and the point of the leak test


class _Transport(httpx.AsyncBaseTransport):
    """Records every request and replies from a per-method script."""

    def __init__(self, replies: dict[str, object]) -> None:
        self.replies = replies
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        method = request.url.path.rsplit("/", 1)[-1]
        reply = self.replies[method]
        if isinstance(reply, list):
            reply = reply.pop(0)
        return httpx.Response(200, json=reply)


@pytest.fixture
def transport(monkeypatch):
    """Route every provider call through a recording transport."""
    holder: dict[str, _Transport] = {}

    real_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        return real_client(*args, transport=holder["t"], **kwargs)

    monkeypatch.setattr(
        "app.infrastructure.messaging.slack.httpx.AsyncClient", _factory
    )

    def _install(replies: dict[str, object]) -> _Transport:
        holder["t"] = _Transport(replies)
        return holder["t"]

    return _install


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    reset_providers()


def _form(request: httpx.Request) -> dict[str, str]:
    return dict(httpx.QueryParams(request.content.decode()))


# ---------------------------------------------------------------------------
# The abstraction itself
# ---------------------------------------------------------------------------

def test_slack_is_registered_under_its_name():
    assert "slack" in available_providers()
    assert get_provider("slack").name == "slack"
    assert get_provider() is get_provider("slack")  # default, and cached


def test_a_second_provider_is_a_class_plus_a_registry_entry():
    """What adding WhatsApp costs: one subclass, one decorator, no step change."""

    @register_provider
    class _Whatsapp(SlackProvider):  # reuses nothing but the shape
        name = "whatsapp-test"

    try:
        assert get_provider("whatsapp-test").name == "whatsapp-test"
        assert "whatsapp-test" in available_providers()
    finally:
        from app.infrastructure.messaging import registry

        registry._PROVIDERS.pop("whatsapp-test", None)


def test_unknown_provider_names_what_is_available():
    with pytest.raises(ValueError, match="Unknown messaging provider 'telepathy'"):
        get_provider("telepathy")


def test_every_interface_method_is_abstract():
    """A partial provider must fail at construction, not at 3am in a run."""

    class _Partial(MessagingProvider):
        name = "partial"

        async def post_message(self, channel, text="", **kw):  # type: ignore[override]
            return None  # pragma: no cover

    with pytest.raises(TypeError):
        _Partial()  # type: ignore[abstract]


def test_neutral_message_shape_has_no_provider_field_names():
    neutral = Message(id="1", channel="C1", text="hi", author="U1").as_dict()
    assert set(neutral) == {"id", "channel", "text", "author", "thread_id"}


def test_slack_message_adds_its_own_native_aliases():
    """The aliases live in the provider, so the neutral shape stays neutral."""
    d = SlackMessage(id="171.5", channel="C1", text="hi", author="U1",
                     thread_id="170.1").as_dict()
    assert d["ts"] == "171.5" and d["user"] == "U1" and d["thread_ts"] == "170.1"
    assert d["id"] == d["ts"] and d["author"] == d["user"]


# ---------------------------------------------------------------------------
# Slack operations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_message_sends_json_with_a_bearer_token(transport):
    t = transport({"chat.postMessage": {"ok": True, "ts": "1.1", "channel": "C1"}})
    posted = await SlackProvider(TOKEN).post_message("C1", "hello")

    assert posted.as_dict() == {"message_id": "1.1", "channel": "C1"}
    request = t.requests[0]
    assert str(request.url) == "https://slack.com/api/chat.postMessage"
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(request.content) == {"channel": "C1", "text": "hello"}


@pytest.mark.asyncio
async def test_reply_in_thread_carries_thread_ts(transport):
    t = transport({"chat.postMessage": {"ok": True, "ts": "2.2", "channel": "C1"}})
    await SlackProvider(TOKEN).reply_in_thread("C1", "1.1", "ack")

    assert json.loads(t.requests[0].content)["thread_ts"] == "1.1"


@pytest.mark.asyncio
async def test_read_history_passes_the_window_and_maps_messages(transport):
    t = transport({"conversations.history": {"ok": True, "messages": [
        {"ts": "9.1", "user": "U9", "text": "FRIST 128498: 3"},
        {"ts": "9.0", "user": "U8", "text": "noise", "thread_ts": "8.0"},
    ]}})
    messages = await SlackProvider(TOKEN).read_history("C1", oldest="100", limit=200)

    assert _form(t.requests[0]) == {"channel": "C1", "oldest": "100", "limit": "200"}
    assert [m.text for m in messages] == ["FRIST 128498: 3", "noise"]
    assert messages[0].as_dict()["ts"] == "9.1"
    assert messages[1].thread_id == "8.0"


@pytest.mark.asyncio
async def test_read_history_omits_an_unset_window(transport):
    t = transport({"conversations.history": {"ok": True, "messages": []}})
    await SlackProvider(TOKEN).read_history("C1")
    assert _form(t.requests[0]) == {"channel": "C1"}


@pytest.mark.asyncio
async def test_read_thread_returns_root_first(transport):
    t = transport({"conversations.replies": {"ok": True, "messages": [
        {"ts": "1.1", "user": "U1", "text": "root"},
        {"ts": "1.2", "user": "U2", "text": "reply", "thread_ts": "1.1"},
    ]}})
    messages = await SlackProvider(TOKEN).read_thread("C1", "1.1")

    assert _form(t.requests[0]) == {"channel": "C1", "ts": "1.1"}
    assert [m.id for m in messages] == ["1.1", "1.2"]


@pytest.mark.asyncio
async def test_open_dm_returns_a_channel_id_post_message_can_use(transport):
    t = transport({
        "conversations.open": {"ok": True, "channel": {"id": "D123"}},
        "chat.postMessage": {"ok": True, "ts": "3.3", "channel": "D123"},
    })
    provider = SlackProvider(TOKEN)
    dm = await provider.open_dm("U0ATR9G06MA")
    posted = await provider.post_message(dm, "no usable export today")

    assert dm == "D123"
    assert _form(t.requests[0]) == {"users": "U0ATR9G06MA"}
    assert json.loads(t.requests[1].content)["channel"] == "D123"
    assert posted.channel == "D123"


@pytest.mark.asyncio
async def test_open_dm_surfaces_a_missing_scope_as_its_own_code(transport):
    """The live bot may not hold im:write — the caller must be able to tell."""
    transport({"conversations.open": {"ok": False, "error": "missing_scope"}})
    with pytest.raises(MessagingError) as exc:
        await SlackProvider(TOKEN).open_dm("U1")
    assert exc.value.code == "missing_scope"


@pytest.mark.asyncio
async def test_delete_message(transport):
    t = transport({"chat.delete": {"ok": True, "ts": "1.1", "channel": "C1"}})
    await SlackProvider(TOKEN).delete_message("C1", "1.1")
    assert _form(t.requests[0]) == {"channel": "C1", "ts": "1.1"}


@pytest.mark.asyncio
async def test_missing_arguments_fail_before_any_request(transport):
    t = transport({})
    provider = SlackProvider(TOKEN)
    for call in (
        provider.reply_in_thread("C1", "", "x"),
        provider.read_thread("C1", ""),
        provider.open_dm(""),
        provider.delete_message("C1", ""),
    ):
        with pytest.raises(MessagingError):
            await call
    assert t.requests == []


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def test_token_comes_from_settings(monkeypatch):
    monkeypatch.setattr(
        "app.infrastructure.messaging.slack.bot_token", lambda: "from-settings"
    )
    assert SlackProvider().token == "from-settings"


@pytest.mark.asyncio
async def test_no_token_configured_is_a_clear_refusal(monkeypatch, transport):
    t = transport({})
    monkeypatch.setattr("app.infrastructure.messaging.slack.bot_token", lambda: "")
    with pytest.raises(MessagingError) as exc:
        await SlackProvider().post_message("C1", "hi")
    assert exc.value.code == "no_token"
    assert t.requests == []


def test_scrub_replaces_a_secret_and_ignores_short_noise():
    assert scrub(f"boom {TOKEN}", TOKEN) == "boom ***"
    assert scrub("boom", "") == "boom"
    assert scrub("a-b", "a-b") == "a-b"  # too short to be a credential


@pytest.mark.asyncio
async def test_a_transport_failure_never_echoes_the_token(monkeypatch):
    def _explode(*args, **kwargs):
        raise RuntimeError(f"connect failed with Authorization: Bearer {TOKEN}")

    monkeypatch.setattr(
        "app.infrastructure.messaging.slack.httpx.AsyncClient", _explode
    )
    with pytest.raises(MessagingError) as exc:
        await SlackProvider(TOKEN).post_message("C1", "hi")
    assert TOKEN not in str(exc.value)
    assert "***" in str(exc.value)


# ---------------------------------------------------------------------------
# Approval threading, moved out of the webhook notifier
# ---------------------------------------------------------------------------

def test_apply_thread_context_is_a_no_op_without_a_thread():
    payload = {"channel": "C1", "text": "hi"}
    assert apply_thread_context(payload, None, "U1") == {"channel": "C1", "text": "hi"}


def test_apply_thread_context_threads_and_mentions_the_approver():
    payload = {
        "channel": "C1",
        "text": "approved",
        "blocks": [
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn", "text": "*plan*"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "second"}},
        ],
    }
    apply_thread_context(payload, "170.1", "U7")

    assert payload["thread_ts"] == "170.1"
    assert payload["text"] == "<@U7> approved"
    # Only the first mrkdwn section is tagged, exactly as before the move.
    assert payload["blocks"][1]["text"]["text"] == "<@U7> *plan*"
    assert payload["blocks"][2]["text"]["text"] == "second"


def test_apply_thread_context_threads_without_an_approver():
    payload = {"channel": "C1", "text": "approved"}
    apply_thread_context(payload, "170.1", "")
    assert payload == {"channel": "C1", "text": "approved", "thread_ts": "170.1"}


def test_registering_a_provider_first_does_not_hide_the_builtins():
    """The bundled providers are registered explicitly, not by import side
    effect — the module is cached after the first import, so a caller that
    registered its own provider first would otherwise never see slack again."""
    from app.infrastructure.messaging import registry

    saved_providers = dict(registry._PROVIDERS)
    registry._PROVIDERS.clear()
    reset_providers()
    try:
        @register_provider
        class _First(SlackProvider):
            name = "first-in"

        assert get_provider("slack").name == "slack"
        assert get_provider("first-in").name == "first-in"
    finally:
        registry._PROVIDERS.clear()
        registry._PROVIDERS.update(saved_providers)
        reset_providers()
