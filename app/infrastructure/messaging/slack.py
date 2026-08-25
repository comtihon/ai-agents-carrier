"""Slack implementation of the messaging provider interface.

Every Slack HTTP call the platform makes goes through this class: the ``slack``
workflow step, the management/MCP messaging tools, and the approval /
ask_context / addon notifications in
``app.infrastructure.notifications.webhook_notifier``.  There is deliberately
no second Slack client anywhere in the backend.

Credentials: the bot token is read from ``SLACK_BOT_TOKEN`` via settings (with a
bare-environment fallback for sandboxed/standalone use).  It is never accepted
from a workflow definition, a data source, or step config, and
:func:`scrub` keeps it out of every error string this module raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.infrastructure.messaging.base import (
    Message,
    MessagingError,
    MessagingProvider,
    PostedMessage,
)
from app.infrastructure.messaging.registry import register_provider

logger = logging.getLogger(__name__)

API_BASE = "https://slack.com/api"
POSTMESSAGE_URL = f"{API_BASE}/chat.postMessage"

_TIMEOUT = 10.0
_HISTORY_TIMEOUT = 30.0

# Slack section/input block text elements are capped at 3000 chars.
SLACK_BLOCK_TEXT_LIMIT = 2900


def scrub(text: str, *secrets: str) -> str:
    """Replace every non-empty secret in *text* with a marker.

    Error strings travel into run state and step output, which the UI and the
    API hand back to callers.  A token that reached either would be a leak that
    outlives the run, so scrubbing happens at the boundary where the text is
    built rather than being left to each caller.
    """
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, "***")
    return text


def bot_token() -> str:
    """The Slack bot token, from settings, falling back to the raw env var."""
    try:
        from app.core.config import get_settings

        token = (get_settings().slack_bot_token or "").strip()
    except Exception:  # noqa: BLE001 — settings must never break a post
        token = ""
    if token:
        return token
    import os

    return (os.environ.get("SLACK_BOT_TOKEN") or "").strip()


@dataclass(frozen=True)
class SlackMessage(Message):
    """A Slack message.

    :meth:`as_dict` adds Slack's own field names next to the neutral ones so
    that scripts and templates written against the Slack Web API (``ts``,
    ``user``, ``thread_ts``) keep working unchanged.  The aliases live here, in
    the provider, precisely so the neutral shape in the base class stays
    neutral — another provider adds its own names, not Slack's.
    """

    def as_dict(self) -> dict[str, Any]:
        data = super().as_dict()
        data["ts"] = self.id
        data["user"] = self.author
        data["thread_ts"] = self.thread_id
        return data


def apply_thread_context(
    payload: dict[str, Any], thread_id: str | None, approver_id: str | None
) -> dict[str, Any]:
    """Turn a ``chat.postMessage`` payload into a threaded, attributed reply.

    Used by the human-approval notify path: once a first approval has created a
    thread, every later notification for the same run belongs in that thread,
    tagged with whoever just decided.  The mention is prepended to ``text`` and
    to the first mrkdwn section block, so it shows up in both the notification
    preview and the rich message body.

    Mutates and returns *payload* — the caller passes the dict it is about to
    send.  This lives in the Slack provider module rather than in the generic
    webhook notifier because it is Slack payload knowledge, and there is one
    copy of it.
    """
    if not thread_id:
        return payload
    payload["thread_ts"] = thread_id
    if approver_id:
        mention = f"<@{approver_id}> "
        payload["text"] = mention + payload.get("text", "")
        for block in payload.get("blocks", []) or []:
            if block.get("type") == "section" and isinstance(block.get("text"), dict):
                block["text"]["text"] = mention + block["text"].get("text", "")
                break
    return payload


@register_provider
class SlackProvider(MessagingProvider):
    """Slack Web API provider."""

    name = "slack"

    def __init__(self, token: str | None = None) -> None:
        # An explicit token is for tests only; production passes nothing and the
        # token is resolved per call, so a secret rotation takes effect without
        # a restart.
        self._token = token

    # ── plumbing ────────────────────────────────────────────────────────────

    @property
    def token(self) -> str:
        return self._token if self._token is not None else bot_token()

    async def _call(
        self,
        method: str,
        *,
        json_body: dict[str, Any] | None = None,
        form: dict[str, Any] | None = None,
        timeout: float = _TIMEOUT,
    ) -> dict[str, Any]:
        """POST one Slack Web API method and return its parsed ``ok`` payload.

        Read methods are sent form-encoded (the shape every Slack method
        accepts) and ``chat.postMessage`` as JSON, because ``blocks`` is a
        nested structure that form encoding cannot carry.
        """
        token = self.token
        if not token:
            raise MessagingError(
                "SLACK_BOT_TOKEN is not configured on this deployment", code="no_token"
            )
        url = f"{API_BASE}/{method}"
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if json_body is not None:
                    response = await client.post(url, headers=headers, json=json_body)
                else:
                    response = await client.post(
                        url,
                        headers=headers,
                        data={k: v for k, v in (form or {}).items() if v not in (None, "")},
                    )
            data = response.json()
        except MessagingError:
            raise
        except Exception as exc:  # noqa: BLE001 — network/JSON, reported not raised raw
            raise MessagingError(
                scrub(f"Slack {method} failed: {exc}", token), code="transport"
            ) from None
        if not isinstance(data, dict) or not data.get("ok"):
            code = str((data or {}).get("error") or "unknown_error")
            raise MessagingError(
                scrub(f"Slack {method} returned error '{code}'", token), code=code
            )
        return data

    @staticmethod
    def _to_message(channel: str, raw: dict[str, Any]) -> SlackMessage:
        return SlackMessage(
            id=str(raw.get("ts") or ""),
            channel=channel,
            text=str(raw.get("text") or ""),
            author=str(raw.get("user") or raw.get("bot_id") or ""),
            thread_id=(str(raw["thread_ts"]) if raw.get("thread_ts") else None),
            raw=raw,
        )

    # ── the interface ───────────────────────────────────────────────────────

    async def post_message(
        self,
        channel: str,
        text: str = "",
        *,
        thread_id: str | None = None,
        blocks: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PostedMessage:
        payload: dict[str, Any] = {**(extra or {}), "channel": channel}
        if text or "text" not in payload:
            payload["text"] = text
        if blocks is not None:
            payload["blocks"] = blocks
        if thread_id:
            payload["thread_ts"] = thread_id
        data = await self._call("chat.postMessage", json_body=payload)
        return PostedMessage(
            id=str(data.get("ts") or ""),
            channel=str(data.get("channel") or channel),
            raw=data,
        )

    async def post_payload(self, payload: dict[str, Any]) -> PostedMessage:
        """Post an already-built ``chat.postMessage`` payload.

        Slack-only escape hatch for the notification helpers, whose payloads are
        JSON templates a user wrote (blocks, attachments, unfurl flags).  It is
        not part of :class:`MessagingProvider`: a caller that uses it has
        already committed to Slack.
        """
        data = await self._call("chat.postMessage", json_body=dict(payload))
        return PostedMessage(
            id=str(data.get("ts") or ""),
            channel=str(data.get("channel") or payload.get("channel") or ""),
            raw=data,
        )

    async def reply_in_thread(
        self, channel: str, thread_id: str, text: str
    ) -> PostedMessage:
        if not thread_id:
            raise MessagingError("reply_in_thread needs a thread id", code="no_thread")
        return await self.post_message(channel, text, thread_id=thread_id)

    async def read_history(
        self, channel: str, oldest: str | None = None, limit: int | None = None
    ) -> list[Message]:
        data = await self._call(
            "conversations.history",
            form={"channel": channel, "oldest": oldest, "limit": limit},
            timeout=_HISTORY_TIMEOUT,
        )
        return [self._to_message(channel, m) for m in (data.get("messages") or [])]

    async def read_thread(self, channel: str, thread_id: str) -> list[Message]:
        if not thread_id:
            raise MessagingError("read_thread needs a thread id", code="no_thread")
        data = await self._call(
            "conversations.replies",
            form={"channel": channel, "ts": thread_id},
            timeout=_HISTORY_TIMEOUT,
        )
        return [self._to_message(channel, m) for m in (data.get("messages") or [])]

    async def open_dm(self, user_id: str) -> str:
        if not user_id:
            raise MessagingError("open_dm needs a user id", code="no_user")
        data = await self._call("conversations.open", form={"users": user_id})
        channel = data.get("channel") or {}
        channel_id = str(channel.get("id") or "") if isinstance(channel, dict) else ""
        if not channel_id:
            raise MessagingError(
                "Slack conversations.open returned no channel id", code="no_channel"
            )
        return channel_id

    async def delete_message(self, channel: str, message_id: str) -> None:
        if not message_id:
            raise MessagingError("delete_message needs a message id", code="no_message")
        await self._call("chat.delete", form={"channel": channel, "ts": message_id})
