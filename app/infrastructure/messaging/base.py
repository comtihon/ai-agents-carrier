"""Provider-neutral messaging: the contract every chat provider implements.

The platform talks to chat systems through this one interface so that the
``slack`` workflow step, the management tools and the notification helpers all
share a single implementation per operation.  Adding WhatsApp or Teams later is
a new :class:`MessagingProvider` subclass plus a registry entry — no new step
type, no change to any caller.

The operation set is deliberately the union of what today's callers actually
need and nothing more: post, reply in a thread, read a channel's history, read
one thread, open a DM channel, and delete a message.  There is no reactions or
emoji support because no caller asks for one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


class MessagingError(RuntimeError):
    """A provider rejected an operation.

    Carries the provider's own error code (``channel_not_found``,
    ``not_in_channel``, ``missing_scope``, …) so callers can branch on it
    without parsing prose.  The message never contains the bot token: see
    :func:`app.infrastructure.messaging.slack.scrub`.
    """

    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Message:
    """One message, in provider-neutral terms.

    ``raw`` keeps the provider's own payload for callers that genuinely need a
    field this interface does not model.  It is never included in step output —
    only :meth:`as_dict` is — so a provider response cannot leak into run state
    by accident.
    """

    id: str
    channel: str
    text: str
    author: str = ""
    thread_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """The neutral shape a workflow step writes into run state."""
        return {
            "id": self.id,
            "channel": self.channel,
            "text": self.text,
            "author": self.author,
            "thread_id": self.thread_id,
        }


@dataclass(frozen=True)
class PostedMessage:
    """The identity of a message this platform just created."""

    id: str
    channel: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {"message_id": self.id, "channel": self.channel}


class MessagingProvider(ABC):
    """A chat provider the platform can post to and read from.

    Implementations are registered under :attr:`name` (see
    ``app.infrastructure.messaging.registry``) and are constructed with no
    arguments, reading their own credentials from settings/env exactly like the
    rest of the backend.  A credential must never arrive through a workflow
    definition, a data source or step config.
    """

    #: Registry key.  ``provider: <name>`` in a step config selects it.
    name: ClassVar[str] = ""

    @abstractmethod
    async def post_message(
        self,
        channel: str,
        text: str = "",
        *,
        thread_id: str | None = None,
        blocks: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> PostedMessage:
        """Post *text* to *channel*; return the new message's identity.

        ``extra`` is a provider-specific passthrough for fields this interface
        does not model (Slack's ``unfurl_links``, ``attachments``, …).  Callers
        that want to stay portable leave it unset.
        """

    @abstractmethod
    async def reply_in_thread(
        self, channel: str, thread_id: str, text: str
    ) -> PostedMessage:
        """Post *text* as a reply inside the thread rooted at *thread_id*."""

    @abstractmethod
    async def read_history(
        self, channel: str, oldest: str | None = None, limit: int | None = None
    ) -> list[Message]:
        """Read *channel*'s recent messages, newest first.

        ``oldest`` is a provider timestamp cursor; ``limit`` caps the page size.
        """

    @abstractmethod
    async def read_thread(self, channel: str, thread_id: str) -> list[Message]:
        """Read every message in one thread, the root message first.

        This is what makes posting idempotent: a caller checks whether its
        reply is already there before adding a second one.
        """

    @abstractmethod
    async def open_dm(self, user_id: str) -> str:
        """Open (or reuse) the direct-message channel with *user_id*.

        Returns a channel id that :meth:`post_message` accepts, so the failure
        path — "DM instead of posting to the channel" — is the ordinary post
        path with a different channel.
        """

    @abstractmethod
    async def delete_message(self, channel: str, message_id: str) -> None:
        """Delete one message this platform posted."""
