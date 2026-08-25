"""The messaging tools, exercised through the MCP surface.

Registration and agent/MCP parity are covered in test_management_mcp_tools.py.
What is pinned here is behaviour: each tool reaches the provider abstraction
(not Slack directly), reports rather than raises, and never accepts a token.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.mcp.management_server import build_management_mcp, register_management_tools
from app.application import management_tools as core
from app.application.management_tools import ManagementDeps
from app.infrastructure.auth.authorization import Permission, set_current_permissions
from app.infrastructure.messaging import (
    Message,
    MessagingError,
    PostedMessage,
    register_provider,
    reset_providers,
)
from app.infrastructure.messaging import registry as provider_registry
from tests.test_yaml_graph_messaging import FAKE, FakeProvider


@pytest.fixture
def provider():
    register_provider(FakeProvider)
    reset_providers()
    yield provider_registry.get_provider(FAKE)
    provider_registry._PROVIDERS.pop(FAKE, None)
    reset_providers()


class _Container:
    def __init__(self) -> None:
        self.yaml_graph_registry = MagicMock()
        self.yaml_graph_registry.list_definitions.return_value = []
        self.run_repository = AsyncMock()
        self.workflow_backend = None
        self.agent_backend = None
        self.data_source_backend = None
        self.event_backend = None
        self.refresh_runner = None
        self.settings = MagicMock()
        self.live_runners: dict = {}


@pytest.fixture
def mcp():
    server = build_management_mcp()
    register_management_tools(server, lambda: _Container())
    return server


def _deps() -> ManagementDeps:
    return ManagementDeps(registry=MagicMock(), run_repository=None)


# ---------------------------------------------------------------------------
# Behaviour over MCP
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_message_goes_through_the_named_provider(mcp, provider):
    result = str(await mcp.call_tool(
        "post_message", {"channel": "C1", "text": "hello", "provider": FAKE}
    ))
    assert provider.posted == [{"channel": "C1", "text": "hello", "thread_id": None}]
    assert "Posted to C1" in result


@pytest.mark.asyncio
async def test_post_message_can_reply_in_a_thread(mcp, provider):
    result = str(await mcp.call_tool(
        "post_message",
        {"channel": "C1", "text": "ack", "thread_id": "1.1", "provider": FAKE},
    ))
    assert provider.posted[0]["thread_id"] == "1.1"
    assert "in thread 1.1" in result


@pytest.mark.asyncio
async def test_read_messages_renders_ids_authors_and_text(mcp, provider):
    provider.history = [
        Message(id="9.1", channel="C1", text="FRIST 128498: 3", author="U9"),
        Message(id="9.0", channel="C1", text="in a thread", author="U8",
                thread_id="8.0"),
    ]
    result = str(await mcp.call_tool(
        "read_messages", {"channel": "C1", "limit": 5, "provider": FAKE}
    ))
    assert "9.1 U9: FRIST 128498: 3" in result
    assert "[thread 8.0]" in result
    assert provider.last_history["limit"] == 5


@pytest.mark.asyncio
async def test_read_messages_on_an_empty_channel(mcp, provider):
    assert "No messages." in str(await mcp.call_tool(
        "read_messages", {"channel": "C1", "provider": FAKE}
    ))


@pytest.mark.asyncio
async def test_read_thread(mcp, provider):
    provider.threads["1.1"] = [
        Message(id="1.1", channel="C1", text="root", author="U1"),
        Message(id="1.2", channel="C1", text="reply", author="U2", thread_id="1.1"),
    ]
    result = str(await mcp.call_tool(
        "read_thread", {"channel": "C1", "thread_id": "1.1", "provider": FAKE}
    ))
    assert "root" in result and "reply" in result


@pytest.mark.asyncio
async def test_send_direct_message_opens_the_dm_first(mcp, provider):
    result = str(await mcp.call_tool(
        "send_direct_message", {"user_id": "U7", "text": "heads up", "provider": FAKE}
    ))
    assert provider.opened == ["U7"]
    assert provider.posted == [{"channel": "D-U7", "text": "heads up",
                                "thread_id": None}]
    assert "Direct message sent to U7" in result


@pytest.mark.asyncio
async def test_delete_message(mcp, provider):
    result = str(await mcp.call_tool(
        "delete_message", {"channel": "C1", "message_id": "1.1", "provider": FAKE}
    ))
    assert provider.deleted == [("C1", "1.1")]
    assert "deleted" in result


@pytest.mark.asyncio
async def test_a_provider_error_is_reported_not_raised(mcp, provider):
    provider.fail_with = MessagingError("not_in_channel", code="not_in_channel")
    result = str(await mcp.call_tool(
        "post_message", {"channel": "C1", "text": "x", "provider": FAKE}
    ))
    assert "Could not post the message" in result
    assert "not_in_channel" in result


@pytest.mark.asyncio
async def test_an_unknown_provider_is_reported_not_raised(mcp):
    result = str(await mcp.call_tool(
        "read_messages", {"channel": "C1", "provider": "telepathy"}
    ))
    assert "Unknown messaging provider" in result


@pytest.mark.asyncio
async def test_no_tool_accepts_a_token(mcp):
    """The credential is deployment configuration, not a tool argument — so
    there is no parameter an LLM or an MCP client could put one in."""
    tools = {t.name: t for t in await mcp.list_tools()}
    for name in ("post_message", "read_messages", "read_thread",
                 "send_direct_message", "delete_message"):
        properties = set(tools[name].inputSchema.get("properties") or {})
        assert not {"token", "bot_token", "api_key", "auth"} & properties


# ---------------------------------------------------------------------------
# Gates (the map itself is asserted in tests/unit/test_tool_permission_gates.py)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_writer_cannot_delete_a_message(provider):
    set_current_permissions(
        frozenset({Permission.ACCESS, Permission.READ, Permission.WRITE})
    )
    result = await core.delete_message(_deps(), "C1", "1.1", FAKE)
    assert "Not permitted" in result and "delete" in result
    assert provider.deleted == []


@pytest.mark.asyncio
async def test_a_reader_cannot_post(provider):
    set_current_permissions(frozenset({Permission.ACCESS, Permission.READ}))
    result = await core.post_message(_deps(), "C1", "x", "", FAKE)
    assert "Not permitted" in result
    assert provider.posted == []


@pytest.mark.asyncio
async def test_a_reader_may_read(provider):
    set_current_permissions(frozenset({Permission.ACCESS, Permission.READ}))
    assert "No messages." in await core.read_messages(_deps(), "C1", 5, "", FAKE)
