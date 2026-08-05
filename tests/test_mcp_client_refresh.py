"""Tests for McpToolsProvider.refresh_server's client lifecycle."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.core.config import McpIntegrationConfig
from app.infrastructure.tools import mcp_client as mcp_client_module
from app.infrastructure.tools.mcp_client import McpToolsProvider


class _FakeMcpClient:
    """Stand-in for MultiServerMCPClient that records aclose() calls."""

    instances: list["_FakeMcpClient"] = []

    def __init__(self, configs) -> None:
        self.configs = configs
        self.closed = False
        _FakeMcpClient.instances.append(self)

    async def get_tools(self, server_name):
        return []

    async def aclose(self) -> None:
        self.closed = True


def _settings_with(name: str) -> MagicMock:
    settings = MagicMock()
    settings.get_mcp_integrations.return_value = [
        McpIntegrationConfig(name=name, enabled=True, transport="streamable_http", url="http://x")
    ]
    return settings


async def test_refresh_server_closes_previously_replaced_client(monkeypatch):
    _FakeMcpClient.instances = []
    monkeypatch.setattr(mcp_client_module, "MultiServerMCPClient", _FakeMcpClient)

    provider = McpToolsProvider(_settings_with("test"))
    await provider.refresh_server("test")
    first = _FakeMcpClient.instances[0]
    assert first.closed is False

    await provider.refresh_server("test")
    second = _FakeMcpClient.instances[1]
    assert first.closed is True  # replaced client is closed
    assert second.closed is False  # currently-active client stays open


async def test_refresh_server_best_effort_when_client_has_no_close(monkeypatch):
    class _NoCloseClient:
        def __init__(self, configs) -> None:
            pass

        async def get_tools(self, server_name):
            return []

    monkeypatch.setattr(mcp_client_module, "MultiServerMCPClient", _NoCloseClient)

    provider = McpToolsProvider(_settings_with("test"))
    await provider.refresh_server("test")
    await provider.refresh_server("test")  # must not raise despite no close/aclose
