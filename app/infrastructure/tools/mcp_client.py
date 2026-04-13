from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import Settings


class McpToolsProvider:
    """
    Manages connections to configured MCP servers and exposes their tools
    as LangChain BaseTool instances.

    Lifecycle: call start() once at application startup and stop() at shutdown.
    When no MCP servers are enabled, start() is a no-op and get_tools() returns [].
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: MultiServerMCPClient | None = None
        self._tools: list[BaseTool] = []

    async def start(self) -> None:
        server_configs = self._build_server_configs()
        if not server_configs:
            return
        self._client = MultiServerMCPClient(server_configs)
        self._tools = await self._client.get_tools()

    async def stop(self) -> None:
        self._client = None
        self._tools = []

    def get_tools(self) -> list[BaseTool]:
        return list(self._tools)

    def get_tool(self, name: str) -> BaseTool | None:
        return next((t for t in self._tools if t.name == name), None)

    def _build_server_configs(self) -> dict[str, dict[str, Any]]:
        configs: dict[str, dict[str, Any]] = {}
        for integration in self._settings.get_mcp_integrations():
            cfg: dict[str, Any] = {
                "transport": integration.transport,
                "url": integration.url,
            }
            if integration.api_key:
                cfg["headers"] = {"Authorization": f"Bearer {integration.api_key}"}
            configs[integration.name] = cfg
        return configs
