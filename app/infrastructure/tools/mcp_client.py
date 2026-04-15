from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.core.config import Settings

logger = logging.getLogger(__name__)


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
        self._tool_server: dict[str, str] = {}  # tool name → server name

    async def start(self) -> None:
        server_configs = self._build_server_configs()
        if not server_configs:
            return
        self._client = MultiServerMCPClient(server_configs)
        self._tools = []
        self._tool_server = {}
        for server_name in server_configs:
            server_tools = await self._client.get_tools(server_name=server_name)
            for tool in server_tools:
                self._tools.append(tool)
                self._tool_server[tool.name] = server_name
            logger.info(
                "MCP server '%s': loaded %d tool(s): %s",
                server_name,
                len(server_tools),
                [t.name for t in server_tools],
            )

    async def stop(self) -> None:
        self._client = None
        self._tools = []
        self._tool_server = {}

    def get_tools(self) -> list[BaseTool]:
        return list(self._tools)

    def get_tool(self, name: str) -> BaseTool | None:
        return next((t for t in self._tools if t.name == name), None)

    def get_tool_server(self, name: str) -> str | None:
        """Return the MCP server name that provides the given tool, or None if unknown."""
        return self._tool_server.get(name)

    def _build_server_configs(self) -> dict[str, dict[str, Any]]:
        configs: dict[str, dict[str, Any]] = {}
        for integration in self._settings.get_mcp_integrations():
            if integration.transport == "stdio":
                cfg: dict[str, Any] = {
                    "transport": "stdio",
                    "command": integration.command,
                    "args": integration.args,
                }
                if integration.env:
                    cfg["env"] = integration.env
            else:
                cfg = {
                    "transport": integration.transport,
                    "url": integration.url,
                }
                if integration.api_key:
                    cfg["headers"] = {"Authorization": f"Bearer {integration.api_key}"}
            configs[integration.name] = cfg
        return configs
