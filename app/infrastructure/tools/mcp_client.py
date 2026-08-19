from __future__ import annotations

import inspect
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
        # Per-server clients created by refresh_server(); kept alive so their
        # tools stay callable.
        self._refresh_clients: dict[str, MultiServerMCPClient] = {}

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
        self._refresh_clients = {}

    def get_tools(self) -> list[BaseTool]:
        return list(self._tools)

    def get_tool(self, name: str) -> BaseTool | None:
        return next((t for t in self._tools if t.name == name), None)

    def get_tool_server(self, name: str) -> str | None:
        """Return the MCP server name that provides the given tool, or None if unknown."""
        return self._tool_server.get(name)

    async def refresh_server(self, name: str) -> None:
        """(Re)connect a single MCP server and swap in its tools atomically.

        Used for servers declared with ``eager_start: false`` and for
        hot-reloading a server whose tool list changed.  Raises on
        connection failure so the caller can retry.
        """
        cfg = self._config_for(name)
        if cfg is None:
            logger.debug("MCP server '%s' is not configured — nothing to refresh", name)
            return
        client = MultiServerMCPClient({name: cfg})
        server_tools = await client.get_tools(server_name=name)
        # Atomic swap: build the new lists first, then rebind.
        tools = [t for t in self._tools if self._tool_server.get(t.name) != name]
        tool_server = {k: v for k, v in self._tool_server.items() if v != name}
        for tool in server_tools:
            tools.append(tool)
            tool_server[tool.name] = name
        self._tools = tools
        self._tool_server = tool_server
        old_client = self._refresh_clients.get(name)
        self._refresh_clients[name] = client
        if old_client is not None:
            await self._maybe_close_client(old_client)
        logger.info(
            "MCP server '%s': refreshed with %d tool(s): %s",
            name, len(server_tools), [t.name for t in server_tools],
        )

    @staticmethod
    async def _maybe_close_client(client: MultiServerMCPClient) -> None:
        """Best-effort close of a replaced MCP client, if it exposes one."""
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if closer is None:
            return
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("failed to close replaced MCP client", exc_info=True)

    def _config_for(self, name: str) -> dict[str, Any] | None:
        """Build the client config for one configured MCP server, or None."""
        for integration in self._settings.get_mcp_integrations():
            if integration.name != name:
                continue
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
                api_key = integration.resolved_api_key()
                if api_key:
                    cfg["headers"] = {"Authorization": f"Bearer {api_key}"}
            return cfg
        return None

    def _build_server_configs(self) -> dict[str, dict[str, Any]]:
        configs: dict[str, dict[str, Any]] = {}
        for integration in self._settings.get_mcp_integrations():
            # A server declared with eager_start: false is never dialled during
            # startup — an agent-side stdio binary absent from the backend
            # container, or a server this very process hosts and which only
            # answers once startup completes (see refresh_server).
            if not integration.eager_start:
                continue
            cfg = self._config_for(integration.name)
            if cfg is not None:
                configs[integration.name] = cfg
        return configs
