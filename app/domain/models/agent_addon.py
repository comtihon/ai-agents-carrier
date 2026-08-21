from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class AgentAddon(BaseModel):
    type: str
    hidden: bool = False


class MCPAddon(AgentAddon):
    type: Literal["mcp"] = "mcp"
    # server-name → enabled toggle
    servers: dict[str, bool] = Field(default_factory=dict)

    def enabled_servers(self) -> set[str]:
        return {name for name, enabled in self.servers.items() if enabled}


class S3Addon(AgentAddon):
    type: Literal["s3"] = "s3"
    bucket: str = ""
    path: str = ""


class ToolsAddon(AgentAddon):
    type: Literal["tools"] = "tools"
    # tool-name → enabled toggle (github / jira / graphify)
    tools: dict[str, bool] = Field(default_factory=dict)

    def enabled_tools(self) -> set[str]:
        return {name for name, enabled in self.tools.items() if enabled}


class DatasourceAddon(AgentAddon):
    """One registered data source an agent may call, and which operations of it.

    Carries no credential.  The agent never talks to the upstream API: it calls
    this backend's own ``/mcp/datasources`` mount, which resolves
    ``DataSourceDefinition.auth`` in-process (see
    ``app.infrastructure.datasources.executor.build_auth_headers``).  For a
    ``service_identity`` source the bearer is the *carrier's own* identity, so
    the checked-operations list below is the only thing standing between an
    agent and every permission that identity holds.

    Several of these addons may be attached to one agent — one per data source.
    The effective grant is the union across them, computed where the agent's
    MCP server list is built (``app.steps.agent_executor._build_agent_config``)
    and enforced server-side on both ``list_tools`` and ``call_tool``
    (``app.api.mcp.datasources_server.ScopedDatasourcesMCP``).
    """

    type: Literal["datasource"] = "datasource"
    source_id: str = ""
    allowed_operations: list[str] = Field(default_factory=list)
    """Operation names of ``source_id`` this agent may invoke.

    **An empty list grants nothing.**  It is a deny, never a wildcard: an agent
    whose author ticked no boxes must not inherit every operation of the source,
    which for an HTTP source can include deletes.  Fail-closed is the
    established convention for every other addon in this module too — no ``mcp``
    addon means no MCP servers, no ``tools`` addon means no tools.

    Checking an operation *is* the authorization for it, writes included.  There
    is no separate allow-writes flag and no approval step: if it is in this
    list, the agent may call it.
    """


AnyAgentAddon = Annotated[
    Union[MCPAddon, S3Addon, ToolsAddon, DatasourceAddon],
    Field(discriminator="type"),
]
