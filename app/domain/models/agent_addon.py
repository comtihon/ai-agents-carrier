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


AnyAgentAddon = Annotated[Union[MCPAddon, S3Addon, ToolsAddon], Field(discriminator="type")]
