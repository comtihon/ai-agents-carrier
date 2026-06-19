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


AnyAgentAddon = Annotated[Union[MCPAddon], Field(discriminator="type")]
