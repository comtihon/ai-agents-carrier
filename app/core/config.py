from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class McpIntegrationConfig(BaseModel):
    """Resolved configuration for a single MCP server."""

    name: str
    enabled: bool
    transport: Literal["streamable_http", "sse"]
    url: str
    api_key: str | None


class Settings(BaseSettings):
    app_name: str = "AI Development Orchestration System"
    environment: Literal["local", "dev", "prod", "test"] = "local"
    api_prefix: str = "/api/v1"
    debug: bool = False
    workflow_definitions_path: str = Field(default="workflows", alias="WORKFLOW_DEFINITIONS_PATH")
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_database: str = Field(default="langgraph_backend", alias="MONGODB_DATABASE")
    workflow_runs_collection: str = "workflow_runs"
    openhands_base_url: str = Field(default="http://openhands:3000", alias="OPENHANDS_BASE_URL")
    openhands_api_key: str | None = Field(default=None, alias="OPENHANDS_API_KEY")
    openhands_timeout_seconds: float = Field(default=60.0, alias="OPENHANDS_TIMEOUT_SECONDS")
    openhands_mock_mode: bool = Field(default=True, alias="OPENHANDS_MOCK_MODE")
    langserve_path: str = Field(default="/langserve/workflow-runner", alias="LANGSERVE_PATH")

    # --- Figma MCP ---
    mcp_figma_enabled: bool = Field(default=False, alias="MCP_FIGMA_ENABLED")
    mcp_figma_transport: Literal["streamable_http", "sse"] = Field(default="streamable_http", alias="MCP_FIGMA_TRANSPORT")
    mcp_figma_url: str = Field(default="", alias="MCP_FIGMA_URL")
    mcp_figma_api_key: str | None = Field(default=None, alias="MCP_FIGMA_API_KEY")

    # --- Jira MCP ---
    mcp_jira_enabled: bool = Field(default=False, alias="MCP_JIRA_ENABLED")
    mcp_jira_transport: Literal["streamable_http", "sse"] = Field(default="streamable_http", alias="MCP_JIRA_TRANSPORT")
    mcp_jira_url: str = Field(default="", alias="MCP_JIRA_URL")
    mcp_jira_api_key: str | None = Field(default=None, alias="MCP_JIRA_API_KEY")

    # --- Miro MCP ---
    mcp_miro_enabled: bool = Field(default=False, alias="MCP_MIRO_ENABLED")
    mcp_miro_transport: Literal["streamable_http", "sse"] = Field(default="streamable_http", alias="MCP_MIRO_TRANSPORT")
    mcp_miro_url: str = Field(default="", alias="MCP_MIRO_URL")
    mcp_miro_api_key: str | None = Field(default=None, alias="MCP_MIRO_API_KEY")

    # --- Notion MCP ---
    mcp_notion_enabled: bool = Field(default=False, alias="MCP_NOTION_ENABLED")
    mcp_notion_transport: Literal["streamable_http", "sse"] = Field(default="streamable_http", alias="MCP_NOTION_TRANSPORT")
    mcp_notion_url: str = Field(default="", alias="MCP_NOTION_URL")
    mcp_notion_api_key: str | None = Field(default=None, alias="MCP_NOTION_API_KEY")

    # --- GitHub MCP ---
    mcp_github_enabled: bool = Field(default=False, alias="MCP_GITHUB_ENABLED")
    mcp_github_transport: Literal["streamable_http", "sse"] = Field(default="streamable_http", alias="MCP_GITHUB_TRANSPORT")
    mcp_github_url: str = Field(default="", alias="MCP_GITHUB_URL")
    mcp_github_api_key: str | None = Field(default=None, alias="MCP_GITHUB_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def get_mcp_integrations(self) -> list[McpIntegrationConfig]:
        """Return the list of enabled MCP integration configs."""
        candidates: list[dict[str, Any]] = [
            dict(name="figma", enabled=self.mcp_figma_enabled, transport=self.mcp_figma_transport, url=self.mcp_figma_url, api_key=self.mcp_figma_api_key),
            dict(name="jira",  enabled=self.mcp_jira_enabled,  transport=self.mcp_jira_transport,  url=self.mcp_jira_url,  api_key=self.mcp_jira_api_key),
            dict(name="miro",  enabled=self.mcp_miro_enabled,  transport=self.mcp_miro_transport,  url=self.mcp_miro_url,  api_key=self.mcp_miro_api_key),
            dict(name="notion",enabled=self.mcp_notion_enabled,transport=self.mcp_notion_transport,url=self.mcp_notion_url,api_key=self.mcp_notion_api_key),
            dict(name="github",enabled=self.mcp_github_enabled,transport=self.mcp_github_transport,url=self.mcp_github_url,api_key=self.mcp_github_api_key),
        ]
        return [McpIntegrationConfig(**c) for c in candidates if c["enabled"] and c["url"]]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
