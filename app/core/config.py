from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_api_key_env(name: str) -> str:
    """Derive the env var name that carries the API key for an integration."""
    return name.upper().replace("-", "_") + "_API_KEY"


class LLMIntegrationConfig(BaseModel):
    """One LLM provider integration — any OpenAI/LiteLLM-compatible endpoint."""

    name: str
    base_url: str
    default_model: str
    # Name of the env var holding the API key. Defaults to `<NAME>_API_KEY`
    # so a single helm secret-ref is enough for built-in providers.
    api_key_env: str | None = None

    def resolved_api_key_env(self) -> str:
        return self.api_key_env or _default_api_key_env(self.name)

    def resolved_api_key(self) -> str | None:
        return os.environ.get(self.resolved_api_key_env())


class McpIntegrationConfig(BaseModel):
    """Resolved configuration for a single MCP server."""

    name: str
    enabled: bool
    transport: Literal["streamable_http", "sse", "stdio"]
    # HTTP transports
    url: str | None = None
    api_key: str | None = None
    # stdio transport
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}


class Settings(BaseSettings):
    app_name: str = "LangGraph Backend"
    environment: str = "local"
    api_prefix: str = "/api/v1"
    debug: bool = False
    graph_definitions_path: str = Field(default="graphs", alias="GRAPH_DEFINITIONS_PATH")

    # --- CORS ---
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000"],
        alias="ALLOWED_ORIGINS",
    )

    # --- Public base URL (used to build callback URLs in approval notifications) ---
    base_url: str = Field(default="http://localhost:8000", alias="BASE_URL")

    # --- Webhook / HTTP trigger ---
    webhook_secret: str | None = Field(default=None, alias="WEBHOOK_SECRET")

    # --- OAuth ---
    oauth_enabled: bool = Field(default=False, alias="OAUTH_ENABLED")
    oauth_jwks_url: str | None = Field(default=None, alias="OAUTH_JWKS_URL")
    oauth_issuer: str | None = Field(default=None, alias="OAUTH_ISSUER")
    oauth_audience: str | None = Field(default=None, alias="OAUTH_AUDIENCE")
    oauth_algorithms: list[str] = Field(default=["RS256"], alias="OAUTH_ALGORITHMS")

    # --- LLM ---
    # Name of the integration to use when a step has no explicit `llm_provider`.
    # Must match one of the entries in `LLM_INTEGRATIONS`.
    llm_provider: str | None = Field(default=None, alias="LLM_PROVIDER")
    # JSON-encoded list of LLM integrations. Each entry: {name, base_url,
    # default_model, api_key_env?}. Delivered via helm `llmIntegrations` list.
    # All integrations are treated as OpenAI/LiteLLM-compatible endpoints.
    llm_integrations_json: str | None = Field(default=None, alias="LLM_INTEGRATIONS")

    # --- Workflow backend ---
    # "localfiles" — read/write YAML files from graph_definitions_path (default).
    # "mongodb"    — read/write the workflow_definitions MongoDB collection.
    workflow_backend_type: Literal["localfiles", "mongodb"] = Field(
        default="localfiles", alias="WORKFLOW_BACKEND"
    )

    # --- MongoDB ---
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_database: str = Field(default="langgraph_backend", alias="MONGODB_DATABASE")

    # --- Slack ---
    slack_signing_secret: str = Field(default="", alias="SLACK_SIGNING_SECRET")
    slack_bot_token: str = Field(default="", alias="SLACK_BOT_TOKEN")
    slack_approvals_channel: str = Field(default="", alias="SLACK_APPROVALS_CHANNEL")

    # --- OpenHands ---
    openhands_base_url: str = Field(default="http://openhands:3000", alias="OPENHANDS_BASE_URL")
    openhands_api_key: str | None = Field(default=None, alias="OPENHANDS_API_KEY")
    openhands_timeout_seconds: float = Field(default=30.0, alias="OPENHANDS_TIMEOUT_SECONDS")
    openhands_task_timeout_seconds: float = Field(default=1800.0, alias="OPENHANDS_TASK_TIMEOUT_SECONDS")
    openhands_poll_interval_seconds: float = Field(default=10.0, alias="OPENHANDS_POLL_INTERVAL_SECONDS")
    openhands_mock_mode: bool = Field(default=True, alias="OPENHANDS_MOCK_MODE")

    # --- Figma MCP ---
    mcp_figma_enabled: bool = Field(default=False, alias="MCP_FIGMA_ENABLED")
    mcp_figma_transport: Literal["streamable_http", "sse"] = Field(default="streamable_http", alias="MCP_FIGMA_TRANSPORT")
    mcp_figma_url: str = Field(default="", alias="MCP_FIGMA_URL")
    mcp_figma_api_key: str | None = Field(default=None, alias="MCP_FIGMA_API_KEY")

    # --- Jira MCP ---
    mcp_jira_enabled: bool = Field(default=False, alias="MCP_JIRA_ENABLED")
    mcp_jira_transport: Literal["streamable_http", "sse", "stdio"] = Field(default="streamable_http", alias="MCP_JIRA_TRANSPORT")
    # HTTP transport fields
    mcp_jira_url: str = Field(default="", alias="MCP_JIRA_URL")
    mcp_jira_api_key: str | None = Field(default=None, alias="MCP_JIRA_API_KEY")
    # stdio transport fields (sooperset/mcp-atlassian via uvx)
    mcp_jira_jira_url: str | None = Field(default=None, alias="MCP_JIRA_JIRA_URL")
    mcp_jira_username: str | None = Field(default=None, alias="MCP_JIRA_USERNAME")
    mcp_jira_api_token: str | None = Field(default=None, alias="MCP_JIRA_API_TOKEN")

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

    def _jira_integration(self) -> dict[str, Any]:
        if self.mcp_jira_transport == "stdio":
            env = {k: v for k, v in {
                "JIRA_URL": self.mcp_jira_jira_url,
                "JIRA_USERNAME": self.mcp_jira_username,
                "JIRA_API_TOKEN": self.mcp_jira_api_token,
            }.items() if v}
            return dict(name="jira", enabled=self.mcp_jira_enabled, transport="stdio",
                        command="uvx", args=["mcp-atlassian"], env=env)
        return dict(name="jira", enabled=self.mcp_jira_enabled, transport=self.mcp_jira_transport,
                    url=self.mcp_jira_url, api_key=self.mcp_jira_api_key)

    def get_llm_integrations(self) -> list[LLMIntegrationConfig]:
        """Parse the LLM_INTEGRATIONS JSON env var into a typed list."""
        if not self.llm_integrations_json:
            return []
        raw = json.loads(self.llm_integrations_json)
        if not isinstance(raw, list):
            raise ValueError("LLM_INTEGRATIONS must be a JSON array of integration objects")
        return [LLMIntegrationConfig.model_validate(item) for item in raw]

    def get_llm_integration(self, name: str) -> LLMIntegrationConfig | None:
        """Look up an integration by name (case-insensitive)."""
        target = name.lower()
        for integration in self.get_llm_integrations():
            if integration.name.lower() == target:
                return integration
        return None

    def get_mcp_integrations(self) -> list[McpIntegrationConfig]:
        candidates: list[dict[str, Any]] = [
            dict(name="figma",  enabled=self.mcp_figma_enabled,  transport=self.mcp_figma_transport,  url=self.mcp_figma_url,  api_key=self.mcp_figma_api_key),
            self._jira_integration(),
            dict(name="miro",   enabled=self.mcp_miro_enabled,   transport=self.mcp_miro_transport,   url=self.mcp_miro_url,   api_key=self.mcp_miro_api_key),
            dict(name="notion", enabled=self.mcp_notion_enabled, transport=self.mcp_notion_transport, url=self.mcp_notion_url, api_key=self.mcp_notion_api_key),
            dict(name="github", enabled=self.mcp_github_enabled, transport=self.mcp_github_transport, url=self.mcp_github_url, api_key=self.mcp_github_api_key),
        ]
        return [McpIntegrationConfig(**c) for c in candidates
                if c["enabled"] and (c.get("url") or c.get("transport") == "stdio")]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
