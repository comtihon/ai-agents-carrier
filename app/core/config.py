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


# Env vars that look like credentials by suffix but belong to the backend only.
_SYSTEM_ONLY_ALIASES = {
    "WEBHOOK_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_SIGNING_SECRET",
    "OPENHANDS_API_KEY",
    "DOCKER_REGISTRY_PASSWORD",
    "DOCKER_REGISTRY_USERNAME",
}

# Suffixes that identify credential/secret fields worth forwarding to agents.
_CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_JSON", "_CREDENTIALS")


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

    # --- Standalone LLM API keys (forwarded to Docker/K8s agent containers) ---
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    google_application_credentials_json: str | None = Field(default=None, alias="GOOGLE_APPLICATION_CREDENTIALS_JSON")

    # --- K8s agent runtime ---
    # Namespace where K8sRuntime deploys agent Helm releases.
    # Must match the namespace the backend pod runs in so its ServiceAccount has RBAC.
    agent_namespace: str = Field(default="langgraph", alias="AGENT_NAMESPACE")
    # Override the callback URL passed to K8s agents. Useful when the default
    # base_url is an OAuth-protected external URL and agents need to call back
    # via an internal cluster URL instead. Defaults to base_url when not set.
    agent_callback_url: str | None = Field(default=None, alias="AGENT_CALLBACK_URL")

    # --- Docker registry auth (used by DockerRuntime to pull private images) ---
    # Set DOCKER_REGISTRY_USERNAME + DOCKER_REGISTRY_PASSWORD to enable auth.
    # GAR:    username=oauth2accesstoken  password=$(gcloud auth print-access-token)
    # ECR:    username=AWS               password=$(aws ecr get-login-password)
    # Other:  plain username / password or personal access token
    docker_registry_username: str | None = Field(default=None, alias="DOCKER_REGISTRY_USERNAME")
    docker_registry_password: str | None = Field(default=None, alias="DOCKER_REGISTRY_PASSWORD")

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

    # --- Agent polling ---
    agent_poll_interval_seconds: int = Field(default=10, alias="AGENT_POLL_INTERVAL_SECONDS")
    agent_max_loops: int = Field(default=3, alias="AGENT_MAX_LOOPS")

    # --- Meta-LLM (lightweight analysis after agent steps complete) ---
    meta_llm_provider: str | None = Field(default="anthropic", alias="META_LLM_PROVIDER")
    meta_llm_model: str = Field(default="claude-haiku-4-5-20251001", alias="META_LLM_MODEL")

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

    def get_forwardable_config(self) -> dict[str, str]:
        """Return {NAME: value} for all credential-like env vars currently set.

        Sources (merged, os.environ wins on collision):
        - .env file via python-dotenv (covers local dev)
        - os.environ (covers Docker / K8s injection)

        Any var matching a credential suffix and not in _SYSTEM_ONLY_ALIASES is
        included.  No Settings field declaration required — users can forward
        any credential to agents by setting the env var, zero code changes.
        """
        from dotenv import dotenv_values
        env_file = self.model_config.get("env_file", ".env")
        dot: dict[str, str | None] = {}
        if env_file:
            try:
                dot = dotenv_values(env_file)  # type: ignore[assignment]
            except Exception:
                pass

        # os.environ takes precedence over .env file values
        merged = {k: v for k, v in dot.items() if v is not None}
        merged.update(os.environ)

        result: dict[str, str] = {}
        for key, val in merged.items():
            if not val:
                continue
            if key in _SYSTEM_ONLY_ALIASES:
                continue
            if any(key.endswith(s) for s in _CREDENTIAL_SUFFIXES):
                result[key] = val
        return result

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
