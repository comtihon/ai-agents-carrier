from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
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


DEFAULT_SERVICE_IDENTITY = "default"


class ServiceIdentityConfig(BaseModel):
    """One outbound OAuth2 service identity (RFC 7523 JWT bearer grant).

    Nothing here is provider-specific: ``scopes`` is forwarded verbatim, so a
    deployment may use plain scopes or any provider's scope URNs. The private
    key may be inlined (``private_key``) or, so a JSON blob in config need not
    embed a PEM, named via ``private_key_env``.
    """

    name: str = DEFAULT_SERVICE_IDENTITY
    # OAuth2 token endpoint of the authorization server.
    token_url: str
    client_id: str
    # `aud` of the client assertion — usually the authorization server issuer.
    audience: str
    # Space-separated scope string, opaque to this code.
    scopes: str = "openid"
    # `kid` header of the signing key, as registered with the provider.
    key_id: str | None = None
    # PEM-encoded RSA private key; literal "\n" escapes are normalized.
    private_key: str | None = None
    # Name of an env var holding the PEM instead of inlining it here.
    private_key_env: str | None = None

    def resolved_private_key(self) -> str | None:
        """The PEM with literal ``\\n`` escapes turned into real newlines.

        Secret managers and env-var injection frequently deliver PEM blocks as
        a single line with escaped newlines, which no PEM parser accepts.
        """
        raw = self.private_key
        if not raw and self.private_key_env:
            raw = os.environ.get(self.private_key_env)
        if not raw:
            return None
        return raw.replace("\\n", "\n")


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
    app_name: str = "AI Agents Carrier"
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

    # --- Pub/Sub trigger ---
    # Off by default: with no GCP project (tests, local runs) a subscriber
    # would only produce credential errors on every startup.
    pubsub_enabled: bool = Field(default=False, alias="PUBSUB_ENABLED")
    # Project the short topic / subscription names of a `pubsub` trigger step
    # resolve against. Fully qualified paths in a step ignore this.
    pubsub_project_id: str | None = Field(default=None, alias="PUBSUB_PROJECT_ID")
    # Prefix for subscriptions the backend creates itself, so they are
    # recognisable in the console and cannot collide with hand-made ones.
    pubsub_subscription_prefix: str = Field(default="aac-", alias="PUBSUB_SUBSCRIPTION_PREFIX")
    pubsub_ack_deadline_seconds: int = Field(default=60, alias="PUBSUB_ACK_DEADLINE_SECONDS")
    # Upper bound on messages pulled concurrently per subscription.
    pubsub_max_messages: int = Field(default=10, alias="PUBSUB_MAX_MESSAGES")
    # A message that does not match the step's schema is acknowledged and
    # dropped by default — redelivering it forever would be a poison pill.
    # Set false to nack instead (topic-level retry / dead-lettering).
    pubsub_drop_invalid_messages: bool = Field(default=True, alias="PUBSUB_DROP_INVALID_MESSAGES")

    # --- OAuth ---
    oauth_enabled: bool = Field(default=False, alias="OAUTH_ENABLED")
    oauth_jwks_url: str | None = Field(default=None, alias="OAUTH_JWKS_URL")
    oauth_issuer: str | None = Field(default=None, alias="OAUTH_ISSUER")
    oauth_audience: str | None = Field(default=None, alias="OAUTH_AUDIENCE")
    oauth_algorithms: list[str] = Field(default=["RS256"], alias="OAUTH_ALGORITHMS")

    # --- Outbound service identity (OAuth2 JWT bearer grant, RFC 7523) ---
    # When enabled, the backend mints its own access tokens from an OAuth2
    # authorization server using a signed client assertion, and attaches them
    # to outbound calls that opt in (`service_identity` auth).
    #
    # Two ways to configure them, and both may be used at once:
    #   SERVICE_AUTH_IDENTITIES — a JSON array of named identities, for
    #     deployments that call several protected services with different
    #     credentials. Mirrors LLM_INTEGRATIONS.
    #   SERVICE_AUTH_* (below) — a single unnamed identity, registered under
    #     DEFAULT_SERVICE_IDENTITY. Simpler for the common one-identity case.
    service_auth_identities_json: str | None = Field(default=None, alias="SERVICE_AUTH_IDENTITIES")
    # Name of the identity used when an auth block names none. Defaults to the
    # sole configured identity when there is exactly one.
    service_auth_default_identity: str | None = Field(default=None, alias="SERVICE_AUTH_DEFAULT_IDENTITY")
    service_auth_enabled: bool = Field(default=False, alias="SERVICE_AUTH_ENABLED")
    # OAuth2 token endpoint of the authorization server.
    service_auth_token_url: str | None = Field(default=None, alias="SERVICE_AUTH_TOKEN_URL")
    service_auth_client_id: str | None = Field(default=None, alias="SERVICE_AUTH_CLIENT_ID")
    # `kid` header of the signing key, as registered with the provider.
    service_auth_key_id: str | None = Field(default=None, alias="SERVICE_AUTH_KEY_ID")
    # PEM-encoded RSA private key; literal "\n" escapes are normalized.
    service_auth_private_key: str | None = Field(default=None, alias="SERVICE_AUTH_PRIVATE_KEY")
    # `aud` of the client assertion — usually the authorization server issuer.
    service_auth_audience: str | None = Field(default=None, alias="SERVICE_AUTH_AUDIENCE")
    # Space-separated scope string, forwarded verbatim (opaque to this code).
    service_auth_scopes: str = Field(default="openid", alias="SERVICE_AUTH_SCOPES")

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

    # --- Semble MCP (stdio; agent-side binary — never launched in backend container) ---
    mcp_semble_enabled: bool = Field(default=True, alias="MCP_SEMBLE_ENABLED")

    # --- Data sources MCP (served in-process at /mcp/datasources) ---
    # Connected lazily in the background, never during startup: the mounted
    # endpoint only answers once the HTTP server is accepting requests.
    mcp_datasources_enabled: bool = Field(default=True, alias="MCP_DATASOURCES_ENABLED")
    mcp_datasources_url: str | None = Field(default=None, alias="MCP_DATASOURCES_URL")
    mcp_datasources_api_key: str | None = Field(default=None, alias="MCP_DATASOURCES_API_KEY")

    # --- Management MCP (served in-process at /mcp/management) ---
    # Exposes the platform's own CRUD + run control tools to external MCP
    # clients. Token-gated and fail-closed (see app.api.app._ManagementAuthWrapper).
    management_mcp_enabled: bool = Field(default=True, alias="MANAGEMENT_MCP_ENABLED")
    management_mcp_api_key: str | None = Field(default=None, alias="MANAGEMENT_MCP_API_KEY")
    # Host header values FastMCP's DNS-rebinding guard accepts for this endpoint.
    # Unset means the loopback defaults only (see
    # app.api.mcp.management_server.DEFAULT_ALLOWED_HOSTS), so a deployment that
    # is reached under a real hostname must list it here, e.g.
    # MANAGEMENT_MCP_ALLOWED_HOSTS='["langgraph.airteam.cloud"]'.
    management_mcp_allowed_hosts: list[str] | None = Field(
        default=None, alias="MANAGEMENT_MCP_ALLOWED_HOSTS"
    )

    # --- Agent polling ---
    agent_poll_interval_seconds: int = Field(default=10, alias="AGENT_POLL_INTERVAL_SECONDS")
    agent_max_loops: int = Field(default=3, alias="AGENT_MAX_LOOPS")

    # --- Meta-LLM (lightweight analysis after agent steps complete) ---
    meta_llm_provider: str | None = Field(default="openrouter", alias="META_LLM_PROVIDER")
    meta_llm_model: str = Field(default="moonshotai/kimi-k2.6", alias="META_LLM_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("mcp_datasources_api_key", "management_mcp_api_key")
    @classmethod
    def _strip_api_key(cls, value: str | None) -> str | None:
        """Trim surrounding whitespace from the in-process MCP API keys.

        Secret Manager values routinely carry a trailing newline and an HTTP
        header value cannot contain one, so ``Bearer <key>\n`` can never equal
        any inbound Authorization header: the endpoint then rejects every caller
        (including this backend's own MCP self-connection) with no other signal.
        Empty / whitespace-only still normalizes to a falsy value, which both
        auth wrappers read as "no usable key" and fail closed on.
        """
        return value.strip() if isinstance(value, str) else value

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

    def resolved_mcp_datasources_url(self) -> str:
        """URL of the in-process data sources MCP server.

        Defaults to loopback so the backend talks to its own mounted endpoint
        without depending on the externally reachable base_url (which may be
        OAuth-protected).
        """
        if self.mcp_datasources_url:
            return self.mcp_datasources_url
        port = os.environ.get("PORT", "8000")
        return f"http://127.0.0.1:{port}/mcp/datasources"

    def resolved_service_auth_private_key(self) -> str | None:
        """Private key of the flat ``SERVICE_AUTH_*`` identity, newlines fixed."""
        if not self.service_auth_private_key:
            return None
        return self.service_auth_private_key.replace("\\n", "\n")

    def get_service_identities(self) -> list[ServiceIdentityConfig]:
        """Every configured outbound identity, in declaration order.

        The flat ``SERVICE_AUTH_*`` set, when present, is registered as
        :data:`DEFAULT_SERVICE_IDENTITY` and listed first — so deployments that
        predate ``SERVICE_AUTH_IDENTITIES`` keep working unchanged. A JSON entry
        of the same name wins, which is how such a deployment migrates.

        Raises ``ValueError`` when the JSON is not an array or an entry is
        missing a required field; the caller surfaces that at startup.
        """
        identities: list[ServiceIdentityConfig] = []
        # Any flat field at all means someone intended a single identity — so a
        # half-filled set is reported field by field rather than as "no identity
        # configured", which would hide what is actually missing.
        if any(
            (
                self.service_auth_token_url,
                self.service_auth_client_id,
                self.service_auth_audience,
                self.service_auth_private_key,
                self.service_auth_key_id,
            )
        ):
            identities.append(
                ServiceIdentityConfig(
                    name=DEFAULT_SERVICE_IDENTITY,
                    # Empty strings keep validation here lenient: completeness is
                    # reported per identity by the token provider, which can say
                    # *which* field is missing.
                    token_url=self.service_auth_token_url or "",
                    client_id=self.service_auth_client_id or "",
                    audience=self.service_auth_audience or "",
                    scopes=self.service_auth_scopes,
                    key_id=self.service_auth_key_id,
                    private_key=self.service_auth_private_key,
                )
            )
        if self.service_auth_identities_json:
            raw = json.loads(self.service_auth_identities_json)
            if not isinstance(raw, list):
                raise ValueError(
                    "SERVICE_AUTH_IDENTITIES must be a JSON array of identity objects"
                )
            for item in raw:
                parsed = ServiceIdentityConfig.model_validate(item)
                identities = [i for i in identities if i.name != parsed.name]
                identities.append(parsed)
        return identities

    def get_service_identity(self, name: str | None = None) -> ServiceIdentityConfig | None:
        """Look up an identity by name (case-insensitive); None → the default."""
        identities = self.get_service_identities()
        if not identities:
            return None
        target = (name or self.resolved_default_service_identity() or "").lower()
        for identity in identities:
            if identity.name.lower() == target:
                return identity
        return None

    def resolved_default_service_identity(self) -> str | None:
        """Identity used when an auth block names none.

        The explicit setting wins; otherwise a single configured identity is
        unambiguously the default. With several and no setting there is no
        default — an auth block must name one rather than silently borrowing
        another service's credentials.
        """
        if self.service_auth_default_identity:
            return self.service_auth_default_identity
        identities = self.get_service_identities()
        if len(identities) == 1:
            return identities[0].name
        return None

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

    def _build_mcp_candidates(self) -> list[dict[str, Any]]:
        return [
            dict(name="figma",  enabled=self.mcp_figma_enabled,  transport=self.mcp_figma_transport,  url=self.mcp_figma_url,  api_key=self.mcp_figma_api_key),
            self._jira_integration(),
            dict(name="miro",   enabled=self.mcp_miro_enabled,   transport=self.mcp_miro_transport,   url=self.mcp_miro_url,   api_key=self.mcp_miro_api_key),
            dict(name="notion", enabled=self.mcp_notion_enabled, transport=self.mcp_notion_transport, url=self.mcp_notion_url, api_key=self.mcp_notion_api_key),
            dict(name="github", enabled=self.mcp_github_enabled, transport=self.mcp_github_transport, url=self.mcp_github_url, api_key=self.mcp_github_api_key),
            # Bare `semble` starts the MCP stdio server; the CLI takes no repo
            # positional (repo is a per-call tool argument) and rejects one.
            dict(name="semble", enabled=self.mcp_semble_enabled, transport="stdio", command="semble", args=[]),
            # Served by this process at /mcp/datasources — connected lazily.
            dict(name="datasources", enabled=self.mcp_datasources_enabled, transport="streamable_http",
                 url=self.resolved_mcp_datasources_url(), api_key=self.mcp_datasources_api_key),
        ]

    def get_mcp_integrations(self) -> list[McpIntegrationConfig]:
        candidates = self._build_mcp_candidates()
        return [McpIntegrationConfig(**c) for c in candidates
                if c["enabled"] and (c.get("url") or c.get("transport") == "stdio")]

    def list_mcp_candidates(self) -> list[dict[str, Any]]:
        """Return name + enabled for ALL known MCP servers regardless of URL/enabled state."""
        return [{"name": c["name"], "enabled": c["enabled"]} for c in self._build_mcp_candidates()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
