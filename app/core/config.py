from __future__ import annotations

import json
import os
import re
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
    """One MCP server an agent may be granted.

    Declared entirely in configuration (``MCP_INTEGRATIONS``) so that neither
    the backend nor the agent image carries knowledge of any specific server —
    the same contract ``AgentToolSpec`` has for bash-level tools.  Adding a
    server is a values.yaml entry, not a code change.

    Secrets are named, not inlined: ``api_key_env`` points at an env var the
    deployment fills from a k8s Secret, so the JSON blob itself is safe to keep
    in a ConfigMap.  ``api_key`` remains available for local ``.env`` use.
    """

    name: str
    enabled: bool = True
    transport: Literal["streamable_http", "sse", "stdio"] = "streamable_http"
    # HTTP transports
    url: str | None = None
    # How an HTTP transport authenticates. "bearer" (the default) sends the
    # token below; "none" states that the endpoint genuinely wants no
    # credential, which is what keeps a tokenless entry from being reported as
    # ready when it is really just unconfigured.  Ignored for stdio servers,
    # which authenticate through ``env`` / ``env_from_config`` instead.
    auth: Literal["bearer", "none"] = "bearer"
    # Bearer token: inlined (local dev) or named (production).
    api_key: str | None = None
    api_key_env: str | None = None
    # stdio transport
    command: str | None = None
    args: list[str] = []
    env: dict[str, str] = {}
    # env var name → env var to copy the value from, for stdio servers whose
    # credentials live in the deployment's own secrets.
    env_from_config: dict[str, str] = {}
    # False for stdio servers whose CLI cannot be re-hosted over HTTP by the
    # agent (no --transport/--port flags): the agent then wires them as plain
    # stdio entries instead of pre-starting an HTTP proxy.
    prestart_http: bool = True
    # False for servers the backend must not dial during startup: an agent-side
    # stdio binary absent from the backend container, or a server this very
    # process hosts (only reachable once the HTTP server accepts requests).
    # Such servers are connected later via ``McpToolsProvider.refresh_server``.
    eager_start: bool = True
    # "self_datasources" resolves ``url`` to this process's own mounted
    # /mcp/datasources endpoint at read time; "value" uses ``url`` verbatim.
    url_from: Literal["value", "self_datasources"] = "value"
    # Endpoint a spawned agent must dial when it differs from the one the
    # backend uses — a loopback or in-cluster address the agent's own pod cannot
    # resolve. Empty means the agent gets ``url``.
    agent_url: str | None = None

    def resolved_agent_url(self) -> str | None:
        """The endpoint to hand a spawned agent for this server."""
        return self.agent_url or self.url

    def resolved_api_key(self) -> str | None:
        """The bearer token, from the inline value or the named env var."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or None
        return None


class AgentToolEnvSpec(BaseModel):
    """How one env var of a tool is filled.

    Either a literal ``value`` or ``from_config`` naming a backend env var /
    ``.env`` key whose value is copied.  Nothing here is tool-specific — the
    operator decides which secret feeds which variable.
    """

    value: str | None = None
    from_config: str | None = None


class AgentToolCliSpec(BaseModel):
    """A CLI invocation exposed to the agent as an MCP-style tool call.

    ``args`` is an argv template: ``{name}`` placeholders are filled from the
    tool-call arguments, ``{name|fallback}`` supplies a default.  ``optional``
    appends extra argv only when that argument is present.  Lets a CLI-only
    tool (no HTTP MCP mode) be routed through the agent's mcp() gateway without
    the agent knowing the command's grammar.
    """

    args: list[str] = Field(default_factory=list)
    # Tool-call arguments that must be present; missing ones are an error.
    required: list[str] = Field(default_factory=list)
    # arg-name → argv fragment appended when the caller supplied that arg.
    optional: dict[str, list[str]] = Field(default_factory=dict)
    # Working directory template, e.g. "{repo}".
    cwd: str | None = None
    # Files that must exist under cwd for the call to make sense.
    requires_files: list[str] = Field(default_factory=list)
    timeout_seconds: int = 60
    description: str = ""


class AgentToolWorkspaceHook(BaseModel):
    """A command run once per restored workspace repo before the agent starts.

    Used by tools that keep a per-repo cache or index — the agent runs the hook
    only in repos that already contain ``requires_files``, and never fails the
    run because of it.
    """

    args: list[str] = Field(default_factory=list)
    requires_files: list[str] = Field(default_factory=list)
    timeout_seconds: int = 120


class AgentToolSpec(BaseModel):
    """One bash-level tool an agent may be granted.

    Declared entirely in configuration (``AGENT_TOOLS``) so that neither the
    backend nor the agent image carries knowledge of any specific tool: the
    backend resolves ``env`` and ships it with the tool, the agent checks that
    ``command`` exists locally and registers what it can actually run.
    """

    label: str = ""
    description: str = ""
    # Binary the agent needs on PATH for this tool to be usable.
    command: str | None = None
    # env var name → how to fill it.
    env: dict[str, AgentToolEnvSpec] = Field(default_factory=dict)
    # Regex matching bash commands that exercise this tool (UI/gating hint).
    bash_match: str | None = None
    # MCP-style tool name → CLI invocation template.
    cli_tools: dict[str, AgentToolCliSpec] = Field(default_factory=dict)
    # Optional per-repo command run after the workspace is restored.
    workspace_hook: AgentToolWorkspaceHook | None = None


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

# Name of the built-in, in-process data sources MCP bridge.  Named here because
# three places have to agree on it: the built-in registry entry, the addon
# picker that must NOT offer it (see list_mcp_candidates), and the agent
# spawn path that reaches it only through a scoped grant
# (app.steps.agent_executor).
DATASOURCES_MCP_NAME = "datasources"


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
    # When the last trigger step using a backend-created subscription goes away
    # (node removed, workflow deleted), delete the subscription too instead of
    # leaving it to accrue a backlog. Subscriptions named by a step or
    # datasource are never deleted — they are not the backend's to remove.
    pubsub_delete_orphaned_subscriptions: bool = Field(
        default=True, alias="PUBSUB_DELETE_ORPHANED_SUBSCRIPTIONS",
    )

    # --- OAuth ---
    oauth_enabled: bool = Field(default=False, alias="OAUTH_ENABLED")
    oauth_jwks_url: str | None = Field(default=None, alias="OAUTH_JWKS_URL")
    oauth_issuer: str | None = Field(default=None, alias="OAUTH_ISSUER")
    oauth_audience: str | None = Field(default=None, alias="OAUTH_AUDIENCE")
    oauth_algorithms: list[str] = Field(default=["RS256"], alias="OAUTH_ALGORITHMS")

    # --- Role-based authorization (see app.infrastructure.auth.authorization) ---
    # Off by default so an upgrade cannot lock out an existing deployment; the
    # policy logs a warning at startup while it is off.
    auth_enforce_permissions: bool = Field(default=False, alias="AUTH_ENFORCE_PERMISSIONS")
    # Identity-provider project id, used to read the standard
    # urn:zitadel:iam:org:project:<id>:roles claim that machine-user tokens carry.
    auth_project_id: str | None = Field(default=None, alias="AUTH_PROJECT_ID")
    # Role names, supplied per deployment. Empty lists are meaningful: with
    # enforcement on, an empty AUTH_ACCESS_ROLES rejects everything (fail closed)
    # and an empty AUTH_ADMIN_ROLES means nobody can create unsandboxed steps.
    # AUTH_ACCESS_ROLES is an allow-list: it is what keeps an identity provider
    # shared with customers from granting them access to this API.
    auth_access_roles: list[str] = Field(default_factory=list, alias="AUTH_ACCESS_ROLES")
    auth_read_roles: list[str] = Field(default_factory=list, alias="AUTH_READ_ROLES")
    auth_write_roles: list[str] = Field(default_factory=list, alias="AUTH_WRITE_ROLES")
    auth_delete_roles: list[str] = Field(default_factory=list, alias="AUTH_DELETE_ROLES")
    auth_admin_roles: list[str] = Field(default_factory=list, alias="AUTH_ADMIN_ROLES")

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

    # --- Data-source deletion approvals ---
    # A workflow that reaches a destructive data-source operation (DELETE, or
    # one flagged ``destructive``) opens an approval case and waits. Turning
    # this off restores the pre-feature behaviour: deletes run unattended.
    approvals_enabled: bool = Field(default=True, alias="APPROVALS_ENABLED")
    # Unbroken identical *human* decisions on one workflow+source+operation
    # after which the meta-LLM decides the next case itself.
    approval_auto_decide_threshold: int = Field(
        default=10, alias="APPROVAL_AUTO_DECIDE_THRESHOLD"
    )
    # How long an autonomous decision stays cancellable before it takes effect.
    # Zero disables the window (the decision applies immediately).
    approval_veto_window_seconds: int = Field(
        default=60, alias="APPROVAL_VETO_WINDOW_SECONDS"
    )
    # How long a surface that cannot suspend (an agent's MCP tool call) blocks
    # waiting for a person before the case expires unapproved.
    approval_wait_timeout_seconds: float = Field(
        default=3600.0, alias="APPROVAL_WAIT_TIMEOUT_SECONDS"
    )
    approval_poll_interval_seconds: float = Field(
        default=3.0, alias="APPROVAL_POLL_INTERVAL_SECONDS"
    )

    # --- OpenHands ---
    openhands_base_url: str = Field(default="http://openhands:3000", alias="OPENHANDS_BASE_URL")
    openhands_api_key: str | None = Field(default=None, alias="OPENHANDS_API_KEY")
    openhands_timeout_seconds: float = Field(default=30.0, alias="OPENHANDS_TIMEOUT_SECONDS")
    openhands_task_timeout_seconds: float = Field(default=1800.0, alias="OPENHANDS_TASK_TIMEOUT_SECONDS")
    openhands_poll_interval_seconds: float = Field(default=10.0, alias="OPENHANDS_POLL_INTERVAL_SECONDS")
    openhands_mock_mode: bool = Field(default=True, alias="OPENHANDS_MOCK_MODE")

    # --- MCP server registry ---
    # JSON array declaring every MCP server agents may be granted:
    #   [{"name": "github", "transport": "streamable_http",
    #     "url": "https://...", "api_key_env": "GITHUB_MCP_TOKEN"}]
    # Empty by default: an agent's mcp addon can only enable what is declared
    # here, and a server that is not declared grants nothing.  The in-process
    # `datasources` server is always present (see _builtin_mcp_integrations).
    mcp_integrations_json: str = Field(default="", alias="MCP_INTEGRATIONS")

    # --- Standalone LLM API keys (forwarded to Docker/K8s agent containers) ---
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    google_application_credentials_json: str | None = Field(default=None, alias="GOOGLE_APPLICATION_CREDENTIALS_JSON")

    # --- Agent tool registry ---
    # JSON object declaring the bash-level tools agents may be granted:
    #   {"<name>": {"command": "...", "env": {"ENV": {"from_config": "KEY"}}}}
    # Empty by default: an agent's tools addon can only enable what is declared
    # here, and a tool that is not declared grants nothing.
    agent_tools_json: str = Field(default="", alias="AGENT_TOOLS")

    # --- K8s agent runtime ---
    # Namespace where K8sRuntime deploys agent Helm releases.
    # Must match the namespace the backend pod runs in so its ServiceAccount has RBAC.
    agent_namespace: str = Field(default="langgraph", alias="AGENT_NAMESPACE")
    # Override the callback URL passed to K8s agents. Useful when the default
    # base_url is an OAuth-protected external URL and agents need to call back
    # via an internal cluster URL instead. Defaults to base_url when not set.
    agent_callback_url: str | None = Field(default=None, alias="AGENT_CALLBACK_URL")
    # Existing KSA that agent pods run as. Empty means the agent chart creates
    # its own ServiceAccount, which carries no Workload Identity annotation and
    # no RBAC — fine locally, wrong in a cluster where agents need an identity
    # of their own (and must not inherit the backend's). A `serviceAccount.*`
    # entry in an agent's helm_values overrides this.
    agent_service_account: str | None = Field(default=None, alias="AGENT_SERVICE_ACCOUNT")

    # --- Local agent runtime (pi-cloud-agent as a child process) ---
    # Directory of a pi-cloud-agent checkout (https://github.com/comtihon/pi-cloud-agent).
    # Empty means the `local` runtime is unavailable and a step must use
    # `docker` / `k8s` instead — there is no in-process agent to fall back to.
    local_agent_dir: str = Field(default="", alias="LOCAL_AGENT_DIR")
    # Command that starts the agent's HTTP server, run with LOCAL_AGENT_DIR as cwd.
    local_agent_command: str = Field(default="node src/server.js", alias="LOCAL_AGENT_COMMAND")

    # --- Python script sandbox (workflow `python` steps with sandbox: true) ---
    # Image used by the docker / k8s sandbox runtimes.  Must contain a Python
    # interpreter on PATH; no backend dependencies are needed inside it.
    script_sandbox_image: str = Field(default="python:3.12-slim", alias="SCRIPT_SANDBOX_IMAGE")
    # Wall-clock limit for a sandboxed script, in seconds.
    script_sandbox_timeout: float = Field(default=60.0, alias="SCRIPT_SANDBOX_TIMEOUT")
    # Memory cap for a sandboxed script, in MiB.
    script_sandbox_memory_mb: int = Field(default=512, alias="SCRIPT_SANDBOX_MEMORY_MB")

    # --- Run-state divergence probe (see stream_graph_to_pause) ---
    # Diagnostic for the dual-state cleanup: after every streamed chunk,
    # compare the hand-merged `current_state` the runner maintains against the
    # LangGraph checkpoint's own reducer-applied values, and log any key that
    # differs. Read-only — it never changes what is persisted. Costs one extra
    # checkpoint read per node, so it is meant to be switched off again once
    # the divergences it finds have been fixed.
    state_divergence_probe: bool = Field(default=False, alias="STATE_DIVERGENCE_PROBE")

    # --- Docker registry auth (used by DockerRuntime to pull private images) ---
    # Set DOCKER_REGISTRY_USERNAME + DOCKER_REGISTRY_PASSWORD to enable auth.
    # GAR:    username=oauth2accesstoken  password=$(gcloud auth print-access-token)
    # ECR:    username=AWS               password=$(aws ecr get-login-password)
    # Other:  plain username / password or personal access token
    docker_registry_username: str | None = Field(default=None, alias="DOCKER_REGISTRY_USERNAME")
    docker_registry_password: str | None = Field(default=None, alias="DOCKER_REGISTRY_PASSWORD")

    # --- Data sources MCP (served in-process at /mcp/datasources) ---
    # Connected lazily in the background, never during startup: the mounted
    # endpoint only answers once the HTTP server is accepting requests.
    mcp_datasources_enabled: bool = Field(default=True, alias="MCP_DATASOURCES_ENABLED")
    mcp_datasources_url: str | None = Field(default=None, alias="MCP_DATASOURCES_URL")
    mcp_datasources_api_key: str | None = Field(default=None, alias="MCP_DATASOURCES_API_KEY")
    # Signs the per-run capability grants an agent presents to /mcp/datasources
    # (see app.infrastructure.auth.datasource_grant).  A grant names the exact
    # source/operation pairs one agent may call; it is NOT a shared credential,
    # so it must not be the same secret an operator might hand out — hence its
    # own key, falling back to MCP_DATASOURCES_API_KEY so a deployment that
    # already has one keeps working without a new secret.  Neither set means no
    # grant can be minted or verified at all, and datasource addons grant
    # nothing (fail closed).
    datasource_grant_signing_key: str | None = Field(
        default=None, alias="DATASOURCE_GRANT_SIGNING_KEY"
    )
    # Lifetime of a minted grant.  Grants are stateless and therefore cannot be
    # revoked before they expire — terminating a run does not instantly kill its
    # grant — so this is the revocation window, not just a hygiene setting.
    datasource_grant_ttl_seconds: int = Field(
        default=86400, alias="DATASOURCE_GRANT_TTL_SECONDS"
    )
    # Host header values the /mcp/datasources DNS-rebinding guard accepts, on
    # top of loopback. Unset derives them from the agent-facing URL; set this
    # only when an agent reaches the backend under a name the backend does not
    # know it has. See Settings.datasources_mcp_allowed_hosts.
    mcp_datasources_allowed_hosts: list[str] | None = Field(
        default=None, alias="MCP_DATASOURCES_ALLOWED_HOSTS"
    )

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

    # --- Google Workspace (Sheets / Drive / Docs) ---
    # Service account the backend impersonates for `google` data source auth.
    # Not a secret: it is an email the backend holds
    # roles/iam.serviceAccountTokenCreator on, and the only principal a
    # `google` auth block is allowed to name. Documents are shared with this
    # address, so it is also what the UI tells the user to grant access to.
    # Unset means the feature is off — better than defaulting to some
    # account's authority nobody asked for.
    google_impersonate_sa: str | None = Field(default=None, alias="GOOGLE_IMPERSONATE_SA")
    # OAuth2 scopes to mint that token with. The metadata server's own token is
    # cloud-platform-scoped and Sheets/Drive reject it, so the scopes have to be
    # stated; an auth block may narrow them further but not widen them.
    google_impersonate_scopes: list[str] = Field(
        default=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
        ],
        alias="GOOGLE_IMPERSONATE_SCOPES",
    )

    # --- Sheet bindings, tier 2 (LLM-generated pure transforms) ---
    # Reads are on by default: a generated read computes a value and hands it
    # back, and the worst a bad one does is return a wrong number to the step
    # that asked.
    sheets_compute_enabled: bool = Field(default=True, alias="SHEETS_COMPUTE_ENABLED")
    # Writes are OFF by default, and this is not timidity. A generated read is
    # wrong in the response; a generated write is wrong *in somebody's
    # spreadsheet*, and a golden fixture over five sample rows is not evidence
    # over five hundred real ones. Turning this on is a deployment-level
    # decision, taken once the operators have looked at what tier 2 actually
    # generates for their sheets.
    sheets_compute_writes_enabled: bool = Field(
        default=False, alias="SHEETS_COMPUTE_WRITES_ENABLED"
    )
    # Human approvals a tier-2 write binding must accumulate before the
    # meta-LLM's autonomy streak is allowed to answer for it. Independent of
    # `approval_auto_decide_threshold` on purpose: that streak is about an
    # operation a person wrote and has watched work, this one is about code a
    # model wrote and nobody has watched at scale yet.
    sheets_compute_write_probation_runs: int = Field(
        default=5, alias="SHEETS_COMPUTE_WRITE_PROBATION_RUNS"
    )
    # Model the compile step calls, pinned. Recorded into every binding it
    # generates (`resolution.model_id`) and part of the cache key, so changing
    # it invalidates generated code rather than silently mixing two models'
    # output across one datasource.
    sheets_compute_model: str | None = Field(default=None, alias="SHEETS_COMPUTE_MODEL")
    # Attempts the compile loop makes, feeding each gate's exact rejection back
    # to the model. Three, then it stops and says so: a fourth attempt on the
    # same instruction has, in practice, nothing new to go on.
    sheets_compute_max_attempts: int = Field(
        default=3, alias="SHEETS_COMPUTE_MAX_ATTEMPTS"
    )
    # Wall-clock ceiling for one transform run in the sandbox. Far below a
    # workflow python step's 60s: a pure transform over a probe's worth of rows
    # either finishes at once or is looping.
    sheets_compute_timeout_seconds: float = Field(
        default=10.0, alias="SHEETS_COMPUTE_TIMEOUT_SECONDS"
    )

    # --- Meta-LLM (lightweight analysis after agent steps complete) ---
    meta_llm_provider: str | None = Field(default="openrouter", alias="META_LLM_PROVIDER")
    meta_llm_model: str = Field(default="moonshotai/kimi-k2.6", alias="META_LLM_MODEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator(
        "mcp_datasources_api_key",
        "management_mcp_api_key",
        "datasource_grant_signing_key",
    )
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

    def datasources_mcp_allowed_hosts(self) -> list[str]:
        """Host header values the ``/mcp/datasources`` mount accepts.

        FastMCP's DNS-rebinding guard defaults to loopback-with-a-port only, so
        a spawned agent dialling this mount at the deployment's real address
        would be rejected with a 421 before its request was ever read.  The
        address is already declared (``AGENT_CALLBACK_URL`` / ``BASE_URL``), so
        it is derived from there rather than made an extra thing to configure;
        ``MCP_DATASOURCES_ALLOWED_HOSTS`` overrides when the agent reaches the
        backend under a name the backend does not know it has (a
        service-mesh or ingress rewrite, say).  Loopback is always allowed on
        top of whatever this returns.
        """
        if self.mcp_datasources_allowed_hosts is not None:
            return [h for h in self.mcp_datasources_allowed_hosts if h]
        from urllib.parse import urlsplit

        netloc = urlsplit(self.resolved_agent_mcp_datasources_url()).netloc
        if not netloc:
            return []
        # Both forms: the declared address may or may not carry a port, and a
        # client is free to include or omit the default one for the scheme.
        host = netloc.split("@")[-1]
        bare = host.rsplit(":", 1)[0] if ":" in host and not host.endswith("]") else host
        return list(dict.fromkeys([host, bare, f"{bare}:*"]))

    def resolved_datasource_grant_signing_key(self) -> str | None:
        """Key that signs and verifies ``/mcp/datasources`` capability grants.

        Falls back to ``MCP_DATASOURCES_API_KEY`` so a deployment that already
        has one needs no new secret to start using scoped datasource addons.
        ``None`` means no grant can be signed or verified, which every caller
        treats as "no datasource access" rather than "unrestricted".
        """
        for candidate in (self.datasource_grant_signing_key, self.mcp_datasources_api_key):
            stripped = (candidate or "").strip()
            if stripped:
                return stripped
        return None

    def resolved_agent_mcp_datasources_url(self) -> str:
        """URL of the data sources MCP server as a *spawned agent* must dial it.

        The backend talks to its own mounted endpoint over loopback, which a
        docker or k8s agent in its own container cannot resolve.  Agents are
        given the same address they call back on instead — the deployment
        already has to make that one reachable from an agent.
        """
        base = (self.agent_callback_url or self.base_url).rstrip("/")
        return f"{base}/mcp/datasources"

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

    def get_agent_tools(self) -> dict[str, AgentToolSpec]:
        """Parse the AGENT_TOOLS JSON object into a typed registry."""
        if not self.agent_tools_json:
            return {}
        raw = json.loads(self.agent_tools_json)
        if not isinstance(raw, dict):
            raise ValueError("AGENT_TOOLS must be a JSON object of {tool_name: spec}")
        return {name: AgentToolSpec.model_validate(spec) for name, spec in raw.items()}

    def get_config_value(self, key: str) -> str | None:
        """Read one config key from os.environ, falling back to the .env file.

        Used to resolve a tool's ``from_config`` references.  Unlike
        ``get_forwardable_config`` this does not filter by credential suffix —
        a tool needs its endpoint and identity variables too, and the operator
        named the key explicitly.
        """
        val = os.environ.get(key)
        if val:
            return val
        from dotenv import dotenv_values
        env_file = self.model_config.get("env_file", ".env")
        if not env_file:
            return None
        try:
            return (dotenv_values(env_file) or {}).get(key) or None
        except Exception:
            return None

    def resolve_tool_env(self, spec: AgentToolSpec) -> dict[str, str]:
        """Resolve a tool's declared env vars to concrete values.

        Unresolvable entries are dropped rather than sent empty, so the agent
        can tell "not configured" from "configured as empty".
        """
        resolved: dict[str, str] = {}
        for env_name, env_spec in spec.env.items():
            if env_spec.value is not None:
                resolved[env_name] = env_spec.value
                continue
            if env_spec.from_config:
                val = self.get_config_value(env_spec.from_config)
                if val:
                    resolved[env_name] = val
        return resolved

    def tool_env_keys(self) -> set[str]:
        """Every env var name claimed by some tool in the registry.

        These are stripped from the generic credential sweep so that a tool's
        secrets reach an agent only through that tool.
        """
        keys: set[str] = set()
        for spec in self.get_agent_tools().values():
            keys.update(spec.env)
            for env_spec in spec.env.values():
                if env_spec.from_config:
                    keys.add(env_spec.from_config)
        return keys

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

    # ── MCP server registry ────────────────────────────────────────────────

    def _builtin_mcp_integrations(self) -> list[McpIntegrationConfig]:
        """Servers this deployment always knows about, without being declared.

        Only the in-process data sources bridge qualifies: it is not an external
        integration an operator could point at, it is this process, and it is
        how every configured data source reaches an agent.  A ``MCP_INTEGRATIONS``
        entry of the same name overrides the defaults set here.
        """
        return [
            McpIntegrationConfig(
                name=DATASOURCES_MCP_NAME,
                enabled=self.mcp_datasources_enabled,
                transport="streamable_http",
                url_from="self_datasources",
                # The mount is gated by MCP_DATASOURCES_API_KEY when there is
                # one, and open to in-cluster callers when there is not — so the
                # declared auth follows the key rather than assuming either way.
                auth="bearer" if (self.mcp_datasources_api_key or "").strip() else "none",
                api_key=self.mcp_datasources_api_key,
                eager_start=False,
            ),
        ]

    def _declared_mcp_integrations(self) -> list[McpIntegrationConfig]:
        """Parse the MCP_INTEGRATIONS JSON array into a typed list."""
        if not self.mcp_integrations_json:
            return []
        raw = json.loads(self.mcp_integrations_json)
        if not isinstance(raw, list):
            raise ValueError("MCP_INTEGRATIONS must be a JSON array of server objects")
        return [McpIntegrationConfig.model_validate(item) for item in raw]

    def all_mcp_integrations(self) -> list[McpIntegrationConfig]:
        """Every known MCP server, declared or built in, regardless of state.

        Declared entries win over built-in ones of the same name, so a
        deployment can retarget the data sources bridge without losing it.
        Placeholders (``url_from``, ``env_from_config``) are resolved here so
        every caller sees concrete values.
        """
        merged: dict[str, McpIntegrationConfig] = {
            intg.name: intg for intg in self._builtin_mcp_integrations()
        }
        for intg in self._declared_mcp_integrations():
            merged[intg.name] = intg
        return [self._resolve_mcp_integration(intg) for intg in merged.values()]

    def _resolve_mcp_integration(self, intg: McpIntegrationConfig) -> McpIntegrationConfig:
        """Fill in the fields that can only be known at read time."""
        updates: dict[str, Any] = {}
        if intg.url_from == "self_datasources":
            updates["url"] = self.resolved_mcp_datasources_url()
            if not intg.agent_url:
                updates["agent_url"] = self.resolved_agent_mcp_datasources_url()
        if intg.env_from_config:
            env = dict(intg.env)
            for env_name, config_key in intg.env_from_config.items():
                value = self.get_config_value(config_key)
                if value:
                    env[env_name] = value
            updates["env"] = env
        return intg.model_copy(update=updates) if updates else intg

    def get_mcp_integrations(self) -> list[McpIntegrationConfig]:
        """Every MCP server that is enabled and reachable as configured."""
        return [
            intg for intg in self.all_mcp_integrations()
            if intg.enabled and (intg.url or intg.transport == "stdio")
        ]

    def legacy_mcp_env_servers(self) -> list[str]:
        """Server names still configured the pre-registry way, if any.

        ``MCP_<NAME>_ENABLED=true`` used to declare a server on its own.  Those
        vars are inert now (``extra="ignore"``), which would silently drop a
        capability on upgrade, so startup logs whichever ones are still set and
        no longer backed by an ``MCP_INTEGRATIONS`` entry.
        """
        declared = {intg.name.upper().replace("-", "_") for intg in self.all_mcp_integrations()}
        reserved = {"DATASOURCES", "INTEGRATIONS"}
        legacy: list[str] = []
        for key, value in os.environ.items():
            match = re.fullmatch(r"MCP_(.+)_ENABLED", key)
            if not match or value.strip().lower() not in {"1", "true", "yes", "on"}:
                continue
            server = match.group(1)
            if server in reserved or server in declared:
                continue
            legacy.append(server.lower())
        return sorted(legacy)

    def mcp_server_enabled(self, name: str) -> bool:
        """True when *name* is declared and enabled (reachable or not)."""
        return any(intg.name == name and intg.enabled for intg in self.all_mcp_integrations())

    def list_mcp_candidates(self) -> list[dict[str, Any]]:
        """Return every grantable MCP server for the UI's mcp addon picker.

        ``configured`` is False when a server is declared but cannot actually be
        dialled: no URL, no stdio command, or ``auth: bearer`` with no token
        resolving behind it.  An HTTP endpoint that needs no credential has to
        say so with ``auth: none`` — otherwise a server whose token was never
        wired up would be reported as ready and fail only once an agent used it.

        The in-process ``datasources`` bridge is deliberately absent.  Ticking
        it here used to hand an agent the static ``MCP_DATASOURCES_API_KEY`` and
        with it *every* operation of *every* registered data source — writes and
        deletes included, executed under this backend's own identity.  Data
        source access is granted per source and per operation by the
        ``datasource`` addon instead, so offering the bridge as a single
        all-or-nothing checkbox would just be a way around that allow-list.
        """
        candidates: list[dict[str, Any]] = []
        for intg in self.all_mcp_integrations():
            if intg.name == DATASOURCES_MCP_NAME:
                continue
            if intg.transport == "stdio":
                configured = bool(intg.command)
            else:
                configured = bool(intg.url) and (
                    intg.auth == "none" or bool(intg.resolved_api_key())
                )
            candidates.append({
                "name": intg.name,
                "enabled": intg.enabled,
                "transport": intg.transport,
                "configured": configured,
            })
        return candidates


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
