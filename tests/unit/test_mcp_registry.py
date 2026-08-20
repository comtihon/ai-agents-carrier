"""Unit tests for the MCP_INTEGRATIONS-driven MCP server registry.

Semantics under test:
- MCP servers exist only because the operator declared them; nothing in the
  backend knows any server name (the in-process ``datasources`` bridge is the
  single exception, and it is this process rather than an integration).
- Bearer tokens are named (``api_key_env``), not inlined, so the JSON blob is
  safe to ship in a ConfigMap; the value is resolved at read time.
- ``eager_start: false`` keeps a server out of the backend's startup dial-out.
- ``list_mcp_candidates`` reports a declared-but-uncredentialed server as
  ``configured: false`` instead of hiding it.
"""
from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.infrastructure.tools.mcp_client import McpToolsProvider

REGISTRY = [
    {
        "name": "github",
        "transport": "streamable_http",
        "url": "https://api.githubcopilot.com/mcp/",
        "api_key_env": "GITHUB_MCP_API_KEY",
    },
    {
        "name": "hubspot",
        "transport": "streamable_http",
        "url": "https://mcp.hubspot.com/anthropic",
        "api_key_env": "HUBSPOT_MCP_API_KEY",
    },
    {
        "name": "tracker",
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-tracker"],
        "env_from_config": {"TRACKER_URL": "TRACKER_BASE_URL"},
    },
    {
        "name": "agent-side",
        "transport": "stdio",
        "command": "agent-side-cli",
        "prestart_http": False,
        "eager_start": False,
    },
    {
        "name": "retired",
        "enabled": False,
        "transport": "sse",
        "url": "https://retired.example/sse",
    },
]


def _settings(registry: list | None = REGISTRY, **kwargs) -> Settings:
    return Settings(
        MCP_INTEGRATIONS=json.dumps(registry) if registry is not None else "",
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _registry_env(monkeypatch):
    monkeypatch.setenv("GITHUB_MCP_API_KEY", "github-secret")
    monkeypatch.setenv("TRACKER_BASE_URL", "https://tracker.example")
    # HUBSPOT_MCP_API_KEY deliberately unset — the "declared but incomplete" case.
    monkeypatch.delenv("HUBSPOT_MCP_API_KEY", raising=False)


def _by_name(integrations, name):
    return next((i for i in integrations if i.name == name), None)


# ── Declaration ──────────────────────────────────────────────────────────────

def test_empty_registry_still_offers_the_builtin_datasources_bridge():
    names = [i.name for i in _settings(registry=None).all_mcp_integrations()]
    assert names == ["datasources"]


def test_declared_servers_are_grantable_without_any_code_change():
    names = sorted(i.name for i in _settings().get_mcp_integrations())
    assert names == ["agent-side", "datasources", "github", "hubspot", "tracker"]


def test_disabled_entry_is_known_but_not_grantable():
    settings = _settings()
    assert _by_name(settings.all_mcp_integrations(), "retired") is not None
    assert _by_name(settings.get_mcp_integrations(), "retired") is None
    assert settings.mcp_server_enabled("retired") is False
    assert settings.mcp_server_enabled("github") is True


def test_entry_with_neither_url_nor_command_is_not_grantable():
    settings = _settings([{"name": "half-configured", "transport": "streamable_http"}])
    assert _by_name(settings.get_mcp_integrations(), "half-configured") is None


def test_malformed_registry_is_reported_not_silently_ignored():
    with pytest.raises(ValueError, match="MCP_INTEGRATIONS must be a JSON array"):
        Settings(MCP_INTEGRATIONS='{"name": "github"}').get_mcp_integrations()


# ── Secret handling ──────────────────────────────────────────────────────────

def test_named_bearer_token_resolves_from_the_environment():
    github = _by_name(_settings().get_mcp_integrations(), "github")
    assert github.api_key is None          # never inlined in the declaration
    assert github.resolved_api_key() == "github-secret"


def test_inline_bearer_token_still_works_for_local_dev():
    settings = _settings([
        {"name": "local", "transport": "streamable_http",
         "url": "http://localhost:9000/mcp", "api_key": "inline-token"},
    ])
    assert _by_name(settings.get_mcp_integrations(), "local").resolved_api_key() == "inline-token"


def test_missing_named_token_resolves_to_none_rather_than_empty_string():
    hubspot = _by_name(_settings().get_mcp_integrations(), "hubspot")
    assert hubspot.resolved_api_key() is None


def test_stdio_env_is_filled_from_backend_config():
    tracker = _by_name(_settings().get_mcp_integrations(), "tracker")
    assert tracker.env == {"TRACKER_URL": "https://tracker.example"}


def test_unresolvable_env_from_config_entry_is_dropped(monkeypatch):
    monkeypatch.delenv("TRACKER_BASE_URL", raising=False)
    tracker = _by_name(_settings().get_mcp_integrations(), "tracker")
    assert tracker.env == {}


# ── Built-in datasources bridge ──────────────────────────────────────────────

def test_datasources_url_points_at_this_process():
    bridge = _by_name(_settings().get_mcp_integrations(), "datasources")
    assert bridge.url.endswith("/mcp/datasources")
    assert bridge.eager_start is False


def test_datasources_bridge_honours_its_disable_switch():
    settings = _settings(MCP_DATASOURCES_ENABLED=False)
    assert _by_name(settings.get_mcp_integrations(), "datasources") is None
    assert settings.mcp_server_enabled("datasources") is False


def test_declared_entry_overrides_the_builtin_of_the_same_name():
    settings = _settings([
        {"name": "datasources", "transport": "streamable_http",
         "url": "http://sidecar:9100/mcp", "eager_start": True},
    ])
    bridge = _by_name(settings.get_mcp_integrations(), "datasources")
    assert bridge.url == "http://sidecar:9100/mcp"
    assert bridge.eager_start is True


# ── Startup dial-out ─────────────────────────────────────────────────────────

def test_backend_only_dials_eager_servers_at_startup():
    configs = McpToolsProvider(_settings())._build_server_configs()
    assert sorted(configs) == ["github", "hubspot", "tracker"]


def test_http_server_config_carries_the_resolved_bearer_header():
    cfg = McpToolsProvider(_settings())._config_for("github")
    assert cfg["url"] == "https://api.githubcopilot.com/mcp/"
    assert cfg["headers"] == {"Authorization": "Bearer github-secret"}


def test_uncredentialed_http_server_sends_no_authorization_header():
    assert "headers" not in McpToolsProvider(_settings())._config_for("hubspot")


def test_stdio_server_config_carries_command_args_and_env():
    cfg = McpToolsProvider(_settings())._config_for("tracker")
    assert cfg == {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-tracker"],
        "env": {"TRACKER_URL": "https://tracker.example"},
    }


def test_non_eager_server_is_still_refreshable_by_name():
    # refresh_server() resolves through the same path, so a server the backend
    # skips at startup must remain configurable after it.
    assert McpToolsProvider(_settings())._config_for("agent-side") is not None


# ── UI picker ────────────────────────────────────────────────────────────────

def test_candidates_flag_a_declared_server_with_no_token_as_unconfigured():
    by_name = {c["name"]: c for c in _settings().list_mcp_candidates()}
    assert by_name["github"] == {
        "name": "github", "enabled": True,
        "transport": "streamable_http", "configured": True,
    }
    assert by_name["hubspot"]["configured"] is False   # api_key_env unresolved
    assert by_name["retired"]["enabled"] is False      # offered, greyed out
    assert by_name["tracker"]["configured"] is True    # stdio: command is enough


def test_candidates_list_every_known_server_including_disabled_ones():
    names = sorted(c["name"] for c in _settings().list_mcp_candidates())
    assert names == ["agent-side", "datasources", "github", "hubspot", "retired", "tracker"]


# ── Split addressing (backend loopback vs agent-reachable) ───────────────────

def test_agents_get_the_callback_address_for_the_datasources_bridge():
    settings = _settings(
        BASE_URL="https://carrier.example",
        AGENT_CALLBACK_URL="http://carrier.langgraph.svc:8000",
    )
    bridge = _by_name(settings.get_mcp_integrations(), "datasources")
    # The backend dials its own mount over loopback…
    assert bridge.url == "http://127.0.0.1:8000/mcp/datasources"
    # …while a spawned agent, in its own container, gets a reachable address.
    assert bridge.resolved_agent_url() == "http://carrier.langgraph.svc:8000/mcp/datasources"


def test_datasources_agent_address_falls_back_to_base_url():
    settings = _settings(BASE_URL="https://carrier.example/")
    bridge = _by_name(settings.get_mcp_integrations(), "datasources")
    assert bridge.resolved_agent_url() == "https://carrier.example/mcp/datasources"


def test_declared_agent_url_overrides_the_backend_one():
    settings = _settings([
        {"name": "internal", "transport": "streamable_http",
         "url": "http://internal.svc/mcp", "agent_url": "http://agent-facing.svc/mcp"},
    ])
    server = _by_name(settings.get_mcp_integrations(), "internal")
    assert server.url == "http://internal.svc/mcp"
    assert server.resolved_agent_url() == "http://agent-facing.svc/mcp"


def test_server_without_agent_url_hands_agents_the_same_endpoint():
    github = _by_name(_settings().get_mcp_integrations(), "github")
    assert github.resolved_agent_url() == github.url


# ── End-to-end grant chain (helm registry → addon → /start payload) ──────────

def _agent_config(servers: dict, settings: Settings):
    from app.domain.models.agent_definition import AgentDefinition
    from app.steps.agent_executor import _build_agent_config

    addons = [{"type": "mcp", "servers": servers}] if servers is not None else []
    agent_def = AgentDefinition(id="a", name="A", default_runtime="k8s", addons=addons)
    return _build_agent_config(agent_def, settings)


def test_no_mcp_addon_sends_no_servers_at_all():
    assert _agent_config(None, _settings())["mcp_servers"] == []


def test_only_checked_servers_reach_the_agent_with_their_credentials():
    cfg = _agent_config(
        {"github": True, "hubspot": False, "tracker": True},
        _settings(BASE_URL="https://carrier.example"),
    )
    by_name = {s["name"]: s for s in cfg["mcp_servers"]}
    assert sorted(by_name) == ["github", "tracker"]
    # HTTP server: endpoint + resolved bearer token, never the env var name.
    assert by_name["github"]["url"] == "https://api.githubcopilot.com/mcp/"
    assert by_name["github"]["api_key"] == "github-secret"
    assert "api_key_env" not in by_name["github"]
    # stdio server: argv the agent runs, plus the env the backend resolved.
    assert by_name["tracker"]["command"] == ["uvx", "mcp-tracker"]
    assert by_name["tracker"]["env"] == {"TRACKER_URL": "https://tracker.example"}
    assert by_name["tracker"]["prestart_http"] is True


def test_agent_side_only_server_is_flagged_as_not_prestartable():
    cfg = _agent_config({"agent-side": True}, _settings())
    entry = next(s for s in cfg["mcp_servers"] if s["name"] == "agent-side")
    assert entry["command"] == ["agent-side-cli"]
    assert entry["prestart_http"] is False


def test_datasources_grant_carries_the_agent_reachable_url_and_its_key():
    settings = _settings(
        BASE_URL="https://carrier.example",
        MCP_DATASOURCES_API_KEY="ds-token",
    )
    entry = next(
        s for s in _agent_config({"datasources": True}, settings)["mcp_servers"]
        if s["name"] == "datasources"
    )
    assert entry["url"] == "https://carrier.example/mcp/datasources"
    assert entry["api_key"] == "ds-token"


def test_disabled_and_undeclared_servers_cannot_be_granted_by_an_addon():
    cfg = _agent_config({"retired": True, "never-declared": True}, _settings())
    assert cfg["mcp_servers"] == []


# ── Upgrade safety ───────────────────────────────────────────────────────────

def test_legacy_enabled_vars_are_reported_so_upgrades_are_not_silent(monkeypatch):
    monkeypatch.setenv("MCP_FIGMA_ENABLED", "true")
    monkeypatch.setenv("MCP_GITHUB_ENABLED", "true")     # declared → already ported
    monkeypatch.setenv("MCP_MIRO_ENABLED", "false")      # was off → nothing lost
    monkeypatch.setenv("MCP_DATASOURCES_ENABLED", "true")  # not an integration var
    assert _settings().legacy_mcp_env_servers() == ["figma"]


def test_no_legacy_vars_reports_nothing(monkeypatch):
    for key in ("MCP_FIGMA_ENABLED", "MCP_MIRO_ENABLED", "MCP_NOTION_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    assert _settings().legacy_mcp_env_servers() == []


# ── Command shadowing across the two registries ──────────────────────────────

def test_a_granted_stdio_server_keeps_its_launcher_off_the_block_list():
    # An ungranted tool names the same binary a granted stdio MCP server needs
    # to start. Stubbing it would revoke the grant that was made.
    settings = _settings(
        [{"name": "tracker", "transport": "stdio", "command": "uvx", "args": ["mcp-tracker"]}],
        AGENT_TOOLS=json.dumps({"scripts": {"command": "uvx"}}),
    )
    cfg = _agent_config({"tracker": True}, settings)
    assert cfg["blocked_commands"] == []
    assert next(s for s in cfg["mcp_servers"] if s["name"] == "tracker")["command"] == ["uvx", "mcp-tracker"]


def test_the_launcher_is_still_blocked_when_neither_entry_is_granted():
    settings = _settings(
        [{"name": "tracker", "transport": "stdio", "command": "uvx", "args": ["mcp-tracker"]}],
        AGENT_TOOLS=json.dumps({"scripts": {"command": "uvx"}}),
    )
    assert _agent_config({}, settings)["blocked_commands"] == ["uvx"]


def test_a_granted_tool_is_never_shadowed_by_another_ungranted_tool():
    settings = _settings(
        registry=None,
        AGENT_TOOLS=json.dumps({
            "reader": {"command": "shared-bin"},
            "writer": {"command": "shared-bin"},
        }),
    )
    cfg = _agent_config({}, settings)          # mcp addon empty; tools addon decides
    assert cfg["blocked_commands"] == ["shared-bin"]

    from app.domain.models.agent_definition import AgentDefinition
    from app.steps.agent_executor import _build_agent_config
    granted = _build_agent_config(
        AgentDefinition(id="a", name="A", default_runtime="k8s",
                        addons=[{"type": "tools", "tools": {"reader": True}}]),
        settings,
    )
    assert granted["blocked_commands"] == []
    assert [t["name"] for t in granted["tools"]] == ["reader"]


# ── Declared auth vs actually-wired auth ─────────────────────────────────────

def test_http_entry_with_no_credential_is_reported_incomplete_not_ready():
    # The failure this prevents: an OAuth-only endpoint pasted in as a URL, with
    # no token anywhere, showing up in the picker as though it worked.
    settings = _settings([
        {"name": "oauth-only", "transport": "streamable_http", "url": "https://oauth.example/mcp"},
    ])
    entry = next(c for c in settings.list_mcp_candidates() if c["name"] == "oauth-only")
    assert entry["configured"] is False


def test_an_endpoint_that_wants_no_credential_says_so_explicitly():
    settings = _settings([
        {"name": "open", "transport": "streamable_http",
         "url": "http://localhost:9000/mcp", "auth": "none"},
    ])
    entry = next(c for c in settings.list_mcp_candidates() if c["name"] == "open")
    assert entry["configured"] is True
    assert next(i for i in settings.get_mcp_integrations() if i.name == "open").resolved_api_key() is None


def test_datasources_bridge_is_ready_with_or_without_a_key():
    # Gated by MCP_DATASOURCES_API_KEY when set, open to in-cluster callers when
    # not — either way it is not "unconfigured".
    with_key = _settings(MCP_DATASOURCES_API_KEY="ds-token")
    without = _settings(MCP_DATASOURCES_API_KEY="")
    for settings in (with_key, without):
        entry = next(c for c in settings.list_mcp_candidates() if c["name"] == "datasources")
        assert entry["configured"] is True
    assert _by_name(with_key.all_mcp_integrations(), "datasources").auth == "bearer"
    assert _by_name(without.all_mcp_integrations(), "datasources").auth == "none"


def test_stdio_servers_ignore_the_auth_field_and_use_env_instead():
    # Their credentials ride in env / env_from_config, so a command is enough.
    entry = next(c for c in _settings().list_mcp_candidates() if c["name"] == "tracker")
    assert entry["configured"] is True
