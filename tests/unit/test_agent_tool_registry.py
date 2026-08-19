"""Unit tests for AGENT_TOOLS-driven tool grants in ``_build_agent_config``.

Semantics under test:
- Tools exist only because the operator declared them in AGENT_TOOLS; the
  agent's tools addon can only toggle declared names.
- A granted tool travels with its resolved env; a tool that was not granted
  contributes its command to ``blocked_commands`` instead.
- Env vars claimed by any registry tool never ride along in the generic
  credential sweep — they reach an agent only through their tool.
- Non-tool credentials (ANTHROPIC / HF) are untouched.
- semble is a stdio MCP candidate on the backend but is never launched inside
  the backend container (mcp_client skips it).
"""
from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.domain.models.agent_definition import AgentDefinition
from app.infrastructure.tools.mcp_client import McpToolsProvider
from app.steps.agent_executor import _build_agent_config

# A registry in the shape an operator would configure — nothing in the backend
# or the agent image knows these names.
REGISTRY = {
    "code-host": {
        "label": "Code host",
        "command": "git",
        "env": {
            "GIT_TOKEN_EXAMPLE_COM": {"from_config": "CODE_HOST_TOKEN"},
        },
        "bash_match": r"\bgit\b",
    },
    "tracker": {
        "env": {
            "TRACKER_URL": {"value": "https://tracker.example"},
            "TRACKER_USER": {"from_config": "TRACKER_USERNAME"},
            "TRACKER_TOKEN": {"from_config": "TRACKER_API_TOKEN"},
        },
    },
    "grapher": {
        "command": "grapher",
        "cli_tools": {
            "grapher_query": {
                "args": ["query", "{question}", "."],
                "required": ["question"],
                "cwd": "{repo}",
            },
        },
    },
}


@pytest.fixture(autouse=True)
def _tool_env(monkeypatch):
    monkeypatch.setenv("CODE_HOST_TOKEN", "code-host-secret")
    monkeypatch.setenv("TRACKER_API_TOKEN", "tracker-secret")
    monkeypatch.setenv("TRACKER_USERNAME", "bot@example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")


def _settings(registry: dict | None = REGISTRY) -> Settings:
    return Settings(AGENT_TOOLS=json.dumps(registry) if registry else "")


def _cfg(addons, registry: dict | None = REGISTRY):
    agent_def = AgentDefinition(id="a", name="A", default_runtime="docker", addons=addons)
    return _build_agent_config(agent_def, _settings(registry))


def _tool(cfg, name):
    return next((t for t in cfg["tools"] if t["name"] == name), None)


def test_no_tools_addon_grants_nothing_and_blocks_every_command():
    cfg = _cfg([])
    assert cfg["tools"] == []
    assert sorted(cfg["blocked_commands"]) == ["git", "grapher"]


def test_all_false_addon_equivalent_to_absent():
    cfg = _cfg([{"type": "tools", "tools": {"code-host": False, "tracker": False}}])
    assert cfg["tools"] == []


def test_granted_tool_carries_its_resolved_env():
    cfg = _cfg([{"type": "tools", "tools": {"tracker": True}}])
    tracker = _tool(cfg, "tracker")
    assert tracker["env"] == {
        "TRACKER_URL": "https://tracker.example",       # literal value
        "TRACKER_USER": "bot@example.com",              # from_config
        "TRACKER_TOKEN": "tracker-secret",              # from_config
    }


def test_tool_env_never_rides_along_in_the_credential_sweep():
    # Enabled for one agent…
    granted = _cfg([{"type": "tools", "tools": {"code-host": True}}])
    assert _tool(granted, "code-host")["env"]["GIT_TOKEN_EXAMPLE_COM"] == "code-host-secret"
    assert "CODE_HOST_TOKEN" not in granted["credentials"]
    # …and invisible to an agent that was not granted it.
    ungranted = _cfg([{"type": "tools", "tools": {"tracker": True}}])
    assert _tool(ungranted, "code-host") is None
    assert "CODE_HOST_TOKEN" not in ungranted["credentials"]
    assert "TRACKER_API_TOKEN" not in ungranted["credentials"]


def test_ungranted_tool_commands_are_blocked_granted_ones_are_not():
    cfg = _cfg([{"type": "tools", "tools": {"grapher": True}}])
    assert cfg["blocked_commands"] == ["git"]
    assert _tool(cfg, "grapher")["command"] == "grapher"


def test_cli_tools_are_forwarded_verbatim():
    cfg = _cfg([{"type": "tools", "tools": {"grapher": True}}])
    cli = _tool(cfg, "grapher")["cli_tools"]["grapher_query"]
    assert cli["args"] == ["query", "{question}", "."]
    assert cli["required"] == ["question"]
    assert cli["cwd"] == "{repo}"


def test_bash_match_is_forwarded():
    cfg = _cfg([{"type": "tools", "tools": {"code-host": True}}])
    assert _tool(cfg, "code-host")["bash_match"] == r"\bgit\b"


def test_tool_not_in_registry_is_ignored():
    cfg = _cfg([{"type": "tools", "tools": {"code-host": True, "not-declared": True}}])
    assert [t["name"] for t in cfg["tools"]] == ["code-host"]


def test_empty_registry_grants_nothing():
    cfg = _cfg([{"type": "tools", "tools": {"code-host": True}}], registry=None)
    assert cfg["tools"] == []
    assert cfg["blocked_commands"] == []


def test_unresolvable_env_entry_is_dropped(monkeypatch):
    monkeypatch.delenv("TRACKER_USERNAME", raising=False)
    cfg = _cfg([{"type": "tools", "tools": {"tracker": True}}])
    env = _tool(cfg, "tracker")["env"]
    assert "TRACKER_USER" not in env
    assert env["TRACKER_TOKEN"] == "tracker-secret"


def test_non_tool_credentials_always_forwarded():
    cfg = _cfg([])  # no tools at all — unrelated creds still go out
    assert cfg["credentials"].get("ANTHROPIC_API_KEY") == "anthropic-secret"
    assert cfg["credentials"].get("HF_TOKEN") == "hf-secret"


def test_mcp_addon_with_semble_included_in_mcp_servers():
    cfg = _cfg([{"type": "mcp", "servers": {"semble": True}}])
    semble = next((s for s in cfg["mcp_servers"] if s["name"] == "semble"), None)
    assert semble is not None
    assert semble["transport"] == "stdio"
    assert semble["command"] == ["semble"]


def test_mcp_addon_without_semble_excluded():
    cfg = _cfg([{"type": "mcp", "servers": {"jira": True}}])
    assert not any(s["name"] == "semble" for s in cfg["mcp_servers"])


def test_backend_mcp_client_skips_semble():
    provider = McpToolsProvider(Settings())
    configs = provider._build_server_configs()
    assert "semble" not in configs
