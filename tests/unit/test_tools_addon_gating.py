"""Unit tests for tools-addon credential gating and semble MCP wiring in
``_build_agent_config`` / the backend MCP client.

Semantics under test:
- No tools addon  → tool_access all False + tool credential keys stripped.
- All-false addon → equivalent to absent.
- Per-tool enable → keep that tool's creds, strip the disabled tools' creds.
- Non-tool credentials (ANTHROPIC / HF) are never touched.
- semble is a stdio MCP candidate on the backend but is never launched inside
  the backend container (mcp_client skips it).
"""
from __future__ import annotations

import pytest

from app.core.config import Settings
from app.domain.models.agent_definition import AgentDefinition
from app.infrastructure.tools.mcp_client import McpToolsProvider
from app.steps.agent_executor import _build_agent_config


@pytest.fixture(autouse=True)
def _tool_env(monkeypatch):
    monkeypatch.setenv("MCP_GITHUB_API_KEY", "gh-secret")
    monkeypatch.setenv("MCP_JIRA_API_TOKEN", "jira-secret")
    monkeypatch.setenv("MCP_JIRA_JIRA_URL", "https://example.atlassian.net")
    monkeypatch.setenv("MCP_JIRA_USERNAME", "bot@example.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")


def _cfg(addons):
    agent_def = AgentDefinition(id="a", name="A", default_runtime="docker", addons=addons)
    return _build_agent_config(agent_def, Settings())


def test_no_tools_addon_strips_tool_creds_and_sends_all_false_tool_access():
    cfg = _cfg([])
    assert cfg["tool_access"] == {"github": False, "jira": False, "graphify": False}
    creds = cfg["credentials"]
    assert "MCP_GITHUB_API_KEY" not in creds
    assert "MCP_JIRA_API_TOKEN" not in creds


def test_all_false_tools_addon_equivalent_to_absent():
    cfg = _cfg([{"type": "tools", "tools": {"github": False, "jira": False, "graphify": False}}])
    assert cfg["tool_access"] == {"github": False, "jira": False, "graphify": False}
    assert "MCP_GITHUB_API_KEY" not in cfg["credentials"]
    assert "MCP_JIRA_API_TOKEN" not in cfg["credentials"]


def test_github_only_enabled_keeps_github_strips_jira():
    cfg = _cfg([{"type": "tools", "tools": {"github": True, "jira": False}}])
    assert cfg["tool_access"] == {"github": True, "jira": False, "graphify": False}
    assert cfg["credentials"].get("MCP_GITHUB_API_KEY") == "gh-secret"
    assert "MCP_JIRA_API_TOKEN" not in cfg["credentials"]


def test_jira_only_enabled_keeps_jira_strips_github():
    cfg = _cfg([{"type": "tools", "tools": {"jira": True, "github": False}}])
    assert cfg["tool_access"] == {"github": False, "jira": True, "graphify": False}
    assert cfg["credentials"].get("MCP_JIRA_API_TOKEN") == "jira-secret"
    assert "MCP_GITHUB_API_KEY" not in cfg["credentials"]


def test_jira_enabled_injects_full_bash_credential_set():
    cfg = _cfg([{"type": "tools", "tools": {"jira": True}}])
    creds = cfg["credentials"]
    assert creds.get("JIRA_URL") == "https://example.atlassian.net"
    assert creds.get("JIRA_USERNAME") == "bot@example.com"
    assert creds.get("JIRA_API_TOKEN") == "jira-secret"


def test_jira_disabled_strips_url_and_username():
    cfg = _cfg([{"type": "tools", "tools": {"jira": False, "github": True}}])
    creds = cfg["credentials"]
    assert "JIRA_URL" not in creds
    assert "JIRA_USERNAME" not in creds
    assert "JIRA_API_TOKEN" not in creds
    assert "MCP_JIRA_API_TOKEN" not in creds


def test_github_enabled_injects_github_token():
    cfg = _cfg([{"type": "tools", "tools": {"github": True}}])
    assert cfg["credentials"].get("GITHUB_TOKEN") == "gh-secret"


def test_github_disabled_strips_github_token():
    cfg = _cfg([{"type": "tools", "tools": {"jira": True}}])
    assert "GITHUB_TOKEN" not in cfg["credentials"]
    assert "MCP_GITHUB_API_KEY" not in cfg["credentials"]


def test_non_tool_credentials_always_forwarded():
    cfg = _cfg([])  # everything disabled — non-tool creds still forwarded
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
