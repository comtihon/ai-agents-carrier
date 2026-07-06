"""Unit tests for the ToolsAddon model and the AnyAgentAddon discriminated union."""
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.domain.models.agent_addon import AnyAgentAddon, ToolsAddon

_adapter: TypeAdapter[AnyAgentAddon] = TypeAdapter(AnyAgentAddon)


def test_tools_addon_validates_via_type_adapter():
    addon = _adapter.validate_python(
        {"type": "tools", "hidden": False, "tools": {"github": True, "jira": False}}
    )
    assert isinstance(addon, ToolsAddon)
    assert addon.type == "tools"
    assert addon.tools == {"github": True, "jira": False}


def test_unknown_addon_type_rejected():
    with pytest.raises(ValidationError):
        _adapter.validate_python({"type": "definitely-not-a-real-addon"})


def test_enabled_tools_filters_false():
    addon = ToolsAddon(tools={"github": True, "jira": False, "graphify": True})
    assert addon.enabled_tools() == {"github", "graphify"}
