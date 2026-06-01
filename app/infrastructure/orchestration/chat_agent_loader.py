"""Load the bundled chat agent YAML config.

The bundled workflow_assistant.yaml is always loaded at startup regardless of
the workflow backend type (mongodb or localfiles).  A custom path can be
supplied via the CHAT_AGENT_CONFIG env var / Settings.chat_agent_config_path.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

BUNDLED_AGENT_PATH = Path(__file__).parent.parent.parent / "agents" / "workflow_assistant.yaml"


def load_chat_agent_config(path: str | Path | None = None) -> dict:
    """Return the chat agent configuration dict.

    Falls back to the bundled workflow_assistant.yaml when *path* is None or
    the specified path does not exist.
    """
    config_path = Path(path) if path else BUNDLED_AGENT_PATH
    if not config_path.exists():
        if path:
            logger.warning(
                "Chat agent config not found at '%s', falling back to bundled default", path
            )
        config_path = BUNDLED_AGENT_PATH
    try:
        with config_path.open() as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded chat agent config '%s' from %s", data.get("id"), config_path)
        return data
    except Exception:
        logger.exception("Failed to load chat agent config from %s", config_path)
        return {}
