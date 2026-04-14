from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.tools.mcp_client import McpToolsProvider

logger = logging.getLogger(__name__)


_STEP_TYPE_MAP: dict[str, str] = {
    "llm_structured": "llm",
    "llm": "llm",
    "mcp": "fetch",
    "human_approval": "approval",
    "execute": "execute",
    "workflow": "workflow",
}


class YamlGraphRegistry:
    def __init__(self, runners: dict[str, YamlGraphRunner]) -> None:
        self._runners = runners

    def get(self, graph_id: str) -> YamlGraphRunner | None:
        return self._runners.get(graph_id)

    def list_ids(self) -> list[str]:
        return list(self._runners.keys())

    def list_definitions(self) -> list[dict]:
        result = []
        for runner in self._runners.values():
            result.append({
                "id": runner.id,
                "name": runner.name,
                "description": runner.description,
                "steps": [
                    {
                        "id": s["id"],
                        "type": _STEP_TYPE_MAP.get(s.get("type", "llm"), s.get("type", "llm")),
                        "name": s.get("name", s["id"]),
                        "description": s.get("description"),
                    }
                    for s in runner.steps
                ],
            })
        return result


def load_yaml_graphs(
    directory: str,
    llm: Any,
    mcp_tools_provider: McpToolsProvider,
    openhands: OpenHandsAdapter | None = None,
    run_repository: Any = None,
) -> YamlGraphRegistry:
    path = Path(directory)
    runners: dict[str, YamlGraphRunner] = {}

    if not path.exists():
        logger.info("Graph definitions directory '%s' not found — no YAML graphs loaded.", directory)
        return YamlGraphRegistry(runners)

    for yaml_file in sorted(path.glob("*.yaml")):
        try:
            definition = yaml.safe_load(yaml_file.read_text())
            runner = YamlGraphRunner(
                definition,
                llm=llm,
                mcp_tools_provider=mcp_tools_provider,
                openhands=openhands,
            )
            runners[runner.id] = runner
            logger.info("Loaded YAML graph '%s' from %s", runner.id, yaml_file.name)
        except Exception:
            logger.exception("Failed to load YAML graph from %s", yaml_file)

    registry = YamlGraphRegistry(runners)

    # Two-pass: inject registry and run_repository into every runner so that
    # workflow steps can look up child runners and persist child runs.
    for runner in runners.values():
        runner._registry = registry
        runner._run_repository = run_repository

    return registry
