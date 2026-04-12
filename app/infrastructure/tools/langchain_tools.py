from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from app.domain.interfaces.tools import ExternalTool


class StubExternalTool(ExternalTool):
    def __init__(self, name: str, response: dict[str, Any] | None = None) -> None:
        self.name = name
        self._response = response or {"status": "stubbed"}

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"tool": self.name, "payload": payload, **self._response}


class ExternalToolAdapter(BaseTool):
    name: str
    description: str
    _tool: ExternalTool = PrivateAttr()

    def __init__(self, tool: ExternalTool, description: str) -> None:
        super().__init__(name=tool.name, description=description)
        self._tool = tool

    def _run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Synchronous tool execution is not supported.")

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs.get("payload", {})
        return await self._tool.ainvoke(payload)


def build_default_tools() -> list[BaseTool]:
    return [
        ExternalToolAdapter(
            StubExternalTool("github", {"capabilities": ["repository_lookup", "pull_request_status"]}),
            "GitHub integration tool.",
        ),
        ExternalToolAdapter(
            StubExternalTool("jira", {"capabilities": ["issue_lookup"], "status": "stubbed"}),
            "Jira integration tool stub.",
        ),
        ExternalToolAdapter(
            StubExternalTool("figma", {"capabilities": ["file_lookup"], "status": "stubbed"}),
            "Figma integration tool stub.",
        ),
    ]
