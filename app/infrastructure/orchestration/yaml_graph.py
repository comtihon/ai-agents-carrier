from __future__ import annotations

import string
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, create_model

from app.infrastructure.tools.mcp_client import McpToolsProvider

if TYPE_CHECKING:
    from app.infrastructure.integrations.openhands import OpenHandsAdapter


def _build_state_schema(steps: list[dict[str, Any]]) -> type:
    """
    Dynamically build a TypedDict (total=False) that includes all output keys
    declared across graph steps plus standard fields.  LangGraph merges node
    return dicts into state key-by-key; any key not in the schema is dropped,
    so we must declare every key upfront.
    """
    fields: dict[str, type] = {
        "request": str,
        "approved": bool,
        "reject_reason": str,
    }
    for step in steps:
        # Regular output nodes store their result under output_key
        if "output_key" in step:
            fields[step["output_key"]] = Any  # type: ignore[assignment]
        # llm_structured stores each named output field directly in state
        if step.get("type") == "llm_structured":
            for out_field in step.get("output", []):
                fields[out_field["name"]] = Any  # type: ignore[assignment]
    return TypedDict("YamlGraphState", fields, total=False)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# YAML graph runner
# ---------------------------------------------------------------------------

class YamlGraphRunner:
    """
    Builds a compiled LangGraph from a plain dict parsed from a YAML file.

    YAML schema (all fields except ``id`` and ``steps`` are optional):

        id: dev-assistant
        description: "..."
        steps:
          - id: <node-id>
            type: llm_structured | llm | mcp | human_approval | execute
            when: <state-key>          # skip node if state[key] is falsy
            system_prompt: "..."       # llm / llm_structured
            user_template: "..."       # {key} placeholders resolved from state
            output_key: <key>          # where to store the LLM/MCP/execute result
            output:                    # llm_structured only
              - name: needs_jira
                type: bool
                description: "..."
            tool: <tool-name>          # mcp only
            tool_input:                # mcp only – dict of {key}-templated values
              query: "{request}"
            repo_template: "{repo}"    # execute only
            instructions_template: "{plan}"  # execute only

    Steps are chained sequentially.  ``human_approval`` calls interrupt() and
    expects the caller to resume with {"approved": bool, "reason": str|None}.
    """

    def __init__(
        self,
        definition: dict[str, Any],
        llm: BaseChatModel,
        mcp_tools_provider: McpToolsProvider,
        openhands: OpenHandsAdapter | None = None,
    ) -> None:
        self.id: str = definition["id"]
        # Human-readable name; fall back to title-casing the id
        self.name: str = definition.get(
            "name",
            self.id.replace("-", " ").replace("_", " ").title(),
        )
        self.description: str = definition.get("description", "")
        self._steps: list[dict[str, Any]] = definition["steps"]
        self._llm = llm
        self._mcp = mcp_tools_provider
        self._openhands = openhands
        self._state_schema = _build_state_schema(self._steps)
        self.graph = self._build()

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self._steps

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build(self):
        sg = StateGraph(self._state_schema)

        prev = START
        for step in self._steps:
            node_fn = self._make_node(step)
            sg.add_node(step["id"], node_fn)
            sg.add_edge(prev, step["id"])
            prev = step["id"]

        sg.add_edge(prev, END)
        return sg.compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------------
    # Node factories
    # ------------------------------------------------------------------

    def _make_node(self, step: dict[str, Any]):
        t = step["type"]
        if t == "llm_structured":
            return self._llm_structured_node(step)
        if t == "llm":
            return self._llm_node(step)
        if t == "mcp":
            return self._mcp_node(step)
        if t == "human_approval":
            return self._approval_node(step)
        if t == "execute":
            return self._execute_node(step)
        raise ValueError(f"Unknown step type '{t}' in graph '{self.id}'")

    def _llm_structured_node(self, step: dict[str, Any]):
        async def node(state: dict) -> dict:
            if not self._when(step, state):
                return {}
            output_model = self._build_output_model(step["output"])
            structured = self._llm.with_structured_output(output_model)
            result = await structured.ainvoke([
                SystemMessage(content=step.get("system_prompt", "")),
                HumanMessage(content=self._render(step.get("user_template", "{request}"), state)),
            ])
            return result.model_dump()
        return node

    def _llm_node(self, step: dict[str, Any]):
        async def node(state: dict) -> dict:
            if not self._when(step, state):
                return {}
            response = await self._llm.ainvoke([
                SystemMessage(content=step.get("system_prompt", "")),
                HumanMessage(content=self._render(step.get("user_template", "{request}"), state)),
            ])
            return {step["output_key"]: response.content}
        return node

    def _mcp_node(self, step: dict[str, Any]):
        async def node(state: dict) -> dict:
            if not self._when(step, state):
                return {}
            tool = self._mcp.get_tool(step["tool"])
            if not tool:
                return {step["output_key"]: f"MCP tool '{step['tool']}' not available"}
            tool_input = {
                k: self._render(v, state)
                for k, v in step.get("tool_input", {}).items()
            }
            try:
                result = await tool.ainvoke(tool_input)
                return {step["output_key"]: str(result)}
            except Exception as exc:
                return {step["output_key"]: f"Error calling '{step['tool']}': {exc}"}
        return node

    def _approval_node(self, step: dict[str, Any]):
        def node(state: dict) -> dict:
            payload = {
                k: self._render(v, state)
                for k, v in step.get("interrupt_payload", {"plan": "{plan}"}).items()
            }
            decision: dict = interrupt(payload)
            return {
                "approved": decision.get("approved", False),
                "reject_reason": decision.get("reason"),
            }
        return node

    def _execute_node(self, step: dict[str, Any]):
        async def node(state: dict) -> dict:
            if not self._when(step, state):
                return {}
            if self._openhands is None:
                return {step["output_key"]: "OpenHands not configured"}
            repo = self._render(step.get("repo_template", "{repo}"), state)
            instructions = self._render(step.get("instructions_template", "{plan}"), state)
            try:
                result = await self._openhands.execute(repo=repo, instructions=instructions)
                return {step["output_key"]: result}
            except Exception as exc:
                return {step["output_key"]: {"error": str(exc)}}
        return node

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _when(step: dict[str, Any], state: dict) -> bool:
        key = step.get("when")
        return bool(state.get(key, False)) if key else True  # type: ignore[arg-type]

    @staticmethod
    def _render(template: str, state: dict) -> str:
        """Render a {key} template against state, leaving unknown keys as-is."""
        try:
            return string.Formatter().vformat(template, [], state)  # type: ignore[arg-type]
        except (KeyError, ValueError):
            return template

    @staticmethod
    def _build_output_model(output_spec: list[dict[str, Any]]) -> type[BaseModel]:
        """Dynamically build a Pydantic model from the ``output`` spec list."""
        _type_map: dict[str, type] = {"bool": bool, "str": str, "int": int, "float": float}
        fields: dict[str, Any] = {
            o["name"]: (
                _type_map.get(o.get("type", "str"), str),
                Field(description=o.get("description", "")),
            )
            for o in output_spec
        }
        return create_model("StructuredOutput", **fields)
