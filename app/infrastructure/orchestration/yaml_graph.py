from __future__ import annotations

import asyncio
import logging
import string
from typing import TYPE_CHECKING, Any, TypedDict
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, create_model

from app.domain.models.graph_run import GraphRun
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
# Shared graph streaming helper (used by workflow steps and default_workflow)
# ---------------------------------------------------------------------------

async def stream_graph_to_pause(
    runner: YamlGraphRunner,
    run: GraphRun,
    run_repository: Any,
    input_value: Any,
) -> None:
    """
    Stream *runner* from *input_value* until it reaches an interrupt or END,
    updating step_statuses and run status in *run_repository* after each node.

    Callers should initialise ``run.step_statuses`` before calling this.
    """
    config = {"configurable": {"thread_id": run.id}}
    current_state: dict = dict(input_value) if isinstance(input_value, dict) else {}
    try:
        async for chunk in runner.graph.astream(input_value, config, stream_mode="updates"):
            for node_name, output in chunk.items():
                if node_name in ("__start__", "__end__"):
                    continue
                status = "skipped" if output == {} else "finished"
                run.step_inputs[node_name] = dict(current_state)
                run.step_statuses[node_name] = status
                run.current_step = node_name
                if output:
                    run.step_outputs[node_name] = output
                    if isinstance(output, dict):
                        current_state.update(output)
                logger.info("run %s: step '%s' → %s", run.id, node_name, status)
                run.touch()
                await run_repository.update(run)
    except Exception as exc:
        logger.exception("run %s: graph execution failed", run.id)
        for sid in run.step_statuses:
            if run.step_statuses.get(sid) == "pending":
                run.step_inputs[sid] = dict(current_state)
                run.step_statuses[sid] = "failed"
                break
        run.status = "failed"
        run.state = {"error": str(exc)}
        run.current_step = None
        run.touch()
        await run_repository.update(run)
        return

    snap = runner.graph.get_state(config)
    run.status = "waiting_approval" if snap.next else "completed"
    run.current_step = snap.next[0] if snap.next else None
    run.state = snap.values
    run.touch()
    await run_repository.update(run)


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
            type: llm_structured | llm | mcp | human_approval | execute | workflow
            when: <state-key>          # skip node if state[key] is falsy
            system_prompt: "..."       # llm / llm_structured
            user_template: "..."       # {key} placeholders resolved from state
            output_key: <key>          # where to store the result
            output:                    # llm_structured only
              - name: needs_jira
                type: bool
                description: "..."
            tool: <tool-name>          # mcp only
            tool_input:                # mcp only – dict of {key}-templated values
              query: "{request}"
            repo_template: "{repo}"    # execute only
            instructions_template: "{plan}"  # execute only
            workflow_id: <id>          # workflow only — child workflow to spawn
            input_template: "{request}"  # workflow only — request passed to child

    Steps are chained sequentially.  ``human_approval`` calls interrupt() and
    expects the caller to resume with {"approved": bool, "reason": str|None}.

    ``workflow`` steps fire-and-forget spawn a child workflow run and store
    {"child_run_id": ..., "workflow_id": ..., "status": "started"} in output_key.
    Registry and run_repository must be injected after construction (done by
    load_yaml_graphs).
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
        # Injected post-construction by load_yaml_graphs
        self._registry: Any = None
        self._run_repository: Any = None
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
        if t == "workflow":
            return self._workflow_node(step)
        raise ValueError(f"Unknown step type '{t}' in graph '{self.id}'")

    _SUBMIT_TOOL = "submit_output"
    _MAX_ITERATIONS = 10

    def _llm_structured_node(self, step: dict[str, Any]):
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            if not self._when(step, state):
                logger.info("[%s] step '%s' skipped (condition not met)", graph_id, step_id)
                return {}
            logger.info("[%s] step '%s' running (llm_structured)", graph_id, step_id)

            output_model = self._build_output_model(step["output"])
            submit_tool = StructuredTool(
                name=self._SUBMIT_TOOL,
                description=(
                    "Call this when you have gathered all necessary information "
                    "and are ready to return the final structured result."
                ),
                args_schema=output_model,
                func=lambda **kwargs: kwargs,  # never actually invoked
            )

            mcp_tools = self._mcp.get_tools()
            llm = self._llm.bind_tools(mcp_tools + [submit_tool])

            messages: list = [
                SystemMessage(content=step.get("system_prompt", "")),
                HumanMessage(content=self._render(step.get("user_template", "{request}"), state)),
            ]
            logger.debug("[%s] step '%s' LLM input: %s", graph_id, step_id, [m.content for m in messages])

            for iteration in range(1, self._MAX_ITERATIONS + 1):
                response = await llm.ainvoke(messages)
                messages.append(response)
                tool_calls = response.tool_calls or []
                logger.debug(
                    "[%s] step '%s' iteration %d tool_calls: %s",
                    graph_id, step_id, iteration, [tc["name"] for tc in tool_calls],
                )

                if not tool_calls:
                    logger.warning(
                        "[%s] step '%s' iteration %d: LLM returned no tool calls, nudging to call %s",
                        graph_id, step_id, iteration, self._SUBMIT_TOOL,
                    )
                    messages.append(HumanMessage(
                        content=f"Please call `{self._SUBMIT_TOOL}` to submit your final answer."
                    ))
                    continue

                # Check for submit_output before executing side-effect tools
                for tc in tool_calls:
                    if tc["name"] == self._SUBMIT_TOOL:
                        output = tc["args"]
                        logger.debug("[%s] step '%s' LLM output: %s", graph_id, step_id, output)
                        logger.info("[%s] step '%s' finished: %s", graph_id, step_id, list(output.keys()))
                        return output

                # Execute MCP tool calls and feed results back
                for tc in tool_calls:
                    tool_name = tc["name"]
                    server = self._mcp.get_tool_server(tool_name)
                    server_tag = f" (server: {server})" if server else ""
                    tool = self._mcp.get_tool(tool_name)
                    if tool:
                        try:
                            result = await tool.ainvoke(tc["args"])
                            content = str(result)
                        except Exception as exc:
                            logger.exception(
                                "[%s] step '%s' tool '%s'%s failed",
                                graph_id, step_id, tool_name, server_tag,
                            )
                            content = f"Error calling '{tool_name}': {exc}"
                    else:
                        logger.warning(
                            "[%s] step '%s' unknown tool requested: '%s'",
                            graph_id, step_id, tool_name,
                        )
                        content = f"Tool '{tool_name}' is not available"
                    logger.debug("[%s] step '%s' tool '%s'%s result: %s", graph_id, step_id, tool_name, server_tag, content)
                    messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

            raise ValueError(
                f"[{graph_id}] step '{step_id}': reached {self._MAX_ITERATIONS} iterations without structured output"
            )

        return node

    def _llm_node(self, step: dict[str, Any]):
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            if not self._when(step, state):
                logger.info("[%s] step '%s' skipped (condition not met)", graph_id, step_id)
                return {}
            logger.info("[%s] step '%s' running (llm)", graph_id, step_id)
            messages = [
                SystemMessage(content=step.get("system_prompt", "")),
                HumanMessage(content=self._render(step.get("user_template", "{request}"), state)),
            ]
            logger.debug("[%s] step '%s' LLM input: %s", graph_id, step_id, [m.content for m in messages])
            response = await self._llm.ainvoke(messages)
            logger.debug("[%s] step '%s' LLM output: %s", graph_id, step_id, response.content)
            logger.info("[%s] step '%s' finished", graph_id, step_id)
            return {step["output_key"]: response.content}
        return node

    def _mcp_node(self, step: dict[str, Any]):
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            if not self._when(step, state):
                logger.info("[%s] step '%s' skipped (condition not met)", graph_id, step_id)
                return {}
            tool_name = step["tool"]
            server = self._mcp.get_tool_server(tool_name)
            server_tag = f" (server: {server})" if server else ""
            logger.info("[%s] step '%s' running (mcp tool='%s'%s)", graph_id, step_id, tool_name, server_tag)
            tool = self._mcp.get_tool(tool_name)
            if not tool:
                logger.warning("[%s] step '%s' MCP tool '%s' not available", graph_id, step_id, tool_name)
                return {step["output_key"]: f"MCP tool '{tool_name}' not available"}
            tool_input = {
                k: self._render(v, state)
                for k, v in step.get("tool_input", {}).items()
            }
            try:
                result = await tool.ainvoke(tool_input)
                logger.info("[%s] step '%s' finished", graph_id, step_id)
                return {step["output_key"]: str(result)}
            except Exception as exc:
                logger.exception("[%s] step '%s' MCP tool '%s'%s failed", graph_id, step_id, tool_name, server_tag)
                return {step["output_key"]: f"Error calling '{tool_name}': {exc}"}
        return node

    def _approval_node(self, step: dict[str, Any]):
        graph_id = self.id

        def node(state: dict) -> dict:
            step_id = step["id"]
            logger.info("[%s] step '%s' waiting for approval", graph_id, step_id)
            payload = {
                k: self._render(v, state)
                for k, v in step.get("interrupt_payload", {"plan": "{plan}"}).items()
            }
            decision: dict = interrupt(payload)
            approved = decision.get("approved", False)
            logger.info("[%s] step '%s' decision: approved=%s", graph_id, step_id, approved)
            return {
                "approved": approved,
                "reject_reason": decision.get("reason"),
            }
        return node

    def _execute_node(self, step: dict[str, Any]):
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            if not self._when(step, state):
                logger.info("[%s] step '%s' skipped (condition not met)", graph_id, step_id)
                return {}
            if self._openhands is None:
                logger.warning("[%s] step '%s' OpenHands not configured", graph_id, step_id)
                return {step["output_key"]: "OpenHands not configured"}
            repo = self._render(step.get("repo_template", "{repo}"), state)
            instructions = self._render(step.get("instructions_template", "{plan}"), state)
            logger.info("[%s] step '%s' running (execute repo='%s')", graph_id, step_id, repo)
            try:
                result = await self._openhands.execute(repo=repo, instructions=instructions)
                logger.info("[%s] step '%s' finished", graph_id, step_id)
                return {step["output_key"]: result}
            except Exception as exc:
                logger.exception("[%s] step '%s' execute failed", graph_id, step_id)
                return {step["output_key"]: {"error": str(exc)}}
        return node

    def _workflow_node(self, step: dict[str, Any]):
        """
        Spawns a child workflow run asynchronously (fire-and-forget).

        The child run is persisted to MongoDB immediately; the parent continues
        to the next step without waiting.  The child's run_id is stored in
        state under ``output_key`` so downstream steps can reference it.
        """
        graph_id = self.id
        step_id = step["id"]
        output_key = step.get("output_key", f"{step_id}_result")

        async def node(state: dict) -> dict:
            if not self._when(step, state):
                logger.info("[%s] step '%s' skipped (condition not met)", graph_id, step_id)
                return {}

            if self._registry is None or self._run_repository is None:
                logger.error(
                    "[%s] step '%s': registry/run_repository not injected — "
                    "ensure load_yaml_graphs is called with run_repository",
                    graph_id, step_id,
                )
                return {output_key: {"error": "workflow step not configured"}}

            child_workflow_id = step["workflow_id"]
            child_runner: YamlGraphRunner | None = self._registry.get(child_workflow_id)
            if child_runner is None:
                logger.error(
                    "[%s] step '%s': child workflow '%s' not found",
                    graph_id, step_id, child_workflow_id,
                )
                return {output_key: {"error": f"workflow '{child_workflow_id}' not found"}}

            child_request = self._render(step.get("input_template", "{request}"), state)
            child_run_id = str(uuid4())
            child_run = GraphRun(
                id=child_run_id,
                graph_id=child_workflow_id,
                user_request=child_request,
                status="running",
                step_statuses={s["id"]: "pending" for s in child_runner.steps},
            )
            await self._run_repository.create(child_run)

            # Fire-and-forget: child runs independently in the background
            asyncio.create_task(
                stream_graph_to_pause(child_runner, child_run, self._run_repository, {"request": child_request})
            )

            logger.info(
                "[%s] step '%s' spawned child workflow '%s' as run %s",
                graph_id, step_id, child_workflow_id, child_run_id,
            )
            return {output_key: {"child_run_id": child_run_id, "workflow_id": child_workflow_id, "status": "started"}}

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
        """Render a {key} template against state; missing keys render as empty string."""
        class _DefaultDict(dict):
            def __missing__(self, key: str) -> str:
                return ""

        try:
            return string.Formatter().vformat(template, [], _DefaultDict(state))  # type: ignore[arg-type]
        except ValueError:
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
