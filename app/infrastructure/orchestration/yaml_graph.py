from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import string
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, TypedDict
from uuid import uuid4

import httpx

from langchain_core.language_models import BaseChatModel
from app.infrastructure.notifications.webhook_notifier import send_approval_notification

logger = logging.getLogger(__name__)
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, create_model

from app.domain.models.graph_run import GraphRun
from app.infrastructure.tools.mcp_client import McpToolsProvider

if TYPE_CHECKING:
    from app.infrastructure.integrations.openhands import OpenHandsAdapter


def _merge_dicts(a: Any, b: Any) -> Any:
    """Reducer for dict-typed state fields updated by concurrent parallel branches."""
    if isinstance(a, dict) and isinstance(b, dict):
        return {**a, **b}
    return b if b is not None else a


def _last_wins(a: Any, b: Any) -> Any:
    """Reducer that keeps the last non-None write; safe for scalar fields."""
    return b if b is not None else a


def _build_state_schema(steps: list[dict[str, Any]]) -> type:
    """
    Dynamically build a TypedDict (total=False) that includes all output keys
    declared across graph steps plus standard fields.  LangGraph merges node
    return dicts into state key-by-key; any key not in the schema is dropped,
    so we must declare every key upfront.

    Fields that can be updated by multiple concurrent parallel branches must use
    ``Annotated[type, reducer]`` so LangGraph knows how to merge the updates.
    """
    fields: dict[str, type] = {
        "request": str,
        "approved": bool,
        "reject_reason": str,
        # Dict fields updated by every node (loop guard, conversation tracking) —
        # multiple parallel branches may write simultaneously, so use _merge_dicts.
        "_conv_map":                  Annotated[Any, _merge_dicts],  # type: ignore[assignment]
        "_visit_counts":              Annotated[Any, _merge_dicts],  # type: ignore[assignment]
        "_slack_thread_ts":           Annotated[Any, _last_wins],    # type: ignore[assignment]
        "_slack_channel":             Annotated[Any, _last_wins],    # type: ignore[assignment]
        "_slack_approver_id":         Annotated[Any, _last_wins],    # type: ignore[assignment]
        "_slack_ask_context_ts":      Annotated[Any, _last_wins],    # type: ignore[assignment]
        "_slack_ask_context_channel": Annotated[Any, _last_wins],    # type: ignore[assignment]
        # ID of the most recent step that caught an internal exception and chose
        # to record the failure in state instead of raising. Read by the chunk
        # handlers to mark step_statuses["that_step"] = "failed" rather than
        # the default "finished" inferred from a non-empty output dict.
        "__failed_step__":            Annotated[Any, _last_wins],    # type: ignore[assignment]
        "_live_token_usage":          Annotated[Any, _last_wins],    # type: ignore[assignment]
    }
    for step in steps:
        # Regular output nodes store their result under output_key
        if "output_key" in step:
            fields[step["output_key"]] = Any  # type: ignore[assignment]
        # llm_structured stores each named output field directly in state
        if step.get("type") == "llm_structured":
            for out_field in step.get("output", []):
                fields[out_field["name"]] = Any  # type: ignore[assignment]
        # http trigger carries the raw webhook body; cron trigger carries schedule metadata
        if step.get("type") == "http":
            fields["trigger_payload"] = Any  # type: ignore[assignment]
        if step.get("type") == "cron":
            fields["trigger_info"] = Any  # type: ignore[assignment]
        # human_approval with a custom output_key writes the bool result there
        if step.get("type") == "human_approval" and "output_key" in step:
            fields[step["output_key"]] = Any  # type: ignore[assignment]
        # mcp steps always store their tool output text for display in the UI
        if step.get("type") == "mcp":
            fields[f"_mcp_output_{step['id']}"] = Any  # type: ignore[assignment]
        # execute steps persist their OpenHands conversation ID for restart resumption
        if step.get("type") == "execute":
            fields[f"_openhands_conv_{step['id']}"] = Any  # type: ignore[assignment]
        # agent steps (langgraph-agent / claude-agent) store their output under output_key
        # when output_mapping is absent; the output_key field is already captured by the
        # generic "output_key" check above, but we also ensure it exists for typing.
        if step.get("type") in ("langgraph-agent", "claude-agent"):
            if "output_key" not in step and step.get("output_mapping"):
                for wf_key in step["output_mapping"].values():
                    fields[wf_key] = Any  # type: ignore[assignment]
            elif "output_key" not in step and not step.get("output_mapping"):
                # No output declaration — agent output keys will be silently
                # dropped by LangGraph's state reducer. Log a warning so this
                # is caught during development rather than at runtime.
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "step '%s' (type=%s) has no output_key or output_mapping — "
                    "agent output will not reach workflow state",
                    step.get("id"), step.get("type"),
                )
            if step.get("slack_input_key"):
                fields[step["slack_input_key"]] = Any  # type: ignore[assignment]
            fields[f"_agent_token_usage_{step['id']}"] = Any  # type: ignore[assignment]
    # Internal field: agent_url stored while a run is in waiting_agent state
    fields["_agent_url"] = Any  # type: ignore[assignment]
    # Internal field: clarification answers from ask_context interrupt, forwarded to agent re-run
    fields["_clarification_answers"] = Any  # type: ignore[assignment]
    return TypedDict("YamlGraphState", fields, total=False)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Shared graph streaming helper (used by workflow steps and default_workflow)
# ---------------------------------------------------------------------------

async def _cleanup_pvc(run, lease_repo, namespace: str) -> None:
    """Delete PVCs for the run immediately and remove their leases."""
    try:
        from app.runtime.pvc_manager import PvcManager
        mgr = PvcManager(namespace)
        leases = await lease_repo.delete_by_run(run.id)
        for lease in leases:
            await mgr.delete_pvc(lease["pvc_name"])
    except Exception as exc:
        logger.warning("_cleanup_pvc: run %s: %s", run.id, exc)


async def _close_openhands_conversations(runner: YamlGraphRunner, state: dict) -> None:
    if runner._openhands is None:
        return
    conv_map: dict = dict((state or {}).get("_conv_map") or {})
    for name, oh_id in conv_map.items():
        try:
            await runner._openhands.close_conversation(oh_id)
            logger.info("run closed OpenHands conversation '%s' (%s)", name, oh_id)
        except Exception:
            logger.warning("Failed to close OpenHands conversation '%s' (%s)", name, oh_id)


def step_status_from_output(node_name: str, output: Any) -> str:
    """Infer step status from the dict a node returned.

    Empty dict → ``skipped``. If the output carries a ``__failed_step__``
    sentinel matching this node, it caught an internal exception and chose to
    record the error in state — surface that as ``failed`` so the UI doesn't
    show a green checkmark over a captured failure. Anything else → ``finished``.
    """
    if not output:
        return "skipped"
    if isinstance(output, dict) and output.get("__failed_step__") == node_name:
        return "failed"
    return "finished"


_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")


def _parse_questions_string(raw: str) -> list[str]:
    """Extract bare question strings from an LLM-emitted ``questions`` field.

    The LLM tends to emit a preamble paragraph followed by a numbered list
    (e.g. ``"Please clarify:\\n1. ...\\n2. ..."``). A naive split on newlines
    treats the preamble as ``question 0``, which then collides with the
    LLM's own ``1.`` prefix when Slack-formatted, producing two ``1.`` lines.

    When two or more lines start with ``N.``/``N)``, treat those as the
    real questions and strip their leading number; the unnumbered preamble
    is dropped. When fewer than two numbered lines are present, fall back
    to one-question-per-non-empty-line.
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    numbered: list[str] = []
    for ln in lines:
        m = _NUMBERED_LINE_RE.match(ln)
        if m:
            numbered.append(m.group(1).strip())
    if len(numbered) >= 2:
        return numbered
    return lines


async def stream_graph_to_pause(
    runner: YamlGraphRunner,
    run: GraphRun,
    run_repository: Any,
    input_value: Any,
    base_url: str | None = None,
) -> None:
    """
    Stream *runner* from *input_value* until it reaches an interrupt or END,
    updating step_statuses and run status in *run_repository* after each node.

    Callers should initialise ``run.step_statuses`` before calling this.
    """
    runner._current_run = run
    runner._current_run_repository = run_repository

    config = {"configurable": {"thread_id": run.id}}
    if isinstance(input_value, dict):
        current_state: dict = dict(input_value)
    else:
        try:
            snap = await runner.graph.aget_state(config)
            current_state = dict(snap.values) if snap and snap.values else {}
        except Exception:
            current_state = {}

    last_processed: str | None = None
    _stream_interrupt_output: list | None = None  # payload from __interrupt__ chunk if seen
    try:
        async for chunk in runner.graph.astream(input_value, config, stream_mode="updates"):
            for node_name, output in chunk.items():
                if node_name in ("__start__", "__end__"):
                    continue
                # __interrupt__ is a LangGraph internal channel, not a real node.
                # Store its output for interrupt-payload lookup but don't pollute
                # step_statuses (it would show up as a phantom "done" step in the UI).
                if node_name == "__interrupt__":
                    _stream_interrupt_output = output
                    run.step_inputs[node_name] = dict(current_state)
                    if output:
                        run.step_outputs[node_name] = output
                    run.touch()
                    await run_repository.update(run)
                    continue
                status = step_status_from_output(node_name, output)
                run.step_inputs[node_name] = dict(current_state)
                run.step_statuses[node_name] = status
                run.current_step = node_name
                if output:
                    run.step_outputs[node_name] = output
                    if isinstance(output, dict):
                        current_state.update(output)
                logger.info("run %s: step '%s' → %s", run.id, node_name, status)
                last_processed = node_name
                run.touch()
                await run_repository.update(run)
    except Exception as exc:
        logger.exception("run %s: graph execution failed", run.id)
        # Attribute the failure to a specific step only when we can identify
        # one with confidence: either the node body raised mid-execution
        # (its wrapper left it "running"), or a previous node recorded a
        # captured failure via the __failed_step__ sentinel. Otherwise leave
        # step_statuses untouched — the run-level error message is the
        # authoritative signal, and falsely flagging the next forward step
        # as failed misleads the UI when the failure is in a retry loop.
        error_msg = f"{type(exc).__name__}: {exc}"
        running_sid = next(
            (sid for sid, st in run.step_statuses.items() if st == "running"),
            None,
        )
        if running_sid is not None:
            run.step_inputs[running_sid] = dict(current_state)
            run.step_statuses[running_sid] = "failed"
            run.step_outputs[running_sid] = {"error": error_msg}
        else:
            failed_sid = current_state.get("__failed_step__") if isinstance(current_state, dict) else None
            if isinstance(failed_sid, str) and failed_sid in run.step_statuses:
                run.step_inputs[failed_sid] = dict(current_state)
                run.step_statuses[failed_sid] = "failed"
                if not run.step_outputs.get(failed_sid):
                    run.step_outputs[failed_sid] = {"error": error_msg}
        run.status = "failed"
        # Preserve accumulated step outputs AND any internal state keys written
        # mid-step by _save_conv_id (e.g. _openhands_conv_*, _conv_map).
        mid_run = {k: v for k, v in (run.state or {}).items() if k.startswith("_")}
        run.state = {**current_state, **mid_run, "error": error_msg}
        run.current_step = None
        run.touch()
        await run_repository.update(run)
        await _close_openhands_conversations(runner, current_state)
        if runner._pvc_lease_repository is not None:
            from app.core.config import get_settings as _get_settings
            _ns = _get_settings().agent_namespace
            await _cleanup_pvc(run, runner._pvc_lease_repository, _ns)
        from app.services.agent_cleanup import cleanup_run_agents
        from app.core.config import get_settings as _get_settings
        await cleanup_run_agents(run.id, _get_settings(), warm_pod_repository=runner._warm_pod_repository)
        return

    snap = await runner.graph.aget_state(config)

    # Extract the type of the active interrupt (if any) from the snapshot.
    # This determines whether we're waiting for an agent or for user input,
    # regardless of which step type raised the interrupt.
    active_interrupt_type: str | None = None
    for task in snap.tasks:
        for intr in task.interrupts:
            if isinstance(intr.value, dict):
                active_interrupt_type = intr.value.get("type")
                break
        if active_interrupt_type:
            break
    if not active_interrupt_type:
        for intr in getattr(snap, "interrupts", ()):
            if isinstance(intr.value, dict):
                active_interrupt_type = intr.value.get("type")
                break
    # LangGraph 1.x: when a resumed node calls interrupt() a second time,
    # the interrupt is recorded in pending_writes but snap.next is empty
    # (aget_state sees the checkpoint as post-resume / completed). Fall back
    # to the __interrupt__ chunk we captured from the stream.
    if not active_interrupt_type and _stream_interrupt_output:
        for intr in (_stream_interrupt_output if isinstance(_stream_interrupt_output, list) else []):
            if isinstance(intr, dict) and isinstance(intr.get("value"), dict):
                active_interrupt_type = intr["value"].get("type")
                break

    # When snap.next is empty but we saw an interrupt in the stream, the
    # graph IS paused — LangGraph just doesn't reflect it in snap.next for
    # second-interrupt-on-resume scenarios.  Reconstruct the paused step from
    # whichever step_status is still "running".
    _snap_next_override: str | None = None
    if not snap.next and active_interrupt_type:
        _snap_next_override = next(
            (sid for sid, st in run.step_statuses.items() if st == "running"),
            None,
        )
        if _snap_next_override:
            logger.info(
                "run %s: snap.next empty but interrupt type=%r detected in stream — "
                "treating step '%s' as paused",
                run.id, active_interrupt_type, _snap_next_override,
            )

    # Determine whether the run paused at a waiting_agent step, a
    # waiting_approval step (or completed).
    _effective_next = snap.next[0] if snap.next else _snap_next_override
    if _effective_next:
        current_step_id = _effective_next
        step_def = next((s for s in runner.steps if s["id"] == current_step_id), None)
        step_type = step_def.get("type") if step_def else None
        if (active_interrupt_type in ("ask_context", "ask_approval")
                and step_type in ("langgraph-agent", "claude-agent")):
            # A Docker/K8s agent raised a clarification or approval interrupt
            # internally (via meta-LLM). Treat as waiting_approval so the UI
            # can prompt the user, and mark the step accordingly.
            run.status = "waiting_approval"
            if current_step_id in run.step_statuses:
                if active_interrupt_type == "ask_context":
                    run.step_statuses[current_step_id] = "waiting_clarification"
                else:
                    run.step_statuses[current_step_id] = "waiting_approval"
        elif step_type in ("langgraph-agent", "claude-agent"):
            run.status = "waiting_agent"
        else:
            run.status = "waiting_approval"
    else:
        run.status = "completed"
    run.current_step = _effective_next
    run.state = snap.values
    run.touch()
    await run_repository.update(run)
    if run.status == "completed":
        await _close_openhands_conversations(runner, snap.values)
        if runner._pvc_lease_repository is not None:
            from app.core.config import get_settings as _get_settings
            _ns = _get_settings().agent_namespace
            await _cleanup_pvc(run, runner._pvc_lease_repository, _ns)
        from app.services.agent_cleanup import cleanup_run_agents
        from app.core.config import get_settings as _get_settings
        await cleanup_run_agents(run.id, _get_settings(), warm_pod_repository=runner._warm_pod_repository)

    if run.status == "waiting_agent" and run.current_step:
        # Extract agent_url from the interrupt payload and persist it on the run
        # so the agent_callbacks route can find and terminate it if needed.
        agent_url: str | None = None
        for task in snap.tasks:
            for intr in task.interrupts:
                if isinstance(intr.value, dict) and intr.value.get("type") == "waiting_agent":
                    agent_url = intr.value.get("agent_url")
                    break
            if agent_url:
                break
        if not agent_url:
            for intr in getattr(snap, "interrupts", ()):
                if isinstance(intr.value, dict) and intr.value.get("type") == "waiting_agent":
                    agent_url = intr.value.get("agent_url")
                    break
        if agent_url:
            run.agent_url = agent_url
            run.touch()
            await run_repository.update(run)

    if run.status == "waiting_approval" and run.current_step:
        # Reset the approval flag at request time so a loop-back through the
        # same approval node does not see a stale True from the prior
        # iteration. The node body sets it back to True on resume.
        step = next((s for s in runner.steps if s["id"] == run.current_step), None)
        if step and step.get("type") == "human_approval":
            approved_key = step.get("output_key", "approved")
            if (snap.values or {}).get(approved_key) is not False:
                config = {"configurable": {"thread_id": run.id}}
                await runner.graph.aupdate_state(config, {approved_key: False})
                run.state = {**(run.state or {}), approved_key: False}
                run.touch()
                await run_repository.update(run)

    if run.status == "waiting_approval" and base_url and run.current_step:
        step = next((s for s in runner.steps if s["id"] == run.current_step), None)
        # Fire Slack notification for explicit ask_context steps AND for agent steps
        # that raised an ask_context interrupt internally via meta-LLM.
        is_agent_ask_context = (
            step and step.get("type") in ("langgraph-agent", "claude-agent")
            and active_interrupt_type == "ask_context"
        )
        if (step and (step.get("type") == "ask_context" or step.get("slack_notifications"))) or is_agent_ask_context:
            from app.core.config import get_settings
            from app.infrastructure.notifications.webhook_notifier import (
                post_slack_ask_context, post_slack_addon_notification, post_slack_thread_questions,
            )
            settings = get_settings()
            # Step-level token/channel override global defaults.
            effective_token = (step.get("slack_token") if step else None) or settings.slack_bot_token
            effective_channel = (step.get("slack_channel") if step else None) or settings.slack_approvals_channel
            if effective_token and effective_channel:
                questions: list[str] = []
                for task in snap.tasks:
                    for intr in task.interrupts:
                        if isinstance(intr.value, dict) and intr.value.get("type") == "ask_context":
                            questions = intr.value.get("questions", [])
                if not questions:
                    for intr in getattr(snap, "interrupts", ()):
                        if isinstance(intr.value, dict) and intr.value.get("type") == "ask_context":
                            questions = intr.value.get("questions", [])
                existing_ts = snap.values.get("_slack_ask_context_ts")
                existing_channel = snap.values.get("_slack_ask_context_channel")
                if questions:
                    if step and step.get("slack_payload"):
                        # Custom payload template — inject step-level channel as {slack_channel}
                        extra: dict = {}
                        if step.get("slack_channel"):
                            extra["slack_channel"] = step["slack_channel"]
                        await post_slack_addon_notification(
                            bot_token=effective_token,
                            payload_template=step["slack_payload"],
                            run_id=run.id,
                            state={**(snap.values or {}), **extra},
                            questions=questions,
                        )
                    elif existing_ts and existing_channel:
                        # Loop-back: post new questions as a reply in the same thread
                        await post_slack_thread_questions(
                            effective_token, existing_channel, existing_ts, questions,
                        )
                    else:
                        # First interrupt: open a new root message
                        notif_resp = await post_slack_ask_context(
                            effective_token, effective_channel,
                            questions, run.id, snap.values,
                        )
                        if notif_resp and notif_resp.get("ok"):
                            ts = notif_resp.get("ts")
                            channel = notif_resp.get("channel")
                            if ts and channel:
                                config = {"configurable": {"thread_id": run.id}}
                                await runner.graph.aupdate_state(config, {
                                    "_slack_ask_context_ts": ts,
                                    "_slack_ask_context_channel": channel,
                                })
                                run.state = {**run.state, "_slack_ask_context_ts": ts, "_slack_ask_context_channel": channel}
                                run.touch()
                                await run_repository.update(run)

        elif step and step.get("notify"):
            notif_resp = await send_approval_notification(step["notify"], run.id, snap.values, base_url)
            if notif_resp and notif_resp.get("ok"):
                ts = notif_resp.get("ts")
                channel = notif_resp.get("channel")
                if ts and channel:
                    config = {"configurable": {"thread_id": run.id}}
                    await runner.graph.aupdate_state(config, {"_slack_thread_ts": ts, "_slack_channel": channel})
                    run.state = {**run.state, "_slack_thread_ts": ts, "_slack_channel": channel}
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
            type: llm_structured | llm | mcp | human_approval | execute | workflow | cron | http | http_call | python
            when: <state-key>          # skip node if state[key] is falsy
            system_prompt: "..."       # llm / llm_structured
            user_template: "..."       # {key} placeholders resolved from state
            output_key: <key>          # where to store the result
            bind_mcp_tools: true       # llm_structured only – set false to hide MCP tools
            max_iterations: 25         # llm_structured only – override default iteration cap
            fail_if_false:             # llm_structured only – fail the run if any listed bool
              - success                #   output field is False (uses 'error'/'summary' as detail)
            output:                    # llm_structured only
              - name: needs_jira
                type: bool
                description: "..."
            tool: <tool-name>          # mcp only
            tool_input:                # mcp only – dict of {key}-templated values
              query: "{request}"
            repo_template: "{repo}"    # execute only
            instructions_template: "{plan}"  # execute only
            stop_on_failure: false     # execute only — when true, an exception
                                       #   inside the node fails the run
                                       #   immediately. When false (default) the
                                       #   error is captured under output_key
                                       #   so the next node can decide to retry.
            workflow_id: <id>          # workflow only — child workflow to spawn
            input_template: "{request}"  # workflow only — request passed to child
            schedule: "0 9 * * 1-5"   # cron only — 5-field cron expression (UTC)
            request_template: "..."    # cron only — initial request; supports {now}, {date}
            url: "https://..."         # http_call only — endpoint; {key} templates resolved
            method: POST               # http_call only — GET | POST | PUT | PATCH | DELETE
            headers:                   # http_call only — request headers; values support {key}
              Authorization: "Bearer {token}"
            body:                      # http_call only — JSON body; values support {key}
              issue_key: "{ticket_id}"
            code: |                    # python only — executed with ``state`` dict in scope;
              output = state["x"] + 1  #   set ``output`` variable to store the result
            routes:                    # llm_structured / switch — multiple branches
              - when: <state-key>      # route taken when state[key] is truthy
                next: <node-id>
                wait_seconds: 60       # optional — sleep before the next node runs
                                       #   (capped at 3600s; useful for retry back-edges)

    ``human_approval`` steps additionally support an optional ``notify`` field
    that fires an HTTP request when the run reaches ``waiting_approval``:

        notify:
          url: "https://hooks.example.com/approval"  # required
          method: POST                                # optional, default POST
          headers:                                    # optional
            X-Custom: "value"
          auth:                                       # optional
            type: bearer                              # bearer | basic
            token: "..."                              # bearer only
            username: "..."                           # basic only
            password: "..."                           # basic only
          payload:                                    # optional JSON body
            text: "Approval needed: {plan}"
            approve_url: "{approve_url}"
            reject_url: "{reject_url}"
            run_id: "{run_id}"

    Template variables in payload / header values / url: {run_id}, {approve_url},
    {reject_url}, and any key from the current graph state.

    Steps are chained sequentially.  ``human_approval`` calls interrupt() and
    expects the caller to resume with {"approved": bool, "reason": str|None}.

    ``workflow`` steps fire-and-forget spawn a child workflow run and store
    {"child_run_id": ..., "workflow_id": ..., "status": "started"} in output_key.

    ``cron`` steps are entry-point triggers: the CronScheduler in the container
    creates a new run on the configured schedule and passes trigger metadata via
    the ``trigger_info`` state key.  When the node executes it simply returns
    that metadata under ``output_key``.

    ``http`` steps are entry-point triggers: the ``POST /api/v1/webhooks/{id}``
    endpoint validates an HMAC-SHA256 signature, then starts a run with the
    webhook body stored in the ``trigger_payload`` state key.  When the node
    executes it returns that payload under ``output_key``.
    Registry and run_repository must be injected after construction (done by
    load_yaml_graphs).
    """

    def __init__(
        self,
        definition: dict[str, Any],
        llm: BaseChatModel,
        mcp_tools_provider: McpToolsProvider,
        openhands: OpenHandsAdapter | None = None,
        llm_factory: Callable[[str | None, str | None], BaseChatModel] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.id: str = definition["id"]
        # Human-readable name; fall back to title-casing the id
        self.name: str = definition.get(
            "name",
            self.id.replace("-", " ").replace("_", " ").title(),
        )
        self.description: str = definition.get("description", "")
        self._max_iterations: int = definition.get("max_iterations", 10)
        self._use_meta_llm: bool = definition.get("use_meta_llm", True)
        self.readonly: bool = False  # Set post-construction by build_registry_from_definitions
        self._steps: list[dict[str, Any]] = definition["steps"]
        self._llm = llm
        self._llm_factory = llm_factory
        self._mcp = mcp_tools_provider
        self._openhands = openhands
        self._checkpointer: BaseCheckpointSaver = checkpointer or MemorySaver()
        # Injected post-construction by load_yaml_graphs
        self._registry: Any = None
        self._run_repository: Any = None
        # Injected post-construction by the application container for agent steps
        self._agent_backend: Any = None
        # Injected post-construction by the application container so agent steps
        # can pass the backend's public base URL to spawned agent servers.
        self._callback_base_url: str = ""
        # Injected post-construction for PVC lease tracking (optional)
        self._pvc_lease_repository: Any = None
        # Injected post-construction for agent task tracking (optional)
        self._agent_task_repository: Any = None
        # Injected post-construction for warm pod reuse tracking (optional)
        self._warm_pod_repository: Any = None
        # Set by stream_graph_to_pause to enable mid-run persistence from nodes
        self._current_run: Any = None
        self._current_run_repository: Any = None
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
        step_ids = [s["id"] for s in self._steps]
        all_ids = set(step_ids)

        for step in self._steps:
            sg.add_node(step["id"], self._make_node(step))

        if not step_ids:
            sg.add_edge(START, END)
            return sg.compile(checkpointer=self._checkpointer)

        sg.add_edge(START, step_ids[0])

        _MULTI_OUTPUT_TYPES = frozenset({"llm_structured", "switch"})

        for i, step in enumerate(self._steps):
            sid = step["id"]
            step_type = step.get("type")

            # parallel: unconditional fan-out to all targets
            if step_type == "parallel":
                targets = step.get("targets") or []
                for t in targets:
                    sg.add_edge(sid, t if t in all_ids else END)
                if not targets:
                    # no targets configured — connect sequentially or to END
                    if i < len(self._steps) - 1:
                        sg.add_edge(sid, step_ids[i + 1])
                    else:
                        sg.add_edge(sid, END)
                continue

            routes = step.get("routes") or []
            next_val = step.get("next")

            if routes:
                if step_type not in _MULTI_OUTPUT_TYPES and len(routes) > 1:
                    raise ValueError(
                        f"Step '{sid}' (type={step_type}) cannot have more than "
                        f"1 route; only llm_structured and switch support multiple routes."
                    )
                # A direct edge skips the router, so any route carrying
                # wait_seconds must go through add_conditional_edges to honor it.
                any_wait = any(r.get("wait_seconds") for r in routes)
                if len(routes) == 1 and "when" not in routes[0] and not any_wait:
                    dest = routes[0]["next"]
                    sg.add_edge(sid, dest if dest in all_ids else END)
                else:
                    route_map = {
                        r["next"]: (r["next"] if r["next"] in all_ids else END)
                        for r in routes
                        if "next" in r
                    }
                    sg.add_conditional_edges(
                        sid,
                        self._make_router_fn(sid, routes),
                        route_map,
                    )
            elif next_val:
                sg.add_edge(sid, next_val if next_val in all_ids else END)
            elif i < len(self._steps) - 1:
                sg.add_edge(sid, step_ids[i + 1])
            else:
                sg.add_edge(sid, END)

        return sg.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------
    # Node factories
    # ------------------------------------------------------------------

    def _get_llm_for_step(self, step: dict[str, Any]) -> BaseChatModel:
        """Return the LLM to use for a step, applying per-step provider/model overrides."""
        provider: str | None = step.get("llm_provider") or None
        model: str | None = step.get("model") or None
        if (provider or model) and self._llm_factory is not None:
            return self._llm_factory(provider, model)
        return self._llm

    def _make_node(self, step: dict[str, Any]):
        t = step["type"]
        if t == "llm_structured":
            fn = self._llm_structured_node(step)
        elif t in ("langgraph-agent", "claude-agent"):
            fn = self._agent_node(step)
        elif t == "llm":
            fn = self._llm_node(step)
        elif t == "mcp":
            fn = self._mcp_node(step)
        elif t == "ask_context":
            fn = self._ask_context_node(step)
        elif t == "human_approval":
            fn = self._approval_node(step)
        elif t == "execute":
            fn = self._execute_node(step)
        elif t == "workflow":
            fn = self._workflow_node(step)
        elif t == "cron":
            fn = self._cron_trigger_node(step)
        elif t == "http":
            fn = self._http_trigger_node(step)
        elif t == "http_call":
            fn = self._http_call_node(step)
        elif t == "python":
            fn = self._python_node(step)
        elif t == "parallel":
            fn = self._parallel_node(step)
        elif t == "join":
            fn = self._join_node(step)
        elif t == "switch":
            fn = self._switch_node(step)
        else:
            raise ValueError(f"Unknown step type '{t}' in graph '{self.id}'")
        wrapped = self._wrap_with_status_running(self._wrap_with_loop_guard(step, fn), step)
        return self._wrap_with_when(step, wrapped)

    _NO_LOOP_GUARD_TYPES: frozenset = frozenset({"ask_context", "human_approval", "cron", "http", "parallel", "join", "switch", "langgraph-agent", "claude-agent"})

    def _wrap_with_when(self, step: dict[str, Any], fn: Callable) -> Callable:
        """Skip node if step has a `when` key and state[when] is falsy."""
        when_key = step.get("when")
        if not when_key:
            return fn

        async def _wrapped(state: dict) -> dict:
            if not state.get(when_key):
                logger.info("[%s] step '%s' skipped (when: %s is falsy)", self.id, step["id"], when_key)
                return {}
            return await fn(state)

        return _wrapped

    def _wrap_with_status_running(self, fn: Callable, step: dict[str, Any]) -> Callable:
        """Persist step_status="running" + current_step before the node executes.

        Without this, step_statuses keeps the value from the previous pass
        through the same node (typically "finished"), so the API can't tell
        a UI which node is actually live during a loop-back. With this hook
        every node briefly publishes "running" before its real result is
        written by stream_graph_to_pause's chunk handler.
        """
        step_id = step["id"]
        is_async = asyncio.iscoroutinefunction(fn)

        async def _wrapped(state: dict) -> dict:
            run = self._current_run
            repo = self._current_run_repository
            if run is not None and repo is not None:
                run.step_statuses[step_id] = "running"
                run.current_step = step_id
                run.touch()
                try:
                    await repo.update(run)
                except Exception:
                    logger.exception(
                        "[%s] failed to persist 'running' status for step '%s'",
                        self.id, step_id,
                    )
            return (await fn(state)) if is_async else fn(state)

        return _wrapped

    def _wrap_with_loop_guard(self, step: dict[str, Any], fn: Callable) -> Callable:
        """Wrap a node function to track visit counts and enforce max_loops."""
        if step.get("type") in self._NO_LOOP_GUARD_TYPES:
            return fn
        step_id = step["id"]
        max_loops = step.get("max_loops", self._max_iterations)
        is_async = asyncio.iscoroutinefunction(fn)
        graph_id = self.id

        async def _guarded(state: dict) -> dict:
            result = (await fn(state)) if is_async else fn(state)
            if not result:  # node was skipped (returned {})
                return result
            counts: dict = dict(state.get("_visit_counts") or {})
            counts[step_id] = counts.get(step_id, 0) + 1
            if counts[step_id] > max_loops:
                raise ValueError(
                    f"[{graph_id}] step '{step_id}' exceeded max_loops={max_loops} "
                    f"(ran {counts[step_id]} times)"
                )
            return {**result, "_visit_counts": counts}

        return _guarded

    _MAX_ROUTE_WAIT_SECONDS: float = 3600.0

    def _make_router_fn(
        self, source_id: str, routes: list[dict[str, Any]]
    ) -> Callable[[dict], Awaitable[str]]:
        """Return an async routing function for add_conditional_edges.


        A route may declare ``wait_seconds: <number>`` to delay the transition
        to its destination. The wait runs after the route is selected and
        before the next node executes; it is capped at ``_MAX_ROUTE_WAIT_SECONDS``.
        While sleeping, ``run.waiting_transition`` is set so the UI can
        visualise the pause; it's cleared in a ``finally`` block so a
        cancellation or exception doesn't leave a stale waiting indicator.
        """
        import ast as _ast
        import builtins as _builtins

        graph_id = self.id

        # AST node types that are never safe to execute in a route condition.
        _UNSAFE_AST = (
            _ast.Import, _ast.ImportFrom,
            _ast.FunctionDef, _ast.AsyncFunctionDef,
            _ast.ClassDef, _ast.Lambda,
            _ast.Global, _ast.Nonlocal,
            _ast.Await, _ast.Yield, _ast.YieldFrom,
            _ast.Delete,
        )

        def _eval_condition(when: str, state: dict) -> bool:
            """Parse and evaluate a route condition against the current state.

            Accepts:
            - Simple state key:  ``approved``
            - Negation:          ``!approved``
            - Any Python expression using state vars and stdlib builtins:
              ``len(hello_out) <= len(world_out)``
              ``score > 4 and status != "skip"``
            JS-style ``&&`` / ``||`` / ``===`` / ``!==`` are rewritten to Python.
            """
            expr = (
                str(when)
                .replace("&&", " and ")
                .replace("||", " or ")
                .replace("!==", " != ")
                .replace("===", " == ")
            )
            try:
                tree = _ast.parse(expr, mode="eval")
                for node in _ast.walk(tree):
                    if isinstance(node, _UNSAFE_AST):
                        raise ValueError(f"unsafe AST node: {type(node).__name__}")
                code = compile(tree, "<route-condition>", "eval")
                result = bool(eval(code, vars(_builtins), dict(state)))  # noqa: S307
                logger.debug(
                    "[%s] router '%s': condition %r → %s",
                    graph_id, source_id, when, result,
                )
                return result
            except Exception:
                # Fallback: simple state-key lookup with optional ! negation
                negate = expr.strip().startswith("!")
                key = expr.strip()[1:].strip() if negate else expr.strip()
                val = bool(state.get(key))
                result = not val if negate else val
                logger.debug(
                    "[%s] router '%s': condition %r → %s (fallback key-lookup, key=%r, raw=%r)",
                    graph_id, source_id, when, result, key, state.get(key),
                )
                return result

        def _select(state: dict) -> dict[str, Any]:
            for route in routes:
                when = route.get("when")
                if when is None:
                    logger.debug(
                        "[%s] router '%s': default route (when=null) → '%s'",
                        graph_id, source_id, route.get("next"),
                    )
                    return route
                if _eval_condition(str(when), state):
                    return route
            # No condition matched and no `when: null` default declared. The
            # previous behaviour was to silently fall back to routes[-1], but
            # that hid bugs: e.g. a develop ↔ deliver-result loop where
            # `success` and `openhands_crashed` both resolved to False got
            # silently routed back to develop and span forever. Fail loudly
            # so the workflow author either adds an explicit default or
            # extends the conditions.
            checked = [r.get("when") for r in routes]
            # Extract the relevant state values for each condition key so the
            # error message explains exactly why nothing matched.
            relevant: dict = {}
            for cond in checked:
                if cond is None:
                    continue
                key = str(cond).strip().lstrip("!").split()[0]
                relevant[str(cond)] = state.get(key)
            logger.error(
                "[%s] router '%s': no route matched | checked=%s | state_values=%s | "
                "non-null state keys=%s",
                graph_id, source_id, checked, relevant,
                [k for k, v in state.items() if v is not None and not k.startswith("_")],
            )
            raise ValueError(
                f"router: no route matched on state and no default "
                f"(when=null) was declared; checked={checked}. "
                f"Add a `when: null` route or a condition that covers this case."
            )

        runner = self

        async def router(state: dict) -> str:
            chosen = _select(state)
            logger.info(
                "[%s] router '%s' → '%s' (condition: %r)",
                graph_id, source_id, chosen.get("next"), chosen.get("when"),
            )
            wait = chosen.get("wait_seconds")
            if wait:
                try:
                    delay = float(wait)
                except (TypeError, ValueError):
                    logger.warning("ignoring non-numeric wait_seconds=%r on route to %s", wait, chosen.get("next"))
                    delay = 0.0
                if delay < 0:
                    logger.warning("ignoring negative wait_seconds=%s on route to %s", delay, chosen.get("next"))
                    delay = 0.0
                if delay > runner._MAX_ROUTE_WAIT_SECONDS:
                    logger.warning("capping wait_seconds=%s at %s on route to %s",
                                   delay, runner._MAX_ROUTE_WAIT_SECONDS, chosen.get("next"))
                    delay = runner._MAX_ROUTE_WAIT_SECONDS
                if delay > 0:
                    logger.info("waiting %.1fs before transitioning to '%s'", delay, chosen.get("next"))
                    run = runner._current_run
                    repo = runner._current_run_repository
                    if run is not None:
                        from app.domain.models.graph_run import WaitingTransition
                        run.waiting_transition = WaitingTransition(
                            source=source_id,
                            target=chosen["next"],
                            wait_seconds=delay,
                            started_at=datetime.now(timezone.utc),
                        )
                        run.touch()
                        if repo is not None:
                            await repo.update(run)
                    try:
                        await asyncio.sleep(delay)
                    finally:
                        if run is not None:
                            run.waiting_transition = None
                            run.touch()
                            if repo is not None:
                                await repo.update(run)
            return chosen["next"]
        return router

    _SUBMIT_TOOL = "submit_output"
    _MAX_ITERATIONS = 25

    def _agent_node(self, step: dict[str, Any]):
        """Node factory for ``langgraph-agent`` and ``claude-agent`` step types.

        Delegates to ``app.steps.agent_executor.execute_agent_step``.  The
        agent backend is resolved lazily from ``self._agent_backend``; it is
        injected post-construction (like ``_registry`` and ``_run_repository``)
        by the application container's ``build_container`` / ``refresh_runner``
        path.
        """
        graph_id = self.id
        use_meta_llm = self._use_meta_llm

        async def node(state: dict) -> dict:
            step_id = step["id"]
            agent_backend = getattr(self, "_agent_backend", None)
            if agent_backend is None:
                logger.error(
                    "[%s] step '%s': _agent_backend not injected — "
                    "ensure the ApplicationContainer has an agent_backend configured",
                    graph_id, step_id,
                )
                return {step.get("output_key", step_id): {"error": "agent backend not configured"}}

            run_id: str = self._current_run.id if self._current_run else "unknown"
            callback_base_url: str = self._callback_base_url or ""

            from app.core.config import get_settings
            from app.steps.agent_executor import execute_agent_step
            logger.info("[%s] step '%s' running (%s)", graph_id, step_id, step["type"])
            try:
                return await execute_agent_step(
                    step, state, agent_backend, run_id, callback_base_url,
                    settings=get_settings(),
                    run_repository=self._current_run_repository,
                    pvc_lease_repository=self._pvc_lease_repository,
                    agent_task_repository=self._agent_task_repository,
                    warm_pod_repository=self._warm_pod_repository,
                    use_meta_llm=use_meta_llm,
                )
            except Exception as _step_exc:
                logger.error("[%s] step '%s' raised: %s", graph_id, step_id, _step_exc)
                return {"__failed_step__": step_id, "error": str(_step_exc)}

        return node

    # DEPRECATED: use langgraph-agent or claude-agent instead
    def _llm_structured_node(self, step: dict[str, Any]):
        graph_id = self.id
        base_llm = self._get_llm_for_step(step)

        async def node(state: dict) -> dict:
            step_id = step["id"]
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

            # bind_mcp_tools defaults to True for backward compat; set to false
            # on steps that only reason about text and should not call MCP tools
            # (prevents the LLM from invoking unneeded/restricted server tools).
            mcp_tools = self._mcp.get_tools() if step.get("bind_mcp_tools", True) else []
            llm = base_llm.bind_tools(mcp_tools + [submit_tool])

            system_prompt = step.get("system_prompt", "")
            user_message = self._render(step.get("user_template", "{request}"), state)
            messages: list = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            logger.info(
                "[%s] step '%s' → LLM | system: %s | user: %s",
                graph_id, step_id, system_prompt, user_message,
            )

            max_iterations = step.get("max_iterations", self._MAX_ITERATIONS)
            for iteration in range(1, max_iterations + 1):
                response = await llm.ainvoke(messages)
                messages.append(response)
                tool_calls = response.tool_calls or []
                logger.info(
                    "[%s] step '%s' ← LLM (iter %d) | content: %r | tool_calls: %s",
                    graph_id, step_id, iteration, response.content,
                    [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls],
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
                submit_tc = next((tc for tc in tool_calls if tc["name"] == self._SUBMIT_TOOL), None)
                if submit_tc is not None:
                    args = submit_tc["args"]
                    required_fields = [o["name"] for o in step.get("output", [])]
                    missing = [f for f in required_fields if f not in args or args[f] is None or args[f] == ""]
                    if missing:
                        logger.warning(
                            "[%s] step '%s' submit_output rejected — missing/empty fields: %s",
                            graph_id, step_id, missing,
                        )
                        messages.append(ToolMessage(
                            content=(
                                f"submit_output rejected: the following required fields are "
                                f"missing or empty: {missing}. "
                                f"Call submit_output again and fill in EVERY required field. "
                                f"Write SHORT summaries (3–5 sentences each) — do NOT try to "
                                f"copy raw file contents into the fields. Summarise what you found."
                            ),
                            tool_call_id=submit_tc["id"],
                        ))
                        continue
                    logger.info("[%s] step '%s' finished: %s", graph_id, step_id, args)
                    # fail_if_false: list of bool output fields that must be True
                    for field in step.get("fail_if_false", []):
                        if field in args and not args[field]:
                            detail = args.get("error") or args.get("summary") or ""
                            raise ValueError(
                                f"[{graph_id}] step '{step_id}' failed: "
                                f"'{field}' is false. {detail}".strip()
                            )
                    return args

                # Execute MCP tool calls and feed results back
                for tc in tool_calls:
                    tool_name = tc["name"]
                    server = self._mcp.get_tool_server(tool_name)
                    server_tag = f" (server: {server})" if server else ""
                    tool = self._mcp.get_tool(tool_name)
                    if tool:
                        logger.info(
                            "[%s] step '%s' → tool '%s'%s | args: %s",
                            graph_id, step_id, tool_name, server_tag, tc["args"],
                        )
                        try:
                            result = await tool.ainvoke(tc["args"])
                            content = self._extract_mcp_text(result)
                        except Exception as exc:
                            logger.exception(
                                "[%s] step '%s' tool '%s'%s failed",
                                graph_id, step_id, tool_name, server_tag,
                            )
                            content = str(exc)
                    else:
                        logger.warning(
                            "[%s] step '%s' unknown tool requested: '%s'",
                            graph_id, step_id, tool_name,
                        )
                        content = f"Tool '{tool_name}' is not available"
                    logger.info(
                        "[%s] step '%s' ← tool '%s'%s | result: %s",
                        graph_id, step_id, tool_name, server_tag, content,
                    )
                    messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

            raise ValueError(
                f"[{graph_id}] step '{step_id}': reached {max_iterations} iterations without structured output"
            )

        return node

    def _llm_node(self, step: dict[str, Any]):
        graph_id = self.id
        llm = self._get_llm_for_step(step)

        async def node(state: dict) -> dict:
            step_id = step["id"]
            output_key = step.get("output_key") or step_id
            logger.info("[%s] step '%s' running (llm)", graph_id, step_id)
            try:
                system_prompt = step.get("system_prompt", "")
                user_message = self._render(step.get("user_template", "{request}"), state)
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
                logger.info(
                    "[%s] step '%s' → LLM | system: %s | user: %s",
                    graph_id, step_id, system_prompt, user_message,
                )
                response = await llm.ainvoke(messages)
                logger.info("[%s] step '%s' ← LLM | content: %r", graph_id, step_id, response.content)
                logger.info("[%s] step '%s' finished", graph_id, step_id)
                return {output_key: response.content}
            except Exception as exc:
                logger.exception("[%s] step '%s' llm failed", graph_id, step_id)
                return {output_key: {"error": str(exc)}}
        return node

    def _mcp_node(self, step: dict[str, Any]):
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            tool_name = step["tool"]
            server = self._mcp.get_tool_server(tool_name)
            server_tag = f" (server: {server})" if server else ""
            logger.info("[%s] step '%s' running (mcp tool='%s'%s)", graph_id, step_id, tool_name, server_tag)
            tool = self._mcp.get_tool(tool_name)
            if not tool:
                logger.warning("[%s] step '%s' MCP tool '%s' not available", graph_id, step_id, tool_name)
                err_msg = f"MCP tool '{tool_name}' not available"
                if "output_key" in step:
                    return {"__failed_step__": step_id, step["output_key"]: err_msg}
                return {"__failed_step__": step_id, "error": err_msg}
            try:
                tool_input = {
                    k: self._render(v, state)
                    for k, v in step.get("tool_input", {}).items()
                }
                logger.info(
                    "[%s] step '%s' MCP tool='%s' input=%r",
                    graph_id, step_id, tool_name, tool_input,
                )
                empty_inputs = [k for k, v in tool_input.items() if v == "" or v is None]
                if empty_inputs:
                    logger.warning(
                        "[%s] step '%s' MCP tool='%s': empty/null input fields: %s "
                        "(template keys may be missing from state)",
                        graph_id, step_id, tool_name, empty_inputs,
                    )
                result = await tool.ainvoke(tool_input)
                logger.info("[%s] step '%s' finished", graph_id, step_id)
                output_text = self._extract_mcp_text(result)
                out = {f"_mcp_output_{step_id}": output_text}
                if "output_key" in step:
                    out[step["output_key"]] = output_text
                return out
            except Exception as exc:
                logger.exception("[%s] step '%s' MCP tool '%s'%s failed", graph_id, step_id, tool_name, server_tag)
                err_msg = f"Error calling '{tool_name}': {exc}"
                if "output_key" in step:
                    return {"__failed_step__": step_id, step["output_key"]: err_msg}
                return {"__failed_step__": step_id, "error": err_msg}
        return node

    def _ask_context_node(self, step: dict[str, Any]):
        """
        Pause execution and present questions to the user.

        Questions come from a previous step via ``questions_key`` (the state key
        that holds a list of strings).  Alternatively they can be hardcoded in
        the YAML via ``questions`` (a list of strings, supports {key} templates).
        Answers are written to ``output_key`` as a dict {str(index): answer}.

        Slack notification (root-level message + read reply from thread) is handled
        in stream_graph_to_pause after the interrupt fires.
        """
        graph_id = self.id
        step_id = step["id"]
        output_key = step.get("output_key", f"{step_id}_answers")
        questions_key: str | None = step.get("questions_key")
        static_questions: list[str] = step.get("questions") or []

        async def node(state: dict) -> dict:
            if questions_key:
                raw = state.get(questions_key) or []
                # llm_structured outputs str, not list — split on newlines if needed
                if isinstance(raw, str):
                    questions = _parse_questions_string(raw)
                else:
                    questions = list(raw)
            else:
                questions = [self._render(q, state) for q in static_questions]
            logger.info("[%s] step '%s' presenting %d question(s)", graph_id, step_id, len(questions))

            answers: dict = interrupt({"type": "ask_context", "questions": questions})
            return {output_key: answers}
        return node

    def _approval_node(self, step: dict[str, Any]):
        graph_id = self.id
        # output_key lets workflows with multiple approvals write to distinct state keys.
        # Defaults to "approved" for backward compatibility.
        approved_key = step.get("output_key", "approved")

        def node(state: dict) -> dict:
            step_id = step["id"]
            logger.info("[%s] step '%s' waiting for approval", graph_id, step_id)
            payload = {
                k: self._render(v, state)
                for k, v in (step.get("interrupt_payload") or {"plan": "{plan}"}).items()
            }
            decision: dict = interrupt(payload)
            approved = decision.get("approved", False)
            corrections: dict = decision.get("corrections") or {}
            logger.info(
                "[%s] step '%s' decision: approved=%s corrections=%s",
                graph_id, step_id, approved, list(corrections.keys()),
            )
            result: dict = {
                approved_key: approved,
                "reject_reason": decision.get("reason"),
            }
            result.update(corrections)
            return result
        return node

    def _execute_node(self, step: dict[str, Any]):
        graph_id = self.id
        step_id = step["id"]
        output_key = step.get("output_key", f"{step_id}_result")

        async def node(state: dict) -> dict:
            if self._openhands is None:
                logger.warning("[%s] step '%s' OpenHands not configured", graph_id, step_id)
                return {output_key: "OpenHands not configured"}
            conv_id_key = f"_openhands_conv_{step_id}"
            conversation_id: str | None = step.get("conversation_id")
            conv_map: dict = dict(state.get("_conv_map") or {})

            if conversation_id:
                existing_conv_id: str | None = conv_map.get(conversation_id)
            else:
                existing_conv_id = state.get(conv_id_key)

            async def _save_conv_id(oh_id: str) -> None:
                if self._current_run is None or self._current_run_repository is None:
                    return
                update: dict = {conv_id_key: oh_id}
                if conversation_id:
                    current_map = dict((self._current_run.state or {}).get("_conv_map") or {})
                    update["_conv_map"] = {**current_map, conversation_id: oh_id}
                self._current_run.state = {**(self._current_run.state or {}), **update}
                self._current_run.touch()
                await self._current_run_repository.update(self._current_run)

            logger.info("[%s] step '%s' running (execute)", graph_id, step_id)
            try:
                repo = self._render(step.get("repo_template", "{repo}"), state)
                instructions = self._render(step.get("instructions_template", "{plan}"), state)
                branch_template = step.get("branch_template")
                branch = self._render(branch_template, state) if branch_template else None
                logger.info("[%s] step '%s' repo='%s'%s", graph_id, step_id, repo,
                            f", resuming conv {existing_conv_id}" if existing_conv_id else "")
                result = await self._openhands.execute(
                    repo=repo,
                    instructions=instructions,
                    existing_conv_id=existing_conv_id,
                    conv_id_callback=_save_conv_id,
                    branch=branch,
                )
                logger.info("[%s] step '%s' finished", graph_id, step_id)
                output: dict = {output_key: result}
                oh_id = result.get("conversation_id")
                if oh_id:
                    output[conv_id_key] = oh_id
                    if conversation_id:
                        output["_conv_map"] = {**conv_map, conversation_id: oh_id}
                return output
            except Exception as exc:
                logger.exception("[%s] step '%s' execute failed", graph_id, step_id)
                # stop_on_failure=True: re-raise so the run is marked failed
                # immediately. Default (False): record the error in state so
                # the next node (typically a deliver-result LLM) can introspect
                # it and decide whether to retry or proceed.
                if step.get("stop_on_failure"):
                    raise
                return {output_key: {"error": str(exc)}, "__failed_step__": step_id}
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

            try:
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
            except Exception as exc:
                logger.exception("[%s] step '%s' workflow spawn failed", graph_id, step_id)
                return {output_key: {"error": str(exc)}}

        return node

    def _cron_trigger_node(self, step: dict[str, Any]):
        """Pass-through node for cron-triggered runs.

        The CronScheduler seeds the state with ``trigger_info`` before the graph
        starts.  This node reads that value and stores it under ``output_key`` so
        downstream steps can reference when/how the run was triggered.
        """
        graph_id = self.id
        output_key = step.get("output_key", "trigger_info")

        async def node(state: dict) -> dict:
            step_id = step["id"]
            logger.info("[%s] step '%s' running (cron trigger)", graph_id, step_id)
            return {output_key: state.get("trigger_info", {})}

        return node

    def _http_trigger_node(self, step: dict[str, Any]):
        """Pass-through node for HTTP-triggered runs.

        The webhook endpoint seeds the state with ``trigger_payload`` (the raw
        request body) before the graph starts.  This node reads that value and
        stores it under ``output_key`` so downstream steps can reference the
        incoming data.

        When a non-empty payload arrives and ``request`` is not already set in
        state (i.e. the run was webhook-triggered rather than manually invoked),
        ``request`` is also populated with the JSON-serialised payload so that
        downstream steps using ``{request}`` work uniformly for both invocation
        paths.
        """
        graph_id = self.id
        output_key = step.get("output_key", "trigger_payload")

        async def node(state: dict) -> dict:
            step_id = step["id"]
            logger.info("[%s] step '%s' running (http trigger)", graph_id, step_id)
            payload = state.get("trigger_payload", {})
            updates: dict[str, Any] = {output_key: payload}
            if payload and not state.get("request"):
                updates["request"] = json.dumps(payload) if isinstance(payload, dict) else str(payload)
            return updates

        return node

    def _http_call_node(self, step: dict[str, Any]):
        """Make an outbound HTTP request.

        Response is stored as ``{"status": <int>, "body": <str>}`` under
        ``output_key`` (defaults to the step id).  All string fields in
        ``url``, ``headers`` values, and ``body`` values are rendered with
        ``{key}`` placeholders resolved from state before the request is sent.
        """
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            method = step.get("method", "GET").upper()
            output_key = step.get("output_key") or step_id

            logger.info("[%s] step '%s' running (http_call %s ...)", graph_id, step_id, method)
            try:
                url = self._render(step.get("url", ""), state)
                raw_headers = step.get("headers", {})
                headers = {k: self._render(str(v), state) for k, v in raw_headers.items()}
                raw_body = step.get("body")
                body = self._render_deep(raw_body, state) if raw_body else None
                logger.info("[%s] step '%s' url=%s", graph_id, step_id, url)
                async with httpx.AsyncClient(timeout=60) as client:
                    if method in ("GET", "DELETE", "HEAD"):
                        resp = await client.request(method, url, headers=headers)
                    else:
                        resp = await client.request(method, url, headers=headers, json=body)
                result: dict[str, Any] = {"status": resp.status_code, "body": resp.text}
                logger.info("[%s] step '%s' finished (status=%d)", graph_id, step_id, resp.status_code)
                if resp.status_code >= 400 and not step.get("ignore_http_errors"):
                    raise ValueError(
                        f"HTTP {resp.status_code} from {method} {url}: {resp.text[:500]}"
                    )
                return {output_key: result}
            except ValueError:
                # Re-raise HTTP error responses so the step is marked failed.
                raise
            except Exception as exc:
                logger.exception("[%s] step '%s' http_call failed", graph_id, step_id)
                return {output_key: {"error": str(exc)}}

        return node

    def _python_node(self, step: dict[str, Any]):
        """Execute inline Python code.

        The code runs with a ``state`` dict injected as a local variable so
        that any state value can be read via ``state["key"]``.  The code
        should assign an ``output`` variable; its value is stored under
        ``output_key`` (defaults to the step id).

        The step runs in a thread-pool executor to avoid blocking the event
        loop.  Standard-library imports are available; builtins are not
        restricted (the workflow is trusted infrastructure code).
        """
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            code = step.get("code", "")
            output_key = step.get("output_key") or step_id

            logger.info("[%s] step '%s' running (python)", graph_id, step_id)
            try:
                local_vars: dict[str, Any] = {"state": dict(state), "output": None}
                compiled = compile(code, f"<workflow:{graph_id}:{step_id}>", "exec")

                def _run() -> None:
                    exec(compiled, {"__builtins__": __builtins__}, local_vars)  # noqa: S102

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _run)
                result = local_vars.get("output")
                logger.info("[%s] step '%s' finished", graph_id, step_id)
                return {output_key: result}
            except Exception as exc:
                logger.exception("[%s] step '%s' python failed", graph_id, step_id)
                return {output_key: {"error": str(exc)}}

        return node

    @staticmethod
    def _parallel_node(step: dict[str, Any]) -> Callable:
        max_parallel: int | None = step.get("max_parallel")
        step_id = step["id"]

        async def node(state: dict) -> dict:
            if max_parallel:
                # Store the limit in state so branch steps can read it via
                # _PARALLEL_LIMIT_KEY if they choose to enforce concurrency.
                return {f"_parallel_limit_{step_id}": max_parallel}
            return {}
        return node

    @staticmethod
    def _join_node(step: dict[str, Any]) -> Callable:
        max_timeout: float | None = (
            float(step["max_timeout"]) if step.get("max_timeout") else None
        )
        step_id = step["id"]
        failure_policy: str = step.get("failure_policy", "and")

        async def node(state: dict) -> dict:
            # Check if any parallel branches recorded a timeout sentinel.
            if max_timeout:
                started_at = state.get(f"_parallel_started_{step_id}")
                if started_at:
                    import time
                    elapsed = time.monotonic() - float(started_at)
                    if elapsed > max_timeout:
                        raise TimeoutError(
                            f"Join '{step_id}' timed out after {elapsed:.1f}s "
                            f"(max_timeout={max_timeout}s)"
                        )
            # Apply branch failure policy.
            failed = state.get("__failed_step__")
            if failed:
                if failure_policy == "and":
                    # AND (default): any branch failure fails the join.
                    return {"__failed_step__": step_id, "error": f"branch failed (AND policy): {failed}"}
                # OR: proceed if at least one branch succeeded.
                # TODO: full OR tracking requires counting successful branches.
                pass
            return {}
        return node

    @staticmethod
    def _switch_node(step: dict[str, Any]) -> Callable:
        async def node(state: dict) -> dict:
            return {}
        return node

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _MAX_TOOL_RESULT_CHARS = 4_000

    # MIME type prefixes whose content should be decoded and passed to the LLM.
    # Everything else (images, PDFs, office docs, …) stays as a placeholder.
    _TEXT_MIME_PREFIXES = ("text/",)

    @staticmethod
    def _extract_mcp_text(result: Any) -> str:
        """Extract plain text from an MCP tool result.

        langchain_mcp_adapters returns content as a list of typed content blocks.
        - text blocks: included as-is.
        - file blocks with a text/* MIME type (e.g. text/html, text/plain): the
          base64-encoded ``data`` field is decoded and included so that e.g. Jira
          HTML attachments reach the LLM as readable content.
        - file blocks with binary MIME types: replaced with a short placeholder.
        The final string is capped at _MAX_TOOL_RESULT_CHARS to prevent context overflow.
        """
        # Only treat the list as MCP content blocks when every dict item carries
        # a recognised "type" field ("text" or "file").  Plain data lists (e.g.
        # mock tool returns in tests) fall through to the str() path unchanged.
        if (
            isinstance(result, list)
            and result
            and all(isinstance(item, dict) and item.get("type") in ("text", "file") for item in result)
        ):
            parts: list[str] = []
            for item in result:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:  # file
                    mime = item.get("mime_type", "unknown")
                    is_text_mime = any(
                        mime.startswith(prefix)
                        for prefix in YamlGraphRunner._TEXT_MIME_PREFIXES
                    )
                    if is_text_mime:
                        raw = item.get("data", "") or item.get("text", "")
                        if raw:
                            try:
                                decoded = base64.b64decode(raw).decode("utf-8", errors="replace")
                            except Exception:
                                decoded = raw  # already plain text, not base64
                            parts.append(f"[attachment: {mime}]\n{decoded}")
                        else:
                            parts.append(f"[attachment: {mime} — no content]")
                    else:
                        parts.append(f"[binary file attachment: {mime}]")
            content = "\n".join(parts)
        else:
            content = str(result)

        if len(content) > YamlGraphRunner._MAX_TOOL_RESULT_CHARS:
            kept = YamlGraphRunner._MAX_TOOL_RESULT_CHARS
            content = content[:kept] + f"\n[truncated — {len(content) - kept} chars omitted]"
        return content

    @staticmethod
    def _render(template: str, state: dict) -> str:
        """Render a {key} template against state; missing keys render as empty string.

        Supports {env.VAR_NAME} and {env[VAR_NAME]} to read environment variables.
        Chained access like {obj[key1][key2]} renders as empty string when any level is missing.
        """
        class _EnvAccessor:
            def __getattr__(self, name: str) -> str:
                return os.environ.get(name, "")
            def __getitem__(self, name: str) -> str:
                return os.environ.get(name, "")

        class _Safe:
            """Returned for missing keys; silently absorbs further attribute/item access."""
            def __getattr__(self, name: str) -> "_Safe":
                return _Safe()
            def __getitem__(self, key: object) -> "_Safe":
                return _Safe()
            def __format__(self, fmt: str) -> str:
                return ""
            def __str__(self) -> str:
                return ""

        class _DefaultDict(dict):
            def __missing__(self, key: str) -> "_Safe":
                return _Safe()

        d = _DefaultDict(state)
        d["env"] = _EnvAccessor()
        try:
            return string.Formatter().vformat(template, [], d)  # type: ignore[arg-type]
        except ValueError:
            return template

    @classmethod
    def _render_deep(cls, value: Any, state: dict) -> Any:
        """Recursively render {key} templates in dicts, lists, and strings."""
        if isinstance(value, str):
            return cls._render(value, state)
        if isinstance(value, dict):
            return {k: cls._render_deep(v, state) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._render_deep(item, state) for item in value]
        return value

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
