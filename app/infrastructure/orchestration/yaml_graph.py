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
from app.domain.models.datastream import as_data_ref, find_data_refs
from app.infrastructure.notifications.webhook_notifier import send_approval_notification

logger = logging.getLogger(__name__)
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from app.application import data_artifacts
from app.application.step_normalization import normalize_edges
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


def _coerce_usage_num(value: Any) -> int | float:
    """Coerce a token-usage value to a number; non-numeric values become 0
    instead of raising, so a single malformed field can't crash the reducer."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


def _sum_usage(a: Any, b: Any) -> Any:
    """Reducer that sums numeric token-usage dicts across re-executions
    (e.g. workflow-loop steps that run the same node multiple times), instead
    of letting the last write clobber earlier usage.

    Must never raise — a raising reducer would crash the graph superstep.
    Non-dict inputs (e.g. a legacy string value written before this reducer
    existed) and non-numeric field values are tolerated defensively."""
    if not b:
        return a
    if not a:
        return b
    if not isinstance(a, dict) or not isinstance(b, dict):
        if isinstance(b, dict):
            return b
        if isinstance(a, dict):
            return a
        return {}
    keys = set(a) | set(b)
    return {key: _coerce_usage_num(a.get(key)) + _coerce_usage_num(b.get(key)) for key in keys}


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
        # Append-only audit trail of every human_approval decision. Each approval
        # node reads the current list, appends its record, and returns the full
        # list, so a plain last-wins field is correct (nodes run sequentially).
        "approval_history": list,
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
        # Quality-gate rejection fields — written by _agent_node when meta-LLM
        # or output-mapping validation fails.  Must be in the schema or LangGraph
        # silently drops them from the update stream before step_outputs is written.
        "_meta_llm_rejection":        Annotated[Any, _last_wins],    # type: ignore[assignment]
        "_meta_llm_result":           Annotated[Any, _last_wins],    # type: ignore[assignment]
        # Id of the approval case the last destructive data_source step went
        # through. Declared or LangGraph drops it, and the run would then carry
        # no trace of which case let the deletion happen.
        "_approval_case_id":          Annotated[Any, _last_wins],    # type: ignore[assignment]
        "error":                      Annotated[Any, _last_wins],    # type: ignore[assignment]
        # Backend judge (meta-LLM) token usage, summed across the whole workflow
        # run — a single global bucket, kept separate from per-step agent/meta usage.
        "_judge_token_usage":         Annotated[Any, _sum_usage],    # type: ignore[assignment]
    }
    for step in steps:
        # proceed_or keeps its first-wins latch in state. It must be declared or
        # LangGraph drops it, and the node would then re-win on every arrival.
        # _merge_dicts (not _last_wins) so a branch that writes the latch in the
        # same superstep as another key cannot blank the counter.
        if step.get("type") == "proceed_or":
            fields[f"_proceed_or_{step['id']}"] = Annotated[Any, _merge_dicts]  # type: ignore[assignment]
        # Regular output nodes store their result under output_key
        if "output_key" in step:
            fields[step["output_key"]] = Any  # type: ignore[assignment]
        # http trigger carries the raw webhook body; cron trigger carries schedule metadata
        if step.get("type") == "http":
            fields["trigger_payload"] = Any  # type: ignore[assignment]
        if step.get("type") == "cron":
            fields["trigger_info"] = Any  # type: ignore[assignment]
        # pubsub trigger carries both: the event body and its delivery metadata
        if step.get("type") == "pubsub":
            fields["trigger_payload"] = Any  # type: ignore[assignment]
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
        # agent steps (langgraph-agent / claude-agent) store their output under
        # output_key; output_mapping additionally lifts individual agent keys
        # into state under names of their own. The two are independent, and a
        # step may declare both — output_key for the whole payload,
        # output_mapping for the fields the rest of the graph routes on. Every
        # mapped name must be declared here regardless, or LangGraph's state
        # merge drops it: the step reports "finished" and its outputs silently
        # never arrive. Gating this on `output_key` being absent made declaring
        # both the one combination that quietly lost the mapped fields.
        if step.get("type") in ("langgraph-agent", "claude-agent"):
            if step.get("output_mapping"):
                for wf_key in step["output_mapping"].values():
                    fields[wf_key] = Any  # type: ignore[assignment]
            if "output_key" not in step and not step.get("output_mapping"):
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
            # Agent-LLM and post-compact meta-LLM token usage are tracked as two
            # separate buckets, summed across workflow-loop re-executions of this
            # step (never merged into one another).
            fields[f"_agent_token_usage_{step['id']}"] = Annotated[Any, _sum_usage]  # type: ignore[assignment]
            fields[f"_meta_token_usage_{step['id']}"] = Annotated[Any, _sum_usage]  # type: ignore[assignment]
            # Live progress trail + in-flight token usage, written directly by
            # POST /agent/progress (see agent_callbacks.py) and scoped per step
            # via run.current_step. Must be declared here too, or LangGraph's
            # state merge silently drops them on the next node transition —
            # they'd flash briefly during live polling then vanish once the
            # step finished, which is exactly what was observed in the UI.
            fields[f"_agent_progress_{step['id']}"] = Any  # type: ignore[assignment]
            fields[f"_live_token_usage_{step['id']}"] = Any  # type: ignore[assignment]
    # Internal field: agent_url stored while a run is in waiting_agent state
    fields["_agent_url"] = Any  # type: ignore[assignment]
    # Internal field: clarification answers from ask_context interrupt, forwarded to agent re-run
    fields["_clarification_answers"] = Any  # type: ignore[assignment]
    return TypedDict("YamlGraphState", fields, total=False)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Shared graph streaming helper (used by workflow steps and default_workflow)
# ---------------------------------------------------------------------------

# State keys that are written to Mongo directly (POST /agent/progress and
# poll-forwarding) and never flow through the LangGraph checkpoint. A run.state
# rebuild from a graph snapshot would clobber them, so they are re-overlaid.
_OUT_OF_BAND_STATE_PREFIXES: tuple[str, ...] = ("_agent_progress_",)


async def merge_out_of_band_state(run_repository, run_id: str, merged: dict) -> dict:
    """Re-overlay Mongo-only keys (written by POST /agent/progress and
    poll-forwarding, never through LangGraph) so a run.state rebuild from
    the graph snapshot can't clobber them. Non-empty fresh values win.
    Best-effort: never raises (must not mask the original error in
    failure-path callers)."""
    if run_repository is None:
        return merged
    try:
        fresh = await run_repository.get(run_id)
        fresh_state = getattr(fresh, "state", None) if fresh is not None else None
        if isinstance(fresh_state, dict):
            for k, v in fresh_state.items():
                if v and any(k.startswith(p) for p in _OUT_OF_BAND_STATE_PREFIXES):
                    merged[k] = v
    except Exception as _e:
        logger.warning("merge_out_of_band_state: run %s: %s", run_id, _e)
    return merged


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


# Step types whose *successful* update is the empty dict. A `data` step records
# what it names out of band and returns state byte-identical on purpose, so
# inferring "skipped" from an empty update would show every working data node
# as one that never ran.
_EMPTY_UPDATE_IS_FINISHED: frozenset = frozenset({"data"})


def step_status_from_output(
    node_name: str, output: Any, *, step_type: str | None = None, ran: bool = False
) -> str:
    """Infer step status from the dict a node returned.

    Empty dict → ``skipped``, except for the types in
    ``_EMPTY_UPDATE_IS_FINISHED`` that ran. A `when` guard also produces an
    empty update, and that *is* a real skip; ``ran`` tells the two apart,
    because the status hook publishes "running" only from inside the node,
    inside the guard.

    If the output carries a ``__failed_step__`` sentinel matching this node, it
    caught an internal exception and chose to record the error in state —
    surface that as ``failed`` so the UI doesn't show a green checkmark over a
    captured failure. Anything else → ``finished``.
    """
    if not output:
        if ran and step_type in _EMPTY_UPDATE_IS_FINISHED:
            return "finished"
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


_PROBE_IGNORED_KEYS = frozenset({"__failed_step__"})


def _probe_repr(value: Any) -> str:
    """Short, never-raising repr for probe logging."""
    try:
        text = repr(value)
    except Exception:
        return f"<unreprable {type(value).__name__}>"
    return text if len(text) <= 200 else text[:197] + "..."


async def _probe_state_divergence(runner: Any, run_id: str, node_name: str,
                                  current_state: dict, config: dict) -> None:
    """Log every key where the runner's hand-merged state disagrees with the
    checkpoint's own reducer-applied values.

    Phase 1 of collapsing the dual state: `stream_graph_to_pause` rebuilds
    state by `dict.update()`-ing each chunk's output, which silently
    reimplements LangGraph's reducers. Any field with a non-last-wins reducer
    (``_sum_usage``, ``_merge_dicts``) can therefore drift. This measures the
    drift instead of guessing at it.

    Purely diagnostic and best-effort: it must never raise, and never writes.
    """
    try:
        snap = await runner.graph.aget_state(config)
    except Exception as exc:
        logger.debug("state probe: run %s: aget_state failed: %s", run_id, exc)
        return
    checkpoint = dict(snap.values) if snap and snap.values else {}
    for key in sorted(set(checkpoint) | set(current_state)):
        if key in _PROBE_IGNORED_KEYS:
            continue
        in_ckpt = key in checkpoint
        in_local = key in current_state
        try:
            same = in_ckpt and in_local and checkpoint[key] == current_state[key]
        except Exception:
            # Values that refuse comparison are not evidence of drift.
            continue
        if same:
            continue
        if not in_local:
            # The checkpoint knows a key the runner never saw — the reducer
            # produced it, or another writer did.
            logger.warning(
                "state probe: run %s node %s key %r MISSING-LOCALLY checkpoint=%s",
                run_id, node_name, key, _probe_repr(checkpoint[key]),
            )
        elif not in_ckpt:
            logger.warning(
                "state probe: run %s node %s key %r MISSING-IN-CHECKPOINT local=%s",
                run_id, node_name, key, _probe_repr(current_state[key]),
            )
        else:
            logger.warning(
                "state probe: run %s node %s key %r DIVERGED local=%s checkpoint=%s",
                run_id, node_name, key,
                _probe_repr(current_state[key]), _probe_repr(checkpoint[key]),
            )


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

    # Restart resilience: if the checkpoint carries a stale __failed_step__
    # sentinel, check the routing_log to see if the last routing decision
    # targeted that very node (= router decided to loop back to retry it).
    # In that case the sentinel is from a previous loop iteration and should
    # be cleared so the fail-fast guard doesn't block legitimate execution.
    _stale_failed = current_state.get("__failed_step__")
    if _stale_failed and run.routing_log:
        _last_route = next(
            (e for e in reversed(run.routing_log) if e.event == "route"),
            None,
        )
        if _last_route is not None and _last_route.target == _stale_failed:
            logger.info(
                "[%s] resume: clearing stale __failed_step__=%r — last route went TO that step (loop-back)",
                run.id, _stale_failed,
            )
            current_state["__failed_step__"] = None
            # Also patch the LangGraph checkpoint so astream sees the cleared value.
            if not isinstance(input_value, dict):
                input_value = {"__failed_step__": None}

    last_processed: str | None = None
    _stream_interrupt_output: list | None = None  # payload from __interrupt__ chunk if seen
    try:
        from app.core.config import get_settings as _get_settings_for_probe
        _probe_enabled = bool(_get_settings_for_probe().state_divergence_probe)
    except Exception:
        _probe_enabled = False
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
                _node_def = next(
                    (s for s in runner.steps if s["id"] == node_name), None
                )
                status = step_status_from_output(
                    node_name, output,
                    step_type=(_node_def or {}).get("type"),
                    # Read before it is overwritten below: the status hook set
                    # it from inside the node, so "running" is the evidence
                    # that the node body actually executed.
                    ran=run.step_statuses.get(node_name) == "running",
                )
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
                if _probe_enabled:
                    await _probe_state_divergence(
                        runner, run.id, node_name, current_state, config,
                    )
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
        # Pull the checkpointer's reducer-applied values (e.g. _sum_usage) so a
        # failure never regresses to the hand-rolled `current_state` overwrite,
        # which would undercount token-usage fields accumulated across loop
        # re-executions. aget_state must never raise here — a failure handler
        # that raises would swallow the original error.
        try:
            _fail_snap = await runner.graph.aget_state(config)
            _checkpoint_state = dict(_fail_snap.values) if _fail_snap and _fail_snap.values else {}
        except Exception:
            _checkpoint_state = {}
        run.state = await merge_out_of_band_state(
            run_repository, run.id,
            {**current_state, **mid_run, **_checkpoint_state, "error": error_msg},
        )
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
            # A data_source step pauses only to have a deletion approved, and
            # unlike human_approval — whose node *is* the gate — it is a
            # working step that will go on to run. Mark it waiting_approval so
            # the canvas shows the halted node in amber rather than leaving it
            # "running" while nothing happens.
            if step_type == "data_source" and current_step_id in run.step_statuses:
                run.step_statuses[current_step_id] = "waiting_approval"
    else:
        run.status = "completed"
    run.current_step = _effective_next
    run.state = await merge_out_of_band_state(run_repository, run.id, dict(snap.values))
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
        # that raised an ask_context interrupt internally via meta-LLM — but only
        # when a Slack addon (slack_notifications) is attached to the agent step.
        # Without it, copilot_ui still surfaces the request for input; Slack is
        # simply not involved.
        is_agent_ask_context = (
            step and step.get("type") in ("langgraph-agent", "claude-agent")
            and active_interrupt_type == "ask_context"
            and step.get("slack_notifications")
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


def _approval_interrupt_payload(case: Any) -> dict[str, Any]:
    """What copilot_ui and the Slack message read off a paused deletion.

    Deliberately flat and self-contained: the panel that renders it must not
    have to fetch the case to know what is about to be deleted, because the
    number in ``affected_rows`` is the one thing the reviewer has to see before
    the buttons.
    """
    verdict = case.meta_llm
    return {
        "type": "datasource_approval",
        "case_id": case.id,
        "datasource_id": case.datasource_id,
        "datasource_name": case.datasource_name,
        "operation": case.operation,
        "method": case.method,
        "endpoint": case.endpoint,
        "affected_rows": case.affected_rows,
        "affected_sample": case.affected_sample,
        "targets": case.targets,
        "params": case.params,
        "meta_llm": verdict.model_dump(mode="json") if verdict is not None else None,
    }


# ---------------------------------------------------------------------------
# YAML graph runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Spilled results in string templates
# ---------------------------------------------------------------------------

class _StreamRef:
    """Stands in for a spilled result inside a ``{key}`` template.

    A handle is a small dict, so interpolating it would not blow anything up
    -- it would just paste ``{'__stream__': 1, 'id': 'sp_...'}`` into a prompt
    or a URL, which is worse than useless because it looks like data. This
    renders the handle's one-line summary instead, while still allowing
    ``{result[items]}`` and ``{result.items}`` for a template that wants the
    count, so a prompt can say "1.2M rows" without ever seeing a row.
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def __getitem__(self, key: Any) -> Any:
        try:
            return getattr(self._handle, str(key))
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __format__(self, fmt: str) -> str:
        return self._handle.summary()

    def __str__(self) -> str:
        return self._handle.summary()

    def __repr__(self) -> str:
        return f"_StreamRef({self._handle.id})"


def _stream_sample_block(ref: Any, sample: list[Any]) -> str:
    """The stated-sample text appended to a prompt for a streamed input.

    The count comes first and the records second, and it says plainly that
    they are a sample.  A model shown five records with no indication that
    there are 400,000 answers as though five is the whole set -- the single
    most dangerous way for this to go wrong.
    """
    import json as _json

    body = _json.dumps(sample, indent=2, default=str)[:6000]
    caveat = (
        " The read was truncated, so even that count is a lower bound."
        if ref.truncated
        else ""
    )
    origin = f" from {ref.source_id}.{ref.operation}" if ref.source_id else ""
    return (
        f"DATA: {ref.items} records are available{origin}, totalling "
        f"{ref.bytes} bytes. They are NOT included here -- far more than fits "
        f"in this prompt.{caveat}\n"
        f"Below are the first {len(sample)} records, as a sample of the shape "
        f"only. Do not treat them as the whole set, do not count them, and do "
        f"not draw totals from them; use the stated count of {ref.items} for "
        f"anything quantitative.\n"
        f"SAMPLE:\n{body}"
    )


def _stream_safe_state(state: dict) -> dict:
    """*state* with every spill handle wrapped so templates cannot inline it.

    Copies only when there is something to wrap, so the ordinary case pays
    one dict scan and no allocation.
    """
    handles = find_data_refs(state)
    if not handles:
        return state
    wrapped = dict(state)
    for key, handle in handles.items():
        wrapped[key] = _StreamRef(handle)
    return wrapped


class YamlGraphRunner:
    """
    Builds a compiled LangGraph from a plain dict parsed from a YAML file.

    YAML schema (all fields except ``id`` and ``steps`` are optional):

        id: dev-assistant
        description: "..."
        steps:
          - id: <node-id>
            type: llm | mcp | human_approval | execute | workflow | cron | http | http_call | python | data_source | parallel | join | proceed_or | switch | storage | slack
            when: <state-key>          # skip node if state[key] is falsy
            system_prompt: "..."       # llm
            user_template: "..."       # {key} placeholders resolved from state
            output_key: <key>          # where to store the result
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
            schedule: "0 9 * * 1-5"   # cron only — 5-field cron expression
            timezone: Europe/Berlin    # cron only — zone the schedule is read in;
                                       #   defaults to UTC. Name the zone for any
                                       #   "09:00 local" schedule: a fixed UTC hour
                                       #   is an hour off for half the year wherever
                                       #   DST applies.
            request_template: "..."    # cron only — initial request; supports {now}, {date}
            url: "https://..."         # http_call only — endpoint; {key} templates resolved
            method: POST               # http_call only — GET | POST | PUT | PATCH | DELETE
            headers:                   # http_call only — request headers; values support {key}
              Authorization: "Bearer {token}"
            auth: service_identity     # http_call only — attach the service's own
                                       #   OAuth2 access token as a bearer header
            auth_identity: afp         # http_call only — which configured service
                                       #   identity to use; omit for the default
            body:                      # http_call only — JSON body; values support {key}
              issue_key: "{ticket_id}"
            code: |                    # python only — executed with ``state`` dict in scope;
              output = state["x"] + 1  #   set ``output`` variable to store the result
            script_id: my-script       # python only — run a library script instead
                                       #   of the inline ``code`` above
            sandbox: true              # python only — default true: run isolated,
                                       #   with no access to the backend's env vars,
                                       #   tools, bash or installed libraries
            sandbox_runtime: local     # python only — local | docker | k8s
            sandbox_image: python:3.12-slim  # python only — docker/k8s image override
            timeout_seconds: 60        # python only — sandbox wall-clock limit
            action: get                # storage only -- get | set | delete | keys
            key: "alert-state"         # storage only -- entry name; {state} templated
            value: {}                  # storage only (set) -- any JSON; {state} templated
            action: post               # slack only -- post | reply | history | thread | dm | delete
            provider: slack            # slack only -- messaging provider name (default slack)
            channel: "C0BLDDSEB1D"     # slack only -- channel id; {state} templated
            text: "{digest}"           # slack only (post|reply|dm) -- message body
            thread_id: "{ts}"          # slack only (reply|thread) -- root message id
            user_id: "{owner}"         # slack only (dm) -- DM recipient
            message_id: "{ts}"         # slack only (delete) -- message to delete
            oldest: "1787600000"       # slack only (history) -- lower time bound
            limit: 200                 # slack only (history) -- max messages
            items: overrides.confirmations   # slack only (reply) -- state path to a
                                       #   list of {thread_id, text} to post in one go
            skip_if_replied: true      # slack only (reply) -- do not post a reply whose
                                       #   text is already in the thread (idempotency)
            ignore_errors: true        # slack only -- capture provider errors under
                                       #   output_key instead of failing the run
            source: github             # data_source only — DataSourceDefinition id
            operation: list_repos      # data_source only — operation to invoke
            result_mode: auto          # data_source only — auto (default)
                                       #   leaves the stream reference in
                                       #   state; ram loads the records back
                                       #   in, and fails if they will not fit
            stream: contacts           # python | llm — the state key holding
                                       #   the data_source result to read
            sample_records: 5          # llm — how many records go in the prompt
            stream_mode: sample        # llm — sample (default) states the count
                                       #   and shows a few records; map_reduce
                                       #   runs the prompt per chunk then once
                                       #   over the answers
            chunk_items: 5000          # llm + map_reduce — records per chunk
            chunk_bytes: 524288        # llm + map_reduce — bytes per chunk
            max_chunks: 200            # llm + map_reduce — stop after N chunks
            chunk_template: ...        # llm + map_reduce — per-chunk prompt,
                                       #   gets {chunk} {chunk_index} {total_items}
            reduce_template: ...       # llm + map_reduce — combining prompt,
                                       #   gets {parts} {total_items}
            params:                    # data_source only — operation inputs;
              owner: "{repo_owner}"    #   values support {key} templates
            routes:                    # switch / langgraph-agent /
                                       #   claude-agent — multiple branches
              - when: <state-key>      # route taken when state[key] is truthy
                next: <node-id>
                wait_seconds: 60       # optional — sleep before the next node runs
                                       #   (capped at 3600s; useful for retry back-edges)

            ``routes`` is mutually exclusive with ``next``: an agent step (or
            switch) declares one or the other, not both. For agent steps the
            route conditions are evaluated against state *after* the agent's
            ``output_mapping`` has been merged in, so a route can reference
            any field the agent's output was mapped onto (e.g. a verdict
            field). Exactly one route should omit ``when`` to act as the
            default fallback; if no condition matches and there is no
            default route, the run fails. This per-route ``when`` is
            evaluated only to pick the next node and is distinct from the
            step-level ``when`` field described above, which instead decides
            whether the step runs at all (skip guard).

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

    Data source results
    -------------------
    A ``data_source`` step never puts its result in state.  The executor
    writes every result -- one record or four million -- to the data stream
    store and the step leaves a small ``DataRef`` behind (see
    ``app.infrastructure.datasources.datastream``).  State, the LangGraph
    checkpoint and the Mongo run document are therefore never functions of
    result size.

    One path, unconditionally, because the alternative -- inline when small,
    a file when large -- means every consumer handles both shapes and every
    workflow behaves differently in production than in test depending on how
    much data happened to come back.

    Consumers read the file:

    * ``python`` with ``stream: <key>`` gets ``records()``, a generator over an
      already-open descriptor, plus ``stream`` (the raw file object) and
      ``stream_records`` (the count, known up front).  It can iterate twice;
      it holds one record at a time.  The descriptor is opened for it before
      the seccomp filter denies ``openat``, so the sandbox is no weaker --
      the script still cannot open a path of its own, including that file.
    * ``data_source`` fan-out streams a referenced upstream, one request per
      record.  Nothing to configure.
    * ``llm`` with ``stream: <key>`` is the one consumer that cannot read a
      descriptor: bytes have to become tokens in a context window, so this
      step reads the file on the model's behalf and puts a bounded selection
      in the prompt (``stream_mode: sample`` or ``map_reduce``).  Reducing
      first with a ``python`` step is cheaper and exact -- prefer it.
    * Any ``{key}`` template renders a reference as a one-line summary, never
      as data.  ``{key[items]}`` reads the count.
    * ``result_mode: ram`` on the ``data_source`` step is the escape hatch for
      a workflow that needs values inline -- a route condition on a field, an
      ``http_call`` body built from the records.  It refuses past
      ``STREAM_READ_ALL_MAX_BYTES`` rather than half-loading.

    Where the sandbox runs decides how the file gets there: ``local`` and
    ``docker`` are handed the path (docker bind-mounts it read-only), while
    ``k8s`` is a different pod with no network and no shared filesystem, so
    the backend streams the bytes into its stdin and it writes its own copy.

    ``data_source`` steps invoke one operation of a registered    ``data_source`` steps invoke one operation of a registered
    ``DataSourceDefinition``.  Upstream operations of the source's DAG are
    resolved by the executor, so the step only supplies the operation's own
    ``params``.  The result is stored under ``output_key`` (defaults to the
    step id); failures are captured as ``{"error": "..."}`` so the next node
    can decide how to react.

    ``join`` and ``proceed_or`` are the two fan-in operators.  ``join`` is an AND:
    it runs after every branch, and ``failure_policy`` decides whether one failed
    branch fails the join.  ``proceed_or`` is an OR: the first branch to arrive
    wins, and every later arrival is routed to END so the steps after it run
    exactly once instead of once per arrival.

    ``proceed_or`` is about *what runs*, not about wall clock.  LangGraph
    supersteps are globally synchronous, so while any branch is still producing
    supersteps the steps after the fan-in wait for them; the OR does not let the
    run overtake a slow branch.  What it does give you is a tail that executes
    once, on the first branch's data, without needing every branch to arrive or
    succeed.

    ``storage`` steps read and write the workflow's own key/value storage, which
    survives between runs -- the place for "which alerts have I already sent",
    not for datasets. It must be switched on for the workflow (``use_storage``);
    a ``storage`` step in a workflow that has it off fails loudly rather than
    silently doing nothing. The owning workflow id is taken from the runner, so a
    step can only ever address its own entries. ``get`` on an absent key yields
    ``None``, because a first run legitimately has no state yet.

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
        # A disabled workflow starts no runs; see app.application.run_control.
        # Read off the definition (not post-construction like readonly) so runners
        # built straight from a YAML dict carry the flag too. Absent means enabled.
        self.enabled: bool = bool(definition.get("enabled", True))
        # Resolve implicit fall-through into an explicit `next`/`targets` up
        # front, so array position stops being load-bearing below this line.
        # Every construction path — stored definition, on-disk YAML, a raw dict
        # in a test — arrives here, so this is the one place that has to do it.
        # Behaviour is unchanged: normalisation writes down the destination the
        # edge builder would have picked anyway, which is why the positional
        # branches further down survive as an unreachable belt-and-braces.
        self._steps: list[dict[str, Any]] = normalize_edges(definition["steps"])
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
        # Injected post-construction for `storage` steps (optional). The backend
        # is shared, the scoping is not: every call passes self.id as the owner,
        # so a step cannot reach another workflow's keys.
        self._storage_backend: Any = None
        self._storage_enabled: bool = bool(definition.get("use_storage", False))
        # Injected post-construction for `data_source` steps (optional)
        self._data_source_backend: Any = None
        self._data_source_executor: Any = None
        # Injected post-construction: reads back a data source result the
        # executor spilled to disk because it was too large for state (see
        # app.infrastructure.datasources.datastream).  Optional — without it a
        # `data_source` step still runs, it just cannot spill, so an oversized
        # result fails the step instead.
        self._stream_store: Any = None
        # Injected post-construction for `data` steps: the run download
        # manifest.  Optional -- without it a `data` step still runs and still
        # returns state unchanged, it simply records nothing (and says so).
        self._data_artifact_backend: Any = None
        # Injected post-construction: the privilege gate that holds a
        # destructive data-source operation until somebody approves it.
        # Optional — without it a `data_source` step runs a DELETE unattended,
        # which is the pre-gate behaviour.
        self._approval_service: Any = None
        # Injected post-construction for `python` steps that reference a
        # library script via `script_id` (optional — inline `code` still works
        # without it).
        self._script_backend: Any = None
        # Injected post-construction for `http_call` steps that use
        # `auth: service_identity` (optional — falls back to the process-wide
        # provider built from settings).
        self._service_token_provider: Any = None
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

        _MULTI_OUTPUT_TYPES = frozenset(
            {"switch", "langgraph-agent", "claude-agent", "human_approval"}
        )

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

            # proceed_or: first arrival continues, later ones are cut to END so
            # the tail of the graph cannot run twice on an uneven-depth fan-in.
            if step_type == "proceed_or":
                dest = next_val or (routes[0].get("next") if routes else None)
                if not dest or dest not in all_ids:
                    dest = step_ids[i + 1] if i < len(self._steps) - 1 else None
                if not dest or dest not in all_ids:
                    sg.add_edge(sid, END)
                else:
                    sg.add_conditional_edges(
                        sid,
                        self._make_proceed_or_router(sid, dest),
                        {dest: dest, END: END},
                    )
                continue

            if routes:
                if step_type not in _MULTI_OUTPUT_TYPES and len(routes) > 1:
                    raise ValueError(
                        f"Step '{sid}' (type={step_type}) cannot have more than "
                        f"1 route; only switch, langgraph-agent, and "
                        f"claude-agent support multiple routes."
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
            raise ValueError(
                f"Step '{step['id']}' in graph '{self.id}' uses the removed step "
                f"type 'llm_structured'. Replace it with 'langgraph-agent' or "
                f"'claude-agent'; the structured `output` spec becomes the agent "
                f"step's `output_mapping`."
            )
        if t in ("langgraph-agent", "claude-agent"):
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
        elif t == "pubsub":
            fn = self._pubsub_trigger_node(step)
        elif t == "http_call":
            fn = self._http_call_node(step)
        elif t == "data_source":
            fn = self._data_source_node(step)
        elif t == "data":
            fn = self._data_node(step)
        elif t == "python":
            fn = self._python_node(step)
        elif t == "parallel":
            fn = self._parallel_node(step)
        elif t == "join":
            fn = self._join_node(step)
        elif t == "proceed_or":
            fn = self._proceed_or_node(step)
        elif t == "storage":
            fn = self._storage_node(step)
        elif t == "slack":
            fn = self._messaging_node(step)
        elif t == "switch":
            fn = self._switch_node(step)
        else:
            raise ValueError(f"Unknown step type '{t}' in graph '{self.id}'")
        wrapped = self._wrap_with_status_running(self._wrap_with_loop_guard(step, fn), step)
        wrapped = self._wrap_with_when(step, wrapped)
        return self._wrap_with_fail_guard(step, wrapped)

    _NO_LOOP_GUARD_TYPES: frozenset = frozenset({"ask_context", "human_approval", "cron", "http", "pubsub", "parallel", "join", "proceed_or", "switch", "langgraph-agent", "claude-agent"})
    # join handles __failed_step__ itself via failure_policy; all others must abort when
    # a previous step has already written the sentinel into state.
    _NO_FAIL_GUARD_TYPES: frozenset = frozenset({"join", "proceed_or"})

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
                # Append node-start event to routing_log for restart resilience.
                try:
                    from app.domain.models.graph_run import RoutingEvent
                    run.routing_log.append(RoutingEvent(
                        event="node_start",
                        node=step_id,
                        iteration=int((state.get("_visit_counts") or {}).get(step_id, 0)),
                    ))
                except Exception:
                    pass
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

    def _wrap_with_fail_guard(self, step: dict[str, Any], fn: Callable) -> Callable:
        """Raise immediately when a previous step already set __failed_step__ in state.

        This enforces fail-fast sequential execution: the moment any node writes the
        failure sentinel, all downstream nodes abort and the graph fails.  Parallel
        join nodes are exempt — they aggregate branch results including failures and
        apply their own failure_policy.
        """
        if step.get("type") in self._NO_FAIL_GUARD_TYPES:
            return fn
        step_id = step["id"]
        graph_id = self.id
        is_async = asyncio.iscoroutinefunction(fn)

        async def _guarded(state: dict) -> dict:
            failed = state.get("__failed_step__")
            if failed and failed != step_id:
                # Only abort for a DIFFERENT upstream failure.  When failed == step_id
                # the checkpoint carries this step's own previous failure sentinel —
                # that happens on restart where the run resumes from the failed step
                # itself.  Blocking self-restart would make every retry instant-fail.
                logger.error(
                    "[%s] step '%s' aborted — upstream step '%s' already failed; "
                    "halting graph execution",
                    graph_id, step_id, failed,
                )
                raise RuntimeError(
                    f"step '{step_id}' aborted — upstream step '{failed}' already failed"
                )
            if failed == step_id:
                logger.info(
                    "[%s] step '%s' restarting (own failure sentinel cleared)",
                    graph_id, step_id,
                )
            result = (await fn(state)) if is_async else fn(state)
            # If the step succeeded (didn't set __failed_step__ itself) and the
            # checkpoint still carried a stale sentinel from a previous failure,
            # explicitly clear it so downstream nodes aren't blocked by the ghost.
            if (
                failed
                and isinstance(result, dict)
                and result.get("__failed_step__") is None
                and "__failed_step__" not in result
            ):
                result = {**result, "__failed_step__": None}
            return result

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
            # Persist routing decision so backend restarts can reconstruct graph
            # position and clear stale __failed_step__ sentinels from prior loops.
            _run = runner._current_run
            _repo = runner._current_run_repository
            if _run is not None and _repo is not None:
                try:
                    from app.domain.models.graph_run import RoutingEvent
                    _event = RoutingEvent(
                        event="route",
                        node=source_id,
                        target=chosen["next"],
                        condition=str(chosen.get("when", "")) or "",
                        iteration=int((state.get("_visit_counts") or {}).get(source_id, 0)),
                    )
                    _run.routing_log.append(_event)
                    _run.touch()
                    await _repo.update(_run)
                except Exception as _re:
                    logger.debug("routing_log update failed (non-critical): %s", _re)
            return chosen["next"]
        return router

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
                from app.steps.agent_executor import MetaLLMRejectionError
                if isinstance(_step_exc, MetaLLMRejectionError):
                    # Include the agent's actual extracted output alongside the
                    # rejection so the UI can show what the agent produced and
                    # why meta-LLM rejected it — not just a blank failed step.
                    logger.error(
                        "[%s] step '%s' meta-LLM rejected: %s",
                        graph_id, step_id, _step_exc.reason,
                    )
                    return {
                        "__failed_step__": step_id,
                        "error": str(_step_exc),
                        "_meta_llm_rejection": _step_exc.reason,
                        **_step_exc.mapped_result,
                    }
                logger.error("[%s] step '%s' raised: %s", graph_id, step_id, _step_exc)
                return {"__failed_step__": step_id, "error": str(_step_exc)}

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

                # A data source result is a file. An LLM cannot read a file:
                # bytes have to become tokens inside a context window, so
                # something has to choose which bytes. This step reads the
                # stream on the model's behalf and puts a bounded selection in
                # the prompt. `stream:` names which result, `stream_mode` says
                # how much:
                #
                #   sample (default) -- the record count and the first few
                #       records, read from the file, labelled as a sample.
                #   map_reduce -- the prompt once per chunk read off the file,
                #       then once over the answers. Reads everything, at N+1
                #       calls, and is lossy: the combining pass sees the
                #       answers, never the records.
                #
                # Neither replaces reducing first. A `python` step reading the
                # same file down to what actually needs judgement is cheaper
                # and exact.
                stream_key = step.get("stream") or step.get("over")
                ref = self._ref_in(state, stream_key) if stream_key else None
                if ref is not None:
                    stream_mode = (
                        step.get("stream_mode") or step.get("spill_mode") or "sample"
                    ).lower()
                    if stream_mode == "map_reduce":
                        content = await self._llm_map_reduce(
                            step, llm, system_prompt, state, ref, stream_key,
                        )
                        logger.info("[%s] step '%s' finished", graph_id, step_id)
                        return {output_key: content}
                    if stream_mode != "sample":
                        raise ValueError(
                            f"step '{step_id}': unknown stream_mode "
                            f"'{stream_mode}'. Valid values are 'sample' and "
                            f"'map_reduce'."
                        )
                    logger.info(
                        "[%s] step '%s' reading '%s' (%d records) as a sample",
                        graph_id, step_id, stream_key, ref.items,
                    )

                user_message = self._render(step.get("user_template", "{request}"), state)
                if ref is not None:
                    sample = await self._read_sample(ref, step_id, step)
                    user_message = (
                        f"{user_message}\n\n{_stream_sample_block(ref, sample)}"
                    )
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
                # an upstream step may emit str, not list — split on newlines
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
            record = {
                "step_id": step_id,
                "approved": approved,
                "reason": decision.get("reason"),
                "corrections": corrections or None,
                "approver_name": decision.get("approver_name"),
                "approver_id": decision.get("approver_id"),
                "approver_source": decision.get("approver_source"),
                "decided_at": decision.get("decided_at"),
            }
            history = list(state.get("approval_history") or [])
            history.append(record)
            result: dict = {
                approved_key: approved,
                "reject_reason": decision.get("reason"),
                "approval_history": history,
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

            # A workflow starting another workflow is an entry point like any
            # other. Imported inside the node: the guard lives in the
            # application layer, which imports this module.
            from app.application.run_control import (
                WorkflowDisabledError,
                ensure_workflow_enabled,
            )
            try:
                ensure_workflow_enabled(child_runner)
            except WorkflowDisabledError as exc:
                logger.warning("[%s] step '%s': %s", graph_id, step_id, exc.detail)
                return {output_key: {"error": exc.detail}}

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

    async def _auth_headers(self, step: dict[str, Any]) -> dict[str, str]:
        """Resolve a step's ``auth`` mode into outbound request headers.

        Currently only ``service_identity`` is supported; any other value is
        rejected rather than silently ignored so a typo can never downgrade a
        step to an unauthenticated call.  ``auth_identity`` picks which
        configured identity to use when the deployment has several.
        """
        mode = step.get("auth")
        if not mode:
            return {}
        if isinstance(mode, str) and mode.strip().lower() == "service_identity":
            provider = self._service_token_provider
            if provider is None:
                from app.infrastructure.auth.service_token_provider import (
                    get_service_token_provider,
                )
                provider = get_service_token_provider()
            identity = step.get("auth_identity")
            return await provider.get_auth_header(
                identity.strip() if isinstance(identity, str) and identity.strip() else None
            )
        raise ValueError(
            f"Unsupported auth mode '{mode}' — supported: 'service_identity'"
        )

    def _pubsub_trigger_node(self, step: dict[str, Any]):
        """Pass-through node for Pub/Sub-triggered runs.

        The subscriber seeds the state with ``trigger_payload`` (the decoded
        message body) and ``trigger_info`` (topic, subscription, message id,
        publish time, attributes) before the graph starts.  This node republishes
        the payload under ``output_key`` so downstream steps can template it as
        ``{event.field}`` — with ``output_key: event`` — and, exactly like the
        http trigger, fills ``request`` from the payload when the run did not
        come with one.

        Runs started by hand (not by an event) simply see an empty payload, so a
        workflow with a pubsub trigger stays testable from the UI.
        """
        graph_id = self.id
        output_key = step.get("output_key", "trigger_payload")

        async def node(state: dict) -> dict:
            step_id = step["id"]
            payload = state.get("trigger_payload", {})
            info = state.get("trigger_info", {})
            logger.info(
                "[%s] step '%s' running (pubsub trigger, message %s)",
                graph_id, step_id, (info or {}).get("message_id", "-"),
            )
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

        ``auth: service_identity`` adds an ``Authorization: Bearer <token>``
        header carrying the service's own OAuth2 access token, so a step can
        call another service protected by the same authorization server.
        Explicit ``headers`` win over the injected one.
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
                headers = {**await self._auth_headers(step), **headers}
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

    def _data_source_node(self, step: dict[str, Any]):
        """Invoke one operation of a registered data source.

        Config: ``source`` (data source id), ``operation`` (operation name),
        ``params`` (dict of {key}-templated operation inputs) and the usual
        ``output_key``.  The executor resolves any upstream operations the
        operation depends on, so ``params`` only carries the operation's own
        declared inputs.

        An operation that *destroys* — a DELETE, or one flagged ``destructive``
        — does not simply run.  The step first resolves what the call would hit
        without making it, opens an approval case naming the row count, and
        raises a ``datasource_approval`` interrupt.  The run parks at
        ``waiting_approval`` exactly as it does for a ``human_approval`` step,
        so every surface that already resumes an approval — the REST routes,
        the Slack buttons, copilot_ui — resumes this one too, and the deletion
        happens only on the way back.
        """
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            source_id = step.get("source", "")
            operation = step.get("operation", "")
            output_key = step.get("output_key") or step_id

            logger.info(
                "[%s] step '%s' running (data_source %s.%s)",
                graph_id, step_id, source_id, operation,
            )
            try:
                if self._data_source_backend is None or self._data_source_executor is None:
                    raise ValueError("data source backend/executor not configured")
                params = self._render_deep(step.get("params") or {}, state)
                source = await self._data_source_backend.get(source_id)
                if source is None:
                    raise ValueError(f"Data source '{source_id}' not found")

                gate = await self._gate_destructive(step, source, operation, params)
                if gate is not None and not gate["approved"]:
                    logger.info(
                        "[%s] step '%s' data_source deletion refused (case %s): %s",
                        graph_id, step_id, gate["case_id"], gate["reason"],
                    )
                    return {output_key: {
                        "skipped": True,
                        "reason": gate["reason"],
                        "approval_case_id": gate["case_id"],
                        "affected_rows": gate["affected_rows"],
                    }}

                result = await self._data_source_executor.execute(source, operation, params)

                # The executor always returns a reference, never the data.
                # `result_mode` decides what this step leaves in state:
                #
                #   auto (default) -- the reference. Downstream steps read the
                #       file; nothing large is ever checkpointed.
                #   ram -- load the records back into state. The escape hatch
                #       for a workflow that needs the value inline: a route
                #       condition on one of its fields, an `http_call` body
                #       built from it. Fails loudly past
                #       stream_read_all_max_bytes rather than half-loading
                #       into a checkpoint that cannot hold it.
                result_mode = (step.get("result_mode") or "auto").lower()
                ref = as_data_ref(result)
                if ref is not None:
                    if ref.truncated:
                        logger.warning(
                            "[%s] step '%s' result is TRUNCATED (%d records, "
                            "%d bytes) -- downstream steps see a prefix, not "
                            "the whole answer",
                            graph_id, step_id, ref.items, ref.bytes,
                        )
                    if result_mode == "ram":
                        result = await self._load_stream(ref, step_id)
                        logger.info(
                            "[%s] step '%s' loaded stream %s into state "
                            "(result_mode: ram)", graph_id, step_id, ref.id,
                        )
                    else:
                        logger.info(
                            "[%s] step '%s' -> stream %s (%d records, %d bytes)",
                            graph_id, step_id, ref.id, ref.items, ref.bytes,
                        )

                logger.info("[%s] step '%s' finished", graph_id, step_id)
                if gate is not None:
                    return {output_key: result, "_approval_case_id": gate["case_id"]}
                return {output_key: result}
            except GraphBubbleUp:
                # The approval gate suspends the node by raising, and LangGraph
                # needs that exception to reach it. Swallowing it into
                # ``{"error": ...}`` would turn every paused deletion into a
                # step that "failed" and a run that carried on.
                raise
            except Exception as exc:
                logger.exception("[%s] step '%s' data_source failed", graph_id, step_id)
                return {output_key: {"error": str(exc)}}

        return node

    async def _gate_destructive(
        self,
        step: dict[str, Any],
        source: Any,
        operation: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Hold a destructive operation until it is approved.

        Returns ``None`` when the call needs no approval at all — not a
        destructive operation, no gate configured, the gate switched off, or a
        preview that found nothing to delete.  Otherwise returns the verdict:
        ``{"approved": bool, "reason": str, "case_id": str, "affected_rows": int}``.

        The node re-runs from the top every time it resumes from the interrupt,
        so the first thing this does is look for the case it already opened.
        Without that, one paused deletion would write a new case — and re-read
        the whole upstream list — on every resume.
        """
        service = self._approval_service
        if service is None:
            return None
        from app.core.config import get_settings
        if not getattr(get_settings(), "approvals_enabled", True):
            return None

        op = source.get_operation(operation)
        if op is None:
            return None
        from app.infrastructure.datasources.destructive import is_destructive
        if not is_destructive(op, source):
            return None

        step_id = step["id"]
        run = self._current_run
        run_id = run.id if run is not None else ""

        case = await service.find_open_case(run_id, step_id)
        if case is None:
            plan = await self._data_source_executor.preview(source, operation, params)
            # Nothing matched upstream, so nothing is destroyed. Asking a human
            # to approve a no-op only teaches them to click Approve.
            if plan.affected_rows < 1:
                return None
            case = await service.open_case(
                source=source,
                operation=operation,
                method=(op.method or "").upper(),
                params=params,
                affected_rows=plan.affected_rows,
                targets=plan.targets,
                sample=plan.sample,
                workflow_id=self.id,
                run_id=run_id,
                step_id=step_id,
                surface="workflow",
                change_kind=plan.change_kind,
                details=plan.details,
            )

        if case.status == "pending":
            decision: dict = interrupt(_approval_interrupt_payload(case))
            approved = bool(decision.get("approved", False))
            decided = await service.decide(
                case.id,
                approved=approved,
                source=decision.get("approver_source") or "ui",
                decided_by_name=decision.get("approver_name") or "",
                decided_by_id=decision.get("approver_id") or "",
                reason=decision.get("reason") or "",
            )
            # A None here means somebody else closed the case first (the Slack
            # button and the UI button racing). Their answer stands.
            if decided is not None:
                case = decided
            else:
                case = await service._backend.get(case.id) or case
        elif case.veto_deadline is not None:
            # The meta-LLM decided on its own. Announced, not silent: the run
            # holds for the veto window so a person can still stop it.
            case = await service.wait_out_veto(case)

        approved = case.status == "approved"
        reason = case.reason or ("approved" if approved else "rejected")
        return {
            "approved": approved,
            "reason": reason,
            "case_id": case.id,
            "affected_rows": case.affected_rows,
        }

    def _data_node(self, step: dict[str, Any]):
        """Name data mid-workflow so a person can download it afterwards.

        Config: ``selections``, a list of ``{name, from, format}``.  ``from`` is
        a state path (``state.projects``, or plain ``projects``) resolved with
        the same walker every other step's path config uses; ``format`` is
        ``jsonl`` (the stored form), ``json`` or ``csv``.

        The step returns ``{}``.  Not "returns nothing much" -- exactly ``{}``:
        no new keys, no mutation, no transform of the data it points at.  A
        `data` node is an observation point, and one that changed the state it
        observed would make a workflow behave differently depending on whether
        somebody wanted to download something from it.  What it writes goes to
        the run's download manifest, out of band, so state stays
        checkpoint-sized however many selections a workflow exports.

        Where a selection resolves to a ``DataRef`` nothing is copied: the bytes
        are already in the stream store, so the stream is pinned (exempted from
        the short spill TTL) and the manifest entry points at it.

        Nothing here fails a run.  A missing path, an unusable selection, no
        backend configured at all -- each is a logged warning and the node
        carries on, because a run dying at the step that was only meant to
        watch it would be the worst possible trade.
        """
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            selections, problems = data_artifacts.parse_selections(step)
            for problem in problems:
                logger.warning(
                    "[%s] step '%s' (data): %s -- nothing recorded for it",
                    graph_id, step_id, problem,
                )

            run = self._current_run
            run_id = run.id if run is not None else ""
            store = self._stream_store
            backend = self._data_artifact_backend
            if not selections:
                return {}
            if store is None or backend is None or not run_id:
                # Worth a warning rather than silence: the workflow author asked
                # for a download and is not going to get one, and the reason is
                # deployment configuration rather than anything in the YAML.
                logger.warning(
                    "[%s] step '%s' (data): no %s configured, so %d selection(s) "
                    "were not recorded",
                    graph_id, step_id,
                    "run" if not run_id else
                    "data stream store" if store is None else "data artifact backend",
                    len(selections),
                )
                return {}

            ttl = float(
                getattr(self._stream_conf(), "data_artifact_ttl_seconds", 0.0) or 0.0
            )
            for selection in selections:
                path = data_artifacts.strip_state_prefix(selection.path)
                value = self._state_path(state, path)
                if value is None:
                    # Absent and present-but-null are one case here, and both
                    # mean there is nothing to offer. Said out loud, since a
                    # download the user expected will simply not be in the list.
                    logger.warning(
                        "[%s] step '%s' (data): selection '%s' path '%s' resolved "
                        "to nothing -- skipped, no artifact recorded",
                        graph_id, step_id, selection.name, selection.path,
                    )
                    continue
                try:
                    artifact = await data_artifacts.capture_selection(
                        store=store,
                        backend=backend,
                        run_id=run_id,
                        step_id=step_id,
                        selection=selection,
                        value=value,
                        ttl_seconds=ttl,
                    )
                except Exception as exc:  # noqa: BLE001 — see the docstring
                    logger.warning(
                        "[%s] step '%s' (data): selection '%s' could not be "
                        "recorded: %s",
                        graph_id, step_id, selection.name, exc, exc_info=True,
                    )
                    continue
                logger.info(
                    "[%s] step '%s' (data): '%s' -> artifact %s (%s, %d item(s), "
                    "%d bytes)%s",
                    graph_id, step_id, selection.name, artifact.id,
                    artifact.format, artifact.items, artifact.bytes,
                    " TRUNCATED" if artifact.truncated else "",
                )
            return {}

        return node

    async def _resolve_script_code(self, step: dict[str, Any]) -> str:
        """Return the code a ``python`` step should run.

        ``script_id`` points at a ScriptDefinition in the library; when set it
        wins over any inline ``code`` left on the step (the UI keeps the last
        loaded body around so the node still shows something when the library
        is unreachable).
        """
        script_id = step.get("script_id")
        if not script_id:
            return step.get("code", "")
        if self._script_backend is None:
            raise ValueError(
                f"step references script '{script_id}' but no script backend is configured"
            )
        script = await self._script_backend.get(script_id)
        if script is None:
            raise ValueError(f"Script '{script_id}' not found in the library")
        return script.code

    def _python_node(self, step: dict[str, Any]):
        """Execute a Python script, inline or from the script library.

        The code runs with a ``state`` dict in scope so that any state value
        can be read via ``state["key"]``.  The code should assign an ``output``
        variable; its value is stored under ``output_key`` (defaults to the
        step id).

        Sandboxing
        ----------
        ``sandbox`` (default ``True``) runs the script in an isolated
        interpreter with no access to the backend's env vars, tools, bash or
        installed libraries — see ``script_sandbox``.  ``sandbox_runtime``
        selects ``local`` (child process, default), ``docker`` or ``k8s``.

        With ``sandbox: false`` the script is exec'd inside the backend process
        in a thread-pool executor: standard-library imports are available and
        builtins are not restricted (trusted infrastructure code only).
        """
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            output_key = step.get("output_key") or step_id
            sandbox = step.get("sandbox", True)
            sandbox_runtime = step.get("sandbox_runtime") or "local"

            logger.info(
                "[%s] step '%s' running (python, sandbox=%s%s)",
                graph_id, step_id, sandbox, f"/{sandbox_runtime}" if sandbox else "",
            )
            try:
                code = await self._resolve_script_code(step)

                # A data source result is a file, so a script that needs
                # one names it with `stream: <state key>` and reads it with
                # records(). Nothing large ever passes through `state`.
                stream_key = step.get("stream") or step.get("over")
                ref = self._ref_in(state, stream_key) if stream_key else None
                if stream_key and ref is None:
                    raise ValueError(
                        f"step '{step_id}' declares `stream: {stream_key}` but "
                        f"state key '{stream_key}' does not hold a data source "
                        f"result reference"
                    )

                if sandbox:
                    from app.core.config import get_settings
                    from app.infrastructure.orchestration.script_sandbox import run_script

                    settings = get_settings()
                    delivery = await self._stream_delivery(ref, sandbox_runtime, step_id)
                    result = await run_script(
                        code,
                        dict(state),
                        runtime=sandbox_runtime,
                        timeout=float(step.get("timeout_seconds") or settings.script_sandbox_timeout),
                        image=step.get("sandbox_image") or settings.script_sandbox_image,
                        memory_mb=settings.script_sandbox_memory_mb,
                        namespace=settings.agent_namespace,
                        **delivery,
                    )
                else:
                    # Single namespace for globals and locals -- see the same
                    # fix in script_sandbox's bootstrap. Two namespaces make a
                    # top-level `def` invisible to another function's body, so
                    # any script with helpers fails on NameError.
                    local_vars: dict[str, Any] = {
                        "__builtins__": __builtins__,
                        "state": dict(state),
                        # Same contract as the sandbox: the data arrives as a
                        # generator over the stream, never as a value.
                        "records": self._records_callable(ref),
                        "stream_records": ref.items if ref is not None else 0,
                        "stream_truncated": bool(ref is not None and ref.truncated),
                        "output": None,
                    }
                    compiled = compile(code, f"<workflow:{graph_id}:{step_id}>", "exec")

                    def _run() -> None:
                        exec(compiled, local_vars, local_vars)  # noqa: S102

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, _run)
                    result = local_vars.get("output")

                logger.info("[%s] step '%s' finished", graph_id, step_id)
                return {output_key: result}
            except Exception as exc:
                logger.exception("[%s] step '%s' python failed", graph_id, step_id)
                # Some failures carry no message (a bare TimeoutError, an
                # ApiException with an empty body), and "error": "" tells the
                # reader nothing at all. Fall back to the exception type so the
                # failure is at least identifiable.
                detail = str(exc) or type(exc).__name__
                return {output_key: {"error": detail}}

        return node

    # ------------------------------------------------------------------
    # Handing a data stream to a step
    # ------------------------------------------------------------------

    async def _stream_delivery(
        self, ref: Any, runtime: str, step_id: str
    ) -> dict[str, Any]:
        """The ``run_script`` kwargs that get *ref* to a sandbox.

        Two mechanisms, picked by where the sandbox runs:

        * same pod (``local``, ``docker``) -- hand over the path. The bootstrap
          opens it before seccomp denies ``openat``, and docker bind-mounts it
          read-only. No copy at all.
        * another pod (``k8s``) -- hand over a copy callable. The backend
          pushes the bytes into that pod's stdin and it writes its own file.
          There is no shared filesystem and the sandbox has no network, so a
          transfer is the only option; it is chunked, so neither side holds the
          whole stream.
        """
        if ref is None:
            return {}
        store = self._require_stream_store(step_id)
        common = {
            "stream_records": ref.items,
            "stream_truncated": bool(ref.truncated),
        }
        if runtime == "k8s":
            async def _copy(sink: Any) -> int:
                return await store.copy_to(ref, sink)

            return {"stream_copy": _copy, **common}

        path = await store.local_path(ref)
        if path is None:
            # A store that keeps bytes off this filesystem (GridFS, object
            # storage) has no path to hand over, so the same transfer the
            # cross-pod case uses is used here too.
            async def _copy_local(sink: Any) -> int:
                return await store.copy_to(ref, sink)

            return {"stream_copy": _copy_local, **common}
        return {"stream_path": path, **common}

    async def _llm_map_reduce(
        self,
        step: dict[str, Any],
        llm: Any,
        system_prompt: str,
        state: dict,
        ref: Any,
        stream_key: str,
    ) -> Any:
        """Run the prompt per chunk read off the file, then over the answers.

        The only way an ``llm`` step reads a whole result. Chunks come off the
        stream, so the backend holds one chunk; but the model still never sees
        the whole thing at once, and the combining pass sees the map answers
        rather than the records, so this is lossy by construction. Opt-in,
        bounded by ``max_chunks``, and logged with the call count so the bill
        is not a surprise.

        ``chunk_template`` is the per-chunk prompt and receives ``{chunk}``
        (the records as JSON), ``{chunk_index}`` and ``{total_items}`` plus the
        usual state keys; ``reduce_template`` receives ``{parts}``, the answers
        joined. Both have defaults that state the job plainly -- a map-reduce
        with a vague map prompt yields N vague summaries and one vaguer answer.
        """
        import json as _json

        step_id = step["id"]
        settings = self._stream_conf()
        store = self._require_stream_store(step_id)

        size = int(step.get("chunk_items") or settings.stream_chunk_items)
        max_bytes = int(step.get("chunk_bytes") or settings.stream_chunk_bytes)
        max_chunks = int(step.get("max_chunks") or settings.stream_max_chunks)

        chunk_template = step.get("chunk_template") or (
            "{request}\n\nBelow is part {chunk_index} of a larger set of "
            "{total_items} records. Answer for these records only, and keep "
            "the answer short enough to be combined with the others.\n"
            "RECORDS:\n{chunk}"
        )
        reduce_template = step.get("reduce_template") or (
            "{request}\n\nBelow are the answers for each part of a set of "
            "{total_items} records, in order. Combine them into one answer. "
            "Do not invent detail that is not in the parts.\n\nPARTS:\n{parts}"
        )

        parts: list[str] = []
        index = 0
        consumed = 0
        async for chunk in store.chunks(ref, size=size, max_bytes=max_bytes):
            if index >= max_chunks:
                logger.warning(
                    "[%s] step '%s' map_reduce stopped at max_chunks (%d) after "
                    "%d record(s) of %d -- the answer covers a prefix only",
                    self.id, step_id, max_chunks, consumed, ref.items,
                )
                break
            chunk_state = dict(state)
            chunk_state["chunk"] = _json.dumps(chunk, default=str)
            chunk_state["chunk_index"] = index
            chunk_state["total_items"] = ref.items
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=self._render(chunk_template, chunk_state)),
            ])
            parts.append(
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            consumed += len(chunk)
            index += 1

        logger.info(
            "[%s] step '%s' map_reduce: %d chunk call(s) over %d record(s), "
            "plus 1 combining call",
            self.id, step_id, index, consumed,
        )
        if not parts:
            return ""
        if len(parts) == 1:
            # One chunk means the map answer *is* the answer; a combining call
            # over a single part only costs money and adds a paraphrase.
            return parts[0]

        reduce_state = dict(state)
        reduce_state["parts"] = "\n\n---\n\n".join(
            f"[part {i}] {part}" for i, part in enumerate(parts)
        )
        reduce_state["total_items"] = ref.items
        reduced = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=self._render(reduce_template, reduce_state)),
        ])
        return reduced.content

    async def _read_sample(
        self, ref: Any, step_id: str, step: dict[str, Any]
    ) -> list[Any]:
        """The first few records, read off the stream for a prompt.

        Read from the file rather than taken from ``ref.preview`` so the
        number of example records is the step's choice, not a constant fixed
        when the stream was written.
        """
        if ref.shape != "list":
            whole = await self._load_stream(ref, step_id)
            return [whole]
        store = self._require_stream_store(step_id)
        count = max(1, int(step.get("sample_records") or 5))
        return [item async for item in store.stream(ref, limit=count)]

    def _records_callable(self, ref: Any) -> Any:
        """A ``records()`` for an in-process (``sandbox: false``) script.

        Synchronous, because the script it is handed to is synchronous: the
        file is read on the calling thread, one line at a time, so an
        unsandboxed script gets exactly the contract a sandboxed one gets.
        """
        if ref is None:
            def _no_stream():
                raise RuntimeError(
                    "no data stream is attached to this step. Add "
                    "`stream: <state key>` naming the data_source output this "
                    "script should read."
                )

            return _no_stream

        store = self._stream_store

        def _records():
            import json as _json

            path = getattr(store, "_path_for", None)
            if path is None:
                raise RuntimeError(
                    "this data stream store cannot be read from an unsandboxed "
                    "script; use sandbox: true"
                )
            with open(path(ref.id), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield _json.loads(line)

        return _records

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

    def _proceed_or_node(self, step: dict[str, Any]) -> Callable:
        """First-wins fan-in: proceed as soon as *one* upstream branch arrives.

        The mirror image of ``join``.  ``join`` is an AND -- LangGraph schedules
        equal-depth branches into a single superstep, so it runs once, after all
        of them.  This node is an OR: the first branch to reach it wins, and every
        later arrival is cut to END by ``_make_proceed_or_router`` so the tail of
        the graph cannot run a second time.

        That guard is the whole point, and it cannot live in the node body: Pregel
        triggers a node once per superstep that writes to it, so an uneven-depth
        fan-in executes this node again when the slow branch lands.  Without the
        router, every step after it would run once per arrival.

        Failure handling is deliberately conservative, because a fan-in cannot
        tell "that was the last branch" from "another is still coming":

        * A failure arriving **before** anything has won does not win and does not
          clear ``__failed_step__``, so the run fails as it would anywhere else.
          Failing loudly beats stranding the run with its tail never run.
        * A failure arriving **after** a branch has already won is discarded --
          sentinel cleared -- because a late loser must not retroactively fail a
          run that already proceeded.

        Note two limits, both shared with ``join``.  Wall clock is not among the
        benefits: supersteps are globally synchronous, so while any branch is
        still producing supersteps the steps after this node wait for them --
        ``proceed_or`` changes *what runs and how often*, not *when*.  And a step
        that records its error in state instead of raising (a ``python`` step with
        ``stop_on_failure`` unset, say) has not "failed" for this purpose: it
        counts as a normal arrival and can win, so gate on its error payload
        downstream if that matters.
        """
        step_id = step["id"]
        graph_id = self.id
        key = f"_proceed_or_{step_id}"

        async def node(state: dict) -> dict:
            latch = state.get(key) or {}
            arrivals = int(latch.get("arrivals") or 0) + 1
            already_won = bool(latch.get("won"))
            failed = state.get("__failed_step__")

            if failed and failed != step_id:
                if already_won:
                    # A branch already carried the run onward; this late failure
                    # must not undo it, so drop the sentinel and stop here.
                    logger.info(
                        "[%s] step '%s' arrival %d discarded -- branch '%s' failed "
                        "after the run had already proceeded",
                        graph_id, step_id, arrivals, failed,
                    )
                    return {
                        key: {"arrivals": arrivals, "won": True, "last_won": False},
                        "__failed_step__": None,
                    }
                # Nothing has won yet. Leave the sentinel in place so the run
                # fails instead of ending quietly with the tail never run.
                logger.warning(
                    "[%s] step '%s' arrival %d failed (branch '%s') with no "
                    "successful branch yet -- failing the run",
                    graph_id, step_id, arrivals, failed,
                )
                return {key: {"arrivals": arrivals, "won": False, "last_won": False}}

            if already_won:
                logger.info(
                    "[%s] step '%s' arrival %d ignored -- already proceeded",
                    graph_id, step_id, arrivals,
                )
                return {key: {"arrivals": arrivals, "won": True, "last_won": False}}

            logger.info(
                "[%s] step '%s' proceeding on arrival %d", graph_id, step_id, arrivals,
            )
            return {key: {"arrivals": arrivals, "won": True, "last_won": True}}

        return node

    def _make_proceed_or_router(
        self, step_id: str, dest: str
    ) -> Callable[[dict], str]:
        """Route the winning arrival onward and every later one to END."""
        graph_id = self.id
        key = f"_proceed_or_{step_id}"

        def _select(state: dict) -> str:
            latch = state.get(key) or {}
            if latch.get("last_won"):
                return dest
            logger.debug(
                "[%s] proceed_or '%s': late arrival routed to END", graph_id, step_id,
            )
            return END

        return _select

    _MESSAGING_ACTIONS = ("post", "reply", "history", "thread", "dm", "delete")

    @staticmethod
    def _state_path(state: dict, path: str) -> Any:
        """Walk a dotted/bracketed state path, returning None when it breaks.

        ``items: overrides.confirmations`` has to reach a *list*, and {key}
        templating renders to a string — so list-valued config is addressed by
        path instead of by template.
        """
        current: Any = state
        for part in path.replace("]", "").replace("[", ".").split("."):
            part = part.strip()
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.lstrip("-").isdigit():
                try:
                    current = current[int(part)]
                except IndexError:
                    return None
            else:
                return None
            if current is None:
                return None
        return current

    def _messaging_node(self, step: dict[str, Any]):
        """Post to, read from, or clean up a chat provider.

        One step type covers every provider: ``provider`` names an
        implementation registered in ``app.infrastructure.messaging`` (``slack``
        today), so adding WhatsApp or Teams later is a new provider class, not a
        new step type.  Config:

        ``action``   post | reply | history | thread | dm | delete
        ``channel``  target channel id (post/reply/history/thread/delete)
        ``text``     message body (post/reply/dm)
        ``thread_id``root message id (reply/thread)
        ``user_id``  DM recipient (dm) — the channel is opened on the fly
        ``message_id`` message to delete (delete)
        ``oldest`` / ``limit``  history window
        ``items``    state path to a list of ``{thread_id, text}`` (reply), so one
                     step can confirm N things without a loop construct
        ``skip_if_replied``  do not post a reply whose exact text is already in
                     the thread — the idempotency guard
        ``ignore_errors``  capture a provider error under ``output_key`` instead
                     of failing the run

        Every string field is ``{state}``-templated like the other steps.  The
        credential is never part of this config: the provider reads
        ``SLACK_BOT_TOKEN`` from settings itself, and provider errors are
        scrubbed of it before they can reach run state.
        """
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            action = (step.get("action") or "post").lower()
            provider_name = (step.get("provider") or "").strip() or None
            output_key = step.get("output_key") or step_id
            list_action = action in ("history", "thread")
            try:
                from app.infrastructure.messaging import MessagingError, get_provider

                if action not in self._MESSAGING_ACTIONS:
                    raise ValueError(
                        f"step '{step_id}' has unknown messaging action '{action}' "
                        f"(expected {' | '.join(self._MESSAGING_ACTIONS)})"
                    )
                provider = get_provider(provider_name)
                channel = self._render(str(step.get("channel") or ""), state).strip()
                text = self._render(str(step.get("text") or ""), state)
                thread_id = self._render(str(step.get("thread_id") or ""), state).strip()

                logger.info(
                    "[%s] step '%s' running (%s %s channel=%s)",
                    graph_id, step_id, provider.name, action, channel or "-",
                )

                if action == "post":
                    self._need(step_id, "channel", channel)
                    posted = await provider.post_message(channel, text)
                    result: Any = posted.as_dict()

                elif action == "reply":
                    self._need(step_id, "channel", channel)
                    result = await self._post_replies(step, state, provider, channel,
                                                      thread_id, text)

                elif action == "history":
                    self._need(step_id, "channel", channel)
                    oldest = self._render(str(step.get("oldest") or ""), state).strip()
                    limit = self._render(str(step.get("limit") or ""), state).strip()
                    messages = await provider.read_history(
                        channel,
                        oldest=oldest or None,
                        limit=int(limit) if limit.isdigit() else None,
                    )
                    result = [m.as_dict() for m in messages]

                elif action == "thread":
                    self._need(step_id, "channel", channel)
                    self._need(step_id, "thread_id", thread_id)
                    messages = await provider.read_thread(channel, thread_id)
                    result = [m.as_dict() for m in messages]

                elif action == "dm":
                    user_id = self._render(str(step.get("user_id") or ""), state).strip()
                    self._need(step_id, "user_id", user_id)
                    dm_channel = await provider.open_dm(user_id)
                    posted = await provider.post_message(dm_channel, text)
                    result = {**posted.as_dict(), "user_id": user_id}

                else:  # delete
                    self._need(step_id, "channel", channel)
                    message_id = self._render(
                        str(step.get("message_id") or ""), state
                    ).strip()
                    self._need(step_id, "message_id", message_id)
                    await provider.delete_message(channel, message_id)
                    result = {"deleted": True, "message_id": message_id,
                              "channel": channel}

                logger.info("[%s] step '%s' finished (%s)", graph_id, step_id, action)
                return {output_key: result}
            except Exception as exc:
                logger.exception("[%s] step '%s' messaging failed", graph_id, step_id)
                if step.get("ignore_errors"):
                    # A provider being unreachable must not be able to take the
                    # whole run down when the caller said so — the CSM watcher
                    # reads its override channel this way, because refusing to
                    # compute deadlines because Slack is down would trade a real
                    # alarm for a missing one.
                    return {output_key: [] if list_action else {"error": str(exc)}}
                return {output_key: {"error": str(exc)}, "__failed_step__": step_id}

        return node

    @staticmethod
    def _need(step_id: str, field: str, value: str) -> None:
        if not value:
            raise ValueError(f"step '{step_id}' needs a non-empty '{field}'")

    @classmethod
    async def _post_replies(
        cls,
        step: dict[str, Any],
        state: dict,
        provider: Any,
        channel: str,
        thread_id: str,
        text: str,
    ) -> dict[str, Any]:
        """Post one thread reply, or a batch of them, skipping duplicates.

        ``items`` names a state path holding a list of ``{thread_id, text}``
        (``thread_ts`` is accepted as well, since that is what Slack calls it).
        With ``skip_if_replied`` the thread is read first and a reply whose text
        is already present is not posted again — that is what makes "confirm
        each accepted override" safe to re-run, which the CSM watcher does every
        morning over an overlapping 26-hour window.
        """
        raw_items = step.get("items")
        if isinstance(raw_items, str) and raw_items.strip():
            items = cls._state_path(state, raw_items.strip())
        elif isinstance(raw_items, list):
            items = cls._render_deep(raw_items, state)
        else:
            items = None

        if items is None:
            cls._need(step["id"], "thread_id", thread_id)
            entries = [{"thread_id": thread_id, "text": text}]
            batch = False
        else:
            if not isinstance(items, list):
                raise ValueError(
                    f"step '{step['id']}' items path '{raw_items}' is not a list"
                )
            entries = items
            batch = True

        skip_if_replied = bool(step.get("skip_if_replied"))
        posted: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_thread = str(entry.get("thread_id") or entry.get("thread_ts") or "")
            entry_text = str(entry.get("text") or "")
            if not entry_thread or not entry_text:
                skipped.append({"thread_id": entry_thread, "reason": "incomplete"})
                continue
            if skip_if_replied:
                existing = await provider.read_thread(channel, entry_thread)
                # The root message is in the reply list too; comparing text
                # rather than authorship keeps the check provider-neutral and
                # deterministic, which is what makes it testable.
                if any(m.text == entry_text and m.id != entry_thread for m in existing):
                    skipped.append({"thread_id": entry_thread, "reason": "already_replied"})
                    continue
            sent = await provider.reply_in_thread(channel, entry_thread, entry_text)
            posted.append({**sent.as_dict(), "thread_id": entry_thread})

        if not batch:
            if skipped:
                return {"skipped": True, "reason": skipped[0]["reason"],
                        "thread_id": thread_id}
            return {**posted[0], "thread_id": thread_id}
        return {"posted": posted, "skipped": skipped,
                "posted_count": len(posted), "skipped_count": len(skipped)}

    def _storage_node(self, step: dict[str, Any]):
        """Read or write this workflow's own key/value storage.

        Config: ``action`` (``get`` | ``set`` | ``delete`` | ``keys``), ``key``
        and, for ``set``, ``value`` -- all ``{state}``-templated -- plus the usual
        ``output_key``.

        The owner is never configurable.  ``workflow_id`` is taken from the
        runner (``self.id``), so a step cannot name another workflow's storage:
        the only id it can address is its own.  That is the whole security model
        and it is structural rather than checked.

        A missing key reads back as ``None`` rather than failing -- a first run
        has no state yet, and that is normal, not an error.  Everything else
        fails loudly: storage switched off for the workflow, an unknown action,
        a missing key name, or a value too large to be bookkeeping.
        """
        graph_id = self.id

        async def node(state: dict) -> dict:
            step_id = step["id"]
            action = (step.get("action") or "get").lower()
            output_key = step.get("output_key") or step_id
            try:
                if not self._storage_enabled:
                    raise ValueError(
                        f"step '{step_id}' uses storage but workflow '{graph_id}' "
                        f"has it switched off -- enable storage in the workflow "
                        f"settings first"
                    )
                if self._storage_backend is None:
                    raise ValueError("workflow storage backend not configured")

                if action == "keys":
                    keys = await self._storage_backend.keys(graph_id)
                    logger.info(
                        "[%s] step '%s' storage keys -> %d entry(ies)",
                        graph_id, step_id, len(keys),
                    )
                    return {output_key: keys}

                key = self._render(str(step.get("key") or ""), state).strip()
                if not key:
                    raise ValueError(
                        f"step '{step_id}' needs a non-empty 'key' for action '{action}'"
                    )

                if action == "get":
                    value = await self._storage_backend.get(graph_id, key)
                    logger.info(
                        "[%s] step '%s' storage get '%s' -> %s",
                        graph_id, step_id, key, "hit" if value is not None else "miss",
                    )
                    return {output_key: value}

                if action == "set":
                    value = self._render_deep(step.get("value"), state)
                    await self._storage_backend.set(graph_id, key, value)
                    logger.info("[%s] step '%s' storage set '%s'", graph_id, step_id, key)
                    return {output_key: {"key": key, "saved": True}}

                if action == "delete":
                    await self._storage_backend.delete(graph_id, key)
                    logger.info("[%s] step '%s' storage delete '%s'", graph_id, step_id, key)
                    return {output_key: {"key": key, "deleted": True}}

                raise ValueError(
                    f"step '{step_id}' has unknown storage action '{action}' "
                    f"(expected get | set | delete | keys)"
                )
            except Exception as exc:
                logger.exception("[%s] step '%s' storage failed", graph_id, step_id)
                return {output_key: {"error": str(exc)}, "__failed_step__": step_id}

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

        d = _DefaultDict(_stream_safe_state(state))
        d["env"] = _EnvAccessor()
        try:
            return string.Formatter().vformat(template, [], d)  # type: ignore[arg-type]
        except ValueError:
            return template

    # A config value that is exactly one placeholder, e.g. "{rows}".
    _WHOLE_PLACEHOLDER = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\]]+\]|\.[A-Za-z_][A-Za-z0-9_]*)*)\}$")

    @classmethod
    def _render_whole(cls, template: str, state: dict) -> Any:
        """The referenced object itself when *template* is one placeholder.

        ``{"values": "{rows}"}`` has to reach a data source as the list, not as
        ``"[['a', 1], ['b', 2]]"`` -- a Python repr, which is not even valid
        JSON, and which an ``array`` param passes through untouched. That is
        what a step composing two data sources needs: read rows from one, write
        them to the other.

        Only lists and dicts are passed through. Scalars keep rendering as
        strings so that ``{"page": "{n}"}`` behaves exactly as it always has
        and the executor's declared-type coercion stays the thing that decides
        what a number is.
        """
        match = cls._WHOLE_PLACEHOLDER.match(template)
        if match is None:
            return None
        head, path = match.group(1), match.group(2)
        if head == "env" or head not in state:
            return None
        current: Any = state[head]
        for part in re.findall(r"\[([^\]]+)\]|\.([A-Za-z_][A-Za-z0-9_]*)", path):
            key = part[0] or part[1]
            try:
                if isinstance(current, (list, tuple)) and key.lstrip("-").isdigit():
                    current = current[int(key)]
                else:
                    current = current[key]
            except (KeyError, IndexError, TypeError):
                return None
        if as_data_ref(current) is not None:
            # Never pass a stream reference through as data. It is a dict, so
            # it would qualify, but inlining one is exactly what the summary
            # rendering exists to prevent -- a step that wants the records
            # names it with `stream:` and reads the file.
            return None
        return current if isinstance(current, (list, dict)) else None

    @classmethod
    def _render_deep(cls, value: Any, state: dict) -> Any:
        """Recursively render {key} templates in dicts, lists, and strings."""
        if isinstance(value, str):
            whole = cls._render_whole(value, state)
            if whole is not None:
                return whole
            return cls._render(value, state)
        if isinstance(value, dict):
            return {k: cls._render_deep(v, state) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._render_deep(item, state) for item in value]
        return value

    # ------------------------------------------------------------------
    # Data source results (always a stream reference)
    # ------------------------------------------------------------------

    def _stream_conf(self) -> Any:
        from app.core.config import get_settings

        return get_settings()

    def _ref_in(self, state: dict, key: str) -> Any:
        """The ``DataRef`` at *key*, or ``None`` when it is not one."""
        return as_data_ref(state.get(key))

    def _require_stream_store(self, step_id: str) -> Any:
        if self._stream_store is None:
            raise ValueError(
                f"step '{step_id}' reads a data source result but no data "
                f"stream store is configured on this backend"
            )
        return self._stream_store

    async def _load_stream(self, ref: Any, step_id: str) -> Any:
        """Load a stream whole, for a step that asked for ``result_mode: ram``.

        The escape hatch for a workflow that must have the value inline -- a
        route condition on one of its fields, an ``http_call`` body built from
        it.  Refuses past ``stream_read_all_max_bytes`` rather than degrading:
        the value is about to go into a checkpoint, and a silent partial read
        is the failure this design exists to remove.
        """
        store = self._require_stream_store(step_id)
        limit = int(getattr(self._stream_conf(), "stream_read_all_max_bytes", 0) or 0)
        return await store.read_all(ref, max_bytes=limit)
