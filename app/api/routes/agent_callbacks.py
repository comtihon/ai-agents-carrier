"""Agent callback routes — used by running agent servers to communicate with the backend.

Protocol
--------
The agent server calls these endpoints over HTTP to:
  - Report its output when done  (POST /runs/{run_id}/agent/output)
  - Ask a clarifying question    (POST /runs/{run_id}/agent/question)
  - Long-poll for an answer      (GET  /runs/{run_id}/agent/input)
  - Receive the answer           (POST /runs/{run_id}/agent/reply)
  - Send a progress message      (POST /runs/{run_id}/agent/progress)

Route ordering note
-------------------
This router uses a ``/runs`` prefix (no ``/workflows`` prefix) so it does not
conflict with the existing workflows router.  Routes with literal segments
(``/agent/output``, ``/agent/input``, etc.) are all registered before any
parameterised ``/{run_id}`` catch-all routes in other routers.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from langgraph.types import Command
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.infrastructure.orchestration.yaml_graph import stream_graph_to_pause

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["agent-callbacks"])

# ---------------------------------------------------------------------------
# In-memory state for question/answer long-poll
# ---------------------------------------------------------------------------
# These are intentionally module-level (not per-request) so all coroutines
# sharing the same process can communicate via them.  They are keyed by run_id.

_answer_events: dict[str, asyncio.Event] = {}
_answers: dict[str, str] = {}
_questions: dict[str, dict[str, Any]] = {}

_LONG_POLL_TIMEOUT = 600.0  # 10 minutes


def _get_or_create_event(run_id: str) -> asyncio.Event:
    if run_id not in _answer_events:
        _answer_events[run_id] = asyncio.Event()
    return _answer_events[run_id]


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class AgentOutputBody(BaseModel):
    output: dict[str, Any]


class AgentQuestionBody(BaseModel):
    question: str
    options: list[str] | None = None


class AgentReplyBody(BaseModel):
    answer: str


class AgentProgressBody(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/{run_id}/agent/output", status_code=202)
async def agent_output(
    run_id: str,
    body: AgentOutputBody,
    background_tasks: BackgroundTasks,
    container: ApplicationContainer = Depends(get_container),
):
    # DEPRECATED: primary output delivery now via GET /poll on the agent.
    # Kept for backward compatibility.
    """Called by the agent when it has finished and has an output to report.

    Resumes the paused LangGraph run with the agent's output.
    """
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status not in ("waiting_agent", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Run is not in a state to receive agent output (status={run.status})",
        )

    runner = container.live_runners.get(run_id)
    if runner is None:
        # Fall back to the workflow registry — covers runs triggered outside this
        # process (e.g. server restart, multi-replica, or direct DB injection).
        runner = container.yaml_graph_registry.get(run.graph_id)
    if runner is None:
        raise HTTPException(
            status_code=404,
            detail=f"Runner for run '{run_id}' not found (workflow '{run.graph_id}' not in registry)",
        )

    # If the agent reported a failure, propagate it as a run failure instead of
    # resuming the graph with an error string as output.
    if "error" in body.output:
        error_msg = str(body.output["error"])
        logger.warning("run %s: agent reported error: %s", run_id, error_msg[:200])
        run.status = "failed"
        run.state = {**(run.state or {}), "error": f"Agent error: {error_msg}"}
        run.agent_url = None
        if run.current_step:
            run.step_statuses = {**(run.step_statuses or {}), run.current_step: "failed"}
            run.step_outputs = {**(run.step_outputs or {}), run.current_step: {"error": error_msg}}
        run.touch()
        await container.run_repository.update(run)
        # Terminate agent container (best-effort — pod sent the callback so it's still alive)
        try:
            from app.runtime.k8s import K8sRuntime
            await K8sRuntime(namespace=container.settings.agent_namespace).terminate_by_run_id(None, run_id)
        except Exception:
            logger.debug("run %s: k8s cleanup on agent error failed", run_id, exc_info=True)
        try:
            from app.runtime.docker import DockerRuntime
            await DockerRuntime(
                registry_username=container.settings.docker_registry_username,
                registry_password=container.settings.docker_registry_password,
            ).terminate_by_run_id(None, run_id)
        except Exception:
            logger.debug("run %s: docker cleanup on agent error failed", run_id, exc_info=True)
        return {"run_id": run_id, "status": "failed"}

    # Transition status immediately so the polling client sees the change
    # even before the background task drains.
    run.status = "running"
    run.agent_url = None
    run.touch()
    await container.run_repository.update(run)

    background_tasks.add_task(
        _resume_with_output,
        runner, run, container, body.output,
    )

    return {"run_id": run_id, "status": "resuming"}


async def _resume_with_output(runner, run, container: ApplicationContainer, output: dict) -> None:
    try:
        await stream_graph_to_pause(
            runner, run, container.run_repository,
            Command(resume={"output": output}),
            base_url=container.settings.base_url,
        )
    except Exception:
        logger.exception("run %s: failed to resume after agent output", run.id)
    finally:
        if run.status in ("completed", "failed", "cancelled", "rejected"):
            container.live_runners.pop(run.id, None)


@router.post("/{run_id}/agent/question", status_code=202)
async def agent_question(
    run_id: str,
    body: AgentQuestionBody,
    container: ApplicationContainer = Depends(get_container),
):
    """Called by the agent when it needs to ask a clarifying question.

    Stores the question so the frontend can display it and wait for a reply.
    The agent should then call GET /runs/{run_id}/agent/input to long-poll for
    the answer.
    """
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    _questions[run_id] = {
        "question": body.question,
        "options": body.options,
    }
    # Reset the answer event so the agent's next poll will block until a reply
    # arrives.
    event = _get_or_create_event(run_id)
    event.clear()
    _answers.pop(run_id, None)

    run.state = {**(run.state or {}), "_pending_question": {"question": body.question, "options": body.options}}
    run.touch()
    await container.run_repository.update(run)

    # Send Slack notification so the user can answer from Slack (thread reply)
    # or from the UI popup. Only fires once per question (first ask).
    try:
        from app.core.config import get_settings
        from app.infrastructure.notifications.webhook_notifier import post_slack_ask_context
        settings = get_settings()
        if settings.slack_bot_token and settings.slack_approvals_channel:
            existing_ts = (run.state or {}).get("_slack_ask_context_ts")
            if not existing_ts:
                notif_resp = await post_slack_ask_context(
                    settings.slack_bot_token,
                    settings.slack_approvals_channel,
                    [body.question],
                    run_id,
                    run.state or {},
                )
                if notif_resp and notif_resp.get("ok"):
                    ts = notif_resp.get("ts")
                    channel = notif_resp.get("channel")
                    if ts and channel:
                        run.state = {**run.state, "_slack_ask_context_ts": ts, "_slack_ask_context_channel": channel}
                        run.touch()
                        await container.run_repository.update(run)
    except Exception:
        logger.exception("run %s: failed to send Slack notification for agent question", run_id)

    logger.info("run %s: agent asked question: %r", run_id, body.question)
    return {"run_id": run_id, "status": "question_stored"}


@router.get("/{run_id}/agent/input")
async def agent_input(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Long-poll endpoint for the agent to receive an answer.

    The agent calls this endpoint and blocks until a human (or the frontend) has
    submitted an answer via ``POST /runs/{run_id}/agent/reply``.

    Returns 408 if no answer arrives within 10 minutes.
    """
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    pending = (run.state or {}).get("_pending_answer")
    if pending is not None:
        new_state = {k: v for k, v in (run.state or {}).items() if k != "_pending_answer"}
        run.state = new_state
        run.touch()
        await container.run_repository.update(run)
        _answers.pop(run_id, None)
        _answer_events.pop(run_id, None)
        logger.info("run %s: delivering persisted answer to agent", run_id)
        return {"answer": pending}

    event = _get_or_create_event(run_id)
    try:
        await asyncio.wait_for(event.wait(), timeout=_LONG_POLL_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail="No answer received within the timeout period",
        )

    answer = _answers.pop(run_id, "")
    _answer_events.pop(run_id, None)
    logger.info("run %s: delivering answer to agent", run_id)
    return {"answer": answer}


@router.post("/{run_id}/agent/reply", status_code=202)
async def agent_reply(
    run_id: str,
    body: AgentReplyBody,
    container: ApplicationContainer = Depends(get_container),
):
    """Called by the frontend (or a test client) to provide an answer to the agent.

    Stores the answer and wakes the long-polling ``GET /agent/input`` request.
    """
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    _answers[run_id] = body.answer
    event = _get_or_create_event(run_id)
    event.set()

    run = await container.run_repository.get(run_id)
    if run:
        run.state = {
            **{k: v for k, v in (run.state or {}).items() if k != "_pending_question"},
            "_pending_answer": body.answer,
        }
        run.touch()
        await container.run_repository.update(run)

    logger.info("run %s: answer stored and event set", run_id)
    return {"run_id": run_id, "status": "answer_delivered"}


@router.post("/{run_id}/agent/progress", status_code=202)
async def agent_progress(
    run_id: str,
    body: AgentProgressBody,
    container: ApplicationContainer = Depends(get_container),
):
    """Called by the agent to report incremental progress.

    Stores the progress message in the run state and, if SSE is available,
    emits it to the frontend.
    """
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    import json as _json

    # Scope live progress/token fields to the step that's actually running —
    # a single workflow run can have several langgraph-agent steps in sequence
    # (researcher, planner, coder, ...) and they all report to this same
    # run_id. Without scoping, opening a finished step's detail view would
    # show whichever step is currently streaming instead of that step's own
    # (now-static) history.
    _step_key = run.current_step or "_unscoped"

    # Structured token-update message — store in _live_token_usage, skip progress list
    if body.message.startswith("__token__:"):
        try:
            usage = _json.loads(body.message[len("__token__:"):])
            run.state = {**(run.state or {}), f"_live_token_usage_{_step_key}": usage}
            run.touch()
            await container.run_repository.update(run)
        except Exception:
            pass  # malformed — fall through to append
        else:
            return {"run_id": run_id, "status": "token_updated"}

    if body.message.startswith("__mcp_start__:"):
        try:
            d = _json.loads(body.message[len("__mcp_start__:"):])
            servers: list = list(run.state.get("_active_mcp_servers", []))
            if d.get("server"):
                servers.append(d["server"])
            run.state = {**(run.state or {}), "_active_mcp_servers": servers}
            run.touch()
            await container.run_repository.update(run)
        except Exception:
            pass
        else:
            return {"run_id": run_id, "status": "mcp_started"}

    if body.message.startswith("__mcp_end__:"):
        try:
            d = _json.loads(body.message[len("__mcp_end__:"):])
            servers = list(run.state.get("_active_mcp_servers", []))
            srv = d.get("server")
            if srv in servers:
                servers.remove(srv)
            run.state = {**(run.state or {}), "_active_mcp_servers": servers}
            run.touch()
            await container.run_repository.update(run)
        except Exception:
            pass
        else:
            return {"run_id": run_id, "status": "mcp_ended"}

    if body.message.startswith("__tool_start__:"):
        try:
            d = _json.loads(body.message[len("__tool_start__:"):])
            tools: list = list(run.state.get("_active_tools", []))
            if d.get("tool"):
                tools.append(d["tool"])
            run.state = {**(run.state or {}), "_active_tools": tools}
            run.touch()
            await container.run_repository.update(run)
        except Exception:
            pass
        else:
            return {"run_id": run_id, "status": "tool_started"}

    if body.message.startswith("__tool_end__:"):
        try:
            d = _json.loads(body.message[len("__tool_end__:"):])
            tools = list(run.state.get("_active_tools", []))
            tl = d.get("tool")
            if tl in tools:
                tools.remove(tl)
            run.state = {**(run.state or {}), "_active_tools": tools}
            run.touch()
            await container.run_repository.update(run)
        except Exception:
            pass
        else:
            return {"run_id": run_id, "status": "tool_ended"}

    if body.message.startswith("__mcp_clear__:"):
        try:
            run.state = {
                **(run.state or {}),
                "_active_mcp_servers": [],
                "_active_tools": [],
            }
            run.touch()
            await container.run_repository.update(run)
        except Exception:
            pass
        else:
            return {"run_id": run_id, "status": "mcp_cleared"}

    # Unknown sentinel messages (prefixed with "__") are never appended to the
    # progress list — they are control signals, not user-visible progress.
    if body.message.startswith("__"):
        return {"run_id": run_id, "status": "sentinel_ignored"}

    # Append progress message to run state for frontend polling.
    _progress_field = f"_agent_progress_{_step_key}"
    progress_list: list = list(run.state.get(_progress_field, []))
    progress_list.append(body.message)
    run.state = {**run.state, _progress_field: progress_list}
    run.touch()
    await container.run_repository.update(run)

    logger.info("run %s: agent progress: %r", run_id, body.message)
    return {"run_id": run_id, "status": "progress_stored"}
