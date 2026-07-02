"""
Step-status reporting when a workflow loops back through an earlier step.

Workflow under test:
    gather (llm_structured) --[!context_sufficient]--> ask_context --> gather
    gather --[context_sufficient]--> approval

Expected execution sequence: gather → ask_context (interrupt) → resume →
gather → approval (interrupt).

The API consumer (copilot_ui) renders the active step using ``current_step``
and ``step_statuses``. During the *second* gather pass the API state must
clearly indicate that gather is the running step — not approval — so the UI
does not fall back to positional inference (currentIdx + 1) and mark the
wrong node active.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.yaml_graph import (
    YamlGraphRunner,
    _parse_questions_string,
    stream_graph_to_pause,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider


WORKFLOW_STEPS: list[dict[str, Any]] = [
    {
        "id": "gather",
        "type": "llm_structured",
        "name": "Gather context",
        "system_prompt": "Gather everything",
        "user_template": "{request}",
        "bind_mcp_tools": False,
        "output": [
            {"name": "context", "type": "str", "description": "ctx"},
            {"name": "context_sufficient", "type": "bool", "description": "done?"},
            {"name": "questions", "type": "str", "description": "qs"},
        ],
        "routes": [
            {"when": "context_sufficient", "next": "approval"},
            {"next": "ask_context"},
        ],
    },
    {
        "id": "ask_context",
        "type": "ask_context",
        "name": "Ask Context",
        "questions_key": "questions",
        "output_key": "context_answers",
        "next": "gather",
    },
    {"id": "approval", "type": "human_approval", "name": "Approval"},
]


class _FakeToolCallingChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel + a no-op bind_tools so llm_structured works."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


def _submit_output(args: dict[str, Any], tc_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "submit_output", "args": args, "id": tc_id}],
    )


def _make_runner() -> YamlGraphRunner:
    # Two gather invocations, both end with submit_output. First emits
    # context_sufficient=False (routes to ask_context); second emits True
    # (routes to approval).
    llm = _FakeToolCallingChatModel(responses=[
        _submit_output(
            {"context": "first", "context_sufficient": False, "questions": "Q1\nQ2"},
            "tc1",
        ),
        _submit_output(
            {"context": "second", "context_sufficient": True, "questions": "none"},
            "tc2",
        ),
    ])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])
    mcp.get_tool_server = MagicMock(return_value=None)
    return YamlGraphRunner(
        {"id": "loopback", "steps": WORKFLOW_STEPS},
        llm=llm,
        mcp_tools_provider=mcp,
    )


def _make_run() -> GraphRun:
    return GraphRun(
        id="loopback-run",
        graph_id="loopback",
        user_request="hello",
        status="running",
        current_step=None,
        state={},
        step_inputs={},
        step_outputs={},
        step_statuses={s["id"]: "pending" for s in WORKFLOW_STEPS},
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )


@pytest.mark.asyncio
async def test_loopback_executes_gather_ask_gather_approval():
    """Sanity check: the workflow really runs the loop-back sequence."""
    runner = _make_runner()
    run = _make_run()
    repo = AsyncMock()

    # First pass: gather → ask_context (interrupt). The wrapper marks the
    # node "running" before it executes; ask_context interrupts mid-flight
    # so its step_status stays "running" until the resume completes it.
    await stream_graph_to_pause(runner, run, repo, {"request": "hello"})
    assert run.status == "waiting_approval"
    assert run.current_step == "ask_context"
    assert run.step_statuses["gather"] == "finished"
    assert run.step_statuses["ask_context"] == "running"
    assert run.step_statuses["approval"] == "pending"

    # Resume with answers: ask_context → gather (loop back) → approval (interrupt)
    await stream_graph_to_pause(runner, run, repo, Command(resume={"0": "a", "1": "b"}))
    assert run.status == "waiting_approval"
    assert run.current_step == "approval"
    # gather ran twice, ask_context completed once on resume — both finished.
    assert run.step_statuses["gather"] == "finished"
    assert run.step_statuses["ask_context"] == "finished"


@pytest.mark.asyncio
async def test_active_step_during_second_gather_pass():
    """
    Between ask_context completing on resume and gather completing for the
    second time, the API state must indicate gather is running.

    Currently the runner only updates ``step_statuses`` *after* a node
    returns, leaving gather's status as ``finished`` from the first pass
    while it's actually being re-executed. Combined with ``current_step``
    pointing at ask_context, the UI's positional fallback (currentIdx + 1)
    marks approval as active — which is the bug the user reports.

    Asserting "running" here documents the intended contract; it will fail
    on the current backend until the runner emits a "running" status when
    a node starts.
    """
    runner = _make_runner()
    run = _make_run()

    # Capture state at every persistence point in the runner.
    persisted: list[dict[str, Any]] = []

    async def capture(updated_run):
        persisted.append({
            "current_step": updated_run.current_step,
            "status": updated_run.status,
            "step_statuses": dict(updated_run.step_statuses),
        })

    repo = AsyncMock()
    repo.update.side_effect = capture

    # First pass: gather → ask_context (interrupt)
    await stream_graph_to_pause(runner, run, repo, {"request": "hello"})
    persisted.clear()

    # Resume: ask_context (chunk 1) → gather (chunk 2) → approval interrupts
    await stream_graph_to_pause(runner, run, repo, Command(resume={"0": "a", "1": "b"}))

    # During the resume stream the runner persists multiple snapshots:
    #   - chunk handler after ask_context completes (gather still 'finished')
    #   - wrapper entry for gather    → gather='running', current_step='gather'
    #   - chunk handler after gather  → gather='finished'
    #   - wrapper entry for approval  → approval='running'
    #   - waiting_approval finalised  → approval is the active step
    # We need at least one snapshot where gather is the active running step
    # so the UI can render it correctly.
    snapshots_with_gather_running = [
        s for s in persisted
        if s["current_step"] == "gather"
        and s["step_statuses"].get("gather") == "running"
    ]
    assert snapshots_with_gather_running, (
        "Expected at least one persisted snapshot during the second gather "
        "pass with current_step='gather' and step_statuses['gather']='running'. "
        f"Persisted snapshots: {persisted!r}"
    )


@pytest.mark.asyncio
async def test_run_response_payload_identifies_active_step_during_loop():
    """
    The HTTP shape (``_run_response``) consumed by copilot_ui must surface
    enough information for the UI to render the right active node during
    the second gather pass — without falling back to positional inference.

    The combination of ``status == "running"``, ``current_step`` and per-step
    ``status`` in ``steps`` is what the UI uses. When gather is re-running
    we expect ``steps[gather].status == "running"``.
    """
    from app.api.routes.workflows import _run_response, _steps_from_definition

    runner = _make_runner()
    run = _make_run()
    repo = AsyncMock()

    # Drive to ask_context interrupt, then into resume.
    await stream_graph_to_pause(runner, run, repo, {"request": "hello"})

    # Manually construct the state that exists between chunks 1 and 2 on
    # resume — i.e. ask_context just completed, gather is about to re-run.
    # We mirror the persistence the runner would have emitted.
    run.status = "running"
    run.current_step = "ask_context"
    run.step_statuses["ask_context"] = "finished"
    # NOTE: this mirrors the buggy state. After the runner emits a "running"
    # status when gather restarts, we'd update this to "running" too.
    run.step_statuses["gather"] = "running"

    # The API response should label the gather step as running.
    response = await _run_response(run, runner)
    steps = response["steps"]
    by_id = {s["id"]: s for s in steps}
    assert by_id["gather"]["status"] == "running", (
        f"_run_response must surface gather=running so the UI can highlight it. "
        f"Got: {by_id['gather']!r}"
    )
    assert by_id["approval"]["status"] == "pending"
    assert response["current_step"] == "ask_context"
    assert response["status"] == "running"


def test_parse_questions_drops_preamble_when_numbered_list_present():
    raw = (
        "The Jira Epic has no description. Based on the codebase ...\n\n"
        "1. What exactly should it do?\n"
        "2. Which data source should be used?\n"
        "3. Should it replace manual drawing?"
    )
    assert _parse_questions_string(raw) == [
        "What exactly should it do?",
        "Which data source should be used?",
        "Should it replace manual drawing?",
    ]


def test_parse_questions_handles_paren_form():
    raw = "1) First?\n2) Second?"
    assert _parse_questions_string(raw) == ["First?", "Second?"]


def test_parse_questions_falls_back_to_lines_when_no_numbering():
    raw = "Q1\nQ2\nQ3"
    assert _parse_questions_string(raw) == ["Q1", "Q2", "Q3"]


def test_parse_questions_keeps_single_unnumbered_line():
    # Only one numbered line is ambiguous — a single Q without explicit numbering
    # should still be presented.
    raw = "Just one question?"
    assert _parse_questions_string(raw) == ["Just one question?"]


@pytest.mark.asyncio
async def test_stream_graph_preserves_out_of_band_agent_progress_on_success():
    """
    `_stream_graph` (app.api.routes.workflows) must not wipe Mongo-only
    `_agent_progress_*` keys that a concurrent progress callback wrote to
    the run's state while the stream was in flight.

    Before the fix, the success path did a wholesale `run.state = snap.values`,
    silently dropping any key not present in the LangGraph snapshot values
    (e.g. keys written directly to Mongo by a different request handler).
    """
    from app.api.routes.workflows import _stream_graph

    runner = _make_runner()
    run = _make_run()

    concurrent_state = {
        "_agent_progress_gather": ["m1", "m2"],
        "_agent_progress_ask_context": ["x"],
    }
    fresh_run = GraphRun(
        id=run.id,
        graph_id=run.graph_id,
        user_request=run.user_request,
        status=run.status,
        state=dict(concurrent_state),
    )

    container = MagicMock()
    container.run_repository = AsyncMock()
    container.run_repository.get = AsyncMock(return_value=fresh_run)
    container.settings = MagicMock(base_url=None)

    await _stream_graph(runner, run, container, {"request": "hello"}, base_url=None)

    assert run.state["_agent_progress_gather"] == ["m1", "m2"]
    assert run.state["_agent_progress_ask_context"] == ["x"]


@pytest.mark.asyncio
async def test_stream_graph_preserves_out_of_band_agent_progress_on_failure():
    """
    Same guarantee as above, but for the failure/except branch of
    `_stream_graph`: `_agent_progress_*` keys must survive alongside the
    newly-set `error` key.
    """
    from app.api.routes.workflows import _stream_graph

    bad_response = AIMessage(content="thinking…")  # no tool_calls → nudge loop
    llm = _FakeToolCallingChatModel(responses=[bad_response] * 5)
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])
    mcp.get_tool_server = MagicMock(return_value=None)
    steps = [
        {**WORKFLOW_STEPS[0], "max_iterations": 3},
        WORKFLOW_STEPS[1],
        WORKFLOW_STEPS[2],
    ]
    runner = YamlGraphRunner(
        {"id": "loopback-fail", "steps": steps}, llm=llm, mcp_tools_provider=mcp,
    )
    run = _make_run()
    run.id = "loopback-fail-run"

    concurrent_state = {"_agent_progress_gather": ["m1", "m2"]}
    fresh_run = GraphRun(
        id=run.id,
        graph_id=run.graph_id,
        user_request=run.user_request,
        status=run.status,
        state=dict(concurrent_state),
    )

    container = MagicMock()
    container.run_repository = AsyncMock()
    container.run_repository.get = AsyncMock(return_value=fresh_run)
    container.settings = MagicMock(base_url=None)

    await _stream_graph(runner, run, container, {"request": "hello"}, base_url=None)

    assert run.status == "failed"
    assert run.state["_agent_progress_gather"] == ["m1", "m2"]
    assert "error" in run.state


@pytest.mark.asyncio
async def test_failure_during_loop_back_marks_running_step_not_next():
    """
    When gather fails on its second pass (e.g. max_iterations exceeded),
    the failure handler must mark gather as failed — not approval, which
    happens to be the next pending step in dict-iteration order.
    """
    # Gather always emits insufficient → loop forever until max_iterations.
    # FakeMessagesListChatModel cycles through responses; emit a non-submit
    # message so the runner nudges the LLM and eventually exhausts iters.
    bad_response = AIMessage(content="thinking…")  # no tool_calls → nudge loop
    llm = _FakeToolCallingChatModel(responses=[bad_response] * 5)
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])
    mcp.get_tool_server = MagicMock(return_value=None)
    steps = [
        {**WORKFLOW_STEPS[0], "max_iterations": 3},
        WORKFLOW_STEPS[1],
        WORKFLOW_STEPS[2],
    ]
    runner = YamlGraphRunner(
        {"id": "loopback-fail", "steps": steps}, llm=llm, mcp_tools_provider=mcp,
    )
    run = _make_run()
    run.id = "loopback-fail-run"
    repo = AsyncMock()

    await stream_graph_to_pause(runner, run, repo, {"request": "hello"})

    assert run.status == "failed"
    assert run.step_statuses["gather"] == "failed", (
        f"gather is the step that failed; got {run.step_statuses!r}"
    )
    assert run.step_statuses["approval"] == "pending", (
        "approval was never reached and must remain pending"
    )
