"""
Integration test: llm_structured conditional routing with a revision loop.

Workflow under test  (code-review-loop):
  plan (llm) → execute (llm_structured) → code_review (llm_structured)
                      ↑                           |
                      └───── needs revision ───────┘
                                                  |
                                          passed → notify (http_call)

Scenario
────────
1. plan runs once.
2. execute runs → code_review detects a bug → routes back to execute.
3. execute runs a second time (fix) → code_review passes → routes to notify.
4. notify (http_call) runs to completion (graceful error on localhost, run still completes).

Visit counts after the run must be: execute=2, code_review=2.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.integration.conftest import build_int_client, make_mock_llm

_WORKFLOW = {
    "id": "code-review-loop",
    "max_iterations": 5,
    "steps": [
        {
            "id": "plan",
            "type": "llm",
            "output_key": "plan",
            "system_prompt": "Create a brief plan.",
            "user_template": "{request}",
        },
        {
            "id": "execute",
            "type": "llm_structured",
            "max_loops": 3,
            "system_prompt": "Implement the plan.",
            "user_template": "Plan: {plan}",
            "output": [
                {"name": "result", "type": "str", "description": "What was implemented"},
                {"name": "code",   "type": "str", "description": "The code written"},
            ],
        },
        {
            "id": "code_review",
            "type": "llm_structured",
            "max_loops": 3,
            "system_prompt": "Review the implementation.",
            "user_template": "Plan: {plan}\nResult: {result}",
            "output": [
                {"name": "feedback", "type": "str",  "description": "Review feedback"},
                {"name": "passed",   "type": "bool", "description": "True if it passes"},
            ],
            "routes": [
                {"when": "passed", "next": "notify"},
                {"next": "execute"},
            ],
        },
        {
            "id": "notify",
            "type": "http_call",
            "url": "http://localhost:19999/notify",  # unreachable — fails gracefully
            "method": "POST",
            "body": {"result": "{result}"},
            "output_key": "notification",
        },
    ],
}


@pytest.mark.asyncio
async def test_code_review_loop_with_revision() -> None:
    """
    execute runs twice (revision loop), then code_review passes and routes to notify.
    Visit counts must reflect 2 runs of execute and 2 runs of code_review.
    """
    llm = make_mock_llm(
        text_responses=["step-by-step plan"],
        structured_responses=[
            # execute run 1 — has a bug
            {"result": "implementation v1 (has a bug)", "code": "def f(): pass  # bug"},
            # code_review run 1 — rejects, routes back to execute
            {"feedback": "Found a bug in the implementation", "passed": False},
            # execute run 2 — fixed
            {"result": "fixed implementation", "code": "def f(): return 42"},
            # code_review run 2 — passes, routes to notify
            {"feedback": "LGTM, implementation is correct", "passed": True},
        ],
    )

    client, mongo = await build_int_client(_WORKFLOW, llm)
    try:
        resp = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "code-review-loop", "user_request": "implement feature X"},
        )
        assert resp.status_code == 200, resp.text
        run_id = resp.json()["id"]

        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        assert get_resp.status_code == 200
        body = get_resp.json()

        assert body["status"] == "completed", f"Expected completed, got {body['status']}: {body.get('intermediate_outputs')}"

        outputs = body["intermediate_outputs"]
        # Final state reflects the second (successful) execute + code_review
        assert outputs["plan"] == "step-by-step plan"
        assert outputs["result"] == "fixed implementation"
        assert outputs["feedback"] == "LGTM, implementation is correct"
        assert outputs["passed"] is True

        # Visit counts: execute × 2, code_review × 2
        counts = outputs.get("_visit_counts", {})
        assert counts.get("execute") == 2,      f"execute visit count: {counts}"
        assert counts.get("code_review") == 2,  f"code_review visit count: {counts}"
        assert counts.get("plan") == 1,         f"plan visit count: {counts}"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_code_review_passes_first_time() -> None:
    """
    If code_review passes on the first attempt, no loop occurs.
    Visit counts: execute=1, code_review=1.
    """
    llm = make_mock_llm(
        text_responses=["short plan"],
        structured_responses=[
            {"result": "perfect implementation", "code": "def f(): return 42"},
            {"feedback": "LGTM", "passed": True},
        ],
    )

    client, mongo = await build_int_client(_WORKFLOW, llm)
    try:
        resp = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "code-review-loop", "user_request": "build something simple"},
        )
        assert resp.status_code == 200
        run_id = resp.json()["id"]

        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = get_resp.json()
        assert body["status"] == "completed"

        counts = body["intermediate_outputs"].get("_visit_counts", {})
        assert counts.get("execute") == 1
        assert counts.get("code_review") == 1
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_loop_guard_fails_run_when_max_loops_exceeded() -> None:
    """
    If code_review always rejects, the run must fail after max_loops=3 executions
    of the execute step (4th attempt raises ValueError → run.status='failed').
    """
    llm = make_mock_llm(
        text_responses=["plan"],
        structured_responses=[
            # execute runs 1, 2, 3 (max_loops=3) — each followed by reject
            {"result": "bad impl", "code": "broken"},
            {"feedback": "still broken", "passed": False},
            {"result": "bad impl", "code": "broken"},
            {"feedback": "still broken", "passed": False},
            {"result": "bad impl", "code": "broken"},
            {"feedback": "still broken", "passed": False},
            # 4th attempt at execute — loop guard fires
            {"result": "bad impl", "code": "broken"},
        ],
    )

    client, mongo = await build_int_client(_WORKFLOW, llm)
    try:
        resp = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "code-review-loop", "user_request": "write broken code"},
        )
        assert resp.status_code == 200
        run_id = resp.json()["id"]

        get_resp = await client.get(f"/api/v1/workflows/runs/{run_id}")
        body = get_resp.json()
        assert body["status"] == "failed", f"Expected failed, got: {body['status']}"

        # The error message should mention max_loops
        error = body.get("error") or body.get("intermediate_outputs", {}).get("error", "")
        assert "max_loops" in str(error).lower() or "exceeded" in str(error).lower(), \
            f"Expected max_loops error, got: {error}"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_single_output_non_llm_structured_raises() -> None:
    """
    A non-llm_structured step with more than 1 route must raise ValueError at build time.
    """
    from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
    from unittest.mock import MagicMock
    from app.infrastructure.tools.mcp_client import McpToolsProvider
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    bad_graph = {
        "id": "bad",
        "steps": [
            {
                "id": "step1",
                "type": "llm",
                "output_key": "out",
                "routes": [{"next": "step2"}, {"next": "step3"}],  # INVALID: llm can't branch
            },
            {"id": "step2", "type": "llm", "output_key": "out2"},
            {"id": "step3", "type": "llm", "output_key": "out3"},
        ],
    }
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="x")])

    with pytest.raises(ValueError, match="cannot have more than 1 route"):
        YamlGraphRunner(bad_graph, llm=llm, mcp_tools_provider=mcp)


def _make_agent_route_runner(verdict: str) -> YamlGraphRunner:
    """Build a runner with a langgraph-agent step that branches on `verdict`.

    execute_agent_step is mocked to return {"verdict": verdict} directly
    (as if output_mapping already merged the agent's output into state).
    """
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
    from app.infrastructure.tools.mcp_client import McpToolsProvider

    definition = {
        "id": "agent-route-graph",
        "steps": [
            {
                "id": "agent_step",
                "type": "langgraph-agent",
                "agent_id": "test-agent",
                "output_mapping": {"verdict": "verdict"},
                "routes": [
                    {"when": "verdict == 'approve'", "next": "route_a"},
                    {"next": "route_b"},
                ],
            },
            # Explicit terminal `next` (pointing outside the graph) so each branch
            # ends the run instead of auto-chaining into the sibling branch.
            {"id": "route_a", "type": "llm", "output_key": "a_out", "next": "__end__"},
            {"id": "route_b", "type": "llm", "output_key": "b_out", "next": "__end__"},
        ],
    }
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="branch output")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner(definition, llm=llm, mcp_tools_provider=mcp)
    runner._agent_backend = MagicMock()
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdict", "expected_key"),
    [("approve", "a_out"), ("reject", "b_out")],
)
async def test_agent_step_multi_route_branches(verdict: str, expected_key: str) -> None:
    """A langgraph-agent step with 2 routes builds without error and branches
    on the merged output_mapping field, same as switch/llm_structured."""
    runner = _make_agent_route_runner(verdict)

    with patch(
        "app.steps.agent_executor.execute_agent_step",
        new=AsyncMock(return_value={"verdict": verdict}),
    ):
        config = {"configurable": {"thread_id": f"agent-route-{verdict}"}}
        state = await runner.graph.ainvoke({"request": "go"}, config)

    assert state.get(expected_key) == "branch output"
    other_key = "b_out" if expected_key == "a_out" else "a_out"
    assert other_key not in state


@pytest.mark.asyncio
async def test_agent_step_single_next_backward_compat() -> None:
    """An agent step using plain `next` (no routes) still works unchanged."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
    from app.infrastructure.tools.mcp_client import McpToolsProvider

    definition = {
        "id": "agent-next-graph",
        "steps": [
            {
                "id": "agent_step",
                "type": "langgraph-agent",
                "agent_id": "test-agent",
                "output_key": "agent_out",
                "next": "after_agent",
            },
            {"id": "after_agent", "type": "llm", "output_key": "after_out"},
        ],
    }
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner(definition, llm=llm, mcp_tools_provider=mcp)
    runner._agent_backend = MagicMock()

    with patch(
        "app.steps.agent_executor.execute_agent_step",
        new=AsyncMock(return_value={"agent_out": {"ok": True}}),
    ):
        config = {"configurable": {"thread_id": "agent-next"}}
        state = await runner.graph.ainvoke({"request": "go"}, config)

    assert state.get("agent_out") == {"ok": True}
    assert state.get("after_out") == "done"


@pytest.mark.asyncio
async def test_agent_step_token_usage_sums_across_loopback_executions() -> None:
    """Regression test for the ``_sum_usage`` reducer (yaml_graph.py).

    A ``langgraph-agent`` step that loops back onto itself (via ``routes``)
    must accumulate ``_agent_token_usage_<step>`` across re-executions
    instead of the last write clobbering the first — this is the exact
    scenario a plain ``LastValue`` field (or a hand-rolled
    ``current_state.update(output)`` failure-path merge) would undercount:
    150 (first execution) + 15 (second execution) must sum to 165, not
    regress to 15.
    """
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
    from app.infrastructure.tools.mcp_client import McpToolsProvider

    definition = {
        "id": "agent-loop-usage-graph",
        "steps": [
            {
                "id": "agent_step",
                "type": "langgraph-agent",
                "agent_id": "test-agent",
                "routes": [
                    {"when": "passed", "next": "__end__"},
                    {"next": "agent_step"},
                ],
            },
        ],
    }
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="unused")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner(definition, llm=llm, mcp_tools_provider=mcp)
    runner._agent_backend = MagicMock()

    # Each execution writes its own delta for the same field; the state
    # schema's Annotated[Any, _sum_usage] reducer must sum them, not
    # overwrite.
    execute_agent_step_mock = AsyncMock(
        side_effect=[
            {
                "passed": False,
                "_agent_token_usage_agent_step": {
                    "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                },
            },
            {
                "passed": True,
                "_agent_token_usage_agent_step": {
                    "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                },
            },
        ]
    )
    with patch(
        "app.steps.agent_executor.execute_agent_step",
        new=execute_agent_step_mock,
    ):
        config = {"configurable": {"thread_id": "agent-loop-usage"}}
        state = await runner.graph.ainvoke({"request": "go"}, config)

    assert execute_agent_step_mock.await_count == 2
    assert state.get("_agent_token_usage_agent_step") == {
        "input_tokens": 110, "output_tokens": 55, "total_tokens": 165,
    }


@pytest.mark.asyncio
async def test_stream_graph_to_pause_failure_persists_checkpoint_usage_not_handrolled_overwrite() -> None:
    """Regression test for the BLOCKER fix in ``stream_graph_to_pause``
    (yaml_graph.py): on failure, ``run.state`` must be built from the
    checkpointer's reducer-applied values (``runner.graph.aget_state``), not
    the hand-rolled ``current_state.update(output)`` accumulation.

    ``current_state.update()`` overwrites a repeated key with the LAST
    chunk's value (15), losing the earlier chunk's contribution (150) that a
    ``_sum_usage``-reducer field would have summed (165) — this mirrors the
    validator's manual repro (165 via checkpointer vs 15 via failure path).

    ``runner.graph.astream``/``aget_state`` are stubbed directly so this test
    exercises exactly the failure-handler code path without depending on
    incidental node-level exception-swallowing behavior elsewhere in the
    runner.
    """
    from datetime import datetime, timezone
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from app.domain.models.graph_run import GraphRun
    from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
    from app.infrastructure.tools.mcp_client import McpToolsProvider

    llm = FakeMessagesListChatModel(responses=[AIMessage(content="unused")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner(
        {"id": "usage-fail-graph", "steps": [{"id": "step1", "type": "llm", "output_key": "out"}]},
        llm=llm, mcp_tools_provider=mcp,
    )

    run = GraphRun(
        id="usage-fail-run",
        graph_id="usage-fail-graph",
        user_request="hello",
        status="running",
        current_step=None,
        state={},
        step_inputs={},
        step_outputs={},
        step_statuses={"step1": "pending"},
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )
    repo = AsyncMock()

    async def fake_astream(*_args, **_kwargs):
        # Two "successful" loop-back chunks writing partial deltas for the
        # same _sum_usage-reducer field, then a stream failure.
        yield {"step1": {"_judge_token_usage": {
            "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
        }}}
        yield {"step1": {"_judge_token_usage": {
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        }}}
        raise RuntimeError("simulated stream failure after loop-back")

    checkpoint_snapshot = MagicMock()
    checkpoint_snapshot.values = {
        "_judge_token_usage": {"input_tokens": 110, "output_tokens": 55, "total_tokens": 165},
    }
    runner.graph.astream = fake_astream
    runner.graph.aget_state = AsyncMock(return_value=checkpoint_snapshot)

    await stream_graph_to_pause(runner, run, repo, {"request": "hello"})

    assert run.status == "failed"
    assert run.state["_judge_token_usage"] == {
        "input_tokens": 110, "output_tokens": 55, "total_tokens": 165,
    }
