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
