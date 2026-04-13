"""
Integration test: MCP fetch → LLM response pipeline.

Replaces: test_mcp_llm_http_flow.py  +  test_llm_tool_call_flow.py

Scenarios
─────────
1. MCP tool is called with the correct templated input derived from state.
2. MCP result is stored under output_key and visible in the final run state.
3. Subsequent LLM node receives the MCP result via user_template substitution
   and its answer is stored under its own output_key.
4. MCP tool error (exception) is captured gracefully — run still completes,
   output_key contains the error string.
5. Multi-step chain: two sequential MCP fetches, each feeding the next LLM.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.integration.conftest import build_int_client, make_mock_llm

_ISSUE_DATA = {"id": "PRJ-99", "summary": "Add dark mode", "priority": "high"}
_REPO_DATA = {"name": "acme/app", "default_branch": "main", "open_prs": 3}


def _mock_tool(return_value):
    t = MagicMock()
    t.ainvoke = AsyncMock(return_value=return_value)
    return t


def _error_tool():
    t = MagicMock()
    t.ainvoke = AsyncMock(side_effect=RuntimeError("upstream 503"))
    return t


# ---------------------------------------------------------------------------
# Graph definitions
# ---------------------------------------------------------------------------

_SINGLE_MCP_GRAPH = {
    "id": "mcp-then-llm",
    "steps": [
        {
            "id": "fetch_issue",
            "type": "mcp",
            "tool": "jira_get_issue",
            "tool_input": {"issue_id": "{request}"},
            "output_key": "issue_data",
        },
        {
            "id": "plan",
            "type": "llm",
            "output_key": "plan",
            "system_prompt": "Given the issue, produce an implementation plan.",
            "user_template": "Issue: {issue_data}\n\nRequest: {request}",
        },
    ],
}

_ERROR_MCP_GRAPH = {
    "id": "mcp-error",
    "steps": [
        {
            "id": "fetch",
            "type": "mcp",
            "tool": "broken_tool",
            "tool_input": {"q": "{request}"},
            "output_key": "data",
        },
        {
            "id": "answer",
            "type": "llm",
            "output_key": "result",
            "user_template": "{request}",
        },
    ],
}

_CHAINED_MCP_GRAPH = {
    "id": "chained-mcp",
    "steps": [
        {
            "id": "fetch_issue",
            "type": "mcp",
            "tool": "jira_get_issue",
            "tool_input": {"issue_id": "PRJ-99"},
            "output_key": "issue_data",
        },
        {
            "id": "fetch_repo",
            "type": "mcp",
            "tool": "github_get_repo",
            "tool_input": {"repo": "acme/app"},
            "output_key": "repo_data",
        },
        {
            "id": "plan",
            "type": "llm",
            "output_key": "plan",
            "system_prompt": "Combine issue and repo context.",
            "user_template": "Issue: {issue_data}\nRepo: {repo_data}\nRequest: {request}",
        },
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_called_with_templated_input() -> None:
    """
    MCP tool receives the rendered tool_input ('{request}' → actual value).
    """
    issue_tool = _mock_tool(_ISSUE_DATA)
    llm = make_mock_llm(text_responses=["plan based on issue"])
    client, mongo = await build_int_client(
        _SINGLE_MCP_GRAPH, llm, mcp_tools={"jira_get_issue": issue_tool}
    )
    try:
        resp = await client.post(
            "/api/v1/graphs/mcp-then-llm/runs",
            json={"request": "PRJ-99"},
        )
        assert resp.status_code == 200, resp.text
        issue_tool.ainvoke.assert_called_once_with({"issue_id": "PRJ-99"})
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_mcp_result_stored_in_state() -> None:
    """
    MCP tool output is stored under the declared output_key and is present
    in the final run state returned by the API.
    """
    issue_tool = _mock_tool(_ISSUE_DATA)
    llm = make_mock_llm(text_responses=["implementation plan"])
    client, mongo = await build_int_client(
        _SINGLE_MCP_GRAPH, llm, mcp_tools={"jira_get_issue": issue_tool}
    )
    try:
        resp = await client.post(
            "/api/v1/graphs/mcp-then-llm/runs",
            json={"request": "PRJ-99"},
        )
        state = resp.json()["state"]
        assert str(state["issue_data"]) == str(_ISSUE_DATA)
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_llm_receives_mcp_output_via_template() -> None:
    """
    The LLM node's user_template contains '{issue_data}'; the rendered prompt
    includes the actual MCP result, not the literal placeholder string.
    """
    issue_tool = _mock_tool(_ISSUE_DATA)
    received_prompts: list[str] = []

    original_ainvoke = AsyncMock(return_value=MagicMock(content="done"))
    llm = make_mock_llm(text_responses=["done"])

    # Capture what the LLM actually receives
    import langchain_core.messages as _lc_msgs  # noqa: PLC0415

    original_factory = llm.ainvoke.side_effect

    async def _capture(messages, **kwargs):
        for m in messages:
            if hasattr(m, "content"):
                received_prompts.append(m.content)
        return await original_factory(messages, **kwargs)

    llm.ainvoke = AsyncMock(side_effect=_capture)

    client, mongo = await build_int_client(
        _SINGLE_MCP_GRAPH, llm, mcp_tools={"jira_get_issue": issue_tool}
    )
    try:
        await client.post(
            "/api/v1/graphs/mcp-then-llm/runs",
            json={"request": "PRJ-99"},
        )
        # At least one prompt should contain the rendered issue data
        assert any(str(_ISSUE_DATA) in p for p in received_prompts), (
            f"LLM prompt did not contain issue data. Prompts: {received_prompts}"
        )
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_mcp_tool_error_captured_gracefully() -> None:
    """
    When an MCP tool raises, the error message is stored under output_key and
    the run still completes (no 500 from the API).
    """
    broken = _error_tool()
    llm = make_mock_llm(text_responses=["fallback answer"])
    client, mongo = await build_int_client(
        _ERROR_MCP_GRAPH, llm, mcp_tools={"broken_tool": broken}
    )
    try:
        resp = await client.post(
            "/api/v1/graphs/mcp-error/runs",
            json={"request": "anything"},
        )
        assert resp.status_code == 200, resp.text
        state = resp.json()["state"]
        assert "Error" in state["data"] or "upstream 503" in state["data"]
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_chained_mcp_both_results_in_state() -> None:
    """
    Two sequential MCP nodes each populate their own output_key; both are
    present in the final state and accessible to the downstream LLM node.
    """
    issue_tool = _mock_tool(_ISSUE_DATA)
    repo_tool = _mock_tool(_REPO_DATA)
    llm = make_mock_llm(text_responses=["combined plan"])
    client, mongo = await build_int_client(
        _CHAINED_MCP_GRAPH,
        llm,
        mcp_tools={"jira_get_issue": issue_tool, "github_get_repo": repo_tool},
    )
    try:
        resp = await client.post(
            "/api/v1/graphs/chained-mcp/runs",
            json={"request": "implement feature"},
        )
        assert resp.status_code == 200, resp.text
        state = resp.json()["state"]

        issue_tool.ainvoke.assert_called_once()
        repo_tool.ainvoke.assert_called_once()
        assert str(state["issue_data"]) == str(_ISSUE_DATA)
        assert str(state["repo_data"]) == str(_REPO_DATA)
        assert state["plan"] == "combined plan"
    finally:
        await mongo.close()
