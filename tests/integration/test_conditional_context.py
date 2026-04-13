"""
Integration test: conditional MCP fetching driven by llm_structured output.

Replaces: test_classifier_flow.py

Scenarios
─────────
1. LLM classifies request as needing GitHub only →
   GitHub MCP tool called, Jira MCP tool skipped.
2. LLM classifies request as needing Jira only →
   Jira MCP tool called, GitHub MCP tool skipped.
3. Both needed → both tools called.
4. Neither needed → both tools skipped.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.integration.conftest import build_int_client, make_mock_llm

# ---------------------------------------------------------------------------
# Graph definition under test
# ---------------------------------------------------------------------------

_GRAPH = {
    "id": "conditional-context",
    "steps": [
        {
            "id": "classify",
            "type": "llm_structured",
            "system_prompt": "Decide which context sources are needed.",
            "user_template": "{request}",
            "output": [
                {"name": "needs_jira", "type": "bool", "description": "Jira context needed"},
                {"name": "needs_github", "type": "bool", "description": "GitHub context needed"},
            ],
        },
        {
            "id": "fetch_jira",
            "type": "mcp",
            "when": "needs_jira",
            "tool": "jira_search",
            "tool_input": {"query": "{request}"},
            "output_key": "jira_context",
        },
        {
            "id": "fetch_github",
            "type": "mcp",
            "when": "needs_github",
            "tool": "github_search",
            "tool_input": {"query": "{request}"},
            "output_key": "github_context",
        },
        {
            "id": "answer",
            "type": "llm",
            "output_key": "result",
            "system_prompt": "Summarise the context.",
            "user_template": "{request}",
        },
    ],
}

_GRAPH_ID = "conditional-context"

_JIRA_DATA = [{"id": "PRJ-42", "summary": "Dark mode request"}]
_GITHUB_DATA = {"repo": "acme/app", "stars": 120}


def _make_tools():
    jira_tool = MagicMock()
    jira_tool.ainvoke = AsyncMock(return_value=_JIRA_DATA)
    github_tool = MagicMock()
    github_tool.ainvoke = AsyncMock(return_value=_GITHUB_DATA)
    return jira_tool, github_tool


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_github_only() -> None:
    """Classifier picks GitHub → GitHub tool called, Jira skipped."""
    jira_tool, github_tool = _make_tools()
    llm = make_mock_llm(
        structured_responses=[{"needs_jira": False, "needs_github": True}],
        text_responses=["summary using github data"],
    )
    client, mongo = await build_int_client(
        _GRAPH, llm, mcp_tools={"jira_search": jira_tool, "github_search": github_tool}
    )
    try:
        resp = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "add dark mode to acme/app"},
        )
        assert resp.status_code == 200, resp.text
        state = resp.json()["state"]

        github_tool.ainvoke.assert_called_once_with({"query": "add dark mode to acme/app"})
        jira_tool.ainvoke.assert_not_called()

        assert str(state["github_context"]) == str(_GITHUB_DATA)
        assert "jira_context" not in state
        assert state["result"] == "summary using github data"
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_jira_only() -> None:
    """Classifier picks Jira → Jira tool called, GitHub skipped."""
    jira_tool, github_tool = _make_tools()
    llm = make_mock_llm(
        structured_responses=[{"needs_jira": True, "needs_github": False}],
        text_responses=["summary using jira data"],
    )
    client, mongo = await build_int_client(
        _GRAPH, llm, mcp_tools={"jira_search": jira_tool, "github_search": github_tool}
    )
    try:
        resp = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "check issues for PRJ-42"},
        )
        assert resp.status_code == 200, resp.text
        state = resp.json()["state"]

        jira_tool.ainvoke.assert_called_once_with({"query": "check issues for PRJ-42"})
        github_tool.ainvoke.assert_not_called()

        assert str(state["jira_context"]) == str(_JIRA_DATA)
        assert "github_context" not in state
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_both_needed() -> None:
    """Classifier picks both → both tools called."""
    jira_tool, github_tool = _make_tools()
    llm = make_mock_llm(
        structured_responses=[{"needs_jira": True, "needs_github": True}],
        text_responses=["summary using both"],
    )
    client, mongo = await build_int_client(
        _GRAPH, llm, mcp_tools={"jira_search": jira_tool, "github_search": github_tool}
    )
    try:
        resp = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "cross-reference jira and github"},
        )
        assert resp.status_code == 200, resp.text

        jira_tool.ainvoke.assert_called_once()
        github_tool.ainvoke.assert_called_once()
    finally:
        await mongo.close()


@pytest.mark.asyncio
async def test_neither_needed() -> None:
    """Classifier picks neither → both MCP steps skipped entirely."""
    jira_tool, github_tool = _make_tools()
    llm = make_mock_llm(
        structured_responses=[{"needs_jira": False, "needs_github": False}],
        text_responses=["answer from request alone"],
    )
    client, mongo = await build_int_client(
        _GRAPH, llm, mcp_tools={"jira_search": jira_tool, "github_search": github_tool}
    )
    try:
        resp = await client.post(
            f"/api/v1/graphs/{_GRAPH_ID}/runs",
            json={"request": "simple question"},
        )
        assert resp.status_code == 200, resp.text
        state = resp.json()["state"]

        jira_tool.ainvoke.assert_not_called()
        github_tool.ainvoke.assert_not_called()
        assert "jira_context" not in state
        assert "github_context" not in state
    finally:
        await mongo.close()
