"""
Unit tests for the CopilotKit router agent.

Covers:
- _action_to_tool() conversion helper
- build_router_graph() compilation
- router_node LLM invocation (with and without frontend actions)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.infrastructure.orchestration.router_agent import _action_to_tool, build_router_graph


# ── _action_to_tool ───────────────────────────────────────────────────────────

def test_action_to_tool_basic():
    action = {
        "name": "startWorkflow",
        "description": "Start a workflow run",
        "parameters": [
            {"name": "workflowId", "type": "string", "description": "Workflow ID", "required": True},
            {"name": "userRequest", "type": "string", "description": "What to do", "required": True},
        ],
    }
    tool = _action_to_tool(action)

    assert tool["type"] == "function"
    fn = tool["function"]
    assert fn["name"] == "startWorkflow"
    assert fn["description"] == "Start a workflow run"

    params = fn["parameters"]
    assert params["type"] == "object"
    assert "workflowId" in params["properties"]
    assert params["properties"]["workflowId"]["type"] == "string"
    assert params["properties"]["workflowId"]["description"] == "Workflow ID"
    assert set(params["required"]) == {"workflowId", "userRequest"}


def test_action_to_tool_optional_params():
    action = {
        "name": "approveWorkflow",
        "parameters": [
            {"name": "runId", "type": "string", "required": True},
            {"name": "feedback", "type": "string"},  # not required
        ],
    }
    tool = _action_to_tool(action)
    fn = tool["function"]
    assert fn["parameters"]["required"] == ["runId"]
    assert "feedback" in fn["parameters"]["properties"]


def test_action_to_tool_no_parameters():
    action = {"name": "reviewPendingApprovals", "description": "Show pending approvals"}
    tool = _action_to_tool(action)
    fn = tool["function"]
    assert fn["parameters"]["properties"] == {}
    assert "required" not in fn["parameters"]


def test_action_to_tool_missing_description_defaults_empty():
    action = {"name": "doSomething", "parameters": []}
    tool = _action_to_tool(action)
    assert tool["function"]["description"] == ""


def test_action_to_tool_preserves_type_default():
    action = {
        "name": "test",
        "parameters": [{"name": "x"}],  # no "type" field
    }
    tool = _action_to_tool(action)
    assert tool["function"]["parameters"]["properties"]["x"]["type"] == "string"


# ── build_router_graph ────────────────────────────────────────────────────────

def test_build_router_graph_returns_compiled_graph():
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="Hello!")])
    graph = build_router_graph(llm)
    assert graph is not None
    # Should be a CompiledStateGraph — has ainvoke
    assert callable(getattr(graph, "ainvoke", None))


# ── router_node invocation ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_router_node_returns_ai_message():
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="Sure, I can help!")])
    graph = build_router_graph(llm)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="What can you do?")],
            "copilotkit": {"actions": [], "context": []},
        }
    )

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_msgs) == 1
    assert ai_msgs[0].content == "Sure, I can help!"


@pytest.mark.asyncio
async def test_router_node_without_actions_uses_plain_llm():
    """When no frontend actions are provided, bind_tools should NOT be called."""
    mock_response = AIMessage(content="Direct answer")
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.bind_tools = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    graph = build_router_graph(mock_llm)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Hello")],
            "copilotkit": {"actions": [], "context": []},
        }
    )

    mock_llm.bind_tools.assert_not_called()
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_router_node_with_actions_binds_tools():
    """When frontend actions are present, the LLM should receive them as tools."""
    mock_response = AIMessage(content="Starting workflow")
    bound_llm = MagicMock(spec=BaseChatModel)
    bound_llm.ainvoke = AsyncMock(return_value=mock_response)

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.bind_tools = MagicMock(return_value=bound_llm)

    graph = build_router_graph(mock_llm)

    actions = [
        {
            "name": "startWorkflow",
            "description": "Start a workflow",
            "parameters": [
                {"name": "workflowId", "type": "string", "required": True},
                {"name": "userRequest", "type": "string", "required": True},
            ],
        }
    ]

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Run the dev-assistant workflow")],
            "copilotkit": {"actions": actions, "context": []},
        }
    )

    mock_llm.bind_tools.assert_called_once()
    # The tools passed to bind_tools should be OpenAI-style dicts
    tools_arg = mock_llm.bind_tools.call_args[0][0]
    assert len(tools_arg) == 1
    assert tools_arg[0]["function"]["name"] == "startWorkflow"
    bound_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_router_node_system_prompt_prepended():
    """System message should be the first message sent to the LLM."""
    from langchain_core.messages import SystemMessage

    captured: list = []

    async def capturing_ainvoke(messages, **kwargs):
        captured.extend(messages)
        return AIMessage(content="ok")

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.ainvoke = capturing_ainvoke

    graph = build_router_graph(mock_llm)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="hi")],
            "copilotkit": {"actions": [], "context": []},
        }
    )

    assert len(captured) >= 2
    assert isinstance(captured[0], SystemMessage)
    assert isinstance(captured[1], HumanMessage)
