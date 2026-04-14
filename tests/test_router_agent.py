"""
Unit tests for the CopilotKit router agent.

The router agent is a pure conversational assistant — it no longer
converts frontend actions to tools.  Workflow operations are handled
by CopilotKit backend actions registered in app.api.app.

Covers:
- build_router_graph() compiles a runnable graph
- router_node LLM invocation (system prompt, messages, no tool binding)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.infrastructure.orchestration.router_agent import build_router_graph


# ── build_router_graph ────────────────────────────────────────────────────────

def test_build_router_graph_returns_compiled_graph():
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="Hello!")])
    graph = build_router_graph(llm)
    assert graph is not None
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
async def test_router_node_does_not_bind_tools():
    """The simplified router never calls bind_tools regardless of provided actions."""
    mock_response = AIMessage(content="Direct answer")
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.bind_tools = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    graph = build_router_graph(mock_llm)

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Hello")],
            "copilotkit": {
                "actions": [
                    {
                        "name": "startWorkflow",
                        "parameters": [{"name": "workflowId", "type": "string", "required": True}],
                    }
                ],
                "context": [],
            },
        }
    )

    mock_llm.bind_tools.assert_not_called()
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_router_node_system_prompt_prepended():
    """System message must be the first message sent to the LLM."""
    captured: list = []

    async def capturing_ainvoke(messages, **kwargs):
        captured.extend(messages)
        return AIMessage(content="ok")

    mock_llm = MagicMock(spec=BaseChatModel)
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


@pytest.mark.asyncio
async def test_router_node_passes_all_messages():
    """All conversation messages are forwarded to the LLM after the system prompt."""
    captured: list = []

    async def capturing_ainvoke(messages, **kwargs):
        captured.extend(messages)
        return AIMessage(content="ok")

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.ainvoke = capturing_ainvoke

    graph = build_router_graph(mock_llm)

    await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="first"),
                AIMessage(content="response"),
                HumanMessage(content="second"),
            ],
            "copilotkit": {"actions": [], "context": []},
        }
    )

    # system + 3 conversation messages
    assert len(captured) == 4
    assert isinstance(captured[0], SystemMessage)
