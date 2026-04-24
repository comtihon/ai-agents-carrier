from __future__ import annotations

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ResumeChatRequest(BaseModel):
    thread_id: str
    answers: dict[str, str]


def _find_ask_context_interrupt(state) -> dict | None:
    """Return the ask_context interrupt payload if the graph is paused waiting for context."""
    for task in (state.tasks or []):
        for intr in (getattr(task, "interrupts", None) or []):
            val = getattr(intr, "value", None)
            if isinstance(val, dict) and val.get("type") == "ask_context":
                return val
    return None


async def _stream_graph(graph, input_value, config: dict, thread_id: str):
    """
    Shared SSE generator for both the chat and resume endpoints.

    Event types emitted:
      {"type": "token",            "content": "..."}
      {"type": "workflow_started", "workflow_id": "...", "workflow_name": "...", "run_id": "..."}
      {"type": "ask_context",      "questions": [...], "thread_id": "..."}
      {"type": "done",             "thread_id": "..."}
      {"type": "error",            "message": "..."}
    """
    reply_had_tokens = False
    try:
        async for event in graph.astream_events(input_value, config, version="v2"):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            if kind == "on_chat_model_stream" and node == "reply":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    reply_had_tokens = True
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

            elif kind == "on_chain_end":
                if node == "reply" and not reply_had_tokens:
                    output = event["data"].get("output") or {}
                    for msg in output.get("messages", []):
                        content = (
                            msg.content
                            if hasattr(msg, "content")
                            else (msg.get("content", "") if isinstance(msg, dict) else "")
                        )
                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                elif node == "spawn_workflow":
                    output = event["data"].get("output") or {}
                    spawned = output.get("spawned_workflow")
                    if spawned:
                        yield f"data: {json.dumps({'type': 'workflow_started', **spawned})}\n\n"

        # Check whether the graph paused at an ask_context interrupt
        state = await graph.aget_state(config)
        interrupt_payload = _find_ask_context_interrupt(state)
        if interrupt_payload:
            questions = interrupt_payload.get("questions", [])
            yield f"data: {json.dumps({'type': 'ask_context', 'questions': questions, 'thread_id': thread_id})}\n\n"
            return  # Don't emit 'done' — the conversation is paused

        yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"

    except Exception as exc:
        logger.exception("Chat stream failed for thread %s", thread_id)
        yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"


@router.post("")
async def chat(body: ChatRequest, request: Request):
    """SSE streaming chat endpoint backed by the default LangGraph workflow."""
    default_graph = request.app.state.default_graph
    thread_id = body.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "messages": [HumanMessage(content=body.message)],
        "copilotkit": {"actions": [], "context": []},
    }

    return StreamingResponse(
        _stream_graph(default_graph, input_state, config, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/resume")
async def resume_chat(body: ResumeChatRequest, request: Request):
    """
    Resume a graph that is paused at an ask_context interrupt.

    Accepts the user's answers (keyed by str(question_index)) and resumes
    execution via Command(resume=answers).  Streams the same SSE events as /chat.
    """
    default_graph = request.app.state.default_graph
    config = {"configurable": {"thread_id": body.thread_id}}

    return StreamingResponse(
        _stream_graph(default_graph, Command(resume=body.answers), config, body.thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
