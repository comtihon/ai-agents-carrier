from __future__ import annotations

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


@router.post("")
async def chat(body: ChatRequest, request: Request):
    """
    SSE streaming chat endpoint backed by the default LangGraph workflow.

    Event types emitted:
      {"type": "token",            "content": "..."}   — LLM token (reply path)
      {"type": "workflow_started", "workflow_id": "...",
                                   "workflow_name": "...",
                                   "run_id": "..."}    — workflow was spawned
      {"type": "done",             "thread_id": "..."}  — stream finished
      {"type": "error",            "message": "..."}   — execution error
    """
    default_graph = request.app.state.default_graph
    thread_id = body.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "messages": [HumanMessage(content=body.message)],
        "copilotkit": {"actions": [], "context": []},
    }

    async def event_stream():
        reply_had_tokens = False
        try:
            async for event in default_graph.astream_events(
                input_state, config, version="v2"
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                # Per-token LLM output from the reply node
                if kind == "on_chat_model_stream" and node == "reply":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        reply_had_tokens = True
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

                elif kind == "on_chain_end":
                    # Static reply (reply_text was set, no LLM streaming)
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

                    # Workflow was spawned
                    elif node == "spawn_workflow":
                        output = event["data"].get("output") or {}
                        spawned = output.get("spawned_workflow")
                        if spawned:
                            yield f"data: {json.dumps({'type': 'workflow_started', **spawned})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"

        except Exception as exc:
            logger.exception("Chat stream failed for thread %s", thread_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
