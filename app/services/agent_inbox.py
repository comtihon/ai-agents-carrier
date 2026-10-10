"""The answer channel to a running agent that asked a question.

An agent asks through ``POST /runs/{run_id}/agent/question`` and long-polls
``GET /runs/{run_id}/agent/input`` for the answer. The answer arrives from a
human (``POST /agent/reply``) or, for a dynamic-workflow job, from the
meta-agent. Both go through ``deliver_answer``: it wakes a long-poll waiting in
this process and persists the answer on the run, so a poll that arrives later
(or reaches another replica) still gets it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Keyed by run id; shared by every coroutine of the process.
answer_events: dict[str, asyncio.Event] = {}
answers: dict[str, str] = {}
questions: dict[str, dict[str, Any]] = {}


def event_for(run_id: str) -> asyncio.Event:
    if run_id not in answer_events:
        answer_events[run_id] = asyncio.Event()
    return answer_events[run_id]


async def deliver_answer(run_repository: Any, run_id: str, answer: str) -> None:
    """Hand *answer* to the agent of *run_id* that waits on its question."""
    answers[run_id] = answer
    event_for(run_id).set()
    if run_repository is None:
        return
    run = await run_repository.get(run_id)
    if run is None:
        return
    run.state = {
        **{k: v for k, v in (run.state or {}).items() if k != "_pending_question"},
        "_pending_answer": answer,
    }
    run.touch()
    await run_repository.update(run)
    logger.info("run %s: answer stored and event set", run_id)


# ─── Messages to a running agent ─────────────────────────────────────────────
# Unlike answers, nobody asked for these: the orchestrator tells a running agent
# something new. Agents that hold a session open between turns (ACP) take them
# as their next turn — or, when urgent, have the current turn cancelled for
# them. Messages still queued when the attempt ends were never read.

messages: dict[str, list[dict[str, Any]]] = {}
message_events: dict[str, asyncio.Event] = {}


def message_event(run_id: str) -> asyncio.Event:
    if run_id not in message_events:
        message_events[run_id] = asyncio.Event()
    return message_events[run_id]


def push_message(run_id: str, text: str, *, interrupt: bool = False) -> None:
    messages.setdefault(run_id, []).append({"text": text, "interrupt": interrupt})
    message_event(run_id).set()


def has_urgent_message(run_id: str) -> bool:
    return any(m.get("interrupt") for m in messages.get(run_id, []))


def take_messages(run_id: str) -> list[dict[str, Any]]:
    """Every queued message for *run_id*, oldest first; the queue is emptied."""
    taken = messages.pop(run_id, [])
    event = message_events.get(run_id)
    if event is not None:
        event.clear()
    return taken


def discard(run_id: str) -> None:
    messages.pop(run_id, None)
    message_events.pop(run_id, None)
