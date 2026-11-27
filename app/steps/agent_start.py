"""The start hook of an agent step (a workflow step's ``on_start_publish``).

The workflow node sets ``agent_on_start`` around its executor call; an executor
calls ``fire_agent_on_start()`` at the moment the agent actually starts working
on the task -- after it got a slot of its runtime, when that runtime has a
concurrency limit -- and not again when it re-attaches after a restart or
resumes after a usage-limit pause. Hook failures never reach the step.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

agent_on_start: ContextVar[Callable[[], Awaitable[None]] | None] = ContextVar("agent_on_start", default=None)


async def fire_agent_on_start() -> None:
    hook = agent_on_start.get()
    if hook is None:
        return
    try:
        await hook()
    except Exception:
        logger.exception("agent start hook failed (step continues)")
