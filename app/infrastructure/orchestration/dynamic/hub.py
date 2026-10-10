"""The attempts a dynamic step has in flight, and the events they produce.

Agents run for minutes to hours, so the job graph does not block on them: its
``schedule`` node launches attempts here and its ``wait`` node takes the next
events — an attempt finished, a running agent asked a question — and hands
them back to ``schedule``. That is what lets the meta-agent answer an agent,
message or stop one, or rearrange the DAG while other agents keep working.

A hub lives in process memory, one per run and dynamic step, and outlives a
pause at a human gate: the attempts keep running and their results queue
until the graph resumes. A restart loses it; the graph then finds attempts it
believes running without a live task and either adopts the result they
recorded on the run before the restart or runs them again.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables.config import var_child_runnable_config

from app.infrastructure.orchestration.dynamic.engine import (
    AttemptResult,
    Delivery,
    JobOutcome,
    JobRunner,
    Launch,
    LiveQuestion,
)

logger = logging.getLogger(__name__)

# (job_id, attempt n, child run id, result or None while running)
OnAttempt = Callable[[str, int, str | None, AttemptResult | None], Awaitable[None]]
# The result an attempt recorded before a restart, if any.
Recorded = Callable[[str, int], AttemptResult | None]
# Deliver an answer to the agent of a child run that waits on its question.
Deliver = Callable[[str, str], Awaitable[None]]


class JobHub:
    def __init__(self, key: tuple[str, str] = ("", "")) -> None:
        self.key = key
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._tasks: dict[tuple[str, int], asyncio.Task] = {}
        # Attempts whose result is queued but not yet taken by the graph.
        self._queued: set[tuple[str, int]] = set()
        self._children: dict[str, tuple[str, int]] = {}
        self.job_runner: JobRunner | None = None
        self.on_attempt: OnAttempt | None = None
        self.recorded: Recorded | None = None
        self.deliver_answer: Deliver | None = None

    def configure(
        self,
        *,
        job_runner: JobRunner,
        on_attempt: OnAttempt | None = None,
        recorded: Recorded | None = None,
        deliver_answer: Deliver | None = None,
    ) -> None:
        self.job_runner = job_runner
        self.on_attempt = on_attempt
        self.recorded = recorded
        self.deliver_answer = deliver_answer

    # ── attempts ────────────────────────────────────────────────────────────

    def launch(self, launch: Launch) -> None:
        key = (launch.job.id, launch.attempt.n)
        if key in self._tasks and not self._tasks[key].done():
            return
        # The attempt runs outside the graph's superstep: it must not see the
        # node's LangGraph config (an interrupt() in it would hit the node's
        # scratchpad).
        ctx = contextvars.copy_context()
        ctx.run(var_child_runnable_config.set, None)
        self._tasks[key] = asyncio.get_running_loop().create_task(self._run(launch), context=ctx)

    def cancel(self, job_id: str, n: int) -> None:
        task = self._tasks.get((job_id, n))
        if task is not None and not task.done():
            task.cancel()

    def alive(self, job_id: str, n: int) -> bool:
        """The attempt runs here, or its result waits to be taken."""
        task = self._tasks.get((job_id, n))
        return (task is not None and not task.done()) or (job_id, n) in self._queued

    def recorded_result(self, job_id: str, n: int) -> AttemptResult | None:
        return self.recorded(job_id, n) if self.recorded is not None else None

    async def _run(self, launch: Launch) -> None:
        job, attempt = launch.job, launch.attempt
        done = self.recorded_result(job.id, attempt.n)
        if done is not None:
            self._put_result(done)
            return
        child: dict[str, str] = {}

        async def started(child_run_id: str) -> None:
            child["id"] = child_run_id
            self._children[child_run_id] = (job.id, attempt.n)
            if self.on_attempt is not None:
                await self.on_attempt(job.id, attempt.n, child_run_id, None)

        assert self.job_runner is not None, "JobHub.configure() was not called"
        fatal: BaseException | None = None
        try:
            outcome = await self.job_runner(job, attempt, launch.payload, started)
        except asyncio.CancelledError:
            outcome = JobOutcome(status="failed", error="cancelled")
        except Exception as exc:  # the runner should not raise, but a crash is just a failed attempt
            logger.exception("dynamic job %s crashed", job.id)
            outcome = JobOutcome(status="failed", error=f"{type(exc).__name__}: {exc}")
        except BaseException as exc:  # still report, so the graph is not left waiting
            fatal = exc
            outcome = JobOutcome(status="failed", error=f"{type(exc).__name__}: {exc}")
        result = AttemptResult(**outcome.model_dump(), job_id=job.id, attempt=attempt.n, child_run_id=child.get("id"))
        if self.on_attempt is not None:
            try:
                await asyncio.shield(self.on_attempt(job.id, attempt.n, child.get("id"), result))
            except Exception:
                logger.exception("dynamic job %s: recording the result failed", job.id)
        self._children.pop(child.get("id", ""), None)
        self._put_result(result)
        if fatal is not None:
            raise fatal

    def _put_result(self, result: AttemptResult) -> None:
        self._queued.add((result.job_id, result.attempt))
        self._queue.put_nowait(result.model_dump(mode="json"))

    # ── live questions ──────────────────────────────────────────────────────

    def owns_child(self, child_run_id: str) -> bool:
        return child_run_id in self._children

    def ask(self, child_run_id: str, question: str) -> bool:
        """A running agent asked a question; False when the child run is not ours."""
        found = self._children.get(child_run_id)
        if found is None:
            return False
        job_id, n = found
        self._queue.put_nowait(LiveQuestion(job_id=job_id, attempt=n, child_run_id=child_run_id, question=question).model_dump(mode="json"))
        return True

    async def deliver(self, delivery: Delivery) -> None:
        if self.deliver_answer is None or not delivery.child_run_id:
            logger.warning("dynamic job %s: no channel to deliver the answer", delivery.job_id)
            return
        try:
            await self.deliver_answer(delivery.child_run_id, delivery.text)
        except Exception:
            logger.exception("dynamic job %s: delivering the answer failed", delivery.job_id)

    # ── events ──────────────────────────────────────────────────────────────

    def push(self, event: dict[str, Any]) -> None:
        self._queue.put_nowait(event)

    async def next_events(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Wait for at least one event (up to *timeout*), then take every one queued."""
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout) if timeout else await self._queue.get()
        except asyncio.TimeoutError:
            return []
        events = [first]
        while not self._queue.empty():
            events.append(self._queue.get_nowait())
        for event in events:
            if event.get("type", "result") == "result":
                self._queued.discard((event["job_id"], event["attempt"]))
        return events

    def close(self) -> None:
        for task in self._tasks.values():
            if not task.done():
                task.cancel()
        self._tasks.clear()
        self._queued.clear()
        self._children.clear()


# ─── registry ────────────────────────────────────────────────────────────────

_HUBS: dict[tuple[str, str], JobHub] = {}


def hub_for(run_id: str, step_id: str) -> JobHub:
    key = (run_id, step_id)
    if key not in _HUBS:
        _HUBS[key] = JobHub(key)
    return _HUBS[key]


def drop_hub(run_id: str, step_id: str) -> None:
    hub = _HUBS.pop((run_id, step_id), None)
    if hub is not None:
        hub.close()


def close_hubs(run_id: str) -> None:
    """Stop every attempt of a run (terminate)."""
    for key in [k for k in _HUBS if k[0] == run_id]:
        drop_hub(*key)


def route_live_question(parent_run_id: str, child_run_id: str, question: str) -> bool:
    """Hand a running job agent's question to its dynamic step; False when no hub owns it."""
    for (run_id, _), hub in list(_HUBS.items()):
        if run_id == parent_run_id and hub.ask(child_run_id, question):
            return True
    return False
