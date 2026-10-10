"""The job DAG as a LangGraph graph.

    START ─► schedule ─┬─ Send × N ─► run_job ─┐
               ▲       ├─ gate (interrupt) ─────┤
               │       └─ END                   │
               └────────────────────────────────┘

``schedule`` folds the finished wave's outcomes (and a gate's resolution) into
the DAG through ``JobDag`` and fans the next wave out with one ``Send`` per job
attempt, so parallel parts are parallel LangGraph tasks. ``run_job`` runs one
attempt; ``gate`` pauses on ``interrupt()`` until a human decides. Loops — a
validator's failed review handing work back, a retry, an answered question —
are ``schedule`` sending the same job again with its next attempt number.

The DAG lives in the graph's state and is checkpointed after every wave, so a
paused or restarted run picks up where it stopped. The ``persist`` callback
mirrors it onto the run document for the UI; ``on_attempt`` records each
attempt's child run and result as soon as they exist, which is also what keeps
a finished attempt from running twice when a wave is replayed after a restart.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from app.domain.models.dynamic import DynamicConfig, DynamicRunState, Job, JobAttempt
from app.infrastructure.orchestration.dynamic.dispatcher import Dispatcher
from app.infrastructure.orchestration.dynamic.engine import (
    AttemptResult,
    JobDag,
    JobOutcome,
    JobRunner,
)

logger = logging.getLogger(__name__)

Persist = Callable[[DynamicRunState], Awaitable[None]]
# (job_id, attempt n, child run id, result or None while running)
OnAttempt = Callable[[str, int, str | None, AttemptResult | None], Awaitable[None]]
# The finished result of an attempt, if one was already recorded (replay after a restart).
RecordedResult = Callable[[str, int], AttemptResult | None]


def _collect(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Parallel run_job tasks append their outcome; ``None`` clears the list."""
    if right is None:
        return []
    return [*(left or []), *right]


def _append(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [*(left or []), *(right or [])]


class JobGraphState(TypedDict, total=False):
    dag: dict[str, Any]
    outcomes: Annotated[list[dict[str, Any]], _collect]
    resolution: dict[str, Any] | None
    # Every human decision taken at a gate, oldest first.
    approvals: Annotated[list[dict[str, Any]], _append]


def gate_payload(state: DynamicRunState, config: DynamicConfig) -> dict[str, Any]:
    decisions = state.open_decisions()
    return {
        "automation": config.automation,
        "decisions": [d.model_dump(mode="json") for d in decisions],
        # A one-line text for the generic approval panel and Slack.
        "plan": "\n".join(f"[{d.kind}] {d.summary}" for d in decisions),
    }


def build_job_graph(
    *,
    config: DynamicConfig,
    dispatcher: Dispatcher,
    roster: str,
    request: str,
    run_id: str,
    job_runner: JobRunner,
    persist: Persist | None = None,
    on_attempt: OnAttempt | None = None,
    recorded: RecordedResult | None = None,
    stop_check: Callable[[], Awaitable[bool]] | None = None,
    workspace: dict[str, Any] | None = None,
    checkpointer: Any = None,
):
    """Compile the job graph for one dynamic step.

    Without a checkpointer the graph is meant to run as a subgraph: called from
    a node of the workflow graph it checkpoints under that node, and a gate's
    interrupt pauses the whole workflow.
    """

    async def schedule(state: JobGraphState) -> Command:
        dag_state = DynamicRunState.model_validate(state["dag"])
        dag = JobDag(
            config=config,
            state=dag_state,
            dispatcher=dispatcher,
            roster=roster,
            request=request,
            run_id=run_id,
            workspace=workspace,
        )
        outcomes = [AttemptResult.model_validate(o) for o in state.get("outcomes") or []]
        stopped = bool(stop_check is not None and await stop_check())
        wave = await dag.schedule(outcomes=outcomes, resolution=state.get("resolution"), stopped=stopped)
        if persist is not None:
            await persist(dag_state)
        update: dict[str, Any] = {"dag": dag_state.model_dump(mode="json"), "outcomes": None, "resolution": None}
        if wave.status == "dispatch":
            return Command(
                update=update,
                goto=[Send("run_job", launch.model_dump(mode="json")) for launch in wave.launches],
            )
        if wave.status == "gate":
            return Command(update=update, goto="gate")
        return Command(update=update, goto=END)

    async def run_job(launch: dict[str, Any]) -> dict[str, Any]:
        job = Job.model_validate(launch["job"])
        attempt = JobAttempt.model_validate(launch["attempt"])
        done = recorded(job.id, attempt.n) if recorded is not None else None
        if done is not None:
            logger.info("dynamic job %s attempt %d already finished; not running it again", job.id, attempt.n)
            return {"outcomes": [done.model_dump(mode="json")]}

        child: dict[str, str] = {}

        async def started(child_run_id: str) -> None:
            child["id"] = child_run_id
            if on_attempt is not None:
                await on_attempt(job.id, attempt.n, child_run_id, None)

        try:
            outcome = await job_runner(job, attempt, launch["payload"], started)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the runner should not raise, but a crash is just a failed attempt
            logger.exception("dynamic job %s crashed", job.id)
            outcome = JobOutcome(status="failed", error=f"{type(exc).__name__}: {exc}")
        result = AttemptResult(**outcome.model_dump(), job_id=job.id, attempt=attempt.n, child_run_id=child.get("id"))
        if on_attempt is not None:
            await on_attempt(job.id, attempt.n, child.get("id"), result)
        return {"outcomes": [result.model_dump(mode="json")]}

    def gate(state: JobGraphState) -> dict[str, Any]:
        dag_state = DynamicRunState.model_validate(state["dag"])
        logger.info("dynamic step '%s' waiting for a decision", dag_state.step_id)
        decision = interrupt({"type": "dynamic_approval", "step_id": dag_state.step_id, **gate_payload(dag_state, config)})
        if not isinstance(decision, dict):
            decision = {"approved": bool(decision)}
        record = {
            "step_id": dag_state.step_id,
            "approved": decision.get("approved", False),
            "reason": decision.get("reason"),
            "corrections": decision.get("corrections") or None,
            "approver_name": decision.get("approver_name"),
            "approver_id": decision.get("approver_id"),
            "approver_source": decision.get("approver_source"),
            "decided_at": decision.get("decided_at"),
        }
        return {"resolution": decision, "approvals": [record]}

    sg = StateGraph(JobGraphState)
    sg.add_node("schedule", schedule, destinations=("run_job", "gate", END))
    sg.add_node("run_job", run_job)
    sg.add_node("gate", gate)
    sg.add_edge(START, "schedule")
    sg.add_edge("run_job", "schedule")
    sg.add_edge("gate", "schedule")
    return sg.compile(checkpointer=checkpointer)


def initial_input(dag: DynamicRunState) -> JobGraphState:
    return {"dag": dag.model_dump(mode="json"), "outcomes": [], "resolution": None, "approvals": []}
