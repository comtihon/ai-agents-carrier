"""The job DAG as a LangGraph graph, driven by events.

    START ─► schedule ─┬─► wait ──┐     (attempts running: take the next events)
               ▲       ├─► gate ──┤     (a decision waits for a human: interrupt)
               │       └─► END    │     (completed or failed)
               └──────────────────┘

``schedule`` folds events and a gate's resolution into the DAG through
``JobDag`` — which consults the meta-agent whenever something needs a
decision — and carries out the result: it launches attempts on the step's
``JobHub``, stops the ones the meta-agent stopped and delivers answers to
agents waiting on a question. ``wait`` takes the next events from the hub as
they come, one finished attempt or one question at a time, so nothing waits
for the slowest agent. Loops — a validator's change request rewinding the DAG,
an answered question, a retry — are ``schedule`` launching the next attempt of
the same job.

The DAG lives in the graph's state and is checkpointed at every step, so a run
paused at a gate resumes where it stopped. The ``persist`` callback mirrors it
onto the run document for the UI.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.domain.models.dynamic import DynamicConfig, DynamicRunState
from app.infrastructure.orchestration.dynamic.engine import JobDag, LostAttempt, parse_event
from app.infrastructure.orchestration.dynamic.hub import JobHub
from app.infrastructure.orchestration.dynamic.meta_agent import MetaAgent

logger = logging.getLogger(__name__)

Persist = Callable[[DynamicRunState], Awaitable[None]]

# How long ``wait`` sleeps before checking whether the run was stopped.
WAKE_SECONDS = 30.0


def _collect(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Events append; ``None`` clears the list once ``schedule`` took them."""
    if right is None:
        return []
    return [*(left or []), *right]


def _append(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [*(left or []), *(right or [])]


class JobGraphState(TypedDict, total=False):
    dag: dict[str, Any]
    events: Annotated[list[dict[str, Any]], _collect]
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
    meta: MetaAgent,
    roster: str,
    request: str,
    run_id: str,
    hub: JobHub,
    persist: Persist | None = None,
    stop_check: Callable[[], Awaitable[bool]] | None = None,
    workspace: dict[str, Any] | None = None,
    checkpointer: Any = None,
    wake_seconds: float = WAKE_SECONDS,
):
    """Compile the job graph for one dynamic step.

    Without a checkpointer the graph is meant to run as a subgraph: called from
    a node of the workflow graph it checkpoints under that node, and a gate's
    interrupt pauses the whole workflow.
    """

    async def schedule(state: JobGraphState) -> Command:
        dag_state = DynamicRunState.model_validate(state["dag"])
        dag = JobDag(config=config, state=dag_state, meta=meta, roster=roster, request=request, run_id=run_id, workspace=workspace)
        events = [parse_event(e) for e in state.get("events") or []]
        stopped = bool(stop_check is not None and await stop_check())
        step = await dag.schedule(events=events, resolution=state.get("resolution"), stopped=stopped)
        if persist is not None:
            await persist(dag_state)
        for ref in step.cancels:
            hub.cancel(ref.job_id, ref.attempt)
        for delivery in step.deliveries:
            await hub.deliver(delivery)
        for launch in step.launches:
            hub.launch(launch)
        update: dict[str, Any] = {"dag": dag_state.model_dump(mode="json"), "events": None, "resolution": None}
        if step.status == "gate":
            return Command(update=update, goto="gate")
        if step.status == "run":
            return Command(update=update, goto="wait")
        return Command(update=update, goto=END)

    async def wait(state: JobGraphState) -> dict[str, Any]:
        dag_state = DynamicRunState.model_validate(state["dag"])
        running = [(j.id, j.last.n) for j in dag_state.running() if j.last is not None]
        # Attempts without a live task (the process restarted): adopt what they
        # recorded before the restart, or report them lost so they run again.
        orphans: list[dict[str, Any]] = []
        for job_id, n in running:
            if hub.alive(job_id, n):
                continue
            done = hub.recorded_result(job_id, n)
            orphans.append(done.model_dump(mode="json") if done is not None else LostAttempt(job_id=job_id, attempt=n).model_dump(mode="json"))
        if orphans or not running:
            return {"events": orphans}
        while True:
            events = await hub.next_events(timeout=wake_seconds)
            if events:
                return {"events": events}
            if stop_check is not None and await stop_check():
                return {"events": []}

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
    sg.add_node("schedule", schedule, destinations=("wait", "gate", END))
    sg.add_node("wait", wait)
    sg.add_node("gate", gate)
    sg.add_edge(START, "schedule")
    sg.add_edge("wait", "schedule")
    sg.add_edge("gate", "schedule")
    return sg.compile(checkpointer=checkpointer)


def initial_input(dag: DynamicRunState) -> JobGraphState:
    return {"dag": dag.model_dump(mode="json"), "events": [], "resolution": None, "approvals": []}
