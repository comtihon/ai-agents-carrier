from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflows"])

_STEP_TYPE_MAP: dict[str, str] = {
    "llm_structured": "llm",
    "llm": "llm",
    "mcp": "fetch",
    "human_approval": "approval",
    "execute": "execute",
    "workflow": "workflow",
}


# ─── Request models ───────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    workflow_id: str
    user_request: str
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict | None = None


class ApproveRequest(BaseModel):
    feedback: str | None = None


class RejectRequest(BaseModel):
    reason: str | None = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _langgraph_status(snap) -> str:
    return "waiting_approval" if snap.next else "completed"


def _get_runner(container: ApplicationContainer, graph_id: str) -> YamlGraphRunner:
    runner = container.yaml_graph_registry.get(graph_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{graph_id}' not found")
    return runner


def _init_step_statuses(runner: YamlGraphRunner) -> dict[str, str]:
    return {s["id"]: "pending" for s in runner.steps}


def _run_response(run: GraphRun, runner: YamlGraphRunner | None = None) -> dict:
    workflow_name = runner.name if runner else run.graph_id
    step_statuses = run.step_statuses
    step_inputs = run.step_inputs
    step_outputs = run.step_outputs
    steps = [
        {
            "id": s["id"],
            "type": _STEP_TYPE_MAP.get(s.get("type", "llm"), s.get("type", "llm")),
            "name": s.get("name", s["id"]),
            "status": step_statuses.get(s["id"], "pending"),
            "input": step_inputs.get(s["id"]),
            "output": step_outputs.get(s["id"]),
        }
        for s in runner.steps
    ] if runner else []
    return {
        "id": run.id,
        "workflow_id": run.graph_id,
        "workflow_name": workflow_name,
        "user_request": run.user_request,
        "status": run.status,
        "current_step": run.current_step,
        "steps": steps,
        "approval_status": "pending" if run.status == "waiting_approval" else "not_required",
        "approval_gates": [],
        "plan": None,
        "tool_call_results": [],
        "llm_agent_results": [],
        "action_results": [],
        "execution_results": [],
        "intermediate_outputs": run.state,
        "error": run.state.get("error") if run.status == "failed" else None,
        "metadata": {},
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }


def _step_status_for_output(output: dict) -> str:
    """Determine step status from its output: empty dict means skipped."""
    return "skipped" if output == {} else "finished"


async def _stream_graph(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
    input_value,
) -> None:
    """
    Stream graph execution, updating step_statuses and current_step in the DB
    after each node completes.  Works for both initial runs and resume-after-approval.
    """
    step_ids = [s["id"] for s in runner.steps]

    # Seed current_state so we can record each step's input before it runs.
    # For Command (approve/reject resume) fetch the checkpointed state instead.
    if isinstance(input_value, dict):
        current_state: dict = dict(input_value)
    else:
        try:
            snap = runner.graph.get_state(_config(run.id))
            current_state = dict(snap.values) if snap.values else {}
        except Exception:
            current_state = {}

    try:
        async for chunk in runner.graph.astream(
            input_value, _config(run.id), stream_mode="updates",
        ):
            for node_name, output in chunk.items():
                if node_name in ("__start__", "__end__"):
                    continue
                status = _step_status_for_output(output)
                run.step_inputs[node_name] = dict(current_state)
                run.step_statuses[node_name] = status
                run.current_step = node_name
                if output:
                    run.step_outputs[node_name] = output
                    if isinstance(output, dict):
                        current_state.update(output)
                logger.info(
                    "run %s: step '%s' → %s", run.id, node_name, status,
                )
                run.touch()
                await container.run_repository.update(run)
    except Exception as exc:
        logger.exception("run %s: graph execution failed", run.id)
        for sid in step_ids:
            if run.step_statuses.get(sid) == "pending":
                run.step_inputs[sid] = dict(current_state)
                run.step_statuses[sid] = "failed"
                break
        run.status = "failed"
        run.state = {"error": str(exc)}
        run.current_step = None
        run.touch()
        await container.run_repository.update(run)
        return

    snap = runner.graph.get_state(_config(run.id))
    run.status = _langgraph_status(snap)
    run.current_step = snap.next[0] if snap.next else None
    run.state = snap.values
    run.touch()
    await container.run_repository.update(run)


async def _execute_graph(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
) -> None:
    """Run the graph to its first interrupt (or completion) and persist the result."""
    run.step_statuses = _init_step_statuses(runner)
    await _stream_graph(runner, run, container, {"request": run.user_request})


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("")
async def list_workflows(container: ApplicationContainer = Depends(get_container)):
    return container.yaml_graph_registry.list_definitions()


@router.post("/runs")
async def start_run(
    body: RunRequest,
    background_tasks: BackgroundTasks,
    container: ApplicationContainer = Depends(get_container),
):
    runner = _get_runner(container, body.workflow_id)
    thread_id = str(uuid4())

    run = GraphRun(
        id=thread_id,
        graph_id=body.workflow_id,
        user_request=body.user_request,
        status="running",
    )
    await container.run_repository.create(run)
    background_tasks.add_task(_execute_graph, runner, run, container)

    return _run_response(run, runner)


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = container.yaml_graph_registry.get(run.graph_id)
    return _run_response(run, runner)


@router.post("/runs/{run_id}/approve")
async def approve_run(
    run_id: str,
    body: ApproveRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner(container, run.graph_id)

    run.status = "running"
    run.touch()
    await container.run_repository.update(run)

    await _stream_graph(
        runner, run, container,
        Command(resume={"approved": True}),
    )

    return _run_response(run, runner)


@router.post("/runs/{run_id}/reject")
async def reject_run(
    run_id: str,
    body: RejectRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    runner = _get_runner(container, run.graph_id)

    run.status = "running"
    run.touch()
    await container.run_repository.update(run)

    await _stream_graph(
        runner, run, container,
        Command(resume={"approved": False, "reason": body.reason if body else None}),
    )

    # Override: if no more gates remain after rejection, mark as cancelled
    if run.status == "completed":
        run.status = "cancelled"
        run.touch()
        await container.run_repository.update(run)

    return _run_response(run, runner)
