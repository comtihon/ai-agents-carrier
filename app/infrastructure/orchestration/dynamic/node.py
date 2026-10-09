"""LangGraph wiring for ``type: dynamic`` steps.

A dynamic step becomes two graph nodes:

    <id>        runs the engine until it completes, fails, or needs a human
    <id>__gate  interrupts for that human, then loops back to <id>

Pausing through a separate gate node (rather than calling ``interrupt()``
inside the long-running node) keeps LangGraph's resume semantics simple: the
gate is re-executed on resume and returns the decision; the dynamic node then
runs a fresh pass that reloads the persisted DAG and applies it. The gate is an
ordinary step in the runner's step list, so the run status (waiting_approval),
the approvals panel and approve/reject all work unchanged.

Each job attempt runs as a child ``GraphRun`` (kind="job") through the same
``execute_agent_step`` every static agent step uses — runtimes, warm pods,
addons, meta-LLM quality gate, progress and clarify callbacks included. The
child run id is what gives parallel instances of one agent distinct pods.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from app.domain.models.dynamic import DynamicConfig, DynamicRunState, Job, JobAttempt
from app.infrastructure.orchestration.dynamic.dispatcher import Dispatcher, build_llm_call, build_roster
from app.infrastructure.orchestration.dynamic.engine import OUTPUT_FIELDS, DynamicEngine, JobOutcome

if TYPE_CHECKING:  # pragma: no cover
    from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner

logger = logging.getLogger(__name__)

GATE_SUFFIX = "__gate"


def gate_key(step_id: str) -> str:
    return f"_dynamic_gate_{step_id}"


def resolution_key(step_id: str) -> str:
    return f"_dynamic_resolution_{step_id}"


def output_key(step: dict[str, Any]) -> str:
    return step.get("output_key") or step["id"]


def expand_dynamic_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Insert each dynamic step's gate right after it (idempotent)."""
    if not isinstance(steps, list):
        return steps
    ids = {s.get("id") for s in steps if isinstance(s, dict)}
    out: list[dict[str, Any]] = []
    for step in steps:
        out.append(step)
        if isinstance(step, dict) and step.get("type") == "dynamic":
            gid = f"{step['id']}{GATE_SUFFIX}"
            if gid not in ids:
                out.append({"id": gid, "type": "dynamic_gate", "dynamic_step": step["id"], "next": step["id"]})
    return out


def state_fields(step: dict[str, Any]) -> dict[str, Any]:
    """State keys a dynamic step writes; LangGraph drops undeclared keys."""
    if step.get("type") != "dynamic":
        return {}
    sid = step["id"]
    return {gate_key(sid): Any, resolution_key(sid): Any, output_key(step): Any}


def router(step: dict[str, Any]):
    """After the dynamic node: to the gate when it paused, else onward."""
    sid = step["id"]
    gate = f"{sid}{GATE_SUFFIX}"
    nxt = step.get("next") or "END"

    def route(state: dict) -> str:
        return gate if state.get(gate_key(sid)) else nxt

    return route, gate, nxt


def make_gate_node(runner: "YamlGraphRunner", step: dict[str, Any]):
    dynamic_id = step["dynamic_step"]

    def node(state: dict) -> dict:
        from langgraph.types import interrupt

        pending = state.get(gate_key(dynamic_id)) or {}
        logger.info("[%s] dynamic step '%s' waiting for a decision", runner.id, dynamic_id)
        decision = interrupt({"type": "dynamic_approval", "step_id": dynamic_id, **pending})
        if not isinstance(decision, dict):
            decision = {"approved": bool(decision)}
        record = {
            "step_id": step["id"],
            "approved": decision.get("approved", False),
            "reason": decision.get("reason"),
            "corrections": decision.get("corrections") or None,
            "approver_name": decision.get("approver_name"),
            "approver_id": decision.get("approver_id"),
            "approver_source": decision.get("approver_source"),
            "decided_at": decision.get("decided_at"),
        }
        history = list(state.get("approval_history") or [])
        history.append(record)
        return {gate_key(dynamic_id): None, resolution_key(dynamic_id): decision, "approval_history": history}

    return node


def _gate_payload(state: DynamicRunState, config: DynamicConfig) -> dict[str, Any]:
    decisions = state.open_decisions()
    return {
        "automation": config.automation,
        "decisions": [d.model_dump(mode="json") for d in decisions],
        # A one-line text for the generic approval panel and Slack.
        "plan": "\n".join(f"[{d.kind}] {d.summary}" for d in decisions),
    }


class _RunStore:
    """Persists the DAG onto the parent run without clobbering concurrent writes."""

    def __init__(self, run: Any, repo: Any, step_id: str) -> None:
        self.run = run
        self.repo = repo
        self.step_id = step_id
        self._lock = asyncio.Lock()

    def load(self) -> DynamicRunState:
        raw = (getattr(self.run, "dynamic", None) or {}).get(self.step_id) if self.run is not None else None
        return DynamicRunState.model_validate(raw) if raw else DynamicRunState(step_id=self.step_id)

    async def save(self, state: DynamicRunState) -> None:
        if self.run is None or self.repo is None:
            return
        data = state.model_dump(mode="json")
        async with self._lock:
            # The run object is shared with the stream loop: keep it current so
            # its own full-document writes carry the DAG too.
            self.run.dynamic = {**(self.run.dynamic or {}), self.step_id: data}
            try:
                set_dynamic = getattr(self.repo, "set_dynamic", None)
                if set_dynamic is not None:
                    await set_dynamic(self.run.id, self.step_id, data)
                else:
                    self.run.touch()
                    await self.repo.update(self.run)
            except Exception:
                logger.exception("dynamic step '%s': failed to persist DAG", self.step_id)


def _questions_from_interrupt(exc: BaseException) -> list[str]:
    questions: list[str] = []
    for item in exc.args[0] if exc.args and isinstance(exc.args[0], (list, tuple)) else []:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            questions += [str(q) for q in value.get("questions") or []]
    return questions or ["The agent needs more information to continue."]


def make_job_runner(runner: "YamlGraphRunner", step: dict[str, Any], config: DynamicConfig, parent_run_id: str):
    """Execute one job attempt as a child run through execute_agent_step."""

    async def run_job(job: Job, attempt: JobAttempt, payload: dict[str, Any], started) -> JobOutcome:
        from langgraph.errors import GraphInterrupt

        from app.core.config import get_settings
        from app.domain.models.graph_run import GraphRun
        from app.services.agent_cleanup import cleanup_run_agents
        from app.steps.agent_executor import MetaLLMRejectionError, execute_agent_step

        settings = get_settings()
        repo = runner._current_run_repository
        child_id = str(uuid.uuid4())
        child = GraphRun(
            id=child_id,
            graph_id=runner.id,
            kind="job",
            parent_run_id=parent_run_id,
            user_request=f"[{job.id} #{attempt.n}] {job.title or job.prompt[:200]}",
            status="running",
            current_step=job.id,
            step_statuses={job.id: "running"},
            step_inputs={job.id: payload},
            state={"request": payload.get("task", "")},
        )
        if repo is not None:
            await repo.create(child)
        await started(child_id)

        fields = OUTPUT_FIELDS[job.category]
        agent_step: dict[str, Any] = {
            "id": job.id,
            "type": "langgraph-agent",
            "agent_id": job.agent_id,
            "input_mapping": {k: k for k in payload if k != "clarification_context"},
            "output_mapping": {f: f for f in fields},
        }
        if config.shared_volume is not None:
            # Every job of the run mounts the same claim.
            agent_step["pvc_mount_point"] = config.shared_volume.mount_point
            agent_step["pvc_name"] = f"pvc-{parent_run_id[:12]}"
            agent_step["pvc_ttl"] = config.shared_volume.ttl
        job_state = dict(payload)
        if payload.get("clarification_context"):
            job_state["_clarification_answers"] = payload["clarification_context"]

        outcome: JobOutcome
        result: dict[str, Any] = {}
        try:
            result = await execute_agent_step(
                agent_step,
                job_state,
                runner._agent_backend,
                child_id,
                runner._callback_base_url or "",
                settings=settings,
                run_repository=repo,
                pvc_lease_repository=runner._pvc_lease_repository,
                agent_task_repository=runner._agent_task_repository,
                warm_pod_repository=runner._warm_pod_repository,
                use_meta_llm=runner._use_meta_llm,
            )
            output = {k: result[k] for k in fields if k in result}
            if "_meta_llm_result" in result:
                output["_meta_llm_result"] = result["_meta_llm_result"]
            outcome = JobOutcome(status="finished", output=output)
        except GraphInterrupt as exc:
            outcome = JobOutcome(status="needs_input", questions=_questions_from_interrupt(exc))
        except MetaLLMRejectionError as exc:
            outcome = JobOutcome(status="failed", error=f"quality gate rejected the output: {exc.reason}", output=exc.mapped_result or None)
        except asyncio.CancelledError:
            outcome = JobOutcome(status="failed", error="cancelled")
            await asyncio.shield(_close_child(repo, child, outcome, result, settings, cleanup_run_agents, runner))
            raise
        except Exception as exc:
            logger.warning("dynamic job %s attempt %d failed: %s", job.id, attempt.n, exc)
            outcome = JobOutcome(status="failed", error=str(exc))
        await asyncio.shield(_close_child(repo, child, outcome, result, settings, cleanup_run_agents, runner))
        return outcome

    return run_job


async def _close_child(repo, child, outcome: JobOutcome, result: dict, settings, cleanup_run_agents, runner) -> None:
    try:
        # Agents are per attempt: nothing reuses a job's pod.
        await cleanup_run_agents(child.id, settings, warm_pod_repository=runner._warm_pod_repository)
    except Exception:
        logger.debug("cleanup for job run %s failed", child.id, exc_info=True)
    if repo is None:
        return
    try:
        fresh = await repo.get(child.id) or child
        status = {"finished": "completed", "failed": "failed", "needs_input": "waiting_approval"}[outcome.status]
        if outcome.error == "cancelled":
            status = "cancelled"
        fresh.status = status  # type: ignore[assignment]
        fresh.step_statuses = {**fresh.step_statuses, child.current_step: {"completed": "finished", "waiting_approval": "waiting_clarification"}.get(status, status)}
        fresh.step_outputs = {**fresh.step_outputs, child.current_step: result or outcome.output or {}}
        if outcome.error:
            fresh.state = {**(fresh.state or {}), "error": outcome.error}
        fresh.touch()
        await repo.update(fresh)
    except Exception:
        logger.exception("failed to close job run %s", child.id)


def make_dynamic_node(runner: "YamlGraphRunner", step: dict[str, Any]):
    step_id = step["id"]
    out_key = output_key(step)

    async def node(state: dict) -> dict:
        config = DynamicConfig.from_step(step)
        run = runner._current_run
        repo = runner._current_run_repository
        run_id = run.id if run is not None else "unknown"
        store = _RunStore(run, repo, step_id)
        dstate = store.load()

        if runner._agent_backend is None:
            return {"__failed_step__": step_id, "error": "agent backend not configured"}
        agents: dict[str, Any] = {}
        for entry in config.agent_pool:
            agent_def = await runner._agent_backend.get(entry.agent_id)
            if agent_def is None:
                return {"__failed_step__": step_id, "error": f"dynamic step '{step_id}': agent '{entry.agent_id}' in the pool does not exist"}
            agents[entry.agent_id] = agent_def

        template = step.get("request_template")
        request = runner._render(template, state) if template else str(state.get("request") or "")

        from app.core.config import get_settings

        async def stopped() -> bool:
            if repo is None or run is None:
                return False
            try:
                fresh = await repo.get(run.id)
            except Exception:
                return False
            return fresh is not None and fresh.status in ("failed", "cancelled")

        engine = DynamicEngine(
            config=config,
            state=dstate,
            dispatcher=Dispatcher(build_llm_call(config, get_settings())),
            roster=build_roster(config, agents),
            request=request,
            job_runner=make_job_runner(runner, step, config, run_id),
            persist=store.save,
            run_id=run_id,
            stop_check=stopped,
        )
        outcome = await engine.advance(state.get(resolution_key(step_id)))
        update: dict[str, Any] = {resolution_key(step_id): None}
        if outcome == "gate":
            update[gate_key(step_id)] = _gate_payload(engine.state, config)
            return update
        update[gate_key(step_id)] = None
        update[out_key] = engine.result()
        if outcome == "failed":
            update["__failed_step__"] = step_id
            update["error"] = engine.state.error
        return update

    return node


def validate_dynamic_steps(steps: Any) -> list[str]:
    """Problems with every ``type: dynamic`` step's config (empty = fine)."""
    from pydantic import ValidationError

    errors: list[str] = []
    if not isinstance(steps, list):
        return errors
    for step in steps:
        if not isinstance(step, dict) or step.get("type") != "dynamic":
            continue
        sid = step.get("id", "?")
        if str(sid).endswith(GATE_SUFFIX):
            errors.append(f"step '{sid}': a dynamic step id may not end with '{GATE_SUFFIX}'")
        try:
            config = DynamicConfig.from_step(step)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err.get("loc", ()))
                errors.append(f"step '{sid}': {loc + ': ' if loc else ''}{err.get('msg')}")
            continue
        if not config.agent_pool:
            errors.append(f"step '{sid}': agent_pool is empty — select the agents this workflow may use")
        can_execute = any(not a.categories or "execution" in a.categories for a in config.agent_pool)
        if config.agent_pool and not can_execute:
            errors.append(f"step '{sid}': no agent in the pool may fill execution jobs")
    return errors


async def child_summaries(run_repository: Any, run_id: str) -> dict[str, Any]:
    """Live view of each job attempt's child run, keyed by child run id."""
    list_children = getattr(run_repository, "list_children", None)
    if list_children is None:
        return {}
    try:
        children = await list_children(run_id)
    except Exception:
        logger.debug("listing job runs of %s failed", run_id, exc_info=True)
        return {}
    out: dict[str, Any] = {}
    for child in children:
        state = child.state or {}
        job_step = child.current_step or ""
        progress = state.get(f"_agent_progress_{job_step}") or []
        out[child.id] = {
            "status": child.status,
            "job_id": job_step,
            "pending_question": state.get("_pending_question"),
            "progress": progress[-8:] if isinstance(progress, list) else [],
            "error": state.get("error"),
            "updated_at": child.updated_at.isoformat() if child.updated_at else None,
        }
    return out
