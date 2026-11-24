"""Run a pool agent's step on the run's member of its runtime pool.

The pool agent (``default_runtime: "pool"``) keeps its own prompt, tools and
addons; the member supplies only the machine and its ACP agent. Each step asks
app.application.runtime_pools for the run's member and runs as a runner step
on it (app.steps.acp_executor). When the member goes offline mid-step and does
not come back within the pool's grace, the run moves to the next member and
the step starts again there: in a fresh session, with the workspace the s3
addon kept if the agent has one, else an empty one.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any

from app.application.runtime_pools import DEFAULT_GRACE_S, DEFAULT_MEMBER_WAIT_S, assign_member

logger = logging.getLogger(__name__)


async def root_run_id(run_repository: Any, run_id: str, max_depth: int = 10) -> str:
    """The run a pool assignment belongs to: the outermost run *run_id* is part of.

    A dynamic workflow runs each job as a child run, and a ``workflow`` step
    runs its sub-workflow as one too; all of them are the same workflow run for
    the pool, so they share its member.
    """
    current = run_id
    if run_repository is None:
        return current
    for _ in range(max_depth):
        try:
            run = await run_repository.get(current)
        except Exception:
            logger.debug("could not load run %s", current, exc_info=True)
            return current
        parent = getattr(run, "parent_run_id", None) if run is not None else None
        if not parent or parent == current:
            return current
        current = parent
    return current


def member_agent_def(agent_def: Any, member: Any, grace_s: float) -> Any:
    """*agent_def* as a runner agent on *member*'s machine."""
    agent_input = dict(agent_def.agent_input or {})
    # A pool step waits for its machine only as long as the grace; then the
    # run moves on instead of waiting the runner default (10 minutes).
    agent_input.setdefault("runner_wait_s", grace_s)
    return agent_def.model_copy(update={
        "default_runtime": "runner",
        "runner_id": member.runner.id,
        "acp_agent": member.acp_agent,
        "agent_input": agent_input,
    })


async def run_pool_agent(*, agent_def: Any, run_id: str, task_key: str, step: dict[str, Any], settings: Any,
                         run_repository: Any = None, agent_task_repository: Any = None, runner_hub: Any = None,
                         agent_backend: Any = None, **kwargs: Any) -> dict[str, Any]:
    from app.application.runners import get_runner_hub
    from app.steps.acp_executor import _ProgressSink, run_acp_agent

    hub = runner_hub or get_runner_hub()
    if hub.pool_backend is None:
        raise RuntimeError(f"[step '{step['id']}'] runtime pools are not configured")
    if not agent_def.pool_id:
        raise RuntimeError(f"[step '{step['id']}'] agent '{agent_def.id}' has runtime 'pool' but no pool_id")
    agent_input = agent_def.agent_input or {}
    grace_s = float(agent_input.get("pool_grace_s") or getattr(settings, "acp_pool_grace_s", DEFAULT_GRACE_S))
    # How long to wait while no member is online; absent/0 = until one is.
    raw_wait = agent_input.get("pool_wait_s")
    if raw_wait is None:
        raw_wait = getattr(settings, "acp_pool_wait_s", DEFAULT_MEMBER_WAIT_S)
    wait_s = float(raw_wait) if raw_wait else None
    sink = _ProgressSink(run_repository, run_id, step["id"])
    wait_key = f"_pool_wait_{step['id']}"
    waiting = False

    async def on_wait(pool: Any) -> None:
        nonlocal waiting
        waiting = True
        sink.add(f"waiting for a member of runtime pool '{pool.name or pool.id}' to come online")
        await sink.flush()
        await sink.set_state(wait_key, {
            "pool_id": pool.id, "pool_name": pool.name or pool.id,
            "since": datetime.now(timezone.utc).isoformat(), "wait_s": wait_s,
        })

    async def end_wait() -> None:
        nonlocal waiting
        if waiting:
            waiting = False
            await sink.set_state(wait_key, None)

    # Jobs of a dynamic workflow and sub-workflow runs keep the member of the
    # workflow run they belong to.
    sticky_run = await root_run_id(run_repository, run_id)
    failed: str | None = None
    tried: set[str] = set()
    while True:
        try:
            member, _ = await assign_member(
                agent_def.pool_id, sticky_run,
                pool_backend=hub.pool_backend, agent_backend=agent_backend, hub=hub,
                failed=failed, exclude=tried, grace_s=grace_s, wait_s=wait_s, on_wait=on_wait,
            )
        finally:
            await end_wait()
        sink.add(f"runtime pool: running on '{member.agent.name or member.agent_id}' "
                 f"(machine '{member.runner.name}')")
        await sink.flush()
        try:
            return await run_acp_agent(
                agent_def=member_agent_def(agent_def, member, grace_s),
                run_id=run_id, task_key=task_key, step=step, settings=settings,
                run_repository=run_repository, agent_task_repository=agent_task_repository,
                runner_hub=hub, agent_backend=agent_backend, **kwargs,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if hub.is_online(member.runner.id):
                raise
            # The machine is gone: whatever failed, it failed because of that.
            logger.info("[step '%s'] pool member %s went offline (%s); moving the run on",
                        step["id"], member.agent_id, exc)
            sink.add(f"machine '{member.runner.name}' went offline; moving to the next member of the pool "
                     "(or waiting for one to come online)")
            await sink.flush()
            failed = member.agent_id
            tried.add(member.agent_id)
            if agent_task_repository is not None:
                # The stored ACP session belongs to the lost machine.
                try:
                    await agent_task_repository.update_task(task_key, {"status": "working", "acp": None})
                except Exception:
                    logger.debug("could not reset task %s", task_key, exc_info=True)
