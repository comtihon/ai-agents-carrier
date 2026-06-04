"""APScheduler safety-net: polls stale agent tasks and handles recovery."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from app.core.container import ApplicationContainer

logger = logging.getLogger(__name__)

_STALE_SECONDS = 30


async def poll_agent(agent_url: str) -> dict[str, Any] | None:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{agent_url}/poll", timeout=5.0)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("Failed to poll agent at %s: %s", agent_url, exc)
        return None


async def _meta_llm_recovery(task: dict[str, Any], anthropic_api_key: str) -> bool:
    try:
        import anthropic
        outputs_text = "\n".join(
            str(o.get("content", "")) for o in task.get("outputs", [])
        )
        ac = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        msg = await ac.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Task: {task.get('input', '')}\n\n"
                    f"Outputs so far:\n{outputs_text}\n\n"
                    "Is this task complete? Answer 'complete' or 'incomplete' only."
                ),
            }],
        )
        return "complete" in msg.content[0].text.lower()
    except Exception as exc:
        logger.warning("Meta LLM recovery failed for task %s: %s", task.get("_id"), exc)
        return False


async def handle_poll_result(
    task: dict[str, Any],
    result: dict[str, Any],
    repo: Any,
    settings: Any,
) -> str:
    """Process one poll result. Returns the resolved status or 'continue'."""
    task_id = task["_id"]
    outputs = result.get("outputs", [])
    if outputs:
        await repo.append_outputs(task_id, outputs)

    status = result.get("status", "idle")

    if status == "finished":
        await repo.update_task(task_id, {"status": "finished"})
        return "finished"

    if status == "failed":
        await repo.update_task(task_id, {"status": "failed"})
        return "failed"

    if status == "working":
        await repo.update_task(task_id, {"status": "running"})
        return "continue"

    if status == "idle":
        complete = await _meta_llm_recovery(task, settings.anthropic_api_key)
        loop_count = task.get("loop_count", 0)
        max_loops = task.get("max_loops", settings.agent_max_loops)
        if complete:
            await repo.update_task(task_id, {"status": "finished"})
            return "finished"
        if loop_count >= max_loops:
            logger.warning("Task %s exceeded max loops (%d), marking failed", task_id, max_loops)
            await repo.update_task(task_id, {"status": "failed"})
            return "failed"
        await repo.update_task(task_id, {"loop_count": loop_count + 1})
        return "idle_retry"

    return "continue"


async def sweep_stale_tasks(container: "ApplicationContainer") -> None:
    repo = container.agent_task_repository
    if repo is None:
        return
    settings = container.settings
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=_STALE_SECONDS)
    try:
        tasks = await repo.list_stale_tasks(cutoff)
    except Exception as exc:
        logger.warning("Failed to list stale tasks: %s", exc)
        return

    for task in tasks:
        agent_url = task.get("agent_url")
        if not agent_url:
            continue
        result = await poll_agent(agent_url)
        if result is None:
            # Unreachable — mark failed after max_loops exhausted
            loop_count = task.get("loop_count", 0)
            max_loops = task.get("max_loops", settings.agent_max_loops)
            if loop_count >= max_loops:
                await repo.update_task(task["_id"], {"status": "failed"})
                logger.warning("Task %s unreachable after %d loops, marked failed", task["_id"], max_loops)
            else:
                await repo.update_task(task["_id"], {"loop_count": loop_count + 1})
            continue
        await handle_poll_result(task, result, repo, settings)
