"""APScheduler safety-net: polls stale agent tasks and handles recovery."""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


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
