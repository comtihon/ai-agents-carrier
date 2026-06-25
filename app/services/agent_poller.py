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


async def _meta_llm_recovery(task: dict[str, Any], settings: Any) -> bool:
    try:
        from app.core.container import build_llm_native
        from langchain_core.messages import HumanMessage
        outputs_text = "\n".join(
            str(o.get("content", "")) for o in task.get("outputs", [])
        )
        provider = settings.meta_llm_provider or settings.llm_provider
        model = settings.meta_llm_model
        llm = build_llm_native(provider, model, settings, max_tokens=100)
        prompt = (
            f"Task: {task.get('input', '')}\n\n"
            f"Outputs so far:\n{outputs_text}\n\n"
            "Is this task complete? Answer 'complete' or 'incomplete' only."
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content if isinstance(response.content, str) else str(response.content)
        return "complete" in text.lower()
    except Exception as exc:
        logger.warning("Meta LLM recovery failed for task %s: %s", task.get("_id"), exc)
        return False
