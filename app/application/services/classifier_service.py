from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

_DEFAULT_PROMPT = (
    "You are a context classifier. Given a user request and a list of available "
    "data-fetching tools, decide which tools are needed to gather context for "
    "fulfilling the request.\n\n"
    "User request: {user_request}\n\n"
    "Available fetchers:\n{tools_description}\n\n"
    "Return only the IDs of the fetchers that are relevant. "
    "If none are needed, return an empty list."
)


class _ClassificationResult(BaseModel):
    selected_step_ids: list[str]
    reasoning: str


class ClassifierService:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def classify(
        self,
        user_request: str,
        fetch_steps: list[dict[str, Any]],
        prompt_override: str | None = None,
    ) -> list[str]:
        if not fetch_steps:
            return []

        tools_description = "\n".join(
            f"- id: {s['id']}, tool: {s['tool']}, description: {s['description']}"
            for s in fetch_steps
        )

        template = prompt_override or _DEFAULT_PROMPT
        prompt = template.format(user_request=user_request, tools_description=tools_description)

        structured_llm = self._llm.with_structured_output(_ClassificationResult)
        result: _ClassificationResult = await structured_llm.ainvoke(prompt)

        valid_ids = {s["id"] for s in fetch_steps}
        return [sid for sid in result.selected_step_ids if sid in valid_ids]
