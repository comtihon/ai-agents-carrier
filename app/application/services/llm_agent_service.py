from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.domain.models.runtime import LlmStepResult, LlmToolCall


class LlmAgentService:
    """
    Runs a tool-calling loop against a LangChain chat model:

      1. Send the user request to the LLM (with tools bound).
      2. If the LLM requests a tool call, execute the tool and append the result.
      3. Call the LLM again with the updated message history.
      4. Repeat until the LLM returns a final text answer (no tool calls).
    """

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]) -> None:
        self._llm = llm
        self._tools: dict[str, BaseTool] = {t.name: t for t in tools}
        # Bind tools lazily — avoids failures when the LLM stub doesn't implement bind_tools.
        self._llm_with_tools: Any = None

    def _get_llm_with_tools(self) -> Any:
        if self._llm_with_tools is None:
            tool_list = list(self._tools.values())
            self._llm_with_tools = self._llm.bind_tools(tool_list) if tool_list else self._llm
        return self._llm_with_tools

    async def run(self, user_request: str) -> LlmStepResult:
        messages: list[Any] = [HumanMessage(content=user_request)]
        tool_calls_made: list[LlmToolCall] = []
        llm_with_tools = self._get_llm_with_tools()

        while True:
            response: AIMessage = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool = self._tools.get(tc["name"])
                if tool is None:
                    raise ValueError(f"LLM requested unknown tool '{tc['name']}'.")

                raw = await tool.ainvoke(tc["args"])
                result: dict[str, Any] = raw if isinstance(raw, dict) else {"result": raw}

                tool_calls_made.append(LlmToolCall(name=tc["name"], args=tc["args"], result=result))
                messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

        final_content = response.content if isinstance(response.content, str) else str(response.content)
        return LlmStepResult(response=final_content, tool_calls_made=tool_calls_made)
