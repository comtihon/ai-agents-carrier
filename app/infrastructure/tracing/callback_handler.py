"""Local LangChain callback handler for in-process run tracing.

Captures LLM calls, tool calls, and state transitions during a LangGraph
run without requiring LangSmith to be configured.  The accumulated data is
persisted to ``GraphRun.trace_data`` after each streaming step.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

_MAX_SUMMARY = 300  # characters kept for prompt/response summaries


def _summarise(text: str) -> str:
    text = str(text).strip()
    if len(text) <= _MAX_SUMMARY:
        return text
    return text[:_MAX_SUMMARY] + "…"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunTraceAccumulator(BaseCallbackHandler):
    """Accumulates trace events during a LangGraph run."""

    def __init__(self) -> None:
        super().__init__()
        self._llm_calls: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._state_transitions: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._token_usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        # Index into _llm_calls keyed by run_id UUID so we can update on_llm_end
        self._llm_index: dict[str, int] = {}
        # Index into _tool_calls keyed by run_id UUID
        self._tool_index: dict[str, int] = {}

    # ── LLM callbacks ─────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("name", "unknown")
        )
        prompt_summary = _summarise(prompts[0]) if prompts else ""
        entry: dict[str, Any] = {
            "model": model,
            "prompt_summary": prompt_summary,
            "timestamp": _now_iso(),
        }
        idx = len(self._llm_calls)
        self._llm_calls.append(entry)
        self._llm_index[str(run_id)] = idx

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        model = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("name", "unknown")
        )
        # Flatten messages to a prompt summary
        flat = " | ".join(
            str(getattr(m, "content", m))
            for batch in messages
            for m in batch
        )
        prompt_summary = _summarise(flat)
        entry: dict[str, Any] = {
            "model": model,
            "prompt_summary": prompt_summary,
            "timestamp": _now_iso(),
        }
        idx = len(self._llm_calls)
        self._llm_calls.append(entry)
        self._llm_index[str(run_id)] = idx

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        idx = self._llm_index.get(str(run_id))
        if idx is None:
            return
        # Extract generation text
        gen_text = ""
        if response.generations:
            first = response.generations[0]
            if first:
                g = first[0]
                gen_text = getattr(g, "text", "") or str(getattr(g, "message", ""))
        self._llm_calls[idx]["response_summary"] = _summarise(gen_text)

        # Token usage
        usage = response.llm_output or {}
        token_usage = usage.get("token_usage") or usage.get("usage") or {}
        if token_usage:
            inp = token_usage.get("prompt_tokens") or token_usage.get("input_tokens", 0)
            out = token_usage.get("completion_tokens") or token_usage.get("output_tokens", 0)
            tot = token_usage.get("total_tokens") or (inp + out)
            self._llm_calls[idx]["input_tokens"] = inp
            self._llm_calls[idx]["output_tokens"] = out
            self._llm_calls[idx]["total_tokens"] = tot
            self._token_usage["input_tokens"] += inp
            self._token_usage["output_tokens"] += out
            self._token_usage["total_tokens"] += tot

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._errors.append(f"LLM error: {error}")

    # ── Tool callbacks ─────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        entry: dict[str, Any] = {
            "tool_name": tool_name,
            "input_summary": _summarise(input_str),
            "timestamp": _now_iso(),
        }
        idx = len(self._tool_calls)
        self._tool_calls.append(entry)
        self._tool_index[str(run_id)] = idx

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        idx = self._tool_index.get(str(run_id))
        if idx is None:
            return
        self._tool_calls[idx]["output_summary"] = _summarise(str(output))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._errors.append(f"Tool error: {error}")

    # ── Chain callbacks (state transitions) ────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or serialized.get("name", "")
        # Skip internal LangGraph bookkeeping nodes
        if not name or name.startswith("__") or name in ("LangGraph", "CompiledStateGraph"):
            return
        self._state_transitions.append({
            "node": name,
            "timestamp": _now_iso(),
        })

    # ── Result ─────────────────────────────────────────────────────────────────

    def to_trace_data(self, latency_ms: float | None = None) -> dict[str, Any]:
        return {
            "llm_calls": list(self._llm_calls),
            "tool_calls": list(self._tool_calls),
            "state_transitions": list(self._state_transitions),
            "token_usage": dict(self._token_usage),
            "latency_ms": latency_ms,
            "errors": list(self._errors),
        }
