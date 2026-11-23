"""``ignore_errors`` on agent steps: a failed agent step stores {error, ok: false}
under its output_key and the run continues to the next step."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.tools.mcp_client import McpToolsProvider
from app.steps.agent_executor import MetaLLMRejectionError


def _runner(agent_step: dict[str, Any]) -> YamlGraphRunner:
    steps = [
        {"agent_id": "a", "type": "claude-agent", "id": "model", **agent_step},
        {
            "id": "after",
            "type": "python",
            "sandbox": False,
            "code": "output = {'seen': state.get('model_out', state.get('model')), 'failed': state.get('__failed_step__')}",
            "output_key": "after_out",
        },
    ]
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner(
        {"id": "wf-ignore", "steps": steps},
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=mcp,
    )
    runner._agent_backend = MagicMock()
    return runner


@contextmanager
def _agent_raises(exc: Exception):
    async def _execute(*args, **kwargs):
        raise exc

    with patch("app.steps.agent_executor.execute_agent_step", new=_execute), \
            patch("app.application.visibility.granted_for_workflow", new=lambda backend, wf: backend):
        yield


async def _invoke(runner: YamlGraphRunner) -> dict[str, Any]:
    return await runner.graph.ainvoke({"request": "go"}, config={"configurable": {"thread_id": "t"}})


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [
    RuntimeError("runner rnr-x did not come online within 600s"),
    RuntimeError("[step 'model'] the agent's account has no quota left: out"),
    ConnectionError("concurrency backend unavailable"),  # the queue wait (ConcurrencySlot.acquire)
])
async def test_ignore_errors_captures_failure_and_continues(exc):
    runner = _runner({"output_key": "model_out", "ignore_errors": True})
    with _agent_raises(exc):
        state = await _invoke(runner)
    assert state["model_out"] == {"error": str(exc), "ok": False}
    assert not state.get("__failed_step__")
    assert state["after_out"] == {"seen": {"error": str(exc), "ok": False}, "failed": None}


@pytest.mark.asyncio
async def test_ignore_errors_defaults_to_step_id_and_keeps_meta_rejection():
    runner = _runner({"output_mapping": {"status": "agent_status"}, "ignore_errors": True})
    exc = MetaLLMRejectionError("rejected", mapped_result={"agent_status": "half"}, reason="bad output")
    with _agent_raises(exc):
        state = await _invoke(runner)
    assert state["model"] == {"error": "rejected", "ok": False, "meta_llm_rejection": "bad output"}
    assert state["agent_status"] == "half"
    assert not state.get("__failed_step__")
    assert state["after_out"]["seen"]["ok"] is False


@pytest.mark.asyncio
async def test_without_ignore_errors_the_run_still_fails():
    runner = _runner({"output_key": "model_out"})
    with _agent_raises(RuntimeError("boom")), pytest.raises(RuntimeError, match="upstream step 'model' already failed"):
        await _invoke(runner)


@pytest.mark.asyncio
async def test_ignore_errors_success_path_unchanged():
    async def _execute(*args, **kwargs):
        return {"model_out": "fine"}

    runner = _runner({"output_key": "model_out", "ignore_errors": True})
    with patch("app.steps.agent_executor.execute_agent_step", new=_execute), \
            patch("app.application.visibility.granted_for_workflow", new=lambda backend, wf: backend):
        state = await _invoke(runner)
    assert state["model_out"] == "fine"
    assert state["after_out"]["seen"] == "fine"
