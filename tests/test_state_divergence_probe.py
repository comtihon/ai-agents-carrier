"""
Phase 1 of collapsing the dual run-state: a read-only probe that reports where
the runner's hand-merged ``current_state`` disagrees with the LangGraph
checkpoint's reducer-applied values.

The probe exists to measure a known class of bug rather than guess at it: the
stream loop rebuilds state with ``dict.update()``, which reimplements the
reducers. Fields with a non-last-wins reducer (``_sum_usage``, ``_merge_dicts``)
drift the moment a node is re-executed.

These tests pin the three properties the probe must have to be safe to switch
on in production: it reports real divergence, it never raises, and it never
writes.
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.orchestration.yaml_graph import (
    _probe_repr,
    _probe_state_divergence,
)


def _runner_with_checkpoint(values: Any) -> MagicMock:
    snapshot = MagicMock()
    snapshot.values = values
    runner = MagicMock()
    runner.graph.aget_state = AsyncMock(return_value=snapshot)
    return runner


_CONFIG = {"configurable": {"thread_id": "probe-run"}}


@pytest.mark.asyncio
async def test_probe_reports_diverged_value(caplog) -> None:
    """The exact bug this exists for: a summed field the local merge clobbered."""
    runner = _runner_with_checkpoint({"_agent_token_usage_x": {"input": 300}})
    local = {"_agent_token_usage_x": {"input": 100}}

    with caplog.at_level(logging.WARNING):
        await _probe_state_divergence(runner, "run-1", "x", local, _CONFIG)

    assert "DIVERGED" in caplog.text
    assert "_agent_token_usage_x" in caplog.text
    # Both sides are reported so the log alone identifies which is wrong.
    assert "300" in caplog.text and "100" in caplog.text


@pytest.mark.asyncio
async def test_probe_reports_key_only_in_checkpoint(caplog) -> None:
    runner = _runner_with_checkpoint({"seen": 1, "unseen": 2})
    with caplog.at_level(logging.WARNING):
        await _probe_state_divergence(runner, "run-2", "n", {"seen": 1}, _CONFIG)

    assert "MISSING-LOCALLY" in caplog.text
    assert "unseen" in caplog.text


@pytest.mark.asyncio
async def test_probe_reports_key_only_in_local_state(caplog) -> None:
    runner = _runner_with_checkpoint({"seen": 1})
    with caplog.at_level(logging.WARNING):
        await _probe_state_divergence(
            runner, "run-3", "n", {"seen": 1, "dropped": 2}, _CONFIG,
        )

    assert "MISSING-IN-CHECKPOINT" in caplog.text
    assert "dropped" in caplog.text


@pytest.mark.asyncio
async def test_probe_silent_when_state_agrees(caplog) -> None:
    state = {"a": 1, "b": {"c": 2}}
    runner = _runner_with_checkpoint(dict(state))
    with caplog.at_level(logging.WARNING):
        await _probe_state_divergence(runner, "run-4", "n", dict(state), _CONFIG)

    assert caplog.text == ""


@pytest.mark.asyncio
async def test_probe_ignores_the_failure_sentinel(caplog) -> None:
    """__failed_step__ is deliberately patched out-of-band on resume, so its
    divergence is expected and would be pure noise."""
    runner = _runner_with_checkpoint({"__failed_step__": "step-a"})
    with caplog.at_level(logging.WARNING):
        await _probe_state_divergence(
            runner, "run-5", "n", {"__failed_step__": None}, _CONFIG,
        )

    assert caplog.text == ""


@pytest.mark.asyncio
async def test_probe_survives_aget_state_failure() -> None:
    """A diagnostic that can break a run is worse than no diagnostic."""
    runner = MagicMock()
    runner.graph.aget_state = AsyncMock(side_effect=RuntimeError("checkpoint down"))

    await _probe_state_divergence(runner, "run-6", "n", {"a": 1}, _CONFIG)


@pytest.mark.asyncio
async def test_probe_survives_empty_snapshot() -> None:
    runner = _runner_with_checkpoint(None)
    await _probe_state_divergence(runner, "run-7", "n", {"a": 1}, _CONFIG)


@pytest.mark.asyncio
async def test_probe_survives_values_that_refuse_comparison(caplog) -> None:
    """A value whose __eq__ raises must not be mistaken for divergence."""

    class _Hostile:
        def __eq__(self, other):  # noqa: ANN001
            raise TypeError("no comparison for you")

    runner = _runner_with_checkpoint({"k": _Hostile()})
    with caplog.at_level(logging.WARNING):
        await _probe_state_divergence(runner, "run-8", "n", {"k": _Hostile()}, _CONFIG)

    assert caplog.text == ""


@pytest.mark.asyncio
async def test_probe_never_writes() -> None:
    """Read-only: the probe may only call aget_state on the graph."""
    runner = _runner_with_checkpoint({"a": 2})
    await _probe_state_divergence(runner, "run-9", "n", {"a": 1}, _CONFIG)

    runner.graph.aupdate_state.assert_not_called()
    runner.graph.astream.assert_not_called()
    runner.graph.ainvoke.assert_not_called()


def test_probe_repr_truncates_large_values() -> None:
    out = _probe_repr("x" * 5000)
    assert len(out) <= 200
    assert out.endswith("...")


def test_probe_repr_survives_unreprable_values() -> None:
    class _Bad:
        def __repr__(self):
            raise ValueError("nope")

    assert "unreprable" in _probe_repr(_Bad())


# ---------------------------------------------------------------------------
# Wiring: the probe must run once per completed node, and only when enabled.
# ---------------------------------------------------------------------------

def _two_step_runner():
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
    from app.infrastructure.tools.mcp_client import McpToolsProvider

    llm = FakeMessagesListChatModel(
        responses=[AIMessage(content="one"), AIMessage(content="two")]
    )
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    return YamlGraphRunner(
        {
            "id": "probe-graph",
            "steps": [
                {"id": "a", "type": "llm", "output_key": "a_out"},
                {"id": "b", "type": "llm", "output_key": "b_out"},
            ],
        },
        llm=llm,
        mcp_tools_provider=mcp,
    )


def _fresh_run(run_id: str):
    from datetime import datetime, timezone

    from app.domain.models.graph_run import GraphRun

    return GraphRun(
        id=run_id,
        graph_id="probe-graph",
        user_request="hello",
        status="running",
        state={},
        step_statuses={"a": "pending", "b": "pending"},
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )


async def _run_stream_with_probe(monkeypatch, enabled: bool, run_id: str) -> list:
    import app.core.config as config_module
    from app.infrastructure.orchestration import yaml_graph as yg

    settings = MagicMock()
    settings.state_divergence_probe = enabled
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)

    calls: list = []

    async def _spy(runner, rid, node_name, current_state, config):
        calls.append(node_name)

    monkeypatch.setattr(yg, "_probe_state_divergence", _spy)

    await yg.stream_graph_to_pause(
        _two_step_runner(), _fresh_run(run_id), AsyncMock(), {"request": "hi"},
    )
    return calls


@pytest.mark.asyncio
async def test_probe_runs_for_each_node_when_enabled(monkeypatch) -> None:
    calls = await _run_stream_with_probe(monkeypatch, True, "probe-on")
    assert calls == ["a", "b"]


@pytest.mark.asyncio
async def test_probe_does_not_run_when_disabled(monkeypatch) -> None:
    calls = await _run_stream_with_probe(monkeypatch, False, "probe-off")
    assert calls == []


@pytest.mark.asyncio
async def test_probe_defaults_to_off() -> None:
    """It costs an extra checkpoint read per node, so it must be opt-in."""
    from app.core.config import Settings

    assert Settings().state_divergence_probe is False
