"""``proceed_or`` is the OR counterpart to ``join``: first branch in wins.

What needs proving is not the happy path but the guard around it. Pregel triggers
a node once per superstep that writes to it, so an uneven-depth fan-in enters the
node twice -- and without the router that cuts late arrivals to END, every step
after the fan-in would run once per arrival. The graph-level tests below assert
that count, because it is the property a user actually feels.

The two failure branches are exercised against the node closure directly. Driving
them through a graph would mean finding a step type that happens to write
``__failed_step__`` (most record their error in state instead), which would test
that step type's error handling rather than this node's.
"""
from __future__ import annotations

import pytest

from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner


class _FakeLLM:
    """Stands in for the chat model: no ``proceed_or`` path ever calls it."""


def _runner(steps: list[dict]) -> YamlGraphRunner:
    return YamlGraphRunner({"id": "t", "steps": steps}, _FakeLLM(), lambda *a, **k: [])


def _py(step_id: str, code: str, **extra) -> dict:
    return {
        "id": step_id, "type": "python", "code": code,
        "output_key": f"o_{step_id}", "sandbox": False, **extra,
    }


_GATE_AND_TAIL = [
    {"id": "gate", "type": "proceed_or", "next": "tail"},
    _py("tail", "output='tail'"),
]


async def _run(steps: list[dict], thread: str) -> tuple[dict[str, int], dict]:
    """Execute *steps* and return per-node execution counts plus merged state."""
    runner = _runner(steps)
    counts: dict[str, int] = {}
    state: dict = {}
    async for chunk in runner.graph.astream(
        {"request": "go"}, {"configurable": {"thread_id": thread}}, stream_mode="updates",
    ):
        for node, update in chunk.items():
            counts[node] = counts.get(node, 0) + 1
            if isinstance(update, dict):
                state.update(update)
    return counts, state


@pytest.mark.asyncio
async def test_uneven_depth_fan_in_runs_the_tail_once():
    """The slow branch re-enters the gate; the tail must still run exactly once."""
    steps = [
        {"id": "fan", "type": "parallel", "targets": ["fast", "slow_a"]},
        _py("fast", "output='f'", next="gate"),
        _py("slow_a", "import time;time.sleep(0.2);output='a'", next="slow_b"),
        _py("slow_b", "output='b'", next="gate"),
        *_GATE_AND_TAIL,
    ]
    counts, _ = await _run(steps, "uneven")

    assert counts["gate"] == 2, "gate should be entered once per arrival superstep"
    assert counts["tail"] == 1, "the tail must not run again for the late branch"


@pytest.mark.asyncio
async def test_equal_depth_fan_in_collapses_to_one_arrival():
    """Equal-depth branches share a superstep, so the gate is entered once."""
    steps = [
        {"id": "fan", "type": "parallel", "targets": ["x", "y"]},
        _py("x", "output='x'", next="gate"),
        _py("y", "output='y'", next="gate"),
        *_GATE_AND_TAIL,
    ]
    counts, _ = await _run(steps, "equal")

    assert counts["gate"] == 1
    assert counts["tail"] == 1


@pytest.mark.asyncio
async def test_a_step_recording_its_error_in_state_can_still_win():
    """A ``python`` step that captures its error has not "failed" for the gate.

    It counts as an ordinary arrival, so the tail runs and can inspect the error
    payload. This is the documented limit that ``join`` shares.
    """
    steps = [
        {"id": "fan", "type": "parallel", "targets": ["boom"]},
        _py("boom", "raise RuntimeError('boom')", next="gate"),
        *_GATE_AND_TAIL,
    ]
    counts, state = await _run(steps, "captured-error")

    assert counts["tail"] == 1
    assert state["o_boom"] == {"error": "boom"}
    assert not state.get("__failed_step__")


def _gate_node(step_id: str = "gate"):
    runner = _runner([{"id": step_id, "type": "proceed_or"}, _py("tail", "output='t'")])
    return runner._proceed_or_node({"id": step_id}), f"_proceed_or_{step_id}"


@pytest.mark.asyncio
async def test_first_arrival_wins_and_marks_itself_the_winner():
    node, key = _gate_node()

    update = await node({})

    assert update[key] == {"arrivals": 1, "won": True, "last_won": True}


@pytest.mark.asyncio
async def test_later_arrival_does_not_win_again():
    node, key = _gate_node()

    update = await node({key: {"arrivals": 1, "won": True, "last_won": True}})

    assert update[key]["last_won"] is False, "only the winner may route onward"
    assert update[key]["won"] is True
    assert update[key]["arrivals"] == 2


@pytest.mark.asyncio
async def test_failure_before_any_win_keeps_the_sentinel_so_the_run_fails():
    """Nothing has proceeded, so the failure must not be swallowed."""
    node, key = _gate_node()

    update = await node({"__failed_step__": "branch_a"})

    assert "__failed_step__" not in update, "the sentinel must be left in place"
    assert update[key]["won"] is False
    assert update[key]["last_won"] is False


@pytest.mark.asyncio
async def test_failure_after_a_win_is_discarded():
    """A late loser must not retroactively fail a run that already proceeded."""
    node, key = _gate_node()

    update = await node({
        key: {"arrivals": 1, "won": True, "last_won": True},
        "__failed_step__": "slow_branch",
    })

    assert update["__failed_step__"] is None, "the win stands; drop the sentinel"
    assert update[key]["won"] is True
    assert update[key]["last_won"] is False


@pytest.mark.asyncio
async def test_the_router_sends_only_the_winner_onward():
    runner = _runner([
        {"id": "gate", "type": "proceed_or", "next": "tail"},
        _py("tail", "output='t'"),
    ])
    route = runner._make_proceed_or_router("gate", "tail")
    key = "_proceed_or_gate"

    assert route({key: {"last_won": True}}) == "tail"
    assert route({key: {"last_won": False}}) == "__end__"
    assert route({}) == "__end__", "no latch at all must not fall through to the tail"
