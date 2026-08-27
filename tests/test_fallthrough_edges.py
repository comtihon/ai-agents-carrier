"""Array position must not decide where a step goes.

A step that declares no outgoing edge falls through to its neighbour in list
order. That shorthand is deliberate, but it makes position load-bearing, and a
convergence node — one every branch routes into — has no natural position. The
editor serialised such a node into the middle of the array, where the neighbour
happened to be a branch, and the tail of the graph looped until the run failed.

The runner now normalises on ingest, so the steps it compiles always say where
they go. These tests drive real graphs rather than inspecting the step list:
what matters is how many times each node actually runs.
"""
from __future__ import annotations

from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner


class _FakeLLM:
    """Stands in for the chat model: no step below ever calls it."""


def _py(step_id: str, **extra) -> dict:
    return {
        "id": step_id, "type": "python", "code": f"output='{step_id}'",
        "output_key": f"o_{step_id}", "sandbox": False, **extra,
    }


async def _counts(steps: list[dict], thread: str) -> dict[str, int]:
    """Execute *steps* and return how many times each node ran."""
    runner = YamlGraphRunner({"id": "t", "steps": steps}, _FakeLLM(), lambda *a, **k: [])
    counts: dict[str, int] = {}
    async for chunk in runner.graph.astream(
        {"request": "go"}, {"configurable": {"thread_id": thread}}, stream_mode="updates",
    ):
        for node in chunk:
            counts[node] = counts.get(node, 0) + 1
    return counts


async def test_explicit_end_stops_a_mid_array_convergence_node() -> None:
    """The CSM deadline-watcher shape with the fix applied to it.

    ``done`` sits mid-array, immediately before ``branch_a``, which the switch
    already routes to. Declaring ``next: END`` makes its position irrelevant;
    without it the tail cycles until the loop guard fails the run.
    """
    steps = [
        _py("start", next="gate"),
        {"id": "gate", "type": "switch", "routes": [
            {"next": "branch_a", "when": "True"},
            {"next": "branch_b"},
        ]},
        _py("done", next="END"),
        _py("branch_a", next="tail"),
        _py("branch_b", next="done"),
        _py("tail", next="done"),
    ]

    counts = await _counts(steps, "fallthrough-explicit-end")

    assert counts.get("branch_a") == 1
    assert counts.get("tail") == 1
    assert counts.get("done") == 1
    assert "branch_b" not in counts


async def test_plain_sequential_fallthrough_still_connects() -> None:
    """Terse authoring is why fall-through exists: consecutive steps that
    declare nothing run in list order, and the last one ends."""
    steps = [_py("first"), _py("second"), _py("third")]

    counts = await _counts(steps, "fallthrough-sequential")

    assert counts == {"first": 1, "second": 1, "third": 1}


async def test_a_declared_loop_back_still_reaches_its_target() -> None:
    """``code-review-loop`` relies on falling through into a step that a router
    also loops back to. Treating that as a mistake strands the rest of the
    graph, so the chain has to survive normalisation intact."""
    steps = [
        _py("plan"),                                   # falls through to execute
        _py("execute"),                                # falls through to review
        {"id": "review", "type": "switch", "routes": [
            {"next": "notify", "when": "o_execute"},
            {"next": "execute"},
        ]},
        _py("notify", next="END"),
    ]

    counts = await _counts(steps, "fallthrough-declared-loop")

    assert counts.get("plan") == 1
    assert counts.get("execute") == 1
    assert counts.get("notify") == 1
