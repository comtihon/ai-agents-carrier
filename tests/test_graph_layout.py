"""What the canvas is told about a graph written through the management API.

The failure being pinned down is not a crash: a definition with no coordinates
renders in the editor's fallback snake, which reads the step array and not the
graph, so a fan-out looks like a sequence. These tests assert the shape of the
layout rather than exact pixels where the pixel is not the point -- what matters
is that depth follows the edges, that an operator lands between the branches it
belongs to, and that nothing overlaps.
"""
from __future__ import annotations

import pytest

from app.application.graph_layout import (
    NODE_H,
    NODE_W,
    X_STEP,
    Y_STEP,
    apply_layout,
    layout_positions,
)


def _no_overlap(positions: dict[str, dict[str, int]]) -> bool:
    items = list(positions.values())
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if abs(a["x"] - b["x"]) < NODE_W and abs(a["y"] - b["y"]) < NODE_H:
                return False
    return True


def test_a_straight_chain_runs_left_to_right_on_one_row():
    steps = [{"id": "a", "type": "llm"}, {"id": "b", "type": "llm"},
             {"id": "c", "type": "llm"}]
    pos = layout_positions(steps)
    assert pos["a"]["x"] < pos["b"]["x"] < pos["c"]["x"]
    assert pos["a"]["y"] == pos["b"]["y"] == pos["c"]["y"]


def test_the_chain_follows_declared_edges_not_array_order():
    """Array order is a red herring here -- the wiring runs backwards through it."""
    steps = [
        {"id": "third", "type": "llm", "next": "END"},
        {"id": "first", "type": "http", "next": "second"},
        {"id": "second", "type": "llm", "next": "third"},
    ]
    pos = layout_positions(steps)
    assert pos["first"]["x"] < pos["second"]["x"] < pos["third"]["x"]


def test_a_fan_out_puts_every_branch_in_the_same_column():
    steps = [
        {"id": "fan", "type": "parallel", "targets": ["a", "b", "c"]},
        {"id": "a", "type": "llm", "next": "END"},
        {"id": "b", "type": "llm", "next": "END"},
        {"id": "c", "type": "llm", "next": "END"},
    ]
    pos = layout_positions(steps)
    assert pos["a"]["x"] == pos["b"]["x"] == pos["c"]["x"] == pos["fan"]["x"] + X_STEP
    ys = sorted(p["y"] for p in (pos["a"], pos["b"], pos["c"]))
    assert ys[1] - ys[0] >= Y_STEP and ys[2] - ys[1] >= Y_STEP


def test_a_fan_out_is_centred_against_its_branches():
    steps = [
        {"id": "fan", "type": "parallel", "targets": ["a", "b", "c"]},
        {"id": "a", "type": "llm", "next": "END"},
        {"id": "b", "type": "llm", "next": "END"},
        {"id": "c", "type": "llm", "next": "END"},
    ]
    pos = layout_positions(steps)
    branch_mid = sum(pos[b]["y"] for b in ("a", "b", "c")) / 3
    assert pos["fan"]["y"] == pytest.approx(branch_mid, abs=1)


def test_a_join_is_centred_against_the_branches_arriving_at_it():
    steps = [
        {"id": "fan", "type": "parallel", "targets": ["a", "b"]},
        {"id": "a", "type": "llm", "next": "gather"},
        {"id": "b", "type": "llm", "next": "gather"},
        {"id": "gather", "type": "join", "next": "END"},
    ]
    pos = layout_positions(steps)
    assert pos["gather"]["x"] > pos["a"]["x"] == pos["b"]["x"]
    assert pos["gather"]["y"] == pytest.approx((pos["a"]["y"] + pos["b"]["y"]) / 2, abs=1)


def test_uneven_branches_still_meet_at_the_join():
    """One branch two steps long, one a single step: the join sits right of both."""
    steps = [
        {"id": "fan", "type": "parallel", "targets": ["long1", "short"]},
        {"id": "long1", "type": "llm", "next": "long2"},
        {"id": "long2", "type": "llm", "next": "gather"},
        {"id": "short", "type": "llm", "next": "gather"},
        {"id": "gather", "type": "join", "next": "END"},
    ]
    pos = layout_positions(steps)
    assert pos["gather"]["x"] > pos["long2"]["x"] > pos["long1"]["x"]
    assert pos["gather"]["x"] > pos["short"]["x"]
    assert _no_overlap(pos)


def test_a_declared_loop_does_not_push_its_target_rightwards():
    """A revision loop's target keeps the depth its forward edges give it.

    Counting the loop-back towards depth is what would make `review` sit right
    of the step that loops into it, which is backwards from how the graph runs.
    """
    steps = [
        {"id": "review", "type": "llm", "next": "gate"},
        {"id": "gate", "type": "switch",
         "routes": [{"next": "revise"}, {"next": "publish"}]},
        {"id": "revise", "type": "llm", "next": "review"},
        {"id": "publish", "type": "slack", "next": "END"},
    ]
    pos = layout_positions(steps)
    assert pos["review"]["x"] < pos["gate"]["x"] < pos["revise"]["x"]
    assert _no_overlap(pos)


def test_a_graph_that_is_all_cycle_still_gets_positions():
    steps = [
        {"id": "a", "type": "llm", "next": "b"},
        {"id": "b", "type": "llm", "next": "a"},
    ]
    pos = layout_positions(steps)
    assert set(pos) == {"a", "b"}
    assert _no_overlap(pos)


def test_switch_branches_do_not_overlap_each_other():
    steps = [
        {"id": "gate", "type": "switch",
         "routes": [{"next": "x"}, {"next": "y"}, {"next": "z"}]},
        {"id": "x", "type": "llm", "next": "END"},
        {"id": "y", "type": "llm", "next": "END"},
        {"id": "z", "type": "llm", "next": "END"},
    ]
    pos = layout_positions(steps)
    assert _no_overlap(pos)


def test_unrelated_triggers_are_stacked_not_drawn_through_each_other():
    steps = [
        {"id": "cron_a", "type": "cron", "next": "a"},
        {"id": "a", "type": "llm", "next": "END"},
        {"id": "cron_b", "type": "cron", "next": "b"},
        {"id": "b", "type": "llm", "next": "END"},
    ]
    pos = layout_positions(steps)
    assert pos["cron_a"]["x"] == pos["cron_b"]["x"]
    assert abs(pos["cron_a"]["y"] - pos["cron_b"]["y"]) >= Y_STEP
    assert _no_overlap(pos)


def test_an_implicit_fall_through_is_laid_out_like_the_edge_it_becomes():
    """Steps declaring nothing chain to their neighbour, so the layout chains too."""
    steps = [{"id": "a", "type": "http"}, {"id": "b", "type": "llm"}]
    assert layout_positions(steps)["b"]["x"] > layout_positions(steps)["a"]["x"]


def test_an_empty_or_malformed_step_list_yields_no_positions():
    assert layout_positions([]) == {}
    assert layout_positions([{"type": "llm"}, "not-a-step"]) == {}


def test_apply_layout_keeps_positions_somebody_already_arranged():
    stored = {"nodes": {"a": {"x": 999, "y": -40}, "b": {"x": 1200, "y": -40}}}
    steps = [{"id": "a", "type": "llm"}, {"id": "b", "type": "llm"}]
    assert apply_layout(stored, steps)["nodes"] == stored["nodes"]


def test_apply_layout_places_a_new_step_clear_of_the_arranged_ones():
    stored = {"nodes": {"a": {"x": 40, "y": 40}, "b": {"x": 320, "y": 40}}}
    steps = [{"id": "a", "type": "llm"}, {"id": "b", "type": "llm"},
             {"id": "c", "type": "llm"}]
    nodes = apply_layout(stored, steps)["nodes"]
    assert nodes["a"] == {"x": 40, "y": 40} and nodes["b"] == {"x": 320, "y": 40}
    assert _no_overlap(nodes)


def test_relayout_replaces_the_stored_arrangement():
    stored = {"nodes": {"a": {"x": 999, "y": -40}, "b": {"x": 1200, "y": -40}}}
    steps = [{"id": "a", "type": "llm"}, {"id": "b", "type": "llm"}]
    nodes = apply_layout(stored, steps, relayout=True)["nodes"]
    assert nodes["a"]["x"] == 40
    assert nodes["b"]["x"] == 40 + X_STEP


def test_apply_layout_drops_positions_for_steps_that_are_gone():
    stored = {"nodes": {"a": {"x": 40, "y": 40}, "deleted": {"x": 320, "y": 40}}}
    nodes = apply_layout(stored, [{"id": "a", "type": "llm"}])["nodes"]
    assert set(nodes) == {"a"}


def test_apply_layout_leaves_the_rest_of_ui_alone():
    """Edge waypoints and addon placements belong to the canvas, not to us."""
    stored = {
        "nodes": {"a": {"x": 40, "y": 40}},
        "edges": {"a:b": {"waypoints": [{"x": 1, "y": 2}]}},
        "addons": [{"id": "slack-1", "type": "slack", "attachedTo": "a"}],
    }
    out = apply_layout(stored, [{"id": "a", "type": "llm"}])
    assert out["edges"] == stored["edges"]
    assert out["addons"] == stored["addons"]
