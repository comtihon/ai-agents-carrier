"""The compiled graph must contain no edge the definition does not state.

This is the property that failed in production. A step list can compile to a
graph with edges that appear nowhere in the document, because a step declaring
no destination falls through to whatever sits next in the array. An editor
rendering the document then draws one graph while the engine runs another, and
the gap is invisible from either side.

Normalising on ingest closes it: every implicit edge is written out, so the
document is the graph. These tests hold that line for every definition shipped
in the repo, so a terse file or a re-serialised step list cannot reintroduce a
phantom edge without failing the build.
"""
from __future__ import annotations

import glob

import pytest
import yaml

from app.application.step_normalization import declared_edges, implicit_edges, normalize_edges
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner


class _FakeLLM:
    """Stands in for the chat model: building a graph never calls it."""


def _definition_files() -> list[str]:
    return sorted(glob.glob("graphs/*.yaml") + glob.glob("graph_definitions/*.yaml"))


def _compiled_step_edges(definition: dict) -> set[tuple[str, str]]:
    """Step-to-step edges LangGraph actually compiled, START/END excluded."""
    runner = YamlGraphRunner(definition, _FakeLLM(), lambda *a, **k: [])
    ids = {s["id"] for s in definition["steps"] if s.get("id")}
    return {
        (e.source, e.target)
        for e in runner.graph.get_graph().edges
        if e.source in ids and e.target in ids
    }


@pytest.mark.parametrize("path", _definition_files())
def test_shipped_definition_compiles_to_exactly_what_it_declares(path: str) -> None:
    definition = yaml.safe_load(open(path).read())
    if not definition or not definition.get("steps"):
        pytest.skip(f"{path} declares no steps")

    definition["steps"] = normalize_edges(definition["steps"])

    assert _compiled_step_edges(definition) == declared_edges(definition["steps"]), (
        f"{path}: the compiled graph and the definition disagree about the edges"
    )


@pytest.mark.parametrize("path", _definition_files())
def test_normalisation_is_idempotent(path: str) -> None:
    """Saving a workflow repeatedly must not keep rewriting it."""
    definition = yaml.safe_load(open(path).read())
    if not definition or not definition.get("steps"):
        pytest.skip(f"{path} declares no steps")

    once = normalize_edges(definition["steps"])
    assert normalize_edges(once) == once


def test_a_fall_through_into_a_branch_is_made_visible_not_silent() -> None:
    """The CSM deadline-watcher shape, which the editor used to produce.

    ``done`` sits immediately before ``branch_a``, a step the switch already
    routes to, so step order alone wires the tail back into a branch. There is
    no structural way to tell this apart from ``code-review-loop``, which closes
    its revision loop the same way on purpose — so normalisation states the edge
    rather than dropping it, and ``implicit_edges`` names it so an author who
    did not mean it can see it. Silence here is what cost eleven Slack posts.
    """
    raw = [
        {"id": "start", "type": "python", "next": "gate"},
        {"id": "gate", "type": "switch", "routes": [
            {"next": "branch_a", "when": "True"},
            {"next": "branch_b"},
        ]},
        {"id": "done", "type": "python"},
        {"id": "branch_a", "type": "python", "next": "done"},
        {"id": "branch_b", "type": "python", "next": "done"},
    ]

    assert ("done", "branch_a") in implicit_edges(raw)

    steps = normalize_edges(raw)
    assert next(s for s in steps if s["id"] == "done")["next"] == "branch_a"
    assert ("done", "branch_a") in declared_edges(steps)
    # Stated outright, it is no longer positional: nothing changes if it moves.
    assert normalize_edges(steps) == steps


def test_an_explicit_end_terminates_wherever_the_step_sits() -> None:
    """The fix applied to both production workflows, and what the editor now
    writes for a node with no outgoing edge: position stops mattering."""
    raw = [
        {"id": "start", "type": "python", "next": "gate"},
        {"id": "gate", "type": "switch", "routes": [{"next": "branch_a"}]},
        {"id": "done", "type": "python", "next": "END"},
        {"id": "branch_a", "type": "python", "next": "done"},
    ]

    assert implicit_edges(raw) == []
    assert ("done", "branch_a") not in declared_edges(normalize_edges(raw))


def test_a_genuine_straight_line_keeps_its_sequencing() -> None:
    """Terse authoring is the reason fall-through exists; expanding it must not
    silently sever a chain that no router ever enters."""
    steps = normalize_edges([
        {"id": "first", "type": "python"},
        {"id": "second", "type": "python"},
        {"id": "third", "type": "python"},
    ])

    assert [s.get("next") for s in steps] == ["second", "third", "END"]


def test_parallel_without_targets_is_terminated_explicitly() -> None:
    """A parallel declares its edges through `targets`, so normalisation has to
    write there rather than leaving a `next` the builder would ignore."""
    steps = normalize_edges([
        {"id": "fan", "type": "parallel"},
        {"id": "after", "type": "python", "next": "END"},
    ])

    assert steps[0]["targets"] == ["after"]
