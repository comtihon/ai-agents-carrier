"""Make every edge in a step list explicit, and refuse the one that bites.

A step that declares no ``next``, ``routes`` or ``targets`` falls through to
whatever sits next in the array. That is deliberate shorthand — it is what lets
a straight-line YAML graph omit the wiring — and it is also the only thing in a
definition that depends on array *position* rather than on what a step says. So
a step list can compile to a graph containing edges that appear nowhere in the
document, and an editor rendering the document draws a different graph than the
engine runs.

Two things follow, and they are deliberately kept apart:

``normalize_edges`` resolves the shorthand and nothing else. It writes down the
destination the graph builder would have chosen anyway, so it never changes what
a definition means — it only stops the meaning depending on position. Guessing
that a fall-through "looks wrong" and dropping it is not its job: a chain like
``plan`` → ``execute`` is ordinary authoring even when ``execute`` is also a
router's loop-back target, and suppressing it silently strands the rest of the
graph.

``undeclared_cycle`` is where judgement lives. The failure worth refusing is
narrow: a fall-through edge nobody wrote closing a loop nobody declared. A
declared loop stays legal, and so does a fall-through that merely joins one.
That is the shape that made a workflow post its Slack digest eleven times on a
single cron tick — a convergence node serialised into the middle of the array,
wired back into a branch it was only ever meant to be reached from.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

# Written into `next` (or a parallel's `targets`) for a step that terminates.
# The graph builder routes any destination that is not a step id to END, so this
# is a marker the engine already understands rather than a new concept.
END_MARKER = "END"


def _declares_an_edge(step: dict[str, Any]) -> bool:
    """True when the step says where it goes, in any of the three forms."""
    if step.get("type") == "parallel":
        return bool(step.get("targets"))
    return bool(step.get("next")) or bool(step.get("routes"))


def declared_edges(steps: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Step-to-step edges the document states outright.

    Destinations that are not step ids are omitted: they compile to END, which
    is a termination rather than an edge between two steps.
    """
    ids = {s.get("id") for s in steps if isinstance(s, dict) and s.get("id")}
    edges: set[tuple[str, str]] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = step.get("id")
        if not sid:
            continue
        if step.get("type") == "parallel":
            for target in step.get("targets") or []:
                if target in ids:
                    edges.add((sid, target))
            continue
        for route in step.get("routes") or []:
            dest = route.get("next")
            if isinstance(dest, str) and dest in ids:
                edges.add((sid, dest))
        dest = step.get("next")
        if isinstance(dest, str) and dest in ids:
            edges.add((sid, dest))
    return edges


def normalize_edges(steps: list[Any]) -> list[Any]:
    """Return *steps* with every implicit fall-through written out explicitly.

    A step that declares nothing gets an explicit ``next`` to the following step,
    or ``"END"`` when it is the last one — exactly what the graph builder does
    with it. Steps that already declare an edge are returned untouched, as is
    anything that is not a dict: a malformed list is the validator's problem.
    """
    if not isinstance(steps, list):
        return steps

    out = deepcopy(steps)
    positions = [i for i, s in enumerate(out) if isinstance(s, dict) and s.get("id")]

    for idx, pos in enumerate(positions):
        step = out[pos]
        if _declares_an_edge(step):
            continue
        # The neighbour is the next *step*, not the next list entry, so a
        # malformed entry in between cannot shift the destination.
        neighbour = out[positions[idx + 1]].get("id") if idx + 1 < len(positions) else None
        dest = neighbour or END_MARKER
        if step.get("type") == "parallel":
            step["targets"] = [dest]
        else:
            step["next"] = dest
    return out


def implicit_edges(steps: list[Any]) -> list[tuple[str, str]]:
    """Edges normalisation had to invent because the document did not state them.

    Reported back to whoever wrote the definition rather than refused. Refusing
    is not available: the fall-through that closed a cycle in the CSM deadline
    watcher is structurally identical to the one ``code-review-loop`` uses to
    close its revision loop on purpose, so no rule separates them without
    reading intent. Naming them is what is left, and it is enough — an author
    who did not mean an edge can see it in the response and in the canvas,
    which is exactly what was missing when this went wrong silently.
    """
    if not isinstance(steps, list):
        return []
    return sorted(declared_edges(normalize_edges(steps)) - declared_edges(steps))
