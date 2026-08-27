"""Where a canvas should draw each step of a workflow.

A definition written through the management API carries no coordinates, and the
editor's fallback for a step it has no position for is a six-column snake: node
*i* goes at column ``i % 6``, wrapping onto a new row. That ordering is the step
array, which for a straight line is right by accident and for anything else is
wrong -- a ``parallel`` and its branches land in a row as if they ran in
sequence, a ``join`` sits wherever the array happens to put it, and an edge that
crosses a row wrap reads as a leap backwards. The gaps compound it: 185px of
pitch for a 176px-wide node leaves nine pixels between neighbours.

So the layout is computed from the graph instead, and stored under
``ui.nodes``, which is where the canvas already looks first.

The shape is the usual layered one, flowing left to right because that is the
direction the node handles face:

* **Layer** is longest-path depth, so a step sits to the right of everything
  that can reach it. Edges that close a loop are left out of the depth
  calculation -- a revision loop would otherwise push its own target rightwards
  forever -- but the loop is still drawn, as an edge running back.
* **Order within a layer** comes from a few barycentre sweeps: a node drifts
  towards the average position of its neighbours in the adjacent layer, which
  is what stops branches from crossing.
* **The y coordinate** is then the closest arrangement to those averages that
  still keeps a clear gap between nodes, solved exactly (pool-adjacent
  violators) rather than approximated. This is what makes fan-out and fan-in
  read correctly: a ``parallel`` ends up centred against its branches and a
  ``join`` centred against the branches arriving at it, because the average of
  their positions is the middle.
* **Disconnected components** -- two triggers with unrelated tails -- are laid
  out separately and stacked, so neither is drawn through the other.
"""
from __future__ import annotations

from app.application.step_normalization import declared_edges, normalize_edges

# Matched to what the canvas renders: a step node is 176px wide (`w-44`) and
# about 44px tall. The remaining pitch is deliberate breathing room -- edge
# labels sit on the horizontal gap, and the vertical one has to survive a node
# with addons unfolded beneath it.
NODE_W = 176
NODE_H = 44
X_STEP = 280
Y_STEP = 110
ORIGIN_X = 40
ORIGIN_Y = 40
COMPONENT_GAP = Y_STEP

Position = dict[str, int]


def _graph(steps: list) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """Step ids in array order, plus successor and predecessor maps.

    Edges come from the same normalisation the engine and the graph endpoint
    use, so what is drawn is what runs -- including fall-through edges the
    document never stated.
    """
    normalized = normalize_edges(steps)
    ids = [
        s["id"] for s in normalized
        if isinstance(s, dict) and isinstance(s.get("id"), str) and s["id"]
    ]
    seen: set[str] = set()
    ordered = [i for i in ids if not (i in seen or seen.add(i))]
    succ: dict[str, list[str]] = {i: [] for i in ordered}
    pred: dict[str, list[str]] = {i: [] for i in ordered}
    for src, dst in sorted(declared_edges(normalized)):
        if src in succ and dst in pred and dst not in succ[src]:
            succ[src].append(dst)
            pred[dst].append(src)
    return ordered, succ, pred


def _components(ids: list[str], succ: dict[str, list[str]],
                pred: dict[str, list[str]]) -> list[list[str]]:
    """Weakly-connected components, each in step-array order."""
    index = {n: i for i, n in enumerate(ids)}
    seen: set[str] = set()
    out: list[list[str]] = []
    for start in ids:
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            node = stack.pop()
            group.append(node)
            for nb in succ[node] + pred[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        out.append(sorted(group, key=index.__getitem__))
    return out


def _forward_only(group: list[str], succ: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Edges that are not loop-backs, found by DFS from the component's roots.

    An edge onto a node still open on the DFS stack closes a cycle. Dropping it
    is what keeps depth finite; nothing else about the edge changes.
    """
    members = set(group)
    incoming = {n: 0 for n in group}
    for node in group:
        for dst in succ[node]:
            if dst in members:
                incoming[dst] += 1
    roots = [n for n in group if incoming[n] == 0] or [group[0]]

    forward: set[tuple[str, str]] = set()
    on_stack: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> None:
        on_stack.add(node)
        for dst in succ[node]:
            if dst not in members or dst == node:
                continue
            if dst in on_stack:
                continue  # loop-back: drawn, but not counted towards depth
            forward.add((node, dst))
            if dst not in done:
                visit(dst)
        on_stack.discard(node)
        done.add(node)

    for root in roots:
        if root not in done:
            visit(root)
    # A node only reachable through a cycle we cut is still unvisited; entering
    # it as its own root gives the rest of its subgraph depths to work from.
    for node in group:
        if node not in done:
            visit(node)
    return forward


def _depths(group: list[str], forward: set[tuple[str, str]]) -> dict[str, int]:
    """Longest-path depth over the cycle-free edges."""
    fsucc: dict[str, list[str]] = {n: [] for n in group}
    indeg = {n: 0 for n in group}
    for src, dst in sorted(forward):
        fsucc[src].append(dst)
        indeg[dst] += 1

    depth = {n: 0 for n in group}
    queue = [n for n in group if indeg[n] == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for dst in fsucc[node]:
            depth[dst] = max(depth[dst], depth[node] + 1)
            indeg[dst] -= 1
            if indeg[dst] == 0:
                queue.append(dst)
    return depth


def _order_layers(layers: dict[int, list[str]], fpred: dict[str, list[str]],
                  fsucc: dict[str, list[str]], sweeps: int = 4) -> None:
    """Reorder each layer in place so adjacent-layer edges cross less."""
    for sweep in range(sweeps):
        keys = sorted(layers)
        if sweep % 2:
            keys.reverse()
        for key in keys:
            rank = {
                n: i
                for other in (key - 1, key + 1)
                for i, n in enumerate(layers.get(other, []))
            }
            neighbours = fpred if sweep % 2 == 0 else fsucc
            current = {n: i for i, n in enumerate(layers[key])}

            def barycentre(node: str) -> float:
                seen = [rank[n] for n in neighbours[node] if n in rank]
                return sum(seen) / len(seen) if seen else float(current[node])

            layers[key].sort(key=lambda n: (barycentre(n), current[n]))


def _pack(desired: list[float], gap: float) -> list[float]:
    """Closest arrangement to *desired* keeping consecutive entries *gap* apart.

    Pool-adjacent-violators on ``desired[i] - i * gap``: the minimum-squared-
    error non-decreasing fit, which after adding the offsets back is the
    minimum-squared-error placement satisfying the separation. Averaging a
    pooled block is exactly the "centre the operator on its branches" behaviour
    -- it falls out of the fit rather than being special-cased.
    """
    blocks: list[list[float]] = []  # [value, weight]
    for i, want in enumerate(desired):
        blocks.append([want - i * gap, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2 = blocks.pop()
            v1, w1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    flat: list[float] = []
    for value, weight in blocks:
        flat.extend([value] * int(weight))
    return [value + i * gap for i, value in enumerate(flat)]


def _component_layout(group: list[str], succ: dict[str, list[str]],
                      iterations: int = 6) -> dict[str, Position]:
    forward = _forward_only(group, succ)
    depth = _depths(group, forward)

    fsucc: dict[str, list[str]] = {n: [] for n in group}
    fpred: dict[str, list[str]] = {n: [] for n in group}
    for src, dst in sorted(forward):
        fsucc[src].append(dst)
        fpred[dst].append(src)

    index = {n: i for i, n in enumerate(group)}
    layers: dict[int, list[str]] = {}
    for node in group:
        layers.setdefault(depth[node], []).append(node)
    for nodes in layers.values():
        nodes.sort(key=index.__getitem__)
    _order_layers(layers, fpred, fsucc)

    y = {n: float(i) * Y_STEP for nodes in layers.values() for i, n in enumerate(nodes)}
    for step in range(iterations):
        keys = sorted(layers)
        neighbours = fpred if step % 2 == 0 else fsucc
        if step % 2:
            keys.reverse()
        for key in keys:
            nodes = layers[key]
            desired = []
            for node in nodes:
                anchors = [y[n] for n in neighbours[node]]
                desired.append(sum(anchors) / len(anchors) if anchors else y[node])
            for node, value in zip(nodes, _pack(desired, Y_STEP)):
                y[node] = value

    top = min(y.values())
    return {
        node: {"x": ORIGIN_X + depth[node] * X_STEP, "y": round(y[node] - top)}
        for node in group
    }


def layout_positions(steps: list) -> dict[str, Position]:
    """Canvas coordinates for every step in *steps*, keyed by step id."""
    ids, succ, pred = _graph(steps)
    if not ids:
        return {}
    positions: dict[str, Position] = {}
    offset = ORIGIN_Y
    for group in _components(ids, succ, pred):
        placed = _component_layout(group, succ)
        for node, pos in placed.items():
            positions[node] = {"x": pos["x"], "y": pos["y"] + offset}
        offset = max(p["y"] for p in positions.values()) + Y_STEP + COMPONENT_GAP
    return positions


def _free_slot(wanted: Position, taken: list[Position]) -> Position:
    """*wanted*, pushed down until it no longer overlaps an occupied box."""
    x, y = wanted["x"], wanted["y"]
    while any(
        abs(p["x"] - x) < NODE_W and abs(p["y"] - y) < NODE_H for p in taken
    ):
        y += Y_STEP
    return {"x": x, "y": y}


def apply_layout(ui: dict | None, steps: list, *, relayout: bool = False) -> dict:
    """Return *ui* with ``nodes`` holding a position for every step.

    Coordinates already stored are kept: someone may have arranged this graph by
    hand on the canvas, and rewriting their arrangement because a step was added
    is not an improvement. A step with no position gets the computed one, nudged
    clear of anything already sitting there. ``relayout=True`` throws the stored
    arrangement away and lays the whole graph out afresh.

    Positions for steps that no longer exist are dropped -- they are invisible
    on the canvas but would otherwise accumulate in the definition forever.
    """
    ui = dict(ui or {})
    ids, _, _ = _graph(steps)
    computed = layout_positions(steps)

    stored = ui.get("nodes")
    kept: dict[str, Position] = {}
    if not relayout and isinstance(stored, dict):
        for node in ids:
            pos = stored.get(node)
            if isinstance(pos, dict) and isinstance(pos.get("x"), (int, float)) \
                    and isinstance(pos.get("y"), (int, float)):
                kept[node] = {"x": round(pos["x"]), "y": round(pos["y"])}

    # Addon positions live under `ui.addons`, so anything in `nodes` that is not
    # a step is stale and goes; non-step keys are never re-added.
    nodes: dict[str, Position] = {}
    for node in ids:
        if node in kept:
            nodes[node] = kept[node]
        elif node in computed:
            nodes[node] = _free_slot(computed[node], list(kept.values()))
    if nodes:
        ui["nodes"] = nodes
    else:
        ui.pop("nodes", None)
    return ui
