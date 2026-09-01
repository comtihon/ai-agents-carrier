"""Which data-source operations are dangerous enough to need a human.

The rule is the same one the UI already draws in its risk badge
(``copilot_ui/src/lib/datasourceRisk.ts``): the HTTP verb says what a call can
do, and DELETE can destroy. Keeping the two in step matters — an operation the
canvas paints red must be the one the runtime stops on.

The verb is the default, not the whole answer. Plenty of APIs delete through
``POST /records/purge``, and plenty of DELETE endpoints clear a cache. So an
operation may state ``destructive`` itself, and when it does that wins in both
directions: ``true`` gates a POST, ``false`` lets a DELETE through unstopped.
"""
from __future__ import annotations

from typing import Any

# Verbs that destroy by default.
DESTRUCTIVE_METHODS = frozenset({"DELETE"})


def is_destructive(operation: Any, source: Any = None) -> bool:
    """Whether invoking *operation* needs an approval case opened first.

    ``operation.destructive`` — a tri-state, ``None`` meaning "not stated" —
    overrides the verb when set. GraphQL sources carry no per-operation method;
    they are only ever destructive when they say so, because a mutation is
    indistinguishable from a query at this layer.
    """
    explicit = getattr(operation, "destructive", None)
    if explicit is not None:
        return bool(explicit)
    if source is not None and getattr(source, "kind", "http") == "graphql":
        return False
    method = (getattr(operation, "method", "") or "GET").upper()
    return method in DESTRUCTIVE_METHODS
