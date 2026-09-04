"""Which data-source operations are dangerous enough to need a human.

The rule is the same one the UI draws in its risk badge
(``copilot_ui/src/lib/datasourceRisk.ts``): what a call can do decides both the
colour on the canvas and whether the runtime stops on it. Keeping the two in
step matters — an operation the canvas paints amber must be the one the
runtime treats as a write.

For an HTTP source the verb answers it, and DELETE can destroy. The verb is
the default, not the whole answer: plenty of APIs delete through
``POST /records/purge`` and plenty of DELETE endpoints clear a cache, so an
operation may state ``destructive`` itself, and when it does that wins in both
directions.

For a GraphQL source the verb says nothing at all — every call is a POST, so
reading the verb painted every read-only query as a write and would have put
one behind the approval gate. The document is what distinguishes them, and it
is right here in the operation, so it is what gets read.
"""
from __future__ import annotations

import re
from typing import Any

# Verbs that destroy by default.
DESTRUCTIVE_METHODS = frozenset({"DELETE"})

# A GraphQL operation definition that writes. Matches `mutation`, `mutation
# Name`, `mutation Name($x: Int)` — always followed by the selection set, which
# is what stops a field or a query merely *called* "mutation" from counting.
# Anchored to the start of the document or to whitespace/a closing brace, so
# `... on Mutation` and `mutationCount` do not match.
_MUTATION_RE = re.compile(
    r"(?:^|[\s}])mutation\b\s*[A-Za-z_]\w*?\s*(?:\([^)]*\))?\s*\{",
    re.IGNORECASE | re.DOTALL,
)
# The same, for the anonymous form `mutation { ... }`.
_ANON_MUTATION_RE = re.compile(r"(?:^|[\s}])mutation\s*\{", re.IGNORECASE)

# Comments and string literals are stripped first: a `# mutation` note or a
# `"mutation"` argument value must not make a query look like a write.
_COMMENT_RE = re.compile(r"#[^\n]*")
_STRING_RE = re.compile(r'"""(?:.|\n)*?"""|"(?:\\.|[^"\\])*"')


def graphql_writes(query: str | None) -> bool:
    """Whether a GraphQL document contains a mutation.

    A document with no operation keyword at all is the shorthand query form
    (``{ projects { id } }``) and reads. ``subscription`` reads too — it
    streams, it does not change anything.
    """
    if not query:
        # No document to judge. Not a write on the strength of nothing: an
        # operation with no query cannot run at all, and calling it a write
        # would gate a definition bug behind an approval nobody can answer.
        return False
    stripped = _STRING_RE.sub('""', _COMMENT_RE.sub("", query))
    return bool(_MUTATION_RE.search(stripped) or _ANON_MUTATION_RE.search(stripped))


def is_destructive(operation: Any, source: Any = None) -> bool:
    """Whether invoking *operation* needs an approval case opened first.

    ``operation.destructive`` — a tri-state, ``None`` meaning "not stated" —
    overrides everything below when set.
    """
    explicit = getattr(operation, "destructive", None)
    if explicit is not None:
        return bool(explicit)
    if source is not None and getattr(source, "kind", "http") == "graphql":
        # Read the document, not the verb. Before this, every GraphQL source
        # was unconditionally non-destructive, which is the safe answer for a
        # query and the wrong one for a mutation.
        return graphql_writes(getattr(operation, "query", None))
    method = (getattr(operation, "method", "") or "GET").upper()
    return method in DESTRUCTIVE_METHODS
