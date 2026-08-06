"""Declarative data source definitions.

A ``DataSourceDefinition`` describes a remote HTTP or GraphQL API together with
the named operations that can be invoked on it.  Operations form a DAG: an
operation template may reference the caller-supplied inputs via
``{params.<name>}`` and the result of another operation of the same source via
``{<operation>.<field.path>}``.  The executor
(``app.infrastructure.datasources.executor``) resolves that DAG at call time.

Auth blocks carry the secret values themselves (token / password / header
value) as part of the stored definition; the executor uses them directly at
request time.  API responses redact these fields (see
``app.api.routes.datasources``).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Auth (discriminated union — mirrors app/domain/models/agent_addon.py)
# ---------------------------------------------------------------------------

# NOTE: secret values (token / password / value) are stored unencrypted in
# the persistence backend (Mongo). The REST API redacts them in responses.

class BearerAuth(BaseModel):
    type: Literal["bearer"] = "bearer"
    token: str


class BasicAuth(BaseModel):
    type: Literal["basic"] = "basic"
    username: str
    password: str


class HeaderAuth(BaseModel):
    type: Literal["header"] = "header"
    header_name: str
    value: str


class NoAuth(BaseModel):
    type: Literal["none"] = "none"


AnyDataSourceAuth = Annotated[
    Union[BearerAuth, BasicAuth, HeaderAuth, NoAuth], Field(discriminator="type")
]


# ---------------------------------------------------------------------------
# Operation building blocks
# ---------------------------------------------------------------------------

class ParamSpec(BaseModel):
    """One caller-supplied input of an operation."""

    name: str
    type: Literal["string", "number", "boolean", "array", "object"] = "string"
    required: bool = True
    description: str = ""


class Paginate(BaseModel):
    """Pagination strategy for a single operation.

    cursor:  ``cursor_path`` is a JMESPath expression pointing at the next
             cursor in the response; it is sent back as ``param``.
    page:    ``param`` carries a 1-based page number.
    offset:  ``param`` carries the number of items already fetched.

    ``items_path`` (all modes) is a JMESPath expression pointing at the
    items array within a raw (unmapped) page response. It is used, when the
    operation has no ``mapping``, to detect an empty page (stop looping) and
    to concatenate pages — without it, a dict-shaped response with no
    ``mapping`` never looks "empty" and pagination silently runs to
    ``max_pages``, returning raw page dicts instead of a concatenated list.
    """

    type: Literal["cursor", "page", "offset"] = "page"
    cursor_path: str | None = None
    items_path: str | None = None
    param: str
    max_pages: int = 10


class CachePolicy(BaseModel):
    ttl_seconds: int = 0
    key_template: str | None = None


class RetryPolicy(BaseModel):
    attempts: int = 1
    backoff: float = 0.5


class OperationDefinition(BaseModel):
    """One named, invocable operation of a data source."""

    name: str
    method: str = "GET"
    # HTTP sources: path appended to base_url; supports {params.x} / {op.field}.
    path: str | None = None
    # GraphQL sources: the query document and its variables.
    query: str | None = None
    variables: dict[str, Any] | None = None
    params: list[ParamSpec] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None
    # JMESPath expression applied to the response payload.
    mapping: str | None = None
    paginate: Paginate | None = None


class DataSourceDefinition(BaseModel):
    """Persistent definition of one remote API exposed as named operations."""

    id: str
    name: str = ""
    description: str | None = None
    kind: Literal["http", "graphql"] = "http"
    base_url: str = ""
    auth: AnyDataSourceAuth = Field(default_factory=NoAuth)
    default_headers: dict[str, str] = Field(default_factory=dict)
    operations: list[OperationDefinition] = Field(default_factory=list)
    cache: CachePolicy = Field(default_factory=CachePolicy)
    timeout_seconds: float = 30
    retries: RetryPolicy = Field(default_factory=RetryPolicy)

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def get_operation(self, name: str) -> OperationDefinition | None:
        for op in self.operations:
            if op.name == name:
                return op
        return None

    def touch(self) -> None:
        from datetime import timezone
        self.updated_at = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = self.updated_at


# ---------------------------------------------------------------------------
# Reference extraction + DAG validation
# ---------------------------------------------------------------------------

# {params.owner}  /  {list_repos.items[].name}
REF_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_-]*)\.([^{}]+)\}")


def extract_refs(text: Any) -> set[tuple[str, str]]:
    """Return every ``(head, path)`` reference found in *text*.

    ``head`` is either the literal ``"params"`` or the name of another
    operation; ``path`` is the remainder of the placeholder.  Dicts and lists
    are walked recursively so operation ``variables`` can be scanned too.
    """
    if isinstance(text, str):
        return {(m.group(1), m.group(2)) for m in REF_PATTERN.finditer(text)}
    if isinstance(text, dict):
        refs: set[tuple[str, str]] = set()
        for key, value in text.items():
            refs |= extract_refs(key)
            refs |= extract_refs(value)
        return refs
    if isinstance(text, list):
        refs = set()
        for item in text:
            refs |= extract_refs(item)
        return refs
    return set()


def operation_refs(operation: OperationDefinition) -> set[tuple[str, str]]:
    """Every reference used by the templated fields of one operation."""
    refs: set[tuple[str, str]] = set()
    refs |= extract_refs(operation.path or "")
    refs |= extract_refs(operation.query or "")
    refs |= extract_refs(operation.variables or {})
    return refs


def validate_operations(definition: DataSourceDefinition) -> None:
    """Raise ``ValueError`` when the operation DAG is not resolvable.

    Checks unknown operation references, unknown ``params.*`` names,
    self-references and cycles.  The "one array upstream per operation" rule
    depends on actual response shapes and is therefore enforced at runtime by
    the executor, not here.
    """
    names = [op.name for op in definition.operations]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate operation name(s): {', '.join(sorted(duplicates))}")

    known = set(names)
    deps: dict[str, set[str]] = {}
    for op in definition.operations:
        declared = {p.name for p in op.params}
        op_deps: set[str] = set()
        for head, path in operation_refs(op):
            if head == "params":
                param_name = path.split(".")[0].split("[")[0]
                if param_name not in declared:
                    raise ValueError(
                        f"Operation '{op.name}' references unknown param "
                        f"'{param_name}'"
                    )
                continue
            if head == op.name:
                raise ValueError(f"Operation '{op.name}' references itself")
            if head not in known:
                raise ValueError(
                    f"Operation '{op.name}' references unknown operation '{head}'"
                )
            op_deps.add(head)
        deps[op.name] = op_deps

    # Kahn topological sort — anything left over is part of a cycle.
    indegree = {name: len(deps[name]) for name in deps}
    queue = [name for name, degree in indegree.items() if degree == 0]
    resolved = 0
    while queue:
        current = queue.pop()
        resolved += 1
        for name, op_deps in deps.items():
            if current in op_deps:
                indegree[name] -= 1
                if indegree[name] == 0:
                    queue.append(name)
    if resolved != len(deps):
        remaining = sorted(name for name in indegree if indegree[name] > 0)
        raise ValueError(
            f"Cyclic operation dependencies: {', '.join(remaining)}"
        )
