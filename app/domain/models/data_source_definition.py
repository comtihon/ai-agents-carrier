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

from app.domain.models.sheet_binding import SheetBinding


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


class ServiceIdentityAuth(BaseModel):
    """Bearer token minted by one of the service's own OAuth2 identities.

    Carries no secret: the token is obtained at request time from the configured
    OAuth2 authorization server (see ``SERVICE_AUTH_*`` settings).  ``identity``
    names which configured identity to use; omit it to use the deployment's
    default one.
    """

    type: Literal["service_identity"] = "service_identity"
    identity: str | None = None


class GoogleAuth(BaseModel):
    """Bearer token for a Google Workspace API, minted by impersonation.

    Carries no secret, like :class:`ServiceIdentityAuth`, but for a different
    reason: on GKE the backend already *is* a Google principal (Workload
    Identity, no key file).  It cannot call Sheets/Drive as itself, because the
    token the metadata server hands out is ``cloud-platform``-scoped and those
    APIs refuse it.  So it mints a token *for another* service account it holds
    ``roles/iam.serviceAccountTokenCreator`` on, with the narrow
    ``target_scopes`` below.  Documents are shared with that account by email,
    which is also what makes access auditable and revocable per document.

    ``impersonate_subject`` names that account.  It is NOT free-form: a caller
    could otherwise point a data source at any service account this backend can
    impersonate and borrow its authority.  It is resolved from
    ``GOOGLE_IMPERSONATE_SA`` at request time and a stored/incoming value that
    disagrees is refused -- see
    ``app.infrastructure.auth.google_token_provider``.
    """

    type: Literal["google"] = "google"
    # None means "the configured one"; a value must equal it.
    impersonate_subject: str | None = None
    scopes: list[str] = Field(default_factory=list)


AnyDataSourceAuth = Annotated[
    Union[BearerAuth, BasicAuth, HeaderAuth, NoAuth, ServiceIdentityAuth, GoogleAuth],
    Field(discriminator="type"),
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
    # Value used when the caller supplies none.  Needed where the target API
    # demands an argument that has no safe implicit value: Google Sheets
    # rejects a write with no ``valueInputOption``, and letting the API pick
    # would mean ``USER_ENTERED`` — which turns a value starting with ``=``
    # into a live formula.  ``None`` means "no default"; a default never makes
    # a ``required`` param optional, it only fills a blank one.
    default: Any | None = None


# Verbs that carry their arguments in the query string rather than a body.
_QUERY_METHODS = frozenset({"GET", "DELETE", "HEAD", "OPTIONS"})


class Paginate(BaseModel):
    """Pagination strategy for a single operation.

    Three kinds, and each works wherever the arguments have to go -- a query
    string, a JSON body, or GraphQL variables (see ``location``), flat or
    nested:

    cursor:  ``cursor_path`` is a JMESPath expression pointing at the next
             cursor in the response; it is sent back as ``param``. Covers
             HubSpot's ``paging.next.after`` and Relay's
             ``pageInfo.endCursor`` alike.
    page:    ``param`` carries a 1-based page number.
    offset:  ``param`` carries the number of items already fetched -- an
             offset/limit API, including GraphQL's usual
             ``pagination: {limit, skip}`` input object.

    Declaring one is what makes paging the SOURCE's business: the executor
    walks the pages and a caller says only how many rows it wants (or
    nothing, for all of them). Without a block here, a workflow has to pass
    page arguments itself, which is how ``pagination`` ended up as visible
    step configuration reading ``[object Object]``.

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
    # Where the cursor / page number / offset goes. For an HTTP source this is
    # a query-string argument. For a GraphQL source it is a variable, and it
    # may be a DOTTED PATH into a nested input object -- control-center takes
    # ``pagination: PaginationInput{limit, skip}``, so ``pagination.skip``
    # reaches the right field where a flat name could not.
    param: str
    # Where the page size goes, same dotted-path rules. Declaring it is what
    # lets the executor choose the page size, and therefore what lets a caller
    # ask for fewer rows than a whole page.
    size_param: str | None = None
    # Rows requested per page when ``size_param`` is set.
    page_size: int = 100
    # WHERE the pagination arguments go.
    #
    # "auto" is right almost always and is chosen from the operation's shape:
    #   graphql            -> the GraphQL variables
    #   GET/DELETE/HEAD    -> the query string
    #   POST/PUT/PATCH     -> the JSON body
    #
    # The last one is the reason this field exists. A search endpoint that
    # takes its filters in a POST body takes its cursor there too --
    # HubSpot's /objects/{type}/search wants ``{"after": ..., "limit": ...}``
    # in the body, and sending them in the query string does nothing at all,
    # so that operation could not declare pagination and had to be walked by
    # hand. Override only for an endpoint that genuinely disagrees with its
    # own verb.
    location: Literal["auto", "query", "body", "variables"] = "auto"

    def resolved_location(self, *, kind: str, method: str) -> str:
        """Where the pagination arguments actually go for this operation."""
        if self.location != "auto":
            return self.location
        if kind == "graphql":
            return "variables"
        return "query" if (method or "GET").upper() in _QUERY_METHODS else "body"
    # Safety ceiling on the number of pages. 0 means no ceiling: walk until
    # the API says there is nothing left. That is what "fetch everything" has
    # to mean for a source whose whole result is wanted, and it is safe to
    # offer because ``max_result_bytes`` still bounds the total.
    max_pages: int = 10
    # JMESPath expression pointing at the API's own total-record count in a
    # raw page response ("total", "totalCount", "meta.total", …).  It is the
    # only way to know the size of a paginated read *before* walking it: after
    # page one, ``total x bytes-per-item`` projects the finished size, so a
    # read that would blow the byte budget is refused (or spilled to disk) at
    # page two instead of at page seven with the pod already at 900 MB.
    #
    # Optional because plenty of APIs do not report one -- cursor pagination
    # typically cannot.  Without it the executor still enforces the budget,
    # just reactively: it measures what has arrived rather than projecting
    # what will.
    total_path: str | None = None


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
    # What this operation does, in one line.  Read by an agent, not a person:
    # it is what the MCP tool description says beyond the source's own blurb,
    # and six similarly named operations of one source are otherwise
    # indistinguishable to a model.
    description: str = ""
    # HTTP sources: path appended to base_url; supports {params.x} / {op.field}.
    path: str | None = None
    # GraphQL sources: the query document and its variables.
    query: str | None = None
    variables: dict[str, Any] | None = None
    params: list[ParamSpec] = Field(default_factory=list)
    # Query-string arguments, rendered for EVERY method and appended to the
    # URL.  Distinct from a declared param the executor has not consumed
    # elsewhere: such a "loose" param goes into the JSON body on POST/PUT,
    # which is wrong for APIs that take a control argument in the query string
    # of a write -- Google Sheets' ``valueInputOption`` /
    # ``insertDataOption`` are exactly that.  Values are templates like any
    # other, so ``{"valueInputOption": "{params.value_input_option}"}`` works.
    #
    # Named ``query_params`` rather than ``query`` because ``query`` already
    # holds the GraphQL document above.
    query_params: dict[str, str] | None = None
    # Per-operation override of the source's retry policy.  ``None`` -- the
    # default -- uses the source's.  Set it on an operation that is not
    # idempotent (an append), where a retry after a timeout that in fact
    # succeeded duplicates the write.
    retries: "RetryPolicy | None" = None
    # Tri-state override of the "is this call destructive?" verdict that
    # otherwise comes from ``method`` (see
    # ``app.infrastructure.datasources.destructive``).  ``True`` puts a POST
    # /purge behind the approval gate; ``False`` lets a DELETE that only clears
    # a cache run unstopped.  ``None`` — the default — means "not stated", so
    # the verb decides.
    destructive: bool | None = None
    response_schema: dict[str, Any] | None = None
    # JMESPath expression applied to the response payload.
    mapping: str | None = None
    paginate: Paginate | None = None


class PubSubSpec(BaseModel):
    """Google Cloud Pub/Sub topic a ``kind="pubsub"`` data source stands for.

    Such a source is not callable like an HTTP/GraphQL one — it carries no
    operations.  It exists so a topic (with its event schema and, once one has
    been created, its subscription) can be configured once and then reused by
    the ``pubsub`` trigger steps of several workflows.
    """

    # Short name ("orders") or a fully qualified path
    # ("projects/p/topics/orders").  The subscriber resolves short names
    # against the configured project.
    topic: str = ""
    # Subscription to pull from.  Empty means "create one on first use" — the
    # subscriber fills this in and the definition is saved back.
    subscription: str = ""
    # Project override; empty means the backend-wide PUBSUB_PROJECT_ID.
    project_id: str = ""
    # JSON-schema-ish description of the message payload, used the same way as
    # an operation's ``response_schema``: top-level type, required keys and
    # property types.  Named ``event_schema`` because ``schema`` collides with
    # pydantic's own attribute.
    event_schema: dict[str, Any] | None = None


class DataSourceDefinition(BaseModel):
    """Persistent definition of one remote API exposed as named operations."""

    id: str
    name: str = ""
    description: str | None = None
    kind: Literal["http", "graphql", "pubsub"] = "http"
    base_url: str = ""
    auth: AnyDataSourceAuth = Field(default_factory=NoAuth)
    default_headers: dict[str, str] = Field(default_factory=dict)
    operations: list[OperationDefinition] = Field(default_factory=list)
    # Declarative read/write descriptions over a Google spreadsheet, authored
    # in a form rather than written as operation templates (see
    # app/domain/models/sheet_binding.py).  A binding is data, not another kind
    # of operation: saving one *compiles* it into an OperationDefinition of the
    # same name in ``operations`` above, which is why nothing downstream --
    # workflow steps, the approval gate, /try-operation, the MCP tool list --
    # needed to learn a new concept.  Both are stored: the binding is what the
    # editor reads back, the operation is what the runtime calls.
    bindings: list[SheetBinding] = Field(default_factory=list)
    # Only meaningful when kind == "pubsub".
    pubsub: PubSubSpec | None = None
    cache: CachePolicy = Field(default_factory=CachePolicy)
    timeout_seconds: float = 30
    retries: RetryPolicy = Field(default_factory=RetryPolicy)

    # --- Result size ceiling ------------------------------------------------
    # Every result is written to the data stream store and passed on as a
    # DataRef, so neither memory nor the 16 MB checkpoint is a function of
    # result size any more (see app.infrastructure.datasources.datastream).
    # What is still finite is the node's disk, and nothing about a remote API
    # bounds what it returns.
    #
    # Past this many *encoded* bytes the fetch stops and the ref is flagged
    # ``truncated``, which every consumer can read -- a bounded, honest prefix
    # beats filling the disk and getting the pod evicted. 0 disables the
    # ceiling (not advised).
    #
    # Default 512 MiB: larger than any realistic business read, and a fraction
    # of the 40 GB node disk even with several runs streaming at once.
    max_result_bytes: int = 512 * 1024 * 1024

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def get_binding(self, name: str) -> SheetBinding | None:
        for binding in self.bindings:
            if binding.name == name:
                return binding
        return None

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
    refs |= extract_refs(operation.query_params or {})
    return refs


def _validate_paginate(
    definition: DataSourceDefinition, op: OperationDefinition
) -> None:
    """Reject a pagination block that cannot do what it says.

    Caught here rather than at request time because the failure is silent
    otherwise: a dotted path in a query string becomes a literal argument
    named ``pagination.skip``, which the API ignores, so every page comes back
    identical and the walk stops looking like it worked.
    """
    paginate = op.paginate
    if paginate is None:
        return
    where = paginate.resolved_location(kind=definition.kind, method=op.method)
    if where == "query":
        for field in ("param", "size_param"):
            value = getattr(paginate, field, None)
            if value and "." in value:
                raise ValueError(
                    f"Operation '{op.name}': paginate.{field} '{value}' is a "
                    f"nested path, but this operation's pagination goes in the "
                    f"query string, where nesting has no meaning. Use a flat "
                    f"name, or set paginate.location to 'body'."
                )
    if paginate.type == "cursor" and not paginate.cursor_path:
        raise ValueError(
            f"Operation '{op.name}': paginate.type is 'cursor' but no "
            f"cursor_path is set, so there is nothing to read the next cursor "
            f"from — the walk would stop after one page."
        )
    if definition.kind == "graphql" and where == "query":
        raise ValueError(
            f"Operation '{op.name}': a GraphQL operation's pagination goes in "
            f"its variables; 'query' is not a place it can go."
        )


def validate_operations(definition: DataSourceDefinition) -> None:
    """Raise ``ValueError`` when the operation DAG is not resolvable.

    Checks unknown operation references, unknown ``params.*`` names,
    self-references and cycles.  The "one array upstream per operation" rule
    depends on actual response shapes and is therefore enforced at runtime by
    the executor, not here.
    """
    for op in definition.operations:
        _validate_paginate(definition, op)

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
