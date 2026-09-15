"""Execution path for ``kind="bigquery"`` data sources.

Why this is not an HTTP data source
-----------------------------------
BigQuery does have a REST API, and an ``http`` source could POST to
``jobs.query``.  That was the first design and it was wrong in three ways that
matter here, all of which this module exists to fix:

* **Parameters.**  An http operation renders its arguments by string
  substitution into a template.  Doing that to SQL is SQL injection by
  construction -- a workflow passing a month boundary that happens to contain a
  quote changes the meaning of the statement.  Here every declared param is
  bound as a BigQuery *named query parameter* (``@month_start``), which the
  server treats as a value and never as syntax.  There is deliberately no way
  to interpolate a param into the SQL text.

* **Reading a job is not one request.**  ``jobs.query`` returns a page and a
  job reference; large results need ``getQueryResults`` polling with a page
  token, and the executor's generic pagination cannot express "poll this job
  until ``jobComplete``".  The client library does it properly.

* **Nothing checked what the statement did.**  A POST body is opaque to the
  executor's destructive-operation gate, so ``DELETE FROM`` would have read as
  an ordinary read.  Here the statement type is established by BigQuery itself
  (see ``_assert_select_only``) before a single byte is billed.

Safety posture
--------------
Three independent guards, in order of authority:

1. A textual pre-check, so an obvious mistake fails immediately with a clear
   message rather than after a round trip.
2. A **dry run**, whose ``statement_type`` is BigQuery's own verdict on what
   the statement is.  Anything but ``SELECT`` is refused.  This is the guard
   that actually holds: it cannot be fooled by comments, casing, whitespace or
   a second statement hidden behind a semicolon.
3. ``maximum_bytes_billed``, set on the real run from the source's configured
   cap.  The dry run also reports the scan estimate up front, so a query that
   would exceed the cap is refused *before* it runs rather than being killed
   mid-flight with the bytes already spent.

The identity is the backend's own Workload Identity token, not an impersonated
one.  Unlike Sheets and Drive -- which refuse the ``cloud-platform``-scoped
token the metadata server hands out, and which is why ``GoogleAuth`` exists --
BigQuery accepts it, so a ``bigquery`` source needs no ``auth`` block at all
and holds no secret.  What it can read is therefore exactly what
``langgraph-backend@`` has been granted per dataset in terraform, and is
changed there rather than here.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    OperationDefinition,
    ParamSpec,
)

logger = logging.getLogger(__name__)


class BigQueryError(RuntimeError):
    """A bigquery operation could not be run, or was refused."""


# Statement keywords that are never acceptable, checked before the dry run so
# the common mistake gets an immediate, readable error.  This list is a
# courtesy, NOT the security boundary -- `_assert_select_only` is.
_FORBIDDEN = re.compile(
    r"(?<![\w.])("
    r"INSERT|UPDATE|DELETE|MERGE|TRUNCATE|DROP|ALTER|CREATE|GRANT|REVOKE|"
    r"EXPORT|LOAD|CALL|EXECUTE\s+IMMEDIATE"
    r")(?![\w.])",
    re.IGNORECASE,
)

# Line and block comments, stripped before the textual check so a forbidden
# keyword cannot be smuggled past it inside a comment that BigQuery ignores.
# (``#`` is a comment in BigQuery too.)
_COMMENTS = re.compile(r"--[^\n]*|#[^\n]*|/\*.*?\*/", re.DOTALL)

# String literals, blanked before the keyword scan for the opposite reason to
# comments: a perfectly ordinary read like ``SELECT 'insert' AS label`` or a
# filter on a status value named "deleted" must not be refused because a
# forbidden word appears inside a *value*. BigQuery's triple-quoted and raw
# forms are covered too. The same stripping is done for GraphQL documents in
# ``destructive.graphql_writes``, for the same reason.
_STRINGS = re.compile(
    r"'''(?:.|\n)*?'''"
    r'|"""(?:.|\n)*?"""'
    r"|'(?:\\.|[^'\\\n])*'"
    r'|"(?:\\.|[^"\\\n])*"'
    r"|`(?:[^`])*`",  # backtick-quoted identifiers: `my-project.ds.table`
    re.DOTALL,
)

_DEFAULT_MAX_BYTES_BILLED = 20 * 1024**3  # 20 GiB
_DEFAULT_MAX_ROWS = 50_000


def _strip_comments(sql: str) -> str:
    return _COMMENTS.sub(" ", sql)


def _pre_check(sql: str, *, op_name: str) -> None:
    """Fail fast and legibly on a statement that is obviously not a read.

    This is a courtesy check for the author, not the security boundary: it can
    be both over- and under-inclusive, and ``_assert_select_only`` is what
    actually decides. So it errs towards letting an odd-looking read through
    -- a false refusal here blocks legitimate work with a confusing message,
    while a false pass costs one dry run and is then caught for certain.
    """
    body = _strip_comments(sql).strip()
    if not body:
        raise BigQueryError(f"operation '{op_name}' has an empty sql statement.")

    head = body.lstrip("( \t\r\n").upper()
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        raise BigQueryError(
            f"operation '{op_name}' must be a SELECT (or a WITH ending in one); "
            f"it starts with '{body.split(None, 1)[0][:32]}'."
        )

    found = _FORBIDDEN.search(_STRINGS.sub("''", body))
    if found:
        raise BigQueryError(
            f"operation '{op_name}' contains '{found.group(1).upper()}', which a "
            f"bigquery data source will not run. These sources are read-only."
        )


def _scalar_type(spec: ParamSpec | None) -> str:
    """BigQuery scalar type for a declared param.

    ``array``/``object`` are deliberately unsupported: a workflow that needs to
    filter on a set passes it as a repeated scalar or builds the set in SQL,
    rather than handing the executor a structure it would have to guess a type
    for.
    """
    if spec is None:
        return "STRING"
    match spec.type:
        case "number":
            return "NUMERIC"
        case "boolean":
            return "BOOL"
        case "string":
            return "STRING"
        case other:
            raise BigQueryError(
                f"param '{spec.name}' is declared '{other}', which a bigquery "
                f"operation cannot bind. Use string, number or boolean."
            )


def _query_parameters(op: OperationDefinition, params: dict[str, Any]) -> list[Any]:
    """Bind declared params as named query parameters.

    Only *declared* params are bound.  An undeclared key in ``params`` is
    ignored rather than bound, so a caller cannot introduce a parameter the
    operation's author never wrote into the SQL.
    """
    from google.cloud import bigquery  # imported late: keeps import cost off startup

    by_name = {p.name: p for p in op.params}
    bound: list[Any] = []
    for name, spec in by_name.items():
        if name not in params and spec.default is None:
            # A required-but-missing param is already rejected upstream by
            # _check_required_params; an optional one with no default binds as
            # NULL so `@x IS NULL` works as the "not supplied" test.
            bound.append(
                bigquery.ScalarQueryParameter(name, _scalar_type(spec), None)
            )
            continue
        value = params.get(name, spec.default)
        bound.append(
            bigquery.ScalarQueryParameter(name, _scalar_type(spec), value)
        )
    return bound


def _assert_select_only(job: Any, *, op_name: str, max_bytes: int) -> None:
    """BigQuery's own verdict on the statement, from a completed dry run.

    ``statement_type`` is what the server parsed, so it is immune to the tricks
    a regex is not: casing, comments, a leading no-op, or a second statement
    after a semicolon (which makes the type ``SCRIPT``, not ``SELECT``).
    """
    statement = getattr(job, "statement_type", None)
    if statement != "SELECT":
        raise BigQueryError(
            f"operation '{op_name}' is a {statement or 'unknown'} statement, not a "
            f"SELECT. bigquery data sources are read-only."
        )

    estimate = getattr(job, "total_bytes_processed", None)
    if estimate is not None and estimate > max_bytes:
        raise BigQueryError(
            f"operation '{op_name}' would scan {estimate / 1024**3:.1f} GiB, over "
            f"the source's cap of {max_bytes / 1024**3:.1f} GiB. Narrow the query "
            f"(the tables are month-partitioned -- filter on the partition column) "
            f"or raise maximum_bytes_billed on the data source."
        )


def _rows(job: Any, *, limit: int | None, max_rows: int) -> list[dict[str, Any]]:
    """Materialise the result, refusing to silently truncate.

    A caller-supplied ``limit`` is a deliberate "give me a taste" and is
    applied quietly.  ``max_rows`` is the safety net, and hitting it raises
    instead of returning a short answer: a truncated aggregate looks exactly
    like a real one, and a workflow would report it as fact.
    """
    cap = max_rows if limit is None else min(limit, max_rows)
    out: list[dict[str, Any]] = []
    for row in job.result():
        if len(out) >= cap:
            if limit is not None and cap == limit:
                break  # caller asked for a prefix; this is that prefix
            raise BigQueryError(
                f"query returned more than {max_rows} rows. Aggregate in SQL "
                f"rather than returning raw rows, or raise max_rows on the "
                f"data source."
            )
        out.append(dict(row.items()))
    return out


def _run_sync(
    *,
    sql: str,
    parameters: list[Any],
    project_id: str | None,
    location: str | None,
    max_bytes: int,
    max_rows: int,
    limit: int | None,
    op_name: str,
) -> list[dict[str, Any]]:
    """The blocking half — a BigQuery client call, run on a worker thread."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id) if project_id else bigquery.Client()

    dry = bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, query_parameters=parameters
    )
    probe = client.query(sql, job_config=dry, location=location)
    _assert_select_only(probe, op_name=op_name, max_bytes=max_bytes)

    config = bigquery.QueryJobConfig(
        query_parameters=parameters,
        maximum_bytes_billed=max_bytes,
        # A read that feeds a report should not invent a table as a side
        # effect; BigQuery writes results to an anonymous cached table either
        # way, and this keeps that the only thing it writes.
        priority=bigquery.QueryPriority.INTERACTIVE,
    )
    job = client.query(sql, job_config=config, location=location)
    return _rows(job, limit=limit, max_rows=max_rows)


async def run_operation(
    source: DataSourceDefinition,
    op: OperationDefinition,
    params: dict[str, Any],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run one ``kind="bigquery"`` operation and return its rows.

    Returns a list of dicts, the same shape a mapped HTTP response arrives in,
    so everything downstream of the executor -- JMESPath mapping, streams,
    python steps -- is unchanged.
    """
    sql = (op.sql or "").strip()
    if not sql:
        raise BigQueryError(
            f"operation '{op.name}' of bigquery source '{source.id}' has no sql."
        )

    _pre_check(sql, op_name=op.name)

    spec = source.bigquery
    max_bytes = (
        spec.maximum_bytes_billed
        if spec and spec.maximum_bytes_billed
        else _DEFAULT_MAX_BYTES_BILLED
    )
    max_rows = spec.max_rows if spec and spec.max_rows else _DEFAULT_MAX_ROWS

    parameters = _query_parameters(op, params)

    logger.info(
        "bigquery source '%s': running '%s' (%d bound params)",
        source.id, op.name, len(parameters),
    )
    try:
        return await asyncio.to_thread(
            _run_sync,
            sql=sql,
            parameters=parameters,
            project_id=(spec.project_id if spec else None) or None,
            location=(spec.location if spec else None) or None,
            max_bytes=max_bytes,
            max_rows=max_rows,
            limit=limit,
            op_name=op.name,
        )
    except BigQueryError:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced as a step error
        raise BigQueryError(
            f"bigquery operation '{op.name}' failed: {exc}"
        ) from exc
