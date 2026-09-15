"""Guards for the ``kind="bigquery"`` execution path.

The security-relevant assertions here are the SELECT-only ones and the
parameter-binding one. They are what stop a data source whose SQL is authored
by a model from writing to BigQuery or from being steered by a caller's
argument, so they are tested against the shapes that would actually be tried:
casing, comments, a trailing second statement, and a quote in a value.
"""
from __future__ import annotations

import pytest

from app.domain.models.data_source_definition import (
    BigQuerySpec,
    DataSourceDefinition,
    OperationDefinition,
    ParamSpec,
)
from app.infrastructure.datasources.bigquery import (
    BigQueryError,
    _assert_select_only,
    _pre_check,
    _query_parameters,
    _rows,
    run_operation,
)
from app.infrastructure.datasources.destructive import is_destructive


class _Job:
    """Stand-in for a completed dry-run job."""

    def __init__(self, statement_type, total_bytes_processed=0):
        self.statement_type = statement_type
        self.total_bytes_processed = total_bytes_processed


# ---------------------------------------------------------------- pre-check

@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM t WHERE 1=1",
        "delete from t",                       # casing
        "  /* harmless */ DROP TABLE t",       # leading comment
        "SELECT 1; DROP TABLE t",              # second statement
        "INSERT INTO t VALUES (1)",
        "MERGE t USING s ON x",
        "CREATE OR REPLACE TABLE t AS SELECT 1",
        "EXPORT DATA OPTIONS(uri='gs://x') AS SELECT 1",
    ],
)
def test_pre_check_refuses_non_reads(sql):
    with pytest.raises(BigQueryError):
        _pre_check(sql, op_name="op")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select count(*) from t",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "-- a note about DELETE semantics\nSELECT 1",   # keyword only in a comment
        "SELECT 'insert' AS label",                      # keyword only in a literal-ish position
    ],
)
def test_pre_check_allows_reads(sql):
    _pre_check(sql, op_name="op")


def test_pre_check_refuses_empty():
    with pytest.raises(BigQueryError, match="empty sql"):
        _pre_check("   ", op_name="op")


# ------------------------------------------------- BigQuery's own verdict

def test_non_select_statement_type_is_refused():
    with pytest.raises(BigQueryError, match="not a SELECT"):
        _assert_select_only(_Job("SCRIPT"), op_name="op", max_bytes=1)


def test_select_statement_type_passes():
    _assert_select_only(_Job("SELECT", 10), op_name="op", max_bytes=100)


def test_scan_estimate_over_cap_is_refused_before_running():
    with pytest.raises(BigQueryError, match="would scan"):
        _assert_select_only(
            _Job("SELECT", 50 * 1024**3), op_name="op", max_bytes=1024**3
        )


# ------------------------------------------------------- parameter binding

def test_params_are_bound_not_interpolated():
    op = OperationDefinition(
        name="monthly",
        sql="SELECT 1 WHERE d >= @month_start",
        params=[ParamSpec(name="month_start", type="string")],
    )
    # A value that would end the string literal if it were substituted.
    bound = _query_parameters(op, {"month_start": "2026-08-01' OR '1'='1"})

    assert len(bound) == 1
    assert bound[0].name == "month_start"
    # The hostile value survives intact AS A VALUE -- it never reaches the SQL
    # text, which is the whole point.
    assert bound[0].value == "2026-08-01' OR '1'='1"
    assert "OR" not in (op.sql or "")


def test_undeclared_params_are_not_bound():
    op = OperationDefinition(
        name="op", sql="SELECT @a", params=[ParamSpec(name="a", type="number")]
    )
    bound = _query_parameters(op, {"a": 1, "smuggled": "x"})
    assert [p.name for p in bound] == ["a"]


def test_optional_param_with_no_value_binds_null():
    op = OperationDefinition(
        name="op",
        sql="SELECT @a",
        params=[ParamSpec(name="a", type="string", required=False)],
    )
    bound = _query_parameters(op, {})
    assert bound[0].value is None


def test_unsupported_param_type_is_reported():
    op = OperationDefinition(
        name="op", sql="SELECT 1", params=[ParamSpec(name="a", type="array")]
    )
    with pytest.raises(BigQueryError, match="cannot bind"):
        _query_parameters(op, {"a": [1]})


# ------------------------------------------------------------------- rows

class _Row:
    def __init__(self, d):
        self._d = d

    def items(self):
        return self._d.items()


class _Result:
    def __init__(self, n):
        self._rows = [_Row({"i": i}) for i in range(n)]

    def result(self):
        return iter(self._rows)


def test_rows_raise_rather_than_truncate_silently():
    with pytest.raises(BigQueryError, match="more than"):
        _rows(_Result(10), limit=None, max_rows=5)


def test_caller_limit_returns_a_prefix_quietly():
    out = _rows(_Result(10), limit=3, max_rows=100)
    assert out == [{"i": 0}, {"i": 1}, {"i": 2}]


# ------------------------------------------------------------ integration

@pytest.mark.asyncio
async def test_missing_sql_is_reported():
    source = DataSourceDefinition(id="bq", kind="bigquery")
    op = OperationDefinition(name="op")
    with pytest.raises(BigQueryError, match="has no sql"):
        await run_operation(source, op, {})


@pytest.mark.asyncio
async def test_write_statement_never_reaches_the_client(monkeypatch):
    """The refusal happens before any BigQuery call is constructed."""
    called = False

    def _boom(*a, **k):
        nonlocal called
        called = True
        raise AssertionError("client must not be reached")

    monkeypatch.setattr(
        "app.infrastructure.datasources.bigquery._run_sync", _boom
    )
    source = DataSourceDefinition(id="bq", kind="bigquery")
    op = OperationDefinition(name="wipe", sql="DELETE FROM t")

    with pytest.raises(BigQueryError):
        await run_operation(source, op, {})
    assert called is False


@pytest.mark.asyncio
async def test_spec_defaults_are_passed_through(monkeypatch):
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return [{"ok": 1}]

    monkeypatch.setattr(
        "app.infrastructure.datasources.bigquery._run_sync", _capture
    )
    source = DataSourceDefinition(
        id="bq",
        kind="bigquery",
        bigquery=BigQuerySpec(
            project_id="p", location="EU", maximum_bytes_billed=123, max_rows=7
        ),
    )
    op = OperationDefinition(name="op", sql="SELECT 1")

    out = await run_operation(source, op, {})

    assert out == [{"ok": 1}]
    assert seen["project_id"] == "p"
    assert seen["location"] == "EU"
    assert seen["max_bytes"] == 123
    assert seen["max_rows"] == 7


# ------------------------------------------------------- approval gating

def test_bigquery_operations_are_never_destructive():
    source = DataSourceDefinition(id="bq", kind="bigquery")
    # Even with a verb that would otherwise gate it.
    op = OperationDefinition(name="op", sql="SELECT 1", method="DELETE")
    assert is_destructive(op, source) is False


def test_explicit_destructive_flag_still_wins():
    """An author who says "gate this" is still obeyed."""
    source = DataSourceDefinition(id="bq", kind="bigquery")
    op = OperationDefinition(name="op", sql="SELECT 1", destructive=True)
    assert is_destructive(op, source) is True


@pytest.mark.parametrize(
    "sql",
    [
        # A forbidden word inside a VALUE must not refuse an ordinary read.
        "SELECT 'insert' AS label",
        "SELECT * FROM t WHERE status = 'deleted'",
        "SELECT * FROM `my-project.ds.drop_table_audit`",
        'SELECT "update" AS k',
        "SELECT 1 # a note mentioning DROP\n",
    ],
)
def test_forbidden_words_inside_literals_do_not_refuse_a_read(sql):
    _pre_check(sql, op_name="op")


def test_keyword_outside_a_literal_is_still_refused():
    """Blanking literals must not open a hole."""
    with pytest.raises(BigQueryError):
        _pre_check("SELECT 'ok' AS a; DROP TABLE t", op_name="op")
