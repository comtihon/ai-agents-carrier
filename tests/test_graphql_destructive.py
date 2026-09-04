"""A GraphQL query is a read; only a mutation is a write.

Every GraphQL call is a POST, so reading the HTTP verb classified read-only
queries as writes. That painted a W badge on control-center's read-only source
in the canvas, and the classifier's previous answer -- "GraphQL is never
destructive" -- was the safe result for a query and the wrong one for a
mutation. The document distinguishes them and it is right there on the
operation.
"""
from __future__ import annotations

import pytest

from app.domain.models.data_source_definition import (
    DataSourceDefinition,
    OperationDefinition,
)
from app.infrastructure.datasources.destructive import graphql_writes, is_destructive


def _gql(**op) -> tuple[DataSourceDefinition, OperationDefinition]:
    operation = OperationDefinition(**op)
    source = DataSourceDefinition(
        id="cc", kind="graphql", base_url="https://x/graphql",
        operations=[operation],
    )
    return source, operation


@pytest.mark.parametrize("query", [
    "query projects { projects { id } }",
    "{ projects { id } }",                       # shorthand query form
    "query { statistics { received } }",
    "query Q { mutationCount }",                 # a field merely named so
    'query Q { f(note: "mutation { x }") }',     # a string argument
    "# mutation { x }\nquery Q { f }",           # a comment
    "subscription S { events { id } }",          # streams, changes nothing
])
def test_a_query_document_is_a_read(query):
    assert graphql_writes(query) is False
    source, op = _gql(name="read_it", query=query)
    assert is_destructive(op, source) is False


@pytest.mark.parametrize("query", [
    "mutation { deleteMachine(id: 1) { ok } }",
    "mutation DeleteMachine($id: Int!) { deleteMachine(id: $id) { ok } }",
    "query A { a }\nmutation B { b }",           # both, in one document
])
def test_a_mutation_document_is_a_write(query):
    assert graphql_writes(query) is True
    source, op = _gql(name="write_it", query=query)
    assert is_destructive(op, source) is True


def test_a_graphql_operation_with_no_document_is_not_a_write():
    """It cannot run at all; gating it would need an approval nobody can give."""
    source, op = _gql(name="broken")
    assert is_destructive(op, source) is False


def test_an_explicit_flag_still_wins_in_both_directions():
    source, op = _gql(name="purge", query="query { x }", destructive=True)
    assert is_destructive(op, source) is True

    source, op = _gql(
        name="harmless_mutation", query="mutation { clearCache { ok } }",
        destructive=False,
    )
    assert is_destructive(op, source) is False


def test_http_sources_are_unchanged():
    """The verb still decides for HTTP; only DELETE destroys by default."""
    for method, expected in (
        ("GET", False), ("POST", False), ("PUT", False),
        ("PATCH", False), ("DELETE", True),
    ):
        op = OperationDefinition(name="op", method=method, path="/x")
        source = DataSourceDefinition(id="h", kind="http", base_url="https://x",
                                      operations=[op])
        assert is_destructive(op, source) is expected, method


def test_the_real_control_center_operations_read_as_reads():
    """The source whose description says query-only must classify that way."""
    queries = [
        "query projects($filterBy: ProjectFilterInput) { projects(filterBy: $filterBy) { id } }",
        "query statistics { statistics { received delivered } }",
        "query machines { machines { id name } }",
    ]
    for q in queries:
        source, op = _gql(name="op", query=q)
        assert is_destructive(op, source) is False, q
