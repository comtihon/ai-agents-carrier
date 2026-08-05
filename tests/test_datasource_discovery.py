"""Unit tests for the pure conversion helpers of datasource schema discovery."""
from __future__ import annotations

from app.infrastructure.datasources.discovery import (
    MAX_DISCOVERED_OPERATIONS,
    gql_param_type,
    graphql_to_operations,
    is_openapi_doc,
    join_url,
    map_param_type,
    openapi_to_operations,
    scalar_selection,
    slugify,
    type_ref_to_string,
    unwrap_type,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def test_join_url_strips_trailing_slashes():
    assert join_url("https://api.test/", "/openapi.json") == "https://api.test/openapi.json"
    assert join_url("https://api.test", "/graphql") == "https://api.test/graphql"


def test_map_param_type():
    assert map_param_type("integer") == "number"
    assert map_param_type("number") == "number"
    assert map_param_type("boolean") == "boolean"
    assert map_param_type("array") == "array"
    assert map_param_type("object") == "object"
    assert map_param_type("string") == "string"
    assert map_param_type(None) == "string"
    assert map_param_type("weird") == "string"


def test_slugify():
    assert slugify("getPetById") == "getpetbyid"
    assert slugify("/pets/{petId}") == "pets_petid"
    assert slugify("__weird--name__") == "weird_name"


def test_is_openapi_doc():
    assert is_openapi_doc({"openapi": "3.0.0", "paths": {}})
    assert is_openapi_doc({"swagger": "2.0", "paths": {}})
    assert not is_openapi_doc({"openapi": "3.0.0"})
    assert not is_openapi_doc({"paths": {}})
    assert not is_openapi_doc(None)
    assert not is_openapi_doc([])


# ---------------------------------------------------------------------------
# OpenAPI conversion
# ---------------------------------------------------------------------------

def _openapi_doc(paths):
    return {"openapi": "3.0.0", "paths": paths}


def test_openapi_path_and_query_params_become_templates():
    doc = _openapi_doc({
        "/pets/{petId}": {
            "get": {
                "operationId": "getPetById",
                "parameters": [
                    {"name": "petId", "in": "path", "schema": {"type": "integer"}},
                    {"name": "verbose", "in": "query", "required": False,
                     "schema": {"type": "boolean"}, "description": "More detail"},
                ],
            },
        },
    })
    ops = openapi_to_operations(doc)
    assert len(ops) == 1
    op = ops[0]
    assert op["name"] == "getpetbyid"
    assert op["method"] == "GET"
    assert op["path"] == "/pets/{params.petId}?verbose={params.verbose}"
    assert op["params"] == [
        {"name": "petId", "type": "number", "required": True, "description": ""},
        {"name": "verbose", "type": "boolean", "required": False, "description": "More detail"},
    ]


def test_openapi_shared_parameters_apply_to_all_methods():
    doc = _openapi_doc({
        "/items/{id}": {
            "parameters": [{"name": "id", "in": "path", "schema": {"type": "string"}}],
            "get": {},
            "delete": {},
        },
    })
    ops = openapi_to_operations(doc)
    assert [op["method"] for op in ops] == ["GET", "DELETE"]
    for op in ops:
        assert op["params"][0]["name"] == "id"
        assert op["path"] == "/items/{params.id}"


def test_openapi_missing_operation_id_uses_method_and_path():
    doc = _openapi_doc({"/pets": {"post": {}}})
    ops = openapi_to_operations(doc)
    assert ops[0]["name"] == "post_pets"


def test_openapi_duplicate_names_are_deduped():
    doc = _openapi_doc({
        "/a": {"get": {"operationId": "op"}},
        "/b": {"get": {"operationId": "op"}},
    })
    names = [op["name"] for op in openapi_to_operations(doc)]
    assert names == ["op", "op_"]


def test_openapi_response_schema_from_200_json_content():
    schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
    doc = _openapi_doc({
        "/pets": {
            "get": {
                "responses": {"200": {"content": {"application/json": {"schema": schema}}}},
            },
        },
    })
    assert openapi_to_operations(doc)[0]["response_schema"] == schema


def test_openapi_swagger2_inline_schema_and_type():
    schema = {"type": "array"}
    doc = {
        "swagger": "2.0",
        "paths": {
            "/pets": {
                "get": {
                    "parameters": [{"name": "limit", "in": "query", "type": "integer"}],
                    "responses": {"200": {"schema": schema}},
                },
            },
        },
    }
    op = openapi_to_operations(doc)[0]
    assert op["params"][0]["type"] == "number"
    assert op["response_schema"] == schema


def test_openapi_operation_count_is_capped():
    doc = _openapi_doc({f"/p{i}": {"get": {}} for i in range(MAX_DISCOVERED_OPERATIONS + 10)})
    assert len(openapi_to_operations(doc)) == MAX_DISCOVERED_OPERATIONS


def test_openapi_body_params_are_ignored():
    doc = _openapi_doc({
        "/pets": {
            "post": {"parameters": [{"name": "body", "in": "body"}]},
        },
    })
    assert openapi_to_operations(doc)[0]["params"] == []


# ---------------------------------------------------------------------------
# GraphQL conversion
# ---------------------------------------------------------------------------

def _non_null(inner):
    return {"kind": "NON_NULL", "name": None, "ofType": inner}


def _scalar(name):
    return {"kind": "SCALAR", "name": name, "ofType": None}


def _obj(name):
    return {"kind": "OBJECT", "name": name, "ofType": None}


def test_unwrap_type_strips_wrappers():
    t = _non_null({"kind": "LIST", "name": None, "ofType": _scalar("Int")})
    assert unwrap_type(t) == _scalar("Int")


def test_type_ref_to_string():
    assert type_ref_to_string(_scalar("Int")) == "Int"
    assert type_ref_to_string(_non_null(_scalar("ID"))) == "ID!"
    assert (
        type_ref_to_string({"kind": "LIST", "name": None, "ofType": _non_null(_scalar("String"))})
        == "[String!]"
    )


def test_gql_param_type():
    assert gql_param_type(_scalar("Int")) == "number"
    assert gql_param_type(_scalar("Float")) == "number"
    assert gql_param_type(_scalar("Boolean")) == "boolean"
    assert gql_param_type(_scalar("ID")) == "string"
    assert gql_param_type({"kind": "LIST", "name": None, "ofType": _scalar("Int")}) == "array"
    assert gql_param_type(_non_null({"kind": "LIST", "name": None, "ofType": _scalar("Int")})) == "array"
    assert gql_param_type({"kind": "INPUT_OBJECT", "name": "Filter", "ofType": None}) == "object"
    assert gql_param_type(_obj("User")) == "string"


def test_scalar_selection_picks_scalar_subfields_capped_at_8():
    types = [
        {
            "name": "User",
            "kind": "OBJECT",
            "fields": (
                [{"name": f"f{i}", "type": _scalar("String")} for i in range(10)]
                + [{"name": "friends", "type": _obj("User")}]
            ),
        },
    ]
    sel = scalar_selection({"name": "user", "type": _obj("User")}, types)
    assert sel == " { f0 f1 f2 f3 f4 f5 f6 f7 }"


def test_scalar_selection_falls_back_to_typename():
    types = [{"name": "Thing", "kind": "OBJECT", "fields": [{"name": "child", "type": _obj("Thing")}]}]
    assert scalar_selection({"name": "thing", "type": _obj("Thing")}, types) == " { __typename }"


def test_scalar_selection_empty_for_scalar_fields():
    assert scalar_selection({"name": "count", "type": _scalar("Int")}, []) == ""


def test_graphql_to_operations_builds_query_variables_and_params():
    types = [
        {
            "name": "User",
            "kind": "OBJECT",
            "fields": [
                {"name": "id", "type": _scalar("ID")},
                {"name": "name", "type": _scalar("String")},
            ],
        },
    ]
    fields = [
        {
            "name": "user",
            "description": None,
            "args": [
                {"name": "id", "description": "User id", "type": _non_null(_scalar("ID"))},
            ],
            "type": _obj("User"),
        },
    ]
    ops = graphql_to_operations(fields, types)
    assert len(ops) == 1
    op = ops[0]
    assert op["name"] == "user"
    assert op["method"] == "POST"
    assert op["query"] == "query user($id: ID!) { user(id: $id) { id name } }"
    assert op["variables"] == {"id": "{params.id}"}
    assert op["params"] == [
        {"name": "id", "type": "string", "required": True, "description": "User id"},
    ]


def test_graphql_to_operations_no_args():
    ops = graphql_to_operations([{"name": "ping", "args": [], "type": _scalar("String")}], [])
    op = ops[0]
    assert op["query"] == "query ping { ping }"
    assert op["variables"] is None
    assert op["params"] == []


def test_graphql_operation_count_is_capped():
    fields = [
        {"name": f"q{i}", "args": [], "type": _scalar("String")}
        for i in range(MAX_DISCOVERED_OPERATIONS + 5)
    ]
    assert len(graphql_to_operations(fields, [])) == MAX_DISCOVERED_OPERATIONS
