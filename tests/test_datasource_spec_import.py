"""Tests for specification import: OpenAPI/Swagger, GraphQL introspection, SDL."""
from __future__ import annotations

import json

import pytest

from app.infrastructure.datasources.spec import (
    MAX_IMPORTED_OPERATIONS,
    SpecParseError,
    deref,
    graphql_schema_to_operations,
    openapi_base_url,
    openapi_to_operations,
    parse_spec,
    sdl_to_introspection,
)


def _doc(paths: dict, **extra) -> dict:
    return {"openapi": "3.0.0", "paths": paths, **extra}


def _op(ops: list, name: str) -> dict:
    return next(o for o in ops if o["name"] == name)


# ---------------------------------------------------------------------------
# $ref inlining
# ---------------------------------------------------------------------------

def test_deref_inlines_local_refs():
    doc = {"components": {"schemas": {"Pet": {"type": "object", "properties": {"id": {"type": "integer"}}}}}}
    resolved = deref(doc, {"$ref": "#/components/schemas/Pet"})
    assert resolved == {"type": "object", "properties": {"id": {"type": "integer"}}}


def test_deref_sibling_keys_override_the_target():
    doc = {"components": {"schemas": {"Pet": {"type": "object", "description": "a pet"}}}}
    resolved = deref(doc, {"$ref": "#/components/schemas/Pet", "description": "the pet to feed"})
    assert resolved["description"] == "the pet to feed"


def test_deref_cuts_recursive_dtos_instead_of_looping():
    doc = {
        "components": {
            "schemas": {
                "Node": {
                    "type": "object",
                    "properties": {"children": {"type": "array", "items": {"$ref": "#/components/schemas/Node"}}},
                },
            },
        },
    }
    resolved = deref(doc, {"$ref": "#/components/schemas/Node"})
    inner = resolved["properties"]["children"]["items"]
    assert inner["type"] == "object"
    assert "recursive reference to Node" in inner["description"]


def test_deref_unresolvable_ref_degrades_to_open_object():
    assert deref({}, {"$ref": "#/components/schemas/Missing"}) == {"type": "object"}
    assert deref({}, {"$ref": "https://elsewhere/schema.json"}) == {"type": "object"}


# ---------------------------------------------------------------------------
# OpenAPI: base URL
# ---------------------------------------------------------------------------

def test_openapi_base_url_from_servers():
    assert openapi_base_url(_doc({}, servers=[{"url": "https://api.test/v2/"}])) == "https://api.test/v2"


def test_openapi_base_url_substitutes_server_variable_defaults():
    doc = _doc({}, servers=[{"url": "https://{region}.api.test", "variables": {"region": {"default": "eu"}}}])
    assert openapi_base_url(doc) == "https://eu.api.test"


def test_openapi_base_url_skips_servers_that_stay_templated():
    doc = _doc({}, servers=[{"url": "https://{tenant}.api.test"}, {"url": "https://api.test"}])
    assert openapi_base_url(doc) == "https://api.test"


def test_openapi_base_url_from_swagger2_host_and_base_path():
    doc = {"swagger": "2.0", "paths": {}, "schemes": ["https"], "host": "api.test", "basePath": "/v1"}
    assert openapi_base_url(doc) == "https://api.test/v1"


def test_openapi_base_url_absent():
    assert openapi_base_url(_doc({})) is None


# ---------------------------------------------------------------------------
# OpenAPI: request bodies and DTOs
# ---------------------------------------------------------------------------

def test_openapi_json_body_properties_become_params():
    doc = _doc({
        "/pets": {
            "post": {
                "operationId": "createPet",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {"type": "string", "description": "Pet name"},
                                    "age": {"type": "integer"},
                                    "tags": {"type": "array"},
                                },
                            },
                        },
                    },
                },
            },
        },
    })
    op = _op(openapi_to_operations(doc), "createpet")
    assert op["method"] == "POST"
    assert op["params"] == [
        {"name": "name", "type": "string", "required": True, "description": "Pet name"},
        {"name": "age", "type": "number", "required": False, "description": ""},
        {"name": "tags", "type": "array", "required": False, "description": ""},
    ]


def test_openapi_body_dto_is_resolved_through_ref():
    doc = _doc(
        {
            "/pets": {
                "post": {
                    "requestBody": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/NewPet"}}}},
                },
            },
        },
        components={"schemas": {"NewPet": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}}},
    )
    op = openapi_to_operations(doc)[0]
    assert op["params"] == [{"name": "name", "type": "string", "required": True, "description": ""}]


def test_openapi_non_object_body_becomes_a_single_body_param():
    doc = _doc({"/raw": {"post": {"requestBody": {"required": True, "content": {"application/json": {"schema": {"type": "array"}}}}}}})
    assert openapi_to_operations(doc)[0]["params"] == [
        {"name": "body", "type": "array", "required": True, "description": "request body"},
    ]


def test_openapi_swagger2_body_parameter_is_expanded():
    doc = {
        "swagger": "2.0",
        "paths": {
            "/pets": {
                "post": {
                    "parameters": [
                        {"name": "body", "in": "body", "required": True,
                         "schema": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}},
                    ],
                },
            },
        },
    }
    assert openapi_to_operations(doc)[0]["params"] == [
        {"name": "name", "type": "string", "required": True, "description": ""},
    ]


def test_openapi_body_property_does_not_shadow_a_path_param():
    doc = _doc({
        "/pets/{id}": {
            "put": {
                "parameters": [{"name": "id", "in": "path", "schema": {"type": "string"}}],
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object", "properties": {"id": {"type": "string"}, "name": {"type": "string"}}}}}},
            },
        },
    })
    params = openapi_to_operations(doc)[0]["params"]
    assert [p["name"] for p in params] == ["id", "name"]
    # The path param survives — it is the one the URL template consumes.
    assert params[0]["required"] is True


def test_openapi_response_schema_is_dereferenced():
    listing = {"type": "array", "items": {"$ref": "#/components/schemas/Pet"}}
    responses = {"200": {"content": {"application/json": {"schema": listing}}}}
    doc = _doc(
        {"/pets": {"get": {"responses": responses}}},
        components={"schemas": {"Pet": {"type": "object", "properties": {"id": {"type": "integer"}}}}},
    )
    schema = openapi_to_operations(doc)[0]["response_schema"]
    assert schema == {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}}}}


def test_openapi_parameter_refs_are_resolved():
    doc = _doc(
        {"/pets": {"get": {"parameters": [{"$ref": "#/components/parameters/Limit"}]}}},
        components={"parameters": {"Limit": {"name": "limit", "in": "query", "schema": {"type": "integer"}, "description": "page size"}}},
    )
    op = openapi_to_operations(doc)[0]
    assert op["params"] == [{"name": "limit", "type": "number", "required": False, "description": "page size"}]
    assert op["path"] == "/pets?limit={params.limit}"


def test_openapi_summary_is_carried_for_the_pick_list():
    doc = _doc({"/pets": {"get": {"summary": "List all pets"}}})
    assert openapi_to_operations(doc)[0]["summary"] == "List all pets"


def test_import_cap_is_far_above_the_discovery_cap():
    doc = _doc({f"/p{i}": {"get": {}} for i in range(60)})
    assert len(openapi_to_operations(doc, max_operations=MAX_IMPORTED_OPERATIONS)) == 60


# ---------------------------------------------------------------------------
# GraphQL
# ---------------------------------------------------------------------------

_SDL = """
type Pet { id: ID!, name: String! }
type Query { pets(limit: Int): [Pet], pet(id: ID!): Pet }
type Mutation { deletePet(id: ID!): Boolean }
"""


def test_graphql_schema_to_operations_is_queries_only_by_default():
    schema = sdl_to_introspection(_SDL)["__schema"]
    names = [op["name"] for op in graphql_schema_to_operations(schema)]
    assert names == ["pets", "pet"]


def test_graphql_schema_to_operations_can_include_mutations():
    schema = sdl_to_introspection(_SDL)["__schema"]
    ops = graphql_schema_to_operations(schema, include_mutations=True)
    delete = _op(ops, "deletePet")
    assert delete["query"].startswith("mutation deletePet($id: ID!)")
    assert delete["params"] == [{"name": "id", "type": "string", "required": True, "description": ""}]


def test_graphql_root_field_names_do_not_collide_across_roots():
    schema = sdl_to_introspection(
        "type Query { sync: Boolean }\ntype Mutation { sync: Boolean }"
    )["__schema"]
    names = [op["name"] for op in graphql_schema_to_operations(schema, include_mutations=True)]
    assert names == ["sync", "sync_"]


# ---------------------------------------------------------------------------
# parse_spec dispatch
# ---------------------------------------------------------------------------

def test_parse_spec_reads_yaml_openapi():
    yaml_text = """
openapi: 3.0.0
servers:
  - url: https://api.test
paths:
  /pets:
    get:
      operationId: listPets
"""
    result = parse_spec(yaml_text, source="petstore.yaml")
    assert result["kind"] == "openapi"
    assert result["base_url"] == "https://api.test"
    assert [op["name"] for op in result["operations"]] == ["listpets"]
    assert result["source"] == "petstore.yaml"


def test_parse_spec_reads_json_openapi_from_bytes():
    raw = json.dumps(_doc({"/pets": {"get": {"operationId": "listPets"}}})).encode()
    assert parse_spec(raw)["operations"][0]["name"] == "listpets"


def test_parse_spec_reads_a_wire_introspection_result():
    introspection = sdl_to_introspection(_SDL)
    result = parse_spec(json.dumps({"data": introspection}))
    assert result["kind"] == "graphql"
    # Imports include mutations — the user picks what to keep.
    assert "deletePet" in [op["name"] for op in result["operations"]]


def test_parse_spec_reads_bare_introspection_result():
    result = parse_spec(json.dumps(sdl_to_introspection(_SDL)))
    assert result["kind"] == "graphql"


def test_parse_spec_reads_sdl():
    result = parse_spec(_SDL, source="schema.graphql")
    assert result["kind"] == "graphql"
    assert [op["name"] for op in result["operations"]] == ["pets", "pet", "deletePet"]


def test_parse_spec_rejects_an_empty_document():
    with pytest.raises(SpecParseError, match="empty"):
        parse_spec("   ")


def test_parse_spec_rejects_unrelated_json():
    with pytest.raises(SpecParseError, match="Unrecognised specification"):
        parse_spec(json.dumps({"hello": "world"}))


def test_parse_spec_names_the_missing_paths_section():
    with pytest.raises(SpecParseError, match="no 'paths' section"):
        parse_spec(json.dumps({"openapi": "3.0.0", "info": {}}))


def test_parse_spec_rejects_invalid_sdl():
    with pytest.raises(SpecParseError, match="Invalid GraphQL SDL"):
        parse_spec("type Query { broken(")


def test_parse_spec_rejects_non_utf8_bytes():
    with pytest.raises(SpecParseError, match="UTF-8"):
        parse_spec(b"\xff\xfe\x00openapi")
