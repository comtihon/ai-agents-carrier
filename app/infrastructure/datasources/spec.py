"""Turn an API specification document into data source operations.

Three kinds of document are understood, all detected from the content rather
than from a file extension or content type:

* **OpenAPI / Swagger** (JSON or YAML) — every path × method becomes one
  operation, with path/query parameters and JSON request-body properties as
  ``params`` and the (de-referenced) success response schema as
  ``response_schema``.
* **GraphQL introspection result** — the JSON returned by an introspection
  query, either bare (``{"__schema": …}``) or as received over the wire
  (``{"data": {"__schema": …}}``).
* **GraphQL SDL** — parsed with ``graphql-core`` and converted to an
  introspection result, so both GraphQL forms share one code path.

``$ref`` pointers are inlined so a stored ``response_schema`` stays meaningful
after the specification is gone; recursive DTOs are cut off with an open
``{"type": "object"}`` instead of looping.

Nothing in this module performs I/O — the caller supplies the document.  The
network side lives in ``app.infrastructure.datasources.discovery``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Auto-discovery (GraphQL introspection against the base URL) stays small: the
# result is filled into the editor unattended, so it has to stay reviewable.
MAX_DISCOVERED_OPERATIONS = 40
# An explicitly imported specification is a pick-list — the user selects what
# they need afterwards, so completeness matters more than brevity here.
MAX_IMPORTED_OPERATIONS = 1000
# Depth guard for $ref inlining; deeper nesting is left as-is.
MAX_REF_DEPTH = 12

HTTP_METHODS = ("get", "post", "put", "patch", "delete")

INTROSPECTION_QUERY = """
query DatasourceIntrospection {
  __schema {
    queryType { name }
    mutationType { name }
    types {
      name
      kind
      fields {
        name
        description
        args {
          name
          description
          type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
        }
        type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
      }
    }
  }
}"""


class SpecParseError(ValueError):
    """The supplied document is not a specification we can map to operations."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def map_param_type(t: str | None) -> str:
    if t in ("integer", "number"):
        return "number"
    if t in ("boolean", "array", "object"):
        return t
    return "string"


def slugify(text: str) -> str:
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_").lower()


def is_openapi_doc(doc: Any) -> bool:
    return (
        isinstance(doc, dict)
        and ("openapi" in doc or "swagger" in doc)
        and isinstance(doc.get("paths"), dict)
    )


def introspection_schema(doc: Any) -> dict[str, Any] | None:
    """Return the ``__schema`` block of an introspection result, if this is one."""
    if not isinstance(doc, dict):
        return None
    for candidate in (doc.get("__schema"), (doc.get("data") or {}).get("__schema") if isinstance(doc.get("data"), dict) else None):
        if isinstance(candidate, dict) and isinstance(candidate.get("types"), list):
            return candidate
    return None


def looks_like_graphql_sdl(text: str) -> bool:
    return bool(re.search(r"^\s*(schema\s*\{|(extend\s+)?type\s+\w)", text, re.MULTILINE))


# ---------------------------------------------------------------------------
# $ref resolution
# ---------------------------------------------------------------------------

def resolve_ref(doc: dict[str, Any], ref: str) -> Any | None:
    """Resolve a local JSON pointer (``#/components/schemas/Pet``) against *doc*."""
    if not ref.startswith("#/"):
        return None  # external documents are not fetched
    node: Any = doc
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def deref(
    doc: dict[str, Any],
    node: Any,
    seen: frozenset[str] = frozenset(),
    depth: int = 0,
) -> Any:
    """Inline every local ``$ref`` inside *node*.

    A ``$ref`` already on the current resolution path is replaced with an open
    object: self-referential DTOs (``Node { children: [Node] }``) are common and
    must not expand forever.
    """
    if depth > MAX_REF_DEPTH:
        return node
    if isinstance(node, list):
        return [deref(doc, item, seen, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        rest = {k: v for k, v in node.items() if k != "$ref"}
        if ref in seen:
            return {
                "type": "object",
                "description": f"recursive reference to {ref.rsplit('/', 1)[-1]}",
                **rest,
            }
        target = resolve_ref(doc, ref)
        if target is None:
            return rest or {"type": "object"}
        resolved = deref(doc, target, seen | {ref}, depth + 1)
        if isinstance(resolved, dict):
            # Sibling keys of a $ref (e.g. an overriding description) win.
            return {**resolved, **deref(doc, rest, seen, depth + 1)}
        return resolved

    return {k: deref(doc, v, seen, depth + 1) for k, v in node.items()}


# ---------------------------------------------------------------------------
# OpenAPI / Swagger → operations
# ---------------------------------------------------------------------------

def openapi_base_url(doc: dict[str, Any]) -> str | None:
    """Best-effort base URL declared by the document itself.

    OpenAPI 3 states it in ``servers``; Swagger 2.0 spreads it over
    ``schemes`` / ``host`` / ``basePath``.  Templated OpenAPI 3 server URLs get
    their variables substituted with the declared defaults, since a URL with
    ``{region}`` still in it is not usable.
    """
    servers = doc.get("servers")
    if isinstance(servers, list):
        for server in servers:
            if not isinstance(server, dict):
                continue
            url = server.get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            variables = server.get("variables")
            if isinstance(variables, dict):
                for name, spec in variables.items():
                    default = (spec or {}).get("default") if isinstance(spec, dict) else None
                    if isinstance(default, str):
                        url = url.replace(f"{{{name}}}", default)
            if "{" in url:
                continue  # still templated — not usable as a base URL
            return url.rstrip("/") or "/"

    host = doc.get("host")
    if isinstance(host, str) and host:
        schemes = doc.get("schemes")
        scheme = schemes[0] if isinstance(schemes, list) and schemes else "https"
        base_path = doc.get("basePath") if isinstance(doc.get("basePath"), str) else ""
        return f"{scheme}://{host}{base_path}".rstrip("/")
    return None


def _param_entries(doc: dict[str, Any], raw: Any) -> list[dict[str, Any]]:
    """De-referenced parameter objects of a ``parameters`` list."""
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in raw:
        resolved = deref(doc, item) if isinstance(item, dict) else None
        if isinstance(resolved, dict) and resolved.get("name"):
            entries.append(resolved)
    return entries


def _body_schema(doc: dict[str, Any], spec: dict[str, Any], params: list[dict[str, Any]]) -> dict[str, Any] | None:
    """JSON schema of the request body — OpenAPI 3 ``requestBody`` or Swagger ``in: body``."""
    body = deref(doc, spec.get("requestBody") or {})
    content = body.get("content") if isinstance(body, dict) else None
    if isinstance(content, dict):
        media_types = [
            m for m in content
            if isinstance(m, str) and (m == "application/json" or m.endswith("+json"))
        ]
        for media in media_types or list(content):
            entry = content.get(media)
            if isinstance(entry, dict) and isinstance(entry.get("schema"), dict):
                schema = dict(entry["schema"])
                if body.get("required"):
                    schema.setdefault("__required__", True)
                return schema

    for param in params:  # Swagger 2.0
        if param.get("in") == "body" and isinstance(param.get("schema"), dict):
            schema = dict(param["schema"])
            if param.get("required"):
                schema.setdefault("__required__", True)
            return schema
    return None


def _body_params(doc: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Body properties as caller-supplied params.

    The executor sends every param a non-GET operation does not consume in its
    path template as the JSON request body, so top-level body properties map
    one-to-one onto ``params``.  A body that is not an object becomes a single
    ``body`` param.
    """
    body_required = bool(schema.pop("__required__", False))
    resolved = deref(doc, schema)
    if not isinstance(resolved, dict):
        return []
    props = resolved.get("properties")
    required = set(resolved.get("required") or [])
    if not isinstance(props, dict) or not props:
        if not resolved.get("type") and not resolved.get("oneOf") and not resolved.get("allOf"):
            return []
        return [{
            "name": "body",
            "type": map_param_type(resolved.get("type")),
            "required": body_required,
            "description": resolved.get("description") or "request body",
        }]
    return [
        {
            "name": name,
            "type": map_param_type(prop.get("type") if isinstance(prop, dict) else None),
            "required": name in required,
            "description": (prop.get("description") if isinstance(prop, dict) else "") or "",
        }
        for name, prop in props.items()
    ]


def _response_schema(doc: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    """De-referenced schema of the operation's success response, if declared."""
    responses = spec.get("responses")
    if not isinstance(responses, dict):
        return None
    for status in ("200", "201", "2XX", "default"):
        ok = responses.get(status)
        ok = deref(doc, ok) if isinstance(ok, dict) else None
        if not isinstance(ok, dict):
            continue
        content = ok.get("content")
        if isinstance(content, dict):
            for media in ("application/json", *[m for m in content if isinstance(m, str) and m.endswith("+json")]):
                entry = content.get(media)
                if isinstance(entry, dict) and isinstance(entry.get("schema"), dict):
                    return deref(doc, entry["schema"])
        if isinstance(ok.get("schema"), dict):  # Swagger 2.0
            return deref(doc, ok["schema"])
    return None


def openapi_to_operations(
    doc: dict[str, Any],
    *,
    max_operations: int = MAX_DISCOVERED_OPERATIONS,
) -> list[dict[str, Any]]:
    """Convert an OpenAPI/Swagger document into operation dicts."""
    paths = doc.get("paths") or {}
    ops: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        shared = _param_entries(doc, methods.get("parameters"))
        for method in HTTP_METHODS:
            spec = methods.get(method)
            if not isinstance(spec, dict):
                continue

            all_params = [*shared, *_param_entries(doc, spec.get("parameters"))]
            addressable = [p for p in all_params if p.get("in") in ("path", "query")]

            params = [
                {
                    "name": p["name"],
                    # swagger 2.0 puts the type inline; OpenAPI 3 nests it.
                    "type": map_param_type(
                        (p.get("schema") or {}).get("type") or p.get("type")
                    ),
                    "required": True if p.get("in") == "path" else bool(p.get("required")),
                    "description": p.get("description") or "",
                }
                for p in addressable
            ]

            body_schema = _body_schema(doc, spec, all_params)
            if body_schema is not None:
                known = {p["name"] for p in params}
                params.extend(
                    p for p in _body_params(doc, body_schema) if p["name"] not in known
                )

            # {petId} in the OpenAPI path becomes the template {params.petId}.
            path = re.sub(r"\{([^{}]+)\}", r"{params.\1}", raw_path)
            query_params = [p for p in addressable if p.get("in") == "query"]
            if query_params:
                path += "?" + "&".join(
                    f"{p['name']}={{params.{p['name']}}}" for p in query_params
                )

            operation_id = spec.get("operationId")
            if isinstance(operation_id, str) and operation_id:
                name = slugify(operation_id)
            else:
                name = f"{method}_{slugify(raw_path) or 'root'}"
            while name in seen:
                name = f"{name}_"
            seen.add(name)

            ops.append(
                {
                    "name": name,
                    "method": method.upper(),
                    "path": path,
                    "params": params,
                    "response_schema": _response_schema(doc, spec),
                    "mapping": None,
                    "summary": (spec.get("summary") or spec.get("description") or "").strip()[:300],
                }
            )
            if len(ops) >= max_operations:
                return ops
    return ops


# ---------------------------------------------------------------------------
# GraphQL → operations
# ---------------------------------------------------------------------------

def unwrap_type(t: dict[str, Any]) -> dict[str, Any]:
    """Strip NON_NULL / LIST wrappers from a GraphQL type ref."""
    cur = t
    while cur.get("ofType") and cur.get("kind") in ("NON_NULL", "LIST"):
        cur = cur["ofType"]
    return cur


def type_ref_to_string(t: dict[str, Any]) -> str:
    if t.get("kind") == "NON_NULL" and t.get("ofType"):
        return f"{type_ref_to_string(t['ofType'])}!"
    if t.get("kind") == "LIST" and t.get("ofType"):
        return f"[{type_ref_to_string(t['ofType'])}]"
    return t.get("name") or "String"


def gql_param_type(t: dict[str, Any]) -> str:
    inner = unwrap_type(t)
    of_type = t.get("ofType") or {}
    if t.get("kind") == "LIST" or (
        t.get("kind") == "NON_NULL" and of_type.get("kind") == "LIST"
    ):
        return "array"
    name = inner.get("name")
    if name in ("Int", "Float"):
        return "number"
    if name == "Boolean":
        return "boolean"
    if name in ("String", "ID"):
        return "string"
    return "object" if inner.get("kind") == "INPUT_OBJECT" else "string"


def scalar_selection(field: dict[str, Any], types: list[dict[str, Any]]) -> str:
    """Selection set of up to 8 scalar sub-fields, or ``__typename`` fallback."""
    inner = unwrap_type(field.get("type") or {})
    if inner.get("kind") in ("SCALAR", "ENUM"):
        return ""
    target = next((t for t in types if t.get("name") == inner.get("name")), None)
    scalars = [
        f["name"]
        for f in (target or {}).get("fields") or []
        if unwrap_type(f.get("type") or {}).get("kind") in ("SCALAR", "ENUM")
    ][:8]
    return f" {{ {' '.join(scalars) if scalars else '__typename'} }}"


def graphql_to_operations(
    fields: list[dict[str, Any]],
    types: list[dict[str, Any]],
    *,
    max_operations: int = MAX_DISCOVERED_OPERATIONS,
    operation_type: str = "query",
    taken: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert introspected root-type fields into operation dicts."""
    ops: list[dict[str, Any]] = []
    used = taken if taken is not None else set()
    for field in fields[:max_operations]:
        args = field.get("args") or []
        var_defs = ", ".join(
            f"${a['name']}: {type_ref_to_string(a.get('type') or {})}" for a in args
        )
        arg_use = ", ".join(f"{a['name']}: ${a['name']}" for a in args)
        document = (
            f"{operation_type} {field['name']}{f'({var_defs})' if var_defs else ''} "
            f"{{ {field['name']}{f'({arg_use})' if arg_use else ''}"
            f"{scalar_selection(field, types)} }}"
        )
        name = field["name"]
        while name in used:
            name = f"{name}_"
        used.add(name)
        ops.append(
            {
                "name": name,
                "method": "POST",
                "query": document,
                "variables": (
                    {a["name"]: f"{{params.{a['name']}}}" for a in args} if args else None
                ),
                "params": [
                    {
                        "name": a["name"],
                        "type": gql_param_type(a.get("type") or {}),
                        "required": (a.get("type") or {}).get("kind") == "NON_NULL",
                        "description": a.get("description") or "",
                    }
                    for a in args
                ],
                "response_schema": None,
                "mapping": None,
                "summary": (field.get("description") or "").strip()[:300],
            }
        )
    return ops


def _root_fields(schema: dict[str, Any], root_key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(fields, types) of the named root type (``queryType`` / ``mutationType``)."""
    types = schema.get("types")
    types = types if isinstance(types, list) else []
    root = schema.get(root_key) or {}
    root_name = root.get("name") if isinstance(root, dict) else None
    if not root_name:
        return [], types
    target = next(
        (t for t in types if isinstance(t, dict) and t.get("name") == root_name),
        None,
    )
    fields = (target or {}).get("fields")
    return (fields if isinstance(fields, list) else []), types


def graphql_schema_to_operations(
    schema: dict[str, Any],
    *,
    max_operations: int = MAX_DISCOVERED_OPERATIONS,
    include_mutations: bool = False,
) -> list[dict[str, Any]]:
    """Operations for a whole introspected schema.

    Mutations are opt-in: they are only wanted where the user reviews and picks
    what to keep, never in the unattended auto-discovery path — a datasource
    silently pre-filled with delete operations is a trap.
    """
    query_fields, types = _root_fields(schema, "queryType")
    taken: set[str] = set()
    ops = graphql_to_operations(
        query_fields, types, max_operations=max_operations, taken=taken
    )
    if include_mutations and len(ops) < max_operations:
        mutation_fields, _ = _root_fields(schema, "mutationType")
        ops.extend(
            graphql_to_operations(
                mutation_fields,
                types,
                max_operations=max_operations - len(ops),
                operation_type="mutation",
                taken=taken,
            )
        )
    return ops


def sdl_to_introspection(sdl: str) -> dict[str, Any]:
    """Parse GraphQL SDL into an introspection result.

    Reuses ``graphql-core`` rather than hand-rolling a parser, so the SDL and
    introspection import paths cannot disagree about a schema.
    """
    try:
        from graphql import build_schema
        from graphql.utilities import introspection_from_schema
    except ImportError as exc:  # pragma: no cover — dependency is declared
        raise SpecParseError(
            "GraphQL SDL import needs the 'graphql-core' package on the backend"
        ) from exc
    try:
        schema = build_schema(sdl, assume_valid=True)
        return introspection_from_schema(schema, descriptions=True)
    except Exception as exc:
        raise SpecParseError(f"Invalid GraphQL SDL: {exc}") from exc


# ---------------------------------------------------------------------------
# Document loading + dispatch
# ---------------------------------------------------------------------------

def load_document(text: str) -> Any:
    """Parse *text* as JSON, then YAML; return the text itself if neither."""
    stripped = text.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except Exception:
        return text
    return loaded if isinstance(loaded, (dict, list)) else text


def parse_spec(
    raw: str | bytes,
    *,
    source: str = "specification",
    max_operations: int = MAX_IMPORTED_OPERATIONS,
) -> dict[str, Any]:
    """Map a specification document onto operations.

    Returns ``{"kind": "openapi"|"graphql", "source": …, "base_url": str|None,
    "operations": [...]}``.  Raises :class:`SpecParseError` when the document is
    not a specification this can map — never for a merely empty one, which
    yields zero operations instead.
    """
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SpecParseError("Specification must be UTF-8 text (JSON, YAML or GraphQL SDL)") from exc
    else:
        text = raw
    if not text.strip():
        raise SpecParseError("Specification is empty")

    doc = load_document(text)

    if is_openapi_doc(doc):
        assert isinstance(doc, dict)
        return {
            "kind": "openapi",
            "source": source,
            "base_url": openapi_base_url(doc),
            "operations": openapi_to_operations(doc, max_operations=max_operations),
        }

    schema = introspection_schema(doc)
    if schema is not None:
        return {
            "kind": "graphql",
            "source": source,
            "base_url": None,
            "operations": graphql_schema_to_operations(
                schema, max_operations=max_operations, include_mutations=True
            ),
        }

    if looks_like_graphql_sdl(text):
        schema = introspection_schema(sdl_to_introspection(text))
        if schema is not None:
            return {
                "kind": "graphql",
                "source": source,
                "base_url": None,
                "operations": graphql_schema_to_operations(
                    schema, max_operations=max_operations, include_mutations=True
                ),
            }

    if isinstance(doc, dict) and ("openapi" in doc or "swagger" in doc):
        raise SpecParseError("OpenAPI document has no 'paths' section")
    raise SpecParseError(
        "Unrecognised specification: expected an OpenAPI/Swagger document "
        "(JSON or YAML), a GraphQL introspection result, or GraphQL SDL"
    )
