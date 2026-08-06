"""Server-side connectivity probing and schema discovery for data sources.

Given a base URL (plus optional auth), this module:

1. Probes the URL — any HTTP response means "reachable"; 401/403 means
   "unauthorized"; a connection/timeout error means "unreachable".
2. When reachable, tries to auto-discover an API schema: OpenAPI/Swagger JSON
   at well-known paths, then GraphQL introspection (order reversed for
   ``kind == "graphql"``), converting whatever is found into
   ``OperationDefinition``-shaped dicts.

This is the backend counterpart of the browser-side prober the UI used to
ship; running it server-side avoids CORS blind spots and lets the stored
secrets authenticate the probe.  Nothing here raises for target-server
failures — every outcome is encoded in the returned ``ProbeOutcome``.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from app.infrastructure.datasources.executor import build_auth_headers

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 8.0

OPENAPI_PATHS = [
    "/openapi.json",
    "/swagger.json",
    "/api-docs",
    "/v2/api-docs",
    "/v3/api-docs",
    "/swagger/v1/swagger.json",
]

MAX_DISCOVERED_OPERATIONS = 40

INTROSPECTION_QUERY = """
query DatasourceIntrospection {
  __schema {
    queryType { name }
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


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


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


def openapi_to_operations(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an OpenAPI/Swagger document into operation dicts."""
    paths = doc.get("paths") or {}
    ops: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        shared = methods.get("parameters")
        shared = shared if isinstance(shared, list) else []
        for method in ("get", "post", "put", "patch", "delete"):
            spec = methods.get(method)
            if not isinstance(spec, dict):
                continue

            spec_params = spec.get("parameters")
            spec_params = spec_params if isinstance(spec_params, list) else []
            all_params = [
                p
                for p in [*shared, *spec_params]
                if isinstance(p, dict)
                and p.get("name")
                and p.get("in") in ("path", "query")
            ]

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
                for p in all_params
            ]

            # {petId} in the OpenAPI path becomes the template {params.petId}.
            path = re.sub(r"\{([^{}]+)\}", r"{params.\1}", raw_path)
            query_params = [p for p in all_params if p.get("in") == "query"]
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

            responses = spec.get("responses") or {}
            ok = responses.get("200") or responses.get("201") or {}
            ok = ok if isinstance(ok, dict) else {}
            content = ok.get("content") or {}
            json_content = content.get("application/json") if isinstance(content, dict) else None
            json_content = json_content if isinstance(json_content, dict) else {}
            response_schema = json_content.get("schema") or ok.get("schema")

            ops.append(
                {
                    "name": name,
                    "method": method.upper(),
                    "path": path,
                    "params": params,
                    "response_schema": response_schema if isinstance(response_schema, dict) else None,
                    "mapping": None,
                }
            )
            if len(ops) >= MAX_DISCOVERED_OPERATIONS:
                return ops
    return ops


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
    fields: list[dict[str, Any]], types: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert introspected query-type fields into operation dicts."""
    ops: list[dict[str, Any]] = []
    for field in fields[:MAX_DISCOVERED_OPERATIONS]:
        args = field.get("args") or []
        var_defs = ", ".join(
            f"${a['name']}: {type_ref_to_string(a.get('type') or {})}" for a in args
        )
        arg_use = ", ".join(f"{a['name']}: ${a['name']}" for a in args)
        query = (
            f"query {field['name']}{f'({var_defs})' if var_defs else ''} "
            f"{{ {field['name']}{f'({arg_use})' if arg_use else ''}"
            f"{scalar_selection(field, types)} }}"
        )
        ops.append(
            {
                "name": field["name"],
                "method": "POST",
                "query": query,
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
            }
        )
    return ops


# ---------------------------------------------------------------------------
# Network probing + discovery
# ---------------------------------------------------------------------------

async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    method: str = "GET",
    json_body: Any | None = None,
) -> Any | None:
    """GET/POST *url*; return the parsed JSON body or None on any failure."""
    try:
        response = await client.request(method, url, headers=headers, json=json_body)
        if response.status_code >= 400:
            return None
        return response.json()
    except Exception:
        return None


async def _try_openapi(
    client: httpx.AsyncClient, base_url: str, headers: dict[str, str]
) -> dict[str, Any] | None:
    candidates = [base_url, *(join_url(base_url, p) for p in OPENAPI_PATHS)]
    for url in candidates:
        doc = await _fetch_json(client, url, headers)
        if is_openapi_doc(doc):
            return {
                "kind": "openapi",
                "source": url,
                "operations": openapi_to_operations(doc),
            }
    return None


async def _try_graphql(
    client: httpx.AsyncClient, base_url: str, headers: dict[str, str]
) -> dict[str, Any] | None:
    for url in (base_url, join_url(base_url, "/graphql")):
        doc = await _fetch_json(
            client,
            url,
            {"Content-Type": "application/json", **headers},
            method="POST",
            json_body={"query": INTROSPECTION_QUERY},
        )
        schema = (doc or {}).get("data", {}).get("__schema") if isinstance(doc, dict) else None
        if not isinstance(schema, dict):
            continue
        query_type = schema.get("queryType") or {}
        types = schema.get("types")
        if not query_type.get("name") or not isinstance(types, list):
            continue
        target = next(
            (t for t in types if isinstance(t, dict) and t.get("name") == query_type["name"]),
            None,
        )
        if not target or not target.get("fields"):
            continue
        return {
            "kind": "graphql",
            "source": f"graphql introspection @ {url}",
            "operations": graphql_to_operations(target["fields"], types),
        }
    return None


async def _discover_schema(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    kind: str,
) -> dict[str, Any] | None:
    attempts = (
        (_try_graphql, _try_openapi) if kind == "graphql" else (_try_openapi, _try_graphql)
    )
    for attempt in attempts:
        found = await attempt(client, base_url, headers)
        if found and found["operations"]:
            return found
    return None


async def probe_and_discover(
    base_url: str, kind: str = "http", auth: Any = None
) -> dict[str, Any]:
    """Probe *base_url* (with the given auth block) and try schema discovery.

    Never raises for target-server failures: the outcome is encoded in the
    returned dict — ``url_status`` (ok | unauthorized | unreachable),
    ``auth_status`` (ok | failed | skipped), ``error`` (detail or None) and
    ``discovered`` (schema dict or None).
    """
    auth_type = getattr(auth, "type", "none")
    result: dict[str, Any] = {
        "url_status": "unreachable",
        "auth_status": "skipped" if auth_type == "none" else "failed",
        "error": None,
        "discovered": None,
    }
    try:
        headers = await build_auth_headers(auth) if auth is not None else {}
        async with httpx.AsyncClient(
            timeout=PROBE_TIMEOUT_SECONDS, follow_redirects=True
        ) as client:
            try:
                response = await client.get(base_url, headers=headers)
            except Exception as exc:
                result["error"] = f"Server not reachable: {exc.__class__.__name__}: {exc}"
                return result

            if response.status_code in (401, 403):
                result["url_status"] = "unauthorized"
                if auth_type == "none":
                    result["error"] = "Server requires authentication"
                else:
                    result["error"] = (
                        f"Credentials rejected ({response.status_code})"
                    )
                return result

            result["url_status"] = "ok"
            if auth_type != "none":
                result["auth_status"] = "ok"

            result["discovered"] = await _discover_schema(client, base_url, headers, kind)
            return result
    except Exception as exc:  # pragma: no cover — absolute last resort
        logger.exception("datasource probe failed unexpectedly")
        result["error"] = f"Probe failed: {exc}"
        return result
