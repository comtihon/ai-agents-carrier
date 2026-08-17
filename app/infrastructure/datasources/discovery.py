"""Server-side connectivity probing and schema fetching for data sources.

Two jobs, both server-side so CORS never applies and the stored secrets can
authenticate the request:

1. :func:`probe_and_discover` probes a base URL — any HTTP response means
   "reachable"; 401/403 means "unauthorized"; a connection/timeout error means
   "unreachable".  For ``kind == "graphql"`` it *also* introspects the schema,
   because a GraphQL endpoint URL **is** the way to fetch its schema.  For HTTP
   sources it does not go looking for a schema: guessing at ``/openapi.json``
   and friends produced wrong or partial documents often enough that the schema
   location is now something the user states (see below).
2. :func:`fetch_and_parse_spec` fetches a specification from an explicit URL
   and maps it onto operations via :mod:`app.infrastructure.datasources.spec`.
   Uploaded specification files go through ``spec.parse_spec`` directly.

Neither raises for target-server failures: the probe encodes every outcome in
its returned dict, and the fetch raises :class:`SpecFetchError`, which the API
layer reports as a 422 with the detail.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.infrastructure.datasources.executor import build_auth_headers
from app.infrastructure.datasources.spec import (  # re-exported for callers/tests
    INTROSPECTION_QUERY,
    MAX_DISCOVERED_OPERATIONS,
    MAX_IMPORTED_OPERATIONS,
    SpecParseError,
    gql_param_type,
    graphql_schema_to_operations,
    graphql_to_operations,
    is_openapi_doc,
    map_param_type,
    openapi_to_operations,
    parse_spec,
    scalar_selection,
    slugify,
    type_ref_to_string,
    unwrap_type,
)

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 8.0
SCHEMA_FETCH_TIMEOUT_SECONDS = 20.0
# Specifications are text; anything larger is not one we should be parsing.
MAX_SPEC_BYTES = 8 * 1024 * 1024

__all__ = [
    "INTROSPECTION_QUERY",
    "MAX_DISCOVERED_OPERATIONS",
    "MAX_IMPORTED_OPERATIONS",
    "MAX_SPEC_BYTES",
    "PROBE_TIMEOUT_SECONDS",
    "SCHEMA_FETCH_TIMEOUT_SECONDS",
    "SpecFetchError",
    "SpecParseError",
    "fetch_and_parse_spec",
    "gql_param_type",
    "graphql_schema_to_operations",
    "graphql_to_operations",
    "is_openapi_doc",
    "join_url",
    "map_param_type",
    "openapi_to_operations",
    "parse_spec",
    "probe_and_discover",
    "scalar_selection",
    "slugify",
    "type_ref_to_string",
    "unwrap_type",
]


class SpecFetchError(ValueError):
    """A specification URL could not be fetched (or returned a non-spec body)."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


# ---------------------------------------------------------------------------
# Network probing + GraphQL introspection
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


async def _introspect(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    *,
    include_mutations: bool = False,
    max_operations: int = MAX_DISCOVERED_OPERATIONS,
) -> dict[str, Any] | None:
    """Run the introspection query against *url*; None when it is not GraphQL."""
    doc = await _fetch_json(
        client,
        url,
        {"Content-Type": "application/json", **headers},
        method="POST",
        json_body={"query": INTROSPECTION_QUERY},
    )
    schema = (doc or {}).get("data", {}).get("__schema") if isinstance(doc, dict) else None
    if not isinstance(schema, dict):
        return None
    operations = graphql_schema_to_operations(
        schema, max_operations=max_operations, include_mutations=include_mutations
    )
    if not operations:
        return None
    return {
        "kind": "graphql",
        "source": f"graphql introspection @ {url}",
        "base_url": url,
        "operations": operations,
    }


async def _try_graphql(
    client: httpx.AsyncClient, base_url: str, headers: dict[str, str]
) -> dict[str, Any] | None:
    for url in (base_url, join_url(base_url, "/graphql")):
        found = await _introspect(client, url, headers)
        if found:
            return found
    return None


async def _discover_schema(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    kind: str,
) -> dict[str, Any] | None:
    """Schema discovered from the base URL alone — GraphQL introspection only.

    HTTP/REST sources get no automatic discovery: their schema location is
    supplied explicitly (schema URL or uploaded file).
    """
    if kind != "graphql":
        return None
    return await _try_graphql(client, base_url, headers)


async def probe_and_discover(
    base_url: str, kind: str = "http", auth: Any = None
) -> dict[str, Any]:
    """Probe *base_url* (with the given auth block); introspect GraphQL schemas.

    Never raises for target-server failures: the outcome is encoded in the
    returned dict — ``url_status`` (ok | unauthorized | unreachable),
    ``auth_status`` (ok | failed | skipped), ``error`` (detail or None) and
    ``discovered`` (schema dict or None; always None for ``kind == "http"``).
    """
    auth_type = getattr(auth, "type", "none")
    result: dict[str, Any] = {
        "url_status": "unreachable",
        "auth_status": "skipped" if auth_type == "none" else "failed",
        "error": None,
        "discovered": None,
    }
    # Resolving our own credentials can fail independently of the target — a
    # `service_identity` block needs a token minted from the backend's key.
    # That is not the target being unreachable, so probe on unauthenticated and
    # report the real cause instead of mislabelling it.
    auth_error: str | None = None
    try:
        headers = await build_auth_headers(auth) if auth is not None else {}
    except Exception as exc:
        headers = {}
        auth_error = f"Could not resolve credentials: {exc}"

    try:
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
                if auth_error is not None:
                    result["error"] = auth_error
                elif auth_type == "none":
                    result["error"] = "Server requires authentication"
                else:
                    result["error"] = (
                        f"Credentials rejected ({response.status_code})"
                    )
                return result

            result["url_status"] = "ok"
            if auth_error is not None:
                result["error"] = auth_error  # auth_status stays "failed"
            elif auth_type != "none":
                result["auth_status"] = "ok"

            result["discovered"] = await _discover_schema(client, base_url, headers, kind)
            return result
    except Exception as exc:  # pragma: no cover — absolute last resort
        logger.exception("datasource probe failed unexpectedly")
        result["error"] = f"Probe failed: {exc}"
        return result


# ---------------------------------------------------------------------------
# Explicit schema fetching
# ---------------------------------------------------------------------------

async def fetch_and_parse_spec(
    schema_url: str,
    kind: str = "http",
    auth: Any = None,
    *,
    max_operations: int = MAX_IMPORTED_OPERATIONS,
) -> dict[str, Any]:
    """Fetch the specification at *schema_url* and map it onto operations.

    For ``kind == "graphql"`` the URL is treated as a GraphQL endpoint first
    (introspection POST) and only then as a document to download, so both
    "here is my /graphql" and "here is my schema.graphql" work.

    Raises :class:`SpecFetchError` when the URL cannot be read and
    :class:`SpecParseError` when the body is not a specification.
    """
    try:
        headers = await build_auth_headers(auth) if auth is not None else {}
    except Exception as exc:
        raise SpecFetchError(f"Could not resolve credentials: {exc}") from exc

    async with httpx.AsyncClient(
        timeout=SCHEMA_FETCH_TIMEOUT_SECONDS, follow_redirects=True
    ) as client:
        if kind == "graphql":
            found = await _introspect(
                client,
                schema_url,
                headers,
                include_mutations=True,
                max_operations=max_operations,
            )
            if found:
                return found

        try:
            response = await client.get(schema_url, headers=headers)
        except Exception as exc:
            raise SpecFetchError(
                f"Could not fetch the schema: {exc.__class__.__name__}: {exc}"
            ) from exc

    if response.status_code >= 400:
        detail = "authentication required" if response.status_code in (401, 403) else "request failed"
        raise SpecFetchError(
            f"Schema URL returned HTTP {response.status_code} ({detail})"
        )
    if len(response.content) > MAX_SPEC_BYTES:
        raise SpecFetchError(
            f"Schema document is larger than {MAX_SPEC_BYTES // (1024 * 1024)} MB"
        )

    return parse_spec(response.content, source=schema_url, max_operations=max_operations)
