"""DAG executor for :class:`DataSourceDefinition` operations.

Calling an operation resolves its dependency closure first: every
``{<operation>.<field>}`` reference in a template is another operation of the
same source, so the closure forms a DAG that is executed level by level with
``asyncio.gather``.  Results are memoised per request, so a diamond-shaped DAG
calls each upstream operation exactly once.

When an operation references a field of an *array* upstream result, the call is
fanned out — once per element — and the result is a list of
``{"<field>": <element value>, "result": <call result>}`` entries.

Values substituted into a URL *path* are percent-encoded and rejected outright
if they contain a traversal segment — see :meth:`DataSourceExecutor._render_path`
for why an unencoded path placeholder makes any per-operation allow-list
meaningless.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any
from urllib.parse import quote

import httpx
import jmespath

from app.domain.models.data_source_definition import (
    REF_PATTERN,
    DataSourceDefinition,
    OperationDefinition,
    operation_refs,
)

logger = logging.getLogger(__name__)

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class DataSourceExecutor:
    """Executes one operation of a data source, resolving its upstream DAG."""

    # Sweep expired cache entries once the cache grows past this size, so a
    # long-lived process doesn't accumulate unbounded stale entries between
    # accesses to any single (source, op, inputs) key.
    _CACHE_SWEEP_THRESHOLD = 500

    def __init__(self, *, fanout_concurrency: int = 5, token_provider: Any = None) -> None:
        self._fanout_concurrency = fanout_concurrency
        # Provides bearer tokens for `service_identity` auth; when None the
        # process-wide provider is resolved lazily in build_auth_headers.
        self._token_provider = token_provider
        # cache key → (expiry monotonic timestamp, value)
        self._cache: dict[str, tuple[float, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        source: DataSourceDefinition,
        operation: str,
        params: dict[str, Any],
    ) -> Any:
        op = source.get_operation(operation)
        if op is None:
            raise ValueError(
                f"Data source '{source.id}' has no operation '{operation}'"
            )
        levels = self._plan(source, operation)
        closure = [
            declared
            for level in levels
            for name in level
            if (declared := source.get_operation(name)) is not None
        ]
        # Coerce before the required check so both see the same values. Applied
        # across the whole closure, not just the target, because `params` is one
        # flat dict shared by every operation in it.
        params = _coerce_params(closure, params)
        # Validate required params for every op in the dependency closure up
        # front — not just the target — so a missing param on an upstream op
        # fails clearly before any HTTP call is made instead of surfacing as
        # a confusing downstream error (or a silently wrong request).
        for op_in_closure in closure:
            self._check_required_params(op_in_closure, params)

        memo: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=source.timeout_seconds) as client:
            for level in levels:
                ops = [source.get_operation(name) for name in level]
                results = await asyncio.gather(
                    *(
                        self._execute_operation(source, o, params, memo, client)
                        for o in ops if o
                    )
                )
                for name, result in zip(level, results):
                    memo[name] = result
        return memo[operation]

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    @staticmethod
    def _dependencies(source: DataSourceDefinition, op: OperationDefinition) -> set[str]:
        return {
            head
            for head, _path in operation_refs(op)
            if head != "params" and source.get_operation(head) is not None
        }

    def _plan(self, source: DataSourceDefinition, target: str) -> list[list[str]]:
        """Return the dependency closure of *target* as topological levels."""
        deps: dict[str, set[str]] = {}
        pending = [target]
        while pending:
            name = pending.pop()
            if name in deps:
                continue
            op = source.get_operation(name)
            if op is None:
                raise ValueError(
                    f"Data source '{source.id}' has no operation '{name}'"
                )
            deps[name] = self._dependencies(source, op)
            pending.extend(deps[name])

        levels: list[list[str]] = []
        done: set[str] = set()
        while len(done) < len(deps):
            level = sorted(
                name for name, d in deps.items()
                if name not in done and d <= done
            )
            if not level:
                remaining = sorted(set(deps) - done)
                raise ValueError(
                    f"Cyclic operation dependencies: {', '.join(remaining)}"
                )
            levels.append(level)
            done.update(level)
        return levels

    # ------------------------------------------------------------------
    # Operation execution (fan-out detection)
    # ------------------------------------------------------------------

    async def _execute_operation(
        self,
        source: DataSourceDefinition,
        op: OperationDefinition,
        params: dict[str, Any],
        memo: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> Any:
        array_refs = {
            (head, path)
            for head, path in operation_refs(op)
            if head != "params" and isinstance(memo.get(head), list)
        }
        array_heads = {head for head, _ in array_refs}
        if len(array_heads) > 1:
            raise ValueError(
                f"Operation '{op.name}' binds more than one array upstream "
                f"({', '.join(sorted(array_heads))}); only one fan-out source "
                f"is supported"
            )
        if not array_heads:
            return await self._invoke(source, op, params, memo, {}, client)

        head = next(iter(array_heads))
        binding_path = sorted(path for h, path in array_refs if h == head)[0]
        binding_name = _leaf_name(binding_path)
        elements = memo[head]
        semaphore = asyncio.Semaphore(self._fanout_concurrency)

        async def _one(element: Any) -> dict[str, Any]:
            async with semaphore:
                result = await self._invoke(source, op, params, memo, {head: element}, client)
            return {
                binding_name: _search(binding_path, element),
                "result": result,
            }

        return list(await asyncio.gather(*(_one(e) for e in elements)))

    # ------------------------------------------------------------------
    # Single operation invocation (cache → pagination → retry → mapping)
    # ------------------------------------------------------------------

    async def _invoke(
        self,
        source: DataSourceDefinition,
        op: OperationDefinition,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
        client: httpx.AsyncClient,
    ) -> Any:
        cache_key = self._cache_key(source, op, params, memo, bound)
        ttl = source.cache.ttl_seconds
        if ttl > 0:
            hit = self._cache.get(cache_key)
            if hit is not None and hit[0] > time.monotonic():
                logger.debug("data source '%s': cache hit for '%s'", source.id, op.name)
                return hit[1]
            if hit is not None:
                # Expired — drop eagerly instead of waiting for a sweep.
                del self._cache[cache_key]

        value = await self._fetch_all_pages(client, source, op, params, memo, bound)

        if ttl > 0:
            self._cache[cache_key] = (time.monotonic() + ttl, value)
            if len(self._cache) > self._CACHE_SWEEP_THRESHOLD:
                self._purge_expired_cache()
        return value

    def _purge_expired_cache(self) -> None:
        """Drop expired entries — called opportunistically so the cache
        doesn't grow unbounded when many distinct cache keys are seen."""
        now = time.monotonic()
        expired = [key for key, (expiry, _) in self._cache.items() if expiry <= now]
        for key in expired:
            del self._cache[key]

    async def _fetch_all_pages(
        self,
        client: httpx.AsyncClient,
        source: DataSourceDefinition,
        op: OperationDefinition,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
    ) -> Any:
        if op.paginate is None:
            raw = await self._request_with_retry(client, source, op, params, memo, bound, {})
            return self._post_process(op, raw)

        paginate = op.paginate
        pages: list[Any] = []
        cursor: Any = None
        page_number = 1
        offset = 0

        for _ in range(max(1, paginate.max_pages)):
            extra: dict[str, Any] = {}
            if paginate.type == "cursor":
                if cursor is not None:
                    extra[paginate.param] = cursor
            elif paginate.type == "page":
                extra[paginate.param] = page_number
            else:  # offset
                extra[paginate.param] = offset

            try:
                raw = await self._request_with_retry(
                    client, source, op, params, memo, bound, extra
                )
            except httpx.HTTPStatusError as exc:
                # Django REST Framework's PageNumberPagination raises NotFound
                # past the last page, so a paginated DRF endpoint never answers
                # with the empty page the stop-check below waits for -- it 404s.
                # Treat that as end-of-pages, but only once a page has already
                # come back: a 404 on the very first request is a wrong path or
                # a deleted resource and must still fail loudly.
                if (
                    paginate.type in ("page", "offset")
                    and exc.response.status_code == 404
                    and pages
                ):
                    logger.debug(
                        "data source '%s' operation '%s': 404 after %d page(s) "
                        "-- treating as end of pages",
                        source.id, op.name, len(pages),
                    )
                    break
                raise
            mapped = self._post_process(op, raw)
            # Without a `mapping`, a dict-shaped page never looks "empty" and
            # page/offset pagination would loop to max_pages, returning raw
            # page dicts. `items_path` extracts the items array explicitly so
            # both the stop-check below and `_combine_pages` see a list.
            if op.mapping is None and paginate.items_path and isinstance(mapped, dict):
                mapped = _search(paginate.items_path, mapped) or []
            pages.append(mapped)

            if paginate.type == "cursor":
                cursor = _search(paginate.cursor_path, raw) if paginate.cursor_path else None
                if not cursor:
                    break
            else:
                if not mapped:
                    break
                page_number += 1
                offset += len(mapped) if isinstance(mapped, list) else 1
        else:
            # The for-loop ran to completion, i.e. max_pages was reached without
            # any stop condition firing. The result is very likely truncated, and
            # a caller that silently believes it has everything is the dangerous
            # case (for an alerting workflow, a short read reads as "nothing to
            # report"). There is no channel to return a warning on, so say it
            # loudly in the log.
            logger.warning(
                "data source '%s' operation '%s': stopped at the max_pages limit "
                "(%d) -- the result is probably incomplete, raise max_pages or "
                "narrow the query",
                source.id, op.name, paginate.max_pages,
            )

        return _combine_pages(pages)

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        source: DataSourceDefinition,
        op: OperationDefinition,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
        extra: dict[str, Any],
    ) -> Any:
        attempts = max(1, source.retries.attempts)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return await self._request_once(client, source, op, params, memo, bound, extra)
            except Exception as exc:  # noqa: BLE001 — retried below, re-raised at the end
                last_error = exc
                if attempt == attempts - 1:
                    break
                delay = source.retries.backoff * (2 ** attempt)
                logger.warning(
                    "data source '%s' operation '%s' attempt %d/%d failed (%s) — "
                    "retrying in %.2fs",
                    source.id, op.name, attempt + 1, attempts, exc, delay,
                )
                await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    async def _request_once(
        self,
        client: httpx.AsyncClient,
        source: DataSourceDefinition,
        op: OperationDefinition,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
        extra: dict[str, Any],
    ) -> Any:
        auth_headers = await build_auth_headers(source.auth, self._token_provider)
        headers = {**source.default_headers, **auth_headers}

        if source.kind == "graphql":
            query = self._render(op.query or "", params, memo, bound)
            variables = self._render(op.variables or {}, params, memo, bound)
            if not isinstance(variables, dict):
                variables = {}
            variables.update(extra)
            resp = await client.post(
                source.base_url,
                json={"query": query, "variables": variables},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

        path = self._render_path(op, params, memo, bound)
        base = source.base_url.rstrip("/")
        url = f"{base}/{path.lstrip('/')}" if path else base
        method = (op.method or "GET").upper()

        # Params already consumed by the path/query template are not repeated
        # as query-string arguments.
        referenced = {
            p.split(".")[0].split("[")[0]
            for head, p in operation_refs(op)
            if head == "params"
        }
        loose = {
            p.name: params[p.name]
            for p in op.params
            if p.name in params and p.name not in referenced and params[p.name] is not None
        }

        if method in ("GET", "DELETE", "HEAD"):
            resp = await client.request(
                method, url, params={**loose, **extra}, headers=headers
            )
        else:
            resp = await client.request(
                method, url, params=extra, json=loose or None, headers=headers
            )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(self, op: OperationDefinition, data: Any) -> Any:
        if op.response_schema:
            _validate_response(op, data, op.response_schema)
        if op.mapping:
            return _search(op.mapping, data)
        return data

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _render_path(
        self,
        op: OperationDefinition,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
    ) -> str:
        """Render ``op.path`` with every substituted value URL-path-encoded.

        Separate from :meth:`_render` because a path is not just another
        template: it is spliced into the request URL, so a value carrying ``/``,
        ``?``, ``#`` or a ``..`` segment does not fill in a placeholder — it
        retargets the request.  Given ``/projects/{params.id}``, an ``id`` of
        ``1/../../admin/users?role=all#`` reaches ``/admin/users?role=all``
        instead, on the same host, carrying the same upstream credential.  That
        makes any per-operation allow-list decorative: a caller granted one
        read of one resource can reach anything the credential can.

        So each value is percent-encoded with nothing left safe (``/`` and
        ``?`` included) and traversal segments are refused rather than encoded,
        because ``%2e%2e`` is only safe until something normalises it.  Literal
        text in the template — including its ``/`` separators — is untouched.

        Unlike :meth:`_render`, there is no whole-value shortcut that preserves
        the native type: a path is a string by the time it reaches the URL, and
        an operation whose entire path is caller-supplied is precisely the case
        that must not skip encoding.
        """
        template = op.path or ""
        if not template:
            return ""

        def _sub(match: "Any") -> str:
            resolved = self._resolve_ref(
                match.group(1), match.group(2), params, memo, bound
            )
            return _encode_path_value(
                resolved, op_name=op.name, placeholder=match.group(0)
            )

        return REF_PATTERN.sub(_sub, template)

    def _render(
        self,
        value: Any,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
    ) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            match = REF_PATTERN.fullmatch(stripped)
            if match:
                # Whole value is a single placeholder — keep the native type.
                return self._resolve_ref(match.group(1), match.group(2), params, memo, bound)

            def _sub(m: "Any") -> str:
                resolved = self._resolve_ref(m.group(1), m.group(2), params, memo, bound)
                return "" if resolved is None else str(resolved)

            return REF_PATTERN.sub(_sub, value)
        if isinstance(value, dict):
            return {k: self._render(v, params, memo, bound) for k, v in value.items()}
        if isinstance(value, list):
            return [self._render(v, params, memo, bound) for v in value]
        return value

    @staticmethod
    def _resolve_ref(
        head: str,
        path: str,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
    ) -> Any:
        if head == "params":
            return _search(path, params)
        if head in bound:
            return _search(path, bound[head])
        if head in memo:
            return _search(path, memo[head])
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_required_params(op: OperationDefinition, params: dict[str, Any]) -> None:
        missing = [
            p.name for p in op.params
            if p.required and params.get(p.name) is None
        ]
        if missing:
            raise ValueError(
                f"Operation '{op.name}' is missing required param(s): "
                f"{', '.join(missing)}"
            )

    @classmethod
    def _cache_key(
        cls,
        source: DataSourceDefinition,
        op: OperationDefinition,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
    ) -> str:
        resolved_params = {
            p.name: params.get(p.name) for p in op.params if p.name in params
        }
        if source.cache.key_template:
            rendered = source.cache.key_template
            for name, value in resolved_params.items():
                rendered = rendered.replace(f"{{params.{name}}}", str(value))
            return f"{source.id}|{op.name}|{rendered}"
        # Include the resolved value of every ref the operation's templates
        # bind — params AND upstream/fan-out results — so an operation that
        # depends on upstream data gets a distinct cache key per upstream
        # value instead of returning another call's stale, cross-contaminated
        # result.
        resolved_refs = {
            (f"{head}.{path}" if path else head): cls._resolve_ref(
                head, path, params, memo, bound
            )
            for head, path in operation_refs(op)
        }
        payload = {"params": resolved_params, "refs": resolved_refs, "bound": bound}
        return f"{source.id}|{op.name}|{_canonical_json(payload)}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Splits a value into path segments the way both httpx and any upstream server
# will: forward slashes, and backslashes too, since some servers normalise them.
_PATH_SEPARATOR_RE = re.compile(r"[/\\]")

# Segments that mean "somewhere else" rather than "a name".
_TRAVERSAL_SEGMENTS = frozenset({".", ".."})


def _encode_path_value(value: Any, *, op_name: str, placeholder: str) -> str:
    """Percent-encode *value* for use as URL path text, or raise.

    ``quote(..., safe="")`` leaves nothing structural intact, so a value can
    contribute only a single path segment's worth of characters — ``/`` becomes
    ``%2F``, ``?`` becomes ``%3F``, ``#`` becomes ``%23``.

    Encoding alone is not enough, though: ``quote("..")`` is ``".."``, so a
    value of exactly ``..`` would still climb a level of a template like
    ``/projects/{params.id}/files``.  Traversal segments are therefore rejected
    before encoding, with the operation named so the error points at the
    definition rather than at the executor.
    """
    if value is None:
        return ""
    text = str(value)
    for segment in _PATH_SEPARATOR_RE.split(text):
        if segment.strip() in _TRAVERSAL_SEGMENTS:
            raise ValueError(
                f"Operation '{op_name}': path placeholder '{placeholder}' "
                f"resolved to {text!r}, which contains a path-traversal "
                f"segment; a path parameter may not escape its operation"
            )
    return quote(text, safe="")


_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


def _coerce_params(
    ops: list[OperationDefinition], params: dict[str, Any]
) -> dict[str, Any]:
    """Coerce caller values to the types their ``ParamSpec`` declares.

    ``ParamSpec.type`` was advisory: an MCP caller got type checking for free
    from the synthesised tool signature, but the ``data_source`` workflow step
    passes rendered state straight through, so ``"5"`` reached a ``number`` param
    and a nonsense value reached it just as easily.  Coercing here means all
    three callers (MCP, workflow step, try-operation) inherit the same
    behaviour, and a value that cannot be the declared type fails with the
    param named instead of producing a silently wrong request.

    Only ``number`` and ``boolean`` are touched — the two with an unambiguous
    parse from a string.  ``string`` / ``array`` / ``object`` are left alone:
    stringifying whatever arrived would hide mistakes rather than surface them,
    and no undeclared key is dropped here because the request builder already
    allow-lists body and query params to declared names.
    """
    coerced = dict(params)
    for spec in (spec for op in ops for spec in op.params):
        if spec.name not in coerced or coerced[spec.name] is None:
            continue
        if spec.type == "number":
            coerced[spec.name] = _as_number(spec.name, coerced[spec.name])
        elif spec.type == "boolean":
            coerced[spec.name] = _as_boolean(spec.name, coerced[spec.name])
    return coerced


def _as_number(name: str, value: Any) -> int | float:
    # bool is an int in Python; letting True through as 1 would silently accept
    # a checkbox where a count was declared.
    if isinstance(value, bool):
        raise ValueError(f"Param '{name}' is declared number but got a boolean")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
    raise ValueError(
        f"Param '{name}' is declared number but got {value!r}, which is not one"
    )


def _as_boolean(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    raise ValueError(
        f"Param '{name}' is declared boolean but got {value!r}, which is not one"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _search(expression: str | None, data: Any) -> Any:
    """JMESPath lookup that never raises on malformed data."""
    if not expression:
        return data
    try:
        return jmespath.search(expression, data)
    except Exception:  # noqa: BLE001 — a bad expression yields no value
        return None


def _leaf_name(path: str) -> str:
    leaf = path.split(".")[-1]
    return leaf.split("[")[0] or path


def _combine_pages(pages: list[Any]) -> Any:
    if len(pages) == 1:
        return pages[0]
    if pages and all(isinstance(p, list) for p in pages):
        combined: list[Any] = []
        for page in pages:
            combined.extend(page)
        return combined
    return pages


async def build_auth_headers(auth: Any, token_provider: Any = None) -> dict[str, str]:
    """Resolve the auth block into request headers using the stored secrets.

    Shared by the executor and the probe/discovery endpoint
    (``app.infrastructure.datasources.discovery``).

    ``service_identity`` carries no stored secret: the bearer token is minted
    at request time by *token_provider* (the process-wide provider is used when
    none is injected), using the identity the auth block names — or the
    deployment's default one when it names none.
    """
    kind = getattr(auth, "type", "none")
    if kind == "service_identity":
        provider = token_provider
        if provider is None:
            from app.infrastructure.auth.service_token_provider import (
                get_service_token_provider,
            )
            provider = get_service_token_provider()
        return await provider.get_auth_header(getattr(auth, "identity", None))
    if kind == "bearer":
        return {"Authorization": f"Bearer {auth.token}"}
    if kind == "basic":
        encoded = base64.b64encode(f"{auth.username}:{auth.password}".encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    if kind == "header":
        return {auth.header_name: auth.value}
    return {}


def _validate_response(op: OperationDefinition, data: Any, schema: dict[str, Any]) -> None:
    """Minimal JSON-schema-ish check: top-level type, required keys, prop types."""
    expected = schema.get("type")
    if expected and expected in _JSON_TYPES and not isinstance(data, _JSON_TYPES[expected]):
        raise ValueError(
            f"Operation '{op.name}' response is {type(data).__name__}, "
            f"expected {expected}"
        )
    if not isinstance(data, dict):
        return
    for key in schema.get("required", []) or []:
        if key not in data:
            raise ValueError(
                f"Operation '{op.name}' response is missing required key '{key}'"
            )
    for key, spec in (schema.get("properties") or {}).items():
        if key not in data or not isinstance(spec, dict):
            continue
        prop_type = spec.get("type")
        if (
            prop_type in _JSON_TYPES
            and data[key] is not None
            and not isinstance(data[key], _JSON_TYPES[prop_type])
        ):
            raise ValueError(
                f"Operation '{op.name}' response key '{key}' is "
                f"{type(data[key]).__name__}, expected {prop_type}"
            )
