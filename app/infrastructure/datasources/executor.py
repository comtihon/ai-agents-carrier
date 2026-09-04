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
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
import jmespath

from app.domain.models.data_source_definition import (
    REF_PATTERN,
    DataSourceDefinition,
    OperationDefinition,
    Paginate,
    operation_refs,
)
from app.domain.models.datastream import as_data_ref, is_data_ref
from app.infrastructure.datasources.datastream import NotStreamable, StreamBuilder

logger = logging.getLogger(__name__)

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


# How many rendered request lines / sample values a preview carries. An
# approval message that lists ten thousand URLs is unreadable, and the number
# that matters (``affected_rows``) is exact regardless of the cap.
_PREVIEW_TARGET_CAP = 20
_PREVIEW_SAMPLE_CAP = 20


@dataclass
class DestructivePlan:
    """What a destructive operation is about to do, before it does it."""

    affected_rows: int
    # Rendered "METHOD url" lines, capped at ``_PREVIEW_TARGET_CAP``.
    targets: list[str] = field(default_factory=list)
    # The values bound from the upstream array (ids, names) — what an approver
    # reads to recognise what is being removed. Capped the same way.
    sample: list[Any] = field(default_factory=list)
    # What kind of change this is, which decides the words an approver reads
    # and the question the meta-LLM is asked. "delete" is the default because
    # removing rows is what the gate was built for; a sheet binding writes
    # cells instead, and calling that a deletion of N rows describes neither
    # the blast radius nor the risk.
    change_kind: str = "delete"
    # Free-form label -> value pairs shown beside the operation and included
    # in the meta-LLM prompt. For a sheet write this is the document, the tab
    # and — the part that changes an answer — whether a model wrote the
    # computation behind it.
    details: dict[str, str] = field(default_factory=dict)
    # WHICH rows are touched, said without reference to their contents
    # ("row 7", "a new row"). The Slack approval message uses this instead of
    # `sample`, so a channel never carries data values; empty when the
    # operation has no row identity (a fan-out delete over API resources),
    # where the count is the whole story anyway.
    rows_label: str = ""


def _binding_details(binding: Any) -> dict[str, str]:
    """Context an approver needs about the binding behind a sheet write.

    The operation name alone does not say which spreadsheet is about to change,
    nor the thing that most affects whether a person says yes: whether the
    values were computed by code a language model wrote. Both are stated here
    rather than left for the approver to go and look up.
    """
    document = getattr(binding, "document", None)
    details: dict[str, str] = {}
    if document is not None:
        name = getattr(document, "name", "") or getattr(document, "file_id", "")
        if name:
            details["Document"] = name
        tab = getattr(document, "sheet", "")
        if tab:
            details["Tab"] = tab
    details["Binding"] = getattr(binding, "name", "") or "—"

    compute = getattr(binding, "compute", None)
    if compute is None:
        details["Values from"] = "a hand-authored column mapping"
    else:
        resolution = getattr(binding, "resolution", None)
        model = getattr(resolution, "model_id", "") or "an unrecorded model"
        edited = bool(getattr(resolution, "edited_by_human", False))
        details["Values from"] = (
            f"generated code, since edited by a person (originally {model})"
            if edited
            else f"generated code written by {model}"
        )
    return details


def _binding_for(source: DataSourceDefinition, operation: str) -> Any:
    """The sheet binding *operation* was compiled from, or ``None``.

    A name comparison against ``source.bindings``, kept out of line so both
    ``execute`` and ``preview`` ask the question the same way. ``getattr`` so a
    definition-shaped object from an older document (or a test double) without
    the field is simply "no bindings".
    """
    for binding in getattr(source, "bindings", None) or []:
        if getattr(binding, "name", None) == operation:
            return binding
    return None


class DataSourceExecutor:
    """Executes one operation of a data source, resolving its upstream DAG."""

    # Sweep expired cache entries once the cache grows past this size, so a
    # long-lived process doesn't accumulate unbounded stale entries between
    # accesses to any single (source, op, inputs) key.
    _CACHE_SWEEP_THRESHOLD = 500

    # Ceiling on reading an intermediate document back for template
    # resolution. A `{op.field}` value ends up in a URL or a GraphQL
    # variable, so anything approaching this is a definition bug.
    _INTERMEDIATE_MAX_BYTES = 8 * 1024 * 1024

    def __init__(
        self,
        *,
        fanout_concurrency: int = 5,
        token_provider: Any = None,
        stream_store: "Any" = None,
    ) -> None:
        self._fanout_concurrency = fanout_concurrency
        # Provides bearer tokens for `service_identity` auth; when None the
        # process-wide provider is resolved lazily in build_auth_headers.
        self._token_provider = token_provider
        # Where every result is written (see
        # app.infrastructure.datasources.datastream).  Required: results do not
        # travel as values, so without a store there is nowhere for one to go
        # and `execute` refuses rather than quietly reverting to holding it all
        # in memory.
        self._stream_store = stream_store
        # cache key → (expiry monotonic timestamp, value)
        self._cache: dict[str, tuple[float, Any]] = {}

    @property
    def stream_store(self) -> "Any":
        return self._stream_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        source: DataSourceDefinition,
        operation: str,
        params: dict[str, Any],
        *,
        limit: int | None = None,
    ) -> Any:
        """Run one operation and return a reference to its result.

        ``limit`` caps how many records are collected. ``None`` -- the default
        -- means everything the source has: a paginated operation is then
        walked page by page until the API says there is nothing left, which is
        the whole point of not stating a limit. Pagination is the data
        source's business, not the caller's; a caller says how much it wants,
        never which page.
        """
        binding = _binding_for(source, operation)
        if binding is not None:
            # A declarative sheet binding compiled to this operation name. Its
            # HTTP work is several Sheets calls with pure transforms between
            # them, so it is served here rather than by rendering one request:
            # the delegate calls back into ``execute`` for each raw operation
            # it needs. Intercepting at the top of ``execute`` is what makes a
            # binding reachable from every caller at once -- workflow step, MCP
            # tool, /try-operation -- with no new step type anywhere.
            from app.infrastructure.datasources.sheet_binding_runtime import run_binding
            return await run_binding(source, self, binding, params)
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
                        self._execute_operation(source, o, params, memo, client, limit=limit)
                        for o in ops if o
                    )
                )
                for name, result in zip(level, results):
                    memo[name] = result
                    if name != operation:
                        # An intermediate result is read back into the memo
                        # when it is a document, because that is what a
                        # `{op.field}` template resolves against. A *list*
                        # upstream stays a reference: it is a fan-out source,
                        # and streaming it is the whole point. So the DAG
                        # keeps working without any intermediate list ever
                        # being resident.
                        memo[name] = await self._resolve_intermediate(
                            source, name, result
                        )
        return memo[operation]

    async def _resolve_intermediate(
        self, source: DataSourceDefinition, name: str, result: Any
    ) -> Any:
        """Read a document-shaped intermediate back; leave a list a reference.

        Bounded by ``_INTERMEDIATE_MAX_BYTES`` rather than by the source's own
        ceiling: this value is about to be interpolated into a URL or a query,
        so a document measured in megabytes is a mistake in the definition,
        not a large read to be accommodated.
        """
        ref = as_data_ref(result)
        if ref is None or ref.shape != "value":
            return result
        if self._stream_store is None:
            return result
        return await self._stream_store.read_all(
            ref, max_bytes=self._INTERMEDIATE_MAX_BYTES
        )

    async def execute_value(
        self,
        source: DataSourceDefinition,
        operation: str,
        params: dict[str, Any],
        *,
        max_bytes: int = 0,
    ) -> Any:
        """Execute an operation and return its records, not a reference.

        The deliberate opt-out of the stream contract, for the callers that
        genuinely need a value in hand: a ``data_source`` step configured
        ``result_mode: ram``, ``/try-operation``'s dry run, a test.

        It refuses past *max_bytes* rather than degrading -- a silent partial
        read is the failure the stream contract exists to remove -- and 0 means
        no limit, which is only safe where the caller already knows the size.
        """
        ref_state = await self.execute(source, operation, params)
        ref = as_data_ref(ref_state)
        if ref is None:
            # A sheet binding serves itself and may return a plain value.
            return ref_state
        if self._stream_store is None:
            raise ValueError(
                f"data source '{source.id}': no data stream store to read "
                f"stream '{ref.id}' back from"
            )
        return await self._stream_store.read_all(ref, max_bytes=max_bytes)

    async def preview(
        self,
        source: DataSourceDefinition,
        operation: str,
        params: dict[str, Any],
    ) -> "DestructivePlan":
        """Resolve everything *around* a call without making the call itself.

        Used before a destructive operation runs, to answer the one question an
        approver actually needs: how many rows is this about to remove, and
        which ones. The dependency closure is executed for real — those are the
        reads that name the targets — and the target operation alone is held
        back, its request URL rendered but never sent.

        The upstream reads run a second time when the approved call finally
        goes out. That is deliberate: re-reading is how an approval that sat in
        Slack for an hour does not act on an hour-old list. Set a
        ``cache.ttl_seconds`` on the source when the extra read is not wanted.

        A compiled sheet binding answers this itself: what an approver needs to
        see there is not a row count but the cells about to change, so the
        binding runtime plans the write (read, resolve the row, check the
        header fingerprint, build) and hands back the before/after of every
        cell as the plan's ``sample``.
        """
        binding = _binding_for(source, operation)
        if binding is not None and binding.operation == "write":
            from app.infrastructure.datasources.sheet_binding_runtime import (
                binding_destructive_plan,
            )
            rows, targets, sample, rows_label = await binding_destructive_plan(
                source, self, binding, params
            )
            return DestructivePlan(
                affected_rows=rows,
                targets=targets,
                sample=sample,
                rows_label=rows_label,
                change_kind="write",
                details=_binding_details(binding),
            )
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
        params = _coerce_params(closure, params)
        for op_in_closure in closure:
            self._check_required_params(op_in_closure, params)

        memo: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=source.timeout_seconds) as client:
            for level in levels:
                ops = [
                    o for name in level
                    if name != operation and (o := source.get_operation(name)) is not None
                ]
                if not ops:
                    continue
                results = await asyncio.gather(
                    *(self._execute_operation(source, o, params, memo, client) for o in ops)
                )
                for o, result in zip(ops, results):
                    memo[o.name] = result

        # Same fan-out rule the real invocation follows: one array upstream
        # binds one request per element, and that element count *is* the number
        # of affected rows.
        # A spilled upstream counts as an array here for the same reason it
        # does in _execute_operation, and the stakes are higher: this number is
        # what an approver is shown. Treating a handle as "not an array" would
        # present a delete fanning out over 400,000 spilled rows as
        # "1 affected row", which is the worst possible thing for this gate to
        # get wrong.
        array_refs = {
            (head, path)
            for head, path in operation_refs(op)
            if head != "params"
            and (isinstance(memo.get(head), list) or is_data_ref(memo.get(head)))
        }
        array_heads = {head for head, _ in array_refs}
        if len(array_heads) > 1:
            raise ValueError(
                f"Operation '{op.name}' binds more than one array upstream "
                f"({', '.join(sorted(array_heads))}); only one fan-out source "
                f"is supported"
            )

        if not array_heads:
            return DestructivePlan(
                affected_rows=1,
                targets=[self._describe_target(source, op, params, memo, {})],
                sample=[params] if params else [],
            )

        head = next(iter(array_heads))
        binding_path = sorted(path for h, path in array_refs if h == head)[0]
        upstream = memo[head]
        handle = as_data_ref(upstream)
        if handle is None:
            elements = upstream
            affected = len(elements)
        else:
            # The count is exact from the handle; only the capped preview is
            # read off disk, so previewing a delete over a spilled upstream
            # costs one short read rather than loading what was spilled
            # precisely because it did not fit.
            affected = handle.items
            if self._stream_store is None:
                raise NotStreamable(
                    f"Operation '{op.name}' fans out over a spilled upstream "
                    f"'{head}' but no spill store is configured to preview it."
                )
            elements = [
                item
                async for item in self._stream_store.stream(
                    handle, limit=max(_PREVIEW_TARGET_CAP, _PREVIEW_SAMPLE_CAP)
                )
            ]
        targets = [
            self._describe_target(source, op, params, memo, {head: element})
            for element in elements[:_PREVIEW_TARGET_CAP]
        ]
        sample = [_search(binding_path, element) for element in elements[:_PREVIEW_SAMPLE_CAP]]
        return DestructivePlan(
            affected_rows=affected,
            targets=targets,
            sample=sample,
        )

    def _describe_target(
        self,
        source: DataSourceDefinition,
        op: OperationDefinition,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
    ) -> str:
        """The request line a call would produce, rendered but not sent."""
        method = (op.method or "GET").upper()
        if source.kind == "graphql":
            return f"{method} {source.base_url} ({op.name})"
        path = self._render_path(op, params, memo, bound)
        base = source.base_url.rstrip("/")
        url = f"{base}/{path.lstrip('/')}" if path else base
        return f"{method} {url}"

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
        *,
        limit: int | None = None,
    ) -> Any:
        # An upstream that spilled is still a fan-out source: it is a list of
        # items, it is just not resident. Treat a handle exactly like a list
        # here so an operation downstream of a large read keeps working --
        # without this it would look like "no array upstream", run once, and
        # render the handle dict into the URL.
        array_refs = {
            (head, path)
            for head, path in operation_refs(op)
            if head != "params"
            and (isinstance(memo.get(head), list) or is_data_ref(memo.get(head)))
        }
        array_heads = {head for head, _ in array_refs}
        if len(array_heads) > 1:
            raise ValueError(
                f"Operation '{op.name}' binds more than one array upstream "
                f"({', '.join(sorted(array_heads))}); only one fan-out source "
                f"is supported"
            )
        if not array_heads:
            return await self._invoke(source, op, params, memo, {}, client, limit=limit)

        head = next(iter(array_heads))
        binding_path = sorted(path for h, path in array_refs if h == head)[0]
        binding_name = _leaf_name(binding_path)
        upstream = memo[head]
        semaphore = asyncio.Semaphore(self._fanout_concurrency)

        async def _one(element: Any) -> dict[str, Any]:
            async with semaphore:
                result = await self._invoke(source, op, params, memo, {head: element}, client)
            return {
                binding_name: _search(binding_path, element),
                # One element's response is a document -- one record fetched by
                # id, one write acknowledged -- so it is read back and put in
                # the entry directly. A reference per element would be a file
                # per element, which is neither useful nor cheap; the
                # *aggregate* is what gets streamed.
                "result": await self._resolve_intermediate(source, op.name, result),
            }

        handle = as_data_ref(upstream)
        if handle is None:
            return list(await asyncio.gather(*(_one(e) for e in upstream)))
        return await self._fanout_over_spill(
            source, op, handle, _one, head, binding_name,
        )

    async def _fanout_over_spill(
        self,
        source: DataSourceDefinition,
        op: OperationDefinition,
        handle: Any,
        one: Any,
        head: str,
        binding_name: str,
    ) -> Any:
        """Fan out over a spilled upstream, a window of elements at a time.

        The in-memory path gathers every call at once, which is fine for a list
        that already fit in memory. Here the upstream deliberately did not, so
        the elements are pulled off disk in windows and only a window's worth
        of calls is ever in flight -- and the *results* go through their own
        budget, because N requests over a large upstream produce a large result
        just as surely as one big response does.
        """
        store = self._stream_store
        if store is None:
            raise NotStreamable(
                f"Operation '{op.name}' fans out over a spilled upstream "
                f"'{head}' but no spill store is configured to read it back."
            )
        builder = StreamBuilder(
            store=store,
            max_result_bytes=source.max_result_bytes,
            source_id=source.id,
            operation=op.name,
        )
        # One window in flight, sized by the fan-out concurrency: enough to
        # keep every slot busy, small enough that the window itself is never
        # the memory problem.
        window = max(1, self._fanout_concurrency) * 4
        done = 0
        try:
            async for chunk in store.chunks(handle, size=window):
                results = await asyncio.gather(*(one(e) for e in chunk))
                await builder.add(list(results))
                done += len(chunk)
                if builder.full:
                    logger.warning(
                        "data source '%s' operation '%s': fan-out stopped at "
                        "max_result_bytes after %d of %d element(s)",
                        source.id, op.name, done, handle.items,
                    )
                    break
            return await builder.finish()
        except BaseException:
            await builder.abort()
            raise

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
        *,
        limit: int | None = None,
    ) -> Any:
        # The limit is part of the key: a 10-row read and a whole-source read
        # are different results, and serving one for the other would be silent.
        cache_key = self._cache_key(source, op, params, memo, bound, limit=limit)
        ttl = source.cache.ttl_seconds
        if ttl > 0:
            hit = self._cache.get(cache_key)
            if hit is not None:
                expiry, cached = hit
                if expiry <= time.monotonic():
                    # Expired — drop eagerly instead of waiting for a sweep.
                    self._cache.pop(cache_key, None)
                elif await self._stream_is_readable(cached):
                    logger.debug(
                        "data source '%s': cache hit for '%s'", source.id, op.name
                    )
                    return cached
                else:
                    # A live entry is not proof the stream is still readable:
                    # files are swept on a TTL and lost on a restart. Treat a
                    # vanished one as a miss rather than handing back a
                    # reference to nothing.
                    logger.info(
                        "data source '%s': cached stream for '%s' is gone -- "
                        "refetching", source.id, op.name,
                    )
                    self._cache.pop(cache_key, None)

        value = await self._fetch_all_pages(
            client, source, op, params, memo, bound, limit=limit
        )

        if ttl > 0:
            self._cache[cache_key] = (time.monotonic() + ttl, value)
            if len(self._cache) > self._CACHE_SWEEP_THRESHOLD:
                self._purge_expired_cache()
        return value

    async def _stream_is_readable(self, value: Any) -> bool:
        """True when *value* is not a reference, or is one whose file is there."""
        ref = as_data_ref(value)
        if ref is None or self._stream_store is None:
            return True
        try:
            await self._stream_store.local_path(ref)
        except FileNotFoundError:
            return False
        except Exception:  # noqa: BLE001 — a store with no local path is fine
            return True
        return True

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
        *,
        limit: int | None = None,
    ) -> Any:
        """Fetch an operation into the data stream store; return its ref.

        Always a ref, never the data.  Pages go to the file as they arrive and
        are dropped, so this holds one page regardless of how many there are.
        """
        if self._stream_store is None:
            raise ValueError(
                f"data source '{source.id}' operation '{op.name}': no data "
                f"stream store is configured. Results are written to a stream "
                f"and passed on as a reference, so one is required."
            )
        builder = StreamBuilder(
            store=self._stream_store,
            max_result_bytes=source.max_result_bytes,
            source_id=source.id,
            operation=op.name,
            limit=limit,
        )
        try:
            if op.paginate is None:
                raw = await self._request_with_retry(
                    client, source, op, params, memo, bound, {}
                )
                await builder.add(self._post_process(op, raw))
            else:
                await self._page_loop(
                    client, source, op, params, memo, bound, op.paginate, builder
                )
            return await builder.finish()
        except BaseException:
            # A half-written stream is not a result. Drop it rather than leave
            # an unreferenced file on a disk the pod has a quota on.
            await builder.abort()
            raise

    async def _page_loop(
        self,
        client: httpx.AsyncClient,
        source: DataSourceDefinition,
        op: OperationDefinition,
        params: dict[str, Any],
        memo: dict[str, Any],
        bound: dict[str, Any],
        paginate: Paginate,
        builder: StreamBuilder,
    ) -> None:
        """Walk the pages of one operation, feeding each into *builder*.

        Pages are handed over one at a time and never retained here, so this
        loop's memory is one page regardless of how many there are.
        """
        cursor: Any = None
        page_number = 1
        offset = 0
        pages = 0

        # max_pages 0 means "no ceiling": walk until the API says it is done.
        # That is what a caller with no row limit is asking for, and
        # max_result_bytes still bounds the total.
        unlimited_pages = paginate.max_pages <= 0
        page_cap = 1 if unlimited_pages else max(1, paginate.max_pages)

        page = 0
        while page < page_cap:
            page += 1
            if unlimited_pages:
                page_cap = page + 1  # keep going; the stop conditions decide
            extra: dict[str, Any] = {}
            if paginate.type == "cursor":
                if cursor is not None:
                    extra[paginate.param] = cursor
            elif paginate.type == "page":
                extra[paginate.param] = page_number
            else:  # offset
                extra[paginate.param] = offset
            if paginate.size_param:
                # Ask for only as many as are still wanted, so a limit of 10
                # against a 100-row page size costs one small page rather
                # than a big one that is then thrown away.
                size = max(1, paginate.page_size)
                remaining = builder.remaining
                if remaining is not None:
                    size = max(1, min(size, remaining))
                extra[paginate.size_param] = size

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
                        source.id, op.name, pages,
                    )
                    return
                raise

            mapped = self._post_process(op, raw)
            # Without a `mapping`, a dict-shaped page never looks "empty" and
            # page/offset pagination would loop to max_pages, returning raw
            # page dicts. `items_path` extracts the items array explicitly so
            # both the stop-check below and the builder see a list.
            if op.mapping is None and paginate.items_path and isinstance(mapped, dict):
                mapped = _search(paginate.items_path, mapped) or []

            # The API's own total, read once from the first page. Where the
            # data goes no longer depends on it -- it always goes to the
            # stream -- but it puts the finished size in the log before the
            # walk, and reports a read that will breach max_result_bytes at
            # page one rather than at page forty.
            if pages == 0 and paginate.total_path:
                builder.project(_search(paginate.total_path, raw), mapped)

            await builder.add(mapped)
            pages += 1

            if builder.full:
                # Either the caller's row limit is satisfied -- nothing more is
                # wanted -- or max_result_bytes was reached and the result is
                # already flagged truncated. Either way there is no reason to
                # ask for another page.
                if builder.limit_reached:
                    logger.debug(
                        "data source '%s' operation '%s': row limit of %d "
                        "reached after %d page(s)",
                        source.id, op.name, builder.items_written, pages,
                    )
                return

            if paginate.type == "cursor":
                cursor = _search(paginate.cursor_path, raw) if paginate.cursor_path else None
                if not cursor:
                    return
            else:
                if not mapped:
                    return
                page_number += 1
                offset += len(mapped) if isinstance(mapped, list) else 1

        # Fell out of the loop with no stop condition having fired, i.e. the
        # max_pages ceiling was reached. The result is very likely incomplete,
        # and a caller that silently believes it has everything is the
        # dangerous case (for an alerting workflow a short read reads as
        # "nothing to report"). Say it loudly in the log and record it on the
        # ref, which is a channel the log is not.
        if not unlimited_pages:
            builder.mark_truncated()
            logger.warning(
                "data source '%s' operation '%s': stopped at the max_pages limit "
                "(%d) -- the result is probably incomplete, raise max_pages "
                "(0 = no ceiling) or narrow the query",
                source.id, op.name, paginate.max_pages,
            )

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
        # An operation may override the source's policy — an append is not
        # idempotent, so a retry after a timeout that in fact succeeded would
        # write the rows twice.
        policy = op.retries or source.retries
        attempts = max(1, policy.attempts)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return await self._request_once(client, source, op, params, memo, bound, extra)
            except Exception as exc:  # noqa: BLE001 — retried below, re-raised at the end
                last_error = exc
                if attempt == attempts - 1:
                    break
                delay = policy.backoff * (2 ** attempt)
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
            for path, value in extra.items():
                _set_path(variables, path, value)
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

        # Declared query-string arguments apply to every method: an API may
        # take a control argument in the query string of a POST/PUT, where a
        # loose param would land in the JSON body instead.  ``extra`` (the
        # pagination cursor) still wins, and so does a loose param of the same
        # name on a method that carries them in the query string.
        query = {
            key: value
            for key, value in (self._render(op.query_params or {}, params, memo, bound)).items()
            if value is not None and value != ""
        }

        if method in ("GET", "DELETE", "HEAD"):
            resp = await client.request(
                method, url, params={**query, **loose, **extra}, headers=headers
            )
        else:
            resp = await client.request(
                method, url, params={**query, **extra}, json=loose or None, headers=headers
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
        limit: int | None = None,
    ) -> str:
        resolved_params = {
            p.name: params.get(p.name) for p in op.params if p.name in params
        }
        if source.cache.key_template:
            rendered = source.cache.key_template
            for name, value in resolved_params.items():
                rendered = rendered.replace(f"{{params.{name}}}", str(value))
            return f"{source.id}|{op.name}|{rendered}|limit={limit}"
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
        payload = {
            "params": resolved_params,
            "refs": resolved_refs,
            "bound": bound,
            "limit": limit,
        }
        return f"{source.id}|{op.name}|{_canonical_json(payload)}"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Splits a value into path segments the way both httpx and any upstream server
# will: forward slashes, and backslashes too, since some servers normalise them.
_PATH_SEPARATOR_RE = re.compile(r"[/\\]")

# Segments that mean "somewhere else" rather than "a name".
_TRAVERSAL_SEGMENTS = frozenset({".", ".."})


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    """Set ``a.b.c`` inside nested dicts, creating the levels on the way.

    Pagination for a GraphQL source has to reach a field of an input object:
    control-center's ``pagination`` is ``PaginationInput{limit, skip}``, so the
    offset belongs at ``pagination.skip``, not at a top-level variable of that
    name. A path with no dots behaves exactly as a plain key assignment did.

    An intermediate level that exists but is not a dict is replaced rather than
    merged into: the paginator owns these fields, and a scalar sitting where an
    object belongs is a definition mistake, not data worth preserving.
    """
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


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
    """Fill declared defaults, then coerce values to their declared types.

    ``ParamSpec.type`` was advisory: an MCP caller got type checking for free
    from the synthesised tool signature, but the ``data_source`` workflow step
    passes rendered state straight through, so ``"5"`` reached a ``number`` param
    and a nonsense value reached it just as easily.  Coercing here means all
    three callers (MCP, workflow step, try-operation) inherit the same
    behaviour, and a value that cannot be the declared type fails with the
    param named instead of producing a silently wrong request.

    A declared ``default`` is filled in first, so it reaches the type coercion
    and the required check like any caller-supplied value.

    Only ``number`` and ``boolean`` are touched — the two with an unambiguous
    parse from a string.  ``string`` / ``array`` / ``object`` are left alone:
    stringifying whatever arrived would hide mistakes rather than surface them,
    and no undeclared key is dropped here because the request builder already
    allow-lists body and query params to declared names.
    """
    coerced = dict(params)
    for spec in (spec for op in ops for spec in op.params):
        if coerced.get(spec.name) is None and spec.default is not None:
            # Filled before the required check, so a declared default also
            # satisfies a required param the caller left out.
            coerced[spec.name] = spec.default
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


async def build_auth_headers(auth: Any, token_provider: Any = None) -> dict[str, str]:
    """Resolve the auth block into request headers using the stored secrets.

    Shared by the executor and the probe/discovery endpoint
    (``app.infrastructure.datasources.discovery``).

    ``service_identity`` carries no stored secret: the bearer token is minted
    at request time by *token_provider* (the process-wide provider is used when
    none is injected), using the identity the auth block names — or the
    deployment's default one when it names none.

    ``google`` carries no stored secret either: the token is minted by
    impersonating the configured service account, with a module-level cache in
    the provider so a ~1h token is not re-minted on every request.  The target
    principal comes from settings, never from the auth block — see
    ``app.infrastructure.auth.google_token_provider``.
    """
    kind = getattr(auth, "type", "none")
    if kind == "google":
        from app.infrastructure.auth.google_token_provider import (
            get_google_auth_header,
        )
        return await get_google_auth_header(auth)
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
