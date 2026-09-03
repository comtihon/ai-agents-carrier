"""Naming data for download, and serving it back.

Two halves, both shared so the step, the REST routes and the MCP tools cannot
disagree:

*capture* — what a `data` step does.  It resolves each declared selection
against workflow state and writes a manifest entry for it.  When the selection
is already a ``DataRef`` **nothing is copied**: the bytes are in the stream
store, so the stream is pinned and the entry points at it.  Re-uploading a
four-million-record result that is already stored is precisely the mistake the
streaming work exists to prevent.  Only a plain state value has to be written,
because there is no stream for it yet.

*serve* — what a download does.  ``jsonl`` is the stored form, so it goes out
untouched, byte for byte, with a real ``Content-Length``.  ``json`` and ``csv``
are transforms, and both are applied *while streaming*: a request for a
million-record CSV must cost the backend one batch of records, not the whole
file.

Neither half touches workflow state.  A `data` node returns ``{}`` — no new
keys, no mutation, no transform of the data it names — because the whole point
of it is to observe a workflow, and an observer that changes what it observes
is not one.

*resolve* sits between the two, and it is what makes a data source result
already in a run's state downloadable by the same mechanism rather than by a
reader of its own.  Those entries are synthesised, are not pinned, and are off
by default in a listing; the resolver takes a *run* rather than a stream id, so
nothing here can open a file a caller merely named.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import queue
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from app.domain.models.data_artifact import FORMATS, DataArtifact
from app.domain.models.datastream import as_data_ref

logger = logging.getLogger(__name__)

# Prefixes a `from` may carry. Every one of them means "workflow state", which
# is the only thing a selection can address, so they are dropped before the
# path is walked. Same set `sheet_binding_compile.param_name_for` accepts, so a
# path copied from a sheet binding works here unchanged.
_STATE_PREFIXES = ("state.", "$.state.", "$.")

# Records read to decide a CSV's header. The header is the union of the keys in
# this prefix, not of the whole file: a header cannot be emitted until the keys
# are known, and reading four million records to find out would mean holding
# the download open with nothing to send. A record beyond the prefix carrying a
# key the header does not have is reported in the log and its extra values are
# dropped, which is the one place this is lossy and is why the bound is
# generous.
CSV_HEADER_SCAN_ITEMS = 500

# Blocks held in flight between the store's copy and the HTTP response. Small:
# its only job is to keep the transfer from stalling on every block boundary,
# and a large one would be a buffer of exactly the kind this avoids.
_COPY_QUEUE_DEPTH = 4

_EOF = object()


class DataArtifactError(RuntimeError):
    """Raised when an artifact cannot be served in the format it was asked in."""


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Selection:
    """One ``selections:`` entry of a `data` step."""

    name: str
    path: str
    format: str = "jsonl"


def strip_state_prefix(path: str) -> str:
    """``state.projects`` -> ``projects``; anything else is returned as given."""
    trimmed = (path or "").strip()
    for prefix in _STATE_PREFIXES:
        if trimmed.startswith(prefix):
            return trimmed[len(prefix):]
    return trimmed


def parse_selections(step: dict[str, Any]) -> tuple[list[Selection], list[str]]:
    """Usable selections of *step*, plus a message per unusable one.

    Unusable entries are reported rather than raised for the same reason a
    missing path is: a `data` node observes, so a typo in one selection must
    not be the reason a workflow dies with the other three unrecorded.
    """
    raw = step.get("selections")
    if raw is None:
        return [], ["declares no `selections`, so it has nothing to record"]
    if not isinstance(raw, list):
        return [], ["`selections` must be a list of {name, from, format} entries"]

    good: list[Selection] = []
    problems: list[str] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        where = f"selection {index}"
        if not isinstance(entry, dict):
            problems.append(f"{where} is not a mapping")
            continue
        name = str(entry.get("name") or "").strip()
        path = str(entry.get("from") or "").strip()
        fmt = str(entry.get("format") or "jsonl").strip().lower()
        if not name:
            problems.append(f"{where} has no `name`")
            continue
        if not path:
            problems.append(f"selection '{name}' has no `from`")
            continue
        if fmt not in FORMATS:
            problems.append(
                f"selection '{name}' asks for format '{fmt}'; "
                f"supported: {', '.join(FORMATS)}"
            )
            continue
        if name in seen:
            # Two entries with one name would produce two artifacts with the
            # same filename, and a user with two identical rows in the UI.
            problems.append(f"selection '{name}' is declared more than once")
            continue
        seen.add(name)
        good.append(Selection(name=name, path=path, format=fmt))
    return good, problems


async def capture_selection(
    *,
    store: Any,
    backend: Any,
    run_id: str,
    step_id: str,
    selection: Selection,
    value: Any,
    ttl_seconds: float,
) -> DataArtifact:
    """Record *value* as a downloadable artifact of ``(run_id, step_id)``.

    A ``DataRef`` is pinned where it already lies and nothing is read or
    written; anything else is written to a new stream, because there is no
    stream for a plain state value yet.  The pin is taken before the manifest
    entry is stored, so a row never outlives the retention exemption that keeps
    its bytes alive.
    """
    ref = as_data_ref(value)
    if ref is None:
        ref = await _write_plain_value(store, selection, value)
        logger.info(
            "data step '%s' selection '%s': wrote plain state value to stream %s "
            "(%d item(s), %d bytes)",
            step_id, selection.name, ref.id, ref.items, ref.bytes,
        )
    else:
        logger.info(
            "data step '%s' selection '%s': pinned existing stream %s "
            "(%d item(s), %d bytes) -- no copy",
            step_id, selection.name, ref.id, ref.items, ref.bytes,
        )

    if selection.format == "csv" and ref.shape != "list":
        raise DataArtifactError(
            f"selection '{selection.name}' asks for csv, but the data behind it "
            f"is a single document, not a list of records. csv needs rows; use "
            f"format: json for a document."
        )

    await store.pin(ref)
    artifact = DataArtifact.for_stream(
        run_id=run_id,
        step_id=step_id,
        name=selection.name,
        fmt=selection.format,
        ref=ref,
        ttl_seconds=ttl_seconds,
    )
    await backend.add(artifact)
    if artifact.truncated:
        # The one failure here that does real damage is somebody downloading a
        # prefix believing it is the whole answer, so it is said loudly at every
        # point it passes through, not only in the manifest.
        logger.warning(
            "data step '%s' selection '%s' -> artifact %s is TRUNCATED: the "
            "stream is a prefix, and a download of it is not the whole answer",
            step_id, selection.name, artifact.id,
        )
    return artifact


async def _write_plain_value(store: Any, selection: Selection, value: Any) -> Any:
    """Write a non-ref state value to a stream and return its ref.

    A list becomes one record per line, matching what a data source result of
    the same shape looks like, so the download formats need no second code
    path.  Anything else is one document (``shape="value"``).
    """
    is_list = isinstance(value, list)
    writer = await store.open_writer(
        source_id="", operation=selection.name, shape="list" if is_list else "value"
    )
    try:
        await writer.append_many(value if is_list else [value])
        return await writer.close()
    except BaseException:
        await writer.abort()
        raise


# ---------------------------------------------------------------------------
# Resolving what a run has to offer
# ---------------------------------------------------------------------------
#
# Two origins, one code path.  A `data` step's selections are stored manifest
# rows; a data source result is a ref already sitting in the run's state and is
# described in the same shape on demand.  Everything below is shared by the
# REST routes and the MCP tools, so a download is the same operation whichever
# kind it names and whichever surface asked.
#
# The security property, and it is the reason this takes a *run* rather than a
# stream id: a datasource ref is only ever found by looking inside the state of
# a run the caller may read.  A resolver that opened whatever ``ds_*`` id it was
# handed would be a cross-run read primitive, which is exactly what the store's
# id validation exists to prevent.  Run-scoped resolution makes a data source
# result exactly as accessible as the run that produced it — the same rule the
# data-node download already follows.

def datasource_artifacts(run: Any, *, ttl_seconds: float) -> list[DataArtifact]:
    """Entries for every data source result in *run*'s state, newest first."""
    from app.domain.models.datastream import find_data_refs

    state = run.state if isinstance(getattr(run, "state", None), dict) else {}
    entries = [
        DataArtifact.for_datasource_ref(
            run_id=run.id, state_key=key, ref=ref, ttl_seconds=ttl_seconds
        )
        for key, ref in find_data_refs(state).items()
    ]
    entries.sort(key=lambda a: a.created_at, reverse=True)
    return entries


def datasource_artifact_in(
    run: Any, artifact_id: str, *, ttl_seconds: float
) -> DataArtifact | None:
    """The entry *artifact_id* names inside *run*'s state, or ``None``."""
    from app.domain.models.data_artifact import stream_id_of_datasource_artifact
    from app.domain.models.datastream import find_data_refs

    stream_id = stream_id_of_datasource_artifact(artifact_id)
    if stream_id is None:
        return None
    state = run.state if isinstance(getattr(run, "state", None), dict) else {}
    for key, ref in find_data_refs(state).items():
        if ref.id == stream_id:
            return DataArtifact.for_datasource_ref(
                run_id=run.id, state_key=key, ref=ref, ttl_seconds=ttl_seconds
            )
    return None


async def list_run_artifacts(
    backend: Any,
    run: Any,
    *,
    include_datasource: bool = False,
    datasource_ttl_seconds: float = 0.0,
) -> list[DataArtifact]:
    """Everything *run* offers for download.

    ``include_datasource`` is off by default, and that default is the point:
    the run UI's Data panel shows the files somebody chose to export, not every
    intermediate result the run happened to fetch.  A caller that wants the raw
    results has to say so.
    """
    stored = [] if backend is None else list(await backend.list_for_run(run.id))
    if not include_datasource:
        return stored
    # A curated export wins where both describe the same stream: it is the one
    # a person named, and it is the one that is pinned.
    pinned_streams = {a.stream_id for a in stored}
    extra = [
        a for a in datasource_artifacts(run, ttl_seconds=datasource_ttl_seconds)
        if a.stream_id not in pinned_streams
    ]
    return stored + extra


async def find_run_artifact(
    backend: Any,
    run: Any,
    artifact_id: str,
    *,
    datasource_ttl_seconds: float = 0.0,
) -> DataArtifact | None:
    """One artifact of *run*, of either origin.

    No opt-in here: the opt-in governs what a *listing* volunteers, and a
    caller holding an id already knows what it is asking for.  Both kinds
    resolve through the same lookup, so a download cannot be told apart by the
    shape of the call.
    """
    stored = None if backend is None else await backend.get(run.id, artifact_id)
    if stored is not None:
        return stored
    return datasource_artifact_in(
        run, artifact_id, ttl_seconds=datasource_ttl_seconds
    )


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

@dataclass
class Download:
    """Everything the HTTP layer needs to answer a download request."""

    filename: str
    content_type: str
    # None when the served size is not known up front, which is every streamed
    # transform: a wrong Content-Length is worse than none, since a client
    # truncates the body to it.
    content_length: int | None
    chunks: AsyncIterator[bytes]


async def prepare_download(store: Any, artifact: DataArtifact) -> Download:
    """Open *artifact* for streaming in the format it was recorded in.

    Raises ``StreamGone`` when the stream behind it has been swept (the route
    turns that into 410) and :class:`DataArtifactError` when the data cannot be
    represented in the requested format.  Both are raised *before* the response
    body starts, which is the only point at which a status code can still be
    chosen.
    """
    ref = artifact.as_ref()
    if artifact.format == "jsonl" or (
        artifact.format == "json" and artifact.shape == "value"
    ):
        # Already exactly the bytes asked for: JSONL is the store's own form,
        # and a `shape="value"` stream is a single JSON document on one line.
        # Serving it through copy_to means no parse, no re-encode and a real
        # Content-Length -- and copy_to is the one read path every store must
        # implement, so this works for object storage as well as local disk.
        return Download(
            filename=artifact.filename,
            content_type=artifact.content_type,
            content_length=artifact.bytes,
            chunks=await _started(raw_chunks(store, ref)),
        )

    if artifact.format == "json":
        return Download(
            filename=artifact.filename,
            content_type=artifact.content_type,
            content_length=None,
            chunks=await _started(_json_array_chunks(store, ref)),
        )

    if artifact.format == "csv":
        if artifact.shape != "list":
            raise DataArtifactError(
                f"artifact '{artifact.id}' is a single document, not a list of "
                f"records, so it cannot be served as csv."
            )
        header = await _csv_header(store, ref)
        return Download(
            filename=artifact.filename,
            content_type=artifact.content_type,
            content_length=None,
            chunks=_csv_chunks(store, ref, header),
        )

    raise DataArtifactError(f"unsupported artifact format '{artifact.format}'")


class _QueueSink:
    """A binary sink that hands each block to the reader through a queue.

    ``copy_to`` pushes; an HTTP response body pulls.  A bounded thread-safe
    queue is the bridge, and its bound is the back-pressure: when the client
    reads slowly the copy blocks on ``put`` instead of buffering ahead, so a
    four-million-record download costs the process a few blocks rather than the
    file.  It has to be a *thread* queue because the stores do their copying on
    a worker thread — ``asyncio.Queue`` is not safe to touch from there.
    """

    def __init__(self, depth: int = _COPY_QUEUE_DEPTH) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=depth)

    def write(self, block: bytes) -> int:
        if block:
            self._q.put(bytes(block))
        return len(block)

    def flush(self) -> None:  # a BinaryIO is expected to have one
        return None

    def finish(self, error: BaseException | None = None) -> None:
        self._q.put(error if error is not None else _EOF)

    def take(self) -> Any:
        return self._q.get()


async def raw_chunks(store: Any, ref: Any) -> AsyncIterator[bytes]:
    """The stored bytes of *ref*, unaltered, in blocks."""
    sink = _QueueSink()

    async def _pump() -> None:
        error: BaseException | None = None
        try:
            await store.copy_to(ref, sink)  # type: ignore[arg-type]
        except BaseException as exc:  # noqa: BLE001 — handed to the consumer
            error = exc
        finally:
            sink.finish(error)

    pump = asyncio.create_task(_pump())
    try:
        while True:
            # Blocking get on a worker thread: the event loop stays free while
            # the copy fills the queue.
            item = await asyncio.to_thread(sink.take)
            if item is _EOF:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # An abandoned download (the client hung up) must not leave the copy
        # blocked on a queue nobody drains.
        if not pump.done():
            pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


async def _json_array_chunks(store: Any, ref: Any) -> AsyncIterator[bytes]:
    """The stream re-framed as one JSON array, a record at a time.

    Never materialised: the array's brackets and commas are emitted around the
    records as they arrive, so the largest thing in memory is one record.

    The opening bracket goes out *with* the first record rather than ahead of
    it, so that starting this generator is already a read of the stream -- see
    ``_started``: a bracket emitted first would mean a missing stream was
    discovered after the response had begun, when the status code is settled.
    """
    first = True
    async for record in store.stream(ref):
        line = json.dumps(record, default=str, ensure_ascii=False).encode("utf-8")
        yield b"[" + line if first else b"," + line
        first = False
    yield b"[]" if first else b"]"


async def _started(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Pull the first block now, then hand back the whole sequence.

    A ``StreamGone`` from inside a generator that has not been started yet
    would surface only once the response body was already going out, by which
    time the status code is settled and the client gets a truncated 200. Doing
    the first read here means the route can still answer 410.
    """
    try:
        first = await chunks.__anext__()
    except StopAsyncIteration:
        first = b""

    async def _chained() -> AsyncIterator[bytes]:
        if first:
            yield first
        async for block in chunks:
            yield block

    return _chained()


async def _csv_header(store: Any, ref: Any) -> list[str]:
    """Column names: the union of the keys in a bounded leading prefix.

    Union rather than the first record's keys, because a sparse API response
    routinely omits a null field, and taking the first row's keys would drop
    every column that happens to be absent from record one.  A record that is
    not a flat mapping is refused here — before a single byte of the response
    has been sent, so the caller gets a status code rather than a corrupt file.
    """
    header: list[str] = []
    seen: set[str] = set()
    scanned = 0
    async for record in store.stream(ref, limit=CSV_HEADER_SCAN_ITEMS):
        scanned += 1
        _require_flat(record, scanned)
        for key in record:
            name = str(key)
            if name not in seen:
                seen.add(name)
                header.append(name)
    if not header:
        raise DataArtifactError(
            "csv needs at least one column, and the first "
            f"{CSV_HEADER_SCAN_ITEMS} record(s) of this stream have no fields."
        )
    return header


def _require_flat(record: Any, position: int) -> None:
    if not isinstance(record, dict):
        raise DataArtifactError(
            f"csv needs a list of flat records, but record {position} is a "
            f"{type(record).__name__}. Use format: jsonl or json for this data, "
            f"or give the operation a `mapping` that flattens it."
        )
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            raise DataArtifactError(
                f"csv needs flat records, but record {position} has a nested "
                f"{type(value).__name__} under '{key}'. Use format: jsonl or "
                f"json for this data, or flatten it upstream."
            )


async def _csv_chunks(
    store: Any, ref: Any, header: list[str]
) -> AsyncIterator[bytes]:
    """Header row, then one CSV row per record, streamed."""
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=header, extrasaction="ignore", lineterminator="\r\n"
    )
    writer.writeheader()
    yield _drain(buffer)

    position = 0
    async for record in store.stream(ref):
        position += 1
        # Past the header scan a nested value can no longer become a status
        # code -- the response is already going out -- so it aborts the
        # transfer loudly rather than writing a row that says "{'a': 1}".
        _require_flat(record, position)
        missing = [k for k in record if k not in header]
        if missing:
            logger.warning(
                "csv download of stream '%s': record %d has column(s) %s that "
                "are not in the header taken from the first %d record(s); their "
                "values are omitted",
                ref.id, position, ", ".join(sorted(missing)), CSV_HEADER_SCAN_ITEMS,
            )
        writer.writerow({k: _csv_cell(record.get(k)) for k in header})
        if buffer.tell() >= 64 * 1024:
            yield _drain(buffer)
    tail = _drain(buffer)
    if tail:
        yield tail


def _csv_cell(value: Any) -> Any:
    """A missing field is an empty cell, not the string ``None``."""
    return "" if value is None else value


def _drain(buffer: io.StringIO) -> bytes:
    text = buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

async def forget_run_artifacts(store: Any, backend: Any, run_id: str) -> int:
    """Drop a run's manifest and unpin the streams it held.

    Called when a run is deleted.  Unpinning is the whole point: without it the
    streams would keep their retention exemption forever and the sweep would
    never reclaim them.  A stream that is already gone is not an error — the
    goal state is "not pinned".
    """
    if backend is None:
        return 0
    rows = await backend.delete_for_run(run_id)
    for artifact in rows:
        if store is None:
            continue
        try:
            await store.unpin(artifact.as_ref())
        except Exception:  # noqa: BLE001 — deleting a run must not fail on this
            logger.debug("could not unpin stream '%s'", artifact.stream_id, exc_info=True)
    if rows:
        logger.info("run %s: forgot %d data artifact(s)", run_id, len(rows))
    return len(rows)
