"""Data source results as files, always — never as values in workflow state.

The rule
--------
Every data source result is written to a file and passed on as a
:class:`~app.domain.models.datastream.DataRef`.  Every result, whatever its
size.  A consumer that wants the data opens a file descriptor and reads it.

Doing this unconditionally is the point.  The alternative — inline below a
threshold, file above it — means every consumer has to handle both shapes,
every workflow behaves differently in production than in test depending on how
much data happened to come back, and the size at which behaviour flips is a
number nobody can predict in advance.  One path is worth the cost of writing a
40-byte result to disk.

What it buys:

* workflow state, the LangGraph checkpoint and the Mongo run document hold a
  ~200-byte ref instead of the payload, so none of them is a function of
  result size (16 MB BSON, ~780 KB ConfigMap, 1 GiB pod);
* the backend never holds a whole result: pages go to the file as they arrive;
* a consumer reads at whatever rate it can, from a real fd, with `lseek`, so
  it can make several passes without a second fetch.

Format
------
Newline-delimited JSON, one record per line (``shape="list"``).  Cheap to
append without re-reading, cheap to read back in pieces, and the line count is
the record count.  A response with no natural record boundary is one line
(``shape="value"``); it can be read but not iterated through.

Getting the fd to the consumer
------------------------------
``open_fd`` returns an OS-level descriptor.  Where the consumer runs in the
same process or on the same pod (``local``, ``docker``) that descriptor, or the
path behind it, is handed straight over.  Where it runs in another pod
(``k8s``) the bytes are streamed into that pod, which writes its own file and
opens its own fd — ``copy_to`` is that transfer.  The sandbox cannot fetch for
itself: its seccomp filter denies both ``socket`` and ``openat``, so the file
must be opened for it before the filter is installed and pushed to it by the
backend.  See ``script_sandbox``.

Consumers, and the one that is different
----------------------------------------
A script reads the fd.  A fan-out reads the fd.  A write back to another API
reads the fd.  All of them genuinely stream, and their memory is flat.

An LLM cannot read an fd.  There is no file input to a model: bytes have to
become tokens inside a context window, so something must decide *which* bytes.
The ``llm`` step therefore reads the fd on the model's behalf and puts a
bounded selection in the prompt — a stated sample, or chunk-by-chunk with a
combining pass.  That is not a limitation of this module; it is what "give an
LLM a file" can mean at all.

Durability
----------
``LocalDiskStreamStore`` writes to the pod's filesystem: right for one run,
wrong across a pause, since a restart loses every file and a run parked at
``waiting_approval`` will come back to a ref pointing at nothing (it gets a
clear error, never a short read).  The :class:`DataStreamStore` ABC is the seam
for a durable backend — ``GcsStreamStore`` in ``datastream_gcs`` is one, and
GridFS would need no new infrastructure at all.  ``persistent`` says which kind
a store is.

Retention
---------
``purge_older_than`` sweeps on the short ``STREAM_TTL_SECONDS`` window, which
is right for a stream that exists only to get from one step to the next.  A
`data` step names data a person downloads later, so those streams are
``pin``-ned: the sweep holds a pinned stream to the much longer
``DATA_ARTIFACT_TTL_SECONDS`` instead, and deleting the download manifest entry
``unpin``s it back into the ordinary window.  Pinning changes when a stream is
swept, never whether the store is durable — a pin on a local-disk store still
does not survive the pod.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any, BinaryIO

from app.domain.models.datastream import PREVIEW_ITEMS, DataRef, shrink_preview

logger = logging.getLogger(__name__)

# Lines pulled per thread hop. Large enough that the hop cost is irrelevant,
# small enough that a batch of fat records is not itself a memory problem.
_READ_BATCH_LINES = 500
# Bytes per hop when copying a stream to another pod.
_COPY_CHUNK_BYTES = 256 * 1024


class StreamReadTooLarge(RuntimeError):
    """Raised when a caller asks to load a whole stream and it will not fit."""


class NotStreamable(RuntimeError):
    """Raised when a caller tries to iterate a ``shape="value"`` stream."""


class StreamGone(FileNotFoundError):
    """Raised when the file behind a ref no longer exists."""


class ResultTooLarge(RuntimeError):
    """Raised when a result passes ``max_result_bytes`` and cannot be truncated."""


class StreamWriter(ABC):
    """Append-only sink for one result."""

    @abstractmethod
    async def append_many(self, items: Iterable[Any]) -> int:
        """Append records; returns bytes written by this call."""

    @abstractmethod
    async def close(self) -> DataRef: ...

    @abstractmethod
    async def abort(self) -> None:
        """Discard a partial write (the fetch failed)."""

    @property
    @abstractmethod
    def bytes_written(self) -> int: ...


class DataStreamStore(ABC):
    """Where data source results live between steps."""

    #: False when contents do not survive a pod restart.
    persistent: bool = False

    @abstractmethod
    async def open_writer(
        self, *, source_id: str = "", operation: str = "", shape: str = "list"
    ) -> StreamWriter: ...

    @abstractmethod
    async def open_fd(self, ref: DataRef) -> int:
        """An OS file descriptor positioned at the start of the stream.

        The caller owns it and must close it.  This is the primitive every
        same-pod consumer uses; ``stream`` and ``chunks`` are conveniences on
        top of it for callers that want parsed records.
        """

    @abstractmethod
    async def local_path(self, ref: DataRef) -> str | None:
        """Filesystem path of the stream, when it has one on this pod.

        ``None`` for a store whose bytes are not on the local filesystem
        (GridFS, object storage); such a store is read with ``copy_to``.
        """

    @abstractmethod
    async def copy_to(self, ref: DataRef, sink: BinaryIO) -> int:
        """Stream the raw bytes into *sink*; returns bytes written.

        How a stream reaches a consumer in another pod: the backend writes the
        bytes into that pod's stdin and the pod writes its own file.  Chunked,
        so neither side holds the whole thing.
        """

    @abstractmethod
    def stream(
        self, ref: DataRef, *, start: int = 0, limit: int | None = None
    ) -> AsyncIterator[Any]:
        """Yield parsed records, holding at most one read batch."""

    @abstractmethod
    def chunks(
        self, ref: DataRef, *, size: int, max_bytes: int = 0, start: int = 0
    ) -> AsyncIterator[list[Any]]:
        """Yield bounded batches: at most *size* records and *max_bytes* encoded."""

    @abstractmethod
    async def read_all(self, ref: DataRef, *, max_bytes: int) -> Any:
        """Load the whole stream into memory, refusing past *max_bytes*.

        The explicit opt-out, for a step configured ``result_mode: ram``.  It
        raises rather than degrading: a silent partial read is the failure this
        design exists to remove.
        """

    @abstractmethod
    async def delete(self, ref: DataRef) -> None: ...

    @abstractmethod
    async def pin(self, ref: DataRef) -> None:
        """Exempt a stream from the ordinary retention sweep.

        A `data` step names data a person will come back for, possibly hours
        later; the 6-hour ``STREAM_TTL_SECONDS`` sweep would delete exactly the
        file they came back for.  Pinning is what a download manifest entry
        holds a stream by, so it is set when the entry is written and cleared
        when the entry is deleted.

        Idempotent, and pinning a stream that is already gone raises
        :class:`StreamGone` — a manifest entry must never be written against a
        stream that cannot be served.
        """

    @abstractmethod
    async def unpin(self, ref: DataRef) -> None:
        """Return a pinned stream to the ordinary sweep.  Idempotent."""

    @abstractmethod
    async def is_pinned(self, ref: DataRef) -> bool: ...

    @abstractmethod
    async def purge_older_than(
        self, seconds: float, *, pinned_seconds: float | None = None
    ) -> int:
        """Drop streams older than *seconds*; returns how many were removed.

        Pinned streams are held to *pinned_seconds* instead — the longer
        ``DATA_ARTIFACT_TTL_SECONDS`` window a download is offered for. ``None``
        keeps them indefinitely, which is what a caller sweeping only the
        ordinary backlog wants.
        """


# ---------------------------------------------------------------------------
# Local disk
# ---------------------------------------------------------------------------

class _LocalDiskWriter(StreamWriter):
    def __init__(
        self, path: Path, fh: Any, *, source_id: str, operation: str, shape: str
    ) -> None:
        self._path = path
        # Opened by the store on a worker thread and held for the whole write:
        # a multi-page fetch is one file handle, not one per page.
        self._fh = fh
        self._source_id = source_id
        self._operation = operation
        self._shape = shape
        self._items = 0
        self._bytes = 0
        self._preview: list[Any] = []
        self._closed = False

    @property
    def bytes_written(self) -> int:
        return self._bytes

    async def append_many(self, items: Iterable[Any]) -> int:
        materialised = list(items)
        if not materialised:
            return 0
        if len(self._preview) < PREVIEW_ITEMS:
            self._preview = shrink_preview(
                self._preview + materialised[: PREVIEW_ITEMS - len(self._preview)]
            )

        def _write() -> int:
            written = 0
            for item in materialised:
                line = json.dumps(item, default=str, ensure_ascii=False)
                self._fh.write(line)
                self._fh.write("\n")
                written += len(line.encode("utf-8")) + 1
            self._fh.flush()
            return written

        written = await asyncio.to_thread(_write)
        self._bytes += written
        self._items += len(materialised)
        return written

    async def close(self) -> DataRef:
        if not self._closed:
            await asyncio.to_thread(self._fh.close)
            self._closed = True
        return DataRef(
            id=self._path.stem,
            shape="value" if self._shape == "value" else "list",
            items=self._items,
            bytes=self._bytes,
            source_id=self._source_id,
            operation=self._operation,
            preview=self._preview,
        )

    async def abort(self) -> None:
        if not self._closed:
            await asyncio.to_thread(self._fh.close)
            self._closed = True
        try:
            await asyncio.to_thread(self._path.unlink)
        except FileNotFoundError:
            pass


class LocalDiskStreamStore(DataStreamStore):
    """Stream store backed by the pod's own filesystem."""

    persistent = False

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, stream_id: str) -> Path:
        # The id becomes a filename, so anything but the shape it was minted
        # with is refused rather than resolved -- a ref reaching us altered
        # must not be able to name another file.
        if not stream_id or "/" in stream_id or stream_id.startswith("."):
            raise ValueError(f"invalid stream id {stream_id!r}")
        return self._dir / f"{stream_id}.jsonl"

    def _pin_path_for(self, stream_id: str) -> Path:
        """Sidecar marking a stream exempt from the ordinary sweep.

        A separate empty file rather than a field inside the data file or an
        index of its own: the data file is append-only JSONL that consumers
        read as bytes, and an index would be a second thing to keep consistent
        with the directory.  The ``.pin`` suffix keeps it out of the
        ``ds_*.jsonl`` glob the sweep walks, so a marker is never mistaken for
        a stream.  Goes through ``_path_for`` first, so a tampered id cannot
        name a file outside the directory here either.
        """
        return self._path_for(stream_id).with_suffix(".pin")

    def _existing_path(self, ref: DataRef) -> Path:
        path = self._path_for(ref.id)
        if not path.exists():
            raise StreamGone(
                f"data stream '{ref.id}' is gone. A local-disk stream does not "
                f"survive a pod restart and is swept on a TTL; re-run the data "
                f"source step that produced it."
            )
        return path

    async def open_writer(
        self, *, source_id: str = "", operation: str = "", shape: str = "list"
    ) -> StreamWriter:
        stream_id = f"ds_{uuid.uuid4().hex}"
        path = self._path_for(stream_id)
        fh = await asyncio.to_thread(path.open, "w", encoding="utf-8")
        return _LocalDiskWriter(
            path, fh, source_id=source_id, operation=operation, shape=shape
        )

    async def open_fd(self, ref: DataRef) -> int:
        path = self._existing_path(ref)
        return await asyncio.to_thread(os.open, str(path), os.O_RDONLY)

    async def local_path(self, ref: DataRef) -> str | None:
        return str(self._existing_path(ref))

    async def copy_to(self, ref: DataRef, sink: BinaryIO) -> int:
        path = self._existing_path(ref)

        def _copy() -> int:
            total = 0
            with path.open("rb") as src:
                while True:
                    block = src.read(_COPY_CHUNK_BYTES)
                    if not block:
                        break
                    sink.write(block)
                    total += len(block)
            return total

        written = await asyncio.to_thread(_copy)
        logger.debug("data stream '%s': copied %d bytes out", ref.id, written)
        return written

    async def _iter_batches(
        self, ref: DataRef, *, start: int, limit: int | None
    ) -> AsyncIterator[list[Any]]:
        path = self._existing_path(ref)
        state: dict[str, Any] = {"fh": None, "index": 0, "yielded": 0}

        def _open() -> None:
            state["fh"] = path.open("r", encoding="utf-8")

        def _next_batch() -> list[Any]:
            fh = state["fh"]
            out: list[Any] = []
            while len(out) < _READ_BATCH_LINES:
                if limit is not None and state["yielded"] >= limit:
                    break
                line = fh.readline()
                if not line:
                    break
                index = state["index"]
                state["index"] = index + 1
                if index < start:
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    out.append(json.loads(stripped))
                except json.JSONDecodeError:
                    logger.warning(
                        "data stream '%s': unparseable line %d skipped", ref.id, index
                    )
                    continue
                state["yielded"] = state["yielded"] + 1
            return out

        await asyncio.to_thread(_open)
        try:
            while True:
                batch = await asyncio.to_thread(_next_batch)
                if not batch:
                    return
                yield batch
                if limit is not None and state["yielded"] >= limit:
                    return
        finally:
            fh = state.get("fh")
            if fh is not None:
                await asyncio.to_thread(fh.close)

    async def stream(
        self, ref: DataRef, *, start: int = 0, limit: int | None = None
    ) -> AsyncIterator[Any]:
        if ref.shape != "list":
            raise NotStreamable(
                f"data stream '{ref.id}' is a single document, not a list of "
                f"records, so it cannot be iterated. Give the operation a "
                f"`mapping` that extracts its record array."
            )
        async for batch in self._iter_batches(ref, start=start, limit=limit):
            for item in batch:
                yield item

    async def chunks(
        self, ref: DataRef, *, size: int, max_bytes: int = 0, start: int = 0
    ) -> AsyncIterator[list[Any]]:
        if size < 1:
            raise ValueError("chunk size must be at least 1")
        pending: list[Any] = []
        pending_bytes = 0
        async for item in self.stream(ref, start=start):
            # Measured before appending, so a chunk is emitted *under* the cap
            # rather than one record over it. A single record that alone
            # exceeds the cap goes out on its own: splitting it would corrupt
            # it, and the caller can see the size.
            encoded = len(json.dumps(item, default=str).encode("utf-8"))
            if pending and (
                len(pending) >= size
                or (max_bytes and pending_bytes + encoded > max_bytes)
            ):
                yield pending
                pending, pending_bytes = [], 0
            pending.append(item)
            pending_bytes += encoded
        if pending:
            yield pending

    async def read_all(self, ref: DataRef, *, max_bytes: int) -> Any:
        if max_bytes and ref.bytes > max_bytes:
            raise StreamReadTooLarge(
                f"data stream '{ref.id}' is {ref.bytes} bytes, over the "
                f"{max_bytes}-byte limit for loading one into memory. Read it "
                f"from the fd or fold over it in chunks instead."
            )
        if ref.shape == "value":
            batches = [b async for b in self._iter_batches(ref, start=0, limit=1)]
            flat = [item for batch in batches for item in batch]
            return flat[0] if flat else None
        out: list[Any] = []
        async for batch in self._iter_batches(ref, start=0, limit=None):
            out.extend(batch)
        return out

    async def delete(self, ref: DataRef) -> None:
        try:
            path = self._path_for(ref.id)
        except ValueError:
            return
        for target in (path, self._pin_path_for(ref.id)):
            try:
                await asyncio.to_thread(target.unlink)
            except FileNotFoundError:
                pass

    async def pin(self, ref: DataRef) -> None:
        # Existence is checked first: a manifest entry written against a
        # stream that is already swept would offer a download that can only
        # ever 410.
        self._existing_path(ref)
        marker = self._pin_path_for(ref.id)
        await asyncio.to_thread(marker.touch)

    async def unpin(self, ref: DataRef) -> None:
        try:
            marker = self._pin_path_for(ref.id)
        except ValueError:
            return
        try:
            await asyncio.to_thread(marker.unlink)
        except FileNotFoundError:
            pass

    async def is_pinned(self, ref: DataRef) -> bool:
        try:
            marker = self._pin_path_for(ref.id)
        except ValueError:
            return False
        return await asyncio.to_thread(marker.exists)

    async def purge_older_than(
        self, seconds: float, *, pinned_seconds: float | None = None
    ) -> int:
        def _purge() -> int:
            if not self._dir.exists():
                return 0
            now = time.time()
            cutoff = now - seconds
            pinned_cutoff = None if pinned_seconds is None else now - pinned_seconds
            removed = 0
            for path in self._dir.glob("ds_*.jsonl"):
                marker = path.with_suffix(".pin")
                pinned = marker.exists()
                # A pinned stream is held to the longer artifact window, or
                # kept outright when the caller named no window for it. The
                # sweep must not be the reason a download offered to somebody
                # stops working.
                limit = pinned_cutoff if pinned else cutoff
                if limit is None:
                    continue
                try:
                    if path.stat().st_mtime >= limit:
                        continue
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    continue
                if pinned:
                    try:
                        marker.unlink()
                    except FileNotFoundError:
                        pass
            return removed

        removed = await asyncio.to_thread(_purge)
        if removed:
            logger.info("data stream store: purged %d expired stream(s)", removed)
        return removed

    async def free_bytes(self) -> int:
        """Space left on the stream filesystem, for a pre-flight check."""
        def _free() -> int:
            return int(shutil.disk_usage(self._dir).free)

        return await asyncio.to_thread(_free)


# ---------------------------------------------------------------------------
# Building one result into a stream
# ---------------------------------------------------------------------------

class StreamBuilder:
    """Writes a data source result to the stream store as it arrives.

    There is no in-memory accumulation and no threshold to cross: the writer
    is open before the first page and each page goes to the file and is
    dropped.  The backend's memory for a fetch is therefore one page, whether
    the result is one record or four million.

    ``max_result_bytes`` is the one remaining ceiling, and it is about the
    disk rather than about memory: past it the fetch stops and the ref is
    flagged ``truncated``, which every consumer can read -- a bounded honest
    prefix beats filling the node's 40 GB.
    """

    def __init__(
        self,
        *,
        store: DataStreamStore,
        max_result_bytes: int,
        source_id: str = "",
        operation: str = "",
        limit: int | None = None,
    ) -> None:
        self._store = store
        self._max = max(0, max_result_bytes)
        # How many records the caller actually wants. None means everything
        # the source has -- the paginator then walks until the API says it is
        # done, which is the point of asking for no limit. A limit is not a
        # truncation: stopping because enough was collected is the caller
        # getting what it asked for, so `truncated` stays False.
        self._limit = limit if limit is None or limit > 0 else None
        self._source_id = source_id
        self._operation = operation
        self._writer: StreamWriter | None = None
        self._bytes = 0
        self._items = 0
        self._saw_list = False
        self._scalar_adds = 0
        self._truncated = False
        self._projected_bytes: int | None = None

    # -- state ----------------------------------------------------------
    @property
    def full(self) -> bool:
        """True once there is no reason to fetch another page.

        Either the byte ceiling was reached (a truncation, flagged as one) or
        the caller's row limit was satisfied (not a truncation).
        """
        return self._truncated or self.limit_reached

    @property
    def limit_reached(self) -> bool:
        return self._limit is not None and self._items >= self._limit

    @property
    def remaining(self) -> int | None:
        """Rows still wanted, or ``None`` when the caller set no limit.

        The paginator uses it to size the next page, so a limit of 10 against
        a 100-row page size fetches ten rows rather than a hundred and throws
        ninety away.
        """
        if self._limit is None:
            return None
        return max(0, self._limit - self._items)

    @property
    def items_written(self) -> int:
        return self._items

    @property
    def bytes_written(self) -> int:
        return self._bytes

    @property
    def projected_bytes(self) -> int | None:
        return self._projected_bytes

    # -- accumulation ---------------------------------------------------
    def project(self, total_items: Any, first_page: Any) -> int | None:
        """Project the finished size from the API's own total count.

        Called once, with the value at ``paginate.total_path`` and the first
        page.  Nothing about *where* the data goes depends on this any more --
        it always goes to the file -- but the projection is still worth having:
        it puts the finished size in the log before the walk, and it lets a
        read that will breach ``max_result_bytes`` be reported at page one
        instead of discovered at page forty.
        """
        if not isinstance(total_items, (int, float)) or isinstance(total_items, bool):
            return None
        total = int(total_items)
        if total < 1 or not isinstance(first_page, list) or not first_page:
            return None
        per_item = _encoded_size(first_page) / len(first_page)
        projected = int(per_item * total)
        self._projected_bytes = projected
        logger.info(
            "data source '%s' operation '%s': API reports %d total records, "
            "~%s projected from the first page",
            self._source_id, self._operation, total, _fmt_bytes(projected),
        )
        if self._max and projected > self._max:
            logger.warning(
                "data source '%s' operation '%s': projected %s exceeds "
                "max_result_bytes (%s) -- the read will be truncated at the "
                "ceiling",
                self._source_id, self._operation,
                _fmt_bytes(projected), _fmt_bytes(self._max),
            )
        return projected

    async def add(self, value: Any) -> None:
        """Add one page of a paginated read, or the whole of an unpaginated one.

        A list is records; anything else -- a dict page the operation has no
        ``mapping`` or ``items_path`` for, a scalar -- is one record.  The
        distinction survives only to keep ``shape`` honest: one non-list add
        is a document, several are a list of pages.
        """
        if self._truncated:
            return
        if isinstance(value, list):
            items = value
            self._saw_list = True
        else:
            items = [value]
            self._scalar_adds += 1
        if not items:
            return
        if self._limit is not None:
            room = self._limit - self._items
            if room <= 0:
                return
            if len(items) > room:
                # The page that crosses the limit is trimmed rather than
                # dropped or kept whole: the caller asked for N rows and gets
                # exactly N.
                items = items[:room]
        writer = await self._ensure_writer()
        self._bytes += await writer.append_many(items)
        self._items += len(items)
        self._enforce_ceiling()

    def mark_truncated(self) -> None:
        """Flag the result a prefix — used when ``max_pages`` cut it short."""
        self._truncated = True

    async def finish(self) -> dict[str, Any]:
        """Close the stream and return the ref, in its state form."""
        writer = await self._ensure_writer()
        ref = await writer.close()
        ref.truncated = self._truncated
        ref.shape = self._final_shape()
        logger.info(
            "data source '%s' operation '%s': %d record(s), %s -> stream %s%s",
            self._source_id, self._operation, ref.items,
            _fmt_bytes(ref.bytes), ref.id,
            " (TRUNCATED)" if self._truncated else "",
        )
        return ref.to_state()

    async def abort(self) -> None:
        if self._writer is not None:
            await self._writer.abort()
            self._writer = None

    # -- internals ------------------------------------------------------
    async def _ensure_writer(self) -> StreamWriter:
        if self._writer is None:
            self._writer = await self._store.open_writer(
                source_id=self._source_id,
                operation=self._operation,
                shape=self._final_shape(),
            )
        return self._writer

    def _final_shape(self) -> str:
        """"value" only for exactly one non-list add and no list ever.

        One dict response is that dict, not a one-record list -- what
        ``result_mode: ram`` gives back and what every existing ``mapping``
        depends on.  Two dict pages are a list of pages; a list page is always
        records.
        """
        if self._saw_list or self._scalar_adds != 1:
            return "list"
        return "value"

    def _enforce_ceiling(self) -> None:
        if not self._max or self._bytes <= self._max:
            return
        self._truncated = True
        if self._final_shape() == "value":
            # A document has no prefix that is still valid JSON, so there is
            # nothing to truncate to.
            raise ResultTooLarge(
                f"data source '{self._source_id}' operation '{self._operation}': "
                f"single response is {_fmt_bytes(self._bytes)}, over "
                f"max_result_bytes ({_fmt_bytes(self._max)}). A single document "
                f"cannot be truncated -- narrow the request."
            )
        logger.warning(
            "data source '%s' operation '%s': reached max_result_bytes (%s) "
            "after %d record(s) -- the stream is a prefix and is flagged "
            "truncated",
            self._source_id, self._operation, _fmt_bytes(self._max), self._items,
        )


def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, default=str, ensure_ascii=False).encode("utf-8"))


def _fmt_bytes(count: int) -> str:
    from app.domain.models.datastream import _human_bytes

    return _human_bytes(count)
