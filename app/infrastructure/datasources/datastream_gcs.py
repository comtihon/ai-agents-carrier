"""The data stream store, backed by a GCS bucket instead of the pod's disk.

Why this is a store and not a feature
-------------------------------------
Object storage is the *second* implementation of
:class:`~app.infrastructure.datasources.datastream.DataStreamStore`, selected by
``STREAM_BACKEND``.  It is deliberately not a separate place that download
artifacts live: a `data` step points at data some earlier step already wrote,
so the only question worth asking is where the store put it.  Giving downloads
their own bucket would mean copying a four-million-record result that is
already stored — exactly the thing the streaming design exists to avoid.

What changes against local disk
-------------------------------
``persistent`` is True: an object survives the pod, so a run that pauses for
approval overnight can still read its refs, and a download offered on Monday
still works on Tuesday.  ``local_path`` returns ``None`` (the bytes are not on
this filesystem), so every consumer that needs a path or a descriptor gets one
from a temporary local copy, and ``copy_to`` is the read path that needs no
copy at all.

Pinning is a custom metadata key on the object rather than a sidecar object:
one round trip to set, and it comes back with the listing the sweep already
does, so a sweep over a bucket does not become one HEAD per object.

The blocking client
-------------------
``google-cloud-storage`` is synchronous.  Every call below therefore goes
through ``asyncio.to_thread``, which is also what makes the queue-based
back-pressure in the download path work: ``copy_to``'s writes happen on a
worker thread, so a sink that blocks blocks the transfer rather than the event
loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from datetime import timedelta
from typing import Any, BinaryIO

from app.domain.models.datastream import PREVIEW_ITEMS, DataRef, shrink_preview
from app.infrastructure.datasources.datastream import (
    DataStreamStore,
    NotStreamable,
    StreamGone,
    StreamReadTooLarge,
    StreamWriter,
)

logger = logging.getLogger(__name__)

# Custom-metadata key marking an object exempt from the ordinary sweep. Its
# presence is what matters; the value is only there to be readable in a
# console.
PIN_METADATA_KEY = "carrier-pinned"

# Bytes per upload/download hop. Matches the local store's copy block so the
# two behave the same from a consumer's point of view.
_CHUNK_BYTES = 256 * 1024
# Lines pulled per thread hop when yielding parsed records.
_READ_BATCH_LINES = 500
# How long a signed download link stays valid. Long enough to survive a user
# noticing the download failed and clicking again, short enough that a link
# pasted into a chat is not a lasting credential.
SIGNED_URL_TTL = timedelta(minutes=15)


class _GcsWriter(StreamWriter):
    """Buffers one result to a local temp file, then uploads it once.

    A resumable upload per page would be the streaming-purist option, but a
    result arrives as many small pages and GCS charges a round trip per chunk;
    the temp file is written and dropped page by page exactly as the local
    store's is, so the backend's memory is still one page.  The pod's disk is
    the scratch space either way — the difference is that here it does not have
    to survive.
    """

    def __init__(
        self, store: "GcsStreamStore", stream_id: str, *,
        source_id: str, operation: str, shape: str,
    ) -> None:
        self._store = store
        self._id = stream_id
        self._source_id = source_id
        self._operation = operation
        self._shape = shape
        self._items = 0
        self._bytes = 0
        self._preview: list[Any] = []
        self._closed = False
        fd, self._temp = tempfile.mkstemp(prefix=f"{stream_id}-", suffix=".jsonl")
        self._fh = os.fdopen(fd, "w", encoding="utf-8")

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
            await asyncio.to_thread(self._store._upload, self._id, self._temp)
            await asyncio.to_thread(_unlink_quietly, self._temp)
        return DataRef(
            id=self._id,
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
        await asyncio.to_thread(_unlink_quietly, self._temp)


class GcsStreamStore(DataStreamStore):
    """Stream store backed by one GCS bucket (and optional key prefix)."""

    persistent = True

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        project: str = "",
        client: Any = None,
    ) -> None:
        if not bucket:
            raise ValueError(
                "STREAM_BACKEND=gcs needs STREAM_GCS_BUCKET; there is nowhere "
                "to write data source results otherwise"
            )
        self._bucket_name = bucket
        self._prefix = prefix.strip("/")
        # Passed to the storage client explicitly. A bare ``storage.Client()``
        # cannot determine the project under Workload Identity -- the
        # credential the metadata server returns carries none -- and raises
        # OSError("Project was not passed and could not be determined from the
        # environment"). Empty is allowed: the client then falls back to
        # GOOGLE_CLOUD_PROJECT and to the credential's own project, which is
        # what works off-cluster.
        self._project = project.strip()
        # Injected in tests; built lazily otherwise so importing this module
        # never needs credentials.
        self._client = client

    # -- plumbing -------------------------------------------------------
    def _bucket(self) -> Any:
        if self._client is None:
            from google.cloud import storage

            self._client = (
                storage.Client(project=self._project)
                if self._project
                else storage.Client()
            )
        return self._client.bucket(self._bucket_name)

    def check_ready(self) -> None:
        """Build the client and confirm the bucket is reachable.

        Called once at startup so a misconfigured store is a boot failure with
        the bucket named, rather than an OSError on the first data source call
        of the deployment -- which is how the missing project surfaced: the
        pod came up Healthy and the fault waited for the first run.
        """
        # Probed by listing one object, NOT by asking whether the bucket
        # exists. A bucket-existence check is storage.buckets.get, which
        # roles/storage.objectAdmin does not grant -- and objectAdmin is
        # exactly what the backend holds, correctly, for a store that only
        # ever reads and writes objects. So the existence check reported
        # "cannot reach gcs bucket ...: 403" against a store that was working
        # perfectly. A one-object listing needs storage.objects.list, which
        # objectAdmin does grant, and it proves the same three things:
        # credentials, network path, and that the bucket resolves.
        try:
            self._bucket()  # forces client construction
            next(iter(self._client.list_blobs(  # type: ignore[union-attr]
                self._bucket_name, max_results=1, prefix=self._prefix or None,
            )), None)
        except Exception as exc:  # noqa: BLE001 — re-raised with the cause named
            raise RuntimeError(
                f"data stream store: cannot reach gcs bucket "
                f"'{self._bucket_name}'"
                f"{f' in project {self._project}' if self._project else ''}: "
                f"{type(exc).__name__}: {exc}. Set STREAM_GCS_PROJECT (or "
                f"GOOGLE_CLOUD_PROJECT) when running under Workload Identity, "
                f"and check the backend service account holds "
                f"roles/storage.objectAdmin on the bucket."
            ) from exc


    def _key_for(self, stream_id: str) -> str:
        # Same refusal as the local store, and for the same reason: the id
        # becomes part of an object name, so a ref reaching us altered must not
        # be able to name another object (``..`` and a leading dot are both
        # rejected by the ``ds_`` shape ids are minted with, but the check is
        # what makes that a guarantee rather than a coincidence).
        if not stream_id or "/" in stream_id or stream_id.startswith("."):
            raise ValueError(f"invalid stream id {stream_id!r}")
        name = f"{stream_id}.jsonl"
        return f"{self._prefix}/{name}" if self._prefix else name

    def _blob(self, ref: DataRef) -> Any:
        return self._bucket().blob(self._key_for(ref.id))

    def _existing_blob(self, ref: DataRef) -> Any:
        blob = self._bucket().get_blob(self._key_for(ref.id))
        if blob is None:
            raise StreamGone(
                f"data stream '{ref.id}' is gone. It was swept on a TTL or "
                f"deleted; re-run the data source step that produced it."
            )
        return blob

    def _upload(self, stream_id: str, path: str) -> None:
        blob = self._bucket().blob(self._key_for(stream_id))
        blob.chunk_size = _CHUNK_BYTES
        blob.upload_from_filename(path, content_type="application/x-ndjson")

    # -- writing --------------------------------------------------------
    async def open_writer(
        self, *, source_id: str = "", operation: str = "", shape: str = "list"
    ) -> StreamWriter:
        stream_id = f"ds_{uuid.uuid4().hex}"
        return _GcsWriter(
            self, stream_id, source_id=source_id, operation=operation, shape=shape
        )

    # -- reading --------------------------------------------------------
    async def open_fd(self, ref: DataRef) -> int:
        """A descriptor over a temporary local copy of the object.

        The consumers that want a descriptor are the sandbox and the k8s
        transfer, and neither can reach the network — the seccomp filter denies
        ``socket``.  So the bytes have to be here before the filter goes on,
        and "here" means a temp file this pod owns.  The caller closes the
        descriptor; the file is unlinked immediately, so the last close frees
        the space.
        """
        blob = self._existing_blob(ref)

        def _fetch() -> int:
            fd, path = tempfile.mkstemp(prefix=f"{ref.id}-", suffix=".jsonl")
            try:
                with os.fdopen(os.dup(fd), "wb") as sink:
                    blob.download_to_file(sink)
                os.lseek(fd, 0, os.SEEK_SET)
            except BaseException:
                os.close(fd)
                _unlink_quietly(path)
                raise
            # Unlinked while open: the data stays reachable through the
            # descriptor and disappears the moment it is closed, so a consumer
            # that crashes cannot leave the file behind.
            _unlink_quietly(path)
            return fd

        return await asyncio.to_thread(_fetch)

    async def local_path(self, ref: DataRef) -> str | None:
        # Per the ABC: no path on this filesystem, so callers must use copy_to.
        return None

    async def copy_to(self, ref: DataRef, sink: BinaryIO) -> int:
        blob = self._existing_blob(ref)

        def _copy() -> int:
            before = getattr(sink, "tell", lambda: -1)()
            blob.download_to_file(sink)
            after = getattr(sink, "tell", lambda: -1)()
            if before >= 0 and after >= before:
                return after - before
            # A sink that is not seekable (a pipe, an HTTP body) cannot report
            # a position; the object's own size is the honest answer.
            return int(blob.size or 0)

        written = await asyncio.to_thread(_copy)
        logger.debug("data stream '%s': copied %d bytes out of GCS", ref.id, written)
        return written

    async def _iter_batches(
        self, ref: DataRef, *, start: int, limit: int | None
    ) -> AsyncIterator[list[Any]]:
        blob = self._existing_blob(ref)
        state: dict[str, Any] = {"fh": None, "index": 0, "yielded": 0}

        def _open() -> None:
            # A streaming text read: the client fetches in chunks, so a
            # four-million-record object is never held whole on either side.
            state["fh"] = blob.open("rt", encoding="utf-8", chunk_size=_CHUNK_BYTES)

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

    # -- lifecycle ------------------------------------------------------
    async def delete(self, ref: DataRef) -> None:
        try:
            blob = self._blob(ref)
        except ValueError:
            return

        def _delete() -> None:
            try:
                blob.delete()
            except Exception:  # noqa: BLE001 — already gone is the goal state
                logger.debug("data stream '%s': delete found nothing", ref.id)

        await asyncio.to_thread(_delete)

    async def pin(self, ref: DataRef) -> None:
        blob = self._existing_blob(ref)

        def _pin() -> None:
            metadata = dict(blob.metadata or {})
            metadata[PIN_METADATA_KEY] = str(int(time.time()))
            blob.metadata = metadata
            blob.patch()

        await asyncio.to_thread(_pin)

    async def unpin(self, ref: DataRef) -> None:
        try:
            key = self._key_for(ref.id)
        except ValueError:
            return

        def _unpin() -> None:
            blob = self._bucket().get_blob(key)
            if blob is None:
                return
            metadata = dict(blob.metadata or {})
            if PIN_METADATA_KEY not in metadata:
                return
            # None removes a key; the whole dict has to be re-sent because
            # patch merges rather than replaces.
            metadata[PIN_METADATA_KEY] = None
            blob.metadata = metadata
            blob.patch()

        await asyncio.to_thread(_unpin)

    async def is_pinned(self, ref: DataRef) -> bool:
        try:
            key = self._key_for(ref.id)
        except ValueError:
            return False

        def _check() -> bool:
            blob = self._bucket().get_blob(key)
            return bool(blob is not None and (blob.metadata or {}).get(PIN_METADATA_KEY))

        return await asyncio.to_thread(_check)

    async def purge_older_than(
        self, seconds: float, *, pinned_seconds: float | None = None
    ) -> int:
        def _purge() -> int:
            now = time.time()
            cutoff = now - seconds
            pinned_cutoff = None if pinned_seconds is None else now - pinned_seconds
            removed = 0
            listing = self._client.list_blobs(  # type: ignore[union-attr]
                self._bucket_name,
                prefix=f"{self._prefix}/ds_" if self._prefix else "ds_",
            )
            for blob in listing:
                pinned = bool((blob.metadata or {}).get(PIN_METADATA_KEY))
                limit = pinned_cutoff if pinned else cutoff
                if limit is None:
                    continue
                updated = blob.updated
                age_ok = updated is None or updated.timestamp() >= limit
                if age_ok:
                    continue
                try:
                    blob.delete()
                    removed += 1
                except Exception:  # noqa: BLE001 — another sweeper got there first
                    continue
            return removed

        def _purge_with_client() -> int:
            # Building the client is itself a blocking call -- credentials, and
            # under Workload Identity a metadata round trip. It used to happen
            # here on the event loop, one line outside the thread that was
            # carefully created for the listing. That was enough to wedge the
            # whole process whenever GCS was slow: the port stayed open while
            # /health and /ready timed out, the kubelet pulled the pod, and
            # nginx served 503. Nothing touching the network belongs on the
            # loop, the client construction included.
            self._bucket()  # listing needs the client itself, not a bucket handle
            return _purge()

        removed = await asyncio.to_thread(_purge_with_client)
        if removed:
            logger.info("data stream store (gcs): purged %d expired stream(s)", removed)
        return removed

    # -- signing --------------------------------------------------------
    async def signed_url(
        self, ref: DataRef, *, filename: str = "", content_type: str = ""
    ) -> str | None:
        """A V4 signed GET URL, or ``None`` when this deployment cannot sign.

        Not on the ABC: signing is a property of object storage, and the
        download route treats it as an optimisation — it redirects when a URL
        comes back and streams the bytes itself when it does not.

        Signing needs a credential that can *sign*.  A service-account key file
        carries a private key and signs locally; under Workload Identity there
        is no key, and the client falls back to the IAM ``signBlob`` API, which
        needs ``roles/iam.serviceAccountTokenCreator`` on the service account
        itself (or an ``impersonated_credentials`` wrapper).  Where neither is
        configured the client raises, and returning ``None`` here is what turns
        that into "stream it through the backend" rather than a failed
        download.
        """
        blob = self._existing_blob(ref)
        disposition = (
            f'attachment; filename="{filename}"' if filename else None
        )

        def _sign() -> str:
            return blob.generate_signed_url(
                version="v4",
                expiration=SIGNED_URL_TTL,
                method="GET",
                response_disposition=disposition,
                response_type=content_type or None,
            )

        try:
            return await asyncio.to_thread(_sign)
        except Exception as exc:  # noqa: BLE001 — every failure means "cannot sign"
            logger.info(
                "data stream '%s': cannot sign a download URL (%s) -- serving "
                "the bytes through the backend instead", ref.id, exc,
            )
            return None


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
