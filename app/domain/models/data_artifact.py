"""One named, downloadable piece of a run's data.

A `data` step does not move data.  It names data that already exists — a
``DataRef`` some earlier step produced, or a plain state value — and records
*that it was named*, so the run UI can offer it as a file afterwards.  This
model is that record: an entry in the run's download manifest.

Why it is out of band
---------------------
The manifest lives in its own collection, keyed by ``(run_id, step_id)``, and
never in workflow state.  State goes into a LangGraph checkpoint and a Mongo
run document, so anything that grows with the number of things a workflow
exports would make both a function of that number.  The same reasoning that
keeps a result out of state keeps its manifest out of state.

What it does *not* hold
-----------------------
Bytes.  ``stream_id`` points at the stream the store already has, pinned for
the artifact's lifetime; serving a download is a read of that stream in the
requested ``format``.  Re-uploading data that is already stored is the mistake
the streaming work exists to prevent.

Two origins, one shape
----------------------
A `data` step's selection is one kind of entry; a data source result already
sitting in a run's state is the other.  The second is *synthesised* on demand
rather than stored, and is described in exactly the same shape, so downloading
a datasource result is the same operation as downloading a curated export
rather than a second mechanism with its own reader.  ``origin`` says which it
is, and it is the only difference a caller can see.  Curated exports are pinned
and offered for the artifact TTL; a datasource result is not pinned and its
``expires_at`` says so, because nothing promised it would still be there
tomorrow.

``truncated`` is copied off the ref and carried all the way to the download UI.
Somebody downloading a truncated prefix in the belief that it is the complete
answer is the one failure here that does real damage, so it travels with every
representation of an artifact and is never inferred.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# Serialisations a selection can be offered in. `jsonl` is the store's own
# form, so it is served untouched; the others are streamed transforms of it.
ArtifactFormat = Literal["jsonl", "json", "csv"]
FORMATS: tuple[str, ...] = ("jsonl", "json", "csv")

# What a browser is told it is receiving.
CONTENT_TYPES: dict[str, str] = {
    "jsonl": "application/x-ndjson",
    "json": "application/json",
    "csv": "text/csv; charset=utf-8",
}


def new_artifact_id() -> str:
    """A fresh artifact id.

    Random rather than derived from ``(run_id, step_id, name)``: a `data` node
    inside a loop runs many times and each pass is its own artifact, so an id
    that is a function of the step could only collide with the previous pass.
    """
    return f"art_{uuid4().hex[:12]}"


class DataArtifact(BaseModel):
    """A run's manifest entry for one named selection of a `data` step."""

    id: str = Field(default_factory=new_artifact_id)
    run_id: str
    step_id: str
    # Where the entry came from. ``data_node`` is a curated export: somebody
    # wrote a `data` step naming this data, and it is pinned and offered for as
    # long as the artifact TTL. ``datasource`` is a result already sitting in
    # the run's state, described in the same shape so that downloading one is
    # the same operation -- but not pinned, because nobody promised it would
    # still be there tomorrow.
    origin: Literal["data_node", "datasource"] = "data_node"
    # The name the workflow author gave the selection; also the filename stem.
    name: str
    format: ArtifactFormat = "jsonl"
    # The pinned stream holding the bytes. Not a path and not a URL: which
    # store it lives in is the store's business, and a local-disk deployment
    # and a GCS one produce the same manifest.
    stream_id: str
    shape: Literal["list", "value"] = "list"
    items: int = 0
    # Size of the *stored* JSONL payload. `jsonl` is served as-is so this is
    # its Content-Length; `json` and `csv` are streamed transforms whose served
    # size is not known until the last record, which is why the download sends
    # no length for them rather than a wrong one.
    bytes: int = 0
    truncated: bool = False
    source_id: str = ""
    operation: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # When the pinned stream falls out of the artifact retention window. Shown
    # to the user, so "download it before this" is a thing they can act on.
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"populate_by_name": True}

    @classmethod
    def for_stream(
        cls,
        *,
        run_id: str,
        step_id: str,
        name: str,
        fmt: str,
        ref: Any,
        ttl_seconds: float,
    ) -> "DataArtifact":
        """Build an entry describing *ref* served as *fmt*."""
        created = datetime.now(timezone.utc)
        return cls(
            run_id=run_id,
            step_id=step_id,
            name=name,
            format=fmt,  # type: ignore[arg-type]
            stream_id=ref.id,
            shape=ref.shape,
            items=ref.items,
            bytes=ref.bytes,
            truncated=ref.truncated,
            source_id=ref.source_id,
            operation=ref.operation,
            created_at=created,
            expires_at=created + timedelta(seconds=max(0.0, ttl_seconds)),
        )

    @classmethod
    def for_datasource_ref(
        cls, *, run_id: str, state_key: str, ref: Any, ttl_seconds: float
    ) -> "DataArtifact":
        """A manifest entry describing a data source result in a run's state.

        Synthesised rather than stored: the ref *is* the record, so a row of
        our own would only be a second copy of counts that could drift from it.
        The id is derived from the stream id so the download can find its way
        back to the same ref -- through the run's state, never by opening a
        stream id it was handed.

        ``expires_at`` is the ordinary spill TTL, not the artifact one, because
        this stream is not pinned.  Saying so is the point: an agent that comes
        back for it in a day gets the same 410 a swept data-node artifact does,
        and the manifest said when that would start.
        """
        created = getattr(ref, "created_at", None) or datetime.now(timezone.utc)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        origin_name = f"{ref.source_id}.{ref.operation}".strip(".")
        return cls(
            id=datasource_artifact_id(ref.id),
            run_id=run_id,
            step_id=state_key,
            origin="datasource",
            name=origin_name or state_key,
            # The stored form. A data source result has no author-declared
            # format to honour, and jsonl is what the bytes already are.
            format="jsonl",
            stream_id=ref.id,
            shape=ref.shape,
            items=ref.items,
            bytes=ref.bytes,
            truncated=ref.truncated,
            source_id=ref.source_id,
            operation=ref.operation,
            created_at=created,
            expires_at=created + timedelta(seconds=max(0.0, ttl_seconds)),
        )

    @property
    def filename(self) -> str:
        """What the file is called when it lands in the user's downloads."""
        return f"{_safe_stem(self.name)}.{self.format}"

    @property
    def content_type(self) -> str:
        return CONTENT_TYPES.get(self.format, "application/octet-stream")

    def as_ref(self) -> Any:
        """The ``DataRef`` needed to read this artifact's bytes back.

        Rebuilt from the manifest entry rather than stored as a nested ref: the
        entry already carries every field a read needs, and one copy of the
        counts cannot drift from another.
        """
        from app.domain.models.datastream import DataRef

        return DataRef(
            id=self.stream_id,
            shape=self.shape,
            items=self.items,
            bytes=self.bytes,
            truncated=self.truncated,
            source_id=self.source_id,
            operation=self.operation,
        )

    def manifest_entry(self, *, download_url: str | None = None) -> dict[str, Any]:
        """The wire form the run-data API and the MCP tools both return.

        One renderer, so the REST manifest, the MCP tool and anything added
        later cannot describe the same artifact differently.

        ``download_url`` defaults to this backend's own endpoint, expressed
        relative to the API root — which is what the frontend joins against its
        base URL and sends its bearer token to.  It is overridden with an
        absolute signed URL where the store can sign one: a redirect to a
        cross-origin signed target would be followed by ``fetch`` and fail
        CORS, whereas an absolute URL in the manifest is simply opened.
        """
        return {
            "id": self.id,
            "origin": self.origin,
            "step_id": self.step_id,
            "name": self.name,
            "format": self.format,
            "filename": self.filename,
            "shape": self.shape,
            "items": self.items,
            "bytes": self.bytes,
            "truncated": self.truncated,
            "source_id": self.source_id,
            "operation": self.operation,
            "created_at": _iso_z(self.created_at),
            "expires_at": _iso_z(self.expires_at),
            "download_url": download_url or self.download_path,
        }

    @property
    def download_path(self) -> str:
        """This backend's own download endpoint, relative to the API root."""
        return f"/runs/{self.run_id}/data/{self.id}"


# Prefix marking an artifact id that is a data source result rather than a
# stored manifest row. The rest of the id is the stream id, which is what lets
# a download resolve it inside the run's own state.
DATASOURCE_ID_PREFIX = "dsr_"


def datasource_artifact_id(stream_id: str) -> str:
    return f"{DATASOURCE_ID_PREFIX}{stream_id}"


def stream_id_of_datasource_artifact(artifact_id: str) -> str | None:
    """The stream id inside a datasource artifact id, or ``None``.

    Only ever used to *match* against a ref found in a run's state — never
    handed to the store on its own. That distinction is the whole security
    property: a stream id the caller supplies must not be able to name a file.
    """
    if not artifact_id.startswith(DATASOURCE_ID_PREFIX):
        return None
    return artifact_id[len(DATASOURCE_ID_PREFIX):] or None


def _safe_stem(name: str) -> str:
    """A selection name reduced to something safe as a filename.

    The name reaches a ``Content-Disposition`` header, where a quote or a
    newline is a header-injection primitive and a slash is a path.  Anything
    outside a conservative set becomes an underscore rather than being
    rejected: a `data` step exists to observe, so a badly named selection
    should still be downloadable.
    """
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-_.") else "_" for ch in (name or "")
    ).strip("._")
    return cleaned or "data"


def _iso_z(value: datetime) -> str:
    """UTC ISO-8601 with a ``Z``, which is what the frontend parses."""
    aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
