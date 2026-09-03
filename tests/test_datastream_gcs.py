"""The GCS stream store: the same ABC over a bucket instead of pod disk.

Driven against a fake client rather than a real bucket, because what needs
proving is the store's own behaviour: that it satisfies the contract every
consumer already depends on, that ``local_path`` is ``None`` so reads go
through ``copy_to``, that a pin is metadata on the object the sweep already
lists, and that a deployment which cannot sign a URL gets a slower download
rather than a broken one.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.models.datastream import DataRef
from app.infrastructure.datasources.datastream import StreamGone
from app.infrastructure.datasources.datastream_gcs import (
    PIN_METADATA_KEY,
    GcsStreamStore,
)

ROWS = [{"id": i, "region": "eu" if i % 2 else "us"} for i in range(1, 21)]


class _FakeBlob:
    def __init__(self, bucket: "_FakeBucket", name: str) -> None:
        self._bucket = bucket
        self.name = name
        self.metadata: dict | None = None
        self.updated: datetime | None = None
        self.chunk_size: int | None = None
        self.content_type = ""
        self.signable = True

    # -- data ------------------------------------------------------------
    @property
    def _data(self) -> bytes:
        return self._bucket.objects[self.name]

    @property
    def size(self) -> int:
        return len(self._bucket.objects.get(self.name, b""))

    def upload_from_filename(self, path: str, content_type: str = "") -> None:
        with open(path, "rb") as fh:
            self._bucket.objects[self.name] = fh.read()
        self.content_type = content_type
        self.updated = datetime.now(timezone.utc)
        self._bucket.blobs[self.name] = self

    def download_to_file(self, sink) -> None:
        sink.write(self._data)

    def open(self, mode: str, encoding: str = "utf-8", chunk_size: int | None = None):
        assert mode == "rt"
        return io.StringIO(self._data.decode(encoding))

    def delete(self) -> None:
        if self.name not in self._bucket.objects:
            raise FileNotFoundError(self.name)
        del self._bucket.objects[self.name]
        self._bucket.blobs.pop(self.name, None)

    def patch(self) -> None:
        # GCS merges a metadata patch and treats a None value as a removal.
        stored = self._bucket.blobs.setdefault(self.name, self)
        merged = dict(stored.metadata or {})
        for key, value in (self.metadata or {}).items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        stored.metadata = merged
        self.metadata = merged

    def generate_signed_url(self, **kwargs) -> str:
        if not self.signable:
            raise RuntimeError(
                "you need a private key to sign credentials -- see the docs"
            )
        self._bucket.signed_with.append(kwargs)
        return f"https://storage.example/{self.name}?X-Goog-Signature=abc"


class _FakeBucket:
    def __init__(self, client: "_FakeClient", name: str) -> None:
        self.client = client
        self.name = name
        self.objects: dict[str, bytes] = client.objects
        self.blobs: dict[str, _FakeBlob] = client.blobs
        self.signed_with: list = client.signed_with

    def blob(self, name: str) -> _FakeBlob:
        blob = _FakeBlob(self, name)
        blob.signable = self.client.signable
        existing = self.blobs.get(name)
        if existing is not None:
            blob.metadata = dict(existing.metadata or {})
            blob.updated = existing.updated
        return blob

    def exists(self) -> bool:
        # The real bucket has this; check_ready calls it to prove the bucket
        # is reachable rather than merely nameable.
        return True

    def get_blob(self, name: str) -> _FakeBlob | None:
        if name not in self.objects:
            return None
        return self.blobs.get(name) or self.blob(name)


class _FakeClient:
    def __init__(self, *, signable: bool = True) -> None:
        self.objects: dict[str, bytes] = {}
        self.blobs: dict[str, _FakeBlob] = {}
        self.signed_with: list = []
        self.signable = signable

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self, name)

    def list_blobs(self, bucket: str, prefix: str = ""):
        return [b for name, b in list(self.blobs.items()) if name.startswith(prefix)]


@pytest.fixture
def client() -> _FakeClient:
    return _FakeClient()


@pytest.fixture
def store(client) -> GcsStreamStore:
    return GcsStreamStore("carrier-test", prefix="streams", client=client)


async def _written(store, rows=ROWS, *, shape: str = "list") -> DataRef:
    writer = await store.open_writer(source_id="crm", operation="list", shape=shape)
    await writer.append_many(rows)
    return await writer.close()


# ---------------------------------------------------------------------------
# The ABC contract
# ---------------------------------------------------------------------------

async def test_a_result_round_trips_through_the_bucket(store, client):
    ref = await _written(store)

    assert ref.items == len(ROWS)
    assert f"streams/{ref.id}.jsonl" in client.objects
    assert await store.read_all(ref, max_bytes=0) == ROWS
    assert [r async for r in store.stream(ref, limit=2)] == ROWS[:2]


async def test_local_path_is_none_so_reads_go_through_copy_to(store):
    """Per the ABC: the bytes are not on this filesystem."""
    ref = await _written(store)
    assert await store.local_path(ref) is None

    sink = io.BytesIO()
    written = await store.copy_to(ref, sink)

    assert written == ref.bytes
    assert sink.getvalue().count(b"\n") == len(ROWS)


async def test_an_fd_is_offered_over_a_temporary_local_copy(store):
    """The sandbox cannot reach the network, so the bytes must be here first."""
    import os

    ref = await _written(store)
    fd = await store.open_fd(ref)
    try:
        assert os.read(fd, 6) == b'{"id":'
    finally:
        os.close(fd)


async def test_a_missing_object_raises_stream_gone(store):
    ref = await _written(store)
    await store.delete(ref)
    with pytest.raises(StreamGone):
        await store.copy_to(ref, io.BytesIO())


async def test_the_store_is_marked_persistent(store):
    """Which is the whole reason to choose it over pod disk."""
    assert store.persistent is True


async def test_a_tampered_stream_id_cannot_name_another_object(store):
    with pytest.raises(ValueError):
        store._key_for("../../secrets")
    with pytest.raises(ValueError):
        store._key_for(".hidden")


async def test_a_bucketless_configuration_is_refused_at_construction():
    with pytest.raises(ValueError) as exc:
        GcsStreamStore("")
    assert "STREAM_GCS_BUCKET" in str(exc.value)


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

async def test_pinning_is_metadata_on_the_object(store, client):
    ref = await _written(store)
    await store.pin(ref)

    assert await store.is_pinned(ref) is True
    assert PIN_METADATA_KEY in (client.blobs[f"streams/{ref.id}.jsonl"].metadata or {})

    await store.unpin(ref)
    assert await store.is_pinned(ref) is False


async def test_a_pinned_object_survives_the_ordinary_sweep(store):
    pinned = await _written(store)
    loose = await _written(store)
    await store.pin(pinned)
    _age(store, pinned, hours=48)
    _age(store, loose, hours=48)

    removed = await store.purge_older_than(3600, pinned_seconds=7 * 24 * 3600)

    assert removed == 1
    assert await store.local_path(pinned) is None  # still there to be read
    await store.copy_to(pinned, io.BytesIO())
    with pytest.raises(StreamGone):
        await store.copy_to(loose, io.BytesIO())


async def test_a_pinned_object_is_swept_once_its_own_window_passes(store):
    ref = await _written(store)
    await store.pin(ref)
    _age(store, ref, hours=24 * 30)

    assert await store.purge_older_than(3600, pinned_seconds=7 * 24 * 3600) == 1


def _age(store, ref, *, hours: float) -> None:
    """Backdate an object so the sweep considers it old."""
    key = f"streams/{ref.id}.jsonl"
    blob = store._client.blobs[key]
    blob.updated = datetime.now(timezone.utc) - timedelta(hours=hours)


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------

async def test_a_signed_url_is_returned_when_the_credential_can_sign(store, client):
    ref = await _written(store)

    url = await store.signed_url(
        ref, filename="open_projects.jsonl", content_type="application/x-ndjson"
    )

    assert url.startswith("https://storage.example/")
    kwargs = client.signed_with[-1]
    assert kwargs["version"] == "v4"
    assert kwargs["method"] == "GET"
    assert kwargs["response_disposition"] == 'attachment; filename="open_projects.jsonl"'


async def test_a_credential_that_cannot_sign_falls_back_rather_than_failing():
    """Workload Identity without SignBlob: slower download, not a broken one."""
    client = _FakeClient(signable=False)
    store = GcsStreamStore("carrier-test", prefix="streams", client=client)
    ref = await _written(store)

    assert await store.signed_url(ref, filename="x.jsonl") is None


async def test_signing_a_missing_object_still_reports_it_as_gone(store):
    ref = await _written(store)
    await store.delete(ref)
    with pytest.raises(StreamGone):
        await store.signed_url(ref)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_the_default_backend_is_still_local_disk():
    """An unconfigured deployment must behave exactly as it did before."""
    from app.core.config import Settings
    from app.core.container import _build_stream_store
    from app.infrastructure.datasources.datastream import LocalDiskStreamStore

    assert isinstance(_build_stream_store(Settings()), LocalDiskStreamStore)


def test_an_unknown_backend_is_refused_rather_than_defaulted():
    """A typo must not put a deployment that meant to be durable on pod disk."""
    from app.core.config import Settings
    from app.core.container import _build_stream_store

    with pytest.raises(ValueError) as exc:
        _build_stream_store(Settings(stream_backend="gcs-typo"))
    assert "gcs-typo" in str(exc.value)


def test_selecting_gcs_builds_the_gcs_store(monkeypatch):
    from app.core.config import Settings
    from app.core.container import _build_stream_store

    monkeypatch.setattr(
        "google.cloud.storage.Client", lambda *a, **k: _FakeClient(), raising=False
    )
    store = _build_stream_store(
        Settings(stream_backend="gcs", stream_gcs_bucket="b", stream_gcs_prefix="p")
    )
    assert isinstance(store, GcsStreamStore)
    assert store._key_for("ds_1") == "p/ds_1.jsonl"


# ---------------------------------------------------------------------------
# the project the client is built with
# ---------------------------------------------------------------------------
#
# Under Workload Identity a bare ``storage.Client()`` cannot determine the
# project: the credential the metadata server returns carries none, so the
# client raises OSError("Project was not passed and could not be determined
# from the environment"). That happened in prod -- and because the only caller
# at boot was the TTL purge, whose exception is caught and logged, the pod came
# up Healthy and the fault waited for the first data source call of the
# deployment. These tests pin both halves of the fix: the project reaches the
# client, and a store that cannot reach its bucket fails at startup instead.

def _fake_storage_module(monkeypatch, record: dict):
    """Replace ``google.cloud.storage`` with a recorder.

    Patched as an attribute of the real ``google.cloud`` package, because that
    is what ``from google.cloud import storage`` resolves to once the genuine
    package is importable -- a sys.modules entry alone is bypassed.
    """
    import types

    import google.cloud

    module = types.ModuleType("google.cloud.storage")

    def _client(**kwargs):
        record["kwargs"] = kwargs
        record["called"] = True
        return _FakeClient()

    module.Client = _client
    monkeypatch.setattr(google.cloud, "storage", module, raising=False)
    return module


def test_the_project_is_passed_to_the_storage_client(monkeypatch):
    """The fix for the prod OSError: the project must reach the client."""
    record: dict = {}
    _fake_storage_module(monkeypatch, record)

    GcsStreamStore("carrier-test", project="some-gcp-project")._bucket()

    assert record["kwargs"] == {"project": "some-gcp-project"}


def test_no_project_configured_leaves_the_client_to_work_it_out(monkeypatch):
    """Off-cluster the ambient credential does carry one; do not override it."""
    record: dict = {}
    _fake_storage_module(monkeypatch, record)

    GcsStreamStore("carrier-test")._bucket()

    assert record["called"] is True
    assert record["kwargs"] == {}, "expected Client() with no project kwarg"


def test_the_project_is_never_defaulted_to_a_literal_in_this_repo():
    """Which project is deployment configuration, not application knowledge.

    It arrives as STREAM_GCS_PROJECT from the infrastructure repo. A default
    baked in here would silently point one environment at another's bucket.
    """
    from app.core.config import Settings

    field = Settings.model_fields["stream_gcs_project"]

    assert field.default == ""
    assert field.alias == "STREAM_GCS_PROJECT"


async def test_check_ready_passes_when_the_bucket_is_reachable(client):
    store = GcsStreamStore("carrier-test", client=client)

    store.check_ready()  # must not raise


def test_check_ready_names_the_bucket_when_the_client_cannot_be_built():
    class _Broken:
        def bucket(self, name):
            raise OSError(
                "Project was not passed and could not be determined from the "
                "environment."
            )

    store = GcsStreamStore("carrier-test", client=_Broken())

    with pytest.raises(RuntimeError) as exc:
        store.check_ready()

    message = str(exc.value)
    assert "carrier-test" in message
    assert "STREAM_GCS_PROJECT" in message
    assert "Project was not passed" in message


def test_check_ready_refuses_a_bucket_that_is_not_there():
    class _Missing:
        def bucket(self, name):
            class _B:
                @staticmethod
                def exists():
                    return False
            return _B()

    store = GcsStreamStore("carrier-test", client=_Missing())

    with pytest.raises(RuntimeError, match="does not exist or is not visible"):
        store.check_ready()
