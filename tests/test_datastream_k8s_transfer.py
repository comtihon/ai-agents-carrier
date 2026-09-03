"""Getting a stream to a sandbox in another pod.

The k8s sandbox shares nothing with the backend: seccomp denies ``socket`` so
it cannot fetch, and there is no common filesystem to mount.  So the backend
pushes the bytes into the pod's stdin and the pod writes its own file --
"streamed from one file to another".  These tests check the transfer and the
pod spec that makes it possible, without a cluster.
"""
from __future__ import annotations

import json
import tempfile
from typing import Any

import pytest

from app.infrastructure.datasources.datastream import LocalDiskStreamStore
from app.infrastructure.orchestration import script_sandbox

ROWS = [{"id": i, "v": "y" * 100} for i in range(5_000)]


@pytest.fixture
async def stream():
    store = LocalDiskStreamStore(tempfile.mkdtemp())
    writer = await store.open_writer(source_id="crm", operation="list_contacts")
    await writer.append_many(ROWS)
    return store, await writer.close()


class _Sink:
    """Stands in for the attach websocket's stdin."""

    def __init__(self) -> None:
        self.blocks: list[bytes] = []

    def write(self, block: bytes) -> int:
        self.blocks.append(block)
        return len(block)

    @property
    def data(self) -> bytes:
        return b"".join(self.blocks)


async def test_copy_to_transfers_every_byte_in_chunks(stream):
    store, ref = stream
    sink = _Sink()

    written = await store.copy_to(ref, sink)

    assert written == ref.bytes
    assert sink.data.decode().count("\n") == len(ROWS)
    assert json.loads(sink.data.decode().splitlines()[0]) == ROWS[0]
    # Chunked, so neither side ever holds the whole stream.
    assert len(sink.blocks) > 1
    assert max(len(b) for b in sink.blocks) <= 256 * 1024


async def test_the_attach_sink_forwards_blocks_to_stdin():
    class _WS:
        def __init__(self) -> None:
            self.writes: list[bytes] = []

        def write_stdin(self, block: bytes) -> None:
            self.writes.append(block)

    ws = _WS()
    sink = script_sandbox._AttachSink(ws)

    sink.write(b"abc")
    sink.write(b"de")

    assert ws.writes == [b"abc", b"de"]
    assert sink.written == 5


async def test_a_k8s_payload_names_stdin_rather_than_carrying_the_data(stream):
    """The ConfigMap must stay small: it carries code and state, never rows."""
    store, ref = stream
    captured: dict[str, Any] = {}

    async def _fake_run_k8s(payload, **kwargs):
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        return {"ok": True}

    original = script_sandbox._run_k8s
    script_sandbox._run_k8s = _fake_run_k8s
    try:
        async def _copy(sink):
            return await store.copy_to(ref, sink)

        await script_sandbox.run_script(
            "output = 1", {"a": 1}, runtime="k8s",
            stream_copy=_copy, stream_records=ref.items,
        )
    finally:
        script_sandbox._run_k8s = original

    payload = json.loads(captured["payload"])
    assert payload["data_from_stdin"] is True
    assert payload["data_dest"] == script_sandbox._K8S_DATA_DEST
    assert payload["stream_records"] == len(ROWS)
    # No records anywhere in it.
    assert "data_path" in payload and payload["data_path"] is None
    assert len(captured["payload"]) < 4096
    assert callable(captured["kwargs"]["stream_copy"])


def test_the_pod_spec_gives_the_data_a_disk_backed_volume():
    """Not tmpfs: a streamed file there would count against pod memory.

    Read off the module rather than by building a pod, so the assertion is
    about the constant the spec uses.
    """
    src = open(script_sandbox.__file__).read()
    data_volume = src[src.index('name="data"'):src.index('name="data"') + 400]

    assert "_K8S_DATA_LIMIT_MB" in data_volume
    assert 'medium="Memory"' not in data_volume
    # And the pod asks for the ephemeral storage it is about to use.
    assert '"ephemeral-storage"' in src


def test_stdin_is_only_opened_when_there_is_a_stream_to_send():
    """An ordinary script must not get a pod that blocks waiting on stdin."""
    src = open(script_sandbox.__file__).read()

    assert "stdin=stream_copy is not None" in src
    assert "stdin_once=stream_copy is not None" in src
