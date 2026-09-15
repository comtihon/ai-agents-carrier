"""Guards for reading an artifact's bytes through the management MCP.

The behaviour that matters is paging: the reason this exists is that an MCP
client could see an artifact and never its content, and a paging bug that
silently returned a prefix would recreate exactly the failure it was built to
fix -- a partial answer that looks whole.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from app.application import management_tools as core


@dataclass
class _Artifact:
    id: str = "a1"
    run_id: str = "r1"
    step_id: str = "s1"
    name: str = "report"
    origin: str = "data"
    format: str = "json"
    filename: str = "report.json"
    shape: str = "value"
    items: int = 1
    bytes: int = 42
    truncated: bool = False
    source_id: str = ""
    operation: str = ""
    created_at: datetime = datetime(2026, 9, 15, tzinfo=UTC)
    expires_at: datetime = datetime(2026, 9, 22, tzinfo=UTC)

    def as_ref(self):
        return object()


class _Download:
    def __init__(self, blocks):
        self._blocks = blocks

    @property
    def chunks(self):
        async def _gen():
            for b in self._blocks:
                yield b
        return _gen()


class _Runs:
    def __init__(self, run=object()):
        self._run = run

    async def get(self, run_id):
        return self._run if run_id == "r1" else None


def _deps(store=object(), artifact=None, run=object()):
    return core.ManagementDeps(
        registry=None,
        run_repository=_Runs(run),
        data_artifact_backend=object(),
        stream_store=store,
    )


@pytest.fixture
def patched(monkeypatch):
    """Point the core at a fake artifact and a fake byte stream."""
    state = {"artifact": _Artifact(), "blocks": [b'{"projects": 4663}']}

    async def _find(backend, run, artifact_id, datasource_ttl_seconds=0):
        return state["artifact"]

    async def _prepare(store, artifact):
        return _Download(state["blocks"])

    monkeypatch.setattr(
        "app.application.data_artifacts.find_run_artifact", _find
    )
    monkeypatch.setattr(
        "app.application.data_artifacts.prepare_download", _prepare
    )
    return state


@pytest.mark.asyncio
async def test_returns_the_content(patched):
    out = await core.read_run_data_artifact(_deps(), "r1", "a1")
    assert '{"projects": 4663}' in out


@pytest.mark.asyncio
async def test_unknown_run_is_reported(patched):
    out = await core.read_run_data_artifact(_deps(run=None), "nope", "a1")
    assert "not found" in out


@pytest.mark.asyncio
async def test_missing_store_is_reported(patched):
    out = await core.read_run_data_artifact(_deps(store=None), "r1", "a1")
    assert "no data stream store" in out.lower()


@pytest.mark.asyncio
async def test_long_content_is_paged_and_says_so(patched):
    patched["blocks"] = [b"x" * 1000]
    out = await core.read_run_data_artifact(_deps(), "r1", "a1", offset=0, limit=100)

    assert "More content follows" in out
    assert "offset=100" in out
    body = out.split("---\n", 1)[1]
    assert body == "x" * 100


@pytest.mark.asyncio
async def test_offset_skips_exactly_that_many_bytes(patched):
    patched["blocks"] = [b"0123456789"]
    out = await core.read_run_data_artifact(_deps(), "r1", "a1", offset=4, limit=3)
    assert out.split("---\n", 1)[1] == "456"


@pytest.mark.asyncio
async def test_offset_spanning_a_chunk_boundary(patched):
    """The window must not depend on how the store happens to block the data."""
    patched["blocks"] = [b"abc", b"def", b"ghi"]
    out = await core.read_run_data_artifact(_deps(), "r1", "a1", offset=2, limit=4)
    assert out.split("---\n", 1)[1] == "cdef"


@pytest.mark.asyncio
async def test_concatenating_pages_reproduces_the_whole(patched):
    """The property that actually matters for a caller."""
    patched["blocks"] = [b"abcde", b"fghij", b"klmno"]
    whole = ""
    offset = 0
    for _ in range(10):
        out = await core.read_run_data_artifact(
            _deps(), "r1", "a1", offset=offset, limit=4
        )
        whole += out.split("---\n", 1)[1]
        if "More content follows" not in out:
            break
        offset += 4
    assert whole == "abcdefghijklmno"


@pytest.mark.asyncio
async def test_truncated_artifact_is_flagged(patched):
    patched["artifact"] = _Artifact(truncated=True)
    out = await core.read_run_data_artifact(_deps(), "r1", "a1")
    assert "truncated prefix" in out


@pytest.mark.asyncio
async def test_swept_bytes_are_distinguished_from_a_missing_artifact(
    patched, monkeypatch
):
    from app.infrastructure.datasources.datastream import StreamGone

    async def _gone(store, artifact):
        raise StreamGone("swept")

    monkeypatch.setattr(
        "app.application.data_artifacts.prepare_download", _gone
    )
    out = await core.read_run_data_artifact(_deps(), "r1", "a1")
    assert "gone" in out.lower()


@pytest.mark.asyncio
async def test_limit_is_capped(patched):
    """A caller cannot ask for an unbounded read."""
    patched["blocks"] = [b"y" * 500_000]
    out = await core.read_run_data_artifact(
        _deps(), "r1", "a1", limit=10_000_000
    )
    body = out.split("---\n", 1)[1]
    assert len(body) == core._ARTIFACT_READ_LIMIT
