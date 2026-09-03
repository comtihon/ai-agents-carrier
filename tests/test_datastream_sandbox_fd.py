"""A sandboxed script reading a data source result off a file descriptor.

The mechanism this rests on: the seccomp filter denies ``openat`` but allows
``read``, ``lseek`` and ``fstat``.  So the bootstrap opens the stream file
*before* installing the filter, and the script inherits a descriptor it could
not have opened for itself.  That is what lets a script read a result far
larger than its own memory limit without weakening the sandbox -- the two
tests at the bottom check that the sandbox is still shut.
"""
from __future__ import annotations

import os
import platform
import tempfile

import pytest

from app.infrastructure.datasources.datastream import LocalDiskStreamStore
from app.infrastructure.orchestration.script_sandbox import run_script

pytestmark = pytest.mark.skipif(
    not (os.name == "posix" and platform.system() == "Linux"
         and platform.machine() == "x86_64"),
    reason="the seccomp filter targets Linux x86_64",
)

RECORDS = 20_000


@pytest.fixture
async def stream():
    store = LocalDiskStreamStore(tempfile.mkdtemp())
    writer = await store.open_writer(source_id="crm", operation="list_contacts")
    # Written in pages, as a paginated fetch would.
    for page in range(0, RECORDS, 5_000):
        await writer.append_many([
            {"id": i, "amount": i, "region": "eu" if i % 2 else "us",
             "blob": "x" * 200}
            for i in range(page, page + 5_000)
        ])
    ref = await writer.close()
    return store, ref


async def _run(store, ref, code: str, state: dict | None = None):
    return await run_script(
        code, state or {},
        runtime="local", timeout=120,
        stream_path=await store.local_path(ref),
        stream_records=ref.items,
        stream_truncated=ref.truncated,
    )


async def test_a_script_reads_the_whole_stream_off_the_descriptor(stream):
    store, ref = stream
    assert ref.bytes > 4 * 1024 * 1024, "fixture should be bigger than a chunk"

    result = await _run(store, ref, """
n = 0
total = 0
by_region = {}
for row in records():
    n += 1
    total += row["amount"]
    by_region[row["region"]] = by_region.get(row["region"], 0) + 1
output = {"count": n, "total": total, "by_region": by_region}
""")

    assert result["count"] == RECORDS
    assert result["total"] == sum(range(RECORDS))
    assert result["by_region"] == {"eu": RECORDS // 2, "us": RECORDS // 2}


async def test_the_script_is_told_how_many_records_to_expect(stream):
    store, ref = stream

    result = await _run(store, ref, """
output = {"declared": stream_records, "read": sum(1 for _ in records()),
          "truncated": stream_truncated}
""")

    assert result["declared"] == RECORDS
    assert result["read"] == RECORDS
    assert result["truncated"] is False


async def test_records_can_be_iterated_twice_without_a_refetch(stream):
    """It seeks back to the start, so a two-pass script needs one stream."""
    store, ref = stream

    result = await _run(store, ref, """
total = sum(r["amount"] for r in records())
mean = total / stream_records
output = sum(1 for r in records() if r["amount"] > mean)
""")

    assert result == RECORDS // 2


async def test_the_raw_file_object_is_available_for_a_script_that_wants_it(stream):
    store, ref = stream

    result = await _run(store, ref, """
stream.seek(0)
output = {"first_line_len": len(stream.readline()), "seekable": stream.seekable()}
""")

    assert result["seekable"] is True
    assert result["first_line_len"] > 0


async def test_a_script_with_no_stream_attached_is_told_so(stream):
    result = await run_script(
        'try:\n    output = sum(1 for _ in records())\nexcept Exception as e:\n'
        '    output = str(e)\n',
        {}, runtime="local", timeout=30,
    )

    assert "no data stream is attached" in result


# ---------------------------------------------------------------------------
# the sandbox is still shut
# ---------------------------------------------------------------------------

async def test_the_script_still_cannot_open_a_path_of_its_own(stream):
    """Handing over one descriptor must not hand over the filesystem."""
    store, ref = stream

    result = await _run(store, ref, """
try:
    open("/etc/passwd").read()
    output = {"blocked": False}
except Exception as exc:
    output = {"blocked": True, "error": type(exc).__name__}
""")

    assert result["blocked"] is True
    assert result["error"] == "PermissionError"


async def test_the_script_still_cannot_open_the_stream_file_by_name(stream):
    """Even the file it is reading is not openable -- only the fd is given."""
    store, ref = stream
    path = await store.local_path(ref)

    result = await _run(store, ref, f"""
try:
    open({path!r}).read()
    output = {{"blocked": False}}
except Exception as exc:
    output = {{"blocked": True, "error": type(exc).__name__}}
""")

    assert result["blocked"] is True


async def test_the_script_still_has_no_network(stream):
    store, ref = stream

    result = await _run(store, ref, """
try:
    import socket
    socket.socket()
    output = {"blocked": False}
except Exception as exc:
    output = {"blocked": True, "error": type(exc).__name__}
""")

    assert result["blocked"] is True
