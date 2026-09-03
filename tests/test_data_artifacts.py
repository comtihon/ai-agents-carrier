"""The `data` step: naming data mid-workflow so it can be downloaded later.

Three properties carry the whole design and each has a test that would fail
loudly if it broke:

* the step returns state *byte-identical* — it observes, it does not transform;
* a selection that resolves to a ``DataRef`` copies **no bytes**, because the
  bytes are already in the store;
* a pinned stream survives the ordinary retention sweep, because the sweep
  would otherwise delete exactly the file a user came back for.

Follows ``test_datastream_steps.py``: the node functions are driven directly,
with a real ``LocalDiskStreamStore`` on ``tmp_path`` and the in-memory manifest
backend.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.application import data_artifacts
from app.application.data_artifacts import (
    CSV_HEADER_SCAN_ITEMS,
    DataArtifactError,
    Selection,
    parse_selections,
    prepare_download,
)
from app.domain.models.data_artifact import DataArtifact
from app.domain.models.graph_run import GraphRun
from app.infrastructure.datasources.datastream import LocalDiskStreamStore, StreamGone
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.persistence.data_artifact_backend import (
    InMemoryDataArtifactBackend,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider

ROWS = [{"id": i, "amount": i * 10, "region": "eu" if i % 2 else "us"}
        for i in range(1, 51)]

_TTL = 7 * 24 * 3600.0


@pytest.fixture
def store(tmp_path) -> LocalDiskStreamStore:
    return LocalDiskStreamStore(tmp_path / "streams")


@pytest.fixture
def backend() -> InMemoryDataArtifactBackend:
    return InMemoryDataArtifactBackend()


async def _streamed(store, rows=ROWS, *, truncated: bool = False, shape: str = "list"):
    writer = await store.open_writer(
        source_id="google-sheets", operation="read_open_projects", shape=shape
    )
    await writer.append_many(rows)
    ref = await writer.close()
    ref.truncated = truncated
    return ref


def _runner(steps, store=None, backend=None, run_id: str = "run_abc") -> YamlGraphRunner:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="answer")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner({"id": "g", "steps": steps}, llm=llm, mcp_tools_provider=mcp)
    runner._stream_store = store
    runner._data_artifact_backend = backend
    if run_id:
        runner._current_run = GraphRun(
            id=run_id, graph_id="g", user_request="hi", status="running"
        )
    return runner


def _step(*selections, step_id: str = "export_rows") -> dict:
    return {"id": step_id, "type": "data", "selections": list(selections)}


# ---------------------------------------------------------------------------
# The step does not touch state
# ---------------------------------------------------------------------------

async def test_state_comes_back_byte_identical(store, backend):
    """The one invariant a `data` node must never break."""
    ref = await _streamed(store)
    step = _step(
        {"name": "open_projects", "from": "state.projects", "format": "jsonl"},
        {"name": "summary", "from": "state.summary", "format": "json"},
    )
    state = {"projects": ref.to_state(), "summary": {"total": 3, "open": 2}}
    before = json.dumps(state, sort_keys=True, default=str)

    out = await _runner([step], store, backend)._data_node(step)(state)

    # No update at all: not "an update that happens to be equal", which a
    # reducer could still turn into a new checkpoint value.
    assert out == {}
    # And the dict it was handed is untouched, keys and values alike.
    assert json.dumps(state, sort_keys=True, default=str) == before
    # The artifacts were still recorded — the step did its job without
    # rewriting anything.
    assert len(await backend.list_for_run("run_abc")) == 2


async def test_the_referenced_data_is_not_transformed(store, backend):
    """Recording a selection must not rewrite the stream it points at."""
    ref = await _streamed(store)
    path = await store.local_path(ref)
    before = open(path, "rb").read()

    step = _step({"name": "rows", "from": "state.projects"})
    await _runner([step], store, backend)._data_node(step)({"projects": ref.to_state()})

    assert open(path, "rb").read() == before


# ---------------------------------------------------------------------------
# A DataRef costs nothing
# ---------------------------------------------------------------------------

async def test_a_data_ref_selection_copies_no_bytes(store, backend, tmp_path):
    """The bytes are already stored. Re-uploading them is the whole mistake."""
    ref = await _streamed(store)
    streams_before = sorted(p.name for p in (tmp_path / "streams").glob("ds_*.jsonl"))

    step = _step({"name": "open_projects", "from": "state.projects"})
    await _runner([step], store, backend)._data_node(step)({"projects": ref.to_state()})

    streams_after = sorted(p.name for p in (tmp_path / "streams").glob("ds_*.jsonl"))
    assert streams_after == streams_before, "a second stream file means a copy"

    artifact = (await backend.list_for_run("run_abc"))[0]
    # The manifest entry points at the stream that already existed.
    assert artifact.stream_id == ref.id
    assert artifact.items == len(ROWS)
    assert artifact.source_id == "google-sheets"
    assert artifact.operation == "read_open_projects"


async def test_a_plain_value_is_written_to_a_new_stream(store, backend, tmp_path):
    """A non-ref state value has no stream yet, so one is created for it."""
    step = _step({"name": "summary", "from": "state.summary", "format": "json"})
    state = {"summary": {"total": 3, "open": 2}}

    await _runner([step], store, backend)._data_node(step)(state)

    artifact = (await backend.list_for_run("run_abc"))[0]
    assert (tmp_path / "streams" / f"{artifact.stream_id}.jsonl").exists()
    # A dict is one document, not a one-record list.
    assert artifact.shape == "value"
    assert artifact.items == 1


async def test_a_plain_list_becomes_one_record_per_line(store, backend):
    step = _step({"name": "rows", "from": "state.rows"})
    await _runner([step], store, backend)._data_node(step)({"rows": ROWS})

    artifact = (await backend.list_for_run("run_abc"))[0]
    assert artifact.shape == "list"
    assert artifact.items == len(ROWS)
    assert await store.read_all(artifact.as_ref(), max_bytes=0) == ROWS


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

async def test_a_pinned_stream_survives_the_purge_and_an_unpinned_one_does_not(store):
    """The 6-hour sweep must not delete what a user was told to come back for."""
    pinned = await _streamed(store)
    loose = await _streamed(store)
    await store.pin(pinned)

    # Everything is older than the ordinary window; the pinned one is held to
    # the longer artifact window instead.
    removed = await store.purge_older_than(-1, pinned_seconds=_TTL)

    assert removed == 1
    assert await store.is_pinned(pinned) is True
    assert await store.local_path(pinned)  # still readable
    with pytest.raises(StreamGone):
        await store.local_path(loose)


async def test_a_pin_with_no_window_named_is_kept_indefinitely(store):
    ref = await _streamed(store)
    await store.pin(ref)
    assert await store.purge_older_than(-1) == 0
    assert await store.local_path(ref)


async def test_a_pinned_stream_is_swept_once_its_own_window_passes(store):
    """Pinned is a longer TTL, not immortality."""
    ref = await _streamed(store)
    await store.pin(ref)
    assert await store.purge_older_than(-1, pinned_seconds=-1) == 1
    with pytest.raises(StreamGone):
        await store.local_path(ref)


async def test_unpinning_returns_a_stream_to_the_normal_purge(store):
    ref = await _streamed(store)
    await store.pin(ref)
    await store.unpin(ref)

    assert await store.is_pinned(ref) is False
    assert await store.purge_older_than(-1, pinned_seconds=_TTL) == 1
    with pytest.raises(StreamGone):
        await store.local_path(ref)


async def test_deleting_a_run_unpins_what_it_held(store, backend):
    ref = await _streamed(store)
    step = _step({"name": "rows", "from": "state.projects"})
    await _runner([step], store, backend)._data_node(step)({"projects": ref.to_state()})
    assert await store.is_pinned(ref) is True

    forgotten = await data_artifacts.forget_run_artifacts(store, backend, "run_abc")

    assert forgotten == 1
    assert await backend.list_for_run("run_abc") == []
    assert await store.is_pinned(ref) is False


async def test_pinning_a_stream_that_is_gone_is_refused(store):
    """A manifest row against missing bytes could only ever 410."""
    ref = await _streamed(store)
    await store.delete(ref)
    with pytest.raises(StreamGone):
        await store.pin(ref)


async def test_a_pin_marker_is_not_mistaken_for_a_stream(store, tmp_path):
    """The sweep globs ds_*.jsonl; the marker must stay out of that."""
    ref = await _streamed(store)
    await store.pin(ref)
    markers = list((tmp_path / "streams").glob("*.pin"))
    assert [p.name for p in markers] == [f"{ref.id}.pin"]
    assert not any(p.suffix == ".jsonl" for p in markers)


async def test_a_tampered_stream_id_cannot_name_another_pin_file(store):
    """The property the local store already had, preserved for the marker."""
    from app.domain.models.datastream import DataRef

    with pytest.raises(ValueError):
        store._pin_path_for("../escape")
    # Through the public surface it is simply not pinned.
    assert await store.is_pinned(DataRef(id="../escape")) is False


# ---------------------------------------------------------------------------
# Nothing here fails a run
# ---------------------------------------------------------------------------

async def test_a_missing_path_is_a_warning_not_a_failure(store, backend, caplog):
    step = _step(
        {"name": "gone", "from": "state.nope"},
        {"name": "rows", "from": "state.rows"},
    )
    with caplog.at_level(logging.WARNING):
        out = await _runner([step], store, backend)._data_node(step)({"rows": ROWS})

    assert out == {}
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("'gone'" in m and "resolved to nothing" in m for m in warnings)
    # The other selection was still recorded: one bad path must not cost the
    # rest of the step.
    assert [a.name for a in await backend.list_for_run("run_abc")] == ["rows"]


async def test_an_unusable_selection_is_reported_and_skipped(store, backend, caplog):
    step = _step(
        {"name": "rows", "from": "state.rows", "format": "parquet"},
        {"name": "", "from": "state.rows"},
        {"name": "dupe", "from": "state.rows"},
        {"name": "dupe", "from": "state.rows"},
    )
    with caplog.at_level(logging.WARNING):
        out = await _runner([step], store, backend)._data_node(step)({"rows": ROWS})

    assert out == {}
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("parquet" in m for m in messages)
    assert any("has no `name`" in m for m in messages)
    assert any("declared more than once" in m for m in messages)
    assert [a.name for a in await backend.list_for_run("run_abc")] == ["dupe"]


async def test_no_backend_configured_is_a_warning_not_a_failure(store, caplog):
    step = _step({"name": "rows", "from": "state.rows"})
    with caplog.at_level(logging.WARNING):
        out = await _runner([step], store, None)._data_node(step)({"rows": ROWS})
    assert out == {}
    assert any("data artifact backend" in (r.getMessage()) for r in caplog.records)


async def test_a_csv_selection_over_a_document_is_refused_without_failing(
    store, backend, caplog
):
    """csv needs rows. Saying so beats emitting a one-column oddity."""
    step = _step({"name": "summary", "from": "state.summary", "format": "csv"})
    with caplog.at_level(logging.WARNING):
        out = await _runner([step], store, backend)._data_node(step)(
            {"summary": {"total": 3}}
        )
    assert out == {}
    assert await backend.list_for_run("run_abc") == []
    assert any("csv" in (r.getMessage()) for r in caplog.records)


def test_parse_selections_reports_a_missing_block():
    good, problems = parse_selections({"id": "x", "type": "data"})
    assert good == []
    assert problems and "no `selections`" in problems[0]


def test_a_from_may_be_written_with_or_without_the_state_prefix():
    assert data_artifacts.strip_state_prefix("state.projects") == "projects"
    assert data_artifacts.strip_state_prefix("$.state.projects") == "projects"
    assert data_artifacts.strip_state_prefix("projects.open") == "projects.open"


# ---------------------------------------------------------------------------
# Many selections, many passes
# ---------------------------------------------------------------------------

async def test_repeated_executions_each_record_their_own_artifacts(store, backend):
    """A `data` node inside a loop: every pass is its own artifact."""
    ref = await _streamed(store)
    step = _step(
        {"name": "open_projects", "from": "state.projects"},
        {"name": "summary", "from": "state.summary", "format": "json"},
    )
    node = _runner([step], store, backend)._data_node(step)
    state = {"projects": ref.to_state(), "summary": {"total": 1}}

    await node(state)
    await node(state)

    rows = await backend.list_for_run("run_abc")
    assert len(rows) == 4
    assert len({a.id for a in rows}) == 4, "artifact ids must be unique"
    assert sorted({a.name for a in rows}) == ["open_projects", "summary"]
    # Both passes pinned the same underlying stream; unpinning it once at
    # deletion is still correct because the pin is a marker, not a count.
    assert await store.is_pinned(ref) is True


async def test_artifacts_of_one_run_are_not_visible_under_another(store, backend):
    ref = await _streamed(store)
    step = _step({"name": "rows", "from": "state.projects"})
    await _runner([step], store, backend)._data_node(step)({"projects": ref.to_state()})
    artifact = (await backend.list_for_run("run_abc"))[0]

    assert await backend.get("run_abc", artifact.id) is artifact
    assert await backend.get("run_other", artifact.id) is None
    assert await backend.list_for_run("run_other") == []


# ---------------------------------------------------------------------------
# truncated, end to end
# ---------------------------------------------------------------------------

async def test_truncated_propagates_into_the_manifest(store, backend):
    """A prefix presented as the whole answer is the damaging failure here."""
    ref = await _streamed(store, truncated=True)
    step = _step({"name": "open_projects", "from": "state.projects"})

    await _runner([step], store, backend)._data_node(step)({"projects": ref.to_state()})

    artifact = (await backend.list_for_run("run_abc"))[0]
    assert artifact.truncated is True
    assert artifact.manifest_entry()["truncated"] is True


async def test_an_untruncated_result_is_not_flagged(store, backend):
    ref = await _streamed(store)
    step = _step({"name": "open_projects", "from": "state.projects"})
    await _runner([step], store, backend)._data_node(step)({"projects": ref.to_state()})
    assert (await backend.list_for_run("run_abc"))[0].truncated is False


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------

async def _download_bytes(store, artifact) -> bytes:
    download = await prepare_download(store, artifact)
    return b"".join([chunk async for chunk in download.chunks])


def _artifact(ref, fmt: str, *, name: str = "rows") -> DataArtifact:
    return DataArtifact.for_stream(
        run_id="run_abc", step_id="export_rows", name=name, fmt=fmt,
        ref=ref, ttl_seconds=_TTL,
    )


async def test_jsonl_is_served_as_stored(store):
    """It is already the requested format; a re-encode could only lose."""
    ref = await _streamed(store)
    stored = open(await store.local_path(ref), "rb").read()

    artifact = _artifact(ref, "jsonl")
    download = await prepare_download(store, artifact)
    body = b"".join([chunk async for chunk in download.chunks])

    assert body == stored
    # Known up front, so the browser can show a progress bar.
    assert download.content_length == artifact.bytes == len(stored)
    assert download.filename == "rows.jsonl"


async def test_json_wraps_a_list_into_an_array(store):
    ref = await _streamed(store)
    artifact = _artifact(ref, "json")

    download = await prepare_download(store, artifact)
    body = b"".join([chunk async for chunk in download.chunks])

    assert json.loads(body) == ROWS
    # Streamed, so the size is not known until the last record — and a wrong
    # Content-Length would truncate the body the client keeps.
    assert download.content_length is None


async def test_json_streams_rather_than_materialising(store):
    """The array is framed around the records, not built out of them."""
    ref = await _streamed(store)
    download = await prepare_download(store, _artifact(ref, "json"))
    chunks = [chunk async for chunk in download.chunks]
    # The opening bracket rides along with the first record, so starting the
    # generator is a real read of the stream (see _started).
    assert chunks[0].startswith(b"[{")
    assert chunks[-1] == b"]"
    assert len(chunks) == len(ROWS) + 1


async def test_json_over_a_document_serves_the_document(store):
    ref = await _streamed(store, [{"total": 3}], shape="value")
    artifact = _artifact(ref, "json", name="summary")
    body = await _download_bytes(store, artifact)
    assert json.loads(body) == {"total": 3}


async def test_csv_header_is_the_union_of_the_keys_it_scanned(store):
    """A sparse response omits null fields; the first row is not the schema."""
    rows = [{"id": 1, "amount": 10}, {"id": 2, "region": "eu"}, {"id": 3}]
    ref = await _streamed(store, rows)

    body = (await _download_bytes(store, _artifact(ref, "csv"))).decode()

    lines = body.splitlines()
    assert lines[0] == "id,amount,region"
    # A field a record does not carry is an empty cell, never "None".
    assert lines[1] == "1,10,"
    assert lines[2] == "2,,eu"
    assert lines[3] == "3,,"


async def test_csv_refuses_a_non_flat_record_with_a_clear_message(store):
    ref = await _streamed(store, [{"id": 1, "owner": {"name": "ada"}}])
    with pytest.raises(DataArtifactError) as exc:
        await prepare_download(store, _artifact(ref, "csv"))
    message = str(exc.value)
    assert "nested" in message and "owner" in message
    assert "jsonl" in message, "the message should name a format that would work"


async def test_csv_refuses_a_list_of_scalars(store):
    ref = await _streamed(store, ["a", "b"])
    with pytest.raises(DataArtifactError) as exc:
        await prepare_download(store, _artifact(ref, "csv"))
    assert "flat records" in str(exc.value)


async def test_csv_rejection_happens_before_any_bytes_are_sent(store):
    """So the caller gets a status code rather than a corrupt file."""
    rows = [{"id": i} for i in range(CSV_HEADER_SCAN_ITEMS)]
    rows[3] = {"id": 3, "tags": ["a"]}
    ref = await _streamed(store, rows)
    with pytest.raises(DataArtifactError):
        await prepare_download(store, _artifact(ref, "csv"))


async def test_a_download_of_a_swept_stream_raises_stream_gone(store, backend):
    ref = await _streamed(store)
    artifact = _artifact(ref, "jsonl")
    await store.delete(ref)
    with pytest.raises(StreamGone):
        await prepare_download(store, artifact)


# ---------------------------------------------------------------------------
# Manifest entry shape
# ---------------------------------------------------------------------------

async def test_the_manifest_entry_carries_the_pinned_contract(store):
    ref = await _streamed(store, truncated=True)
    entry = _artifact(ref, "jsonl", name="open_projects").manifest_entry()

    assert set(entry) == {
        "id", "origin", "step_id", "name", "format", "filename", "shape",
        "items", "bytes", "truncated", "source_id", "operation", "created_at",
        "expires_at", "download_url",
    }
    assert entry["origin"] == "data_node"
    assert entry["id"].startswith("art_")
    assert entry["filename"] == "open_projects.jsonl"
    assert entry["download_url"] == f"/runs/run_abc/data/{entry['id']}"
    assert entry["truncated"] is True
    assert entry["created_at"].endswith("Z") and entry["expires_at"].endswith("Z")


async def test_a_signed_url_replaces_the_download_path_when_offered(store):
    ref = await _streamed(store)
    entry = _artifact(ref, "jsonl").manifest_entry(
        download_url="https://storage.example/signed?sig=1"
    )
    assert entry["download_url"] == "https://storage.example/signed?sig=1"


def test_a_hostile_selection_name_cannot_escape_the_filename():
    """The name reaches a Content-Disposition header."""
    artifact = DataArtifact(
        run_id="r", step_id="s", name='../../etc/passwd"; x=y', stream_id="ds_1"
    )
    assert "/" not in artifact.filename
    assert '"' not in artifact.filename
    assert artifact.filename.endswith(".jsonl")


def test_selection_defaults_to_jsonl():
    assert Selection(name="n", path="state.x").format == "jsonl"


# ---------------------------------------------------------------------------
# The tool surfaces
# ---------------------------------------------------------------------------

class _FakeRunRepository:
    def __init__(self, *runs: GraphRun) -> None:
        self._runs = {r.id: r for r in runs if r is not None}

    async def get(self, run_id: str) -> GraphRun | None:
        return self._runs.get(run_id)


def _deps(backend, run_id: str = "run_abc"):
    from app.application.management_tools import ManagementDeps

    # Two runs exist, and only one of them exported anything: that is what
    # makes the scoping assertions below about the *manifest* key rather than
    # about a missing run.
    runs = [
        GraphRun(id=rid, graph_id="g", user_request="hi", status="completed")
        for rid in (run_id, "run_other")
    ]
    return ManagementDeps(
        registry=MagicMock(),
        run_repository=_FakeRunRepository(*runs),
        data_artifact_backend=backend,
    )


async def _recorded(store, backend, **kwargs) -> DataArtifact:
    ref = await _streamed(store, **kwargs)
    step = _step({"name": "open_projects", "from": "state.projects"})
    await _runner([step], store, backend)._data_node(step)({"projects": ref.to_state()})
    return (await backend.list_for_run("run_abc"))[0]


async def test_list_run_data_renders_the_manifest_for_an_agent(store, backend):
    from app.application import management_tools as core

    artifact = await _recorded(store, backend)
    out = await core.list_run_data(_deps(backend), "run_abc")

    assert artifact.id in out
    assert "open_projects" in out
    # A URL a tool caller can actually fetch, prefix included.
    assert f"/api/v1/runs/run_abc/data/{artifact.id}" in out


async def test_list_run_data_shouts_about_a_truncated_artifact(store, backend):
    """An LLM skimming `truncated: true` is exactly the reader that misses it."""
    from app.application import management_tools as core

    await _recorded(store, backend, truncated=True)
    out = await core.list_run_data(_deps(backend), "run_abc")
    assert "INCOMPLETE" in out


async def test_list_run_data_says_so_when_a_run_exported_nothing(store, backend):
    from app.application import management_tools as core

    out = await core.list_run_data(_deps(backend), "run_abc")
    assert "no downloadable data" in out


async def test_list_run_data_reports_an_unknown_run(store, backend):
    from app.application import management_tools as core

    out = await core.list_run_data(_deps(backend), "run_missing")
    assert "not found" in out


async def test_get_run_data_artifact_reports_the_metadata_and_the_url(store, backend):
    from app.application import management_tools as core

    artifact = await _recorded(store, backend, truncated=True)
    out = await core.get_run_data_artifact(_deps(backend), "run_abc", artifact.id)

    assert f"Artifact: {artifact.id}" in out
    assert "Format: jsonl (filename open_projects.jsonl)" in out
    assert "Truncated: True" in out
    assert "WARNING" in out
    assert f"/api/v1/runs/run_abc/data/{artifact.id}" in out


async def test_get_run_data_artifact_is_scoped_to_its_run(store, backend):
    from app.application import management_tools as core

    artifact = await _recorded(store, backend)
    out = await core.get_run_data_artifact(_deps(backend), "run_other", artifact.id)
    assert "No data artifact" in out


async def test_the_tools_say_so_when_no_backend_is_configured(store):
    from app.application import management_tools as core

    assert "unavailable" in await core.list_run_data(_deps(None), "run_abc")
    # With no store there is no artifact to find either, and the id is named
    # back so the caller can see it was not a typo on our side.
    out = await core.get_run_data_artifact(_deps(None), "run_abc", "art_x")
    assert "No data artifact 'art_x'" in out


async def test_both_tool_surfaces_publish_the_run_data_tools():
    """A half-finished registration is what the parity tests exist to catch."""
    from app.api.mcp.management_server import (
        build_management_mcp,
        register_management_tools,
    )

    mcp = build_management_mcp()
    register_management_tools(mcp, lambda: None)
    names = {t.name for t in await mcp.list_tools()}
    assert {"list_run_data", "get_run_data_artifact"} <= names


# ---------------------------------------------------------------------------
# Run status
# ---------------------------------------------------------------------------

def test_a_data_step_that_ran_reads_as_finished_not_skipped():
    """Its successful update is `{}`, which for every other type means skipped."""
    from app.infrastructure.orchestration.yaml_graph import step_status_from_output

    assert step_status_from_output("d", {}, step_type="data", ran=True) == "finished"
    # A `when` guard skips the node without running it, and that is a real skip.
    assert step_status_from_output("d", {}, step_type="data", ran=False) == "skipped"
    # Nothing else changes: an empty update from any other type still means
    # the node did not produce anything.
    assert step_status_from_output("x", {}, step_type="llm", ran=True) == "skipped"
    assert step_status_from_output("x", {"a": 1}) == "finished"


# ---------------------------------------------------------------------------
# Data source results, on the same rails
# ---------------------------------------------------------------------------
#
# A data source result already in the run's state is offered in the *same*
# artifact shape and downloaded from the same URL, rather than through a reader
# of its own — the data node is how data leaves the system, so a datasource
# result rides those rails. Two things must stay true: the default listing does
# not volunteer them (the Data panel shows curated exports), and one is only
# ever found by looking inside the state of the run named in the call.

def _run_with_ref(ref, *, run_id="run_abc", key="projects") -> GraphRun:
    return GraphRun(
        id=run_id, graph_id="g", user_request="hi", status="completed",
        state={key: ref.to_state(), "note": "not a stream"},
    )


def _deps_for(run, backend=None):
    from app.application.management_tools import ManagementDeps

    return ManagementDeps(
        registry=MagicMock(),
        run_repository=_FakeRunRepository(run),
        data_artifact_backend=backend,
    )


async def test_a_datasource_result_is_described_in_the_artifact_shape(store):
    from app.application.data_artifacts import datasource_artifacts

    ref = await _streamed(store, truncated=True)
    entries = datasource_artifacts(_run_with_ref(ref), ttl_seconds=21600.0)

    assert len(entries) == 1
    entry = entries[0].manifest_entry()
    # Identical field set to a curated export, plus origin.
    assert set(entry) == set(
        _artifact(ref, "jsonl").manifest_entry()
    )
    assert entry["origin"] == "datasource"
    assert entry["name"] == "google-sheets.read_open_projects"
    assert entry["filename"] == "google-sheets.read_open_projects.jsonl"
    # The state key is the step that produced it.
    assert entry["step_id"] == "projects"
    assert entry["items"] == len(ROWS)
    assert entry["truncated"] is True


async def test_the_default_listing_shows_only_curated_exports(store, backend):
    """The Data panel is what somebody chose to export, not every fetch."""
    from app.application.data_artifacts import list_run_artifacts

    ref = await _streamed(store)
    run = _run_with_ref(ref)

    assert await list_run_artifacts(backend, run) == []
    opted_in = await list_run_artifacts(
        backend, run, include_datasource=True, datasource_ttl_seconds=21600.0
    )
    assert [a.origin for a in opted_in] == ["datasource"]


async def test_a_curated_export_wins_over_the_raw_result_it_names(store, backend):
    """One stream, one entry: the pinned one a person actually named."""
    from app.application.data_artifacts import list_run_artifacts

    ref = await _streamed(store)
    step = _step({"name": "open_projects", "from": "state.projects"})
    await _runner([step], store, backend)._data_node(step)({"projects": ref.to_state()})

    rows = await list_run_artifacts(
        backend, _run_with_ref(ref), include_datasource=True,
        datasource_ttl_seconds=21600.0,
    )

    assert [a.origin for a in rows] == ["data_node"]
    assert rows[0].name == "open_projects"


async def test_a_datasource_entry_expires_on_the_ordinary_stream_ttl(store):
    """It is not pinned, and the manifest must not imply that it is."""
    from app.application.data_artifacts import datasource_artifacts

    ref = await _streamed(store)
    entry = datasource_artifacts(_run_with_ref(ref), ttl_seconds=21600.0)[0]

    assert (entry.expires_at - entry.created_at).total_seconds() == 21600.0
    assert await store.is_pinned(ref) is False


async def test_a_datasource_artifact_resolves_and_downloads_like_any_other(store):
    from app.application.data_artifacts import find_run_artifact

    ref = await _streamed(store)
    run = _run_with_ref(ref)
    entry = await find_run_artifact(
        None, run, f"dsr_{ref.id}", datasource_ttl_seconds=21600.0
    )

    assert entry is not None and entry.origin == "datasource"
    body = await _download_bytes(store, entry)
    assert body == open(await store.local_path(ref), "rb").read()


async def test_a_stream_of_another_run_cannot_be_resolved_through_this_one(store):
    """Otherwise the download endpoint is a cross-run read primitive."""
    from app.application.data_artifacts import find_run_artifact

    mine = await _streamed(store)
    theirs = await _streamed(store)

    found = await find_run_artifact(
        None, _run_with_ref(mine), f"dsr_{theirs.id}", datasource_ttl_seconds=21600.0
    )

    assert found is None


async def test_a_bare_stream_id_is_not_an_artifact_id(store):
    """The prefix is what makes an id resolvable, and it is still run-scoped."""
    from app.application.data_artifacts import find_run_artifact

    ref = await _streamed(store)
    run = _run_with_ref(ref)

    assert await find_run_artifact(None, run, ref.id) is None
    assert await find_run_artifact(None, run, "dsr_") is None
    assert await find_run_artifact(None, run, "dsr_../escape") is None


async def test_a_plain_state_value_is_not_offered_as_a_datasource_result(store):
    from app.application.data_artifacts import datasource_artifacts

    run = GraphRun(
        id="run_abc", graph_id="g", user_request="hi", status="completed",
        state={"answer": "42", "rows": [{"a": 1}]},
    )
    assert datasource_artifacts(run, ttl_seconds=21600.0) == []


async def test_list_run_data_can_be_asked_for_the_raw_results(store, backend):
    from app.application import management_tools as core

    ref = await _streamed(store, truncated=True)
    run = _run_with_ref(ref)

    default = await core.list_run_data(_deps_for(run, backend), "run_abc")
    assert "no downloadable data" in default
    # And it says how to see more, so the absence is not a dead end.
    assert "include_datasource" in default

    opted_in = await core.list_run_data(
        _deps_for(run, backend), "run_abc", include_datasource=True
    )
    assert "[datasource]" in opted_in
    assert "google-sheets.read_open_projects" in opted_in
    assert "INCOMPLETE" in opted_in
    assert f"/api/v1/runs/run_abc/data/dsr_{ref.id}" in opted_in


async def test_get_run_data_artifact_serves_a_datasource_result_identically(
    store, backend
):
    from app.application import management_tools as core

    ref = await _streamed(store)
    run = _run_with_ref(ref)

    out = await core.get_run_data_artifact(
        _deps_for(run, backend), "run_abc", f"dsr_{ref.id}"
    )

    assert f"Artifact: dsr_{ref.id}" in out
    assert "Origin: datasource" in out
    assert "Format: jsonl" in out
    # And it is honest about what it is not: a curated, pinned export.
    assert "not pinned" in out


async def test_get_run_data_artifact_will_not_reach_another_runs_stream(
    store, backend
):
    from app.application import management_tools as core

    mine = await _streamed(store)
    theirs = await _streamed(store)

    out = await core.get_run_data_artifact(
        _deps_for(_run_with_ref(mine), backend), "run_abc", f"dsr_{theirs.id}"
    )

    assert "No data artifact" in out


# ---------------------------------------------------------------------------
# Through a real graph
# ---------------------------------------------------------------------------

async def test_a_data_step_inside_a_compiled_graph_leaves_state_alone(
    store, backend
):
    """Driving the node directly proves the body; this proves the wiring.

    A `data` step declares no `output_key`, so it contributes nothing to the
    state schema — and LangGraph drops any key that is not in the schema, which
    is exactly why the step must not want one.
    """
    ref = await _streamed(store)
    steps = [
        # Stands in for a `data_source` step: the ref is what one of those
        # leaves under its output_key.
        {"id": "seed", "type": "python", "sandbox": False, "output_key": "projects",
         "code": f"output = {ref.to_state()!r}"},
        {"id": "export_rows", "type": "data", "selections": [
            {"name": "open_projects", "from": "state.projects"},
        ]},
    ]
    runner = _runner(steps, store, backend)

    out = await runner.graph.ainvoke(
        {"request": "go"}, {"configurable": {"thread_id": "t1"}}
    )

    assert out["projects"] == ref.to_state()
    # The data step added nothing of its own to the final state.
    assert "export_rows" not in out
    rows = await backend.list_for_run("run_abc")
    assert [a.name for a in rows] == ["open_projects"]
    assert rows[0].stream_id == ref.id
