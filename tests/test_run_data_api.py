"""The run-data download endpoints, and the gate in front of them.

The manifest shape here is the contract the frontend was built against, so the
assertions are deliberately literal about field names and about the two error
codes that mean different things to a user: 404 "no such download" and 410 "it
existed and the bytes are gone".
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.domain.models.data_artifact import DataArtifact
from app.domain.models.graph_run import GraphRun
from app.infrastructure.datasources.datastream import LocalDiskStreamStore
from app.infrastructure.persistence.data_artifact_backend import (
    InMemoryDataArtifactBackend,
)
from tests.test_graphs_api import _build_container, _build_registry

_RUNS = "/api/v1/runs"

ROWS = [
    {"id": 1, "amount": 10, "region": "eu"},
    {"id": 2, "region": "us"},
    {"id": 3, "amount": 30},
]


@pytest.fixture
async def client(tmp_path):
    container = _build_container(_build_registry())
    container.stream_store = LocalDiskStreamStore(tmp_path / "streams")
    container.data_artifact_backend = InMemoryDataArtifactBackend()
    container.run_repository.get = AsyncMock(
        return_value=GraphRun(
            id="run_abc", graph_id="simple", user_request="hi", status="completed"
        )
    )
    app = create_app()
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, container


async def _record(container, *, fmt: str = "jsonl", rows=ROWS, truncated=False,
                  name: str = "open_projects", shape: str = "list") -> DataArtifact:
    writer = await container.stream_store.open_writer(
        source_id="google-sheets", operation="read_open_projects", shape=shape
    )
    await writer.append_many(rows)
    ref = await writer.close()
    ref.truncated = truncated
    await container.stream_store.pin(ref)
    artifact = DataArtifact.for_stream(
        run_id="run_abc", step_id="export_rows", name=name, fmt=fmt,
        ref=ref, ttl_seconds=7 * 24 * 3600.0,
    )
    await container.data_artifact_backend.add(artifact)
    return artifact


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------

async def test_the_manifest_matches_the_pinned_contract(client):
    c, container = client
    artifact = await _record(container, truncated=True)

    resp = await c.get(f"{_RUNS}/run_abc/data")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run_id"] == "run_abc"
    assert len(body["artifacts"]) == 1
    entry = body["artifacts"][0]
    assert entry == {
        "id": artifact.id,
        "origin": "data_node",
        "step_id": "export_rows",
        "name": "open_projects",
        "format": "jsonl",
        "filename": "open_projects.jsonl",
        "shape": "list",
        "items": 3,
        "bytes": artifact.bytes,
        "truncated": True,
        "source_id": "google-sheets",
        "operation": "read_open_projects",
        "created_at": entry["created_at"],
        "expires_at": entry["expires_at"],
        "download_url": f"/runs/run_abc/data/{artifact.id}",
    }


async def test_a_run_with_no_data_node_gets_an_empty_list(client):
    """Not an error: the frontend renders "nothing to download" from this."""
    c, _ = client
    resp = await c.get(f"{_RUNS}/run_abc/data")
    assert resp.status_code == 200
    assert resp.json() == {"run_id": "run_abc", "artifacts": []}


async def test_the_manifest_404s_for_an_unknown_run(client):
    c, container = client
    container.run_repository.get = AsyncMock(return_value=None)
    resp = await c.get(f"{_RUNS}/nope/data")
    assert resp.status_code == 404


async def test_the_manifest_lists_every_pass_of_a_looping_node(client):
    c, container = client
    first = await _record(container, name="open_projects")
    second = await _record(container, name="open_projects")

    body = (await c.get(f"{_RUNS}/run_abc/data")).json()

    assert {e["id"] for e in body["artifacts"]} == {first.id, second.id}


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

async def test_jsonl_downloads_the_stored_bytes(client):
    c, container = client
    artifact = await _record(container)
    stored = open(await container.stream_store.local_path(artifact.as_ref()), "rb").read()

    resp = await c.get(f"{_RUNS}/run_abc/data/{artifact.id}")

    assert resp.status_code == 200
    assert resp.content == stored
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    assert (
        resp.headers["content-disposition"]
        == 'attachment; filename="open_projects.jsonl"'
    )
    assert resp.headers["content-length"] == str(artifact.bytes)


async def test_a_truncated_download_says_so_on_the_response(client):
    """A client that only ever sees the file must still be able to tell."""
    c, container = client
    artifact = await _record(container, truncated=True)
    resp = await c.get(f"{_RUNS}/run_abc/data/{artifact.id}")
    assert resp.headers["x-data-truncated"] == "true"


async def test_an_untruncated_download_is_marked_complete(client):
    c, container = client
    artifact = await _record(container)
    resp = await c.get(f"{_RUNS}/run_abc/data/{artifact.id}")
    assert resp.headers["x-data-truncated"] == "false"


async def test_json_downloads_a_streamed_array(client):
    c, container = client
    artifact = await _record(container, fmt="json")

    resp = await c.get(f"{_RUNS}/run_abc/data/{artifact.id}")

    assert resp.status_code == 200
    assert json.loads(resp.content) == ROWS
    assert resp.headers["content-type"].startswith("application/json")
    # A streamed transform has no length to promise.
    assert "content-length" not in resp.headers


async def test_csv_downloads_a_header_union_and_rows(client):
    c, container = client
    artifact = await _record(container, fmt="csv")

    resp = await c.get(f"{_RUNS}/run_abc/data/{artifact.id}")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    lines = resp.text.splitlines()
    assert lines[0] == "id,amount,region"
    assert lines[1] == "1,10,eu"
    assert lines[2] == "2,,us"
    assert lines[3] == "3,30,"


async def test_csv_over_nested_records_is_refused_rather_than_malformed(client):
    c, container = client
    artifact = await _record(
        container, fmt="csv", rows=[{"id": 1, "owner": {"name": "ada"}}]
    )

    resp = await c.get(f"{_RUNS}/run_abc/data/{artifact.id}")

    assert resp.status_code == 409
    assert "nested" in resp.json()["detail"]


async def test_an_unknown_artifact_404s(client):
    c, _ = client
    resp = await c.get(f"{_RUNS}/run_abc/data/art_nope")
    assert resp.status_code == 404
    assert "art_nope" in resp.json()["detail"]


async def test_an_artifact_of_another_run_404s(client):
    """The run in the path is part of the key, not decoration on it."""
    c, container = client
    artifact = await _record(container)
    other = DataArtifact.for_stream(
        run_id="run_other", step_id="s", name="x", fmt="jsonl",
        ref=artifact.as_ref(), ttl_seconds=1.0,
    )
    await container.data_artifact_backend.add(other)

    resp = await c.get(f"{_RUNS}/run_abc/data/{other.id}")

    assert resp.status_code == 404


async def test_a_swept_stream_gives_410_not_404(client):
    """"It existed and is gone" is a different fact from "no such download"."""
    c, container = client
    artifact = await _record(container)
    await container.stream_store.delete(artifact.as_ref())

    resp = await c.get(f"{_RUNS}/run_abc/data/{artifact.id}")

    assert resp.status_code == 410
    assert "gone" in resp.json()["detail"].lower()


async def test_a_signed_url_is_offered_in_the_manifest_when_the_store_can_sign(client):
    """The frontend opens an absolute URL directly instead of fetching it."""
    c, container = client
    artifact = await _record(container)

    async def _sign(ref, *, filename="", content_type=""):
        assert ref.id == artifact.stream_id
        assert filename == "open_projects.jsonl"
        return "https://storage.example/o/x?X-Goog-Signature=abc"

    container.stream_store.signed_url = _sign  # type: ignore[attr-defined]

    body = (await c.get(f"{_RUNS}/run_abc/data")).json()

    assert (
        body["artifacts"][0]["download_url"]
        == "https://storage.example/o/x?X-Goog-Signature=abc"
    )


async def test_a_store_that_cannot_sign_keeps_the_backend_endpoint(client):
    c, container = client
    artifact = await _record(container)

    async def _sign(ref, *, filename="", content_type=""):
        return None

    container.stream_store.signed_url = _sign  # type: ignore[attr-defined]

    body = (await c.get(f"{_RUNS}/run_abc/data")).json()

    assert body["artifacts"][0]["download_url"] == f"/runs/run_abc/data/{artifact.id}"


async def test_the_download_endpoint_redirects_when_the_store_can_sign(client):
    """Kept for API clients; the browser path goes through the manifest."""
    c, container = client
    artifact = await _record(container)

    async def _sign(ref, *, filename="", content_type=""):
        return "https://storage.example/o/x?X-Goog-Signature=abc"

    container.stream_store.signed_url = _sign  # type: ignore[attr-defined]

    resp = await c.get(f"{_RUNS}/run_abc/data/{artifact.id}", follow_redirects=False)

    assert resp.status_code == 302
    assert resp.headers["location"].startswith("https://storage.example/")


async def test_a_transform_is_never_handed_to_a_signed_url(client):
    """Object storage will not apply the transform, so it must not be asked to."""
    c, container = client
    artifact = await _record(container, fmt="csv")
    container.stream_store.signed_url = AsyncMock(  # type: ignore[attr-defined]
        return_value="https://storage.example/raw"
    )

    resp = await c.get(f"{_RUNS}/run_abc/data/{artifact.id}", follow_redirects=False)

    assert resp.status_code == 200
    assert resp.text.splitlines()[0] == "id,amount,region"
    container.stream_store.signed_url.assert_not_called()


# ---------------------------------------------------------------------------
# Deleting a run forgets its downloads
# ---------------------------------------------------------------------------

async def test_deleting_a_run_drops_the_manifest_and_unpins_the_streams(
    client, monkeypatch
):
    c, container = client
    monkeypatch.setattr("app.services.agent_cleanup.cleanup_run_agents", AsyncMock())
    artifact = await _record(container)
    container.run_repository.delete = AsyncMock()
    assert await container.stream_store.is_pinned(artifact.as_ref()) is True

    resp = await c.delete("/api/v1/workflows/runs/run_abc")

    assert resp.status_code == 204
    assert await container.data_artifact_backend.list_for_run("run_abc") == []
    assert await container.stream_store.is_pinned(artifact.as_ref()) is False


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_the_run_data_paths_are_not_exempt_from_authentication():
    """A run's data is exactly as sensitive as the run itself.

    The agent-callback exemption covers ``/runs/{id}/agent/...`` only, where the
    run id is a spawned container's bearer capability. These paths must need a
    user token, exactly as reading the run does.
    """
    from app.api.middleware.auth import _is_unprotected

    assert not _is_unprotected("GET", "/api/v1/runs/run_abc/data")
    assert not _is_unprotected("GET", "/api/v1/runs/run_abc/data/art_1")
    # The neighbouring exemption is still in place and still narrow.
    assert _is_unprotected("POST", "/api/v1/runs/run_abc/agent/output")


def test_a_get_of_run_data_requires_the_same_permission_as_reading_a_run():
    from app.infrastructure.auth.authorization import Permission, permission_for_method

    assert permission_for_method("GET") is Permission.READ


def test_the_router_is_mounted_under_the_api_prefix():
    """Mounted at the bare root the frontend would 404 on every download."""
    app = create_app()
    paths = {r.path for r in app.routes}
    assert "/api/v1/runs/{run_id}/data" in paths
    assert "/api/v1/runs/{run_id}/data/{artifact_id}" in paths


# ---------------------------------------------------------------------------
# Data source results, behind an opt-in
# ---------------------------------------------------------------------------

async def _run_holding(container, *, run_id="run_abc", key="projects", truncated=False):
    """A run whose state carries a data source result, and the ref itself."""
    writer = await container.stream_store.open_writer(
        source_id="google-sheets", operation="read_open_projects"
    )
    await writer.append_many(ROWS)
    ref = await writer.close()
    ref.truncated = truncated
    run = GraphRun(
        id=run_id, graph_id="simple", user_request="hi", status="completed",
        state={key: ref.to_state(), "note": "not a stream"},
    )
    container.run_repository.get = AsyncMock(
        side_effect=lambda rid: run if rid == run_id else None
    )
    return ref


async def test_the_default_manifest_hides_datasource_results(client):
    """The frontend shipped against the curated-exports-only shape."""
    c, container = client
    await _run_holding(container)

    body = (await c.get(f"{_RUNS}/run_abc/data")).json()

    assert body == {"run_id": "run_abc", "artifacts": []}


async def test_the_opt_in_lists_datasource_results_in_the_same_shape(client):
    c, container = client
    ref = await _run_holding(container, truncated=True)

    body = (await c.get(f"{_RUNS}/run_abc/data?include_datasource=true")).json()

    assert len(body["artifacts"]) == 1
    entry = body["artifacts"][0]
    assert entry["origin"] == "datasource"
    assert entry["id"] == f"dsr_{ref.id}"
    assert entry["name"] == "google-sheets.read_open_projects"
    assert entry["filename"] == "google-sheets.read_open_projects.jsonl"
    assert entry["step_id"] == "projects"
    assert entry["items"] == len(ROWS)
    assert entry["truncated"] is True
    assert entry["download_url"] == f"/runs/run_abc/data/dsr_{ref.id}"


async def test_a_datasource_result_downloads_through_the_same_endpoint(client):
    """Same path, same headers — the call cannot tell the two kinds apart."""
    c, container = client
    ref = await _run_holding(container)
    stored = open(await container.stream_store.local_path(ref), "rb").read()

    resp = await c.get(f"{_RUNS}/run_abc/data/dsr_{ref.id}")

    assert resp.status_code == 200
    assert resp.content == stored
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    assert (
        resp.headers["content-disposition"]
        == 'attachment; filename="google-sheets.read_open_projects.jsonl"'
    )
    assert resp.headers["x-data-truncated"] == "false"


async def test_a_truncated_datasource_download_says_so_too(client):
    c, container = client
    ref = await _run_holding(container, truncated=True)
    resp = await c.get(f"{_RUNS}/run_abc/data/dsr_{ref.id}")
    assert resp.headers["x-data-truncated"] == "true"


async def test_a_swept_datasource_result_gives_the_same_410(client):
    c, container = client
    ref = await _run_holding(container)
    await container.stream_store.delete(ref)

    resp = await c.get(f"{_RUNS}/run_abc/data/dsr_{ref.id}")

    assert resp.status_code == 410


async def test_a_datasource_download_needs_no_opt_in(client):
    """The opt-in governs what a listing volunteers, not what a URL serves."""
    c, container = client
    ref = await _run_holding(container)
    assert (await c.get(f"{_RUNS}/run_abc/data/dsr_{ref.id}")).status_code == 200


async def test_a_datasource_result_is_not_pinned_by_listing_it(client):
    """Pinning is the data node's promise. Nothing promised this one."""
    c, container = client
    ref = await _run_holding(container)

    await c.get(f"{_RUNS}/run_abc/data?include_datasource=true")
    await c.get(f"{_RUNS}/run_abc/data/dsr_{ref.id}")

    assert await container.stream_store.is_pinned(ref) is False


async def test_a_datasource_entry_expires_on_the_stream_ttl_not_the_artifact_ttl(
    client,
):
    c, container = client
    from datetime import datetime

    await _run_holding(container)
    entry = (
        await c.get(f"{_RUNS}/run_abc/data?include_datasource=true")
    ).json()["artifacts"][0]

    created = datetime.fromisoformat(entry["created_at"].replace("Z", "+00:00"))
    expires = datetime.fromisoformat(entry["expires_at"].replace("Z", "+00:00"))
    assert (expires - created).total_seconds() == container.settings.stream_ttl_seconds


async def test_another_runs_stream_cannot_be_downloaded_through_this_run(client):
    """The security property: no bare stream id opens a file."""
    c, container = client
    # A stream that exists in the store but is in no state this run can see.
    writer = await container.stream_store.open_writer(source_id="crm", operation="x")
    await writer.append_many([{"secret": "theirs"}])
    other = await writer.close()
    await _run_holding(container)

    listing = await c.get(f"{_RUNS}/run_abc/data?include_datasource=true")
    download = await c.get(f"{_RUNS}/run_abc/data/dsr_{other.id}")

    assert other.id not in listing.text
    assert download.status_code == 404


async def test_a_raw_stream_id_is_not_a_download_url(client):
    c, container = client
    ref = await _run_holding(container)
    assert (await c.get(f"{_RUNS}/run_abc/data/{ref.id}")).status_code == 404
