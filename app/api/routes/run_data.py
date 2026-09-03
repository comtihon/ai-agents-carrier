"""Downloads for the data a run produced.

Two endpoints: the run's manifest, and one artifact's bytes.

Two origins, one mechanism.  A `data` step's selections are curated exports and
are listed by default; the data source results sitting in the run's state are
described in exactly the same shape behind ``?include_datasource=true``, and
both are downloaded through the same endpoint.  A data source result is found
by looking inside the state of the run in the path -- never by opening a stream
id a caller supplied -- which is what makes it exactly as accessible as the run
that produced it, and no more.

Route ordering note
-------------------
This router uses a ``/runs`` prefix, like ``agent_callbacks``, and its literal
``/data`` segments cannot collide with that router's ``/agent/...`` ones.

Authorisation
-------------
There is no gate of this module's own, deliberately.  A run's data is exactly
as sensitive as the run, so it is guarded by the same thing that guards reading
a run: ``OAuthMiddleware`` authenticates every ``/api/v1/...`` request and
``permission_for_method`` requires READ of a GET.  The exemption list in
``app.api.middleware.auth`` covers only ``/api/v1/runs/{id}/agent/...`` — the
spawned-agent callbacks, whose run id is their bearer capability — so these
paths need a user token exactly as ``GET /api/v1/workflows/runs/{id}`` does.
Adding a permission of its own here would be a second answer to a question
already answered, and inventing a weaker one (a capability URL, say) would make
a run's data easier to read than the run.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse, StreamingResponse

from app.api.dependencies import get_container
from app.application.data_artifacts import (
    DataArtifactError,
    find_run_artifact,
    list_run_artifacts,
    prepare_download,
)
from app.core.container import ApplicationContainer
from app.infrastructure.datasources.datastream import StreamGone

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["run-data"])


@router.get("/{run_id}/data")
async def list_run_data(
    run_id: str,
    include_datasource: bool = Query(
        False,
        description=(
            "Also list the data source results sitting in this run's state, "
            "in the same artifact shape (origin=\"datasource\"). Off by "
            "default: the Data panel shows the files somebody chose to export, "
            "not every intermediate result the run fetched."
        ),
    ),
    container: ApplicationContainer = Depends(get_container),
):
    """The run's download manifest: one entry per selection a `data` step named.

    An empty list is the normal answer for a run with no `data` step, or one
    whose selections all resolved to nothing — it is not an error, and the
    caller does not have to distinguish it from "no manifest backend
    configured" to render the page.

    With ``include_datasource`` the raw results the run fetched are listed too,
    described identically and marked ``origin: "datasource"``.  Those are not
    pinned, so their ``expires_at`` is the ordinary spill TTL rather than the
    artifact one — the listing says how long they are actually good for.
    """
    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    artifacts = await list_run_artifacts(
        getattr(container, "data_artifact_backend", None),
        run,
        include_datasource=include_datasource,
        datasource_ttl_seconds=container.settings.stream_ttl_seconds,
    )
    return {
        "run_id": run_id,
        "artifacts": [
            a.manifest_entry(
                download_url=await _signed_download_url(container.stream_store, a)
            )
            for a in artifacts
        ],
    }


async def _signed_download_url(store, artifact) -> str | None:
    """An absolute signed URL for *artifact*, when one is available.

    Put in the manifest rather than served as a 302 from the download
    endpoint: the frontend fetches that endpoint with a bearer token, and a
    redirect to a cross-origin signed target would be followed by ``fetch``
    and rejected by CORS.  An absolute URL in the manifest is opened directly
    instead, so the bytes never pass through this process.

    Only the stored form can be signed.  ``json`` and ``csv`` are transforms
    and something has to apply them; object storage will not, so those keep
    this backend's own endpoint.
    """
    if store is None or artifact.format != "jsonl":
        return None
    signer = getattr(store, "signed_url", None)
    if signer is None:
        return None
    try:
        return await signer(
            artifact.as_ref(),
            filename=artifact.filename,
            content_type=artifact.content_type,
        )
    except Exception:  # noqa: BLE001 — a manifest must render without the store
        # Including StreamGone: an expired artifact still belongs in the
        # listing (with its expiry shown), and finding out that the bytes are
        # gone is the download's job, not the listing's.
        logger.debug(
            "could not sign a URL for artifact '%s'", artifact.id, exc_info=True
        )
        return None


@router.get("/{run_id}/data/{artifact_id}")
async def download_run_data(
    run_id: str,
    artifact_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Stream one artifact's bytes as a file.

    Serves both origins through this one path: a `data` step's curated export
    and a data source result found in the run's state are downloaded the same
    way, with the same headers and the same 404 / 410, so a caller cannot tell
    which kind it is fetching from the shape of the call.

    ``jsonl`` is the stored form and goes out untouched with a real
    ``Content-Length``; ``json`` and ``csv`` are streamed transforms of it and
    send no length, because their size is not known until the last record and a
    wrong ``Content-Length`` truncates the body a client accepts.

    Where the store can sign a URL (object storage with a signing credential)
    the response is a 302 to it and the bytes never pass through this process
    at all.  Where it cannot, they do — a deployment without a signer gets a
    slower download, not a broken one.
    """
    # Resolved through the run, which is what makes a data source result
    # exactly as accessible as the run that produced it -- and no more.
    run = await container.run_repository.get(run_id)
    artifact = None if run is None else await find_run_artifact(
        getattr(container, "data_artifact_backend", None),
        run,
        artifact_id,
        datasource_ttl_seconds=container.settings.stream_ttl_seconds,
    )
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data artifact '{artifact_id}' for run '{run_id}'",
        )

    store = container.stream_store
    if store is None:
        raise HTTPException(
            status_code=503, detail="No data stream store is configured on this backend"
        )

    # A signed URL is only offered for the stored form. A transform has to be
    # applied by something, and object storage will not apply it.
    signer = getattr(store, "signed_url", None)
    if signer is not None and artifact.format == "jsonl":
        try:
            url = await signer(
                artifact.as_ref(),
                filename=artifact.filename,
                content_type=artifact.content_type,
            )
        except StreamGone as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        if url:
            return RedirectResponse(url=url, status_code=302)

    try:
        download = await prepare_download(store, artifact)
    except StreamGone as exc:
        # The bytes were swept or deleted. 410 rather than 404: the artifact is
        # a real thing this run produced, and "it existed and is gone" is a
        # different fact for a user than "no such download".
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except DataArtifactError as exc:
        # The data cannot be represented in the format it was recorded in — a
        # nested value in a csv, say. Refused with the reason rather than
        # streamed as something malformed.
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    headers = {
        "Content-Disposition": f'attachment; filename="{download.filename}"',
        # Carried onto the response as well as into the manifest: a client that
        # only ever sees the file must still be able to tell that it holds a
        # prefix rather than the whole answer.
        "X-Data-Truncated": "true" if artifact.truncated else "false",
    }
    if download.content_length is not None:
        headers["Content-Length"] = str(download.content_length)
    return StreamingResponse(
        download.chunks, media_type=download.content_type, headers=headers
    )
