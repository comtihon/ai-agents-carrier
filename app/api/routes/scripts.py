"""REST API for the Python script library.

Scripts are reusable Python bodies that ``python`` workflow steps reference by
``script_id``.  Saving from a workflow node is a save-by-name: when the name is
already taken the API answers 409 so the UI can ask the user to confirm an
overwrite, and the retry carries ``overwrite: true``.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_container
# Shared with the workflow-save capture path so a name-derived id is the same
# whichever route created it.
from app.application.script_capture import slugify
from app.core.container import ApplicationContainer
from app.domain.models.script_definition import ScriptDefinition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scripts", tags=["scripts"])


# ─── Request models ───────────────────────────────────────────────────────────

class ScriptRequest(BaseModel):
    # Omit to derive the id from the name (slugified) — that is what makes
    # "save to library by name" idempotent.
    id: str | None = None
    name: str
    description: str | None = None
    code: str = ""
    # Set by the client after the user confirmed the overwrite warning.
    overwrite: bool = False


class ScriptUpdateRequest(BaseModel):
    name: str
    description: str | None = None
    code: str = ""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_backend(container: ApplicationContainer) -> None:
    if container.script_backend is None:
        raise HTTPException(status_code=501, detail="Script backend not configured")


# ─── Collection routes ────────────────────────────────────────────────────────

@router.get("")
async def list_scripts(
    container: ApplicationContainer = Depends(get_container),
):
    """List every script in the library."""
    _require_backend(container)
    assert container.script_backend is not None
    scripts = await container.script_backend.list()
    scripts.sort(key=lambda s: (s.name or s.id).lower())
    return [s.model_dump(mode="json") for s in scripts]


@router.post("", status_code=201)
async def create_script(
    body: ScriptRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Create a script, or overwrite an existing one when ``overwrite`` is set.

    Returns 409 when the id or the name is already taken and the caller has not
    opted into overwriting.
    """
    _require_backend(container)
    assert container.script_backend is not None

    script_id = body.id or slugify(body.name)
    existing = await container.script_backend.get(script_id)
    if existing is None:
        by_name = await container.script_backend.get_by_name(body.name)
        existing = by_name
        if by_name is not None:
            script_id = by_name.id

    if existing is not None and not body.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Script '{existing.name or existing.id}' already exists",
        )

    defn = ScriptDefinition(
        id=script_id,
        name=body.name,
        description=body.description,
        code=body.code,
        created_at=existing.created_at if existing else None,
    )
    saved = await container.script_backend.create(defn)
    return saved.model_dump(mode="json")


# ─── Item routes ──────────────────────────────────────────────────────────────

@router.get("/{script_id}")
async def get_script(
    script_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Get one script by id."""
    _require_backend(container)
    assert container.script_backend is not None

    defn = await container.script_backend.get(script_id)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    return defn.model_dump(mode="json")


@router.put("/{script_id}")
async def update_script(
    script_id: str,
    body: ScriptUpdateRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Update a script in place.  The id never changes, even if the name does."""
    _require_backend(container)
    assert container.script_backend is not None

    existing = await container.script_backend.get(script_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")

    # A rename must not collide with a different script's name.
    clash = await container.script_backend.get_by_name(body.name)
    if clash is not None and clash.id != script_id:
        raise HTTPException(
            status_code=409, detail=f"Script '{body.name}' already exists",
        )

    existing.name = body.name
    existing.description = body.description
    existing.code = body.code
    saved = await container.script_backend.update(script_id, existing)
    return saved.model_dump(mode="json")


@router.delete("/{script_id}", status_code=204)
async def delete_script(
    script_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Delete a script.  Workflow steps referencing it will fail at run time."""
    _require_backend(container)
    assert container.script_backend is not None

    existing = await container.script_backend.get(script_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")
    await container.script_backend.delete(script_id)
