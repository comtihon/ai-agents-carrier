"""REST API for AgentDefinition CRUD.

Route ordering note
-------------------
Literal-segment routes (``/types``) are registered BEFORE parameterised routes
(``/{agent_id}``) so that Starlette does not swallow "types" as a path param.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException
from google.cloud import storage
from pydantic import BaseModel, Field

from app.api.dependencies import get_container
from app.core.config import get_settings
from app.core.container import ApplicationContainer
from app.domain.models.agent_addon import AnyAgentAddon
from app.domain.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


# ─── Request / response models ────────────────────────────────────────────────

class AgentDefinitionRequest(BaseModel):
    id: str
    name: str = ""
    description: str | None = None
    default_runtime: str = "local"
    # Sent to the agent on every run (system_prompt, model, tools, etc.)
    agent_input: dict = Field(default_factory=dict)
    # Docker-specific
    image: str | None = None
    # K8s-specific
    helm_chart: str | None = None
    helm_values: dict = Field(default_factory=dict)
    addons: list[dict] = Field(default_factory=list)


class AgentDefinitionUpdateRequest(BaseModel):
    name: str = ""
    description: str | None = None
    default_runtime: str = "local"
    agent_input: dict = Field(default_factory=dict)
    image: str | None = None
    helm_chart: str | None = None
    helm_values: dict = Field(default_factory=dict)
    # None = omitted by caller -> preserve existing addons on update.
    # Explicit [] = caller intentionally clears addons.
    addons: list[dict] | None = None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_backend(container: ApplicationContainer) -> None:
    if container.agent_backend is None:
        raise HTTPException(status_code=501, detail="Agent backend not configured")


# ─── Literal-segment routes (must come before /{agent_id}) ───────────────────

@router.get("/types")
async def list_agent_types():
    """Return the supported runtimes."""
    return {
        "runtimes": ["local", "docker", "k8s"],
    }


@router.get("/mcp-integrations")
async def list_mcp_integrations():
    """Return all known MCP servers with their enabled state."""
    settings = get_settings()
    return settings.list_mcp_candidates()


# ─── Collection routes ────────────────────────────────────────────────────────

@router.get("")
async def list_agents(
    container: ApplicationContainer = Depends(get_container),
):
    """List all registered agent definitions."""
    _require_backend(container)
    assert container.agent_backend is not None
    agents = await container.agent_backend.list()
    return [a.model_dump(mode="json") for a in agents]


@router.post("", status_code=201)
async def create_agent(
    body: AgentDefinitionRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Register a new agent definition."""
    _require_backend(container)
    assert container.agent_backend is not None

    existing = await container.agent_backend.get(body.id)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Agent '{body.id}' already exists")

    from pydantic import TypeAdapter
    _addon_adapter: TypeAdapter[list[AnyAgentAddon]] = TypeAdapter(list[AnyAgentAddon])
    defn = AgentDefinition(
        id=body.id,
        name=body.name,
        description=body.description,
        default_runtime=body.default_runtime,  # type: ignore[arg-type]
        agent_input=body.agent_input,
        image=body.image,
        helm_chart=body.helm_chart,
        helm_values=body.helm_values,
        addons=_addon_adapter.validate_python(body.addons),
    )
    saved = await container.agent_backend.create(defn)
    return saved.model_dump(mode="json")


# ─── Item routes (parameterised — must come AFTER literal-segment routes) ─────

@router.get("/{agent_id}")
async def get_agent(
    agent_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Get a specific agent definition by ID."""
    _require_backend(container)
    assert container.agent_backend is not None

    defn = await container.agent_backend.get(agent_id)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return defn.model_dump(mode="json")


@router.put("/{agent_id}")
async def update_agent(
    agent_id: str,
    body: AgentDefinitionUpdateRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Update an existing agent definition."""
    _require_backend(container)
    assert container.agent_backend is not None

    existing = await container.agent_backend.get(agent_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    from pydantic import TypeAdapter
    _addon_adapter: TypeAdapter[list[AnyAgentAddon]] = TypeAdapter(list[AnyAgentAddon])
    # body.addons is None when the caller omits the field entirely (e.g. a UI
    # form that only round-trips system_prompt/model) — preserve existing
    # addons in that case instead of silently wiping them. An explicit []
    # still clears addons, since that's a deliberate value, not an omission.
    addons = (
        existing.addons
        if body.addons is None
        else _addon_adapter.validate_python(body.addons)
    )
    defn = AgentDefinition(
        id=agent_id,
        name=body.name,
        description=body.description,
        default_runtime=body.default_runtime,  # type: ignore[arg-type]
        agent_input=body.agent_input,
        image=body.image,
        helm_chart=body.helm_chart,
        helm_values=body.helm_values,
        created_at=existing.created_at,
        addons=addons,
    )
    saved = await container.agent_backend.update(agent_id, defn)
    return saved.model_dump(mode="json")


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Delete an agent definition."""
    _require_backend(container)
    assert container.agent_backend is not None

    existing = await container.agent_backend.get(agent_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    await container.agent_backend.delete(agent_id)


@router.get("/{agent_id}/addons")
async def get_agent_addons(
    agent_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Return the addons list for an agent."""
    _require_backend(container)
    assert container.agent_backend is not None

    defn = await container.agent_backend.get(agent_id)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return {"addons": [addon.model_dump(mode="json") for addon in defn.addons]}


class AddonsUpdateRequest(BaseModel):
    addons: list[dict] = Field(default_factory=list)


@router.put("/{agent_id}/addons")
async def update_agent_addons(
    agent_id: str,
    body: AddonsUpdateRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Replace the addons list for an agent."""
    _require_backend(container)
    assert container.agent_backend is not None

    existing = await container.agent_backend.get(agent_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    from pydantic import TypeAdapter
    _addon_adapter: TypeAdapter[list[AnyAgentAddon]] = TypeAdapter(list[AnyAgentAddon])
    existing.addons = _addon_adapter.validate_python(body.addons)
    saved = await container.agent_backend.update(agent_id, existing)
    return saved.model_dump(mode="json")


# ─── S3 addon file browser helpers ───────────────────────────────────────────

async def _resolve_s3_addon(
    container: ApplicationContainer,
    agent_id: str,
    run_id: str,
) -> tuple[str, str]:
    """Load agent + run, resolve template placeholders in addon path.

    Returns (bucket, resolved_path).
    """
    _require_backend(container)
    assert container.agent_backend is not None

    defn = await container.agent_backend.get(agent_id)
    if defn is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    addon = defn.s3_addon
    if addon is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' has no S3 addon")

    run = await container.run_repository.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    path = addon.path.replace("{workflow_id}", run.id or "")
    path = path.replace("{project_id}", (run.state or {}).get("project_id", ""))
    return (addon.bucket, path)


@router.get("/{agent_id}/s3/files")
async def list_s3_addon_files(
    agent_id: str,
    run_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """List files in the GCS bucket/path resolved from the agent's S3 addon."""
    bucket_name, path = await _resolve_s3_addon(container, agent_id, run_id)

    if not path or not path.strip():
        return {"bucket": bucket_name, "path": path, "files": []}

    def _list() -> list[dict]:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=path))
        return [
            {
                "name": blob.name.split("/")[-1],
                "path": blob.name,
                "size": blob.size,
                "updated": blob.updated.isoformat() if blob.updated else None,
            }
            for blob in blobs
        ]

    files = await anyio.to_thread.run_sync(_list)
    return {"bucket": bucket_name, "path": path, "files": files}


@router.get("/{agent_id}/s3/download")
async def get_s3_signed_url(
    agent_id: str,
    run_id: str,
    file_path: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Generate a short-lived signed URL for a GCS file."""
    bucket_name, resolved_path = await _resolve_s3_addon(container, agent_id, run_id)

    if not resolved_path:
        raise HTTPException(status_code=400, detail="resolved path is empty; cannot validate file_path")
    if not file_path.startswith(resolved_path):
        raise HTTPException(status_code=400, detail="file_path is outside the allowed prefix")

    def _sign() -> str:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"File '{file_path}' not found")
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET",
        )

    signed_url = await anyio.to_thread.run_sync(_sign)
    return {"url": signed_url, "expires_in": 900}
