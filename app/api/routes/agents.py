"""REST API for AgentDefinition CRUD.

Route ordering note
-------------------
Literal-segment routes (``/types``) are registered BEFORE parameterised routes
(``/{agent_id}``) so that Starlette does not swallow "types" as a path param.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
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
    addons: list[dict] = Field(default_factory=list)


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
        addons=_addon_adapter.validate_python(body.addons),
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
