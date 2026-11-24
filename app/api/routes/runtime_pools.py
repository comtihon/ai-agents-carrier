"""Runtime pools (app.application.runtime_pools).

    GET    /runtime-pools                         pools the caller can see, with live member status
    POST   /runtime-pools                         {id, name, description?, visibility?, team_id?}
    GET    /runtime-pools/{id}
    PUT    /runtime-pools/{id}                    {name?, description?, visibility?, team_id?}
    DELETE /runtime-pools/{id}                    refused while an agent uses it
    POST   /runtime-pools/{id}/members            {agent_id} — the caller's own runner agent joins
    DELETE /runtime-pools/{id}/members/{agent_id} the agent's owner, the pool's editors or its team's admins
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.application.runners import get_runner_hub
from app.application.runtime_pools import add_member, present, remove_member
from app.application.visibility import unwrap, visibility_kwargs
from app.core.container import ApplicationContainer
from app.domain.models.runtime_pool import RuntimePool

router = APIRouter(prefix="/runtime-pools", tags=["runtime-pools"])


def _backend(container: ApplicationContainer) -> Any:
    backend = getattr(container, "runtime_pool_backend", None)
    if backend is None:
        raise HTTPException(status_code=404, detail="Runtime pools are not enabled")
    return backend


async def _pool(container: ApplicationContainer, pool_id: str) -> RuntimePool:
    pool = await _backend(container).get(pool_id)
    if pool is None:
        raise HTTPException(status_code=404, detail=f"Runtime pool '{pool_id}' not found")
    return pool


async def _present(container: ApplicationContainer, pool: RuntimePool) -> dict[str, Any]:
    return await present(
        pool, agent_backend=container.agent_backend, hub=get_runner_hub(),
        pool_backend=unwrap(_backend(container)), run_repository=container.run_repository,
    )


class PoolCreate(BaseModel):
    id: str
    name: str = ""
    description: str | None = None
    visibility: Literal["private", "team", "public"] | None = None
    team_id: str | None = None


class PoolUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    visibility: Literal["private", "team", "public"] | None = None
    team_id: str | None = None


class MemberAdd(BaseModel):
    agent_id: str


@router.get("")
async def list_pools(container: ApplicationContainer = Depends(get_container)):
    return [await _present(container, p) for p in await _backend(container).list()]


@router.post("", status_code=201)
async def create_pool(body: PoolCreate, container: ApplicationContainer = Depends(get_container)):
    backend = _backend(container)
    if await backend.get(body.id) is not None:
        raise HTTPException(status_code=409, detail=f"Runtime pool '{body.id}' already exists")
    pool = RuntimePool(
        id=body.id, name=body.name, description=body.description,
        **visibility_kwargs(body.visibility, body.team_id),
    )
    return await _present(container, await backend.create(pool))


@router.get("/{pool_id}")
async def get_pool(pool_id: str, container: ApplicationContainer = Depends(get_container)):
    return await _present(container, await _pool(container, pool_id))


@router.put("/{pool_id}")
async def update_pool(pool_id: str, body: PoolUpdate, container: ApplicationContainer = Depends(get_container)):
    existing = await _pool(container, pool_id)
    pool = RuntimePool(
        id=pool_id,
        name=existing.name if body.name is None else body.name,
        description=existing.description if body.description is None else (body.description or None),
        created_at=existing.created_at,
        **visibility_kwargs(body.visibility, body.team_id),
    )
    await _backend(container).update(pool_id, pool)
    return await _present(container, await _pool(container, pool_id))


@router.delete("/{pool_id}", status_code=204)
async def delete_pool(pool_id: str, container: ApplicationContainer = Depends(get_container)):
    await _pool(container, pool_id)
    agents = await unwrap(container.agent_backend).list() if container.agent_backend is not None else []
    users = [a.id for a in agents if a.default_runtime == "pool" and a.pool_id == pool_id]
    if users:
        raise HTTPException(
            status_code=409,
            detail=f"Runtime pool '{pool_id}' is used by agent(s) {', '.join(users)}; point them elsewhere first",
        )
    await _backend(container).delete(pool_id)
    return Response(status_code=204)


@router.post("/{pool_id}/members", status_code=201)
async def join_pool(pool_id: str, body: MemberAdd, container: ApplicationContainer = Depends(get_container)):
    pool = await _pool(container, pool_id)
    agent = await container.agent_backend.get(body.agent_id) if container.agent_backend is not None else None
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{body.agent_id}' not found")
    await add_member(unwrap(_backend(container)), pool, agent)
    return await _present(container, await _pool(container, pool_id))


@router.delete("/{pool_id}/members/{agent_id}")
async def leave_pool(pool_id: str, agent_id: str, container: ApplicationContainer = Depends(get_container)):
    pool = await _pool(container, pool_id)
    agent = await unwrap(container.agent_backend).get(agent_id) if container.agent_backend is not None else None
    await remove_member(unwrap(_backend(container)), pool, agent_id, agent)
    return await _present(container, await _pool(container, pool_id))
