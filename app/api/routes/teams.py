"""Teams and users. Who may do what is app.application.teams."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.application import teams
from app.core.container import ApplicationContainer

router = APIRouter(tags=["teams"])


class TeamRequest(BaseModel):
    name: str


class MemberRequest(BaseModel):
    role: Literal["member", "admin"] = "member"


def _directory(container: ApplicationContainer):
    directory = teams.directory_of(container)
    if directory is None:
        raise HTTPException(status_code=503, detail="Teams are not available: no user directory")
    return directory


@router.get("/teams")
async def list_teams(container: ApplicationContainer = Depends(get_container)) -> list[dict]:
    """Every team for a super admin; the caller's own team (or nothing) otherwise."""
    return await teams.list_teams(_directory(container))


@router.post("/teams", status_code=201)
async def create_team(
    body: TeamRequest, container: ApplicationContainer = Depends(get_container)
) -> dict:
    return await teams.create_team(_directory(container), body.name)


@router.put("/teams/{team_id}")
async def rename_team(
    team_id: str, body: TeamRequest, container: ApplicationContainer = Depends(get_container)
) -> dict:
    return await teams.rename_team(_directory(container), team_id, body.name)


@router.delete("/teams/{team_id}", status_code=204)
async def delete_team(
    team_id: str, container: ApplicationContainer = Depends(get_container)
) -> Response:
    await teams.delete_team(
        _directory(container),
        team_id,
        [
            container.agent_backend,
            container.data_source_backend,
            container.script_backend,
            container.workflow_backend,
        ],
    )
    return Response(status_code=204)


@router.put("/teams/{team_id}/members/{sub}")
async def set_member(
    team_id: str,
    sub: str,
    body: MemberRequest | None = Body(default=None),
    container: ApplicationContainer = Depends(get_container),
) -> dict:
    """Add a user to the team, or change their role in it (default "member")."""
    return await teams.set_member(
        _directory(container), team_id, sub, body.role if body else None
    )


@router.delete("/teams/{team_id}/members/{sub}")
async def remove_member(
    team_id: str, sub: str, container: ApplicationContainer = Depends(get_container)
) -> dict:
    return await teams.remove_member(_directory(container), team_id, sub)


@router.get("/users")
async def list_users(container: ApplicationContainer = Depends(get_container)) -> list[dict]:
    """Super admin: everyone. Team admin: users in no team plus their own team."""
    return await teams.list_users(_directory(container))
