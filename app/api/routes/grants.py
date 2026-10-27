"""Grant routes, shared by data sources, agents and workflows.

Mounted under each resource's own router (``/{id}/grants``…) by
:func:`add_grant_routes`; who may do what is app.application.grants.
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.application import grants as grant_service
from app.application.visibility import can_change_visibility, workflow_grant
from app.core.container import ApplicationContainer


class GrantRequest(BaseModel):
    grantee_type: Literal["user", "workflow"]
    grantee_id: str
    access: Literal["read", "read_write"] = "read"


def present_sharing(
    payload: dict[str, Any], resource: Any, workflow_id: str | None = None,
) -> dict[str, Any]:
    """Grant fields of a resource as the caller may read them: the grant list
    only for someone who may manage it (a grantee learns their own access from
    ``granted_access``, not who else holds one)."""
    payload["can_share"] = can_change_visibility(resource)
    if not payload["can_share"]:
        payload.pop("grants", None)
    if workflow_id and workflow_grant(resource, workflow_id):
        payload["granted_to_workflow"] = workflow_grant(resource, workflow_id)
    return payload


def add_grant_routes(router: APIRouter, kind: str) -> None:
    """``GET/POST /{id}/grants``, ``GET /{id}/grants/candidates`` and
    ``DELETE /{id}/grants/{grant_id}`` for one kind."""

    @router.get("/{resource_id}/grants", name=f"list_{kind}_grants")
    async def list_grants(
        resource_id: str, container: ApplicationContainer = Depends(get_container)
    ) -> list[dict[str, Any]]:
        """Who this resource has been opened to beyond its visibility."""
        return await grant_service.list_grants(container, kind, resource_id)

    @router.get("/{resource_id}/grants/candidates", name=f"list_{kind}_grant_candidates")
    async def list_candidates(
        resource_id: str, container: ApplicationContainer = Depends(get_container)
    ) -> dict[str, list[dict[str, Any]]]:
        """Users and workflows the caller could grant this resource to."""
        return await grant_service.candidates(container, kind, resource_id)

    @router.post("/{resource_id}/grants", status_code=201, name=f"create_{kind}_grant")
    async def create_grant(
        resource_id: str,
        body: GrantRequest,
        container: ApplicationContainer = Depends(get_container),
    ) -> dict[str, Any]:
        """Grant a user or a workflow access. Granting to one that already
        holds a grant changes its access."""
        return await grant_service.grant(
            container, kind, resource_id,
            grantee_type=body.grantee_type, grantee_id=body.grantee_id, access=body.access,
        )

    @router.delete("/{resource_id}/grants/{grant_id}", status_code=204, name=f"revoke_{kind}_grant")
    async def revoke_grant(
        resource_id: str, grant_id: str, container: ApplicationContainer = Depends(get_container)
    ):
        """Revoke a grant. Whatever relied on it fails on its next save or run."""
        await grant_service.revoke(container, kind, resource_id, grant_id)
        return Response(status_code=204)
