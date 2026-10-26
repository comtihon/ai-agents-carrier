"""Granting a data source, agent or workflow to a user or a workflow outside
its visibility.

A grant is stored on the resource itself (``GrantableResource.grants``), so
the resource always knows who it has been opened to and any grant can be
revoked from there. What a grant permits is app.application.visibility; this
module decides who may grant and to whom:

* **Who grants** — whoever may change the resource's visibility: its owner, a
  super admin, or an admin of the team it is shared with
  (``can_change_visibility``). A grantee, even with ``read_write``, may not
  grant further.
* **To a user** — someone the resource's scope leaves out: not its owner, not
  a member of its team. A public resource needs no grants at all.
* **To a workflow** — one whose scope does not already cover the resource, so
  the grant is what lets it use the resource (a team A workflow calling team
  B's agent or workflow). Granting to a workflow names it by id: the granter
  may not be able to see it (it may be private to whoever asked for the
  grant). On an agent or a workflow, a workflow grant is use only (``read``):
  editing is for users.

Granting to someone who already holds a grant changes its access level.
Revoking is always allowed to a granter and takes effect on the next save or
run that relied on the grant — nothing that used it is rewritten.

Every function takes a *holder* — the application container or the
management tools' deps — and reads ``data_source_backend``, ``agent_backend``,
``workflow_backend``, ``team_directory`` and ``refresh_runner`` off it. Grants
are written to the *raw* backend: saving through the scoped one restamps the
stored grants, which is what keeps a PUT from changing them.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.application.visibility import (
    _LABEL,
    NotOwner,
    ResourceNotFound,
    VisibilityError,
    VisibilityScopedBackend,
    _team,
    _visibility,
    can_change_visibility,
    can_see,
    grants_of,
    unwrap,
)
from app.domain.models.ownership import ResourceGrant
from app.infrastructure.auth.authorization import get_current_subject


class GrantInvalid(VisibilityError):
    status_code = 422


class GrantConflict(VisibilityError):
    status_code = 409


KINDS = ("datasource", "agent", "workflow")
ACCESS_LEVELS = ("read", "read_write")

_BACKEND = {
    "datasource": "data_source_backend",
    "agent": "agent_backend",
    "workflow": "workflow_backend",
}


def _backend(holder: Any, kind: str) -> Any:
    if kind not in _BACKEND:
        raise GrantInvalid(f"Only {', '.join(KINDS)} can be granted")
    backend = getattr(holder, _BACKEND[kind], None)
    if backend is None:
        raise ResourceNotFound(f"No {_LABEL[kind]} backend is configured")
    return backend


def _guard(holder: Any) -> Any:
    """The reference guard behind the scoped backends, if they are scoped."""
    for name in _BACKEND.values():
        backend = getattr(holder, name, None)
        if isinstance(backend, VisibilityScopedBackend):
            return backend._guard
    return None


async def _resource_to_share(holder: Any, kind: str, resource_id: str) -> Any:
    """The raw resource, if the caller may manage its grants."""
    label = _LABEL[kind] if kind in _LABEL else kind
    resource = await unwrap(_backend(holder, kind)).get(resource_id)
    if resource is None or not can_see(resource):
        raise ResourceNotFound(f"{label.capitalize()} '{resource_id}' not found")
    if not can_change_visibility(resource):
        raise NotOwner(
            f"Only the owner of {label} '{resource_id}', an admin of its team, "
            f"or a super admin may share it"
        )
    if getattr(resource, "readonly", False):
        raise GrantInvalid(f"{label.capitalize()} '{resource_id}' is read-only and cannot be shared")
    return resource


def grant_dict(grant: Any) -> dict[str, Any]:
    return grant.model_dump(mode="json")


async def list_grants(holder: Any, kind: str, resource_id: str) -> list[dict[str, Any]]:
    resource = await _resource_to_share(holder, kind, resource_id)
    return [grant_dict(g) for g in grants_of(resource)]


async def _check_user_grantee(holder: Any, resource: Any, user_id: str) -> str | None:
    """The grantee's display name; refuses someone who needs no grant."""
    if user_id == resource.owner_id:
        raise GrantConflict("The owner already has full access")
    directory = getattr(holder, "team_directory", None)
    user = await directory.get_user(user_id) if directory is not None else None
    if directory is not None and user is None:
        raise GrantInvalid(f"User '{user_id}' not found")
    team = _team(resource)
    if team is not None and user is not None and user.team_id == team:
        raise GrantConflict(f"User '{user_id}' is in its team and can already use it")
    return (user.name or user.email) if user is not None else None


async def _check_workflow_grantee(
    holder: Any, kind: str, resource: Any, workflow_id: str,
) -> str | None:
    if kind == "workflow" and workflow_id == resource.id:
        raise GrantConflict("A workflow can always use itself")
    workflows = getattr(holder, "workflow_backend", None)
    workflow = await unwrap(workflows).get(workflow_id) if workflows is not None else None
    if workflow is None:
        raise GrantInvalid(f"Workflow '{workflow_id}' not found")
    guard = _guard(holder)
    # Scope alone, ignoring any grant already there (which is being replaced).
    if guard is not None and await guard._scope_violation(workflow, "workflow", resource, kind) is None:
        raise GrantConflict(f"Workflow '{workflow_id}' can already use this {_LABEL[kind]}")
    # The granter may not see the workflow; its name is only shown back to
    # them if they can.
    return (workflow.name or workflow_id) if can_see(workflow) else None


async def _save(holder: Any, kind: str, resource: Any) -> None:
    await unwrap(_backend(holder, kind)).update(resource.id, resource)
    if kind == "workflow":
        # The registry's listing carries the grants; rebuild the runner.
        refresh = getattr(holder, "refresh_runner", None)
        if refresh is not None:
            await refresh(resource.id)


async def grant(
    holder: Any,
    kind: str,
    resource_id: str,
    *,
    grantee_type: str,
    grantee_id: str,
    access: str = "read",
) -> dict[str, Any]:
    resource = await _resource_to_share(holder, kind, resource_id)
    grantee_id = (grantee_id or "").strip()
    if not grantee_id:
        raise GrantInvalid("Say who to grant access to")
    if access not in ACCESS_LEVELS:
        raise GrantInvalid(f"access must be one of {', '.join(ACCESS_LEVELS)}")
    if _visibility(resource) == "public":
        raise GrantConflict(f"A public {_LABEL[kind]} is open to everyone; it needs no grants")
    if grantee_type == "user":
        name = await _check_user_grantee(holder, resource, grantee_id)
    elif grantee_type == "workflow":
        if kind != "datasource" and access != "read":
            raise GrantInvalid(
                f"A workflow can only use a {_LABEL[kind]} (access 'read'); "
                f"grant read_write to a user to let them edit it"
            )
        name = await _check_workflow_grantee(holder, kind, resource, grantee_id)
    else:
        raise GrantInvalid("grantee_type must be 'user' or 'workflow'")

    subject = get_current_subject()
    existing = [
        g for g in grants_of(resource)
        if g.grantee_type == grantee_type and g.grantee_id == grantee_id
    ]
    record = ResourceGrant(
        id=existing[0].id if existing else uuid.uuid4().hex[:12],
        grantee_type=grantee_type,  # type: ignore[arg-type]
        grantee_id=grantee_id,
        grantee_name=name,
        access=access,  # type: ignore[arg-type]
        granted_by=subject.id if subject is not None else None,
        granted_by_name=subject.name if subject is not None else None,
        granted_at=datetime.now(timezone.utc),
    )
    resource.grants = [g for g in grants_of(resource) if g not in existing] + [record]
    await _save(holder, kind, resource)
    return grant_dict(record)


async def revoke(holder: Any, kind: str, resource_id: str, grant_id: str) -> dict[str, Any]:
    """Revoke one grant; returns it."""
    resource = await _resource_to_share(holder, kind, resource_id)
    gone = [g for g in grants_of(resource) if g.id == grant_id]
    if not gone:
        raise ResourceNotFound(f"Grant '{grant_id}' not found")
    resource.grants = [g for g in grants_of(resource) if g.id != grant_id]
    await _save(holder, kind, resource)
    return grant_dict(gone[0])


async def find_grant(
    holder: Any, kind: str, resource_id: str, grantee_type: str, grantee_id: str,
) -> dict[str, Any] | None:
    for g in await list_grants(holder, kind, resource_id):
        if g["grantee_type"] == grantee_type and g["grantee_id"] == grantee_id:
            return g
    return None


async def candidates(holder: Any, kind: str, resource_id: str) -> dict[str, list[dict[str, Any]]]:
    """Who the caller could grant this resource to, for a picker.

    Users: everyone the directory knows except the owner and the resource's
    team. Workflows: those the caller can see whose scope does not already
    cover the resource. A workflow the caller cannot see is granted by id.
    """
    resource = await _resource_to_share(holder, kind, resource_id)
    team = _team(resource)
    users: list[dict[str, Any]] = []
    directory = getattr(holder, "team_directory", None)
    if directory is not None:
        for user in await directory.list_users():
            if user.sub == resource.owner_id or (team is not None and user.team_id == team):
                continue
            users.append({"id": user.sub, "name": user.name, "email": user.email})
    users.sort(key=lambda u: (u["name"] or u["email"] or u["id"]).lower())

    workflows: list[dict[str, Any]] = []
    backend = getattr(holder, "workflow_backend", None)
    guard = _guard(holder)
    if backend is not None:
        for workflow in await backend.list():
            if getattr(workflow, "readonly", False):
                continue
            if kind == "workflow" and workflow.id == resource.id:
                continue
            if guard is not None and await guard._scope_violation(
                workflow, "workflow", resource, kind
            ) is None:
                continue
            workflows.append({
                "id": workflow.id,
                "name": workflow.name or workflow.id,
                "owner_name": workflow.owner_name,
                "visibility": _visibility(workflow),
            })
    workflows.sort(key=lambda w: w["name"].lower())
    return {"users": users, "workflows": workflows}
