"""Granting a data source to a user or a workflow outside its visibility.

A grant is stored on the data source itself (``DataSourceDefinition.grants``),
so the source always knows who it has been opened to and any grant can be
revoked from there. What a grant permits is app.application.visibility; this
module decides who may grant and to whom:

* **Who grants** — whoever may change the source's visibility: its owner, a
  super admin, or an admin of the team it is shared with
  (``can_change_visibility``). A grantee, even with ``read_write``, may not
  grant further.
* **To a user** — someone the source's scope leaves out: not its owner, not a
  member of its team. A public source needs no grants at all.
* **To a workflow** — one whose scope does not already cover the source, so
  the grant is what lets it use the source. Granting to a workflow names it
  by id: the granter may not be able to see it (it may be private to whoever
  asked for the grant).

Granting to someone who already holds a grant changes its access level.
Revoking is always allowed to a granter and takes effect on the next save or
run that relied on the grant — nothing that used it is rewritten.

Grants are written to the *raw* backend: saving through the scoped one
restamps the stored grants, which is what keeps a PUT from changing them.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.application.visibility import (
    NotOwner,
    ResourceNotFound,
    VisibilityError,
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


ACCESS_LEVELS = ("read", "read_write")


async def _source_to_share(container: Any, source_id: str) -> Any:
    """The raw source, if the caller may manage its grants."""
    backend = getattr(container, "data_source_backend", None)
    if backend is None:
        raise ResourceNotFound("Data sources are not configured")
    source = await unwrap(backend).get(source_id)
    if source is None or not can_see(source):
        raise ResourceNotFound(f"Data source '{source_id}' not found")
    if not can_change_visibility(source):
        raise NotOwner(
            f"Only the owner of data source '{source_id}', an admin of its team, "
            f"or a super admin may share it"
        )
    return source


def grant_dict(grant: Any) -> dict[str, Any]:
    return grant.model_dump(mode="json")


async def list_grants(container: Any, source_id: str) -> list[dict[str, Any]]:
    source = await _source_to_share(container, source_id)
    return [grant_dict(g) for g in grants_of(source)]


async def _check_user_grantee(container: Any, source: Any, user_id: str) -> str | None:
    """The grantee's display name; refuses someone who needs no grant."""
    if user_id == source.owner_id:
        raise GrantConflict("The owner already has full access")
    directory = getattr(container, "team_directory", None)
    user = await directory.get_user(user_id) if directory is not None else None
    if directory is not None and user is None:
        raise GrantInvalid(f"User '{user_id}' not found")
    team = _team(source)
    if team is not None and user is not None and user.team_id == team:
        raise GrantConflict(
            f"User '{user_id}' is in the source's team and can already use it"
        )
    return (user.name or user.email) if user is not None else None


async def _check_workflow_grantee(container: Any, source: Any, workflow_id: str) -> str | None:
    workflows = getattr(container, "workflow_backend", None)
    workflow = await unwrap(workflows).get(workflow_id) if workflows is not None else None
    if workflow is None:
        raise GrantInvalid(f"Workflow '{workflow_id}' not found")
    guard = container._visibility_guard() if hasattr(container, "_visibility_guard") else None
    if guard is not None:
        # Scope alone, ignoring any grant already there (which is being replaced).
        if await guard._scope_violation(workflow, "workflow", source, "datasource") is None:
            raise GrantConflict(
                f"Workflow '{workflow_id}' can already use this data source"
            )
    # The granter may not see the workflow; its name is only shown back to
    # them if they can.
    return (workflow.name or workflow_id) if can_see(workflow) else None


async def grant(
    container: Any,
    source_id: str,
    *,
    grantee_type: str,
    grantee_id: str,
    access: str = "read",
) -> dict[str, Any]:
    source = await _source_to_share(container, source_id)
    grantee_id = (grantee_id or "").strip()
    if not grantee_id:
        raise GrantInvalid("Say who to grant access to")
    if access not in ACCESS_LEVELS:
        raise GrantInvalid(f"access must be one of {', '.join(ACCESS_LEVELS)}")
    if _visibility(source) == "public":
        raise GrantConflict("A public data source is open to everyone; it needs no grants")
    if grantee_type == "user":
        name = await _check_user_grantee(container, source, grantee_id)
    elif grantee_type == "workflow":
        name = await _check_workflow_grantee(container, source, grantee_id)
    else:
        raise GrantInvalid("grantee_type must be 'user' or 'workflow'")

    subject = get_current_subject()
    existing = [
        g for g in grants_of(source)
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
    source.grants = [g for g in grants_of(source) if g not in existing] + [record]
    await unwrap(container.data_source_backend).update(source.id, source)
    return grant_dict(record)


async def revoke(container: Any, source_id: str, grant_id: str) -> None:
    source = await _source_to_share(container, source_id)
    kept = [g for g in grants_of(source) if g.id != grant_id]
    if len(kept) == len(grants_of(source)):
        raise ResourceNotFound(f"Grant '{grant_id}' not found")
    source.grants = kept
    await unwrap(container.data_source_backend).update(source.id, source)


async def candidates(container: Any, source_id: str) -> dict[str, list[dict[str, Any]]]:
    """Who the caller could grant this source to, for a picker.

    Users: everyone the directory knows except the owner and the source's
    team. Workflows: those the caller can see whose scope does not already
    cover the source. A workflow the caller cannot see is granted by id.
    """
    source = await _source_to_share(container, source_id)
    team = _team(source)
    users: list[dict[str, Any]] = []
    directory = getattr(container, "team_directory", None)
    if directory is not None:
        for user in await directory.list_users():
            if user.sub == source.owner_id or (team is not None and user.team_id == team):
                continue
            users.append({"id": user.sub, "name": user.name, "email": user.email})
    users.sort(key=lambda u: (u["name"] or u["email"] or u["id"]).lower())

    workflows: list[dict[str, Any]] = []
    backend = getattr(container, "workflow_backend", None)
    guard = container._visibility_guard() if hasattr(container, "_visibility_guard") else None
    if backend is not None:
        for workflow in await backend.list():
            if getattr(workflow, "readonly", False):
                continue
            if guard is not None and await guard._scope_violation(
                workflow, "workflow", source, "datasource"
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
