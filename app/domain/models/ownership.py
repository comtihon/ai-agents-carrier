from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, computed_field

Visibility = Literal["public", "team", "private"]
GrantAccess = Literal["read", "read_write"]
GranteeType = Literal["user", "workflow"]

# Computed per caller (or per lookup) on every read; never written to a store.
COMPUTED_FIELDS: frozenset[str] = frozenset({"can_edit", "team_name", "granted_access"})


class ResourceGrant(BaseModel):
    """Access to a resource for someone its visibility leaves out.

    Granted to a ``user`` — who may then see it and use it in their own private
    workflows and agents — or to one ``workflow``, which may then use it while
    whoever edits that workflow still cannot pick it anywhere else. ``access``
    says which operations: ``read`` ones only, or all of them (``read_write``).
    Neither lets the grantee change the resource itself. What a grant permits
    is app.application.visibility; who may grant is app.application.grants.
    """

    id: str
    grantee_type: GranteeType
    grantee_id: str
    # Display name at grant time (a user's name, a workflow's name).
    grantee_name: str | None = None
    access: GrantAccess = "read"
    granted_by: str | None = None
    granted_by_name: str | None = None
    granted_at: datetime | None = None


class OwnedResource(BaseModel):
    """Owner and visibility, shared by agents, data sources, scripts and workflows.

    The defaults are what a document stored before these fields existed loads
    as: public, owned by nobody, in no team — so no migration is needed. What
    the fields mean, and who may change them, is app.application.visibility;
    nothing that builds one of these models from a request may set ``owner_*``
    (the scoped backend overwrites them on every save). ``team_id`` may be
    requested, but the scoped backend decides whether the caller may use it.
    """

    # Team members may edit a team resource of this kind (workflows), rather
    # than only use it (agents, data sources, scripts).
    team_editable: ClassVar[bool] = False

    visibility: Visibility = "public"
    owner_id: str | None = None
    owner_name: str | None = None
    # The team a "team" resource belongs to; None otherwise. It stays with the
    # team when the owner leaves it.
    team_id: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def team_name(self) -> str | None:
        """Display name of ``team_id``, looked up so a rename shows everywhere."""
        from app.application.teams import team_name
        return team_name(self.team_id)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def can_edit(self) -> bool:
        """Whether the *current caller* may change this resource. Never stored."""
        from app.application.visibility import can_edit
        return can_edit(self)
