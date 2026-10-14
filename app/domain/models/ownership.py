from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, computed_field

Visibility = Literal["public", "team", "private"]

# Computed per caller (or per lookup) on every read; never written to a store.
COMPUTED_FIELDS: frozenset[str] = frozenset({"can_edit", "team_name"})


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
