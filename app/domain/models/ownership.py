from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, computed_field


class OwnedResource(BaseModel):
    """Owner and visibility, shared by agents, data sources, scripts and workflows.

    The defaults are what a document stored before these fields existed loads
    as: public, owned by nobody — so no migration is needed. What the fields
    mean, and who may change them, is app.application.visibility; nothing that
    builds one of these models from a request may set ``owner_*`` (the scoped
    backend overwrites them on every save).
    """

    visibility: Literal["public", "private"] = "public"
    owner_id: str | None = None
    owner_name: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def can_edit(self) -> bool:
        """Whether the *current caller* may change this resource. Never stored."""
        from app.application.visibility import can_edit
        return can_edit(self)
