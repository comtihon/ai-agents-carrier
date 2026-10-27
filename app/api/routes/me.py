"""Who the backend thinks the caller is.

The UI uses it to decide whose resources are "mine", whether to offer the
visibility controls at all (with authentication off there is no owner, every
resource is public, and the controls mean nothing), whether "team" is an option,
and which team-management screens to show.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.application.teams import team_name
from app.core.config import get_settings
from app.infrastructure.auth.authorization import (
    Permission,
    get_current_permissions_or_none,
    get_current_subject,
)

router = APIRouter(tags=["me"])


@router.get("/me")
async def who_am_i() -> dict:
    subject = get_current_subject()
    is_super_admin = bool(subject and subject.is_super_admin)
    permissions = get_current_permissions_or_none() or frozenset()
    team = None
    if subject is not None and subject.team_id:
        team = {
            "id": subject.team_id,
            "name": team_name(subject.team_id),
            "role": subject.team_role or "member",
        }
    return {
        "sub": subject.id if subject else None,
        "name": subject.name if subject else None,
        # The ADMIN permission (unsandboxed python steps). Confers no visibility.
        "is_admin": Permission.ADMIN in permissions,
        "is_super_admin": is_super_admin,
        "auth_enabled": bool(get_settings().oauth_enabled),
        "team": team,
    }
