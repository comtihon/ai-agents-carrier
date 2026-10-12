"""Who the backend thinks the caller is.

The UI uses it to decide whose resources are "mine" and whether to offer the
visibility controls at all: with authentication off there is no owner, every
resource is public, and the controls mean nothing.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.infrastructure.auth.authorization import get_current_subject

router = APIRouter(tags=["me"])


@router.get("/me")
async def who_am_i() -> dict:
    subject = get_current_subject()
    return {
        "sub": subject.id if subject else None,
        "name": subject.name if subject else None,
        "is_admin": bool(subject and subject.is_admin),
        "auth_enabled": bool(get_settings().oauth_enabled),
    }
