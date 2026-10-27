"""The login script, editable at runtime by super admins.

How it is chosen, validated and picked up is app.application.login_admin.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.dependencies import get_container
from app.application import login_admin
from app.application.login import login_script_store_of
from app.application.teams import directory_of
from app.core.container import ApplicationContainer
from app.infrastructure.auth.authorization import AuthorizationPolicy

router = APIRouter(prefix="/auth/login-script", tags=["auth"])


class LoginScriptRequest(BaseModel):
    # None or blank disables the stored script.
    script: str | None = None


class LoginScriptTestRequest(BaseModel):
    # Omitted: test the active script.
    script: str | None = None
    claims: dict[str, Any] = Field(default_factory=dict)
    # A stored user to pass as ctx.user; omitted: ctx.user is null.
    sub: str | None = None


def _settings(request: Request, container: ApplicationContainer) -> Any:
    return getattr(request.app.state, "settings", None) or container.settings


def _policy(request: Request, settings: Any) -> AuthorizationPolicy:
    return getattr(request.app.state, "authz_policy", None) or AuthorizationPolicy.from_settings(
        settings
    )


@router.get("")
async def get_login_script(
    request: Request, container: ApplicationContainer = Depends(get_container)
) -> dict:
    return await login_admin.describe(
        _settings(request, container), login_script_store_of(container)
    )


@router.put("")
async def put_login_script(
    body: LoginScriptRequest,
    request: Request,
    container: ApplicationContainer = Depends(get_container),
) -> dict:
    return await login_admin.save(
        _settings(request, container),
        login_script_store_of(container),
        getattr(request.app.state, "login_service", None),
        body.script,
    )


@router.post("/test")
async def test_login_script(
    body: LoginScriptTestRequest,
    request: Request,
    container: ApplicationContainer = Depends(get_container),
) -> dict:
    settings = _settings(request, container)
    return await login_admin.dry_run(
        settings,
        login_script_store_of(container),
        directory_of(container),
        _policy(request, settings),
        script=body.script,
        claims=body.claims,
        sub=body.sub,
    )
