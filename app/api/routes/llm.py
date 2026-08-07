from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/llm", tags=["llm"])


class LLMProviderInfo(BaseModel):
    name: str
    default_model: str


class LLMProvidersResponse(BaseModel):
    providers: list[LLMProviderInfo]
    default_provider: str | None = None


@router.get("/providers", response_model=LLMProvidersResponse)
async def list_providers(settings: Settings = Depends(get_settings)) -> LLMProvidersResponse:
    """List configured LLM provider integrations."""
    integrations = settings.get_llm_integrations()
    return LLMProvidersResponse(
        providers=[
            LLMProviderInfo(name=i.name, default_model=i.default_model)
            for i in integrations
        ],
        default_provider=settings.llm_provider,
    )


@router.get("/config/keys")
def get_config_keys(settings: Settings = Depends(get_settings)) -> dict:
    """Return names (not values) of forwardable config keys that are currently set."""
    return {"keys": list(settings.get_forwardable_config().keys())}


class ServiceIdentityInfo(BaseModel):
    """Non-secret description of the backend's own outbound service identity."""

    enabled: bool
    # Usable right now: enabled and every required setting present.
    configured: bool
    # Zitadel machine-user id the assertion is signed as (a subject id, not a secret).
    client_id: str | None = None
    token_url: str | None = None
    audience: str | None = None
    scopes: str | None = None
    # Why it is unusable, when it is not.
    error: str | None = None


@router.get("/service-identity", response_model=ServiceIdentityInfo)
def get_service_identity(settings: Settings = Depends(get_settings)) -> ServiceIdentityInfo:
    """Describe the outbound service identity used by ``service_identity`` auth.

    There is exactly one per deployment, and the secret behind it is a signing
    key that never leaves the backend — so this reports only what a caller needs
    in order to decide whether selecting ``service_identity`` will work: whether
    it is configured, which machine user it authenticates as, and against which
    authorization server.
    """
    from app.infrastructure.auth.service_token_provider import (
        ServiceAuthError,
        ServiceTokenProvider,
    )

    error: str | None = None
    if not settings.service_auth_enabled:
        error = "SERVICE_AUTH_ENABLED is not set on this backend"
    else:
        try:
            ServiceTokenProvider(settings).validate_configuration()
        except ServiceAuthError as exc:
            error = exc.message

    return ServiceIdentityInfo(
        enabled=bool(settings.service_auth_enabled),
        configured=error is None,
        client_id=settings.service_auth_client_id,
        token_url=settings.service_auth_token_url,
        audience=settings.service_auth_audience,
        scopes=settings.service_auth_scopes,
        error=error,
    )
