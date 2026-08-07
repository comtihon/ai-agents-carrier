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
    """Non-secret description of one outbound service identity."""

    name: str
    # Usable right now: every field needed to mint a token is present.
    configured: bool
    # Client id the assertion is signed as (a subject id, not a secret).
    client_id: str | None = None
    token_url: str | None = None
    audience: str | None = None
    scopes: str | None = None
    # Why it is unusable, when it is not.
    error: str | None = None


class ServiceIdentitiesResponse(BaseModel):
    """Every configured outbound identity, plus which one is the default."""

    enabled: bool
    identities: list[ServiceIdentityInfo] = []
    # Used when an auth block names no identity; null when the deployment has
    # several and has not designated one.
    default_identity: str | None = None
    # Set when the identities cannot be read at all (e.g. malformed JSON).
    error: str | None = None


@router.get("/service-identities", response_model=ServiceIdentitiesResponse)
def list_service_identities(
    settings: Settings = Depends(get_settings),
) -> ServiceIdentitiesResponse:
    """List the outbound identities available to ``service_identity`` auth.

    The secret behind each is a signing key that never leaves the backend, so
    this reports only what a caller needs in order to pick one: its name,
    whether it is usable, which subject it authenticates as, and against which
    authorization server. Key material and key ids are deliberately omitted.
    """
    from app.infrastructure.auth.service_token_provider import ServiceTokenProvider

    enabled = bool(settings.service_auth_enabled)
    if not enabled:
        return ServiceIdentitiesResponse(
            enabled=False, error="SERVICE_AUTH_ENABLED is not set on this backend"
        )

    provider = ServiceTokenProvider(settings)
    try:
        configured = settings.get_service_identities()
    except ValueError as exc:
        return ServiceIdentitiesResponse(
            enabled=True, error=f"SERVICE_AUTH_IDENTITIES is not valid: {exc}"
        )

    identities: list[ServiceIdentityInfo] = []
    for identity in configured:
        _, error = provider.describe(identity.name)
        identities.append(
            ServiceIdentityInfo(
                name=identity.name,
                configured=error is None,
                client_id=identity.client_id or None,
                token_url=identity.token_url or None,
                audience=identity.audience or None,
                scopes=identity.scopes,
                error=error,
            )
        )

    return ServiceIdentitiesResponse(
        enabled=True,
        identities=identities,
        default_identity=settings.resolved_default_service_identity(),
        error=None if identities else "No service identity is configured",
    )
