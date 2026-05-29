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
