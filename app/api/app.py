from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.auth import OAuthMiddleware
from app.api.routes.graphs import router as graphs_router
from app.api.routes.health import router as health_router
from app.api.routes.workflows import router as workflows_router
from app.core.config import get_settings
from app.core.container import ApplicationContainer, build_container
from app.infrastructure.auth.auth_service import AuthService


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = build_container(get_settings())
    await container.startup()
    app.state.container = container
    yield
    await container.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    if settings.oauth_enabled:
        auth_service = AuthService(
            jwks_url=settings.oauth_jwks_url,
            issuer=settings.oauth_issuer,
            algorithms=settings.oauth_algorithms,
            audience=settings.oauth_audience,
        )
        app.add_middleware(OAuthMiddleware, auth_service=auth_service)
    # CORSMiddleware must be outermost — added after OAuthMiddleware so it wraps it
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(graphs_router, prefix=settings.api_prefix)
    app.include_router(workflows_router, prefix=settings.api_prefix)
    return app
