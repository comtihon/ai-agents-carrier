from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from app.api.routes.health import router as health_router
from app.api.routes.workflows import router as workflows_router
from app.core.config import get_settings
from app.core.container import ApplicationContainer, build_container
from app.domain.models.runtime import WorkflowRequest


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
    app.include_router(health_router)
    app.include_router(workflows_router, prefix=settings.api_prefix)
    _register_langserve_routes(app, settings.langserve_path)
    return app


def _register_langserve_routes(app: FastAPI, path: str) -> None:
    async def invoke(request: WorkflowRequest) -> dict:
        container: ApplicationContainer = app.state.container
        run = await container.orchestration_service.submit(request)
        return run.model_dump(mode="json")

    add_routes(app, RunnableLambda(invoke), path=path)
