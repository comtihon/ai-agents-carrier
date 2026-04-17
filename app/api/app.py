from __future__ import annotations

import app.compat  # noqa: F401 — must be first, patches langgraph.graph.graph

from contextlib import asynccontextmanager
from uuid import uuid4

from copilotkit import Action, CopilotKitRemoteEndpoint, LangGraphAgent
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit.integrations.fastapi import handler as _ck_handler
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command

from app.api.middleware.auth import OAuthMiddleware
from app.api.routes.callbacks import router as callbacks_router
from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.webhooks import router as webhooks_router
from app.api.routes.workflows import router as workflows_router
from app.core.config import get_settings
from app.core.container import ApplicationContainer, build_container
from app.domain.models.graph_run import GraphRun
from app.infrastructure.auth.auth_service import AuthService
from app.infrastructure.orchestration.default_workflow import build_default_workflow
from app.infrastructure.orchestration.router_agent import build_router_graph


def _langgraph_status(snap) -> str:
    return "waiting_approval" if snap.next else "completed"


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _build_actions(container: ApplicationContainer) -> list[Action]:
    """Create CopilotKit backend actions backed by the application container."""

    async def list_graphs_handler() -> dict:
        runners = container.yaml_graph_registry.list_ids()
        return {
            "graphs": [
                {
                    "id": gid,
                    "name": container.yaml_graph_registry.get(gid).name,
                    "description": container.yaml_graph_registry.get(gid).description,
                }
                for gid in runners
            ]
        }

    async def start_graph_run_handler(graph_id: str, request: str) -> dict:
        runner = container.yaml_graph_registry.get(graph_id)
        if runner is None:
            return {"error": f"Graph '{graph_id}' not found"}
        thread_id = str(uuid4())
        run = GraphRun(id=thread_id, graph_id=graph_id, status="running")
        await container.run_repository.create(run)
        try:
            await runner.graph.ainvoke({"request": request}, _config(thread_id))
        except Exception as exc:  # noqa: BLE001
            run.status = "failed"
            await container.run_repository.update(run)
            return {"error": str(exc)}
        snap = runner.graph.get_state(_config(thread_id))
        run.status = _langgraph_status(snap)
        run.state = snap.values
        await container.run_repository.update(run)
        return {"graph_id": graph_id, "thread_id": thread_id, "status": run.status}

    async def get_graph_run_handler(graph_id: str, thread_id: str) -> dict:
        run = await container.run_repository.get(thread_id)
        if run is None:
            return {"error": "Run not found"}
        return {"graph_id": run.graph_id, "thread_id": run.id, "status": run.status}

    async def approve_graph_run_handler(graph_id: str, thread_id: str) -> dict:
        runner = container.yaml_graph_registry.get(graph_id)
        if runner is None:
            return {"error": f"Graph '{graph_id}' not found"}
        run = await container.run_repository.get(thread_id)
        if run is None:
            return {"error": "Run not found"}
        await runner.graph.ainvoke(Command(resume={"approved": True}), _config(thread_id))
        snap = runner.graph.get_state(_config(thread_id))
        run.status = _langgraph_status(snap)
        run.state = snap.values
        await container.run_repository.update(run)
        return {"graph_id": graph_id, "thread_id": thread_id, "status": run.status}

    async def reject_graph_run_handler(
        graph_id: str, thread_id: str, reason: str = ""
    ) -> dict:
        runner = container.yaml_graph_registry.get(graph_id)
        if runner is None:
            return {"error": f"Graph '{graph_id}' not found"}
        run = await container.run_repository.get(thread_id)
        if run is None:
            return {"error": "Run not found"}
        await runner.graph.ainvoke(
            Command(resume={"approved": False, "reason": reason or None}),
            _config(thread_id),
        )
        snap = runner.graph.get_state(_config(thread_id))
        run.status = _langgraph_status(snap)
        run.state = snap.values
        await container.run_repository.update(run)
        return {"graph_id": graph_id, "thread_id": thread_id, "status": run.status}

    return [
        Action(
            name="listGraphs",
            handler=list_graphs_handler,
            description="List all available workflow graphs with their IDs, names, and descriptions.",
            parameters=[],
        ),
        Action(
            name="startGraphRun",
            handler=start_graph_run_handler,
            description="Start a new run of a workflow graph.",
            parameters=[
                {"name": "graph_id", "type": "string", "description": "The workflow graph ID", "required": True},
                {"name": "request", "type": "string", "description": "The user request / task description", "required": True},
            ],
        ),
        Action(
            name="getGraphRun",
            handler=get_graph_run_handler,
            description="Get the current status of a workflow run.",
            parameters=[
                {"name": "graph_id", "type": "string", "description": "The workflow graph ID", "required": True},
                {"name": "thread_id", "type": "string", "description": "The run thread ID", "required": True},
            ],
        ),
        Action(
            name="approveGraphRun",
            handler=approve_graph_run_handler,
            description="Approve a workflow run that is waiting for human approval.",
            parameters=[
                {"name": "graph_id", "type": "string", "description": "The workflow graph ID", "required": True},
                {"name": "thread_id", "type": "string", "description": "The run thread ID", "required": True},
            ],
        ),
        Action(
            name="rejectGraphRun",
            handler=reject_graph_run_handler,
            description="Reject a workflow run that is waiting for human approval.",
            parameters=[
                {"name": "graph_id", "type": "string", "description": "The workflow graph ID", "required": True},
                {"name": "thread_id", "type": "string", "description": "The run thread ID", "required": True},
                {"name": "reason", "type": "string", "description": "Optional rejection reason"},
            ],
        ),
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = build_container(get_settings())
    await container.startup()
    app.state.container = container

    router_graph = build_router_graph(container.llm)
    default_graph = build_default_workflow(
        container.llm,
        container.yaml_graph_registry,
        container.run_repository,
    )
    sdk = CopilotKitRemoteEndpoint(
        agents=[
            LangGraphAgent(
                name="default",
                description=(
                    "Intelligent assistant that decides whether to reply directly "
                    "or route the request to the appropriate workflow."
                ),
                graph=default_graph,
            ),
            LangGraphAgent(
                name="router",
                description=(
                    "Conversational assistant that explains the workflow platform "
                    "and guides users through available workflows."
                ),
                graph=router_graph,
            ),
        ],
        actions=_build_actions(container),
    )
    app.state.default_graph = default_graph
    add_fastapi_endpoint(app, sdk, "/copilotkit")

    # add_fastapi_endpoint only registers /copilotkit/{path:path}.
    # FastAPI's redirect_slashes redirects POST /copilotkit → 307 /copilotkit/,
    # which breaks streaming. Register the bare path explicitly so no redirect fires.
    async def _ck_root(request: Request) -> None:
        request.scope.setdefault("path_params", {})["path"] = ""
        return await _ck_handler(request, sdk)

    app.add_api_route(
        "/copilotkit",
        _ck_root,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        include_in_schema=False,
    )

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
    app.include_router(workflows_router, prefix=settings.api_prefix)
    app.include_router(chat_router, prefix=settings.api_prefix)
    app.include_router(webhooks_router, prefix=settings.api_prefix)
    app.include_router(callbacks_router, prefix=settings.api_prefix)
    return app
