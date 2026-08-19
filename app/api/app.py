from __future__ import annotations

import app.compat  # noqa: F401 — must be first, patches langgraph.graph.graph

import logging
import secrets
from contextlib import AsyncExitStack, asynccontextmanager
from uuid import uuid4

from copilotkit import Action, CopilotKitRemoteEndpoint, LangGraphAGUIAgent
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit.integrations.fastapi import handler as _ck_handler
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from starlette.routing import get_route_path

from app.api.mcp.datasources_server import get_datasources_mcp, rebuild_datasource_tools
from app.api.mcp.management_server import get_management_mcp, register_management_tools
from app.api.middleware.auth import OAuthMiddleware
from app.infrastructure.auth.authorization import (
    AuthorizationPolicy,
    Permission,
    reset_current_permissions,
    set_current_permissions,
)
from app.api.routes.agent_callbacks import router as agent_callbacks_router
from app.api.routes.agents import router as agents_router
from app.api.routes.callbacks import router as callbacks_router
from app.api.routes.chat import router as chat_router
from app.api.routes.datasources import router as datasources_router
from app.api.routes.scripts import router as scripts_router
from app.api.routes.health import router as health_router
from app.api.routes.llm import router as llm_router
from app.api.routes.webhooks import router as webhooks_router
from app.api.routes.workflows import router as workflows_router
from app.core.config import get_settings
from app.core.container import ApplicationContainer, build_container
from app.domain.models.graph_run import GraphRun
from app.infrastructure.auth.auth_service import AuthError, AuthService
from app.infrastructure.orchestration.chat_agent_loader import load_chat_agent_config
from app.infrastructure.orchestration.default_workflow import build_default_workflow
from app.infrastructure.orchestration.router_agent import build_router_graph


logger = logging.getLogger(__name__)


class _DatasourcesAuthWrapper:
    """Pure-ASGI bearer-auth gate for the mounted ``/mcp/datasources`` app.

    ``OAuthMiddleware`` exempts ``/mcp/datasources`` (see
    ``app.api.middleware.auth._UNPROTECTED_PREFIXES``) so this backend can
    reach its own mounted MCP endpoint without a user JWT — there is no user
    in that flow, only the container's own MCP client. This wrapper is the
    actual gate for that endpoint:

    - ``api_key`` set: every request must carry an exact
      ``Authorization: Bearer <api_key>`` header, or it gets 401.
    - ``api_key`` unset and ``oauth_enabled``: fail closed (401 on every
      request) — without a key the endpoint would otherwise be reachable
      with zero credentials, which is strictly worse than the OAuth posture
      of the rest of the API.
    - ``api_key`` unset and OAuth disabled: pass through, matching the
      unauthenticated posture of the rest of the API.
    - ``api_key`` configured but empty / whitespace-only: unusable as a
      credential, so it fails closed regardless of ``oauth_enabled`` rather
      than passing through.
    """

    def __init__(self, app, api_key: str | None, oauth_enabled: bool) -> None:
        self._app = app
        # Surrounding whitespace is stripped (Secret Manager values often carry a
        # trailing newline, which no HTTP header value can hold — such a key can
        # never match anything).  A key that is *configured but empty* after
        # stripping is not a usable credential either: keeping it would make the
        # bare header "Bearer " valid.  It counts as unset for matching, but it
        # fails closed instead of passing through unauthenticated — an empty
        # secret must never open the endpoint.
        stripped = (api_key or "").strip()
        self._api_key = stripped or None
        configured_empty = api_key is not None and not stripped
        self._fail_closed = self._api_key is None and (oauth_enabled or configured_empty)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if self._api_key is not None:
            headers = dict(scope.get("headers") or ())
            # Compare the raw header bytes: a decoded str can hold non-ASCII
            # characters (any header byte >= 0x80), and secrets.compare_digest
            # raises TypeError on those -> 500 instead of 401.
            expected = b"Bearer " + self._api_key.encode()
            if not secrets.compare_digest(headers.get(b"authorization", b""), expected):
                await _send_401(send)
                return
        elif self._fail_closed:
            await _send_401(send)
            return

        await self._app(scope, receive, send)


class _ManagementAuthWrapper:
    """Pure-ASGI bearer-auth gate for the mounted ``/mcp/management`` app.

    Stricter than ``_DatasourcesAuthWrapper``: it always fails closed, because
    nothing inside this backend calls the management endpoint — it exists for
    external MCP clients only, and it can delete workflows and control runs.

    - ``api_key`` set and the request carries ``Authorization: Bearer <api_key>``
      → pass.
    - otherwise, ``oauth_enabled`` with a bearer token that ``auth_service``
      validates → pass.
    - everything else → 401.  In particular: no ``api_key`` and OAuth disabled
      means every request is rejected.  An empty / whitespace-only ``api_key``
      counts as *unset*, so ``MANAGEMENT_MCP_API_KEY=""`` cannot turn the bare
      header ``"Bearer "`` into a valid credential.

    Only ``http`` scopes can reach the inner app: anything else (websocket
    included) is refused here rather than forwarded unauthenticated.
    """

    def __init__(
        self,
        app,
        api_key: str | None,
        oauth_enabled: bool,
        auth_service: AuthService | None = None,
        policy: AuthorizationPolicy | None = None,
    ) -> None:
        self._app = app
        self._policy = policy or AuthorizationPolicy()
        # Strip surrounding whitespace (a Secret Manager value often carries a
        # trailing newline, which no HTTP header value can hold, so an unstripped
        # key could never match); empty / whitespace-only counts as unset.
        self._api_key = (api_key or "").strip() or None
        self._oauth_enabled = oauth_enabled and auth_service is not None
        self._auth_service = auth_service

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return

        headers = dict(scope.get("headers") or ())
        # Keep the raw bytes for the constant-time comparison: a latin-1 decode
        # turns any byte >= 0x80 into a non-ASCII character and
        # secrets.compare_digest raises TypeError on those (500 instead of 401).
        auth_bytes = headers.get(b"authorization", b"")

        if self._api_key is not None and secrets.compare_digest(
            auth_bytes, b"Bearer " + self._api_key.encode()
        ):
            # The static key is capped below ADMIN, so tools guarded on ADMIN
            # (unsandboxed python steps) refuse it regardless of role config.
            await self._call_with_principal(
                self._policy.permissions_for_api_key(), scope, receive, send
            )
            return

        auth_header = auth_bytes.decode("latin-1")

        if self._oauth_enabled and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            try:
                claims = await self._auth_service.validate_token(token)
            except AuthError as exc:
                logger.warning("management MCP auth rejected: %s", exc.message)
                await _send_401(send)
                return
            permissions = self._policy.permissions_for_claims(claims)
            if Permission.ACCESS not in permissions:
                logger.warning(
                    "management MCP authorization rejected: subject %s holds no role "
                    "granting access",
                    claims.get("sub"),
                )
                await _send_403(send)
                return
            await self._call_with_principal(permissions, scope, receive, send)
            return

        await _send_401(send)


    async def _call_with_principal(self, permissions, scope, receive, send) -> None:
        """Run the inner app with *permissions* bound for this request's task.

        FastMCP passes no request object to tool bodies, so a tool cannot ask who
        is calling; it reads the ambient principal instead. Set and reset around
        the call so nothing leaks between requests.
        """
        token = set_current_permissions(permissions)
        try:
            await self._app(scope, receive, send)
        finally:
            reset_current_permissions(token)


class _McpDispatcher:
    """Route ``/mcp/<name>`` to the matching mounted MCP app.

    Starlette matches mounts in registration order and ``Mount("/mcp")``
    swallows every ``/mcp/*`` path, so a second ``app.mount("/mcp", ...)``
    would be dead code.  This dispatcher sits behind the single ``/mcp`` mount
    and picks the inner app by path prefix instead.  Each inner FastMCP keeps
    its native ``streamable_http_path`` (``/datasources``, ``/management``), so
    the inner apps see exactly the paths they already expect.
    """

    def __init__(self, routes: dict) -> None:
        self._routes = routes

    async def __call__(self, scope, receive, send) -> None:
        # Starlette's Mount does not rewrite scope["path"] — it only sets
        # root_path ("/mcp"). get_route_path strips root_path, which is what the
        # inner-app prefixes ("/datasources", "/management") are relative to.
        path = get_route_path(scope) if scope.get("type") in ("http", "websocket") else ""
        for prefix, app in self._routes.items():
            if path == prefix or path.startswith(f"{prefix}/"):
                await app(scope, receive, send)
                return
        if scope["type"] == "http":
            await _send_404(send)


async def _send_401(send) -> None:
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({
        "type": "http.response.body",
        "body": b'{"detail":"Missing or invalid Authorization header"}',
    })


async def _send_403(send) -> None:
    await send({
        "type": "http.response.start",
        "status": 403,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({
        "type": "http.response.body",
        "body": b'{"detail":"Not authorized to access this service"}',
    })


async def _send_404(send) -> None:
    await send({
        "type": "http.response.start",
        "status": 404,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({
        "type": "http.response.body",
        "body": b'{"detail":"Not Found"}',
    })


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
        snap = await runner.graph.aget_state(_config(thread_id))
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
        snap = await runner.graph.aget_state(_config(thread_id))
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
        snap = await runner.graph.aget_state(_config(thread_id))
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


def _make_datasources_refresher(container: ApplicationContainer):
    """Return a coroutine that republishes data source MCP tools."""
    async def refresh() -> None:
        from app.api.routes.datasources import _refresh_datasource_tools
        await _refresh_datasource_tools(container)
    return refresh


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create_app stashes the Settings it resolved; reuse it so the mount and the
    # tool registration below cannot disagree with it.
    settings = getattr(app.state, "settings", None) or get_settings()
    container = build_container(settings)
    await container.startup()
    app.state.container = container

    # Data sources MCP server: publish the current tool list, then keep its
    # streamable-http session manager running for the app's lifetime.
    datasources_mcp = get_datasources_mcp()
    try:
        await rebuild_datasource_tools(
            datasources_mcp, container.data_source_backend, lambda: app.state.container
        )
    except Exception:
        logger.exception("failed to build initial datasources MCP tools")

    # Management MCP server: static tool set, registered once. Disabled by flag
    # means the server is never built and never mounted (404 at /mcp/management).
    management_mcp = None
    if settings.management_mcp_enabled:
        management_mcp = get_management_mcp(settings.management_mcp_allowed_hosts)
        register_management_tools(management_mcp, lambda: app.state.container)

    router_graph = build_router_graph(container.llm)

    # Load bundled chat agent config (always, regardless of workflow backend type)
    agent_config = load_chat_agent_config(
        getattr(settings, "chat_agent_config_path", None)
    )

    # Resolve the LLM for the chat agent — use llm_provider from YAML if set,
    # otherwise fall back to the container's default LLM.
    chat_llm = container.llm
    agent_provider = agent_config.get("llm_provider")
    if agent_provider and container.llm_factory:
        chat_llm = container.llm_factory(agent_provider, agent_config.get("model"))

    default_graph = build_default_workflow(
        chat_llm,
        container.yaml_graph_registry,
        container.run_repository,
        checkpointer=container.checkpointer,
        agent_config=agent_config,
        workflow_backend=container.workflow_backend,
        refresh_runner=container.refresh_runner,
        agent_backend=container.agent_backend,
        data_source_backend=container.data_source_backend,
        refresh_datasources=_make_datasources_refresher(container),
        # The chat agent is the platform's internal agent: it gets the same MCP
        # tools the workflow steps do, resolved per invocation.
        mcp_tools_provider=container.mcp_tools_provider,
        # Needed by the run-control tools (terminate/retry/restart/approve/reject).
        container=container,
    )
    sdk = CopilotKitRemoteEndpoint(
        agents=[
            LangGraphAGUIAgent(
                name="default",
                description=(
                    "Intelligent assistant that decides whether to reply directly "
                    "or route the request to the appropriate workflow."
                ),
                graph=default_graph,
            ),
            LangGraphAGUIAgent(
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

    async with AsyncExitStack() as stack:
        # FastMCP's streamable-http app requires an active session manager.
        await stack.enter_async_context(datasources_mcp.session_manager.run())
        if management_mcp is not None:
            await stack.enter_async_context(management_mcp.session_manager.run())
        yield
    await container.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    app.state.settings = settings
    # Built once and shared by OAuthMiddleware and the management MCP gate.
    authz_policy = AuthorizationPolicy.from_settings(settings)
    app.state.authz_policy = authz_policy
    auth_service: AuthService | None = None
    if settings.oauth_enabled:
        auth_service = AuthService(
            jwks_url=settings.oauth_jwks_url,
            issuer=settings.oauth_issuer,
            algorithms=settings.oauth_algorithms,
            audience=settings.oauth_audience,
        )
        app.add_middleware(OAuthMiddleware, auth_service=auth_service, policy=authz_policy)
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
    app.include_router(agents_router, prefix=settings.api_prefix)
    app.include_router(datasources_router, prefix=settings.api_prefix)
    app.include_router(scripts_router, prefix=settings.api_prefix)
    app.include_router(llm_router, prefix=settings.api_prefix)
    app.include_router(chat_router, prefix=settings.api_prefix)
    app.include_router(webhooks_router, prefix=settings.api_prefix)
    app.include_router(callbacks_router, prefix=settings.api_prefix)
    # Agent callback routes — registered after callbacks but the /runs prefix
    # means they don't conflict with the /{run_id} routes in workflows.
    app.include_router(agent_callbacks_router, prefix=settings.api_prefix)
    # Data sources MCP server — the streamable-http endpoint lands exactly on
    # /mcp/datasources (streamable_http_path="/datasources" inside the mount).
    # OAuthMiddleware exempts this prefix (see _UNPROTECTED_PREFIXES); the
    # actual gate is _DatasourcesAuthWrapper below.
    # Same normalization as _DatasourcesAuthWrapper: an empty / whitespace-only
    # key counts as unset (and fails closed), so it must warn too.
    if not (settings.mcp_datasources_api_key or "").strip() and settings.oauth_enabled:
        logger.warning(
            "OAUTH_ENABLED is true but MCP_DATASOURCES_API_KEY is not set — "
            "/mcp/datasources will reject every request (fail closed). Set "
            "MCP_DATASOURCES_API_KEY so the backend's own agents can reach it."
        )
    mcp_routes: dict = {
        "/datasources": _DatasourcesAuthWrapper(
            get_datasources_mcp().streamable_http_app(),
            api_key=settings.mcp_datasources_api_key,
            oauth_enabled=settings.oauth_enabled,
        ),
    }
    # Management MCP — the platform's own CRUD + run control tools for external
    # MCP clients. Flag off means it is not mounted at all (404).
    if settings.management_mcp_enabled:
        # Same normalization as _ManagementAuthWrapper: an empty / whitespace-only
        # key counts as unset, so it must warn too instead of silently 401-ing.
        if not (settings.management_mcp_api_key or "").strip() and not settings.oauth_enabled:
            logger.warning(
                "MANAGEMENT_MCP_ENABLED is true but MANAGEMENT_MCP_API_KEY is not "
                "set and OAuth is disabled — /mcp/management will reject every "
                "request (fail closed). Set MANAGEMENT_MCP_API_KEY to use it."
            )
        mcp_routes["/management"] = _ManagementAuthWrapper(
            get_management_mcp(settings.management_mcp_allowed_hosts).streamable_http_app(),
            api_key=settings.management_mcp_api_key,
            oauth_enabled=settings.oauth_enabled,
            auth_service=auth_service,
            policy=authz_policy,
        )
    app.mount("/mcp", _McpDispatcher(mcp_routes))
    return app
