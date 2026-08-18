from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.infrastructure.auth.auth_service import AuthError, AuthService

logger = logging.getLogger(__name__)

# Exact paths that bypass authentication (health/readiness probes)
# /mcp/management is the bare form of the "/mcp/management/" prefix below; it is
# guarded by _ManagementAuthWrapper instead (see app.api.app).
_UNPROTECTED_PATHS = {"/health", "/ready", "/mcp/management"}

# Path prefixes that bypass authentication.
# /copilotkit is the CopilotKit runtime endpoint — it has no user-specific data
# and cannot reach the backend API without the frontend actions providing their
# own authenticated calls.  Secure at the network / API-gateway level instead.
# /api/v1/callbacks are approval callback URLs sent to external systems (Slack, etc.)
# where the caller has no credentials; the run_id UUID in the path is the secret.
_UNPROTECTED_PREFIXES = (
    "/copilotkit",
    "/api/v1/callbacks/",
    "/api/v1/webhooks/",
    # Agent output/progress callbacks — posted by spawned agent containers that
    # have no user credentials.  The run_id UUID in the path acts as the shared
    # secret; secure at the network level (cluster-internal traffic only).
    "/api/v1/runs/",
    # Data sources MCP — this backend must be able to reach its own mounted
    # endpoint at startup/refresh time without a user JWT (there is no user
    # in that flow). The mount is guarded instead by a dedicated bearer-token
    # wrapper around the mounted app (see app.api.app._DatasourcesAuthWrapper).
    "/mcp/datasources",
    # Management MCP — same reasoning as /mcp/datasources: BaseHTTPMiddleware in
    # front of a FastMCP streamable-HTTP app is unverified, so the prefix is
    # exempted here and guarded by a dedicated bearer-token wrapper around the
    # mounted app instead (see app.api.app._ManagementAuthWrapper), which fails
    # closed when no API key is configured.  Listed as a "/"-terminated prefix so
    # it cannot also exempt a sibling path like /mcp/managementfoo.
    "/mcp/management/",
)


class OAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth_service: AuthService) -> None:
        super().__init__(app)
        self.auth_service = auth_service

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (
            path in _UNPROTECTED_PATHS
            or path.startswith(_UNPROTECTED_PREFIXES)
        ):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        token: str | None = None
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]

        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or malformed Authorization header"},
            )

        try:
            claims = await self.auth_service.validate_token(token)
        except AuthError as e:
            logger.warning("Auth rejected: %s", e.message)
            return JSONResponse(status_code=401, content={"detail": e.message})

        request.state.jwt_claims = claims
        return await call_next(request)
