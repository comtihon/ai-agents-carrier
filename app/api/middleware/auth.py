from __future__ import annotations

import logging
import re

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.infrastructure.auth.auth_service import AuthError, AuthService

logger = logging.getLogger(__name__)

# Exact paths that bypass authentication (health/readiness probes)
# /mcp/management is the bare form of the "/mcp/management/" prefix below; it is
# guarded by _ManagementAuthWrapper instead (see app.api.app).
_UNPROTECTED_PATHS = {"/health", "/ready", "/mcp/management"}

# Prefixes exempt for every method, because each already authenticates itself by
# another mechanism. Nothing belongs here on the grounds that it is "internal":
# the prod ingress serves "/", so every route below is reachable from the
# internet and needs a credential of its own.
_UNPROTECTED_PREFIXES = (
    # Slack posts these with an X-Slack-Signature HMAC, verified in
    # app.api.routes.callbacks._verify_slack_signature.
    "/api/v1/callbacks/slack/",
    # Webhook triggers carry an X-Webhook-Signature HMAC over the raw body,
    # keyed per workflow or by WEBHOOK_SECRET.
    "/api/v1/webhooks/",
    # Data sources MCP — this backend must reach its own mounted endpoint at
    # startup/refresh time without a user JWT (there is no user in that flow).
    # Guarded by a dedicated bearer-token wrapper around the mounted app
    # (see app.api.app._DatasourcesAuthWrapper).
    "/mcp/datasources",
    # Management MCP — same reasoning: BaseHTTPMiddleware in front of a FastMCP
    # streamable-HTTP app is unverified, so the prefix is exempted here and
    # guarded by app.api.app._ManagementAuthWrapper instead, which fails closed
    # when no API key is configured. Listed "/"-terminated so it cannot also
    # exempt a sibling path like /mcp/managementfoo.
    "/mcp/management/",
)

# (method, prefix) pairs exempt only for that HTTP method.
_UNPROTECTED_METHOD_PREFIXES = (
    # Approve/reject links rendered as Slack buttons and opened in a browser. A
    # link click cannot carry an Authorization header, so the GET forms stay
    # open and the run_id is their only credential. The POST forms are API calls
    # and are authenticated normally.
    ("GET", "/api/v1/callbacks/"),
)

# Callbacks posted by spawned agent containers, which hold no user credentials —
# the run_id acts as their bearer capability. Matched exactly rather than by a
# "/api/v1/runs/" prefix so that a route added to this router later is not
# silently unauthenticated. (Run control is unaffected either way: it lives under
# /api/v1/workflows/runs/..., which this prefix never covered.) Note
# "/agent/reply" is deliberately absent — the frontend calls it and sends a token.
_AGENT_CALLBACK_PATH = re.compile(
    r"^/api/v1/runs/[^/]+/agent/(?:output|question|input|progress)$"
)


def _is_unprotected(method: str, path: str) -> bool:
    """Whether `method path` may bypass user authentication."""
    if path in _UNPROTECTED_PATHS or path.startswith(_UNPROTECTED_PREFIXES):
        return True
    for exempt_method, prefix in _UNPROTECTED_METHOD_PREFIXES:
        if method == exempt_method and path.startswith(prefix):
            return True
    return bool(_AGENT_CALLBACK_PATH.match(path))


class OAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth_service: AuthService) -> None:
        super().__init__(app)
        self.auth_service = auth_service

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _is_unprotected(request.method, path):
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
