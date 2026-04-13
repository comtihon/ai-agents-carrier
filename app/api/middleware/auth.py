from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.infrastructure.auth.auth_service import AuthError, AuthService

logger = logging.getLogger(__name__)

# Exact paths that bypass authentication (health/readiness probes)
_UNPROTECTED_PATHS = {"/health", "/ready"}

# Path prefixes that bypass authentication.
# /copilotkit is the CopilotKit runtime endpoint — it has no user-specific data
# and cannot reach the backend API without the frontend actions providing their
# own authenticated calls.  Secure at the network / API-gateway level instead.
_UNPROTECTED_PREFIXES = ("/copilotkit",)


class OAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth_service: AuthService) -> None:
        super().__init__(app)
        self.auth_service = auth_service

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _UNPROTECTED_PATHS or path.startswith(_UNPROTECTED_PREFIXES):
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
