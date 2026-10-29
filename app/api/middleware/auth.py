from __future__ import annotations

import logging
import re
from dataclasses import replace

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.infrastructure.auth.auth_service import AuthError, AuthService
from app.application.login import (
    LoginService,
    effective_roles,
    login_decision,
    login_script_store_of,
    with_profile,
)
from app.application.teams import directory_of, resolve_subject
from app.infrastructure.auth.authorization import (
    AuthorizationPolicy,
    Permission,
    permission_for_method,
    reset_current_permissions,
    reset_current_subject,
    set_current_permissions,
    set_current_subject,
    subject_from_claims,
)

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
# "/api/v1/runs/" prefix: that prefix also covers the run-control API in
# app.api.routes.workflows (read a run, its trace, terminate, retry, delete,
# approve, reject, restart-from-step), which must NOT be reachable without a
# user token. Note "/agent/reply" is deliberately absent — the frontend calls it
# and does send a token.
_AGENT_CALLBACK_PATH = re.compile(
    r"^/api/v1/runs/[^/]+/agent/(?:output|question|input|progress)$"
)


# Runner enrollment and self-service (app.api.routes.runners): a machine that
# has no credential yet asks for an enrollment and polls it with its own poll
# token; an enrolled runner revokes itself with its runner token. Approving
# (lookup/approve/deny) and managing runners stays user-authenticated. The
# runner sockets are WebSockets, which this HTTP middleware never sees; they
# check the runner token themselves.
_RUNNER_PUBLIC = (
    ("GET", re.compile(r"^/api/v1/acp/runners/(?:info|install\.md)$")),
    ("POST", re.compile(r"^/api/v1/acp/runners/enrollments$")),
    ("GET", re.compile(r"^/api/v1/acp/runners/enrollments/[^/]+/poll$")),
    ("DELETE", re.compile(r"^/api/v1/acp/runners/self$")),
)


def _is_unprotected(method: str, path: str) -> bool:
    """Whether `method path` may bypass user authentication."""
    if path in _UNPROTECTED_PATHS or path.startswith(_UNPROTECTED_PREFIXES):
        return True
    if any(method == m and rx.match(path) for m, rx in _RUNNER_PUBLIC):
        return True
    for exempt_method, prefix in _UNPROTECTED_METHOD_PREFIXES:
        if method == exempt_method and path.startswith(prefix):
            return True
    return bool(_AGENT_CALLBACK_PATH.match(path))


class OAuthMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        auth_service: AuthService,
        policy: AuthorizationPolicy | None = None,
        login_service: LoginService | None = None,
    ) -> None:
        super().__init__(app)
        self.auth_service = auth_service
        # No policy configured behaves as authentication-only, which is what every
        # deployment did before permissions existed.
        self.policy = policy or AuthorizationPolicy()
        # The login script, if one is configured (app.application.login).
        self.login_service = login_service

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

        container = getattr(request.app.state, "container", None)
        directory = directory_of(container)
        decision = await login_decision(
            self.login_service, token, claims, directory, login_script_store_of(container)
        )
        roles = effective_roles(self.policy, claims, decision)
        permissions = self.policy.permissions_for_roles(roles)
        request.state.permissions = permissions
        # Strict evaluation: in shadow mode `permissions` holds everything for
        # everyone, which must not make every private resource visible.
        would_grant = self.policy.permissions_for_roles_strict(roles)
        subject = replace(
            with_profile(
                subject_from_claims(claims, Permission.SUPER_ADMIN in would_grant), decision
            ),
            roles=frozenset(roles),
        )

        # Shadow mode. With enforcement off the policy grants everything, so the
        # checks below never fire — but the mapping can still be evaluated and
        # reported. This is what makes turning enforcement on an evidence-based
        # decision rather than a hopeful one: any caller that would be locked out
        # (most likely one holding an opaque token whose userinfo response carries
        # no project roles) shows up in logs first, while still being served.
        if not self.policy.enforce:
            required_if_enforced = permission_for_method(request.method)
            denied = self.policy.denied_roles(roles)
            if denied:
                logger.warning(
                    "RBAC shadow: subject %s would be DENIED access (holds denied "
                    "role %s); roles=%s path=%s %s",
                    claims.get("sub"), sorted(denied), sorted(roles), request.method, path,
                )
            elif Permission.ACCESS not in would_grant:
                logger.warning(
                    "RBAC shadow: subject %s would be DENIED access (no matching role); "
                    "roles=%s path=%s %s",
                    claims.get("sub"), sorted(roles), request.method, path,
                )
            elif required_if_enforced not in would_grant:
                logger.warning(
                    "RBAC shadow: subject %s would be DENIED %s; roles=%s path=%s %s",
                    claims.get("sub"), required_if_enforced.value,
                    sorted(roles), request.method, path,
                )
            return await self._call_with_principal(permissions, subject, directory, call_next, request)

        # ACCESS is the tenancy gate: an identity the provider issued a valid token
        # for, but which is not entitled to this API at all (a customer, on an
        # identity provider shared with staff).
        if Permission.ACCESS not in permissions:
            logger.warning(
                "Authorization rejected: subject %s holds no role granting access "
                "(denied roles: %s)",
                claims.get("sub"), sorted(self.policy.denied_roles(roles)),
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Not authorized to access this service"},
            )

        required = permission_for_method(request.method)
        if required not in permissions:
            logger.warning(
                "Authorization rejected: subject %s lacks %s for %s %s",
                claims.get("sub"), required.value, request.method, path,
            )
            return JSONResponse(
                status_code=403,
                content={"detail": f"Missing '{required.value}' permission"},
            )

        return await self._call_with_principal(permissions, subject, directory, call_next, request)

    async def _call_with_principal(self, permissions, subject, directory, call_next, request: Request):
        """Serve the request with *permissions* bound as the ambient principal.

        The HTTP method is not the whole story. A single authenticated request can
        reach operations of several tiers: ``POST /api/v1/chat`` (WRITE by method)
        hands the chat agent a toolset that includes ``delete_workflow``, and the
        management MCP tools run the same shared cores. Those entrypoints get no
        request object, so they read the ambient principal instead — binding it
        here is what stops WRITE from becoming DELETE by going through the agent.

        The value is set before ``call_next`` on purpose: BaseHTTPMiddleware runs
        the downstream app in a child task, which copies this context at creation
        and therefore keeps seeing the principal for its whole life, including
        while a streaming response is still being produced after this method has
        returned and reset its own copy.
        """
        token = set_current_permissions(permissions)
        # The caller's team comes from carrier's own directory, read per
        # request so a membership change applies at once on every replica.
        subject = await resolve_subject(directory, subject)
        subject_token = set_current_subject(subject)
        try:
            return await call_next(request)
        finally:
            reset_current_subject(subject_token)
            reset_current_permissions(token)
