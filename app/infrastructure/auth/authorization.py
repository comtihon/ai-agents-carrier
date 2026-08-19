"""Role-based authorization.

Authentication answers *who is calling*; this module answers *what they may do*.

The permission vocabulary is fixed and deliberately coarse. The mapping from
identity-provider roles onto it is entirely configuration, so no deployment's
role names appear in this repository:

    ACCESS  — may reach the API at all
    READ    — read workflows, runs, definitions, data sources, traces
    WRITE   — create, edit and run workflows; control runs
    DELETE  — delete workflows, runs, agents, data sources
    ADMIN   — privileged operations that can compromise the backend itself

ADMIN is not "WRITE plus a bit". It gates operations whose blast radius is the
backend process rather than the data — today that means unsandboxed `python`
steps, which execute inside this process alongside every credential it holds.

Two principal kinds reach the API, and they are not equivalent. A user
authenticates with an OAuth token and is granted permissions by role. The
management MCP static API key is a single long-lived shared secret that cannot be
attributed to a person or revoked per user, so it is capped: it never receives
ADMIN, no matter how it is configured.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    ACCESS = "access"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


# Permissions the management MCP API key may never hold, regardless of config.
# A shared static secret has no user identity behind it, so an action that can
# execute code in this process must not be reachable with it — there would be no
# way to say who did it.
_KEY_FORBIDDEN_PERMISSIONS = frozenset({Permission.ADMIN})

# Standard Zitadel claim carrying the roles granted on one project. Machine users
# do not run the custom-claims action, so their tokens carry this instead of the
# flattened `roles` claim that human tokens get.
_PROJECT_ROLES_CLAIM_TEMPLATE = "urn:zitadel:iam:org:project:{project_id}:roles"


def extract_roles(claims: dict[str, Any], project_id: str | None = None) -> set[str]:
    """Role names from a validated token's claims.

    Reads, in order:

    1. a flattened ``roles`` claim (list, or dict keyed by role name);
    2. ``urn:zitadel:iam:org:project:<project_id>:roles`` when *project_id* is
       configured — the shape machine-user tokens carry;
    3. any ``urn:zitadel:iam:org:project:*:roles`` claim, so a deployment that
       requests all project roles at once still resolves without having to
       enumerate project ids.

    Unknown shapes yield no roles rather than raising: a malformed claim must
    deny access, never 500.
    """
    def _names(value: Any) -> set[str]:
        if isinstance(value, dict):
            return {str(k) for k in value}
        if isinstance(value, (list, tuple, set)):
            return {str(v) for v in value}
        if isinstance(value, str):
            return {part for part in value.split() if part}
        return set()

    roles = _names(claims.get("roles"))
    if roles:
        return roles

    if project_id:
        roles = _names(claims.get(_PROJECT_ROLES_CLAIM_TEMPLATE.format(project_id=project_id)))
        if roles:
            return roles

    for key, value in claims.items():
        if key.startswith("urn:zitadel:iam:org:project") and key.endswith(":roles"):
            roles |= _names(value)
    return roles


@dataclass(frozen=True)
class AuthorizationPolicy:
    """Maps role names onto permissions.

    Every field holds role names supplied by configuration. A caller is granted a
    permission when it holds at least one of that permission's roles.

    ``access_roles`` is the tenancy gate, and it is an allow-list on purpose. An
    identity provider shared between staff and customers will grow new customer
    roles over time; with a deny-list every such addition would silently gain
    access, whereas an unlisted role here is denied by default.

    ``enforce`` off preserves pre-RBAC behaviour — any authenticated caller may do
    anything — because flipping to deny-by-default on upgrade would lock out every
    existing deployment. It logs a warning at startup so it cannot be forgotten.
    """

    enforce: bool = False
    project_id: str | None = None
    access_roles: frozenset[str] = field(default_factory=frozenset)
    read_roles: frozenset[str] = field(default_factory=frozenset)
    write_roles: frozenset[str] = field(default_factory=frozenset)
    delete_roles: frozenset[str] = field(default_factory=frozenset)
    admin_roles: frozenset[str] = field(default_factory=frozenset)

    @staticmethod
    def _norm(roles: Iterable[str] | None) -> frozenset[str]:
        return frozenset(r.strip() for r in (roles or ()) if r and r.strip())

    @classmethod
    def from_settings(cls, settings: Any) -> "AuthorizationPolicy":
        policy = cls(
            enforce=bool(getattr(settings, "auth_enforce_permissions", False)),
            project_id=getattr(settings, "auth_project_id", None),
            access_roles=cls._norm(getattr(settings, "auth_access_roles", None)),
            read_roles=cls._norm(getattr(settings, "auth_read_roles", None)),
            write_roles=cls._norm(getattr(settings, "auth_write_roles", None)),
            delete_roles=cls._norm(getattr(settings, "auth_delete_roles", None)),
            admin_roles=cls._norm(getattr(settings, "auth_admin_roles", None)),
        )
        policy.warn_if_misconfigured()
        return policy

    def warn_if_misconfigured(self) -> None:
        if not self.enforce:
            logger.warning(
                "AUTH_ENFORCE_PERMISSIONS is false — every authenticated caller has "
                "full access, including unsandboxed python steps. Set it to true and "
                "configure AUTH_ACCESS_ROLES / AUTH_READ_ROLES / AUTH_WRITE_ROLES / "
                "AUTH_DELETE_ROLES / AUTH_ADMIN_ROLES."
            )
            return
        if not self.access_roles:
            logger.error(
                "AUTH_ENFORCE_PERMISSIONS is true but AUTH_ACCESS_ROLES is empty — "
                "every request will be rejected. This is fail-closed and intentional; "
                "configure the roles that may reach this API."
            )
        if not self.admin_roles:
            logger.info(
                "AUTH_ADMIN_ROLES is empty — no caller can create unsandboxed python "
                "steps. This is the safe configuration."
            )

    def permissions_for_roles(self, roles: set[str]) -> frozenset[Permission]:
        """Permissions granted to a caller holding *roles*."""
        if not self.enforce:
            return frozenset(Permission)
        # The tenancy gate is checked first and denies everything on failure, so a
        # customer who happens to hold a role named in read_roles still gets nothing.
        return self.permissions_for_roles_strict(roles)

    def permissions_for_claims(self, claims: dict[str, Any]) -> frozenset[Permission]:
        return self.permissions_for_roles(extract_roles(claims, self.project_id))

    def roles_of(self, claims: dict[str, Any]) -> set[str]:
        """Roles this policy would read from *claims*. Exposed for diagnostics."""
        return extract_roles(claims, self.project_id)

    def evaluate_shadow(self, claims: dict[str, Any]) -> frozenset[Permission]:
        """Permissions that *would* be granted if enforcement were on.

        Identical to :meth:`permissions_for_claims` except that it ignores the
        ``enforce`` flag, so a deployment can measure the effect of turning
        enforcement on before it does.
        """
        return self.permissions_for_roles_strict(self.roles_of(claims))

    def permissions_for_roles_strict(self, roles: set[str]) -> frozenset[Permission]:
        """Role mapping with the ``enforce`` short-circuit removed."""
        if not (roles & self.access_roles):
            return frozenset()
        granted = {Permission.ACCESS}
        for permission, allowed in (
            (Permission.READ, self.read_roles),
            (Permission.WRITE, self.write_roles),
            (Permission.DELETE, self.delete_roles),
            (Permission.ADMIN, self.admin_roles),
        ):
            if roles & allowed:
                granted.add(permission)
        return frozenset(granted)

    def permissions_for_api_key(self) -> frozenset[Permission]:
        """Permissions for the management MCP static key.

        Full data access, never ADMIN — see the module docstring.
        """
        if not self.enforce:
            return frozenset(Permission) - _KEY_FORBIDDEN_PERMISSIONS
        return frozenset(
            {Permission.ACCESS, Permission.READ, Permission.WRITE, Permission.DELETE}
        ) - _KEY_FORBIDDEN_PERMISSIONS


# HTTP method → the permission it requires.
#
# Method-derived rather than declared per route: the set of routes changes far
# more often than the verbs, and a route whose permission was simply forgotten
# would default to unprotected. Here a new route inherits a sane requirement
# automatically. Read-only verbs map to READ, mutations to WRITE, and DELETE gets
# its own permission so that read/write access does not imply destruction.
_METHOD_PERMISSIONS: dict[str, Permission] = {
    "GET": Permission.READ,
    "HEAD": Permission.READ,
    "OPTIONS": Permission.READ,
    "POST": Permission.WRITE,
    "PUT": Permission.WRITE,
    "PATCH": Permission.WRITE,
    "DELETE": Permission.DELETE,
}


def permission_for_method(method: str) -> Permission:
    """Permission required by an HTTP method. Unknown verbs require WRITE."""
    return _METHOD_PERMISSIONS.get(method.upper(), Permission.WRITE)


# ── Ambient principal ────────────────────────────────────────────────────────
# The management MCP reaches its tools through FastMCP, which passes no request
# object down to them, so a tool cannot ask "who is calling". The ASGI wrapper
# that authenticates the request stores the caller's permissions here instead;
# the tool body reads them back. A ContextVar is correct rather than convenient:
# it is set inside the request's own task, so concurrent requests cannot observe
# one another's principal.
_current_permissions: "ContextVar[frozenset[Permission] | None]" = ContextVar(
    "current_permissions", default=None
)


def set_current_permissions(permissions: frozenset[Permission]) -> "Token":
    """Bind the calling principal's permissions for this task."""
    return _current_permissions.set(permissions)


def reset_current_permissions(token: "Token") -> None:
    _current_permissions.reset(token)


def get_current_permissions() -> frozenset[Permission]:
    """Permissions bound for this task, or none at all when unauthenticated.

    Defaults to the empty set, so a caller that somehow reaches a guarded
    operation without passing through an authenticating wrapper is denied. This
    is the right default for the ADMIN gate (unsandboxed python steps): code
    execution in this process must never be reachable by an unattributable
    caller. For the graded data permissions use :func:`missing_permission`, which
    tells "no principal bound at all" apart from "bound and holds nothing".
    """
    return _current_permissions.get() or frozenset()


def get_current_permissions_or_none() -> "frozenset[Permission] | None":
    """The bound principal, or ``None`` when nothing bound one.

    ``None`` and ``frozenset()`` mean different things and must not be conflated:
    the first is "no authenticating wrapper ran", the second is "a real caller
    holds no permission at all".
    """
    return _current_permissions.get()


def missing_permission(required: Permission) -> Permission | None:
    """*required*, when the bound principal lacks it; ``None`` when it is allowed.

    Shared by every non-HTTP entrypoint (management MCP tools, the chat agent's
    platform tools, the run-control cores) so that one operation cannot be
    reachable at a lower privilege through a different surface than the REST
    route that performs it.

    An *unbound* principal is allowed. Nothing bound one either because there is
    no authentication at all (OAuth disabled, or a direct in-process call in a
    test), or because the caller is the platform itself rather than a user: a
    Pub/Sub trigger, a cron trigger, a Slack callback, a webhook. Those carry no
    roles and never will, so denying them would break the platform rather than
    protect it — and each already authenticates by its own mechanism. This
    mirrors the fallback the REST routes use when no middleware ran.

    A *bound but empty* principal is denied: that is an authenticated identity
    whose roles grant nothing, exactly the case enforcement exists for.
    """
    permissions = _current_permissions.get()
    if permissions is None:
        return None
    return None if required in permissions else required
