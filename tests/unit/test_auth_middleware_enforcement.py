"""End-to-end behaviour of OAuthMiddleware: 401, 403, shadow mode, principal.

The policy's own table is unit-tested in ``test_authorization.py``; what is
asserted here is the wiring — that a role really does turn into a status code,
that shadow mode serves the request while reporting what it *would* have done,
and that the caller's permissions are bound as the ambient principal for the
duration of the request (the only way a tool body reached through an LLM can know
who is calling).
"""
from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware.auth import OAuthMiddleware
from app.infrastructure.auth.auth_service import AuthError
from app.infrastructure.auth.authorization import (
    AuthorizationPolicy,
    get_current_permissions,
    get_current_permissions_or_none,
)


class _FakeAuthService:
    """Maps a token to claims; the token "bad" is rejected."""

    def __init__(self, roles_by_token: dict[str, list[str]]) -> None:
        self._roles = roles_by_token

    async def validate_token(self, token: str) -> dict:
        if token not in self._roles:
            raise AuthError("Invalid token")
        return {"sub": f"user-of-{token}", "roles": self._roles[token]}


POLICY_ROLES = dict(
    access_roles=frozenset({"staff", "manager"}),
    read_roles=frozenset({"staff", "manager"}),
    write_roles=frozenset({"staff"}),
    delete_roles=frozenset({"admin-role"}),
    admin_roles=frozenset({"admin-role"}),
)

TOKENS = {
    "staff": ["staff"],
    "manager": ["manager"],          # read only
    "admin": ["staff", "admin-role"],
    "customer": ["client"],          # valid token, outside the access list
    "roleless": [],                  # e.g. an opaque token whose userinfo has no roles
}


def _client(enforce: bool) -> TestClient:
    app = FastAPI()

    @app.get("/api/v1/thing")
    async def read() -> dict:
        return {"permissions": sorted(p.value for p in get_current_permissions())}

    @app.post("/api/v1/thing")
    async def write() -> dict:
        return {"ok": True}

    @app.delete("/api/v1/thing")
    async def remove() -> dict:
        return {"ok": True}

    @app.get("/health")
    async def health() -> dict:
        return {"principal_bound": get_current_permissions_or_none() is not None}

    app.add_middleware(
        OAuthMiddleware,
        auth_service=_FakeAuthService(TOKENS),
        policy=AuthorizationPolicy(enforce=enforce, **POLICY_ROLES),
    )
    return TestClient(app)


@pytest.fixture
def enforcing() -> TestClient:
    return _client(enforce=True)


@pytest.fixture
def shadow() -> TestClient:
    return _client(enforce=False)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── Authentication ────────────────────────────────────────────────────────────

def test_missing_header_is_401(enforcing: TestClient) -> None:
    assert enforcing.get("/api/v1/thing").status_code == 401


def test_non_bearer_header_is_401(enforcing: TestClient) -> None:
    resp = enforcing.get("/api/v1/thing", headers={"Authorization": "Basic abc"})
    assert resp.status_code == 401


def test_rejected_token_is_401(enforcing: TestClient) -> None:
    assert enforcing.get("/api/v1/thing", headers=_auth("bad")).status_code == 401


def test_exempt_path_needs_no_token_and_binds_no_principal(
    enforcing: TestClient,
) -> None:
    """An exempt caller is not a principal: internal gates must see "unbound"."""
    resp = enforcing.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"principal_bound": False}


# ── Authorization, enforced ───────────────────────────────────────────────────

def test_role_outside_the_access_list_is_403(enforcing: TestClient) -> None:
    resp = enforcing.get("/api/v1/thing", headers=_auth("customer"))
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Not authorized to access this service"


def test_token_without_roles_is_403(enforcing: TestClient) -> None:
    """The expected shadow denial: an opaque token whose userinfo carries no roles."""
    assert enforcing.get("/api/v1/thing", headers=_auth("roleless")).status_code == 403


def test_reader_may_read(enforcing: TestClient) -> None:
    resp = enforcing.get("/api/v1/thing", headers=_auth("manager"))
    assert resp.status_code == 200
    assert resp.json()["permissions"] == ["access", "read"]


def test_reader_may_not_write(enforcing: TestClient) -> None:
    resp = enforcing.post("/api/v1/thing", headers=_auth("manager"))
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Missing 'write' permission"


def test_writer_may_write(enforcing: TestClient) -> None:
    assert enforcing.post("/api/v1/thing", headers=_auth("staff")).status_code == 200


def test_writer_may_not_delete(enforcing: TestClient) -> None:
    resp = enforcing.delete("/api/v1/thing", headers=_auth("staff"))
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Missing 'delete' permission"


def test_deleter_may_delete(enforcing: TestClient) -> None:
    assert enforcing.delete("/api/v1/thing", headers=_auth("admin")).status_code == 200


# ── The ambient principal ─────────────────────────────────────────────────────

def test_principal_is_bound_for_the_request(enforcing: TestClient) -> None:
    resp = enforcing.get("/api/v1/thing", headers=_auth("admin"))
    assert resp.json()["permissions"] == ["access", "admin", "delete", "read", "write"]


def test_principal_does_not_outlive_the_request(enforcing: TestClient) -> None:
    enforcing.get("/api/v1/thing", headers=_auth("admin"))
    assert get_current_permissions_or_none() is None


def test_shadow_mode_binds_the_full_principal(shadow: TestClient) -> None:
    """With enforcement off the policy grants everything, tools included."""
    resp = shadow.get("/api/v1/thing", headers=_auth("customer"))
    assert resp.status_code == 200
    assert resp.json()["permissions"] == ["access", "admin", "delete", "read", "write"]


# ── Shadow reporting ──────────────────────────────────────────────────────────

def test_shadow_reports_a_would_be_access_denial(
    shadow: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.api.middleware.auth"):
        assert shadow.get("/api/v1/thing", headers=_auth("customer")).status_code == 200
    line = "\n".join(caplog.messages + [r.getMessage() for r in caplog.records])
    assert "RBAC shadow" in line
    assert "would be DENIED access" in line
    assert "user-of-customer" in line
    assert "client" in line  # the roles it actually holds, for diagnosis


def test_shadow_reports_a_would_be_tier_denial(
    shadow: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.api.middleware.auth"):
        assert shadow.post("/api/v1/thing", headers=_auth("manager")).status_code == 200
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "would be DENIED write" in line
    assert "user-of-manager" in line


def test_shadow_stays_quiet_for_a_caller_that_would_be_allowed(
    shadow: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.api.middleware.auth"):
        assert shadow.get("/api/v1/thing", headers=_auth("staff")).status_code == 200
    assert not [r for r in caplog.records if "RBAC shadow" in r.getMessage()]


def test_shadow_still_rejects_an_invalid_token(shadow: TestClient) -> None:
    """Shadow mode is about *authorization*; authentication is unchanged."""
    assert shadow.get("/api/v1/thing", headers=_auth("bad")).status_code == 401
