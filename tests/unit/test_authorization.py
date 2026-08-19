"""Role → permission mapping, and the claim shapes it has to survive."""
from __future__ import annotations

import pytest

from app.infrastructure.auth.authorization import (
    AuthorizationPolicy,
    Permission,
    extract_roles,
    get_current_permissions,
    permission_for_method,
    reset_current_permissions,
    set_current_permissions,
)


# ── extract_roles ────────────────────────────────────────────────────────────

def test_flattened_roles_claim_as_list() -> None:
    assert extract_roles({"roles": ["STAFF", "AUTHOR"]}) == {"STAFF", "AUTHOR"}


def test_flattened_roles_claim_as_dict() -> None:
    """Zitadel's custom-claims action emits a dict keyed by role name."""
    claims = {"roles": {"STAFF": {"orgid": "example.com"}, "AUTHOR": {}}}
    assert extract_roles(claims) == {"STAFF", "AUTHOR"}


def test_flattened_roles_claim_as_space_delimited_string() -> None:
    assert extract_roles({"roles": "STAFF AUTHOR"}) == {"STAFF", "AUTHOR"}


def test_project_roles_claim_used_when_flattened_claim_absent() -> None:
    """Machine users skip the custom-claims action and carry only this shape."""
    claims = {"urn:zitadel:iam:org:project:12345:roles": {"SERVICE": {}}}
    assert extract_roles(claims, project_id="12345") == {"SERVICE"}


def test_project_roles_claim_found_without_configured_project_id() -> None:
    """A deployment requesting all project roles need not enumerate project ids."""
    claims = {"urn:zitadel:iam:org:project:999:roles": ["STAFF"]}
    assert extract_roles(claims) == {"STAFF"}


def test_flattened_claim_wins_over_project_claim() -> None:
    claims = {
        "roles": ["FROM_FLAT"],
        "urn:zitadel:iam:org:project:1:roles": ["FROM_PROJECT"],
    }
    assert extract_roles(claims, project_id="1") == {"FROM_FLAT"}


@pytest.mark.parametrize("claims", [{}, {"roles": None}, {"roles": 42}, {"roles": []}])
def test_missing_or_malformed_roles_yield_nothing(claims: dict) -> None:
    """A malformed claim must deny, never raise — the caller turns this into a 403."""
    assert extract_roles(claims) == set()


# ── enforcement disabled ─────────────────────────────────────────────────────

def test_disabled_policy_grants_everything() -> None:
    """Pre-RBAC behaviour, so an upgrade cannot lock out a live deployment."""
    policy = AuthorizationPolicy(enforce=False)
    assert policy.permissions_for_claims({}) == frozenset(Permission)


def test_disabled_policy_still_caps_the_api_key_below_admin() -> None:
    policy = AuthorizationPolicy(enforce=False)
    assert Permission.ADMIN not in policy.permissions_for_api_key()


# ── the access gate ──────────────────────────────────────────────────────────

def _policy(**kw) -> AuthorizationPolicy:
    defaults = dict(
        enforce=True,
        access_roles=frozenset({"STAFF"}),
        read_roles=frozenset({"STAFF"}),
        write_roles=frozenset({"AUTHOR"}),
        delete_roles=frozenset({"OWNER"}),
        admin_roles=frozenset({"PLATFORM_ADMIN"}),
    )
    defaults.update(kw)
    return AuthorizationPolicy(**defaults)


def test_role_outside_the_access_list_gets_nothing() -> None:
    """A valid token from a shared identity provider is not entitlement."""
    assert _policy().permissions_for_claims({"roles": ["CUSTOMER_VIEWER"]}) == frozenset()


def test_access_gate_overrides_other_grants() -> None:
    """Holding a write role means nothing without access — the gate is checked first."""
    policy = _policy(access_roles=frozenset({"STAFF"}))
    assert policy.permissions_for_claims({"roles": ["AUTHOR"]}) == frozenset()


def test_empty_access_roles_denies_everyone_when_enforcing() -> None:
    """Fail closed: a half-configured deployment rejects rather than admits."""
    policy = _policy(access_roles=frozenset())
    assert policy.permissions_for_claims({"roles": ["STAFF"]}) == frozenset()


def test_access_role_alone_grants_only_access() -> None:
    policy = _policy(read_roles=frozenset({"READER"}))
    assert policy.permissions_for_claims({"roles": ["STAFF"]}) == frozenset({Permission.ACCESS})


# ── permission mapping ───────────────────────────────────────────────────────

def test_read_role_grants_read_but_not_write_or_delete() -> None:
    granted = _policy().permissions_for_claims({"roles": ["STAFF"]})
    assert Permission.READ in granted
    assert Permission.WRITE not in granted
    assert Permission.DELETE not in granted
    assert Permission.ADMIN not in granted


def test_write_role_does_not_imply_delete() -> None:
    """Delete is its own permission so edit rights do not include destruction."""
    granted = _policy().permissions_for_claims({"roles": ["STAFF", "AUTHOR"]})
    assert Permission.WRITE in granted
    assert Permission.DELETE not in granted


def test_write_role_does_not_imply_admin() -> None:
    """Authoring workflows must not confer backend code execution."""
    granted = _policy().permissions_for_claims({"roles": ["STAFF", "AUTHOR"]})
    assert Permission.ADMIN not in granted


def test_admin_role_grants_admin() -> None:
    granted = _policy().permissions_for_claims({"roles": ["STAFF", "PLATFORM_ADMIN"]})
    assert Permission.ADMIN in granted


def test_roles_accumulate() -> None:
    granted = _policy().permissions_for_claims(
        {"roles": ["STAFF", "AUTHOR", "OWNER", "PLATFORM_ADMIN"]}
    )
    assert granted == frozenset(Permission)


def test_role_names_are_whitespace_normalised() -> None:
    """Config arrives as env-var strings, which routinely carry stray spaces."""
    policy = AuthorizationPolicy.from_settings(
        type("S", (), {
            "auth_enforce_permissions": True,
            "auth_project_id": None,
            "auth_access_roles": [" STAFF ", ""],
            "auth_read_roles": ["STAFF"],
            "auth_write_roles": [],
            "auth_delete_roles": [],
            "auth_admin_roles": [],
        })()
    )
    assert policy.access_roles == frozenset({"STAFF"})
    assert Permission.READ in policy.permissions_for_claims({"roles": ["STAFF"]})


# ── the management MCP static key ────────────────────────────────────────────

def test_api_key_gets_data_access_but_never_admin() -> None:
    policy = _policy()
    granted = policy.permissions_for_api_key()
    assert {Permission.ACCESS, Permission.READ, Permission.WRITE, Permission.DELETE} <= granted
    assert Permission.ADMIN not in granted


def test_api_key_cannot_be_configured_into_admin() -> None:
    """The cap is structural: no role configuration reaches the key path at all."""
    policy = _policy(admin_roles=frozenset({"PLATFORM_ADMIN", "ANY", "EVERYTHING"}))
    assert Permission.ADMIN not in policy.permissions_for_api_key()


# ── method → permission ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("GET", Permission.READ),
        ("HEAD", Permission.READ),
        ("OPTIONS", Permission.READ),
        ("POST", Permission.WRITE),
        ("PUT", Permission.WRITE),
        ("PATCH", Permission.WRITE),
        ("DELETE", Permission.DELETE),
        ("get", Permission.READ),
    ],
)
def test_method_permissions(method: str, expected: Permission) -> None:
    assert permission_for_method(method) is expected


def test_unknown_method_requires_write_not_read() -> None:
    """An unrecognised verb must not fall through to the weakest requirement."""
    assert permission_for_method("TRACE") is Permission.WRITE


# ── ambient principal ────────────────────────────────────────────────────────

def test_ambient_principal_defaults_to_nothing() -> None:
    """A guarded operation reached without an authenticating wrapper is denied."""
    assert get_current_permissions() == frozenset()


def test_ambient_principal_set_and_reset() -> None:
    token = set_current_permissions(frozenset({Permission.READ}))
    try:
        assert get_current_permissions() == frozenset({Permission.READ})
    finally:
        reset_current_permissions(token)
    assert get_current_permissions() == frozenset()
