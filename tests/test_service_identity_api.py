"""Tests for GET /api/v1/llm/service-identities.

The endpoint tells a UI which outbound identities exist, which are usable and
what each authenticates as. It must never echo key material.
"""
from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.core.config import Settings, get_settings

PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\nnot-a-real-key\n-----END RSA PRIVATE KEY-----"

# The flat single-identity form, as deployed manifests set it.
FLAT_ENV = {
    "SERVICE_AUTH_ENABLED": "true",
    "SERVICE_AUTH_TOKEN_URL": "https://auth.example.test/oauth/v2/token",
    "SERVICE_AUTH_AUDIENCE": "https://auth.example.test",
    "SERVICE_AUTH_SCOPES": "openid some:scope",
    "SERVICE_AUTH_CLIENT_ID": "flat-client",
    "SERVICE_AUTH_KEY_ID": "flat-key-id",
    "SERVICE_AUTH_PRIVATE_KEY": PRIVATE_KEY,
}

ALL_SERVICE_AUTH_VARS = (
    *FLAT_ENV,
    "SERVICE_AUTH_IDENTITIES",
    "SERVICE_AUTH_DEFAULT_IDENTITY",
)


@pytest.fixture
async def client():
    app = create_app()
    # Settings is rebuilt per request so env changes inside a test take effect
    # (the real dependency is lru_cached for the process).
    app.dependency_overrides[get_settings] = lambda: Settings()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start every test from no service-auth configuration at all."""
    for name in ALL_SERVICE_AUTH_VARS:
        monkeypatch.delenv(name, raising=False)


def _set_env(monkeypatch, env: dict):
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def _identity(name: str, **overrides) -> dict:
    body = {
        "name": name,
        "token_url": f"https://{name}.auth.test/oauth/v2/token",
        "audience": f"https://{name}.auth.test",
        "client_id": f"{name}-client",
        "key_id": f"{name}-key-id",
        "private_key": PRIVATE_KEY,
        "scopes": f"openid {name}:aud",
    }
    body.update(overrides)
    return body


async def test_flat_settings_appear_as_the_default_identity(client, monkeypatch):
    """Deployments predating SERVICE_AUTH_IDENTITIES keep working unchanged."""
    _set_env(monkeypatch, FLAT_ENV)

    data = (await client.get("/api/v1/llm/service-identities")).json()
    assert data["enabled"] is True
    assert data["error"] is None
    assert [i["name"] for i in data["identities"]] == ["default"]
    # A single identity is unambiguously the default without extra config.
    assert data["default_identity"] == "default"
    only = data["identities"][0]
    assert only["configured"] is True
    assert only["client_id"] == "flat-client"
    assert only["token_url"] == "https://auth.example.test/oauth/v2/token"


async def test_lists_multiple_named_identities(client, monkeypatch):
    _set_env(monkeypatch, {
        "SERVICE_AUTH_ENABLED": "true",
        "SERVICE_AUTH_IDENTITIES": json.dumps([_identity("afp"), _identity("control_center")]),
    })

    data = (await client.get("/api/v1/llm/service-identities")).json()
    assert [i["name"] for i in data["identities"]] == ["afp", "control_center"]
    assert all(i["configured"] for i in data["identities"])
    # Several identities and no designated default — a caller must name one.
    assert data["default_identity"] is None


async def test_explicit_default_identity_is_reported(client, monkeypatch):
    _set_env(monkeypatch, {
        "SERVICE_AUTH_ENABLED": "true",
        "SERVICE_AUTH_IDENTITIES": json.dumps([_identity("afp"), _identity("control_center")]),
        "SERVICE_AUTH_DEFAULT_IDENTITY": "afp",
    })

    data = (await client.get("/api/v1/llm/service-identities")).json()
    assert data["default_identity"] == "afp"


async def test_json_entry_overrides_the_flat_identity_of_the_same_name(client, monkeypatch):
    """How a flat deployment migrates to the JSON list without a gap."""
    _set_env(monkeypatch, {
        **FLAT_ENV,
        "SERVICE_AUTH_IDENTITIES": json.dumps([_identity("default", client_id="json-client")]),
    })

    data = (await client.get("/api/v1/llm/service-identities")).json()
    assert [i["name"] for i in data["identities"]] == ["default"]
    assert data["identities"][0]["client_id"] == "json-client"


async def test_never_echoes_key_material(client, monkeypatch):
    _set_env(monkeypatch, {
        **FLAT_ENV,
        "SERVICE_AUTH_IDENTITIES": json.dumps([_identity("afp")]),
    })

    body = (await client.get("/api/v1/llm/service-identities")).text
    assert "not-a-real-key" not in body
    assert "PRIVATE KEY" not in body
    # Key ids are signing material too — no caller needs them.
    assert "flat-key-id" not in body
    assert "afp-key-id" not in body


async def test_incomplete_identity_is_listed_but_not_configured(client, monkeypatch):
    _set_env(monkeypatch, {
        "SERVICE_AUTH_ENABLED": "true",
        "SERVICE_AUTH_IDENTITIES": json.dumps([
            _identity("afp"),
            _identity("broken", private_key=None),
        ]),
    })

    data = (await client.get("/api/v1/llm/service-identities")).json()
    by_name = {i["name"]: i for i in data["identities"]}
    assert by_name["afp"]["configured"] is True
    assert by_name["broken"]["configured"] is False
    assert "private_key" in by_name["broken"]["error"]


async def test_private_key_env_indirection(client, monkeypatch):
    """A JSON blob in config need not embed a PEM."""
    monkeypatch.setenv("AFP_SIGNING_KEY", PRIVATE_KEY)
    _set_env(monkeypatch, {
        "SERVICE_AUTH_ENABLED": "true",
        "SERVICE_AUTH_IDENTITIES": json.dumps([
            _identity("afp", private_key=None, private_key_env="AFP_SIGNING_KEY"),
        ]),
    })

    data = (await client.get("/api/v1/llm/service-identities")).json()
    assert data["identities"][0]["configured"] is True
    assert "not-a-real-key" not in (await client.get("/api/v1/llm/service-identities")).text


async def test_disabled_reports_reason(client, monkeypatch):
    _set_env(monkeypatch, {**FLAT_ENV, "SERVICE_AUTH_ENABLED": "false"})

    data = (await client.get("/api/v1/llm/service-identities")).json()
    assert data["enabled"] is False
    assert data["identities"] == []
    assert "SERVICE_AUTH_ENABLED" in data["error"]


async def test_enabled_with_nothing_configured(client, monkeypatch):
    _set_env(monkeypatch, {"SERVICE_AUTH_ENABLED": "true"})

    data = (await client.get("/api/v1/llm/service-identities")).json()
    assert data["enabled"] is True
    assert data["identities"] == []
    assert "No service identity is configured" in data["error"]


async def test_malformed_identities_json_is_reported_not_raised(client, monkeypatch):
    _set_env(monkeypatch, {
        "SERVICE_AUTH_ENABLED": "true",
        "SERVICE_AUTH_IDENTITIES": json.dumps({"name": "afp"}),  # object, not array
    })

    resp = await client.get("/api/v1/llm/service-identities")
    assert resp.status_code == 200
    assert "must be a JSON array" in resp.json()["error"]


async def test_config_keys_never_expose_service_auth_material(client, monkeypatch):
    """The credential-name list backs a *bearer token* picker — a signing key
    must not appear there, or it could be pasted in as one."""
    _set_env(monkeypatch, FLAT_ENV)

    keys = (await client.get("/api/v1/llm/config/keys")).json()["keys"]
    assert not [k for k in keys if k.startswith("SERVICE_AUTH_")]
