"""Tests for GET /api/v1/llm/service-identity.

The endpoint tells a UI whether picking ``service_identity`` auth will work and
which machine user it would authenticate as. It must never echo key material.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.core.config import Settings, get_settings

PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\nnot-a-real-key\n-----END RSA PRIVATE KEY-----"

SERVICE_AUTH_ENV = {
    "SERVICE_AUTH_ENABLED": "true",
    "SERVICE_AUTH_TOKEN_URL": "https://auth.uat.airteam.cloud/oauth/v2/token",
    "SERVICE_AUTH_AUDIENCE": "https://auth.uat.airteam.cloud",
    "SERVICE_AUTH_SCOPES": "openid urn:zitadel:iam:org:project:id:368003225346900776:aud",
    "SERVICE_AUTH_CLIENT_ID": "385067191013213465",
    "SERVICE_AUTH_KEY_ID": "key-1",
    "SERVICE_AUTH_PRIVATE_KEY": PRIVATE_KEY,
}


@pytest.fixture
async def client():
    app = create_app()
    # Settings is rebuilt per request so env changes inside a test take effect
    # (the real dependency is lru_cached for the process).
    app.dependency_overrides[get_settings] = lambda: Settings()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _set_env(monkeypatch, **overrides):
    env = {**SERVICE_AUTH_ENV, **overrides}
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


async def test_reports_configured_identity(client, monkeypatch):
    _set_env(monkeypatch)

    resp = await client.get("/api/v1/llm/service-identity")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["configured"] is True
    assert data["error"] is None
    assert data["client_id"] == "385067191013213465"
    assert data["token_url"] == "https://auth.uat.airteam.cloud/oauth/v2/token"
    assert data["audience"] == "https://auth.uat.airteam.cloud"
    assert "368003225346900776" in data["scopes"]


async def test_never_echoes_key_material(client, monkeypatch):
    _set_env(monkeypatch)

    resp = await client.get("/api/v1/llm/service-identity")
    body = resp.text
    assert "not-a-real-key" not in body
    assert "PRIVATE KEY" not in body
    # The key id is part of the signing material too — not needed by any caller.
    assert "key-1" not in body


async def test_disabled_reports_reason(client, monkeypatch):
    _set_env(monkeypatch, SERVICE_AUTH_ENABLED="false")

    data = (await client.get("/api/v1/llm/service-identity")).json()
    assert data["enabled"] is False
    assert data["configured"] is False
    assert "SERVICE_AUTH_ENABLED" in data["error"]


async def test_enabled_but_incomplete_names_missing_settings(client, monkeypatch):
    _set_env(monkeypatch, SERVICE_AUTH_PRIVATE_KEY=None, SERVICE_AUTH_CLIENT_ID=None)

    data = (await client.get("/api/v1/llm/service-identity")).json()
    assert data["enabled"] is True
    assert data["configured"] is False
    assert "SERVICE_AUTH_PRIVATE_KEY" in data["error"]
    assert "SERVICE_AUTH_CLIENT_ID" in data["error"]


async def test_config_keys_never_expose_service_auth_material(client, monkeypatch):
    """The credential-name list backs a *bearer token* picker — a signing key
    must not appear there, or it could be pasted in as one."""
    _set_env(monkeypatch)

    keys = (await client.get("/api/v1/llm/config/keys")).json()["keys"]
    assert not [k for k in keys if k.startswith("SERVICE_AUTH_")]
