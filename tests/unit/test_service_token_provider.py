"""Tests for the OAuth2 JWT bearer grant (RFC 7523) service token provider.

The token endpoint is stubbed with a fake httpx.AsyncClient: each test supplies
a payload/status and the recorder captures every posted form.
"""
from __future__ import annotations

import asyncio
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.core.config import Settings
from app.infrastructure.auth import service_token_provider as provider_module
from app.infrastructure.auth.service_token_provider import (
    JWT_BEARER_GRANT_TYPE,
    ServiceAuthError,
    ServiceTokenProvider,
)

TOKEN_URL = "https://auth.example.test/oauth/token"


# ---------------------------------------------------------------------------
# Key material + httpx stub
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def keypair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("error", request=None, response=self)  # type: ignore[arg-type]

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeClient:
    def __init__(self, recorder) -> None:
        self._recorder = recorder

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def post(self, url, data=None, headers=None):
        self._recorder.calls.append(
            {"url": url, "data": dict(data or {}), "headers": dict(headers or {})}
        )
        return self._recorder.responder(len(self._recorder.calls) - 1)


@pytest.fixture
def token_endpoint(monkeypatch):
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.responder = lambda index: FakeResponse(
                {"access_token": f"token-{index}", "expires_in": 3600}
            )

    recorder = Recorder()
    monkeypatch.setattr(
        provider_module.httpx, "AsyncClient", lambda *a, **kw: FakeClient(recorder)
    )
    return recorder


def _settings(private_key: str, **overrides) -> Settings:
    data = {
        "SERVICE_AUTH_ENABLED": True,
        "SERVICE_AUTH_TOKEN_URL": TOKEN_URL,
        "SERVICE_AUTH_CLIENT_ID": "client-123",
        "SERVICE_AUTH_KEY_ID": "key-abc",
        "SERVICE_AUTH_PRIVATE_KEY": private_key,
        "SERVICE_AUTH_AUDIENCE": "https://auth.example.test",
        "SERVICE_AUTH_SCOPES": "openid custom:scope",
    }
    data.update(overrides)
    return Settings(**data)


# ---------------------------------------------------------------------------
# Grant request + assertion contents
# ---------------------------------------------------------------------------

async def test_posts_jwt_bearer_grant_with_scopes(keypair, token_endpoint):
    private_key, _ = keypair
    provider = ServiceTokenProvider(_settings(private_key))

    token = await provider.get_token()

    assert token == "token-0"
    call = token_endpoint.calls[0]
    assert call["url"] == TOKEN_URL
    assert call["data"]["grant_type"] == JWT_BEARER_GRANT_TYPE
    # Scopes are forwarded verbatim — opaque strings to this code.
    assert call["data"]["scope"] == "openid custom:scope"
    assert call["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


async def test_assertion_claims_and_kid_header(keypair, token_endpoint):
    private_key, public_key = keypair
    provider = ServiceTokenProvider(_settings(private_key))

    await provider.get_token()

    assertion = token_endpoint.calls[0]["data"]["assertion"]
    assert jwt.get_unverified_header(assertion)["kid"] == "key-abc"
    claims = jwt.decode(
        assertion,
        public_key,
        algorithms=["RS256"],
        audience="https://auth.example.test",
    )
    assert claims["iss"] == "client-123"
    assert claims["sub"] == "client-123"
    assert claims["aud"] == "https://auth.example.test"
    assert 0 < claims["exp"] - claims["iat"] <= 600


async def test_private_key_escaped_newlines_are_normalized(keypair, token_endpoint):
    private_key, public_key = keypair
    escaped = private_key.replace("\n", "\\n")
    provider = ServiceTokenProvider(_settings(escaped))

    await provider.get_token()

    assertion = token_endpoint.calls[0]["data"]["assertion"]
    jwt.decode(
        assertion,
        public_key,
        algorithms=["RS256"],
        audience="https://auth.example.test",
    )


async def test_get_auth_header_carries_bearer_token(keypair, token_endpoint):
    private_key, _ = keypair
    provider = ServiceTokenProvider(_settings(private_key))

    assert await provider.get_auth_header() == {"Authorization": "Bearer token-0"}


# ---------------------------------------------------------------------------
# Caching / refresh
# ---------------------------------------------------------------------------

async def test_token_is_cached_between_calls(keypair, token_endpoint):
    private_key, _ = keypair
    provider = ServiceTokenProvider(_settings(private_key))

    first = await provider.get_token()
    second = await provider.get_token()

    assert first == second == "token-0"
    assert len(token_endpoint.calls) == 1


async def test_token_refreshed_when_near_expiry(keypair, token_endpoint):
    private_key, _ = keypair
    # Shorter than the refresh buffer, so the cached token is never reused.
    token_endpoint.responder = lambda index: FakeResponse(
        {"access_token": f"token-{index}", "expires_in": 30}
    )
    provider = ServiceTokenProvider(_settings(private_key))

    assert await provider.get_token() == "token-0"
    assert await provider.get_token() == "token-1"
    assert len(token_endpoint.calls) == 2


async def test_expiry_falls_back_to_unverified_exp_claim(keypair, token_endpoint):
    private_key, _ = keypair
    long_lived = jwt.encode(
        {"exp": int(time.time()) + 3600}, private_key, algorithm="RS256"
    )
    token_endpoint.responder = lambda index: FakeResponse({"access_token": long_lived})
    provider = ServiceTokenProvider(_settings(private_key))

    await provider.get_token()
    await provider.get_token()

    assert len(token_endpoint.calls) == 1


async def test_unusable_lifetime_falls_back_to_default_ttl(keypair, token_endpoint):
    private_key, _ = keypair
    # Neither expires_in nor a future exp: the default TTL applies.
    stale = jwt.encode({"exp": int(time.time()) - 10}, private_key, algorithm="RS256")
    token_endpoint.responder = lambda index: FakeResponse({"access_token": stale})
    provider = ServiceTokenProvider(_settings(private_key))

    await provider.get_token()
    await provider.get_token()

    assert len(token_endpoint.calls) == 1


async def test_concurrent_callers_trigger_one_request(keypair, token_endpoint):
    private_key, _ = keypair
    provider = ServiceTokenProvider(_settings(private_key))

    tokens = await asyncio.gather(*(provider.get_token() for _ in range(5)))

    assert set(tokens) == {"token-0"}
    assert len(token_endpoint.calls) == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

async def test_disabled_provider_raises_actionable_error(keypair, token_endpoint):
    private_key, _ = keypair
    provider = ServiceTokenProvider(_settings(private_key, SERVICE_AUTH_ENABLED=False))

    with pytest.raises(ServiceAuthError, match="SERVICE_AUTH_ENABLED"):
        await provider.get_token()
    assert token_endpoint.calls == []


async def test_missing_configuration_lists_the_missing_keys(keypair, token_endpoint):
    private_key, _ = keypair
    provider = ServiceTokenProvider(
        _settings(private_key, SERVICE_AUTH_TOKEN_URL=None, SERVICE_AUTH_CLIENT_ID=None)
    )

    with pytest.raises(ServiceAuthError) as excinfo:
        await provider.get_token()
    assert "SERVICE_AUTH_TOKEN_URL" in str(excinfo.value)
    assert "SERVICE_AUTH_CLIENT_ID" in str(excinfo.value)


def test_validate_configuration_is_noop_when_disabled(keypair):
    private_key, _ = keypair
    provider = ServiceTokenProvider(
        _settings(private_key, SERVICE_AUTH_ENABLED=False, SERVICE_AUTH_TOKEN_URL=None)
    )
    provider.validate_configuration()


def test_validate_configuration_fails_fast_when_incomplete(keypair):
    private_key, _ = keypair
    provider = ServiceTokenProvider(_settings(private_key, SERVICE_AUTH_PRIVATE_KEY=None))

    with pytest.raises(ServiceAuthError, match="SERVICE_AUTH_PRIVATE_KEY"):
        provider.validate_configuration()


async def test_invalid_private_key_raises_service_auth_error(token_endpoint):
    provider = ServiceTokenProvider(_settings("not-a-pem-key"))

    with pytest.raises(ServiceAuthError, match="SERVICE_AUTH_PRIVATE_KEY"):
        await provider.get_token()
    assert token_endpoint.calls == []


async def test_rejected_grant_raises_with_status_and_body(keypair, token_endpoint):
    private_key, _ = keypair
    token_endpoint.responder = lambda index: FakeResponse(
        {"error": "invalid_client"}, status_code=400, text='{"error":"invalid_client"}'
    )
    provider = ServiceTokenProvider(_settings(private_key))

    with pytest.raises(ServiceAuthError) as excinfo:
        await provider.get_token()
    assert "400" in str(excinfo.value)
    assert "invalid_client" in str(excinfo.value)


async def test_response_without_access_token_raises(keypair, token_endpoint):
    private_key, _ = keypair
    token_endpoint.responder = lambda index: FakeResponse({"token_type": "Bearer"})
    provider = ServiceTokenProvider(_settings(private_key))

    with pytest.raises(ServiceAuthError, match="no access_token"):
        await provider.get_token()
