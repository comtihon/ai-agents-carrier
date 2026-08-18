"""
Unit tests for AuthService.

Covers:
- Opaque (non-JWT) token fallback to OIDC userinfo endpoint
- Userinfo response caching (TTL)
- Userinfo rejection (401/403)
- Missing issuer raises AuthError for opaque tokens
- JWT DecodeError path
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.infrastructure.auth.auth_service import AuthError, AuthService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_service(issuer: str | None = "https://auth.example.com") -> AuthService:
    return AuthService(
        jwks_url="https://auth.example.com/oauth/v2/keys",
        issuer=issuer,
    )


def _mock_http_client(status: int = 200, payload: dict | None = None):
    """Return a context-manager mock for httpx.AsyncClient."""
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=payload or {})

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = AsyncMock(return_value=resp)
    return client


# ── Opaque token → userinfo fallback ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_opaque_token_calls_userinfo_endpoint():
    service = _make_service()
    mock_client = _mock_http_client(
        status=200, payload={"sub": "user-123", "email": "test@example.com"}
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        claims = await service.validate_token("opaque_access_token")

    assert claims["sub"] == "user-123"
    assert claims["email"] == "test@example.com"
    mock_client.get.assert_called_once_with(
        "https://auth.example.com/oidc/v1/userinfo",
        headers={"Authorization": "Bearer opaque_access_token"},
    )


@pytest.mark.asyncio
async def test_opaque_token_userinfo_result_is_cached():
    service = _make_service()
    mock_client = _mock_http_client(
        status=200, payload={"sub": "user-abc"}
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        first = await service.validate_token("opaque_xyz")
        second = await service.validate_token("opaque_xyz")

    # HTTP client should only be called once; second hit comes from cache
    assert mock_client.get.call_count == 1
    assert first == second == {"sub": "user-abc"}


@pytest.mark.asyncio
async def test_opaque_token_different_tokens_not_shared_in_cache():
    service = _make_service()
    mock_client = _mock_http_client(status=200, payload={"sub": "user-1"})

    with patch("httpx.AsyncClient", return_value=mock_client):
        await service.validate_token("token_a")
        await service.validate_token("token_b")

    # Two distinct tokens → two HTTP calls
    assert mock_client.get.call_count == 2


@pytest.mark.asyncio
async def test_opaque_token_expired_cache_entry_re_fetches():
    service = _make_service()
    mock_client = _mock_http_client(status=200, payload={"sub": "user-1"})

    # Manually insert an already-expired cache entry
    import hashlib

    token = "expired_token"
    cache_key = hashlib.sha256(token.encode()).hexdigest()[:24]
    service._userinfo_cache[cache_key] = (
        {"sub": "stale"},
        datetime.now(timezone.utc) - timedelta(seconds=1),  # already expired
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        claims = await service.validate_token(token)

    # Should have gone to the network, not used stale entry
    mock_client.get.assert_called_once()
    assert claims["sub"] == "user-1"


@pytest.mark.asyncio
async def test_opaque_token_rejected_401_raises_auth_error():
    service = _make_service()
    mock_client = _mock_http_client(status=401)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(AuthError) as exc_info:
            await service.validate_token("bad_opaque_token")

    assert "rejected" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_opaque_token_rejected_403_raises_auth_error():
    service = _make_service()
    mock_client = _mock_http_client(status=403)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(AuthError) as exc_info:
            await service.validate_token("forbidden_token")

    assert "rejected" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_opaque_token_unexpected_status_raises_auth_error():
    service = _make_service()
    mock_client = _mock_http_client(status=500)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(AuthError) as exc_info:
            await service.validate_token("some_token")

    assert "500" in exc_info.value.message


@pytest.mark.asyncio
async def test_opaque_token_no_issuer_raises_auth_error():
    service = _make_service(issuer=None)

    with pytest.raises(AuthError) as exc_info:
        await service._validate_via_userinfo("any_token")

    msg = exc_info.value.message.lower()
    assert "userinfo" in msg or "issuer" in msg


# ── Userinfo URL derivation ───────────────────────────────────────────────────

def test_userinfo_url_derived_from_issuer():
    service = _make_service(issuer="https://auth.example.com")
    assert service._userinfo_url == "https://auth.example.com/oidc/v1/userinfo"


def test_userinfo_url_trailing_slash_stripped():
    service = _make_service(issuer="https://auth.example.com/")
    assert service._userinfo_url == "https://auth.example.com/oidc/v1/userinfo"


def test_userinfo_url_none_when_no_issuer():
    service = _make_service(issuer=None)
    assert service._userinfo_url is None


# ── JWT path: DecodeError triggers userinfo fallback ─────────────────────────

@pytest.mark.asyncio
async def test_jwt_decode_error_falls_back_to_userinfo():
    """A malformed/opaque token that triggers jwt.DecodeError → userinfo path."""
    import jwt as pyjwt

    service = _make_service()
    mock_client = _mock_http_client(status=200, payload={"sub": "from-userinfo"})

    with patch("jwt.get_unverified_header", side_effect=pyjwt.DecodeError("bad")):
        with patch("httpx.AsyncClient", return_value=mock_client):
            claims = await service.validate_token("not.a.valid.jwt")

    assert claims["sub"] == "from-userinfo"


# ── Non-ASCII bearer token → AuthError, never an outbound call ────────────────

class _ExplodingTransport(httpx.AsyncBaseTransport):
    """Real httpx transport that fails the test if it is ever reached."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request):  # pragma: no cover - must not run
        self.calls.append(request)
        raise AssertionError(f"unexpected outbound request to {request.url}")


@pytest.mark.asyncio
async def test_non_ascii_token_raises_auth_error_without_userinfo_call(monkeypatch):
    """A latin-1 decoded header byte >= 0x80 must 401, not blow up in httpx.

    The token is placed in an *outbound* Authorization header on the userinfo
    path and httpx ascii-encodes header values, so a non-ASCII token used to
    raise UnicodeEncodeError -> unauthenticated 500.  Patched at the transport
    level (not the auth service) so the real httpx call would actually happen
    if the guard regressed.
    """
    service = _make_service()
    transport = _ExplodingTransport()
    real_client = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)

    with pytest.raises(AuthError) as exc_info:
        await service.validate_token("\xff\xfe\xc3junk")

    assert "non-ascii" in exc_info.value.message.lower()
    assert transport.calls == []


@pytest.mark.asyncio
async def test_non_ascii_token_raises_auth_error_on_jwks_path(monkeypatch):
    """Same guard on the JWKS branch: no JWKS fetch, no crash, just AuthError."""
    service = _make_service()
    transport = _ExplodingTransport()
    real_client = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)
    # Force the JWT branch: without the guard this would reach _fetch_jwks.
    monkeypatch.setattr("jwt.get_unverified_header", lambda token: {"kid": "k1"})

    with pytest.raises(AuthError):
        await service.validate_token("hdr.payl\xffoad.sig")

    assert transport.calls == []


@pytest.mark.asyncio
async def test_ascii_junk_token_still_reaches_userinfo():
    """Guard must not swallow ordinary ASCII junk tokens (still a 401 path)."""
    service = _make_service()
    mock_client = _mock_http_client(status=401)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(AuthError):
            await service.validate_token("plain-junk")

    mock_client.get.assert_called_once()
