"""Tests for the /mcp/datasources auth posture:

- OAuthMiddleware exempts the /mcp/datasources prefix (so the backend can
  reach its own mounted MCP endpoint without a user JWT).
- _DatasourcesAuthWrapper is the actual gate for that endpoint.
"""
from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from app.api.app import _DatasourcesAuthWrapper
from app.api.middleware.auth import OAuthMiddleware
from app.infrastructure.auth.auth_service import AuthService


def _inner_app() -> Starlette:
    async def ok(request):
        return PlainTextResponse("ok")

    return Starlette(routes=[Route("/{path:path}", ok, methods=["GET", "POST"])])


# ---------------------------------------------------------------------------
# _DatasourcesAuthWrapper
# ---------------------------------------------------------------------------

async def test_wrapper_401_without_token_when_key_set():
    wrapper = _DatasourcesAuthWrapper(_inner_app(), api_key="secret", oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://test") as c:
        resp = await c.post("/datasources")
    assert resp.status_code == 401


async def test_wrapper_401_with_wrong_token_when_key_set():
    wrapper = _DatasourcesAuthWrapper(_inner_app(), api_key="secret", oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://test") as c:
        resp = await c.post("/datasources", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


async def test_wrapper_200_pass_through_with_correct_token():
    wrapper = _DatasourcesAuthWrapper(_inner_app(), api_key="secret", oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://test") as c:
        resp = await c.post("/datasources", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.text == "ok"


async def test_wrapper_fails_closed_when_no_key_and_oauth_enabled():
    wrapper = _DatasourcesAuthWrapper(_inner_app(), api_key=None, oauth_enabled=True)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://test") as c:
        resp = await c.post("/datasources")
    assert resp.status_code == 401


async def test_wrapper_passes_through_when_no_key_and_oauth_disabled():
    wrapper = _DatasourcesAuthWrapper(_inner_app(), api_key=None, oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://test") as c:
        resp = await c.post("/datasources")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# OAuthMiddleware exemption
# ---------------------------------------------------------------------------

async def test_oauth_middleware_exempts_mcp_datasources_prefix():
    service = AuthService(jwks_url="https://auth.example.com/keys", issuer="https://auth.example.com")
    app = OAuthMiddleware(_inner_app(), auth_service=service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/mcp/datasources")
    assert resp.status_code == 200


async def test_oauth_middleware_still_rejects_other_paths_without_token():
    service = AuthService(jwks_url="https://auth.example.com/keys", issuer="https://auth.example.com")
    app = OAuthMiddleware(_inner_app(), auth_service=service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/some/protected/path")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Non-ASCII bearer token: 401, not 500 (raw-ASGI probe with byte headers)
# ---------------------------------------------------------------------------

async def _raw_asgi_get(app, path: str, auth: bytes | None = None) -> int:
    """Drive an ASGI app directly so the Authorization header keeps raw bytes.

    An httpx/TestClient call cannot carry a byte >= 0x80 in a header the way a
    real socket does, and that byte is the whole bug: Starlette latin-1 decodes
    it into a non-ASCII token which httpx then refuses to ascii-encode into the
    outbound userinfo header.
    """
    headers = [(b"host", b"test")]
    if auth is not None:
        headers.append((b"authorization", auth))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    status: dict = {}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]

    await app(scope, receive, send)
    return status["code"]


class _ExplodingTransport(httpx.AsyncBaseTransport):
    """Transport-level stub: any outbound call at all fails the test."""

    def __init__(self) -> None:
        self.calls: list = []

    async def handle_async_request(self, request):  # pragma: no cover - must not run
        self.calls.append(request)
        raise AssertionError(f"unexpected outbound request to {request.url}")


@pytest.fixture
def exploding_httpx(monkeypatch):
    transport = _ExplodingTransport()
    real_client = httpx.AsyncClient

    def _client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)
    return transport


async def test_oauth_middleware_non_ascii_bearer_is_401_not_500(exploding_httpx):
    """Real (unmocked) AuthService, OAuth enabled, non-ASCII bearer → 401."""
    service = AuthService(
        jwks_url="https://auth.example.com/keys", issuer="https://auth.example.com"
    )
    app = OAuthMiddleware(_inner_app(), auth_service=service)

    status = await _raw_asgi_get(app, "/api/v1/workflows", auth=b"Bearer \xff\xfe\xc3\xbf")

    assert status == 401
    assert exploding_httpx.calls == []


async def test_oauth_middleware_ascii_junk_bearer_is_also_401(exploding_httpx):
    """Sanity check the probe itself: ASCII junk keeps its ordinary 401 path."""
    service = AuthService(
        jwks_url="https://auth.example.com/keys", issuer=None
    )
    app = OAuthMiddleware(_inner_app(), auth_service=service)

    status = await _raw_asgi_get(app, "/api/v1/workflows", auth=b"Bearer junk")

    assert status == 401


# ---------------------------------------------------------------------------
# API keys with surrounding whitespace (Secret Manager trailing newline)
# ---------------------------------------------------------------------------

async def test_wrapper_accepts_key_configured_with_trailing_newline():
    wrapper = _DatasourcesAuthWrapper(
        _inner_app(), api_key="secret\n", oauth_enabled=True
    )
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://test") as c:
        resp = await c.post("/datasources", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


async def test_wrapper_accepts_key_configured_with_surrounding_spaces():
    wrapper = _DatasourcesAuthWrapper(
        _inner_app(), api_key="  secret  ", oauth_enabled=True
    )
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://test") as c:
        resp = await c.post("/datasources", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


async def test_wrapper_empty_key_fails_closed_and_rejects_bare_bearer():
    """An empty key must never become "no key configured, pass through"."""
    for key in ("", "   ", "\n"):
        wrapper = _DatasourcesAuthWrapper(_inner_app(), api_key=key, oauth_enabled=False)
        async with AsyncClient(
            transport=ASGITransport(app=wrapper), base_url="http://test"
        ) as c:
            assert (await c.post("/datasources")).status_code == 401
            resp = await c.post("/datasources", headers={"Authorization": "Bearer "})
            assert resp.status_code == 401


def test_settings_strip_api_keys():
    from app.core.config import Settings

    settings = Settings(
        MCP_DATASOURCES_API_KEY="ds-key\n", MANAGEMENT_MCP_API_KEY="  mgmt-key  "
    )
    assert settings.mcp_datasources_api_key == "ds-key"
    assert settings.management_mcp_api_key == "mgmt-key"

    blank = Settings(MCP_DATASOURCES_API_KEY="  \n", MANAGEMENT_MCP_API_KEY="   ")
    # Whitespace-only stays falsy → both wrappers keep reading it as "no usable
    # key" and fail closed.
    assert blank.mcp_datasources_api_key == ""
    assert blank.management_mcp_api_key == ""
    assert Settings().mcp_datasources_api_key is None
