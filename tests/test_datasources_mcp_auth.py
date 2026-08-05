"""Tests for the /mcp/datasources auth posture:

- OAuthMiddleware exempts the /mcp/datasources prefix (so the backend can
  reach its own mounted MCP endpoint without a user JWT).
- _DatasourcesAuthWrapper is the actual gate for that endpoint.
"""
from __future__ import annotations

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
