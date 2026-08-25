"""Tests for the /mcp/management auth posture and the /mcp dispatcher.

- ``_ManagementAuthWrapper`` is the real gate for the management MCP endpoint
  and fails closed: no API key and no OAuth means every request is rejected.
- ``_McpDispatcher`` routes /mcp/<name> to the matching mounted app and 404s
  everything else; /mcp/datasources must keep working through it.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from app.api.app import _DatasourcesAuthWrapper, _ManagementAuthWrapper, _McpDispatcher
from app.api.middleware.auth import OAuthMiddleware
from app.infrastructure.auth.auth_service import AuthError, AuthService


@pytest.fixture(autouse=True)
def _reset_management_mcp():
    """Drop the process-wide management MCP singleton around every test.

    ``get_management_mcp`` memoizes and ``allowed_hosts`` only applies on the
    building call, so any earlier ``create_app()`` — in this file or any other —
    would otherwise freeze this file's server with whatever
    MANAGEMENT_MCP_ALLOWED_HOSTS the ambient environment happened to hold.
    Rebuilding per test makes the allow-list pins below actually take effect and
    keeps the file order-independent.
    """
    from app.api.mcp import management_server

    management_server._mcp = None
    yield
    management_server._mcp = None


def _inner_app(body: str = "ok") -> Starlette:
    async def ok(request):
        return PlainTextResponse(body)

    return Starlette(routes=[Route("/{path:path}", ok, methods=["GET", "POST"])])


def _auth_service(valid: bool = True) -> MagicMock:
    service = MagicMock(spec=AuthService)
    if valid:
        service.validate_token = AsyncMock(return_value={"sub": "u1"})
    else:
        service.validate_token = AsyncMock(side_effect=AuthError("bad token"))
    return service


# ---------------------------------------------------------------------------
# _ManagementAuthWrapper
# ---------------------------------------------------------------------------

async def test_correct_api_key_passes():
    wrapper = _ManagementAuthWrapper(_inner_app(), api_key="secret", oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        resp = await c.post("/management", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.text == "ok"


async def test_wrong_api_key_is_rejected():
    wrapper = _ManagementAuthWrapper(_inner_app(), api_key="secret", oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        resp = await c.post("/management", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


async def test_missing_header_is_rejected():
    wrapper = _ManagementAuthWrapper(_inner_app(), api_key="secret", oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        resp = await c.post("/management")
    assert resp.status_code == 401


async def test_valid_jwt_passes_when_no_api_key_and_oauth_enabled():
    service = _auth_service(valid=True)
    wrapper = _ManagementAuthWrapper(
        _inner_app(), api_key=None, oauth_enabled=True, auth_service=service
    )
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        resp = await c.post("/management", headers={"Authorization": "Bearer jwt"})
    assert resp.status_code == 200
    service.validate_token.assert_awaited_once_with("jwt")


async def test_invalid_jwt_is_rejected():
    wrapper = _ManagementAuthWrapper(
        _inner_app(), api_key=None, oauth_enabled=True, auth_service=_auth_service(False)
    )
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        resp = await c.post("/management", headers={"Authorization": "Bearer jwt"})
    assert resp.status_code == 401


async def test_fails_closed_without_api_key_and_without_oauth():
    wrapper = _ManagementAuthWrapper(_inner_app(), api_key=None, oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        no_header = await c.post("/management")
        with_header = await c.post("/management", headers={"Authorization": "Bearer any"})
    assert no_header.status_code == 401
    assert with_header.status_code == 401


async def test_api_key_holder_passes_even_when_oauth_enabled():
    wrapper = _ManagementAuthWrapper(
        _inner_app(), api_key="secret", oauth_enabled=True, auth_service=_auth_service(False)
    )
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        resp = await c.post("/management", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# OAuthMiddleware exemption
# ---------------------------------------------------------------------------

async def test_oauth_middleware_exempts_mcp_management_prefix():
    service = AuthService(
        jwks_url="https://auth.example.com/keys", issuer="https://auth.example.com"
    )
    app = OAuthMiddleware(_inner_app(), auth_service=service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/mcp/management")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _McpDispatcher
# ---------------------------------------------------------------------------

def _dispatcher(with_management: bool = True) -> _McpDispatcher:
    routes: dict = {
        "/datasources": _DatasourcesAuthWrapper(
            _inner_app("datasources"), api_key=None, oauth_enabled=False
        )
    }
    if with_management:
        routes["/management"] = _ManagementAuthWrapper(
            _inner_app("management"), api_key="secret", oauth_enabled=False
        )
    return _McpDispatcher(routes)


async def test_dispatcher_routes_both_endpoints():
    app = _dispatcher()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        ds = await c.post("/datasources")
        mg = await c.post("/management", headers={"Authorization": "Bearer secret"})
    assert (ds.status_code, ds.text) == (200, "datasources")
    assert (mg.status_code, mg.text) == (200, "management")


async def test_dispatcher_routes_subpaths():
    app = _dispatcher()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.post("/datasources/sub/path")
    assert resp.status_code == 200


async def test_dispatcher_404s_unknown_paths():
    app = _dispatcher()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.post("/nope")
    assert resp.status_code == 404


async def test_dispatcher_404s_management_when_flag_off_but_keeps_datasources():
    app = _dispatcher(with_management=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        mg = await c.post("/management", headers={"Authorization": "Bearer secret"})
        ds = await c.post("/datasources")
    assert mg.status_code == 404
    assert ds.status_code == 200


# ---------------------------------------------------------------------------
# create_app wiring
# ---------------------------------------------------------------------------

async def test_create_app_mounts_management_when_enabled(monkeypatch):
    from app.api import app as app_module

    settings = app_module.get_settings()
    monkeypatch.setattr(settings, "management_mcp_enabled", True, raising=False)
    monkeypatch.setattr(settings, "management_mcp_api_key", "k", raising=False)
    fastapi_app = app_module.create_app()
    mount = next(r for r in fastapi_app.routes if getattr(r, "path", None) == "/mcp")
    assert set(mount.app._routes) == {"/datasources", "/management"}


async def test_create_app_omits_management_when_disabled(monkeypatch):
    from app.api import app as app_module

    settings = app_module.get_settings()
    monkeypatch.setattr(settings, "management_mcp_enabled", False, raising=False)
    fastapi_app = app_module.create_app()
    mount = next(r for r in fastapi_app.routes if getattr(r, "path", None) == "/mcp")
    assert set(mount.app._routes) == {"/datasources"}


# ---------------------------------------------------------------------------
# End-to-end through create_app: the real /mcp mount + dispatcher
#
# The standalone _McpDispatcher tests above feed it an unmounted scope shape
# (path="/management", root_path=""), which the real mount never produces:
# Starlette's Mount leaves scope["path"] alone and only sets root_path="/mcp".
# This test exercises the mounted scope, so a dispatcher that compares against
# the wrong path field turns every /mcp/* request into a 404 and fails here.
#
# ``Host: localhost:8000`` is required — FastMCP's DNS-rebinding guard only
# accepts a host:port form from its allow list.
# ---------------------------------------------------------------------------

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "1"},
    },
}

_TOOLS_LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}


def _rpc_result(response) -> dict:
    """Extract the JSON-RPC result from a streamable-http (SSE) response."""
    for line in response.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:])["result"]
    raise AssertionError(f"no SSE data frame in {response.text!r}")


async def test_mounted_mcp_endpoints_answer_the_real_handshake(monkeypatch):
    from contextlib import AsyncExitStack

    from app.api import app as app_module
    from app.api.mcp.datasources_server import get_datasources_mcp
    from app.api.mcp.management_server import get_management_mcp, register_management_tools

    settings = app_module.get_settings()
    monkeypatch.setattr(settings, "oauth_enabled", False, raising=False)
    monkeypatch.setattr(settings, "management_mcp_enabled", True, raising=False)
    monkeypatch.setattr(settings, "management_mcp_api_key", "mgmt-key", raising=False)
    monkeypatch.setattr(settings, "mcp_datasources_api_key", "ds-key", raising=False)
    # The handshake below hardcodes ``Host: localhost:8000``; pin the allow list
    # to the loopback defaults so a deployment value in the environment
    # (MANAGEMENT_MCP_ALLOWED_HOSTS) cannot turn this into a 421.  This only
    # bites because ``_reset_management_mcp`` clears the memoized singleton, so
    # ``create_app`` below is the building call.
    monkeypatch.setattr(settings, "management_mcp_allowed_hosts", None, raising=False)

    fastapi_app = app_module.create_app()
    register_management_tools(get_management_mcp(), lambda: MagicMock())

    async with AsyncExitStack() as stack:
        # Both FastMCP apps need a live session manager, which create_app's
        # lifespan starts; entered directly here so the test needs no Mongo.
        await stack.enter_async_context(get_datasources_mcp().session_manager.run())
        await stack.enter_async_context(get_management_mcp().session_manager.run())
        async with AsyncClient(
            transport=ASGITransport(app=fastapi_app), base_url="http://localhost:8000"
        ) as c:
            ds_auth = {**_MCP_HEADERS, "Authorization": "Bearer ds-key"}
            mg_auth = {**_MCP_HEADERS, "Authorization": "Bearer mgmt-key"}

            ds_init = await c.post("/mcp/datasources", headers=ds_auth, json=_INITIALIZE)
            mg_init = await c.post("/mcp/management", headers=mg_auth, json=_INITIALIZE)
            mg_tools = await c.post("/mcp/management", headers=mg_auth, json=_TOOLS_LIST)

            # Wrong credential must not 404 into oblivion — it must be a 401.
            ds_bad = await c.post(
                "/mcp/datasources",
                headers={**_MCP_HEADERS, "Authorization": "Bearer wrong"},
                json=_INITIALIZE,
            )
            mg_bad = await c.post(
                "/mcp/management",
                headers={**_MCP_HEADERS, "Authorization": "Bearer wrong"},
                json=_INITIALIZE,
            )
            unknown = await c.post("/mcp/nope", headers=_MCP_HEADERS, json=_INITIALIZE)
            # Not covered by the "/mcp/management" prefix — must not reach it.
            sibling = await c.post(
                "/mcp/managementfoo", headers=mg_auth, json=_INITIALIZE
            )

    # Reached the inner apps (a dispatcher path bug makes these 404).
    assert ds_init.status_code == 200, ds_init.text
    assert mg_init.status_code == 200, mg_init.text
    assert _rpc_result(ds_init)["serverInfo"]["name"] == "datasources"
    assert _rpc_result(mg_init)["serverInfo"]["name"] == "management"

    names = {t["name"] for t in _rpc_result(mg_tools)["tools"]}
    assert len(names) == 42
    assert "ask_user" not in names

    assert (ds_bad.status_code, mg_bad.status_code) == (401, 401)
    assert unknown.status_code == 404
    assert sibling.status_code == 404


async def test_empty_api_key_is_treated_as_unset():
    """MANAGEMENT_MCP_API_KEY="" must not make "Bearer " a valid credential."""
    wrapper = _ManagementAuthWrapper(_inner_app(), api_key="", oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        bare = await c.post("/management", headers={"Authorization": "Bearer "})
        empty = await c.post("/management", headers={"Authorization": "Bearer"})
        none = await c.post("/management")
    assert (bare.status_code, empty.status_code, none.status_code) == (401, 401, 401)


async def test_whitespace_api_key_is_treated_as_unset():
    wrapper = _ManagementAuthWrapper(_inner_app(), api_key="   ", oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        resp = await c.post("/management", headers={"Authorization": "Bearer    "})
    assert resp.status_code == 401


async def test_non_http_scopes_never_reach_the_inner_app():
    """A websocket scope must be closed here, not forwarded unauthenticated."""
    reached = False

    async def inner(scope, receive, send):
        nonlocal reached
        reached = True

    wrapper = _ManagementAuthWrapper(inner, api_key="secret", oauth_enabled=False)
    sent: list = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "websocket.connect"}

    await wrapper({"type": "websocket", "path": "/management", "headers": []}, receive, send)
    assert reached is False
    assert sent == [{"type": "websocket.close", "code": 1008}]

    sent.clear()
    await wrapper({"type": "lifespan"}, receive, send)
    assert reached is False
    assert sent == []


# ---------------------------------------------------------------------------
# DNS-rebinding allow list (FastMCP returns 421 for a host it was not told about)
# ---------------------------------------------------------------------------

async def test_configured_allowed_hosts_admit_a_real_hostname():
    from app.api.mcp.management_server import build_management_mcp

    mcp = build_management_mcp(["langgraph.airteam.cloud"])
    inner = mcp.streamable_http_app()  # also creates the session manager
    async with mcp.session_manager.run():
        async with AsyncClient(
            transport=ASGITransport(app=inner),
            base_url="http://langgraph.airteam.cloud",
        ) as c:
            allowed = await c.post("/management", headers=_MCP_HEADERS, json=_INITIALIZE)
        async with AsyncClient(
            transport=ASGITransport(app=inner),
            base_url="http://evil.example.com",
        ) as c:
            rejected = await c.post("/management", headers=_MCP_HEADERS, json=_INITIALIZE)
    assert allowed.status_code == 200, allowed.text
    assert rejected.status_code == 421


async def test_default_allowed_hosts_keep_local_dev_working():
    from app.api.mcp.management_server import build_management_mcp

    mcp = build_management_mcp()
    inner = mcp.streamable_http_app()  # also creates the session manager
    async with mcp.session_manager.run():
        for host in ("localhost", "localhost:8000", "127.0.0.1:8000"):
            async with AsyncClient(
                transport=ASGITransport(app=inner),
                base_url=f"http://{host}",
            ) as c:
                resp = await c.post("/management", headers=_MCP_HEADERS, json=_INITIALIZE)
            assert resp.status_code == 200, (host, resp.text)
        async with AsyncClient(
            transport=ASGITransport(app=inner),
            base_url="http://langgraph.airteam.cloud",
        ) as c:
            resp = await c.post("/management", headers=_MCP_HEADERS, json=_INITIALIZE)
    assert resp.status_code == 421


async def test_non_ascii_authorization_header_is_401_not_500(monkeypatch):
    """A header byte >= 0x80 must not reach secrets.compare_digest as a str.

    ``secrets.compare_digest`` raises TypeError on non-ASCII ``str``, so a
    latin-1-decoded ``Authorization`` header turned every such request into a
    500 (with a full traceback logged) instead of a 401.  Driven at the raw-ASGI
    level because an HTTP client normalizes/rejects that header value.
    """
    from app.api import app as app_module

    settings = app_module.get_settings()
    monkeypatch.setattr(settings, "oauth_enabled", False, raising=False)
    monkeypatch.setattr(settings, "management_mcp_enabled", True, raising=False)
    monkeypatch.setattr(settings, "management_mcp_api_key", "mgmt-key", raising=False)
    monkeypatch.setattr(settings, "mcp_datasources_api_key", "ds-key", raising=False)
    monkeypatch.setattr(settings, "management_mcp_allowed_hosts", None, raising=False)

    fastapi_app = app_module.create_app()

    async def probe(path: str) -> int:
        statuses: list[int] = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        await fastapi_app(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "root_path": "",
                "query_string": b"",
                "headers": [
                    (b"host", b"localhost:8000"),
                    (b"content-type", b"application/json"),
                    (b"authorization", b"Bearer \xff\xfe"),
                ],
                "client": ("127.0.0.1", 51234),
                "server": ("localhost", 8000),
            },
            receive,
            send,
        )
        return statuses[0]

    assert await probe("/mcp/datasources") == 401
    assert await probe("/mcp/management") == 401


# ---------------------------------------------------------------------------
# Non-ASCII bearer through the OAuth fallback → 401, never a 500
# ---------------------------------------------------------------------------

class _ExplodingTransport(httpx.AsyncBaseTransport):
    """Transport-level stub: any outbound call at all fails the test."""

    def __init__(self) -> None:
        self.calls: list = []

    async def handle_async_request(self, request):  # pragma: no cover - must not run
        self.calls.append(request)
        raise AssertionError(f"unexpected outbound request to {request.url}")


async def test_non_ascii_bearer_through_oauth_fallback_is_401(monkeypatch):
    """Raw byte header + real AuthService: the wrapper must still 401.

    The auth service is deliberately NOT mocked here — the prod bug lived in the
    real outbound httpx call, so it is stubbed at the transport level instead and
    must never be reached.
    """
    transport = _ExplodingTransport()
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **kw: real_client(*a, **{**kw, "transport": transport}),
    )
    service = AuthService(
        jwks_url="https://auth.example.com/keys", issuer="https://auth.example.com"
    )
    wrapper = _ManagementAuthWrapper(
        _inner_app(), api_key="secret", oauth_enabled=True, auth_service=service
    )

    status = await _raw_asgi_post(wrapper, "/management", auth=b"Bearer \xff\xfe\xc3\xbf")

    assert status == 401
    assert transport.calls == []


async def _raw_asgi_post(app, path: str, auth: bytes | None = None) -> int:
    """Drive an ASGI app directly so the Authorization header keeps raw bytes."""
    headers = [(b"host", b"t")]
    if auth is not None:
        headers.append((b"authorization", auth))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("t", 80),
    }
    status: dict = {}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]

    await app(scope, receive, send)
    return status["code"]


# ---------------------------------------------------------------------------
# API keys with surrounding whitespace (Secret Manager trailing newline)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["secret\n", "  secret  ", "\tsecret\r\n"])
async def test_api_key_with_surrounding_whitespace_still_matches(key):
    wrapper = _ManagementAuthWrapper(_inner_app(), api_key=key, oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        resp = await c.post("/management", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


@pytest.mark.parametrize("key", ["", "   ", "\n"])
async def test_empty_api_key_counts_as_unset_and_fails_closed(key):
    wrapper = _ManagementAuthWrapper(_inner_app(), api_key=key, oauth_enabled=False)
    async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
        bare = await c.post("/management", headers={"Authorization": "Bearer "})
        none = await c.post("/management")
    assert bare.status_code == 401
    assert none.status_code == 401
