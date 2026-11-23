"""Shared company memory (gbrain) support, end to end.

- brain grants mint, verify, and fail closed (:class:`TestBrainGrant`);
- effective access is min(agent, workflow) (:class:`TestEffectiveAccess`);
- ``_build_agent_config`` adds the ``brain`` entry exactly when it should
  (:class:`TestAgentConfig`);
- the mounted server filters list_tools AND refuses call_tool by grant, forces
  the write namespace and injects provenance (:class:`TestBrainTools`);
- the gate accepts only a brain grant (:class:`TestBrainAuthGate`);
- ``/health/brain`` reports three states without leaking (:class:`TestHealth`);
- ``brain_access`` round-trips through the agent and workflow APIs
  (:class:`TestApiRoundTrip`);
- the app boots without gbrain and does not mount ``/mcp/brain``
  (:class:`TestAppMount`).
"""
from __future__ import annotations

import itertools
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from mcp.server.fastmcp.exceptions import ToolError

from app.api.app import _BrainAuthWrapper
from app.api.mcp.brain_server import (
    READ_TOOLS,
    WRITE_TOOLS,
    build_brain_mcp,
    current_brain_grant,
    namespaced_slug,
    set_current_brain_grant,
)
from app.core.config import Settings
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.brain_access import effective_brain_access, normalize_brain_access
from app.infrastructure.auth.brain_grant import (
    BrainGrant,
    looks_like_brain_grant,
    mint_brain_grant,
    verify_brain_grant,
)
from app.infrastructure.auth.datasource_grant import DatasourceGrant, mint_grant
from app.infrastructure.brain.gbrain_client import BrainStatus, GbrainClient, GbrainError
from app.steps.agent_executor import _build_agent_config

KEY = "brain-signing-key"
GBRAIN_URL = "http://gbrain.internal:3131/mcp"
GBRAIN_SECRET = "gbrain-upstream-secret"


def _settings(configured: bool = True, signing_key: str | None = KEY, **kw) -> Settings:
    extra = {}
    if configured:
        extra["GBRAIN_URL"] = GBRAIN_URL
        extra["GBRAIN_API_KEY"] = GBRAIN_SECRET
    if signing_key:
        extra["BRAIN_GRANT_SIGNING_KEY"] = signing_key
    return Settings(
        MCP_INTEGRATIONS="",
        BASE_URL="https://carrier.example",
        AGENT_CALLBACK_URL="http://carrier.svc:8000",
        **extra,
        **kw,
    )


def _agent(access: str = "none") -> AgentDefinition:
    return AgentDefinition(id="writer", name="Writer", default_runtime="k8s", brain_access=access)


def _brain_entry(cfg: dict) -> dict | None:
    return next((s for s in cfg["mcp_servers"] if s["name"] == "brain"), None)


# ---------------------------------------------------------------------------

class TestBrainGrant:
    def test_round_trip(self):
        token = mint_brain_grant(
            BrainGrant(run_id="r", agent_id="a", workflow_id="w", access="write"), KEY
        )
        assert token.startswith("brg1.") and looks_like_brain_grant(token)
        grant = verify_brain_grant(token, KEY)
        assert (grant.run_id, grant.agent_id, grant.workflow_id, grant.access) == (
            "r", "a", "w", "write",
        )
        assert grant.can_write

    def test_no_key_mints_nothing_and_verifies_nothing(self):
        assert mint_brain_grant(BrainGrant(), None) is None
        assert mint_brain_grant(BrainGrant(), "  ") is None
        token = mint_brain_grant(BrainGrant(), KEY)
        assert verify_brain_grant(token, None) is None

    def test_wrong_key_rejected(self):
        token = mint_brain_grant(BrainGrant(), KEY)
        assert verify_brain_grant(token, "other-key") is None

    def test_tampered_payload_rejected(self):
        import base64
        import json

        token = mint_brain_grant(BrainGrant(access="read"), KEY)
        prefix, body, sig = token.split(".")
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        data["access"] = "write"
        forged = base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
        assert verify_brain_grant(f"{prefix}.{forged}.{sig}", KEY) is None

    def test_expired_rejected(self):
        token = mint_brain_grant(BrainGrant(), KEY, ttl_seconds=10, now=time.time() - 100)
        assert verify_brain_grant(token, KEY) is None

    def test_explicit_zero_expiry_is_filled_from_ttl(self):
        token = mint_brain_grant(BrainGrant(), KEY, ttl_seconds=60)
        assert verify_brain_grant(token, KEY).expires_at > time.time()

    def test_datasource_grant_is_not_a_brain_grant(self):
        ds = mint_grant(DatasourceGrant(grants={"x": ["y"]}), KEY)
        assert not looks_like_brain_grant(ds)
        assert verify_brain_grant(ds, KEY) is None

    def test_garbage_rejected(self):
        for junk in ("", "brg1", "brg1.a.b", "brg1.!!!.###", "Bearer x"):
            assert verify_brain_grant(junk, KEY) is None


class TestEffectiveAccess:
    @pytest.mark.parametrize(
        "agent,workflow,expected",
        [
            ("write", "write", "write"),
            ("write", "read", "read"),
            ("read", "write", "read"),
            ("read", "read", "read"),
            ("none", "write", "none"),
            ("write", "none", "none"),
            ("write", None, "none"),
            ("write", "bogus", "none"),
        ],
    )
    def test_min(self, agent, workflow, expected):
        assert effective_brain_access(agent, workflow) == expected

    def test_normalize(self):
        assert normalize_brain_access(" Write ") == "write"
        assert normalize_brain_access(None) == "none"
        assert normalize_brain_access(3) == "none"


# ---------------------------------------------------------------------------

LEVELS = ["none", "read", "write"]


class TestAgentConfig:
    @pytest.mark.parametrize(
        "agent_level,workflow_level,configured",
        list(itertools.product(LEVELS, LEVELS + [None], [True, False])),
    )
    def test_every_combination(self, agent_level, workflow_level, configured):
        cfg = _build_agent_config(
            _agent(agent_level),
            _settings(configured=configured),
            run_id="run-1",
            workflow_id="wf-1",
            workflow_brain_access=workflow_level,
        )
        entry = _brain_entry(cfg)
        expected = effective_brain_access(agent_level, workflow_level)
        if not configured or expected == "none":
            assert entry is None
            assert "Shared company memory" not in (cfg["system_prompt"] or "")
            return
        assert entry["transport"] == "streamable_http"
        assert entry["env"] == {}
        assert entry["url"] == "http://carrier.svc:8000/mcp/brain"
        grant = verify_brain_grant(entry["api_key"], KEY)
        assert grant.access == expected
        assert (grant.run_id, grant.agent_id, grant.workflow_id) == ("run-1", "writer", "wf-1")
        prompt = cfg["system_prompt"]
        assert "Shared company memory" in prompt
        assert f"{expected} access" in prompt
        if expected == "write":
            assert "carrier/" in prompt and "PII" in prompt
        else:
            assert "brain_put_page" not in prompt

    def test_no_signing_key_means_no_entry(self):
        cfg = _build_agent_config(
            _agent("write"),
            _settings(signing_key=None),
            run_id="r", workflow_id="w", workflow_brain_access="write",
        )
        assert _brain_entry(cfg) is None

    def test_signing_key_falls_back_to_datasource_key(self):
        cfg = _build_agent_config(
            _agent("read"),
            _settings(signing_key=None, DATASOURCE_GRANT_SIGNING_KEY="ds-key"),
            run_id="r", workflow_id="w", workflow_brain_access="read",
        )
        assert verify_brain_grant(_brain_entry(cfg)["api_key"], "ds-key").access == "read"

    def test_requested_but_unconfigured_warns(self, caplog):
        caplog.set_level("WARNING")
        _build_agent_config(
            _agent("read"), _settings(configured=False),
            run_id="r", workflow_id="w", workflow_brain_access="read",
        )
        assert "gbrain is not configured" in caplog.text

    def test_upstream_key_never_reaches_the_agent(self, monkeypatch):
        monkeypatch.setenv("GBRAIN_API_KEY", GBRAIN_SECRET)
        cfg = _build_agent_config(
            _agent("write"), _settings(),
            run_id="r", workflow_id="w", workflow_brain_access="write",
        )
        assert "GBRAIN_API_KEY" not in cfg["credentials"]
        assert GBRAIN_SECRET not in repr(cfg)

    def test_mcp_addon_cannot_grant_brain(self):
        agent = AgentDefinition(
            id="x", addons=[{"type": "mcp", "servers": {"brain": True}}],
        )
        cfg = _build_agent_config(agent, _settings(), run_id="r")
        assert _brain_entry(cfg) is None

    def test_runner_url_is_public(self):
        assert _settings().resolved_runner_mcp_brain_url() == "https://carrier.example/mcp/brain"


# ---------------------------------------------------------------------------

class _FakeBrainClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, dict(arguments or {})))
        return {"ok": True, "tool": name}


class _Container:
    def __init__(self) -> None:
        self.brain_client = _FakeBrainClient()


@pytest.fixture
def brain():
    container = _Container()
    mcp = build_brain_mcp(lambda: container, write_namespace="carrier")
    return mcp, container.brain_client


def _grant(access: str = "read") -> BrainGrant:
    return BrainGrant(run_id="run-9", agent_id="writer", workflow_id="wf-3", access=access)


class TestBrainTools:
    async def test_no_grant_lists_and_calls_nothing(self, brain):
        mcp, client = brain
        set_current_brain_grant(None)
        assert await mcp.list_tools() == []
        with pytest.raises(ToolError):
            await mcp.call_tool("brain_search", {"query": "x"})
        assert client.calls == []

    async def test_read_grant_lists_only_read_tools(self, brain):
        mcp, _ = brain
        set_current_brain_grant(_grant("read"))
        assert {t.name for t in await mcp.list_tools()} == READ_TOOLS

    async def test_write_grant_lists_everything(self, brain):
        mcp, _ = brain
        set_current_brain_grant(_grant("write"))
        assert {t.name for t in await mcp.list_tools()} == READ_TOOLS | WRITE_TOOLS

    @pytest.mark.parametrize("tool,args", [
        ("brain_put_page", {"slug": "a", "content": "c", "reason": "r"}),
        ("brain_remember", {"fact": "f", "entity": "e"}),
    ])
    async def test_read_grant_cannot_call_write_tools(self, brain, tool, args):
        mcp, client = brain
        set_current_brain_grant(_grant("read"))
        with pytest.raises(ToolError, match="not granted"):
            await mcp.call_tool(tool, args)
        assert client.calls == []

    async def test_read_tools_map_to_upstream_names(self, brain):
        mcp, client = brain
        set_current_brain_grant(_grant("read"))
        await mcp.call_tool("brain_search", {"query": "acme"})
        await mcp.call_tool("brain_query", {"query": "how", "limit": 3})
        await mcp.call_tool("brain_get_page", {"slug": "people/x"})
        await mcp.call_tool("brain_recall", {"entity": "acme"})
        await mcp.call_tool("brain_list_pages", {"limit": 500})
        names = [c[0] for c in client.calls]
        assert names == ["search", "query", "get_page", "recall", "list_pages"]
        assert client.calls[0][1] == {"query": "acme", "limit": 10}
        assert client.calls[1][1] == {"query": "how", "limit": 3}
        assert client.calls[4][1]["limit"] == 100  # capped

    async def test_put_page_namespace_provenance_and_args(self, brain):
        mcp, client = brain
        set_current_brain_grant(_grant("write"))
        await mcp.call_tool(
            "brain_put_page",
            {"slug": "customers/acme", "content": "body", "reason": "learned it"},
        )
        name, args = client.calls[-1]
        assert name == "put_page"
        assert args["slug"] == "carrier/customers/acme"
        assert args["content_origin"] == "inferred"
        assert args["reason"] == "carrier run=run-9 workflow=wf-3 agent=writer: learned it"
        assert len(args["request_id"]) == 36
        assert set(args) <= {
            "slug", "content", "content_origin", "reason", "request_id", "expected_revision",
        }

    async def test_put_page_forwards_expected_revision(self, brain):
        mcp, client = brain
        set_current_brain_grant(_grant("write"))
        await mcp.call_tool(
            "brain_put_page",
            {"slug": "carrier/x", "content": "c", "reason": "r", "expected_revision": 4},
        )
        assert client.calls[-1][1]["slug"] == "carrier/x"
        assert client.calls[-1][1]["expected_revision"] == 4

    @pytest.mark.parametrize("bad", ["", "  ", "/etc/x", "../x", "carrier/../other", "a//b", "a\\b"])
    async def test_put_page_rejects_bad_slugs(self, brain, bad):
        mcp, client = brain
        set_current_brain_grant(_grant("write"))
        with pytest.raises(ToolError):
            await mcp.call_tool("brain_put_page", {"slug": bad, "content": "c", "reason": "r"})
        assert client.calls == []

    def test_namespaced_slug(self):
        assert namespaced_slug("x/y", "carrier") == "carrier/x/y"
        assert namespaced_slug("carrier/x", "carrier") == "carrier/x"
        assert namespaced_slug("carrierx/y", "carrier") == "carrier/carrierx/y"

    async def test_remember_forces_provenance(self, brain):
        mcp, client = brain
        set_current_brain_grant(_grant("write"))
        await mcp.call_tool("brain_remember", {"fact": "uses SAP", "entity": "acme"})
        name, args = client.calls[-1]
        assert name == "remember"
        assert args == {
            "fact": "uses SAP",
            "entity": "acme",
            "kind": "fact",
            "provenance": "carrier run=run-9 workflow=wf-3 agent=writer",
            "content_origin": "inferred",
        }
        tool = next(t for t in await mcp.list_tools() if t.name == "brain_remember")
        assert "provenance" not in tool.inputSchema["properties"]

    async def test_upstream_error_becomes_tool_error(self, brain):
        mcp, client = brain

        async def _boom(name, arguments=None):
            raise GbrainError("gbrain tool 'search' failed: nope")

        client.call_tool = _boom
        set_current_brain_grant(_grant("read"))
        with pytest.raises(ToolError, match="nope"):
            await mcp.call_tool("brain_search", {"query": "x"})


class TestGbrainClient:
    async def test_call_tool_prefixes_and_drops_none(self, monkeypatch):
        client = GbrainClient(GBRAIN_URL, GBRAIN_SECRET, tool_prefix="gbrain_")
        seen = {}

        class _Session:
            async def call_tool(self, name, args):
                seen.update(name=name, args=args)
                return MagicMock(isError=False, content=[MagicMock(type="text", text='{"a": 1}')])

        async def _with_session(fn):
            return await fn(_Session())

        monkeypatch.setattr(client, "_with_session", _with_session)
        result = await client.call_tool("recall", {"entity": "x", "query": None})
        assert result == {"a": 1}
        assert seen == {"name": "gbrain_recall", "args": {"entity": "x"}}

    async def test_tool_error_raises(self, monkeypatch):
        client = GbrainClient(GBRAIN_URL)

        class _Session:
            async def call_tool(self, name, args):
                return MagicMock(isError=True, content=[MagicMock(type="text", text="conflict")])

        async def _with_session(fn):
            return await fn(_Session())

        monkeypatch.setattr(client, "_with_session", _with_session)
        with pytest.raises(GbrainError, match="conflict"):
            await client.call_tool("put_page", {})

    async def test_status_unreachable_hides_url_and_is_cached(self):
        client = GbrainClient("http://127.0.0.1:9/mcp", GBRAIN_SECRET, timeout_seconds=2)
        first = await client.status()
        assert first.status == "unavailable"
        assert "127.0.0.1" not in (first.detail or "") and GBRAIN_SECRET not in (first.detail or "")
        assert await client.status() is first

    async def test_status_falls_back_to_listing(self, monkeypatch):
        client = GbrainClient(GBRAIN_URL)
        monkeypatch.setattr(client, "call_tool", AsyncMock(side_effect=GbrainError("no tool")))
        monkeypatch.setattr(client, "list_tool_names", AsyncMock(return_value=["search"]))
        assert (await client.status()).status == "ok"

    def test_from_settings(self):
        assert GbrainClient.from_settings(_settings(configured=False)) is None
        assert GbrainClient.from_settings(_settings()) is not None


# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self) -> None:
        self.seen: list = []

    async def __call__(self, scope, receive, send) -> None:
        self.seen.append(current_brain_grant())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


class TestBrainAuthGate:
    async def _post(self, wrapper, auth: str | None):
        headers = {"Authorization": auth} if auth is not None else {}
        async with AsyncClient(transport=ASGITransport(app=wrapper), base_url="http://t") as c:
            return await c.post("/brain", headers=headers)

    async def test_valid_grant_passes_with_scope(self):
        rec = _Recorder()
        token = mint_brain_grant(_grant("write"), KEY)
        resp = await self._post(_BrainAuthWrapper(rec, KEY), f"Bearer {token}")
        assert resp.status_code == 200
        assert rec.seen[0].access == "write"

    @pytest.mark.parametrize("auth", [None, "", "Bearer ", "Bearer static", "Basic abc"])
    async def test_anything_else_is_401(self, auth):
        rec = _Recorder()
        assert (await self._post(_BrainAuthWrapper(rec, KEY), auth)).status_code == 401
        assert rec.seen == []

    async def test_datasource_grant_is_401(self):
        rec = _Recorder()
        ds = mint_grant(DatasourceGrant(grants={"a": ["b"]}), KEY)
        assert (await self._post(_BrainAuthWrapper(rec, KEY), f"Bearer {ds}")).status_code == 401

    async def test_no_signing_key_fails_closed(self):
        rec = _Recorder()
        token = mint_brain_grant(_grant(), KEY)
        assert (await self._post(_BrainAuthWrapper(rec, None), f"Bearer {token}")).status_code == 401


# ---------------------------------------------------------------------------

def _app_with_container(container):
    from app.api.app import create_app
    from app.core.config import get_settings

    # get_settings is lru-cached; read the environment this test set up.
    get_settings.cache_clear()
    app = create_app()
    get_settings.cache_clear()
    app.state.container = container
    return app


class TestHealth:
    async def _get(self, brain_client):
        container = MagicMock()
        container.brain_client = brain_client
        app = _app_with_container(container)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            return await c.get("/health/brain")

    async def test_not_configured(self):
        resp = await self._get(None)
        assert resp.status_code == 200
        assert resp.json() == {"status": "not_configured", "detail": None}

    async def test_ok(self):
        client = MagicMock()
        client.status = AsyncMock(return_value=BrainStatus("ok"))
        assert (await self._get(client)).json() == {"status": "ok", "detail": None}

    async def test_unavailable(self):
        client = MagicMock()
        client.status = AsyncMock(return_value=BrainStatus("unavailable", "timed out"))
        assert (await self._get(client)).json() == {"status": "unavailable", "detail": "timed out"}

    async def test_never_raises(self):
        client = MagicMock()
        client.status = AsyncMock(side_effect=RuntimeError(GBRAIN_URL))
        resp = await self._get(client)
        assert resp.status_code == 200
        assert resp.json()["status"] == "unavailable"
        assert GBRAIN_URL not in resp.text

    async def test_unauthenticated_under_oauth(self):
        from app.api.middleware.auth import _is_unprotected

        assert _is_unprotected("GET", "/health/brain")
        assert _is_unprotected("POST", "/mcp/brain")
        assert _is_unprotected("POST", "/mcp/brain/")
        assert not _is_unprotected("POST", "/mcp/brainx")


# ---------------------------------------------------------------------------

class TestApiRoundTrip:
    async def test_agent_brain_access(self):
        from tests.test_agents_api import InMemoryAgentBackend, _build_container

        backend = InMemoryAgentBackend()
        app = _app_with_container(_build_container(backend))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            created = await c.post("/api/v1/agents", json={"id": "a", "name": "A", "brain_access": "write"})
            assert created.status_code == 201, created.text
            assert created.json()["brain_access"] == "write"
            assert (await c.get("/api/v1/agents/a")).json()["brain_access"] == "write"
            # Omitted on update keeps the stored value.
            put = await c.put("/api/v1/agents/a", json={"name": "A2"})
            assert put.status_code == 200, put.text
            assert put.json()["brain_access"] == "write"
            put = await c.put("/api/v1/agents/a", json={"name": "A2", "brain_access": "read"})
            assert put.json()["brain_access"] == "read"
            bad = await c.put("/api/v1/agents/a", json={"name": "A2", "brain_access": "admin"})
            assert bad.status_code == 422
            default = await c.post("/api/v1/agents", json={"id": "b", "name": "B"})
            assert default.json()["brain_access"] == "none"

    async def test_workflow_brain_access(self):
        from tests.test_workflow_flag_preservation import _container

        container, backend = _container()
        stored = backend.get.return_value
        stored.brain_access = "read"
        app = _app_with_container(container)
        steps = [{"id": "s1", "type": "llm", "output_key": "answer", "user_template": "{request}"}]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.put("/api/v1/workflows/wf", json={"name": "WF", "steps": steps})
            assert resp.status_code == 200, resp.text
            assert resp.json()["brain_access"] == "read"
            resp = await c.put(
                "/api/v1/workflows/wf",
                json={"name": "WF", "steps": steps, "brain_access": "write"},
            )
            assert resp.json()["brain_access"] == "write"

            backend.get = AsyncMock(return_value=None)
            backend.create = AsyncMock(side_effect=lambda defn: defn)
            created = await c.post(
                "/api/v1/workflows", json={"id": "wf2", "name": "W2", "steps": steps},
            )
            assert created.status_code in (200, 201), created.text
            assert created.json()["brain_access"] == "none"

    def test_workflow_definition_carries_it_to_the_runner(self):
        from app.domain.models.workflow_definition import WorkflowDefinition
        from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner

        steps = [{"id": "s1", "type": "llm", "output_key": "a", "user_template": "{request}"}]
        defn = WorkflowDefinition(id="wf", steps=steps, brain_access="write")
        assert defn.to_raw_dict()["brain_access"] == "write"
        assert YamlGraphRunner(defn.to_raw_dict(), MagicMock(), MagicMock())._brain_access == "write"
        legacy = {"id": "old", "steps": steps}
        assert YamlGraphRunner(legacy, MagicMock(), MagicMock())._brain_access == "none"

    async def test_management_tools(self):
        from app.application import management_tools as core
        from tests.test_agents_api import InMemoryAgentBackend

        deps = MagicMock()
        deps.agent_backend = InMemoryAgentBackend()
        out = await core.create_agent(deps, "m", "M", brain_access="read")
        assert "created" in out
        assert (await deps.agent_backend.get("m")).brain_access == "read"
        await core.update_agent(deps, "m", name="M2")
        assert (await deps.agent_backend.get("m")).brain_access == "read"
        await core.update_agent(deps, "m", brain_access="write")
        assert (await deps.agent_backend.get("m")).brain_access == "write"
        assert "Invalid brain_access" in await core.update_agent(deps, "m", brain_access="x")


# ---------------------------------------------------------------------------

class TestAppMount:
    async def test_unconfigured_app_boots_and_brain_is_404(self, monkeypatch):
        monkeypatch.delenv("GBRAIN_URL", raising=False)
        from tests.test_agents_api import InMemoryAgentBackend, _build_container

        app = _app_with_container(_build_container(InMemoryAgentBackend()))
        assert app.state.brain_mcp is None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            assert (await c.post("/mcp/brain", json={})).status_code == 404
            assert (await c.get("/health")).json() == {"status": "ok"}

    async def test_configured_app_serves_scoped_tools_over_http(self, monkeypatch):
        monkeypatch.setenv("GBRAIN_URL", GBRAIN_URL)
        monkeypatch.setenv("BRAIN_GRANT_SIGNING_KEY", KEY)
        container = _Container()
        app = _app_with_container(container)
        brain_mcp = app.state.brain_mcp
        assert brain_mcp is not None
        read_token = mint_brain_grant(_grant("read"), KEY)

        async with brain_mcp.session_manager.run():
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://127.0.0.1:8000"
            ) as c:
                async def rpc(token, method, params=None):
                    return await c.post(
                        "/mcp/brain",
                        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Accept": "application/json, text/event-stream",
                            "Content-Type": "application/json",
                            "MCP-Protocol-Version": "2025-06-18",
                        },
                    )

                assert (await rpc("nope", "tools/list")).status_code == 401
                listed = await rpc(read_token, "tools/list")
                assert listed.status_code == 200, listed.text
                assert "brain_search" in listed.text
                assert "brain_put_page" not in listed.text
                called = await rpc(
                    read_token, "tools/call",
                    {"name": "brain_remember", "arguments": {"fact": "f", "entity": "e"}},
                )
                assert "not granted" in called.text
                ok = await rpc(
                    read_token, "tools/call",
                    {"name": "brain_search", "arguments": {"query": "acme"}},
                )
                assert ok.status_code == 200
                assert container.brain_client.calls == [("search", {"query": "acme", "limit": 10})]
