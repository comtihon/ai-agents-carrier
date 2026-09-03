"""Tests for the scoped ``datasource`` agent addon, end to end.

The property under test throughout is one sentence: **the checked operations
list is the authorization, and it is enforced by this backend, not by the
agent.**  Everything else follows from that —

- the addon round-trips through the REST API, so what an operator ticks is what
  gets stored (:class:`TestAddonRoundTrip`);
- a grant is minted from the union of every attached addon and delivered as the
  MCP entry's ``api_key``, carrying no data source secret
  (:class:`TestGrantMinting`);
- the mounted server filters ``list_tools`` **and** refuses ``call_tool``, so
  guessing a tool name that was never listed does not work
  (:class:`TestScopedToolSurface`);
- a forged, tampered, or expired grant does not verify (:class:`TestGrantTokens`).
"""
from __future__ import annotations

import time

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from app.api.app import _DatasourcesAuthWrapper
from app.api.mcp.datasources_server import (
    build_datasources_mcp,
    grant_tool_names,
    rebuild_datasource_tools,
    reset_current_grant,
    set_current_grant,
    tool_name_for,
)
from app.core.config import Settings
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.auth.datasource_grant import (
    DatasourceGrant,
    looks_like_grant,
    mint_grant,
    verify_grant,
)
from app.infrastructure.datasources.executor import DataSourceExecutor
from app.steps.agent_executor import _build_agent_config
from tests.test_datasources_api import InMemoryDataSourceBackend


def _executor(**kwargs):
    """An executor with a throw-away stream store.

    Every data source result is written to a stream and returned as a
    reference, so an executor needs somewhere to write. Tests that assert on
    records call ``execute_value``, which reads the stream back.
    """
    import tempfile

    from app.infrastructure.datasources.datastream import LocalDiskStreamStore

    kwargs.setdefault("stream_store", LocalDiskStreamStore(tempfile.mkdtemp()))
    return DataSourceExecutor(**kwargs)

SIGNING_KEY = "grant-signing-key"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _projects_source() -> DataSourceDefinition:
    """A source with a read, a write and a delete — the realistic shape.

    Modelled on the live ``afp-projects`` source, which authenticates as the
    carrier's own service identity and holds 23 mutating operations.
    """
    return DataSourceDefinition.model_validate({
        "id": "afp-projects",
        "name": "AFP Projects",
        "description": "Project data of the AFP backend",
        "base_url": "https://afp.example",
        "auth": {"type": "bearer", "token": "upstream-secret"},
        "operations": [
            {
                "name": "get_project",
                "method": "GET",
                "path": "/projects/{params.id}",
                "params": [{"name": "id"}],
            },
            {
                "name": "deliver_project",
                "method": "POST",
                "path": "/projects/{params.id}/deliver",
                "params": [{"name": "id"}],
            },
            {
                "name": "delete_project",
                "method": "DELETE",
                "path": "/projects/{params.id}",
                "params": [{"name": "id"}],
            },
        ],
    })


def _crm_source() -> DataSourceDefinition:
    return DataSourceDefinition.model_validate({
        "id": "hubspot-crm",
        "name": "HubSpot",
        "base_url": "https://crm.example",
        "operations": [
            {"name": "list_contacts", "method": "GET", "path": "/contacts"},
        ],
    })


class _Container:
    def __init__(self, backend) -> None:
        self.data_source_backend = backend
        self.data_source_executor = _executor()


@pytest.fixture
async def bridge():
    """A built MCP server carrying both sources, plus its container."""
    backend = InMemoryDataSourceBackend()
    await backend.create(_projects_source())
    await backend.create(_crm_source())
    container = _Container(backend)
    mcp = build_datasources_mcp()
    await rebuild_datasource_tools(mcp, backend, lambda: container)
    return mcp, container


@pytest.fixture
async def api():
    """The real FastAPI app over an in-memory agent backend."""
    from app.api.app import create_app
    from tests.test_agents_api import InMemoryAgentBackend, _build_container

    backend = InMemoryAgentBackend()
    app = create_app()
    app.state.container = _build_container(backend)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def scoped():
    """Apply a grant inside the test body.

    No teardown: each async test runs as its own asyncio task, which copies the
    context rather than sharing it, so a value set here cannot outlive the test.
    ``test_the_grant_does_not_leak_into_the_next_request`` covers the set/reset
    pairing that the production ASGI gate relies on.
    """
    return set_current_grant


def _settings(**kwargs) -> Settings:
    return Settings(
        MCP_INTEGRATIONS="",
        BASE_URL="https://carrier.example",
        DATASOURCE_GRANT_SIGNING_KEY=SIGNING_KEY,
        **kwargs,
    )


def _agent(*addons) -> AgentDefinition:
    return AgentDefinition(
        id="planner", name="Planner", default_runtime="k8s", addons=list(addons)
    )


def _datasource_addon(source_id: str, operations: list[str]) -> dict:
    return {
        "type": "datasource",
        "source_id": source_id,
        "allowed_operations": operations,
    }


def _ds_entry(cfg: dict) -> dict | None:
    return next(
        (s for s in cfg["mcp_servers"] if s["name"] == "datasources"), None
    )


# ---------------------------------------------------------------------------
# Addon model + REST round-trip
# ---------------------------------------------------------------------------

class TestAddonRoundTrip:
    def test_addon_parses_out_of_the_discriminated_union(self):
        agent = _agent(_datasource_addon("afp-projects", ["get_project"]))
        addon = agent.datasource_addons[0]
        assert addon.type == "datasource"
        assert addon.source_id == "afp-projects"
        assert addon.allowed_operations == ["get_project"]

    def test_several_addons_are_all_returned_not_just_the_first(self):
        agent = _agent(
            _datasource_addon("afp-projects", ["get_project"]),
            _datasource_addon("hubspot-crm", ["list_contacts"]),
        )
        assert [a.source_id for a in agent.datasource_addons] == [
            "afp-projects", "hubspot-crm",
        ]
        # get_addon() returns only the first — the reason datasource_addons exists.
        assert agent.get_addon("datasource").source_id == "afp-projects"

    def test_allowed_operations_defaults_to_empty_not_to_everything(self):
        agent = _agent({"type": "datasource", "source_id": "afp-projects"})
        assert agent.datasource_addons[0].allowed_operations == []

    def test_addon_survives_a_dump_and_reload(self):
        agent = _agent(
            _datasource_addon("afp-projects", ["get_project", "deliver_project"]),
            {"type": "mcp", "servers": {"github": True}},
        )
        reloaded = AgentDefinition.model_validate(agent.model_dump(mode="json"))
        assert [a.model_dump() for a in reloaded.datasource_addons] == [
            a.model_dump() for a in agent.datasource_addons
        ]
        # The sibling addon types still parse alongside it.
        assert reloaded.mcp_addon.enabled_servers() == {"github"}

    async def test_addon_round_trips_through_the_rest_api(self, api):
        """POST /agents → GET → PUT /addons, the path the UI actually uses."""
        addon = _datasource_addon("afp-projects", ["get_project", "delete_project"])
        created = await api.post(
            "/api/v1/agents",
            json={"id": "planner", "name": "Planner", "addons": [addon]},
        )
        assert created.status_code in (200, 201), created.text
        assert created.json()["addons"] == [{**addon, "hidden": False}]

        fetched = await api.get("/api/v1/agents/planner")
        assert fetched.json()["addons"][0]["allowed_operations"] == [
            "get_project", "delete_project",
        ]

        # Two addons on one agent survive the round trip as two entries.
        both = [addon, _datasource_addon("hubspot-crm", ["list_contacts"])]
        put = await api.put("/api/v1/agents/planner/addons", json={"addons": both})
        assert put.status_code == 200, put.text
        assert [a["source_id"] for a in put.json()["addons"]] == [
            "afp-projects", "hubspot-crm",
        ]

    async def test_rest_response_carries_no_auth_material(self, api):
        addon = _datasource_addon("afp-projects", ["get_project"])
        await api.post(
            "/api/v1/agents",
            json={"id": "planner", "name": "Planner", "addons": [addon]},
        )
        body = (await api.get("/api/v1/agents/planner")).text
        for secret in ("upstream-secret", SIGNING_KEY, "api_key", "token"):
            assert secret not in body


# ---------------------------------------------------------------------------
# Grant minting and delivery
# ---------------------------------------------------------------------------

class TestGrantMinting:
    def test_no_datasource_addon_means_no_datasources_entry(self):
        cfg = _build_agent_config(_agent(), _settings())
        assert _ds_entry(cfg) is None

    def test_entry_carries_the_agent_reachable_url_and_a_verifying_grant(self):
        cfg = _build_agent_config(
            _agent(_datasource_addon("afp-projects", ["get_project"])),
            _settings(),
            run_id="run-1",
        )
        entry = _ds_entry(cfg)
        assert entry["url"] == "https://carrier.example/mcp/datasources"

        grant = verify_grant(entry["api_key"], SIGNING_KEY)
        assert grant is not None
        assert grant.grants == {"afp-projects": ["get_project"]}
        assert grant.run_id == "run-1"
        assert grant.agent_id == "planner"
        assert grant.expires_at > int(time.time())

    def test_the_grant_is_the_union_across_every_attached_addon(self):
        cfg = _build_agent_config(
            _agent(
                _datasource_addon("afp-projects", ["get_project"]),
                _datasource_addon("afp-projects", ["deliver_project", "get_project"]),
                _datasource_addon("hubspot-crm", ["list_contacts"]),
            ),
            _settings(),
        )
        grant = verify_grant(_ds_entry(cfg)["api_key"], SIGNING_KEY)
        # Same source twice merges and deduplicates; a second source is added.
        assert grant.grants == {
            "afp-projects": ["get_project", "deliver_project"],
            "hubspot-crm": ["list_contacts"],
        }

    def test_an_addon_with_no_ticked_operations_produces_no_entry(self):
        """Empty allowed_operations is a deny, so there is nothing to grant."""
        cfg = _build_agent_config(
            _agent(_datasource_addon("afp-projects", [])), _settings()
        )
        assert _ds_entry(cfg) is None

    def test_writes_need_no_extra_flag_to_be_granted(self):
        cfg = _build_agent_config(
            _agent(
                _datasource_addon("afp-projects", ["deliver_project", "delete_project"])
            ),
            _settings(),
        )
        grant = verify_grant(_ds_entry(cfg)["api_key"], SIGNING_KEY)
        assert grant.grants["afp-projects"] == ["deliver_project", "delete_project"]

    def test_no_signing_key_grants_nothing_rather_than_something_unsigned(self):
        settings = Settings(
            MCP_INTEGRATIONS="",
            BASE_URL="https://carrier.example",
            MCP_DATASOURCES_API_KEY="",
            DATASOURCE_GRANT_SIGNING_KEY="",
        )
        assert settings.resolved_datasource_grant_signing_key() is None
        cfg = _build_agent_config(
            _agent(_datasource_addon("afp-projects", ["get_project"])), settings
        )
        assert _ds_entry(cfg) is None

    def test_the_signing_key_falls_back_to_the_static_datasources_key(self):
        settings = Settings(
            MCP_INTEGRATIONS="",
            BASE_URL="https://carrier.example",
            MCP_DATASOURCES_API_KEY="ds-token",
        )
        cfg = _build_agent_config(
            _agent(_datasource_addon("afp-projects", ["get_project"])), settings
        )
        assert verify_grant(_ds_entry(cfg)["api_key"], "ds-token") is not None

    def test_the_payload_never_carries_the_data_source_credential(self):
        """The one assertion that proves credential isolation at the boundary."""
        import json

        cfg = _build_agent_config(
            _agent(_datasource_addon("afp-projects", ["get_project"])),
            _settings(),
            run_id="run-1",
        )
        dumped = json.dumps(cfg, default=str)
        # Neither the upstream secret nor the key that signs grants travels.
        assert "upstream-secret" not in dumped
        assert SIGNING_KEY not in dumped
        # And the grant itself is decodable by anyone — it is a capability, so
        # it must not be relied on to hide anything.
        grant = verify_grant(_ds_entry(cfg)["api_key"], SIGNING_KEY)
        assert set(grant.model_dump()) == {
            "version", "run_id", "agent_id", "grants", "expires_at",
        }


# ---------------------------------------------------------------------------
# Grant token integrity
# ---------------------------------------------------------------------------

class TestGrantTokens:
    def _grant(self, **kwargs) -> DatasourceGrant:
        return DatasourceGrant(
            run_id="run-1",
            agent_id="planner",
            grants={"afp-projects": ["get_project"]},
            **kwargs,
        )

    def test_a_minted_grant_verifies_and_is_recognisable(self):
        token = mint_grant(self._grant(), SIGNING_KEY)
        assert looks_like_grant(token)
        assert verify_grant(token, SIGNING_KEY).grants == {
            "afp-projects": ["get_project"]
        }

    def test_a_different_key_does_not_verify(self):
        token = mint_grant(self._grant(), SIGNING_KEY)
        assert verify_grant(token, "some-other-key") is None

    def test_a_tampered_payload_does_not_verify(self):
        """The whole point: an agent cannot widen its own grant."""
        import base64
        import json

        token = mint_grant(self._grant(), SIGNING_KEY)
        prefix, body, signature = token.split(".")
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        payload["grants"]["afp-projects"].append("delete_project")
        forged_body = (
            base64.urlsafe_b64encode(json.dumps(payload).encode())
            .decode()
            .rstrip("=")
        )
        assert verify_grant(f"{prefix}.{forged_body}.{signature}", SIGNING_KEY) is None

    def test_a_wholly_self_signed_grant_does_not_verify(self):
        forged = mint_grant(
            DatasourceGrant(grants={"afp-projects": ["delete_project"]}),
            "a-key-the-agent-made-up",
        )
        assert verify_grant(forged, SIGNING_KEY) is None

    def test_an_expired_grant_does_not_verify(self):
        token = mint_grant(self._grant(), SIGNING_KEY, ttl_seconds=60)
        assert verify_grant(token, SIGNING_KEY, now=time.time() + 61) is None
        assert verify_grant(token, SIGNING_KEY, now=time.time() + 30) is not None

    def test_a_grant_with_no_stated_expiry_does_not_verify(self):
        """Minting always sets one; a hand-built eternal grant is refused."""
        import base64
        import hashlib
        import hmac
        import json

        payload = self._grant(expires_at=0).model_dump(mode="json")
        body = (
            base64.urlsafe_b64encode(json.dumps(payload).encode())
            .decode()
            .rstrip("=")
        )
        signature = (
            base64.urlsafe_b64encode(
                hmac.new(
                    SIGNING_KEY.encode(), body.encode(), hashlib.sha256
                ).digest()
            )
            .decode()
            .rstrip("=")
        )
        assert verify_grant(f"dsg1.{body}.{signature}", SIGNING_KEY) is None

    def test_nothing_verifies_without_a_signing_key(self):
        token = mint_grant(self._grant(), SIGNING_KEY)
        assert verify_grant(token, None) is None
        assert verify_grant(token, "   ") is None

    def test_junk_is_rejected_without_raising(self):
        for junk in ("", "not-a-grant", "dsg1.x", "dsg1..", "dsg1.@@@.@@@",
                     "dsg9.a.b", "Bearer dsg1.a.b"):
            assert verify_grant(junk, SIGNING_KEY) is None

    def test_a_non_ascii_signature_is_rejected_not_a_crash(self):
        """A header arrives latin-1 decoded, so any byte can appear in it."""
        token = mint_grant(self._grant(), SIGNING_KEY)
        prefix, body, _ = token.split(".")
        assert verify_grant(f"{prefix}.{body}.\xff\xfe", SIGNING_KEY) is None

    def test_an_empty_grant_authorizes_no_tool(self):
        assert DatasourceGrant().is_empty()
        assert DatasourceGrant(grants={"afp-projects": []}).is_empty()
        assert grant_tool_names(DatasourceGrant(grants={"afp-projects": []})) == frozenset()


# ---------------------------------------------------------------------------
# The scoped tool surface — list AND call
# ---------------------------------------------------------------------------

class TestScopedToolSurface:
    async def test_without_a_grant_the_caller_sees_everything(self, bridge):
        """The backend's own loopback client stays unscoped."""
        mcp, _ = bridge
        names = {t.name for t in await mcp.list_tools()}
        assert names == {
            "ds_afp-projects_get_project",
            "ds_afp-projects_deliver_project",
            "ds_afp-projects_delete_project",
            "ds_hubspot-crm_list_contacts",
        }

    async def test_list_tools_shows_only_the_granted_operations(self, bridge, scoped):
        mcp, _ = bridge
        scoped(DatasourceGrant(grants={"afp-projects": ["get_project"]}))
        assert {t.name for t in await mcp.list_tools()} == {
            "ds_afp-projects_get_project"
        }

    async def test_list_tools_unions_across_sources(self, bridge, scoped):
        mcp, _ = bridge
        scoped(DatasourceGrant(grants={
            "afp-projects": ["get_project", "deliver_project"],
            "hubspot-crm": ["list_contacts"],
        }))
        assert {t.name for t in await mcp.list_tools()} == {
            "ds_afp-projects_get_project",
            "ds_afp-projects_deliver_project",
            "ds_hubspot-crm_list_contacts",
        }

    async def test_an_empty_grant_lists_no_tools_at_all(self, bridge, scoped):
        mcp, _ = bridge
        scoped(DatasourceGrant(grants={"afp-projects": []}))
        assert await mcp.list_tools() == []

    async def test_a_granted_operation_is_callable(self, bridge, scoped, monkeypatch):
        mcp, container = bridge
        seen: dict = {}

        async def _fake_execute(source, operation, params):
            seen.update({"source": source.id, "operation": operation})
            return {"ok": True}

        monkeypatch.setattr(container.data_source_executor, "execute", _fake_execute)
        scoped(DatasourceGrant(grants={"afp-projects": ["get_project"]}))

        await mcp.call_tool("ds_afp-projects_get_project", {"id": "42"})
        assert seen == {"source": "afp-projects", "operation": "get_project"}

    async def test_a_granted_write_is_callable_with_no_extra_ceremony(
        self, bridge, scoped, monkeypatch
    ):
        """The allow-list is the whole gate: a ticked DELETE just works."""
        mcp, container = bridge
        seen: dict = {}

        async def _fake_execute(source, operation, params):
            seen["operation"] = operation
            return {"ok": True}

        monkeypatch.setattr(container.data_source_executor, "execute", _fake_execute)
        scoped(DatasourceGrant(grants={"afp-projects": ["delete_project"]}))

        await mcp.call_tool("ds_afp-projects_delete_project", {"id": "42"})
        assert seen == {"operation": "delete_project"}

    async def test_an_ungranted_operation_is_refused_even_when_guessed(
        self, bridge, scoped, monkeypatch
    ):
        """Listing is not authorization. The name is guessable; the call is not."""
        from mcp.server.fastmcp.exceptions import ToolError

        mcp, container = bridge

        async def _must_not_run(source, operation, params):  # pragma: no cover
            raise AssertionError(f"executor reached for ungranted {operation!r}")

        monkeypatch.setattr(container.data_source_executor, "execute", _must_not_run)
        scoped(DatasourceGrant(grants={"afp-projects": ["get_project"]}))

        # It is not listed …
        assert "ds_afp-projects_delete_project" not in {
            t.name for t in await mcp.list_tools()
        }
        # … and naming it directly is refused, before the executor is reached.
        with pytest.raises(ToolError, match="not granted"):
            await mcp.call_tool("ds_afp-projects_delete_project", {"id": "42"})

    async def test_a_different_source_is_refused(self, bridge, scoped, monkeypatch):
        """source_id is baked into the tool name at registration, never read
        from the request body — so there is nothing to substitute."""
        from mcp.server.fastmcp.exceptions import ToolError

        mcp, container = bridge

        async def _must_not_run(source, operation, params):  # pragma: no cover
            raise AssertionError(f"executor reached for {source.id!r}")

        monkeypatch.setattr(container.data_source_executor, "execute", _must_not_run)
        scoped(DatasourceGrant(grants={"afp-projects": ["get_project"]}))

        with pytest.raises(ToolError, match="not granted"):
            await mcp.call_tool("ds_hubspot-crm_list_contacts", {})

    async def test_a_source_id_in_the_arguments_changes_nothing(
        self, bridge, scoped, monkeypatch
    ):
        mcp, container = bridge
        seen: dict = {}

        async def _fake_execute(source, operation, params):
            seen.update({"source": source.id, "params": params})
            return {"ok": True}

        monkeypatch.setattr(container.data_source_executor, "execute", _fake_execute)
        scoped(DatasourceGrant(grants={"afp-projects": ["get_project"]}))

        await mcp.call_tool(
            "ds_afp-projects_get_project",
            {"id": "42", "source_id": "hubspot-crm"},
        )
        assert seen["source"] == "afp-projects"
        # The undeclared key never becomes a param of the operation either.
        assert "source_id" not in seen["params"]

    async def test_the_grant_does_not_leak_into_the_next_request(self, bridge):
        """set/reset is balanced, so one run's scope is not another's."""
        mcp, _ = bridge
        token = set_current_grant(
            DatasourceGrant(grants={"afp-projects": ["get_project"]})
        )
        assert len(await mcp.list_tools()) == 1
        reset_current_grant(token)
        assert len(await mcp.list_tools()) == 4

    async def test_write_operations_are_labelled_as_writes(self, bridge):
        mcp, _ = bridge
        by_name = {t.name: t for t in await mcp.list_tools()}
        assert "[GET · READ]" in by_name["ds_afp-projects_get_project"].description
        assert "[POST · WRITE]" in by_name["ds_afp-projects_deliver_project"].description
        assert "[DELETE · WRITE]" in by_name["ds_afp-projects_delete_project"].description

    async def test_the_input_schema_is_still_derived_per_operation(self, bridge):
        """One tool per operation, not one generic call — so the ParamSpec
        schema survives."""
        mcp, _ = bridge
        schema = {t.name: t.inputSchema for t in await mcp.list_tools()}
        assert set(schema["ds_afp-projects_get_project"]["properties"]) == {"id"}
        assert schema["ds_afp-projects_get_project"]["required"] == ["id"]


# ---------------------------------------------------------------------------
# GraphQL operations that splice caller input into the query document
# ---------------------------------------------------------------------------

class TestGraphqlQueryInjection:
    async def test_an_op_templating_params_into_its_query_is_not_exposed(self, caplog):
        backend = InMemoryDataSourceBackend()
        await backend.create(DataSourceDefinition.model_validate({
            "id": "gql",
            "kind": "graphql",
            "base_url": "https://gql.example",
            "operations": [
                {
                    # Caller input in the query *document* — it could select
                    # other fields or invoke a mutation.
                    "name": "unsafe",
                    "query": "query { project(id: \"{params.id}\") { name } }",
                    "params": [{"name": "id"}],
                },
                {
                    # Caller input in `variables`, where it travels as data.
                    "name": "safe",
                    "query": "query($id: ID!) { project(id: $id) { name } }",
                    "variables": {"id": "{params.id}"},
                    "params": [{"name": "id"}],
                },
            ],
        }))
        mcp = build_datasources_mcp()
        with caplog.at_level("WARNING"):
            await rebuild_datasource_tools(mcp, backend, lambda: _Container(backend))

        names = {t.name for t in await mcp.list_tools()}
        assert names == {"ds_gql_safe"}
        assert any("query document" in r.message for r in caplog.records)

    async def test_a_graphql_op_is_labelled_read_not_get(self):
        backend = InMemoryDataSourceBackend()
        await backend.create(DataSourceDefinition.model_validate({
            "id": "gql",
            "kind": "graphql",
            "base_url": "https://gql.example",
            "operations": [{"name": "safe", "query": "query { me { id } }"}],
        }))
        mcp = build_datasources_mcp()
        await rebuild_datasource_tools(mcp, backend, lambda: _Container(backend))
        tool = (await mcp.list_tools())[0]
        assert "[GRAPHQL · READ]" in tool.description


# ---------------------------------------------------------------------------
# The ASGI gate in front of the mount
# ---------------------------------------------------------------------------

def _inner_app() -> Starlette:
    async def ok(request):
        return PlainTextResponse("ok")

    return Starlette(routes=[Route("/{path:path}", ok, methods=["GET", "POST"])])


class _GrantRecorder:
    """Inner ASGI app that records the grant visible to the mounted server."""

    def __init__(self) -> None:
        self.seen: list = []

    async def __call__(self, scope, receive, send) -> None:
        from app.api.mcp.datasources_server import current_grant

        self.seen.append(current_grant())
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({"type": "http.response.body", "body": b"ok"})


class TestAuthGate:
    def _grant_token(self, **kwargs) -> str:
        return mint_grant(
            DatasourceGrant(
                run_id="run-1",
                agent_id="planner",
                grants={"afp-projects": ["get_project"]},
                **kwargs,
            ),
            SIGNING_KEY,
        )

    async def test_a_valid_grant_passes_and_reaches_the_server_as_scope(self):
        recorder = _GrantRecorder()
        wrapper = _DatasourcesAuthWrapper(
            recorder,
            api_key="static-key",
            oauth_enabled=True,
            grant_signing_key=SIGNING_KEY,
        )
        async with AsyncClient(
            transport=ASGITransport(app=wrapper), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/datasources",
                headers={"Authorization": f"Bearer {self._grant_token()}"},
            )
        assert resp.status_code == 200
        assert recorder.seen[0].grants == {"afp-projects": ["get_project"]}

    async def test_the_static_key_still_passes_unscoped(self):
        recorder = _GrantRecorder()
        wrapper = _DatasourcesAuthWrapper(
            recorder,
            api_key="static-key",
            oauth_enabled=True,
            grant_signing_key=SIGNING_KEY,
        )
        async with AsyncClient(
            transport=ASGITransport(app=wrapper), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/datasources", headers={"Authorization": "Bearer static-key"}
            )
        assert resp.status_code == 200
        # None means "unscoped", and it is set explicitly rather than left over.
        assert recorder.seen == [None]

    async def test_a_tampered_grant_is_401(self):
        wrapper = _DatasourcesAuthWrapper(
            _inner_app(),
            api_key="static-key",
            oauth_enabled=True,
            grant_signing_key=SIGNING_KEY,
        )
        token = self._grant_token()
        async with AsyncClient(
            transport=ASGITransport(app=wrapper), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/datasources",
                headers={"Authorization": f"Bearer {token[:-4]}AAAA"},
            )
        assert resp.status_code == 401

    async def test_an_expired_grant_is_401(self):
        wrapper = _DatasourcesAuthWrapper(
            _inner_app(),
            api_key="static-key",
            oauth_enabled=True,
            grant_signing_key=SIGNING_KEY,
        )
        expired = mint_grant(
            DatasourceGrant(
                grants={"afp-projects": ["get_project"]},
                expires_at=int(time.time()) - 1,
            ),
            SIGNING_KEY,
        )
        async with AsyncClient(
            transport=ASGITransport(app=wrapper), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/datasources", headers={"Authorization": f"Bearer {expired}"}
            )
        assert resp.status_code == 401

    async def test_a_grant_works_with_no_static_key_configured(self):
        """A deployment can run on grants alone — the fail-closed posture still
        rejects everything else."""
        recorder = _GrantRecorder()
        wrapper = _DatasourcesAuthWrapper(
            recorder,
            api_key=None,
            oauth_enabled=True,
            grant_signing_key=SIGNING_KEY,
        )
        async with AsyncClient(
            transport=ASGITransport(app=wrapper), base_url="http://test"
        ) as c:
            granted = await c.post(
                "/datasources",
                headers={"Authorization": f"Bearer {self._grant_token()}"},
            )
            bare = await c.post("/datasources")
        assert granted.status_code == 200
        assert bare.status_code == 401

    async def test_an_open_endpoint_still_honours_a_presented_grant(self):
        """Arriving through an unauthenticated deployment must not widen scope."""
        recorder = _GrantRecorder()
        wrapper = _DatasourcesAuthWrapper(
            recorder,
            api_key=None,
            oauth_enabled=False,
            grant_signing_key=SIGNING_KEY,
        )
        async with AsyncClient(
            transport=ASGITransport(app=wrapper), base_url="http://test"
        ) as c:
            await c.post(
                "/datasources",
                headers={"Authorization": f"Bearer {self._grant_token()}"},
            )
        assert recorder.seen[0].grants == {"afp-projects": ["get_project"]}

    async def test_no_signing_key_means_no_grant_verifies(self):
        wrapper = _DatasourcesAuthWrapper(
            _inner_app(),
            api_key="static-key",
            oauth_enabled=True,
            grant_signing_key=None,
        )
        async with AsyncClient(
            transport=ASGITransport(app=wrapper), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/datasources",
                headers={"Authorization": f"Bearer {self._grant_token()}"},
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Full-stack: the grant survives the real mount, not just a direct call
# ---------------------------------------------------------------------------

async def test_the_grant_reaches_the_mounted_server_over_real_http():
    """The context variable is set by an ASGI wrapper and read inside FastMCP's
    own request handling — different code, possibly a different task.  This
    drives a real JSON-RPC ``tools/list`` through the mount to prove it
    actually arrives, rather than trusting that it should.
    """
    backend = InMemoryDataSourceBackend()
    await backend.create(_projects_source())
    container = _Container(backend)
    mcp = build_datasources_mcp(allowed_hosts=["agents.example"])
    await rebuild_datasource_tools(mcp, backend, lambda: container)

    wrapper = _DatasourcesAuthWrapper(
        mcp.streamable_http_app(),
        api_key="static-key",
        oauth_enabled=True,
        grant_signing_key=SIGNING_KEY,
    )
    token = mint_grant(
        DatasourceGrant(
            run_id="run-1",
            agent_id="planner",
            grants={"afp-projects": ["get_project"]},
        ),
        SIGNING_KEY,
    )

    async with mcp.session_manager.run():
        # The agent-facing host, not loopback: FastMCP's DNS-rebinding guard
        # 421s any Host it was not told about, before the request is read — so
        # this also covers the deployment address reaching the mount at all.
        async with AsyncClient(
            transport=ASGITransport(app=wrapper), base_url="http://agents.example"
        ) as client:
            async def _rpc(auth: str, method: str, params: dict | None = None):
                resp = await client.post(
                    "/datasources",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": params or {},
                    },
                    headers={
                        "Authorization": f"Bearer {auth}",
                        "Accept": "application/json, text/event-stream",
                        "Content-Type": "application/json",
                        "MCP-Protocol-Version": "2025-06-18",
                    },
                )
                return resp

            scoped_list = await _rpc(token, "tools/list")
            assert scoped_list.status_code == 200, scoped_list.text
            assert "ds_afp-projects_get_project" in scoped_list.text
            # The two write operations exist in the process-global tool set and
            # must not appear in this caller's listing.
            assert "deliver_project" not in scoped_list.text
            assert "delete_project" not in scoped_list.text

            # Guessing the name is refused too, over the wire.
            called = await _rpc(
                token,
                "tools/call",
                {
                    "name": "ds_afp-projects_delete_project",
                    "arguments": {"id": "42"},
                },
            )
            assert "not granted" in called.text

            # The static key is the backend's own client and still sees it all.
            unscoped = await _rpc("static-key", "tools/list")
            assert "delete_project" in unscoped.text
