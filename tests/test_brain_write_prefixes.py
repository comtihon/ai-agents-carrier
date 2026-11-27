"""Workflow-scoped extra gbrain write prefixes (``brain_write_prefixes``).

A workflow may let its agents write under slug prefixes besides the carrier
namespace (e.g. ``kb/initiatives/``), bounded by the operator allowlist
``GBRAIN_WRITE_PREFIXES_ALLOWED`` at mint time *and* at call time.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from app.api.mcp.brain_server import (
    build_brain_mcp,
    resolve_write_slug,
    set_current_brain_grant,
)
from app.core.config import Settings, normalize_brain_write_prefix
from app.infrastructure.auth._signed_token import sign_payload
from app.infrastructure.auth.brain_grant import (
    GRANT_PREFIX,
    BrainGrant,
    mint_brain_grant,
    verify_brain_grant,
)
from app.steps.agent_executor import _build_agent_config
from tests.test_brain_shared_memory import (
    KEY,
    _agent,
    _brain_entry,
    _Container,
    _settings,
)

PREFIX = "kb/initiatives/"


# ---------------------------------------------------------------------------
# Settings

class TestSettingsAllowlist:
    def test_default_is_empty(self, monkeypatch):
        monkeypatch.delenv("GBRAIN_WRITE_PREFIXES_ALLOWED", raising=False)
        assert Settings(_env_file=None).gbrain_write_prefixes_allowed == []

    def test_comma_separated_env_is_normalized(self, monkeypatch):
        monkeypatch.setenv(
            "GBRAIN_WRITE_PREFIXES_ALLOWED",
            " /kb/initiatives , kb/initiatives/, team/docs/x/ ",
        )
        assert Settings(_env_file=None).gbrain_write_prefixes_allowed == [
            "kb/initiatives/", "team/docs/x/",
        ]

    def test_unsafe_and_too_broad_are_dropped(self, monkeypatch, caplog):
        monkeypatch.setenv(
            "GBRAIN_WRITE_PREFIXES_ALLOWED",
            "kb/,kb/initiatives/../services/,a\\b/c/,a//b/,./x/,ok/fine/",
        )
        with caplog.at_level(logging.WARNING):
            allowed = Settings(_env_file=None).gbrain_write_prefixes_allowed
        assert allowed == ["ok/fine/"]
        assert "too broad" in caplog.text

    @pytest.mark.parametrize("raw,expected", [
        ("kb/initiatives", "kb/initiatives/"),
        ("/kb/initiatives/", "kb/initiatives/"),
        ("kb/", None),
        ("kb", None),
        ("", None),
        ("  ", None),
        ("a/../b/", None),
        ("a/./b/", None),
        ("a\\b/c", None),
        (None, None),
    ])
    def test_normalize(self, raw, expected):
        assert normalize_brain_write_prefix(raw) == expected


# ---------------------------------------------------------------------------
# Grant

class TestGrantWritePrefixes:
    def test_round_trip_with_prefixes(self):
        token = mint_brain_grant(
            BrainGrant(run_id="r", access="write", write_prefixes=[PREFIX]), KEY,
        )
        grant = verify_brain_grant(token, KEY)
        assert grant is not None
        assert grant.write_prefixes == [PREFIX]
        assert grant.version == 1

    def test_round_trip_without_prefixes(self):
        token = mint_brain_grant(BrainGrant(run_id="r", access="write"), KEY)
        grant = verify_brain_grant(token, KEY)
        assert grant is not None and grant.write_prefixes == []

    def test_old_token_shape_still_validates(self):
        import time

        old_payload = {
            "version": 1, "run_id": "r", "agent_id": "a", "workflow_id": "w",
            "access": "write", "expires_at": int(time.time()) + 600,
        }
        grant = verify_brain_grant(sign_payload(GRANT_PREFIX, old_payload, KEY), KEY)
        assert grant is not None
        assert grant.access == "write"
        assert grant.write_prefixes == []


# ---------------------------------------------------------------------------
# Mint: workflow prefixes intersected with the allowlist

def _minted(cfg: dict) -> BrainGrant:
    entry = _brain_entry(cfg)
    assert entry is not None
    grant = verify_brain_grant(entry["api_key"], KEY)
    assert grant is not None
    return grant


class TestMintIntersection:
    def test_only_allowed_prefixes_reach_the_grant(self, caplog):
        settings = _settings(GBRAIN_WRITE_PREFIXES_ALLOWED="kb/initiatives/,x/y/")
        with caplog.at_level(logging.WARNING):
            cfg = _build_agent_config(
                _agent("write"), settings, run_id="r", workflow_id="w",
                workflow_brain_access="write",
                workflow_brain_write_prefixes=[
                    "/kb/initiatives", "kb/services/", "kb/",
                ],
            )
        grant = _minted(cfg)
        assert grant.write_prefixes == [PREFIX]
        assert "kb/services/" in caplog.text
        assert f"`{PREFIX}`" in cfg["system_prompt"]
        assert "exactly the slug" in cfg["system_prompt"]

    def test_empty_allowlist_grants_nothing_extra(self):
        cfg = _build_agent_config(
            _agent("write"), _settings(), run_id="r", workflow_id="w",
            workflow_brain_access="write", workflow_brain_write_prefixes=[PREFIX],
        )
        assert _minted(cfg).write_prefixes == []
        assert PREFIX not in cfg["system_prompt"]

    def test_read_access_never_carries_prefixes(self):
        settings = _settings(GBRAIN_WRITE_PREFIXES_ALLOWED=PREFIX)
        cfg = _build_agent_config(
            _agent("write"), settings, run_id="r", workflow_id="w",
            workflow_brain_access="read", workflow_brain_write_prefixes=[PREFIX],
        )
        grant = _minted(cfg)
        assert grant.access == "read"
        assert grant.write_prefixes == []
        assert PREFIX not in cfg["system_prompt"]


# ---------------------------------------------------------------------------
# brain_put_page

@pytest.fixture
def brain(monkeypatch):
    container = _Container()
    mcp = build_brain_mcp(lambda: container, write_namespace="carrier")
    allow = {"value": [PREFIX]}
    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: MagicMock(gbrain_write_prefixes_allowed=allow["value"]),
    )
    return mcp, container.brain_client, allow


def _write_grant(prefixes: list[str] | None = None, access: str = "write") -> BrainGrant:
    return BrainGrant(
        run_id="run-1", agent_id="architect", workflow_id="wf",
        access=access, write_prefixes=prefixes if prefixes is not None else [PREFIX],
    )


async def _put(mcp, slug: str):
    return await mcp.call_tool("brain_put_page", {"slug": slug, "content": "c", "reason": "r"})


class TestPutPageWithPrefixes:
    async def test_allowed_prefix_slug_is_kept(self, brain):
        mcp, client, _ = brain
        set_current_brain_grant(_write_grant())
        await _put(mcp, "kb/initiatives/foo/design/solution")
        assert client.calls[-1][1]["slug"] == "kb/initiatives/foo/design/solution"

    async def test_non_matching_slug_is_namespaced(self, brain):
        mcp, client, _ = brain
        set_current_brain_grant(_write_grant())
        await _put(mcp, "kb/services/x")
        assert client.calls[-1][1]["slug"] == "carrier/kb/services/x"

    async def test_bare_prefix_is_not_a_match(self, brain):
        mcp, client, _ = brain
        set_current_brain_grant(_write_grant())
        await _put(mcp, "kb/initiatives")
        assert client.calls[-1][1]["slug"] == "carrier/kb/initiatives"

    @pytest.mark.parametrize("bad", [
        "kb/initiatives/../services/x",
        "/kb/initiatives/x",
        "kb/initiatives//x",
        "kb/initiatives/a\\b",
        "kb/initiatives/./x",
    ])
    async def test_traversal_is_rejected(self, brain, bad):
        mcp, client, _ = brain
        set_current_brain_grant(_write_grant())
        with pytest.raises(ToolError):
            await _put(mcp, bad)
        assert client.calls == []

    async def test_refused_when_allowlist_no_longer_contains_prefix(self, brain):
        mcp, client, allow = brain
        allow["value"] = []
        set_current_brain_grant(_write_grant())
        with pytest.raises(ToolError, match="no longer allowed"):
            await _put(mcp, "kb/initiatives/foo/design")
        assert client.calls == []
        # Slugs outside the stale prefix still go to the namespace as before.
        await _put(mcp, "notes/x")
        assert client.calls[-1][1]["slug"] == "carrier/notes/x"

    async def test_grant_without_prefixes_namespaces_everything(self, brain):
        mcp, client, _ = brain
        set_current_brain_grant(_write_grant(prefixes=[]))
        await _put(mcp, "kb/initiatives/foo")
        assert client.calls[-1][1]["slug"] == "carrier/kb/initiatives/foo"

    async def test_read_only_grant_still_cannot_write(self, brain):
        mcp, client, _ = brain
        set_current_brain_grant(_write_grant(access="read"))
        with pytest.raises(ToolError, match="not granted"):
            await _put(mcp, "kb/initiatives/foo")
        assert client.calls == []

    def test_resolve_write_slug(self):
        assert resolve_write_slug("kb/initiatives/x", "carrier", [PREFIX], [PREFIX]) == (
            "kb/initiatives/x", PREFIX,
        )
        assert resolve_write_slug("kb/initiatives/x", "carrier", [PREFIX], []) == (
            "carrier/kb/initiatives/x", None,
        )
        assert resolve_write_slug("kb/initiativesx/y", "carrier", [PREFIX], [PREFIX]) == (
            "carrier/kb/initiativesx/y", None,
        )


# ---------------------------------------------------------------------------
# Workflow definition: persistence, runner, management tools

class TestWorkflowDefinition:
    async def test_local_files_round_trip(self, tmp_path):
        from app.domain.models.workflow_definition import WorkflowDefinition
        from app.infrastructure.persistence.workflow_backend import LocalFilesWorkflowBackend

        backend = LocalFilesWorkflowBackend(str(tmp_path))
        steps = [{"id": "s1", "type": "llm", "output_key": "a", "user_template": "{request}"}]
        await backend.create(WorkflowDefinition(
            id="wf", steps=steps, brain_access="write", brain_write_prefixes=[PREFIX],
        ))
        assert (await backend.get("wf")).brain_write_prefixes == [PREFIX]

        await backend.create(WorkflowDefinition(id="plain", steps=steps))
        assert "brain_write_prefixes" not in (tmp_path / "plain.yaml").read_text()
        assert (await backend.get("plain")).brain_write_prefixes == []

    def test_mongo_doc_round_trip(self):
        from app.domain.models.workflow_definition import WorkflowDefinition
        from app.infrastructure.persistence.workflow_backend import MongoWorkflowBackend

        defn = WorkflowDefinition(id="wf", brain_write_prefixes=[PREFIX])
        doc = MongoWorkflowBackend._to_doc(defn)
        assert MongoWorkflowBackend._from_doc(doc).brain_write_prefixes == [PREFIX]
        legacy = {k: v for k, v in doc.items() if k != "brain_write_prefixes"}
        assert MongoWorkflowBackend._from_doc(legacy).brain_write_prefixes == []

    def test_carried_to_the_runner(self):
        from app.domain.models.workflow_definition import WorkflowDefinition
        from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner

        steps = [{"id": "s1", "type": "llm", "output_key": "a", "user_template": "{request}"}]
        defn = WorkflowDefinition(id="wf", steps=steps, brain_write_prefixes=[PREFIX])
        assert defn.to_raw_dict()["brain_write_prefixes"] == [PREFIX]
        runner = YamlGraphRunner(defn.to_raw_dict(), MagicMock(), MagicMock())
        assert runner._brain_write_prefixes == [PREFIX]
        legacy = YamlGraphRunner({"id": "old", "steps": steps}, MagicMock(), MagicMock())
        assert legacy._brain_write_prefixes == []


class _MemWorkflowBackend:
    def __init__(self) -> None:
        self.items: dict = {}

    async def get(self, workflow_id):
        return self.items.get(workflow_id)

    async def list(self):
        return list(self.items.values())

    async def create(self, defn):
        self.items[defn.id] = defn
        return defn

    async def update(self, workflow_id, defn):
        self.items[workflow_id] = defn
        return defn


class TestManagementTools:
    def _deps(self):
        deps = MagicMock()
        deps.workflow_backend = _MemWorkflowBackend()
        deps.refresh_runner = None
        deps.script_backend = None
        deps.data_source_backend = None
        return deps

    async def test_create_and_update(self):
        from app.application import management_tools as core

        deps = self._deps()
        out = await core.create_workflow(
            deps, "wf", "WF", "", "[]", brain_access="write",
            brain_write_prefixes="kb/initiatives, team/docs/",
        )
        assert "created" in out, out
        stored = deps.workflow_backend.items["wf"]
        assert stored.brain_write_prefixes == [PREFIX, "team/docs/"]
        assert "Brain write prefixes: kb/initiatives/, team/docs/" in (
            await core.get_workflow(deps, "wf", include_steps=False)
        )

        # Omitted keeps, a list replaces, an empty value clears.
        await core.update_workflow(deps, "wf", name="WF2")
        assert deps.workflow_backend.items["wf"].brain_write_prefixes == [PREFIX, "team/docs/"]
        await core.update_workflow(deps, "wf", brain_write_prefixes=["x/y"])
        assert deps.workflow_backend.items["wf"].brain_write_prefixes == ["x/y/"]
        await core.update_workflow(deps, "wf", brain_write_prefixes="")
        assert deps.workflow_backend.items["wf"].brain_write_prefixes == []

    async def test_unsafe_prefix_rejected(self):
        from app.application import management_tools as core

        deps = self._deps()
        out = await core.create_workflow(
            deps, "wf", "WF", "", "[]", brain_write_prefixes=["kb/"],
        )
        assert "Invalid brain_write_prefixes" in out
        assert "wf" not in deps.workflow_backend.items
        await core.create_workflow(deps, "wf", "WF", "", "[]")
        out = await core.update_workflow(
            deps, "wf", brain_write_prefixes="kb/initiatives/../x/",
        )
        assert "Invalid brain_write_prefixes" in out
        assert deps.workflow_backend.items["wf"].brain_write_prefixes == []

    async def test_mcp_surface_exposes_the_param(self):
        from tests.test_management_mcp_tools import _Container as _MgmtContainer
        from tests.test_management_mcp_tools import _register
        from app.api.mcp.management_server import build_management_mcp

        mcp = build_management_mcp()
        _register(mcp, _MgmtContainer())
        schemas = {t.name: t.inputSchema for t in await mcp.list_tools()}
        assert "brain_write_prefixes" in schemas["create_workflow"]["properties"]
        assert "brain_write_prefixes" in schemas["update_workflow"]["properties"]


class TestRestRoutes:
    async def test_workflow_routes_carry_prefixes(self):
        from unittest.mock import AsyncMock

        from httpx import ASGITransport, AsyncClient

        from tests.test_brain_shared_memory import _app_with_container
        from tests.test_workflow_flag_preservation import _container

        container, backend = _container()
        stored = backend.get.return_value
        stored.brain_write_prefixes = [PREFIX]
        app = _app_with_container(container)
        steps = [{"id": "s1", "type": "llm", "output_key": "answer", "user_template": "{request}"}]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            resp = await c.put("/api/v1/workflows/wf", json={"name": "WF", "steps": steps})
            assert resp.status_code == 200, resp.text
            assert resp.json()["brain_write_prefixes"] == [PREFIX]
            resp = await c.put(
                "/api/v1/workflows/wf",
                json={"name": "WF", "steps": steps, "brain_write_prefixes": ["x/y"]},
            )
            assert resp.json()["brain_write_prefixes"] == ["x/y/"]
            bad = await c.put(
                "/api/v1/workflows/wf",
                json={"name": "WF", "steps": steps, "brain_write_prefixes": ["x"]},
            )
            assert bad.status_code == 422

            backend.get = AsyncMock(return_value=None)
            backend.create = AsyncMock(side_effect=lambda defn: defn)
            created = await c.post(
                "/api/v1/workflows",
                json={"id": "wf2", "name": "W2", "steps": steps,
                      "brain_write_prefixes": [PREFIX]},
            )
            assert created.status_code in (200, 201), created.text
            assert created.json()["brain_write_prefixes"] == [PREFIX]
