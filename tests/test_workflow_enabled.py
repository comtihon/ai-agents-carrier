"""A disabled workflow starts no runs, from any entry point.

Covers the enabled flag end to end: its backward-compatible default, its
round-trip through both persistence backends, and the shared guard
(``app.application.run_control.ensure_workflow_enabled``) at every surface that
can start a run — REST, webhook, Pub/Sub, cron, the MCP tool, a parent
workflow's ``workflow`` step, and replaying an existing run.
"""
from __future__ import annotations

import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.app import create_app
from app.application import management_tools, run_control
from app.application.run_control import (
    WORKFLOW_DISABLED_STATUS,
    WorkflowDisabledError,
    ensure_workflow_enabled,
)
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.config.graph_loader import (
    YamlGraphRegistry,
    build_registry_from_definitions,
    build_runner_from_definition,
)
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.persistence.workflow_backend import (
    LocalFilesWorkflowBackend,
    MongoWorkflowBackend,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider

_STEPS = [{"id": "step1", "type": "llm", "output_key": "answer", "user_template": "{request}"}]


def _defn(workflow_id: str = "wf-off", *, enabled: bool) -> WorkflowDefinition:
    return WorkflowDefinition(
        id=workflow_id, name="Off Workflow", steps=list(_STEPS), enabled=enabled
    )


def _mcp() -> MagicMock:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.start = AsyncMock()
    mcp.stop = AsyncMock()
    mcp.get_tool = MagicMock(return_value=None)
    return mcp


def _runner(definition: dict) -> YamlGraphRunner:
    return YamlGraphRunner(
        definition,
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="done")]),
        mcp_tools_provider=_mcp(),
    )


def _container(
    registry: YamlGraphRegistry, workflow_backend=None
) -> ApplicationContainer:
    repo = AsyncMock(spec=MongoGraphRunRepository)
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    mongo_provider = MagicMock()
    mongo_provider.close = AsyncMock()
    return ApplicationContainer(
        settings=Settings(),
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=_mcp(),
        yaml_graph_registry=registry,
        mongo_provider=mongo_provider,
        run_repository=repo,
        openhands=MagicMock(spec=OpenHandsAdapter),
        workflow_backend=workflow_backend,
    )


def _backend_stub(definition: WorkflowDefinition | None) -> MagicMock:
    backend = MagicMock()
    backend.get = AsyncMock(return_value=definition)
    return backend


# ---------------------------------------------------------------------------
# Backward compatibility: no flag stored means enabled
# ---------------------------------------------------------------------------

class TestBackwardCompatibleDefault:
    def test_model_without_the_field_is_enabled(self) -> None:
        """A document written before the flag existed must keep running."""
        defn = WorkflowDefinition.model_validate(
            {"id": "legacy", "name": "Legacy", "steps": _STEPS}
        )
        assert defn.enabled is True

    def test_mongo_document_without_the_field_is_enabled(self) -> None:
        legacy_doc = {"_id": "legacy", "name": "Legacy", "steps": _STEPS}
        defn = MongoWorkflowBackend._from_doc(legacy_doc)
        assert defn.id == "legacy"
        assert defn.enabled is True

    @pytest.mark.asyncio
    async def test_yaml_file_without_the_field_is_enabled(self, tmp_path) -> None:
        (tmp_path / "legacy.yaml").write_text(textwrap.dedent("""\
            id: legacy
            name: Legacy
            steps:
              - id: s1
                type: llm
                output_key: out
        """))
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        defn = await backend.get("legacy")
        assert defn is not None
        assert defn.enabled is True

    def test_runner_built_from_a_definition_without_the_field_is_enabled(self) -> None:
        runner = _runner({"id": "legacy", "steps": _STEPS})
        assert runner.enabled is True

    def test_guard_treats_an_object_with_no_flag_as_enabled(self) -> None:
        ensure_workflow_enabled(object())  # must not raise


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistenceRoundTrip:
    @pytest.mark.asyncio
    async def test_local_files_backend_round_trips_disabled(self, tmp_path) -> None:
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        await backend.create(_defn("wf-a", enabled=False))

        reloaded = await backend.get("wf-a")
        assert reloaded is not None
        assert reloaded.enabled is False
        assert "enabled: false" in (tmp_path / "wf-a.yaml").read_text()

        reloaded.enabled = True
        await backend.update("wf-a", reloaded)
        again = await backend.get("wf-a")
        assert again is not None
        assert again.enabled is True

    @pytest.mark.asyncio
    async def test_local_files_backend_round_trips_through_list(self, tmp_path) -> None:
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        await backend.create(_defn("wf-on", enabled=True))
        await backend.create(_defn("wf-off", enabled=False))
        by_id = {d.id: d.enabled for d in await backend.list()}
        assert by_id == {"wf-on": True, "wf-off": False}

    def test_mongo_document_carries_the_flag_both_ways(self) -> None:
        doc = MongoWorkflowBackend._to_doc(_defn("wf-off", enabled=False))
        assert doc["enabled"] is False
        assert MongoWorkflowBackend._from_doc(doc).enabled is False


# ---------------------------------------------------------------------------
# The flag reaches the runner, the registry listing and the REST responses
# ---------------------------------------------------------------------------

class TestFlagIsVisible:
    def test_build_runner_from_definition_carries_the_flag(self) -> None:
        registry = YamlGraphRegistry({})
        runner = build_runner_from_definition(
            _defn(enabled=False),
            llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
            mcp_tools_provider=_mcp(),
            registry=registry,
        )
        assert runner.enabled is False

    def test_registry_listing_exposes_the_flag(self) -> None:
        registry = build_registry_from_definitions(
            [_defn("wf-on", enabled=True), _defn("wf-off", enabled=False)],
            llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
            mcp_tools_provider=_mcp(),
        )
        by_id = {d["id"]: d["enabled"] for d in registry.list_definitions()}
        assert by_id == {"wf-on": True, "wf-off": False}

    def test_mcp_list_workflows_marks_a_disabled_workflow(self) -> None:
        registry = build_registry_from_definitions(
            [_defn("wf-on", enabled=True), _defn("wf-off", enabled=False)],
            llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
            mcp_tools_provider=_mcp(),
        )
        deps = MagicMock()
        deps.registry = registry
        out = management_tools.list_workflows(deps)
        assert "**wf-off** (Off Workflow) [DISABLED]" in out
        assert "**wf-on** (Off Workflow):" in out  # the enabled one is unmarked


# ---------------------------------------------------------------------------
# Entry point: REST POST /workflows/runs (the manual start)
# ---------------------------------------------------------------------------

async def _rest_client(container: ApplicationContainer):
    app = create_app()
    app.state.container = container
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestManualStartIsRefused:
    @pytest.mark.asyncio
    async def test_disabled_definition_from_the_backend_is_409(self) -> None:
        container = _container(
            YamlGraphRegistry({}), workflow_backend=_backend_stub(_defn(enabled=False))
        )
        async with await _rest_client(container) as c:
            resp = await c.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "wf-off", "user_request": "go"},
            )
        assert resp.status_code == WORKFLOW_DISABLED_STATUS == 409
        assert "wf-off" in resp.json()["detail"]
        assert "disabled" in resp.json()["detail"]
        container.run_repository.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabled_registry_runner_is_409_without_a_backend(self) -> None:
        runner = _runner({"id": "wf-off", "steps": _STEPS, "enabled": False})
        container = _container(YamlGraphRegistry({"wf-off": runner}))
        async with await _rest_client(container) as c:
            resp = await c.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "wf-off", "user_request": "go"},
            )
        assert resp.status_code == 409
        container.run_repository.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_enabled_workflow_still_starts(self) -> None:
        runner = _runner({"id": "wf-on", "steps": _STEPS})
        container = _container(YamlGraphRegistry({"wf-on": runner}))
        async with await _rest_client(container) as c:
            resp = await c.post(
                "/api/v1/workflows/runs",
                json={"workflow_id": "wf-on", "user_request": "go"},
            )
        assert resp.status_code == 200
        container.run_repository.create.assert_awaited()


# ---------------------------------------------------------------------------
# Entry point: the trigger surfaces (webhook, Pub/Sub, cron)
# ---------------------------------------------------------------------------

class TestTriggerDispatchIsRefused:
    @pytest.mark.asyncio
    async def test_webhook_trigger_is_409(self) -> None:
        steps = [
            {"id": "trigger", "type": "http", "auth_mode": "bearer", "bearer_token": "t0k"},
            *_STEPS,
        ]
        defn = WorkflowDefinition(id="wf-off", name="Off", steps=steps, enabled=False)
        container = _container(YamlGraphRegistry({}), workflow_backend=_backend_stub(defn))
        async with await _rest_client(container) as c:
            resp = await c.post(
                "/api/v1/webhooks/wf-off",
                json={"request": "go"},
                headers={"Authorization": "Bearer t0k"},
            )
        assert resp.status_code == 409
        assert "disabled" in resp.json()["detail"]
        container.run_repository.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_webhook_trigger_refuses_before_it_leaks_state(self) -> None:
        """A bad token still gets 403 — the flag is no unauthenticated oracle."""
        steps = [
            {"id": "trigger", "type": "http", "auth_mode": "bearer", "bearer_token": "t0k"},
            *_STEPS,
        ]
        defn = WorkflowDefinition(id="wf-off", name="Off", steps=steps, enabled=False)
        container = _container(YamlGraphRegistry({}), workflow_backend=_backend_stub(defn))
        async with await _rest_client(container) as c:
            resp = await c.post(
                "/api/v1/webhooks/wf-off",
                json={"request": "go"},
                headers={"Authorization": "Bearer wrong"},
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_pubsub_trigger_starts_no_run(self) -> None:
        defn = _defn(enabled=False)
        container = _container(YamlGraphRegistry({}), workflow_backend=_backend_stub(defn))
        job = container._make_pubsub_job("wf-off", {"id": "trigger", "type": "pubsub"})
        await job({"hello": "world"}, {"message_id": "m1"})
        container.run_repository.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cron_trigger_starts_no_run(self) -> None:
        defn = _defn(enabled=False)
        container = _container(YamlGraphRegistry({}), workflow_backend=_backend_stub(defn))
        job = container._make_cron_job("wf-off", "scheduled at {now}")
        await job()
        container.run_repository.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# Entry point: the MCP run_workflow tool
# ---------------------------------------------------------------------------

class TestMcpRunWorkflowIsRefused:
    @pytest.mark.asyncio
    async def test_returns_a_plain_error_string(self) -> None:
        registry = build_registry_from_definitions(
            [_defn(enabled=False)],
            llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
            mcp_tools_provider=_mcp(),
        )
        deps = MagicMock()
        deps.registry = registry
        deps.run_repository = AsyncMock()
        stream = AsyncMock()

        out = await management_tools.run_workflow(deps, "wf-off", "go", stream_fn=stream)

        assert "wf-off" in out and "disabled" in out
        assert "__event__" not in out
        deps.run_repository.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# Entry point: a parent workflow's `workflow` step
# ---------------------------------------------------------------------------

class TestChildWorkflowSpawnIsRefused:
    @pytest.mark.asyncio
    async def test_parent_step_reports_the_refusal_and_spawns_nothing(self) -> None:
        child = _runner({"id": "wf-off", "steps": _STEPS, "enabled": False})
        parent = _runner({
            "id": "parent",
            "steps": [{"id": "spawn", "type": "workflow", "workflow_id": "wf-off"}],
        })
        parent._registry = YamlGraphRegistry({"wf-off": child, "parent": parent})
        parent._run_repository = AsyncMock()

        node = parent._workflow_node({"id": "spawn", "type": "workflow", "workflow_id": "wf-off"})
        result = await node({"request": "go"})

        assert "disabled" in result["spawn_result"]["error"]
        parent._run_repository.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# Replaying an existing run of a workflow that has since been disabled
# ---------------------------------------------------------------------------

class TestReplayOfAPreExistingRun:
    """retry / restart-from-step re-execute steps, so they are starts and are
    refused. approve / reject deliberately stay open: a run already parked at a
    human gate must remain closable, otherwise disabling a workflow would strand
    every in-flight approval with no way to finish or reject it."""

    def _container_with_disabled(self) -> ApplicationContainer:
        return _container(
            YamlGraphRegistry({}), workflow_backend=_backend_stub(_defn(enabled=False))
        )

    @pytest.mark.asyncio
    async def test_retry_is_refused(self) -> None:
        container = self._container_with_disabled()
        container.run_repository.get = AsyncMock(
            return_value=GraphRun(id="r1", graph_id="wf-off", user_request="go", status="failed")
        )
        with pytest.raises(WorkflowDisabledError) as err:
            await run_control.retry_run(container, "r1")
        assert err.value.status_code == 409
        assert "wf-off" in err.value.detail

    @pytest.mark.asyncio
    async def test_restart_from_step_is_refused(self) -> None:
        container = self._container_with_disabled()
        container.run_repository.get = AsyncMock(
            return_value=GraphRun(id="r1", graph_id="wf-off", user_request="go", status="failed")
        )
        with pytest.raises(WorkflowDisabledError):
            await run_control.restart_from_step(container, "r1", "step1")

    @pytest.mark.asyncio
    async def test_the_id_guard_allows_an_unknown_workflow(self) -> None:
        """Nothing to check means nothing to refuse — 404 stays 404's job."""
        container = _container(YamlGraphRegistry({}), workflow_backend=_backend_stub(None))
        await run_control.ensure_workflow_id_enabled(container, "nope")
