"""Pub/Sub-triggered runs: ack on save, one run per message, recoverable.

Prod incident (agentic-mf): the trigger callback awaited the whole run, so the
message stayed unacked for the run's duration. A carrier restart redelivered
both open messages, and each started a second run next to the one startup
recovery resumed. The callback now returns once the run is saved (the
subscriber acks then), a redelivered message is refused by a unique index, and
the saved run carries its trigger so recovery can start it over.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pymongo.errors import DuplicateKeyError

import app.core.container as container_module
from app.core.config import Settings
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.config.graph_loader import YamlGraphRegistry
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import DuplicateTriggerError, MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider

_STEPS = [
    {"id": "trigger", "type": "pubsub", "output_key": "event"},
    {"id": "step1", "type": "llm", "output_key": "answer", "user_template": "{request}"},
]
_META = {"topic": "service.backoffice.mf_agentic_requested", "message_id": "m-1", "attributes": {}}
_PAYLOAD = {"projectId": "147498", "requestedAt": "2026-11-26T10:58:04Z"}


def _mcp() -> MagicMock:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    return mcp


def _container(repo=None) -> ApplicationContainer:
    if repo is None:
        repo = AsyncMock(spec=MongoGraphRunRepository)
        repo.create = AsyncMock()
        repo.update = AsyncMock()
    backend = MagicMock()
    backend.get = AsyncMock(return_value=WorkflowDefinition(id="wf", name="wf", steps=list(_STEPS)))
    return ApplicationContainer(
        settings=Settings(mcp_datasources_enabled=False),
        llm=FakeMessagesListChatModel(responses=[AIMessage(content="x")]),
        mcp_tools_provider=_mcp(),
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=MagicMock(),
        run_repository=repo,
        openhands=MagicMock(spec=OpenHandsAdapter),
        workflow_backend=backend,
    )


@pytest.fixture
def blocked_stream(monkeypatch):
    """Replace the graph execution with one that runs until released."""
    release = asyncio.Event()
    started: list[GraphRun] = []

    async def fake_stream(runner, run, repo, initial_state, base_url=None):
        started.append(run)
        await release.wait()
        run.status = "completed"

    monkeypatch.setattr(container_module, "stream_graph_to_pause", fake_stream)
    return release, started


@pytest.mark.asyncio
async def test_callback_returns_once_the_run_is_saved(blocked_stream) -> None:
    release, started = blocked_stream
    container = _container()
    job = container._make_pubsub_job("wf", {"id": "trigger", "type": "pubsub"})

    # Returns although the run is still executing: the subscriber acks now.
    await asyncio.wait_for(job(_PAYLOAD, _META), timeout=1)

    container.run_repository.create.assert_awaited_once()
    saved: GraphRun = container.run_repository.create.await_args.args[0]
    assert saved.trigger_message_id == "service.backoffice.mf_agentic_requested/m-1"
    assert saved.state["trigger_payload"] == _PAYLOAD
    assert saved.state["trigger_info"]["message_id"] == "m-1"
    assert saved.step_statuses == {"trigger": "pending", "step1": "pending"}

    await asyncio.sleep(0)
    assert [r.id for r in started] == [saved.id]
    assert len(container.background_runs) == 1
    assert saved.id in container.live_runners

    release.set()
    await asyncio.gather(*container.background_runs)
    assert not container.background_runs
    assert saved.id not in container.live_runners


@pytest.mark.asyncio
async def test_redelivered_message_starts_no_second_run(blocked_stream) -> None:
    _, started = blocked_stream
    container = _container()
    container.run_repository.create = AsyncMock(
        side_effect=DuplicateTriggerError("wf", "service.backoffice.mf_agentic_requested/m-1")
    )
    job = container._make_pubsub_job("wf", {"id": "trigger", "type": "pubsub"})

    await job(_PAYLOAD, _META)  # returns normally -> the message is acked

    await asyncio.sleep(0)
    assert started == []
    assert not container.background_runs
    assert not container.live_runners


@pytest.mark.asyncio
async def test_failing_save_propagates_so_the_message_is_redelivered(blocked_stream) -> None:
    _, started = blocked_stream
    container = _container()
    container.run_repository.create = AsyncMock(side_effect=RuntimeError("mongo down"))
    job = container._make_pubsub_job("wf", {"id": "trigger", "type": "pubsub"})

    with pytest.raises(RuntimeError):
        await job(_PAYLOAD, _META)
    assert started == []
    assert not container.live_runners


@pytest.mark.asyncio
async def test_message_without_id_is_not_deduplicated(blocked_stream) -> None:
    release, _ = blocked_stream
    container = _container()
    job = container._make_pubsub_job("wf", {"id": "trigger", "type": "pubsub"})

    await job(_PAYLOAD, {"topic": "t"})

    saved: GraphRun = container.run_repository.create.await_args.args[0]
    assert saved.trigger_message_id is None
    release.set()
    await asyncio.gather(*container.background_runs)


@pytest.mark.asyncio
async def test_mongo_create_maps_the_unique_index_to_duplicate_trigger() -> None:
    collection = MagicMock()
    collection.insert_one = AsyncMock(side_effect=DuplicateKeyError("E11000 uniq_graph_trigger_message"))
    repo = MongoGraphRunRepository(collection)

    run = GraphRun(id="r2", graph_id="wf", status="running", trigger_message_id="t/m-1")
    with pytest.raises(DuplicateTriggerError):
        await repo.create(run)

    # A run without a trigger id keeps the original error.
    plain = GraphRun(id="r3", graph_id="wf", status="running")
    with pytest.raises(DuplicateKeyError):
        await repo.create(plain)


@pytest.mark.asyncio
async def test_unique_index_covers_only_runs_with_a_trigger_id() -> None:
    collection = MagicMock()
    collection.create_index = AsyncMock()
    await MongoGraphRunRepository(collection).ensure_indexes()

    calls = [c for c in collection.create_index.await_args_list if c.kwargs.get("name") == "uniq_graph_trigger_message"]
    assert len(calls) == 1
    assert calls[0].args[0] == [("graph_id", 1), ("trigger_message_id", 1)]
    assert calls[0].kwargs["unique"] is True
    assert calls[0].kwargs["partialFilterExpression"] == {"trigger_message_id": {"$type": "string"}}


@pytest.mark.asyncio
async def test_recovery_restarts_an_unstarted_run_with_its_trigger() -> None:
    """Acked, saved, then the pod died before the first step finished."""
    container = _container()
    runner = MagicMock()
    runner.steps = list(_STEPS)
    container._build_runner_for_recovery = MagicMock(return_value=runner)
    container._resume_run = AsyncMock()

    run = GraphRun(
        id="r1", graph_id="wf", status="running", user_request="{}",
        state={"trigger_payload": _PAYLOAD, "trigger_info": dict(_META)},
        step_statuses={"trigger": "pending", "step1": "pending"},
        trigger_message_id="t/m-1",
    )
    await container._recover_run(run)
    await asyncio.sleep(0)

    container._resume_run.assert_awaited_once()
    resume_input = container._resume_run.await_args.args[2]
    assert resume_input["trigger_payload"] == _PAYLOAD
    assert resume_input["trigger_info"]["message_id"] == "m-1"
    assert resume_input["request"] == "{}"
