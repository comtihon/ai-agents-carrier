"""Unit tests for the per-visit agent task key (planner handoff: new task per
NEW agent execution, incl. workflow loop re-entry), and for the save_task
$setOnInsert/$set split that lets a re-entrant save update ``input`` without
clobbering accumulated ``outputs``."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_agent_def() -> MagicMock:
    agent_def = MagicMock()
    agent_def.agent_input = {"system_prompt": "Research.", "model": "claude-sonnet-4-5"}
    agent_def.runtime = "k8s"
    agent_def.default_runtime = "k8s"
    agent_def.mcp_addon = None
    agent_def.s3_addon = None
    return agent_def


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.get_mcp_integrations.return_value = []
    settings.anthropic_api_key = "sk-test"
    settings.openai_api_key = None
    settings.llm_provider = "anthropic"
    settings.llm_base_url = None
    settings.get_llm_integration.return_value = None
    settings.get_llm_integrations.return_value = []
    settings.get_forwardable_config.return_value = {}
    return settings


def _make_step() -> dict:
    return {"id": "loop_step", "type": "langgraph-agent", "agent_id": "researcher-fast"}


def _poll_settings() -> MagicMock:
    s = MagicMock()
    s.agent_poll_interval_seconds = 1
    s.agent_max_loops = 3
    s.anthropic_api_key = "sk-test"
    return s


def _finished_response(raw_output: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "status": "finished",
        "run_id": "run1",
        "outputs": [{"id": "x", "type": "final", "content": raw_output}],
    }
    return resp


@pytest.mark.asyncio
async def test_new_visit_count_produces_new_key_and_input():
    """Two executions of the same step_id with different _visit_counts must
    save distinct task docs (_0, _1 suffix) and send distinct /start inputs —
    proving loop re-entry spawns a fresh agent task rather than resuming."""
    from app.steps.agent_executor import execute_agent_step

    step = _make_step()
    fake_agent_def = _make_agent_def()
    settings = _make_settings()

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(return_value=_finished_response({"result": "ok"}))
    mock_http_client.post = AsyncMock(return_value=_finished_response({"result": "ok"}))

    fake_backend = AsyncMock()
    fake_backend.get = AsyncMock(return_value=fake_agent_def)

    task_repo = AsyncMock()
    task_repo.save_task = AsyncMock()
    task_repo.get_task = AsyncMock(return_value=None)
    task_repo.update_task = AsyncMock()
    task_repo.append_outputs = AsyncMock()

    with (
        patch("app.runtime.factory.get_runtime") as mock_get_runtime,
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("httpx.AsyncClient", return_value=mock_http_client),
        patch("app.steps.agent_executor.get_settings", return_value=_poll_settings()),
    ):
        mock_runtime = MagicMock()
        mock_runtime.has_container_for_run = AsyncMock(return_value=False)
        mock_runtime.terminate_by_run_id = AsyncMock()
        mock_runtime.rewrite_callback_url = MagicMock(return_value="http://localhost")
        mock_runtime.spawn = AsyncMock(return_value="http://agent-host:8080")
        mock_get_runtime.return_value = mock_runtime

        # First execution: fresh visit (count 0)
        await execute_agent_step(
            step=step,
            state={"request": "first pass", "_visit_counts": {}},
            agent_backend=fake_backend,
            run_id="run1",
            callback_base_url="http://localhost",
            settings=settings,
            run_repository=None,
            agent_task_repository=task_repo,
        )

        # Second execution: loop re-entry (visit count bumped to 1)
        await execute_agent_step(
            step=step,
            state={"request": "second pass", "_visit_counts": {"loop_step": 1}},
            agent_backend=fake_backend,
            run_id="run1",
            callback_base_url="http://localhost",
            settings=settings,
            run_repository=None,
            agent_task_repository=task_repo,
        )

    assert task_repo.save_task.call_count == 2
    first_doc = task_repo.save_task.call_args_list[0].args[0]
    second_doc = task_repo.save_task.call_args_list[1].args[0]
    assert first_doc["_id"] == "run1_loop_step_0"
    assert second_doc["_id"] == "run1_loop_step_1"

    first_start_body = mock_http_client.post.call_args_list[0].kwargs["json"]
    second_start_body = mock_http_client.post.call_args_list[1].kwargs["json"]
    assert first_start_body["input"] != second_start_body["input"]
    assert first_start_body["task_id"] == "run1_loop_step_0"
    assert second_start_body["task_id"] == "run1_loop_step_1"


@pytest.mark.asyncio
async def test_same_attempt_resume_reuses_key():
    """Same-attempt crash recovery: when the container already exists for the
    run and _visit_counts is unchanged, the resume path must look up the task
    using the SAME task key (no new task created)."""
    from app.steps.agent_executor import execute_agent_step

    step = _make_step()
    fake_agent_def = _make_agent_def()
    settings = _make_settings()

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(return_value=_finished_response({"result": "ok"}))
    mock_http_client.post = AsyncMock(return_value=_finished_response({"result": "ok"}))

    fake_backend = AsyncMock()
    fake_backend.get = AsyncMock(return_value=fake_agent_def)

    task_repo = AsyncMock()
    task_repo.get_task = AsyncMock(return_value={"agent_url": "http://agent-host:8080"})
    task_repo.save_task = AsyncMock()
    task_repo.update_task = AsyncMock()
    task_repo.append_outputs = AsyncMock()

    with (
        patch("app.runtime.factory.get_runtime") as mock_get_runtime,
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("httpx.AsyncClient", return_value=mock_http_client),
        patch("app.steps.agent_executor.get_settings", return_value=_poll_settings()),
    ):
        mock_runtime = MagicMock()
        mock_runtime.has_container_for_run = AsyncMock(return_value=True)
        mock_runtime.terminate_by_run_id = AsyncMock()
        mock_runtime.rewrite_callback_url = MagicMock(return_value="http://localhost")
        mock_runtime.get_agent_url_for_run = AsyncMock(return_value="http://agent-host:8080")
        mock_get_runtime.return_value = mock_runtime

        await execute_agent_step(
            step=step,
            state={"request": "first pass", "_visit_counts": {}},
            agent_backend=fake_backend,
            run_id="run1",
            callback_base_url="http://localhost",
            settings=settings,
            run_repository=None,
            agent_task_repository=task_repo,
        )

    task_repo.get_task.assert_called_once_with("run1_loop_step_0")


@pytest.mark.asyncio
async def test_save_task_updates_input_preserves_outputs():
    """save_task called twice with the same _id but different input must
    route ``input`` through $set (so it updates) and ``outputs`` through
    $setOnInsert (so accumulated outputs from append_outputs aren't reset)."""
    from app.infrastructure.persistence.mongo import MongoAgentTaskRepository

    fake_collection = MagicMock()
    fake_collection.update_one = AsyncMock()
    repo = MongoAgentTaskRepository(fake_collection)

    from datetime import datetime, timezone

    base_doc = {
        "_id": "run1_loop_step_0",
        "run_id": "run1",
        "step_id": "loop_step",
        "agent_id": "researcher-fast",
        "agent_url": "http://agent-host:8080",
        "agent_config": {},
        "status": "pending",
        "loop_count": 0,
        "max_loops": 20,
        "outputs": [],
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    await repo.save_task({**base_doc, "input": {"request": "first"}})
    await repo.save_task({**base_doc, "input": {"request": "second"}})

    assert fake_collection.update_one.call_count == 2
    for call in fake_collection.update_one.call_args_list:
        _filter, update = call.args
        assert "input" in update["$set"]
        assert "input" not in update.get("$setOnInsert", {})
        assert "outputs" in update["$setOnInsert"]
        assert "outputs" not in update["$set"]

    first_update = fake_collection.update_one.call_args_list[0].args[1]
    second_update = fake_collection.update_one.call_args_list[1].args[1]
    assert first_update["$set"]["input"] == {"request": "first"}
    assert second_update["$set"]["input"] == {"request": "second"}
