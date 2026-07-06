"""Unit tests for the ``previous_task`` context block injected into
``input_data`` on re-entrant agent executions (planner handoff), and for
the ``_build_previous_task`` / ``_truncate_for_prompt`` helpers it relies
on. Follows the AsyncMock style of ``test_agent_task_key.py``."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.steps.agent_executor import (
    _PREVIOUS_TASK_GUIDANCE,
    _build_previous_task,
    _truncate_for_prompt,
)


def _make_agent_def() -> MagicMock:
    agent_def = MagicMock()
    agent_def.agent_input = {"system_prompt": "Plan.", "model": "claude-sonnet-4-5"}
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
    return {"id": "planner_step", "type": "langgraph-agent", "agent_id": "planner"}


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


# --- _truncate_for_prompt --------------------------------------------------


def test_truncate_for_prompt_leaves_short_string_untouched():
    assert _truncate_for_prompt("short") == "short"


def test_truncate_for_prompt_caps_long_string_with_marker():
    long_str = "x" * 5000
    result = _truncate_for_prompt(long_str, cap=4000)
    assert result.endswith("…[truncated]")
    assert result == "x" * 4000 + "…[truncated]"


def test_truncate_for_prompt_walks_dict_string_values():
    value = {"short": "ok", "long": "y" * 5000, "num": 42}
    result = _truncate_for_prompt(value, cap=4000)
    assert result["short"] == "ok"
    assert result["long"].endswith("…[truncated]")
    assert result["num"] == 42


# --- _build_previous_task ---------------------------------------------------


@pytest.mark.asyncio
async def test_build_previous_task_absent_when_prior_doc_missing():
    repo = AsyncMock()
    repo.get_task = AsyncMock(return_value=None)
    result = await _build_previous_task(repo, "run1", "step1", 1)
    assert result is None
    repo.get_task.assert_called_once_with("run1_step1_0")


@pytest.mark.asyncio
async def test_build_previous_task_absent_when_prior_unfinished():
    repo = AsyncMock()
    repo.get_task = AsyncMock(return_value={"status": "pending", "outputs": []})
    result = await _build_previous_task(repo, "run1", "step1", 1)
    assert result is None


@pytest.mark.asyncio
async def test_build_previous_task_absent_on_first_visit():
    repo = AsyncMock()
    repo.get_task = AsyncMock()
    result = await _build_previous_task(repo, "run1", "step1", 0)
    assert result is None
    repo.get_task.assert_not_called()


@pytest.mark.asyncio
async def test_build_previous_task_output_field_truncated_with_marker():
    repo = AsyncMock()
    long_output = "z" * 5000
    repo.get_task = AsyncMock(
        return_value={
            "status": "finished",
            "input": {"request": "first pass"},
            "outputs": [{"type": "final", "content": long_output}],
        }
    )
    result = await _build_previous_task(repo, "run1", "step1", 1)
    assert result is not None
    assert result["output"].endswith("…[truncated]")
    assert len(result["output"]) == 4000 + len("…[truncated]")


@pytest.mark.asyncio
async def test_build_previous_task_guidance_present():
    repo = AsyncMock()
    repo.get_task = AsyncMock(
        return_value={
            "status": "finished",
            "input": {"request": "first pass"},
            "outputs": [{"type": "final", "content": "done"}],
        }
    )
    result = await _build_previous_task(repo, "run1", "step1", 1)
    assert result is not None
    assert result["guidance"] == _PREVIOUS_TASK_GUIDANCE


# --- execute_agent_step wiring ----------------------------------------------


async def _run_execute_agent_step(state: dict, task_repo) -> dict:
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

        await execute_agent_step(
            step=step,
            state=state,
            agent_backend=fake_backend,
            run_id="run1",
            callback_base_url="http://localhost",
            settings=settings,
            run_repository=None,
            agent_task_repository=task_repo,
        )

    return mock_http_client.post.call_args_list[0].kwargs["json"]["input"]


@pytest.mark.asyncio
async def test_previous_task_injected_when_visit_positive():
    """visit_count=1 with a finished prior task doc → previous_task is added
    to the outgoing /start input, built from the prior visit's input/output."""
    prior_doc = {
        "status": "finished",
        "input": {"request": "first pass"},
        "outputs": [{"type": "final", "content": {"plan": "step 1"}}],
    }

    async def _get_task(key):
        return prior_doc if key == "run1_planner_step_0" else None

    task_repo = AsyncMock()
    task_repo.save_task = AsyncMock()
    task_repo.get_task = AsyncMock(side_effect=_get_task)
    task_repo.update_task = AsyncMock()
    task_repo.append_outputs = AsyncMock()

    sent_input = await _run_execute_agent_step(
        {"request": "second pass", "_visit_counts": {"planner_step": 1}}, task_repo
    )

    assert "previous_task" in sent_input
    assert sent_input["previous_task"]["input"] == {"request": "first pass"}
    assert sent_input["previous_task"]["output"] == {"plan": "step 1"}
    assert sent_input["previous_task"]["guidance"] == _PREVIOUS_TASK_GUIDANCE


@pytest.mark.asyncio
async def test_previous_task_absent_on_first_visit():
    """visit_count=0 → no prior visit to look up, previous_task must be
    absent and get_task must not be called for the previous_task lookup."""
    task_repo = AsyncMock()
    task_repo.save_task = AsyncMock()
    task_repo.get_task = AsyncMock(return_value=None)
    task_repo.update_task = AsyncMock()
    task_repo.append_outputs = AsyncMock()

    sent_input = await _run_execute_agent_step(
        {"request": "first pass", "_visit_counts": {}}, task_repo
    )

    assert "previous_task" not in sent_input
    task_repo.get_task.assert_not_called()


@pytest.mark.asyncio
async def test_previous_task_absent_when_prior_unfinished():
    """visit_count=1 but prior task doc is not ``finished`` → previous_task
    is omitted rather than injecting a partial/incorrect context block."""
    async def _get_task(key):
        return {"status": "running", "input": {}, "outputs": []}

    task_repo = AsyncMock()
    task_repo.save_task = AsyncMock()
    task_repo.get_task = AsyncMock(side_effect=_get_task)
    task_repo.update_task = AsyncMock()
    task_repo.append_outputs = AsyncMock()

    sent_input = await _run_execute_agent_step(
        {"request": "second pass", "_visit_counts": {"planner_step": 1}}, task_repo
    )

    assert "previous_task" not in sent_input


@pytest.mark.asyncio
async def test_previous_task_absent_when_prior_doc_missing():
    """visit_count=1 but no doc exists for the prior visit key → previous_task
    is omitted (acceptable no-op, e.g. prior visit key carried a clarification
    discriminator that this plain-key lookup doesn't reconstruct)."""
    task_repo = AsyncMock()
    task_repo.save_task = AsyncMock()
    task_repo.get_task = AsyncMock(return_value=None)
    task_repo.update_task = AsyncMock()
    task_repo.append_outputs = AsyncMock()

    sent_input = await _run_execute_agent_step(
        {"request": "second pass", "_visit_counts": {"planner_step": 1}}, task_repo
    )

    assert "previous_task" not in sent_input
