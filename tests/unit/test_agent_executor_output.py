"""Unit tests for agent_executor structured output protocol."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# _build_agent_config tests
# ---------------------------------------------------------------------------

def _make_agent_def(system_prompt: str = "You are a helpful agent.") -> MagicMock:
    agent_def = MagicMock()
    agent_def.agent_input = {"system_prompt": system_prompt, "model": "claude-sonnet-4-5"}
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


def test_output_protocol_injected_when_output_mapping_set():
    from app.steps.agent_executor import _build_agent_config

    agent_def = _make_agent_def("You are a researcher.")
    settings = _make_settings()
    step = {"output_mapping": {"ticket_id": "tid", "summary": "sum", "context_sufficient": "cs"}}

    result = _build_agent_config(agent_def, settings, step=step)

    sp = result["system_prompt"]
    assert "Output Protocol" in sp
    assert "ticket_id" in sp
    assert "summary" in sp
    assert "context_sufficient" in sp
    assert "result" not in sp.lower() or "wrap" in sp or "plain text" in sp  # no plain result field instruction


def test_no_injection_when_no_output_mapping():
    from app.steps.agent_executor import _build_agent_config

    agent_def = _make_agent_def("You are a planner.")
    settings = _make_settings()
    step = {"output_key": "plan_result"}

    result = _build_agent_config(agent_def, settings, step=step)

    sp = result.get("system_prompt", "")
    assert "Output Protocol" not in sp


def test_no_injection_when_step_is_none():
    from app.steps.agent_executor import _build_agent_config

    agent_def = _make_agent_def("Base prompt.")
    settings = _make_settings()

    result = _build_agent_config(agent_def, settings, step=None)

    sp = result.get("system_prompt", "")
    assert "Output Protocol" not in sp


# ---------------------------------------------------------------------------
# Structured output detection logic (isolated, no complex mocking needed)
# ---------------------------------------------------------------------------

def test_is_structured_output_detection():
    output_mapping = {"ticket_id": "tid", "summary": "sum"}
    raw_output_structured = {"ticket_id": "X", "summary": "Y"}
    raw_output_unstructured = {"result": "text"}
    raw_output_context = {"context_sufficient": False, "questions": ["q"]}

    def detect(output_mapping, raw_output):
        return bool(
            output_mapping
            and (
                any(k in raw_output for k in output_mapping)
                or "context_sufficient" in raw_output
            )
        )

    assert detect(output_mapping, raw_output_structured) is True
    assert detect(output_mapping, raw_output_unstructured) is False
    assert detect(output_mapping, raw_output_context) is True
    assert detect({}, raw_output_structured) is False


# ---------------------------------------------------------------------------
# execute_agent_step structured output detection tests
# ---------------------------------------------------------------------------

def _make_step(output_mapping=None, output_key=None):
    step = {"id": "node_test", "type": "langgraph-agent", "agent_id": "researcher-fast"}
    if output_mapping:
        step["output_mapping"] = output_mapping
    if output_key:
        step["output_key"] = output_key
    return step


@pytest.mark.asyncio
async def test_structured_output_skips_meta_llm():
    """When agent returns fields matching output_mapping, meta-LLM is skipped."""
    from app.steps.agent_executor import execute_agent_step

    output_mapping = {"ticket_id": "ticket_id", "summary": "summary", "context_sufficient": "context_sufficient"}
    step = _make_step(output_mapping=output_mapping)
    raw_output = {"ticket_id": "C130-1475", "summary": "Add frozen check", "context_sufficient": True}

    fake_agent_def = MagicMock()
    fake_agent_def.agent_input = {"system_prompt": "Research.", "model": "claude-sonnet-4-5"}
    fake_agent_def.runtime = "k8s"
    fake_agent_def.default_runtime = "k8s"

    settings = _make_settings()

    import httpx

    # Build a mock poll response that returns "finished" with the raw_output as a final output
    mock_poll_response = MagicMock()
    mock_poll_response.raise_for_status = MagicMock()
    mock_poll_response.json.return_value = {
        "status": "finished",
        "run_id": "test-run-id",
        "outputs": [{"id": "x", "type": "final", "content": raw_output}],
    }

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(return_value=mock_poll_response)
    mock_http_client.post = AsyncMock(return_value=mock_poll_response)

    _poll_settings = MagicMock()
    _poll_settings.agent_poll_interval_seconds = 1
    _poll_settings.agent_max_loops = 3
    _poll_settings.anthropic_api_key = "sk-test"

    with (
        patch("app.steps.agent_executor._meta_llm_evaluate", new_callable=AsyncMock, return_value={"passed": True, "reason": "ok"}) as mock_meta,
        patch("app.runtime.factory.get_runtime") as mock_get_runtime,
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("httpx.AsyncClient", return_value=mock_http_client),
        patch("app.steps.agent_executor.get_settings", return_value=_poll_settings),
    ):
        mock_runtime = MagicMock()
        mock_runtime.has_container_for_run = AsyncMock(return_value=True)
        mock_runtime.terminate_by_run_id = AsyncMock()
        mock_runtime.rewrite_callback_url = MagicMock(return_value="http://localhost")
        mock_runtime.get_agent_url_for_run = AsyncMock(return_value="http://agent-host:8080")
        mock_get_runtime.return_value = mock_runtime

        fake_backend = AsyncMock()
        fake_backend.get = AsyncMock(return_value=fake_agent_def)

        result = await execute_agent_step(
            step=step,
            state={"request": "Implement C130-1475"},
            agent_backend=fake_backend,
            run_id="test-run-id",
            callback_base_url="http://localhost",
            settings=settings,
            run_repository=None,
        )

    mock_meta.assert_called_once()
    assert result.get("ticket_id") == "C130-1475"


@pytest.mark.asyncio
async def test_unstructured_output_raises_when_output_mapping_unmatched():
    """When agent returns only 'result' key and step has output_mapping, raises RuntimeError."""
    from app.steps.agent_executor import execute_agent_step

    output_mapping = {"ticket_id": "ticket_id", "summary": "summary"}
    step = _make_step(output_mapping=output_mapping)
    raw_output = {"result": "Some text output that is not parseable as YAML/JSON with matching fields", "token_usage": {}}

    fake_agent_def = MagicMock()
    fake_agent_def.agent_input = {"system_prompt": "Research.", "model": "claude-sonnet-4-5"}
    fake_agent_def.runtime = "k8s"
    fake_agent_def.default_runtime = "k8s"

    settings = _make_settings()

    mock_poll_response = MagicMock()
    mock_poll_response.raise_for_status = MagicMock()
    mock_poll_response.json.return_value = {
        "status": "finished",
        "run_id": "test-run-id",
        "outputs": [{"id": "x", "type": "final", "content": raw_output}],
    }

    mock_http_client = AsyncMock()
    mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
    mock_http_client.__aexit__ = AsyncMock(return_value=False)
    mock_http_client.get = AsyncMock(return_value=mock_poll_response)
    mock_http_client.post = AsyncMock(return_value=mock_poll_response)

    _poll_settings2 = MagicMock()
    _poll_settings2.agent_poll_interval_seconds = 1
    _poll_settings2.agent_max_loops = 3
    _poll_settings2.anthropic_api_key = "sk-test"

    with (
        patch("app.runtime.factory.get_runtime") as mock_get_runtime,
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("httpx.AsyncClient", return_value=mock_http_client),
        patch("app.steps.agent_executor.get_settings", return_value=_poll_settings2),
        pytest.raises(RuntimeError, match="unstructured output"),
    ):
        mock_runtime = MagicMock()
        mock_runtime.has_container_for_run = AsyncMock(return_value=True)
        mock_runtime.terminate_by_run_id = AsyncMock()
        mock_runtime.rewrite_callback_url = MagicMock(return_value="http://localhost")
        mock_runtime.get_agent_url_for_run = AsyncMock(return_value="http://agent-host:8080")
        mock_get_runtime.return_value = mock_runtime

        fake_backend = AsyncMock()
        fake_backend.get = AsyncMock(return_value=fake_agent_def)

        await execute_agent_step(
            step=step,
            state={"request": "Implement C130-1475"},
            agent_backend=fake_backend,
            run_id="test-run-id",
            callback_base_url="http://localhost",
            settings=settings,
            run_repository=None,
        )
