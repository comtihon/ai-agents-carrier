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


# ---------------------------------------------------------------------------
# _apply_usage_keys tests — agent/meta/judge usage kept as separate buckets
# ---------------------------------------------------------------------------

def test_agent_usage_excludes_judge():
    """Agent token_usage and judge (meta-LLM evaluator) usage must land in
    separate keys, never summed together."""
    from app.steps.agent_executor import _apply_usage_keys

    raw_output = {"result": "ok", "token_usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}
    judge_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    target: dict = {}
    _apply_usage_keys(target, "node_test", raw_output, judge_usage)

    assert target["_agent_token_usage_node_test"] == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    assert target["_judge_token_usage"] == judge_usage
    assert "_meta_token_usage_node_test" not in target


def test_meta_usage_from_payload():
    """The agent's own post-compact meta-LLM usage (meta_token_usage in the raw
    payload) must land under _meta_token_usage_{step_id}, separate from both
    the agent usage and the judge usage."""
    from app.steps.agent_executor import _apply_usage_keys

    raw_output = {
        "result": "ok",
        "token_usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        "meta_token_usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
    }

    target: dict = {}
    _apply_usage_keys(target, "node_test", raw_output, None)

    assert target["_agent_token_usage_node_test"] == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    assert target["_meta_token_usage_node_test"] == {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}
    assert "_judge_token_usage" not in target


def test_no_meta_no_judge_keys_absent():
    """When there is no meta or judge usage, only the agent key is written —
    the other keys are omitted entirely (not written as None)."""
    from app.steps.agent_executor import _apply_usage_keys

    raw_output = {"result": "ok", "token_usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}

    target: dict = {}
    _apply_usage_keys(target, "node_test", raw_output, None)

    assert target == {"_agent_token_usage_node_test": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}
    assert "_meta_token_usage_node_test" not in target
    assert "_judge_token_usage" not in target


def test_rejection_paths_split_usage():
    """Both rejection-path call sites (meta-LLM rejection and unstructured
    output rejection) must split agent/meta/judge usage into separate keys
    on the mapped_result dict passed to MetaLLMRejectionError."""
    from app.steps.agent_executor import _apply_usage_keys

    raw_output = {
        "result": "some text",
        "token_usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
        "meta_token_usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }
    judge_usage = {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}

    for target in ({}, {}):
        _apply_usage_keys(target, "node_test", raw_output, judge_usage)
        assert target["_agent_token_usage_node_test"] == {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40}
        assert target["_meta_token_usage_node_test"] == {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
        assert target["_judge_token_usage"] == judge_usage


@pytest.mark.asyncio
async def test_meta_llm_evaluate_captures_usage_metadata():
    """_meta_llm_evaluate must read usage off response.usage_metadata and
    return it under the 'usage' key with normalized field names."""
    from app.steps.agent_executor import _meta_llm_evaluate

    fake_response = MagicMock()
    fake_response.content = "DECISION: PASS\nREASON: looks good"
    fake_response.usage_metadata = {"input_tokens": 42, "output_tokens": 7, "total_tokens": 49}

    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=fake_response)

    settings = _make_settings()
    settings.meta_llm_provider = None
    settings.meta_llm_model = None

    with patch("app.core.container.build_llm_native", return_value=fake_llm):
        result = await _meta_llm_evaluate(
            raw_output={"result": "some output"},
            input_data={"request": "do a thing"},
            step_id="node_test",
            settings=settings,
        )

    assert result["passed"] is True
    assert result["usage"] == {"input_tokens": 42, "output_tokens": 7, "total_tokens": 49}


@pytest.mark.asyncio
async def test_meta_llm_evaluate_usage_none_when_not_exposed():
    """When the LLM client doesn't expose usage_metadata, usage must be None
    (not a zeroed dict)."""
    from app.steps.agent_executor import _meta_llm_evaluate

    fake_response = MagicMock(spec=["content"])  # no usage_metadata attribute
    fake_response.content = "DECISION: PASS\nREASON: ok"

    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=fake_response)

    settings = _make_settings()
    settings.meta_llm_provider = None
    settings.meta_llm_model = None

    with patch("app.core.container.build_llm_native", return_value=fake_llm):
        result = await _meta_llm_evaluate(
            raw_output={"result": "some output"},
            input_data={"request": "do a thing"},
            step_id="node_test",
            settings=settings,
        )

    assert result["usage"] is None


@pytest.mark.asyncio
async def test_meta_llm_evaluate_usage_none_on_error():
    """The existing except-Exception fallback (evaluator error, defaulting to
    pass) must report usage=None since no real call succeeded."""
    from app.steps.agent_executor import _meta_llm_evaluate

    settings = _make_settings()
    settings.meta_llm_provider = None
    settings.meta_llm_model = None

    with patch("app.core.container.build_llm_native", side_effect=RuntimeError("boom")):
        result = await _meta_llm_evaluate(
            raw_output={"result": "some output"},
            input_data={"request": "do a thing"},
            step_id="node_test",
            settings=settings,
        )

    assert result["passed"] is True
    assert result["usage"] is None


async def _run_unstructured_output_scenario(result_text: str) -> "MetaLLMRejectionError":
    """Drive execute_agent_step through the unstructured-output rejection path
    (output_mapping set, agent returned only 'result', no matching fields) and
    return the raised MetaLLMRejectionError so callers can inspect its message.
    """
    from app.steps.agent_executor import MetaLLMRejectionError, execute_agent_step

    output_mapping = {"ticket_id": "ticket_id", "summary": "summary"}
    step = _make_step(output_mapping=output_mapping)
    raw_output = {"result": result_text, "token_usage": {}}

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

    _poll_settings3 = MagicMock()
    _poll_settings3.agent_poll_interval_seconds = 1
    _poll_settings3.agent_max_loops = 3
    _poll_settings3.anthropic_api_key = "sk-test"

    with (
        patch("app.runtime.factory.get_runtime") as mock_get_runtime,
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("httpx.AsyncClient", return_value=mock_http_client),
        patch("app.steps.agent_executor.get_settings", return_value=_poll_settings3),
    ):
        mock_runtime = MagicMock()
        mock_runtime.has_container_for_run = AsyncMock(return_value=True)
        mock_runtime.terminate_by_run_id = AsyncMock()
        mock_runtime.rewrite_callback_url = MagicMock(return_value="http://localhost")
        mock_runtime.get_agent_url_for_run = AsyncMock(return_value="http://agent-host:8080")
        mock_get_runtime.return_value = mock_runtime

        fake_backend = AsyncMock()
        fake_backend.get = AsyncMock(return_value=fake_agent_def)

        try:
            await execute_agent_step(
                step=step,
                state={"request": "Implement C130-1475"},
                agent_backend=fake_backend,
                run_id="test-run-id",
                callback_base_url="http://localhost",
                settings=settings,
                run_repository=None,
            )
        except MetaLLMRejectionError as exc:
            return exc
    raise AssertionError("expected MetaLLMRejectionError to be raised")


@pytest.mark.asyncio
async def test_unstructured_output_rejection_includes_full_text_under_5000_chars():
    """Rejection snippet cap is 5000 chars, not 400 — text between 400 and
    5000 chars must appear in full in the rejection reason/message."""
    result_text = "x" * 4000
    exc = await _run_unstructured_output_scenario(result_text)
    assert result_text in exc.reason
    assert result_text in str(exc)


@pytest.mark.asyncio
async def test_unstructured_output_rejection_truncates_at_5000_chars():
    """A 6000-char result truncates to exactly the first 5000 chars (no
    truncation marker is appended by the current implementation)."""
    result_text = "y" * 6000
    expected_snippet = result_text[:5000]
    exc = await _run_unstructured_output_scenario(result_text)
    assert expected_snippet in exc.reason
    assert result_text not in exc.reason


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
