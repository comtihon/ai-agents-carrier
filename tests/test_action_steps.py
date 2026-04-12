from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.domain.models.runtime import WorkflowRun
from app.domain.models.workflow_definition import WorkflowStepDefinition
from app.infrastructure.actions.http_executor import HttpStepExecutor
from app.infrastructure.actions.registry import ActionRegistry
from app.infrastructure.actions.templates import build_template_context, resolve_templates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(**kwargs: Any) -> WorkflowRun:
    return WorkflowRun(
        workflow_id=kwargs.get("workflow_id", "wf-test"),
        workflow_name=kwargs.get("workflow_name", "Test Workflow"),
        user_request=kwargs.get("user_request", "Do something useful"),
    )


def _make_http_step(**kwargs: Any) -> WorkflowStepDefinition:
    return WorkflowStepDefinition(
        id=kwargs.get("id", "notify"),
        name=kwargs.get("name", "Notify"),
        type="http",
        url=kwargs.get("url", "https://internal.example.com/notify"),
        method=kwargs.get("method", "POST"),
        body=kwargs.get("body", {}),
        http_headers=kwargs.get("http_headers", {}),
        output_key=kwargs.get("output_key", None),
    )


def _mock_httpx_client(response_json: Any = None, status_code: int = 200, raises: Exception | None = None):
    """Return a patched httpx.AsyncClient context manager."""
    mock_response = MagicMock()
    if raises:
        mock_response.raise_for_status.side_effect = raises
    else:
        mock_response.raise_for_status = MagicMock()
        if response_json is not None:
            mock_response.json.return_value = response_json
        else:
            mock_response.json.side_effect = ValueError("not json")
            mock_response.text = "ok"
            mock_response.status_code = status_code

    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


# ---------------------------------------------------------------------------
# Template resolution tests
# ---------------------------------------------------------------------------

def test_resolve_templates_replaces_run_id() -> None:
    run = _make_run()
    ctx = build_template_context(run)
    result = resolve_templates("id={{ run.id }}", ctx)
    assert result == f"id={run.id}"


def test_resolve_templates_nested_dict() -> None:
    run = _make_run()
    ctx = build_template_context(run)
    payload = {"run_id": "{{ run.id }}", "nested": {"wf": "{{ run.workflow_id }}"}}
    result = resolve_templates(payload, ctx)
    assert result == {"run_id": run.id, "nested": {"wf": run.workflow_id}}


def test_resolve_templates_list_items() -> None:
    run = _make_run()
    ctx = build_template_context(run)
    result = resolve_templates(["{{ run.id }}", "{{ run.workflow_name }}"], ctx)
    assert result == [run.id, run.workflow_name]


def test_resolve_templates_unknown_key_is_left_unchanged() -> None:
    run = _make_run()
    ctx = build_template_context(run)
    result = resolve_templates("{{ unknown.key }}", ctx)
    assert result == "{{ unknown.key }}"


def test_resolve_templates_non_string_values_pass_through() -> None:
    run = _make_run()
    ctx = build_template_context(run)
    assert resolve_templates(42, ctx) == 42
    assert resolve_templates(True, ctx) is True
    assert resolve_templates(None, ctx) is None


# ---------------------------------------------------------------------------
# HTTP step executor tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_http_executor_calls_correct_url_and_method() -> None:
    mock_client = _mock_httpx_client(response_json={"notified": True})
    with patch("httpx.AsyncClient", return_value=mock_client):
        executor = HttpStepExecutor(timeout=10.0)
        step = _make_http_step(url="https://internal.example.com/notify", method="POST")
        result = await executor.execute(step, _make_run())

    assert result == {"notified": True}
    mock_client.request.assert_called_once_with(
        method="POST",
        url="https://internal.example.com/notify",
        json=None,
        headers={},
    )


@pytest.mark.asyncio
async def test_http_executor_resolves_body_templates() -> None:
    mock_client = _mock_httpx_client(response_json={"ok": True})
    with patch("httpx.AsyncClient", return_value=mock_client):
        executor = HttpStepExecutor(timeout=10.0)
        run = _make_run()
        step = _make_http_step(
            body={"run_id": "{{ run.id }}", "workflow": "{{ run.workflow_id }}"}
        )
        await executor.execute(step, run)

    _, call_kwargs = mock_client.request.call_args
    assert call_kwargs["json"] == {"run_id": run.id, "workflow": run.workflow_id}


@pytest.mark.asyncio
async def test_http_executor_resolves_header_templates() -> None:
    mock_client = _mock_httpx_client(response_json={})
    with patch("httpx.AsyncClient", return_value=mock_client):
        executor = HttpStepExecutor(timeout=10.0)
        run = _make_run()
        step = _make_http_step(http_headers={"X-Run-Id": "{{ run.id }}"})
        await executor.execute(step, run)

    _, call_kwargs = mock_client.request.call_args
    assert call_kwargs["headers"] == {"X-Run-Id": run.id}


@pytest.mark.asyncio
async def test_http_executor_raises_on_http_error() -> None:
    error = httpx.HTTPStatusError("404 Not Found", request=MagicMock(), response=MagicMock())
    mock_client = _mock_httpx_client(raises=error)
    with patch("httpx.AsyncClient", return_value=mock_client):
        executor = HttpStepExecutor(timeout=10.0)
        with pytest.raises(httpx.HTTPStatusError):
            await executor.execute(_make_http_step(), _make_run())


@pytest.mark.asyncio
async def test_http_executor_falls_back_to_text_when_response_is_not_json() -> None:
    mock_client = _mock_httpx_client(status_code=200)  # no response_json → text fallback
    with patch("httpx.AsyncClient", return_value=mock_client):
        executor = HttpStepExecutor(timeout=10.0)
        result = await executor.execute(_make_http_step(), _make_run())

    assert result == {"status_code": 200, "body": "ok"}


@pytest.mark.asyncio
async def test_http_executor_uses_get_method() -> None:
    mock_client = _mock_httpx_client(response_json={"data": []})
    with patch("httpx.AsyncClient", return_value=mock_client):
        executor = HttpStepExecutor(timeout=10.0)
        step = _make_http_step(method="GET")
        await executor.execute(step, _make_run())

    _, call_kwargs = mock_client.request.call_args
    assert call_kwargs["method"] == "GET"


# ---------------------------------------------------------------------------
# Action registry tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_action_registry_executes_registered_handler() -> None:
    registry = ActionRegistry()

    async def my_handler(handler_input: dict, run: WorkflowRun) -> dict:
        return {"handled": True, "received": handler_input}

    registry.register("my.handler", my_handler)
    result = await registry.execute("my.handler", {"key": "value"}, _make_run())
    assert result == {"handled": True, "received": {"key": "value"}}


@pytest.mark.asyncio
async def test_action_registry_resolves_input_templates() -> None:
    registry = ActionRegistry()
    captured: dict = {}

    async def capture(handler_input: dict, run: WorkflowRun) -> dict:
        captured.update(handler_input)
        return {}

    registry.register("capture", capture)
    run = _make_run()
    await registry.execute("capture", {"run_id": "{{ run.id }}", "wf": "{{ run.workflow_id }}"}, run)

    assert captured["run_id"] == run.id
    assert captured["wf"] == run.workflow_id


@pytest.mark.asyncio
async def test_action_registry_passes_run_to_handler() -> None:
    registry = ActionRegistry()
    received_run: WorkflowRun | None = None

    async def run_receiver(handler_input: dict, run: WorkflowRun) -> dict:
        nonlocal received_run
        received_run = run
        return {}

    registry.register("run.receiver", run_receiver)
    run = _make_run(workflow_id="wf-captured")
    await registry.execute("run.receiver", {}, run)

    assert received_run is not None
    assert received_run.workflow_id == "wf-captured"


@pytest.mark.asyncio
async def test_action_registry_raises_key_error_for_unknown_handler() -> None:
    registry = ActionRegistry()
    with pytest.raises(KeyError, match="not.registered"):
        await registry.execute("not.registered", {}, _make_run())


@pytest.mark.asyncio
async def test_action_registry_error_message_lists_available_handlers() -> None:
    registry = ActionRegistry()

    async def existing(handler_input: dict, run: WorkflowRun) -> dict:
        return {}

    registry.register("existing.handler", existing)
    with pytest.raises(KeyError, match="existing.handler"):
        await registry.execute("missing.handler", {}, _make_run())


def test_action_registry_registered_names() -> None:
    registry = ActionRegistry()

    async def h(handler_input: dict, run: WorkflowRun) -> dict:
        return {}

    registry.register("a.b", h)
    registry.register("c.d", h)
    assert set(registry.registered_names()) == {"a.b", "c.d"}


# ---------------------------------------------------------------------------
# WorkflowStepDefinition validation tests
# ---------------------------------------------------------------------------

def test_http_step_requires_url() -> None:
    with pytest.raises(ValueError, match="must specify a 'url'"):
        WorkflowStepDefinition(id="s", name="S", type="http")


def test_action_step_requires_handler() -> None:
    with pytest.raises(ValueError, match="must specify a 'handler'"):
        WorkflowStepDefinition(id="s", name="S", type="action")


def test_http_step_default_method_is_post() -> None:
    step = WorkflowStepDefinition(id="s", name="S", type="http", url="https://example.com")
    assert step.method == "POST"


def test_http_step_accepts_all_methods() -> None:
    for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        step = WorkflowStepDefinition(
            id="s", name="S", type="http", url="https://example.com", method=method  # type: ignore[arg-type]
        )
        assert step.method == method
