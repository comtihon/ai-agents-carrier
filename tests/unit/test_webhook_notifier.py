"""Unit tests for webhook_notifier Slack block-text summarization."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.infrastructure.notifications.webhook_notifier import (
    _SLACK_BLOCK_TEXT_LIMIT,
    _render_value,
    _summarize_for_slack,
    _truncate,
)


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.meta_llm_provider = "openrouter"
    settings.meta_llm_model = "moonshotai/kimi-k2.6"
    settings.llm_provider = "openrouter"
    return settings


def _make_llm(contents):
    """Build a fake LLM whose ainvoke returns MagicMock(spec=["content"]) responses.

    ``contents`` is either a single string or a list of strings (side_effect).
    """
    llm = MagicMock()
    if isinstance(contents, list):
        responses = [MagicMock(spec=["content"], content=c) for c in contents]
        llm.ainvoke = AsyncMock(side_effect=responses)
    else:
        llm.ainvoke = AsyncMock(return_value=MagicMock(spec=["content"], content=contents))
    return llm


@pytest.mark.asyncio
async def test_short_block_text_no_llm_call():
    settings = _make_settings()
    value = {"type": "mrkdwn", "text": "x" * 2900}

    with patch("app.core.container.build_llm_native") as mock_build:
        result = await _render_value(value, {}, settings)

    mock_build.assert_not_called()
    assert result["text"] == "x" * 2900
    assert "…(truncated)" not in result["text"]


@pytest.mark.asyncio
async def test_long_non_block_text_untouched_no_llm():
    settings = _make_settings()
    # Nested under a non-block-text dict key so _in_block_text never becomes true.
    value = {"note": "y" * 5000}

    with patch("app.core.container.build_llm_native") as mock_build:
        result = await _render_value(value, {}, settings)

    mock_build.assert_not_called()
    assert result["note"] == "y" * 5000


@pytest.mark.asyncio
async def test_long_block_text_summarized():
    settings = _make_settings()
    value = {"type": "mrkdwn", "text": "x" * 5000}
    fake_llm = _make_llm("short summary")

    with patch("app.core.container.build_llm_native", return_value=fake_llm) as mock_build:
        result = await _render_value(value, {}, settings)

    assert result["text"] == "short summary"
    mock_build.assert_called_once_with(
        settings.meta_llm_provider, settings.meta_llm_model, settings, max_tokens=1024
    )


@pytest.mark.asyncio
async def test_llm_error_falls_back_to_truncate():
    settings = _make_settings()
    rendered = "x" * 5000
    value = {"type": "mrkdwn", "text": rendered}

    with patch("app.core.container.build_llm_native", side_effect=RuntimeError("boom")):
        result = await _render_value(value, {}, settings)

    assert result["text"] == _truncate(rendered)


@pytest.mark.asyncio
async def test_llm_output_still_too_long_gets_truncated():
    settings = _make_settings()
    value = {"type": "mrkdwn", "text": "x" * 5000}
    llm_content = "z" * 6000
    fake_llm = _make_llm(llm_content)

    with patch("app.core.container.build_llm_native", return_value=fake_llm):
        result = await _render_value(value, {}, settings)

    assert result["text"] == _truncate(llm_content)
    assert len(result["text"]) <= _SLACK_BLOCK_TEXT_LIMIT + len("\n…(truncated)")
    assert result["text"].endswith("…(truncated)")


@pytest.mark.asyncio
async def test_empty_summary_falls_back_to_truncate_original():
    settings = _make_settings()
    rendered = "x" * 5000
    value = {"type": "mrkdwn", "text": rendered}
    fake_llm = _make_llm("")

    with patch("app.core.container.build_llm_native", return_value=fake_llm):
        result = await _render_value(value, {}, settings)

    assert result["text"] == _truncate(rendered)


@pytest.mark.asyncio
async def test_multiple_oversized_fields_independent_calls():
    settings = _make_settings()
    payload = {
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": "a" * 5000}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "b" * 5000}},
        ]
    }
    fake_llm = _make_llm(["summary one", "summary two"])

    with patch("app.core.container.build_llm_native", return_value=fake_llm) as mock_build:
        result = await _render_value(payload, {}, settings)

    assert mock_build.call_count == 2
    texts = [b["text"]["text"] for b in result["blocks"]]
    assert texts == ["summary one", "summary two"]
    assert texts[0] != texts[1]
