from __future__ import annotations

import logging
import string
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _render(template: str, ctx: dict) -> str:
    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return ""
    try:
        return string.Formatter().vformat(template, [], _DefaultDict(ctx))  # type: ignore[arg-type]
    except ValueError:
        return template


# Slack section/input block text elements are capped at 3000 chars.
_SLACK_BLOCK_TEXT_LIMIT = 2900


def _truncate(value: str, limit: int = _SLACK_BLOCK_TEXT_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n…(truncated)"


def _render_value(value: Any, ctx: dict, _in_block_text: bool = False) -> Any:
    if isinstance(value, str):
        rendered = _render(value, ctx)
        return _truncate(rendered) if _in_block_text else rendered
    if isinstance(value, dict):
        # Detect a Slack block text object: {"type": "mrkdwn"|"plain_text", "text": "..."}
        is_block_text = value.get("type") in ("mrkdwn", "plain_text") and "text" in value
        return {k: _render_value(v, ctx, _in_block_text=is_block_text and k == "text")
                for k, v in value.items()}
    if isinstance(value, list):
        return [_render_value(v, ctx) for v in value]
    return value


async def send_approval_notification(
    notify: dict[str, Any],
    run_id: str,
    state: dict[str, Any],
    base_url: str,
) -> None:
    """POST an approval notification to a configured URL.

    Template variables available in ``payload`` values, header values, and the URL:
      {run_id}       — the workflow run ID
      {approve_url}  — callback URL to approve the run
      {reject_url}   — callback URL to reject the run
      Any key from the current graph state (e.g. {plan}, {request}).
    """
    url = notify.get("url")
    if not url:
        logger.warning("run %s: notify config missing 'url', skipping", run_id)
        return

    ctx: dict[str, Any] = dict(state)
    ctx["run_id"] = run_id
    base = base_url.rstrip("/")
    ctx["approve_url"] = f"{base}/api/v1/callbacks/{run_id}/approve"
    ctx["reject_url"] = f"{base}/api/v1/callbacks/{run_id}/reject"

    url = _render(url, ctx)
    method = notify.get("method", "POST").upper()

    headers: dict[str, str] = {
        k: _render(str(v), ctx)
        for k, v in notify.get("headers", {}).items()
    }

    httpx_auth: tuple[str, str] | None = None
    auth_config = notify.get("auth", {})
    auth_type = auth_config.get("type", "").lower()
    if auth_type == "bearer":
        token = _render(auth_config.get("token", ""), ctx)
        headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "basic":
        username = _render(auth_config.get("username", ""), ctx)
        password = _render(auth_config.get("password", ""), ctx)
        httpx_auth = (username, password)

    payload = _render_value(notify.get("payload", {}), ctx)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(
                method,
                url,
                json=payload,
                headers=headers,
                auth=httpx_auth,
            )
            response.raise_for_status()
            logger.info("run %s: approval notification sent (HTTP %d)", run_id, response.status_code)
    except Exception:
        logger.exception("run %s: failed to send approval notification", run_id)
