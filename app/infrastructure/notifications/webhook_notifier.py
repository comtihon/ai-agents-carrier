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
) -> dict[str, Any] | None:
    """POST an approval notification to a configured URL.

    Template variables available in ``payload`` values, header values, and the URL:
      {run_id}                  — the workflow run ID
      {approve_url}             — callback URL to approve the run
      {reject_url}              — callback URL to reject the run
      {slack_bot_token}         — injected from SLACK_BOT_TOKEN setting
      {slack_approvals_channel} — injected from SLACK_APPROVALS_CHANNEL setting
      Any key from the current graph state (e.g. {plan}, {request}).

    Returns the parsed JSON response body if the endpoint returned one, otherwise None.
    When using the Slack Web API (chat.postMessage), the response contains ``ts`` and
    ``channel`` which callers can use to post follow-up messages in the same thread.

    Threading: when ``_slack_thread_ts`` is already in state, a chat.postMessage call
    will automatically be sent as a thread reply.  If ``_slack_approver_id`` is also
    in state the approver is tagged at the start of the message.
    """
    from app.core.config import get_settings
    settings = get_settings()

    url = notify.get("url")
    if not url:
        logger.warning("run %s: notify config missing 'url', skipping", run_id)
        return None

    ctx: dict[str, Any] = dict(state)
    ctx["run_id"] = run_id
    base = base_url.rstrip("/")
    ctx["approve_url"] = f"{base}/api/v1/callbacks/{run_id}/approve"
    ctx["reject_url"] = f"{base}/api/v1/callbacks/{run_id}/reject"
    # Inject Slack credentials so notify configs can reference them as {slack_bot_token}
    # and {slack_approvals_channel} without storing sensitive values in the database.
    ctx.setdefault("slack_bot_token", settings.slack_bot_token)
    ctx.setdefault("slack_approvals_channel", settings.slack_approvals_channel)

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

    # For Slack chat.postMessage: if a previous approval already created a thread,
    # reply in that thread and tag whoever approved it.
    if "slack.com/api/chat.postMessage" in url:
        thread_ts = state.get("_slack_thread_ts")
        approver_id = state.get("_slack_approver_id") or ""
        if thread_ts:
            payload["thread_ts"] = thread_ts
            if approver_id:
                mention = f"<@{approver_id}> "
                payload["text"] = mention + payload.get("text", "")
                # Also prepend to the first mrkdwn section block so rich formatting includes it
                for block in payload.get("blocks", []):
                    if block.get("type") == "section" and isinstance(block.get("text"), dict):
                        block["text"]["text"] = mention + block["text"].get("text", "")
                        break

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
            try:
                return response.json()
            except Exception:
                return None
    except Exception:
        logger.exception("run %s: failed to send approval notification", run_id)
        return None


async def post_slack_thread_questions(
    bot_token: str,
    channel: str,
    thread_ts: str,
    questions: list[str],
) -> None:
    """Post ask_context questions as a reply in an existing Slack thread."""
    if not questions:
        return
    lines = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    text = f"I need a bit more information to proceed:\n\n{lines}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {bot_token}"},
                json={"channel": channel, "thread_ts": thread_ts, "text": text},
            )
            data = response.json()
            if not data.get("ok"):
                logger.warning("Slack thread post failed: %s", data.get("error"))
    except Exception:
        logger.exception("Failed to post ask_context questions to Slack thread")


async def post_slack_ask_context(
    bot_token: str,
    channel: str,
    questions: list[str],
    run_id: str,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """Post ask_context questions as a new root-level Slack message.

    Returns the Slack API response (contains ``ts`` and ``channel``).
    """
    if not questions:
        return None
    ticket_id = state.get("ticket_id") or run_id
    lines = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    n = len(questions)
    hint = "Reply in this thread with your answer." if n == 1 else \
        f"Reply in this thread with {n} numbered answers, one per line."
    text = f"*Context needed for `{ticket_id}`*\n\n{lines}\n\n_{hint}_"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {bot_token}"},
                json={"channel": channel, "text": text},
            )
            data = response.json()
            if not data.get("ok"):
                logger.warning("Slack ask_context post failed: %s", data.get("error"))
                return None
            return data
    except Exception:
        logger.exception("Failed to post ask_context questions to Slack")
        return None
