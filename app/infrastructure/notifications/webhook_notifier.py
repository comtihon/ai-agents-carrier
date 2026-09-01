from __future__ import annotations

import logging
import re
import string
from typing import Any, cast

import httpx

from app.infrastructure.messaging.base import MessagingError
from app.infrastructure.messaging.slack import (
    SLACK_BLOCK_TEXT_LIMIT as _SLACK_BLOCK_TEXT_LIMIT,
)
from app.infrastructure.messaging.slack import SlackProvider, apply_thread_context

logger = logging.getLogger(__name__)


def _slack(bot_token: str | None = None) -> SlackProvider:
    """The one Slack client in the backend.

    ``bot_token`` is what the call sites already carry (a step-level override, or
    ``settings.slack_bot_token``); passing ``None`` lets the provider resolve the
    token from settings/env itself.  Either way the HTTP work, the error
    handling and the token scrubbing live in
    ``app.infrastructure.messaging.slack`` and nowhere else.
    """
    from app.infrastructure.messaging.registry import get_provider

    if bot_token:
        return SlackProvider(bot_token)
    return cast(SlackProvider, get_provider("slack"))


def _md_to_slack(text: str) -> str:
    """Convert common GitHub-flavoured markdown to Slack mrkdwn."""
    return re.sub(r'\*\*(.+?)\*\*', r'*\1*', text, flags=re.DOTALL)


def _render(template: str, ctx: dict) -> str:
    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return ""
    try:
        return string.Formatter().vformat(template, [], _DefaultDict(ctx))  # type: ignore[arg-type]
    except ValueError:
        return template


def _truncate(value: str, limit: int = _SLACK_BLOCK_TEXT_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n…(truncated)"


_SUMMARY_TARGET_CHARS = 2600
_SUMMARIZE_INPUT_CAP = 20000

_SUMMARIZE_PROMPT = (
    "Cut text below to fit Slack block. HARD LIMIT: {max_chars} chars. Not a target — a wall.\n"
    "Keep: headers, key names/IDs/files/numbers, bullets.\n"
    "Drop: prose, repetition, detail.\n"
    "Write caveman-style: drop articles/filler words, short fragments ok, keep technical terms exact.\n"
    "Length beats completeness. Cut harder if unsure.\n"
    "Output: summary only. No preamble, no fences.\n\n"
    "TEXT:\n{text}"
)


async def _summarize_for_slack(text: str, settings: Any) -> str:
    """Summarize an oversized Slack block-text field via the meta-LLM.

    Falls back to the existing hard _truncate() if the LLM call fails for any
    reason, or if its output is itself still over the Slack block limit —
    Slack must never receive an over-limit block regardless of LLM behavior.
    """
    try:
        from app.core.container import build_llm_native
        from langchain_core.messages import HumanMessage

        provider = settings.meta_llm_provider or settings.llm_provider
        model = settings.meta_llm_model
        llm = build_llm_native(provider, model, settings, max_tokens=1024)

        prompt = _SUMMARIZE_PROMPT.format(
            max_chars=_SUMMARY_TARGET_CHARS, text=text[:_SUMMARIZE_INPUT_CAP]
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        summary = response.content if isinstance(response.content, str) else str(response.content)
        summary = summary.strip()

        if not summary:
            return _truncate(text)
        if len(summary) <= _SLACK_BLOCK_TEXT_LIMIT:
            return summary
        return _truncate(summary)
    except Exception as exc:
        logger.warning("Slack block-text summarization failed, falling back to truncate: %s", exc)
        return _truncate(text)


async def _render_value(value: Any, ctx: dict, settings: Any = None,
                         _in_block_text: bool = False) -> Any:
    if isinstance(value, str):
        rendered = _render(value, ctx)
        if not _in_block_text:
            return rendered
        if len(rendered) <= _SLACK_BLOCK_TEXT_LIMIT:
            return rendered
        if settings is None:
            return _truncate(rendered)
        return await _summarize_for_slack(rendered, settings)
    if isinstance(value, dict):
        # Detect a Slack block text object: {"type": "mrkdwn"|"plain_text", "text": "..."}
        is_block_text = value.get("type") in ("mrkdwn", "plain_text") and "text" in value
        return {k: await _render_value(v, ctx, settings, _in_block_text=is_block_text and k == "text")
                for k, v in value.items()}
    if isinstance(value, list):
        return [await _render_value(v, ctx, settings) for v in value]
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

    payload = await _render_value(notify.get("payload", {}), ctx, settings)

    # For Slack chat.postMessage: if a previous approval already created a thread,
    # reply in that thread and tag whoever approved it.  The payload surgery is
    # Slack knowledge, so it lives with the Slack provider — this stays the
    # generic webhook sender it has always been (any url, method, auth).
    if "slack.com/api/chat.postMessage" in url:
        apply_thread_context(
            payload,
            state.get("_slack_thread_ts"),
            state.get("_slack_approver_id") or "",
        )

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


def _format_questions(questions: list[str]) -> str:
    """Format a list of questions for Slack mrkdwn.

    Converts **bold** to *bold*, and only prepends a counter when the question
    does not already start with its own number (e.g. "1. Are you...").
    """
    lines: list[str] = []
    for i, q in enumerate(questions):
        q = _md_to_slack(q)
        if re.match(r"^\d+[.)]\s", q.lstrip()):
            lines.append(q)
        else:
            lines.append(f"{i + 1}. {q}")
    return "\n".join(lines)


async def post_slack_thread_reply(
    bot_token: str,
    channel: str,
    thread_ts: str,
    text: str,
) -> None:
    """Post a plain text reply in an existing Slack thread."""
    try:
        await _slack(bot_token).reply_in_thread(channel, thread_ts, text)
    except MessagingError as exc:
        logger.warning("Slack thread reply failed: %s", exc)
    except Exception:
        logger.exception("Failed to post Slack thread reply")


async def post_slack_thread_questions(
    bot_token: str,
    channel: str,
    thread_ts: str,
    questions: list[str],
) -> None:
    """Post ask_context questions as a reply in an existing Slack thread."""
    if not questions:
        return
    text = f"I need a bit more information to proceed:\n\n{_format_questions(questions)}"
    try:
        await _slack(bot_token).reply_in_thread(channel, thread_ts, text)
    except MessagingError as exc:
        logger.warning("Slack thread post failed: %s", exc)
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
    n = len(questions)
    hint = "Reply in this thread with your answer." if n == 1 else \
        f"Reply in this thread with {n} numbered answers, one per line."
    text = f"*Context needed for `{ticket_id}`*\n\n{_format_questions(questions)}\n\n_{hint}_"
    try:
        posted = await _slack(bot_token).post_message(channel, text)
        return posted.raw
    except MessagingError as exc:
        logger.warning("Slack ask_context post failed: %s", exc)
        return None
    except Exception:
        logger.exception("Failed to post ask_context questions to Slack")
        return None


async def post_slack_addon_notification(
    bot_token: str,
    payload_template: str,
    run_id: str,
    state: dict[str, Any],
    questions: list[str] | None = None,
) -> None:
    """Send a custom Slack message configured on a workflow addon.

    ``payload_template`` is a JSON string with template variables:
      {run_id}       — workflow run id
      {request}      — user request from state
      {questions}    — formatted clarification questions (if any)
      Any other state key.
    The message always goes out as ``chat.postMessage``.
    """
    import json as _json
    from app.core.config import get_settings
    settings = get_settings()

    ctx: dict[str, Any] = dict(state)
    ctx["run_id"] = run_id
    ctx["questions"] = _format_questions(questions) if questions else ""
    ctx.setdefault("request", "")
    # Inject Slack credentials so payload templates can reference {slack_bot_token}
    # and {slack_approvals_channel} without storing them in the workflow definition.
    ctx.setdefault("slack_bot_token", settings.slack_bot_token or "")
    ctx.setdefault("slack_approvals_channel", settings.slack_approvals_channel or "")

    try:
        rendered_str = _render(payload_template, ctx)
        payload = _json.loads(rendered_str)
    except Exception as exc:
        logger.warning("run %s: slack addon payload JSON parse failed: %s", run_id, exc)
        return

    payload = await _render_value(payload, ctx, settings)

    try:
        await _slack(bot_token).post_payload(payload)
    except MessagingError as exc:
        logger.warning("run %s: slack addon notification failed: %s", run_id, exc)
    except Exception:
        logger.exception("run %s: failed to send slack addon notification", run_id)


# ── Data-source approval cases ────────────────────────────────────────────────

# Slack action ids for the approval-case buttons. Distinct from the plain
# "approve"/"reject" pair the human_approval gate uses, because those carry a
# run id in ``value`` and these carry a case id — one handler must be able to
# tell them apart without guessing at the shape of the value.
APPROVAL_APPROVE_ACTION = "ds_approval_approve"
APPROVAL_REJECT_ACTION = "ds_approval_reject"
APPROVAL_VETO_ACTION = "ds_approval_veto"

_APPROVAL_SAMPLE_LINES = 10


def _approval_blocks(case: Any, *, mode: str = "request") -> list[dict[str, Any]]:
    """The message body an approver reads before answering.

    Ordered by what decides the answer: the blast radius first (it is the whole
    reason this message exists), then what is being hit, then the meta-LLM's
    recommendation — last, and clearly labelled as advice, so it informs the
    decision instead of standing in for it.

    ``mode`` picks which of three messages this is: ``request`` (Approve /
    Reject), ``veto`` (the meta-LLM decided; one Cancel button and a deadline),
    or ``notice`` (already done, no buttons — offering an action on a closed
    case would be a lie).
    """
    rows = "1 row" if case.affected_rows == 1 else f"{case.affected_rows} rows"
    headline = {
        "request": f"*Data deletion awaiting approval* — `{rows}`",
        "veto": f"*Data deletion auto-approved* — `{rows}`",
        "notice": f"*Data deletion confirmed* — `{rows}`",
    }.get(mode, f"*Data deletion* — `{rows}`")
    fields = [
        {"type": "mrkdwn", "text": f"*Data source*\n{case.datasource_name or case.datasource_id}"},
        {"type": "mrkdwn", "text": f"*Operation*\n`{case.operation}` [{case.method}]"},
        {"type": "mrkdwn", "text": f"*Workflow*\n{case.workflow_name or case.workflow_id or '—'}"},
        {"type": "mrkdwn", "text": f"*Run*\n`{case.run_id or '—'}`"},
    ]
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": headline}},
        {"type": "section", "fields": fields},
    ]

    if case.endpoint:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Endpoint*\n`{_truncate(case.endpoint, 400)}`"},
        })
    if case.params:
        import json as _json
        body = _json.dumps(case.params, default=str, indent=2)
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Input*\n```{_truncate(body, 1200)}```"},
        })
    if case.affected_sample:
        shown = case.affected_sample[:_APPROVAL_SAMPLE_LINES]
        more = case.affected_rows - len(shown)
        listing = "\n".join(f"• {s}" for s in shown)
        if more > 0:
            listing += f"\n• …and {more} more"
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Targets*\n{_truncate(listing, 1200)}"},
        })

    verdict = case.meta_llm
    if verdict is not None and verdict.decision != "abstain":
        label = "would approve" if verdict.decision == "approve" else "would reject"
        note = (
            f"*Meta-LLM {label}* — {verdict.reason or 'no reason given'}\n"
            f"_Based on {verdict.history_size} prior decision(s) on this operation. "
            + ("This is the decision; cancel below to stop it._"
               if verdict.autonomous else "Advisory only — you decide._")
        )
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": _truncate(note, 2800)}})

    if mode == "notice":
        who = case.decided_by_name or "someone"
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"Confirmed by {who} in the data source editor."},
        })
    elif mode == "veto":
        blocks.append({
            "type": "actions",
            "elements": [{
                "type": "button",
                "style": "danger",
                "text": {"type": "plain_text", "text": "Cancel this deletion"},
                "action_id": APPROVAL_VETO_ACTION,
                "value": case.id,
            }],
        })
    else:
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "style": "primary",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "action_id": APPROVAL_APPROVE_ACTION,
                    "value": case.id,
                },
                {
                    "type": "button",
                    "style": "danger",
                    "text": {"type": "plain_text", "text": "Reject"},
                    "action_id": APPROVAL_REJECT_ACTION,
                    "value": case.id,
                },
            ],
        })
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"case `{case.id}`"}],
    })
    return blocks


async def post_slack_approval_case(
    bot_token: str,
    channel: str,
    case: Any,
    *,
    mode: str = "request",
) -> dict[str, Any] | None:
    """Announce an approval case in Slack; returns the raw API response.

    See :func:`_approval_blocks` for what ``mode`` selects.
    """
    rows = "1 row" if case.affected_rows == 1 else f"{case.affected_rows} rows"
    prefix = {
        "request": "Approval needed",
        "veto": "Auto-approved",
        "notice": "Deletion confirmed",
    }.get(mode, "Deletion")
    fallback = (
        f"{prefix}: "
        f"{case.datasource_name or case.datasource_id}.{case.operation} "
        f"[{case.method}] affecting {rows}"
    )
    try:
        posted = await _slack(bot_token).post_message(
            channel, fallback, blocks=_approval_blocks(case, mode=mode)
        )
        return posted.raw
    except MessagingError as exc:
        logger.warning("approval case %s: Slack post failed: %s", case.id, exc)
        return None
    except Exception:
        logger.exception("approval case %s: failed to post to Slack", case.id)
        return None


async def post_slack_approval_outcome(
    bot_token: str,
    channel: str,
    thread_ts: str,
    text: str,
) -> None:
    """Close the loop in the case's own thread once it has been decided."""
    if not (channel and thread_ts):
        return
    await post_slack_thread_reply(bot_token, channel, thread_ts, text)
