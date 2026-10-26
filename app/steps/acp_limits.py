"""Recognise an ACP agent running out of quota, and when it can continue.

Neither adapter has a typed "usage limit" signal yet, so this reads what they
send today (claude-agent-acp 0.97, codex-acp 0.16):

Claude
    ``usage_update._meta["_claude/rateLimit"]``: ``{status, resetsAt (unix s),
    rateLimitType, …}`` as the rate limit changes during a turn; and a failed
    ``session/prompt`` (``-32603``) whose message carries the CLI's text, e.g.
    ``Internal error: You've hit your limit · resets 3pm (Europe/Berlin)``,
    with ``data.errorKind``.
Codex
    a failed ``session/prompt`` (``-32603``) with ``data: {message,
    codex_error_info: "usage_limit_exceeded"}``; the message ends in
    ``Try again at 3:42 PM.`` (the runner's local time; no date when it is
    today) or ``Try again at Oct 27th, 2026 3:42 PM.``.

Kinds:
    usage_limit     wait until ``resets_at`` (None: unknown — retry later)
    quota_exhausted no reset will help (credits, spend cap, plan): fail
    rate_limited    transient 429 the adapter's own retries did not absorb
    overloaded      the provider is overloaded
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# The Claude Agent SDK's USAGE_LIMIT_ERROR_PREFIXES, split by whether waiting helps.
_CLAUDE_RESETTING = ("You've hit your", "You've reached your")
_CLAUDE_EXHAUSTED = (
    "You're out of usage credits",
    "Your org is out of usage",
    "Your seat type doesn't include",
    "Your usage allocation has been disabled",
    "Your group's usage limit is set to $0",
    "requires usage credits",
    "You're out of extra usage",
)
_CODEX_EXHAUSTED = re.compile(
    r"out of credits|spend cap|Quota exceeded|not included in your plan|Upgrade to Plus to continue", re.I
)

_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}

# "resets 3pm (Europe/Berlin)", "resets 2:30pm", "resets Oct 30, 9am (UTC)", "resets Oct 30 at 9am"
_CLAUDE_RESET = re.compile(
    r"resets\s+(?:(?P<mon>[A-Za-z]{3})[a-z]*\.?\s+(?P<day>\d{1,2})(?:,|\s+at)?\s+)?"
    r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>am|pm)(?:\s*\((?P<tz>[^)]+)\))?",
    re.I,
)
# "Try again at 3:42 PM." / "try again at Oct 27th, 2026 3:42 PM."
_CODEX_RESET = re.compile(
    r"try again at\s+(?:(?P<mon>[A-Za-z]{3})[a-z]*\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,\s*(?P<year>\d{4})\s+)?"
    r"(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ampm>AM|PM)",
    re.I,
)
_LEGACY_UNIX = re.compile(r"usage limit reached\|(\d{10})", re.I)


@dataclass
class LimitInfo:
    kind: str
    resets_at: datetime | None
    message: str


def _zone(name: str | None) -> timezone | ZoneInfo:
    if name:
        try:
            return ZoneInfo(name.strip())
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return timezone.utc


def _at(m: re.Match[str], tz: timezone | ZoneInfo, now: datetime) -> datetime | None:
    hour = int(m.group("h")) % 12 + (12 if m.group("ampm").lower() == "pm" else 0)
    minute = int(m.group("m") or 0)
    local_now = now.astimezone(tz)
    mon = (m.groupdict().get("mon") or "").lower()[:3]
    try:
        if mon in _MONTHS:
            year = int(m.groupdict().get("year") or local_now.year)
            at = datetime(year, _MONTHS[mon], int(m.group("day")), hour, minute, tzinfo=tz)
            if not m.groupdict().get("year") and at < local_now - timedelta(days=1):
                at = at.replace(year=year + 1)
        else:
            at = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if at <= local_now - timedelta(minutes=1):
                at += timedelta(days=1)
    except ValueError:
        return None
    return at.astimezone(timezone.utc)


def parse_claude_reset(text: str, now: datetime, default_tz: str | None = None) -> datetime | None:
    legacy = _LEGACY_UNIX.search(text)
    if legacy:
        return datetime.fromtimestamp(int(legacy.group(1)), timezone.utc)
    m = _CLAUDE_RESET.search(text)
    return _at(m, _zone(m.group("tz") or default_tz), now) if m else None


def parse_codex_reset(text: str, now: datetime, runner_tz: str | None = None) -> datetime | None:
    m = _CODEX_RESET.search(text)
    return _at(m, _zone(runner_tz), now) if m else None


def _from_meta(meta: dict[str, Any] | None) -> datetime | None:
    if not isinstance(meta, dict):
        return None
    for key in ("resetsAt", "overageResetsAt"):
        value = meta.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return datetime.fromtimestamp(value, timezone.utc)
    return None


def error_text(message: str, data: Any) -> str:
    parts = [message or ""]
    if isinstance(data, dict):
        for key in ("message", "details", "detail"):
            if isinstance(data.get(key), str):
                parts.append(data[key])
    elif isinstance(data, str):
        parts.append(data)
    return " ".join(p for p in parts if p).strip()


def classify_limit(
    message: str,
    data: Any,
    *,
    rate_limit: dict[str, Any] | None = None,
    tz: str | None = None,
    now: datetime | None = None,
) -> LimitInfo | None:
    """The limit a failed ``session/prompt`` reports, or None for any other error.

    *rate_limit* is the last ``_claude/rateLimit`` meta seen in the turn; *tz*
    the agent machine's time zone (Codex prints local times).
    """
    now = now or datetime.now(timezone.utc)
    text = error_text(message, data)
    info = data if isinstance(data, dict) else {}
    short = re.sub(r"^Internal error:?\s*", "", text)[:500]

    codex = info.get("codex_error_info")
    if isinstance(codex, dict):  # newer shapes: {"usage_limit_exceeded": {...}}
        codex = next(iter(codex), None)
    if codex == "usage_limit_exceeded":
        if _CODEX_EXHAUSTED.search(text) and not _CODEX_RESET.search(text):
            return LimitInfo("quota_exhausted", None, short)
        return LimitInfo("usage_limit", parse_codex_reset(text, now, tz), short)
    if codex == "server_overloaded":
        return LimitInfo("overloaded", None, short)

    if any(p in text for p in _CLAUDE_EXHAUSTED) or info.get("errorKind") in ("billing_error", "account_on_hold"):
        return LimitInfo("quota_exhausted", None, short)
    meta_rejected = isinstance(rate_limit, dict) and rate_limit.get("status") == "rejected"
    if any(p in text for p in _CLAUDE_RESETTING) or _LEGACY_UNIX.search(text):
        resets = (_from_meta(rate_limit) if meta_rejected else None) or parse_claude_reset(text, now, tz)
        return LimitInfo("usage_limit", resets, short)
    if info.get("errorKind") == "rate_limit":
        return LimitInfo("usage_limit" if meta_rejected else "rate_limited", _from_meta(rate_limit) if meta_rejected else None, short)
    if info.get("errorKind") == "overloaded":
        return LimitInfo("overloaded", None, short)
    return None
