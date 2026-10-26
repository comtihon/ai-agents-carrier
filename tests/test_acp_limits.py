"""Usage-limit recognition for ACP agents (claude-agent-acp, codex-acp)."""

from __future__ import annotations

from datetime import datetime, timezone

from app.api.middleware.auth import _is_unprotected
from app.steps.acp_limits import classify_limit

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)


def test_claude_limit_with_reset_time_and_zone():
    info = classify_limit("Internal error: You've hit your limit · resets 3pm (Europe/Berlin)", {"errorKind": "rate_limit"}, now=NOW)
    assert info.kind == "usage_limit"
    assert info.resets_at == datetime(2026, 10, 26, 14, 0, tzinfo=timezone.utc)
    assert info.message == "You've hit your limit · resets 3pm (Europe/Berlin)"


def test_claude_reset_rolls_to_tomorrow_and_reads_dates():
    info = classify_limit("Internal error: You've hit your session limit · resets 9:30am", {}, tz="UTC", now=NOW)
    assert info.resets_at == datetime(2026, 10, 27, 9, 30, tzinfo=timezone.utc)
    info = classify_limit("Internal error: You've hit your weekly limit · resets Oct 30, 9am (UTC)", {}, now=NOW)
    assert info.resets_at == datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc)


def test_claude_rate_limit_meta_wins():
    meta = {"status": "rejected", "resetsAt": 1793100000, "rateLimitType": "five_hour"}
    info = classify_limit("Internal error: You've hit your limit · resets 3pm", {"errorKind": "rate_limit"}, rate_limit=meta, now=NOW)
    assert info.resets_at == datetime.fromtimestamp(1793100000, timezone.utc)


def test_claude_out_of_credits_is_not_waited_for():
    info = classify_limit("Internal error: You're out of usage credits", {}, now=NOW)
    assert info.kind == "quota_exhausted"
    assert classify_limit("Internal error: x", {"errorKind": "billing_error"}, now=NOW).kind == "quota_exhausted"


def test_codex_limit_in_the_runner_time_zone():
    data = {"message": "You've hit your usage limit. Try again at 3:42 PM.", "codex_error_info": "usage_limit_exceeded"}
    info = classify_limit("Internal error", data, tz="America/New_York", now=NOW)
    assert info.kind == "usage_limit"
    assert info.resets_at == datetime(2026, 10, 26, 19, 42, tzinfo=timezone.utc)
    assert info.message.startswith("You've hit your usage limit")


def test_codex_limit_with_a_date_and_without_a_time():
    data = {"message": "You've hit your usage limit. Visit https://x to purchase more credits or try again at Oct 28th, 2026 1:05 AM.",
            "codex_error_info": "usage_limit_exceeded"}
    assert classify_limit("Internal error", data, tz="UTC", now=NOW).resets_at == datetime(2026, 10, 28, 1, 5, tzinfo=timezone.utc)
    later = {"message": "You've hit your usage limit. Try again later.", "codex_error_info": "usage_limit_exceeded"}
    info = classify_limit("Internal error", later, now=NOW)
    assert info.kind == "usage_limit" and info.resets_at is None


def test_codex_spend_cap_and_overload():
    cap = {"message": "You hit your spend cap set in your workspace. Increase your spend cap to continue.", "codex_error_info": "usage_limit_exceeded"}
    assert classify_limit("Internal error", cap, now=NOW).kind == "quota_exhausted"
    assert classify_limit("Internal error", {"codex_error_info": "server_overloaded"}, now=NOW).kind == "overloaded"


def test_other_errors_are_not_limits():
    assert classify_limit("Internal error: model stream dropped", {}, now=NOW) is None
    assert classify_limit("agent exited before answering session/prompt", {"code": 1}, now=NOW) is None


def test_runner_public_paths_are_exact():
    assert _is_unprotected("POST", "/api/v1/acp/runners/enrollments")
    assert _is_unprotected("GET", "/api/v1/acp/runners/enrollments/abc/poll")
    assert _is_unprotected("GET", "/api/v1/acp/runners/info")
    assert _is_unprotected("DELETE", "/api/v1/acp/runners/self")
    assert not _is_unprotected("POST", "/api/v1/acp/runners/enrollments/lookup")
    assert not _is_unprotected("POST", "/api/v1/acp/runners/enrollments/abc/approve")
    assert not _is_unprotected("GET", "/api/v1/acp/runners")
    assert not _is_unprotected("GET", "/api/v1/acp/runners/self")
    assert not _is_unprotected("DELETE", "/api/v1/acp/runners/rnr-1")
