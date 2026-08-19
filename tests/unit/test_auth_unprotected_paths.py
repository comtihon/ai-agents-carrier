"""Which (method, path) pairs may bypass user authentication.

The prod ingress serves "/", so every route here is reachable from the internet.
A route belongs in the unprotected set only if it authenticates itself some other
way (Slack HMAC, webhook HMAC, a dedicated bearer wrapper) or genuinely cannot
carry a user token (a spawned agent container, a browser link click).

These are table tests rather than request tests on purpose: the predicate is the
security boundary, and asserting on it directly means a future edit to the
exemption lists cannot quietly widen the boundary without failing here.
"""
from __future__ import annotations

import pytest

from app.api.middleware.auth import _is_unprotected


# ── Must bypass auth: they authenticate themselves, or hold no user token ──────

@pytest.mark.parametrize(
    ("method", "path"),
    [
        # Liveness/readiness probes.
        ("GET", "/health"),
        ("GET", "/ready"),
        # Slack — verified by X-Slack-Signature HMAC in the route.
        ("POST", "/api/v1/callbacks/slack/interactive"),
        ("POST", "/api/v1/callbacks/slack/events"),
        # Webhook triggers — verified by X-Webhook-Signature HMAC over the body.
        ("POST", "/api/v1/webhooks/my-workflow"),
        # MCP mounts — guarded by their own bearer wrappers in app.api.app.
        ("POST", "/mcp/datasources"),
        ("POST", "/mcp/management"),
        ("POST", "/mcp/management/"),
        # Slack button URLs opened in a browser: a click sends no Authorization
        # header, so the GET forms must stay open.
        ("GET", "/api/v1/callbacks/abc-123/approve"),
        ("GET", "/api/v1/callbacks/abc-123/reject"),
        # Callbacks from spawned agent containers, which have no user credentials.
        ("POST", "/api/v1/runs/abc-123/agent/output"),
        ("POST", "/api/v1/runs/abc-123/agent/question"),
        ("GET", "/api/v1/runs/abc-123/agent/input"),
        ("POST", "/api/v1/runs/abc-123/agent/progress"),
    ],
)
def test_unprotected(method: str, path: str) -> None:
    assert _is_unprotected(method, path) is True


# ── Must require auth ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("method", "path"),
    [
        # Run-control API. These used to be swallowed by a blanket
        # "/api/v1/runs/" exemption, leaving them internet-reachable with a
        # run_id as the only credential — and run ids leak into pod names, helm
        # release names, Slack messages, logs, and every agent's own RUN_ID env
        # var, so an agent could approve its own human-approval gate.
        ("GET", "/api/v1/runs/abc-123"),
        ("GET", "/api/v1/runs/abc-123/trace"),
        ("POST", "/api/v1/runs/abc-123/approve"),
        ("POST", "/api/v1/runs/abc-123/reject"),
        ("POST", "/api/v1/runs/abc-123/terminate"),
        ("POST", "/api/v1/runs/abc-123/retry"),
        ("POST", "/api/v1/runs/abc-123/restart-from-step"),
        ("DELETE", "/api/v1/runs/abc-123"),
        # Called by the frontend, which does attach a bearer token.
        ("POST", "/api/v1/runs/abc-123/agent/reply"),
        # POST approve/reject are API calls, not browser link clicks.
        ("POST", "/api/v1/callbacks/abc-123/approve"),
        ("POST", "/api/v1/callbacks/abc-123/reject"),
        # CopilotKit runtime: unauthenticated LLM access on the org's API keys.
        ("POST", "/copilotkit"),
        ("POST", "/copilotkit/info"),
        # Ordinary API surface.
        ("GET", "/api/v1/runs"),
        ("POST", "/api/v1/runs"),
        ("GET", "/api/v1/workflows"),
        ("PUT", "/api/v1/agents/researcher-fast"),
    ],
)
def test_protected(method: str, path: str) -> None:
    assert _is_unprotected(method, path) is False


# ── Boundary cases ───────────────────────────────────────────────────────────

def test_agent_callback_pattern_is_anchored_not_a_prefix() -> None:
    """A deeper or decorated path must not inherit the agent-callback exemption."""
    assert _is_unprotected("POST", "/api/v1/runs/abc/agent/output/extra") is False
    assert _is_unprotected("POST", "/api/v1/runs/abc/agent/outputx") is False
    assert _is_unprotected("POST", "/api/v1/runs/abc/agent/") is False
    assert _is_unprotected("POST", "/api/v1/runs/abc/agent") is False


def test_agent_callback_run_id_segment_cannot_span_slashes() -> None:
    """`[^/]+` must not let an attacker walk into another route."""
    assert _is_unprotected("POST", "/api/v1/runs/a/b/agent/output") is False


def test_sibling_prefixes_are_not_exempted() -> None:
    """"/"-terminated prefixes must not match a longer sibling name."""
    assert _is_unprotected("POST", "/mcp/managementfoo") is False
    assert _is_unprotected("POST", "/api/v1/webhooksfoo") is False
    assert _is_unprotected("POST", "/api/v1/callbacks/slackfoo/x") is False


def test_method_exemption_does_not_leak_to_other_methods() -> None:
    """The GET exemption on callbacks must not cover mutating methods."""
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert _is_unprotected(method, "/api/v1/callbacks/abc/approve") is False


def test_health_is_exact_not_a_prefix() -> None:
    assert _is_unprotected("GET", "/healthz") is False
    assert _is_unprotected("GET", "/health/detail") is False
