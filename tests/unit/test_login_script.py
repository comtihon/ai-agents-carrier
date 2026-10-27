"""The login script engine: loading, running, and validating what it returns."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.infrastructure.auth.login_script import (
    KEEP,
    LoginDecision,
    LoginScript,
    LoginScriptConfigError,
    LoginScriptError,
    TeamAssignment,
    load_login_script,
    parse_decision,
)

CTX = {
    "claims": {"sub": "u1", "groups": ["eng", "ops"], "email": "u1@example.com"},
    "user": None,
    "teams": [{"id": "t1", "name": "Eng"}],
}


def _run(body: str, ctx: dict | None = None, **kw) -> LoginDecision:
    return LoginScript(f"function onLogin(ctx) {{ {body} }}", **kw).run(ctx or CTX)


# ── Success ───────────────────────────────────────────────────────────────────

def test_maps_claims_to_roles_profile_and_team() -> None:
    decision = _run(
        """
        return {
          roles: ctx.claims.groups.map(g => "carrier-" + g),
          name: "User One",
          email: ctx.claims.email,
          team: { name: "Eng", role: "admin" },
        };
        """
    )
    assert decision == LoginDecision(
        roles=frozenset({"carrier-eng", "carrier-ops"}),
        name="User One",
        email="u1@example.com",
        team=TeamAssignment(role="admin", name="Eng"),
    )


def test_sees_teams_and_stored_user() -> None:
    ctx = dict(CTX, user={"sub": "u1", "name": "Old", "email": None,
                          "team": {"id": "t1", "name": "Eng", "role": "member"}})
    decision = _run("return { team: { id: ctx.teams[0].id }, name: ctx.user.team.name };", ctx)
    assert decision.team == TeamAssignment(role="member", id="t1")
    assert decision.name == "Eng"


@pytest.mark.parametrize("body", ["return;", "return null;", "return {};", ""])
def test_nothing_returned_changes_nothing(body: str) -> None:
    assert _run(body) == LoginDecision(roles=None, name=None, email=None, team=KEEP)


def test_team_null_means_no_team() -> None:
    assert _run("return { team: null };").team is None


def test_empty_roles_is_an_explicit_answer() -> None:
    assert _run("return { roles: [] };").roles == frozenset()


def test_each_run_gets_a_fresh_isolate() -> None:
    script = LoginScript(
        "var seen = 0; function onLogin(ctx) { seen += 1; return { name: 'run ' + seen }; }"
    )
    assert script.run(CTX).name == "run 1"
    assert script.run(CTX).name == "run 1"


async def test_async_run() -> None:
    script = LoginScript("function onLogin(ctx) { return { roles: ['a'] }; }")
    assert (await script.arun(CTX)).roles == frozenset({"a"})


def test_has_no_network_filesystem_or_process_access() -> None:
    decision = _run(
        "return { roles: [typeof fetch, typeof require, typeof process, typeof XMLHttpRequest] };"
    )
    assert decision.roles == frozenset({"undefined"})


# ── Errors ────────────────────────────────────────────────────────────────────

def test_a_throwing_script_is_an_error() -> None:
    with pytest.raises(LoginScriptError, match="boom"):
        _run("throw new Error('boom');")


def test_a_runaway_script_times_out() -> None:
    with pytest.raises(LoginScriptError, match="timed out"):
        _run("while (true) {}", timeout_ms=100)


def test_a_memory_hog_is_stopped() -> None:
    with pytest.raises(LoginScriptError):
        _run(
            "const a = []; for (;;) { a.push(new Array(1e6).fill(1)); }",
            timeout_ms=10_000,
            max_memory_bytes=32 * 1024 * 1024,
        )


def test_a_promise_is_refused() -> None:
    with pytest.raises(LoginScriptError, match="promise"):
        _run("return Promise.resolve({ roles: ['a'] });")


# ── Invalid output ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "result, message",
    [
        ("['a']", "must return an object"),
        ("'admin'", "must return an object"),
        ("{ role: ['a'] }", "unknown field"),
        ("{ roles: 'admin' }", "array of strings"),
        ("{ roles: ['a', 1] }", "must be a string"),
        ("{ roles: [' '] }", "must not be empty"),
        ("{ name: 42 }", "must be a string"),
        ("{ email: '' }", "must not be empty"),
        ("{ team: 'Eng' }", "object or null"),
        ("{ team: {} }", "exactly one"),
        ("{ team: { id: 'x', name: 'y' } }", "exactly one"),
        ("{ team: { name: 'Eng', role: 'owner' } }", "team.role"),
        ("{ team: { name: 'Eng', extra: 1 } }", "unknown field"),
        ("{ name: 'x'.repeat(10000) }", "longer than"),
    ],
)
def test_invalid_output_is_rejected_whole(result: str, message: str) -> None:
    with pytest.raises(LoginScriptError, match=message):
        _run(f"return {result};")


def test_parse_decision_null_fields_mean_keep() -> None:
    assert parse_decision({"roles": None, "name": None, "email": None}) == LoginDecision()


# ── Loading ───────────────────────────────────────────────────────────────────

def _settings(**kw) -> SimpleNamespace:
    base = dict(auth_login_script=None, auth_login_script_path=None,
                auth_login_script_timeout_ms=1000)
    return SimpleNamespace(**(base | kw))


def test_no_script_configured() -> None:
    assert load_login_script(_settings()) is None
    assert load_login_script(_settings(auth_login_script="  ")) is None


def test_inline_script() -> None:
    script = load_login_script(
        _settings(auth_login_script="function onLogin() { return {}; }",
                  auth_login_script_timeout_ms=250)
    )
    assert script is not None and script.timeout_ms == 250


def test_script_from_file(tmp_path) -> None:
    path = tmp_path / "login.js"
    path.write_text("function onLogin(ctx) { return { roles: ['x'] }; }")
    script = load_login_script(_settings(auth_login_script_path=str(path)))
    assert script.run(CTX).roles == frozenset({"x"})


def test_both_sources_is_a_config_error(tmp_path) -> None:
    path = tmp_path / "login.js"
    path.write_text("function onLogin() {}")
    with pytest.raises(LoginScriptConfigError, match="not both"):
        load_login_script(
            _settings(auth_login_script="function onLogin() {}", auth_login_script_path=str(path))
        )


def test_missing_file_is_a_config_error(tmp_path) -> None:
    with pytest.raises(LoginScriptConfigError, match="Cannot read"):
        load_login_script(_settings(auth_login_script_path=str(tmp_path / "nope.js")))


@pytest.mark.parametrize(
    "source", ["function onlogin() {}", "var onLogin = 1;", "function onLogin( {", "throw 1;"]
)
def test_unusable_script_is_a_config_error(source: str) -> None:
    with pytest.raises(LoginScriptConfigError):
        LoginScript(source)
