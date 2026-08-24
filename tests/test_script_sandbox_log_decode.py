"""Pod logs arrive as bytes; the result parser must still find the marker.

This cost three round-trips in production to find. The kubernetes client returns
`read_namespaced_pod_log` as bytes, and every `line.startswith(RESULT_MARKER)`
then compared bytes against str and was quietly False -- so a script that had run
perfectly was reported as "produced no result". `bytes.strip()` is truthy, so the
empty-output guard waved it through as well. Both halves are pinned here.
"""
from __future__ import annotations

import pytest

from app.infrastructure.orchestration.script_sandbox import (
    RESULT_MARKER,
    ScriptSandboxError,
    _as_text,
    _parse_result,
)

_PAYLOAD = '{"output": {"ok": true, "py": "3.12.14"}}'


def test_a_bytes_log_is_parsed_like_a_str_one():
    as_bytes = f"\n{RESULT_MARKER}{_PAYLOAD}\n".encode()

    assert _parse_result(as_bytes) == {"ok": True, "py": "3.12.14"}


def test_the_str_path_still_works():
    as_str = f"\n{RESULT_MARKER}{_PAYLOAD}\n"

    assert _parse_result(as_str) == {"ok": True, "py": "3.12.14"}


def test_script_output_before_the_marker_is_ignored():
    log = f"hello from the script\n{RESULT_MARKER}{_PAYLOAD}\n".encode()

    assert _parse_result(log) == {"ok": True, "py": "3.12.14"}


def test_a_genuinely_markerless_log_still_raises_and_shows_what_it_got():
    with pytest.raises(ScriptSandboxError, match="Traceback"):
        _parse_result(b"Traceback (most recent call last):\n  boom\n")


def test_undecodable_bytes_do_not_crash_the_parser():
    """A log with invalid UTF-8 must fail as "no marker", not as a decode error."""
    with pytest.raises(ScriptSandboxError):
        _parse_result(b"\xff\xfe not utf-8")


def test_a_bytes_repr_trapped_in_a_string_is_still_parsed():
    """The kubernetes client str()s the raw bytes when the declared response type
    is `str`, producing the *text* "b'...'". The real fix is to bypass that
    deserialization, but the marker is recoverable either way."""
    trapped = repr(f"\n{RESULT_MARKER}{_PAYLOAD}\n".encode())

    assert _parse_result(trapped) == {"ok": True, "py": "3.12.14"}


def test_real_output_beginning_with_b_quote_is_not_mangled():
    """The unwrap must be narrow: legitimate output may start with b'."""
    log = f"b'not really bytes'\n{RESULT_MARKER}{_PAYLOAD}\n"

    assert _parse_result(log) == {"ok": True, "py": "3.12.14"}


@pytest.mark.parametrize(
    "value,expected",
    [(b"abc", "abc"), ("abc", "abc"), (None, ""), (b"", ""), ("", "")],
)
def test_as_text_normalises_every_shape_the_client_returns(value, expected):
    assert _as_text(value) == expected
