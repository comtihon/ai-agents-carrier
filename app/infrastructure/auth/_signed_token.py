"""Shared encoding and signing for the stateless capability grants.

``datasource_grant`` (``dsg1``) and ``brain_grant`` (``brg1``) are the same
token shape — ``<prefix>.<b64url(json)>.<b64url(hmac-sha256)>`` — carrying
different payloads.  The mechanics live here once so the two cannot drift: a
fix to signature comparison or payload decoding applies to both.

Everything here fails closed: :func:`verify_payload` returns ``None`` for any
token it cannot vouch for, and never raises on attacker-controlled input.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def b64u_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def signature(body: str, signing_key: str) -> str:
    """HMAC over the encoded body, not over re-serialized JSON.

    Signing the exact bytes that travel means verification never has to
    reproduce a canonical JSON form, so a difference in key order or separators
    can never turn a valid token into an invalid one (or vice versa).
    """
    digest = hmac.new(
        signing_key.encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).digest()
    return b64u_encode(digest)


def looks_like(token: str, prefix: str) -> bool:
    """Whether *token* has the ``<prefix>.<body>.<sig>`` shape, unverified."""
    return token.startswith(f"{prefix}.") and token.count(".") == 2


def sign_payload(prefix: str, payload: dict[str, Any], signing_key: str) -> str:
    """Encode and sign *payload* (a JSON-able dict) under *prefix*."""
    body = b64u_encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return f"{prefix}.{body}.{signature(body, signing_key)}"


def verify_payload(
    prefix: str, token: str, signing_key: str | None, *, label: str
) -> dict[str, Any] | None:
    """Return the decoded payload dict of a validly signed token, else ``None``.

    Checks shape, signature and that the payload is a JSON object.  Schema,
    version and expiry are the caller's business — they differ per grant kind.
    *label* names the grant kind in log lines.
    """
    key = (signing_key or "").strip()
    if not key or not looks_like(token, prefix):
        return None
    _, body, sig = token.split(".", 2)
    expected = signature(body, key)
    # Compare bytes: a token arrives from an HTTP header, which Starlette
    # latin-1 decodes, so `sig` can hold characters that would make
    # compare_digest on str raise TypeError (a 500) instead of failing.
    if not hmac.compare_digest(sig.encode("utf-8"), expected.encode("utf-8")):
        logger.warning("%s rejected: bad signature", label)
        return None
    try:
        data = json.loads(b64u_decode(body))
    except Exception:  # noqa: BLE001 — any malformed payload is simply invalid
        logger.warning("%s rejected: payload is not valid JSON", label)
        return None
    if not isinstance(data, dict):
        logger.warning("%s rejected: payload is not an object", label)
        return None
    return data
