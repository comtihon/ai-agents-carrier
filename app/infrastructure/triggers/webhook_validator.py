from __future__ import annotations

import hashlib
import hmac
from typing import Mapping

# Header carrying the signature unless the trigger names another
# (``signature_header`` on the http step).
DEFAULT_SIGNATURE_HEADER = "X-Webhook-Signature"


def validate_webhook_signature(body: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature for an incoming webhook request.

    The expected signature is HMAC-SHA256(key=secret, msg=body) as a hex digest,
    bare or prefixed ``sha256=``. Any other algorithm prefix is refused. Uses hmac.compare_digest for timing-safe comparison.
    """
    signature = signature.strip()
    if "=" in signature:
        algorithm, _, signature = signature.partition("=")
        if algorithm.strip().lower() != "sha256":
            return False
    computed = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature.strip().lower())


def valid_signature(body: bytes, headers: Mapping[str, str], secret: str, header: str | None = None) -> bool:
    """Whether *header* (default DEFAULT_SIGNATURE_HEADER) signs *body* with *secret*."""
    signature = headers.get(header or DEFAULT_SIGNATURE_HEADER) or ""
    return bool(signature) and validate_webhook_signature(body, signature, secret)


def valid_bearer(headers: Mapping[str, str], token: str) -> bool:
    if not token:
        return False
    # As before: the "Bearer " prefix is optional (some senders put the bare token).
    provided = headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return bool(provided) and hmac.compare_digest(provided, token)
