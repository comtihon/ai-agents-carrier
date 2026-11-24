from __future__ import annotations

import hashlib
import hmac
from typing import Mapping

# Where senders put an HMAC-SHA256 signature of the raw body, in the order
# checked: carrier's own header, gbrain's webhook notification sink, Jira
# system webhooks (and older GitHub-style senders), GitHub-style senders.
SIGNATURE_HEADERS: tuple[str, ...] = (
    "X-Webhook-Signature",
    "X-Gbrain-Signature",
    "X-Hub-Signature",
    "X-Hub-Signature-256",
)


def validate_webhook_signature(body: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature for an incoming webhook request.

    The expected signature is HMAC-SHA256(key=secret, msg=body) as a hex digest,
    bare or prefixed ``sha256=`` (gbrain, Jira, GitHub). Any other algorithm
    prefix is refused. Uses hmac.compare_digest for timing-safe comparison.
    """
    signature = signature.strip()
    if "=" in signature:
        algorithm, _, signature = signature.partition("=")
        if algorithm.strip().lower() != "sha256":
            return False
    computed = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature.strip().lower())


def signatures_from(headers: Mapping[str, str]) -> list[str]:
    """Every signature the request carries, from any of SIGNATURE_HEADERS."""
    return [headers[h] for h in SIGNATURE_HEADERS if headers.get(h)]


def valid_signature(body: bytes, headers: Mapping[str, str], secret: str) -> bool:
    return any(validate_webhook_signature(body, sig, secret) for sig in signatures_from(headers))


def valid_bearer(headers: Mapping[str, str], token: str) -> bool:
    if not token:
        return False
    # As before: the "Bearer " prefix is optional (some senders put the bare token).
    provided = headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return bool(provided) and hmac.compare_digest(provided, token)
