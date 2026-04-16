from __future__ import annotations

import hashlib
import hmac


def validate_webhook_signature(body: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature for an incoming webhook request.

    The expected signature is HMAC-SHA256(key=secret, msg=body) as a hex digest.
    Uses hmac.compare_digest for timing-safe comparison.
    """
    computed = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature)
