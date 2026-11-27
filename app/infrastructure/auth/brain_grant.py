"""Stateless, HMAC-signed capability grants for ``/mcp/brain``.

The sibling of :mod:`app.infrastructure.auth.datasource_grant`, for the shared
company memory (gbrain).  A spawned agent holds no credential of its own, so it
is *handed* a scope: "this run's agent, in this workflow, may read (or read and
write) the shared memory until this instant".  It is minted at spawn time by
``app.steps.agent_executor._build_agent_config`` from the effective access
level (the lower of the agent's and the workflow's ``brain_access``), travels
as the MCP entry's ``api_key``, and is verified on every request by
``app.api.app._BrainAuthWrapper``.

The upstream gbrain credential (``GBRAIN_API_KEY``) never leaves this process:
the grant names an access level, nothing else.  It cannot be revoked before
``expires_at``, so ``BRAIN_GRANT_TTL_SECONDS`` is the revocation window.

It fails closed exactly like the data source grant: no key, bad signature,
malformed payload, unknown version, unknown access level, or a missing/elapsed
expiry all yield ``None``.
"""
from __future__ import annotations

import logging
import time
from typing import Literal

from pydantic import BaseModel, Field

from app.infrastructure.auth._signed_token import (
    looks_like,
    sign_payload,
    verify_payload,
)

logger = logging.getLogger(__name__)

# Distinct from "dsg1" so a data source grant can never be presented as a brain
# grant (or the other way round), even when both are signed with the same key.
GRANT_PREFIX = "brg1"

DEFAULT_TTL_SECONDS = 24 * 60 * 60


class BrainGrant(BaseModel):
    """What one run's agent may do on the shared memory bridge.

    There is deliberately no ``"none"`` level: an agent without access gets no
    grant at all, rather than a token that authorizes nothing.
    """

    version: int = 1
    run_id: str = ""
    agent_id: str = ""
    workflow_id: str = ""
    access: Literal["read", "write"] = "read"
    # Extra slug prefixes (``a/b/`` form) a ``write`` grant may write under
    # verbatim, besides GBRAIN_WRITE_NAMESPACE: the workflow's
    # ``brain_write_prefixes`` intersected with GBRAIN_WRITE_PREFIXES_ALLOWED at
    # mint time. Absent in tokens minted before the field existed (= none).
    write_prefixes: list[str] = Field(default_factory=list)
    # Unix seconds; 0 ("no expiry stated") is rejected by verification.
    expires_at: int = 0

    @property
    def can_write(self) -> bool:
        return self.access == "write"


def looks_like_brain_grant(token: str) -> bool:
    return looks_like(token, GRANT_PREFIX)


def mint_brain_grant(
    grant: BrainGrant,
    signing_key: str | None,
    *,
    ttl_seconds: int | None = None,
    now: float | None = None,
) -> str | None:
    """Sign *grant* into a bearer token, or ``None`` when there is no key."""
    key = (signing_key or "").strip()
    if not key:
        logger.warning(
            "cannot mint a brain grant: none of BRAIN_GRANT_SIGNING_KEY, "
            "DATASOURCE_GRANT_SIGNING_KEY or MCP_DATASOURCES_API_KEY is set — "
            "the agent will get no shared memory access"
        )
        return None
    payload = grant.model_copy()
    if not payload.expires_at:
        ttl = DEFAULT_TTL_SECONDS if ttl_seconds is None else max(1, int(ttl_seconds))
        payload.expires_at = int((now if now is not None else time.time()) + ttl)
    return sign_payload(GRANT_PREFIX, payload.model_dump(mode="json"), key)


def verify_brain_grant(
    token: str,
    signing_key: str | None,
    *,
    now: float | None = None,
) -> BrainGrant | None:
    """Return the grant *token* carries, or ``None`` if it is not a valid one."""
    data = verify_payload(GRANT_PREFIX, token, signing_key, label="brain grant")
    if data is None:
        return None
    try:
        grant = BrainGrant.model_validate(data)
    except Exception:  # noqa: BLE001 — includes an unknown access level
        logger.warning("brain grant rejected: payload does not match the schema")
        return None
    if grant.version != 1:
        logger.warning("brain grant rejected: unsupported version %r", grant.version)
        return None
    if not grant.expires_at or grant.expires_at <= int(
        now if now is not None else time.time()
    ):
        logger.warning(
            "brain grant rejected: missing or elapsed expiry (run '%s')", grant.run_id
        )
        return None
    return grant
