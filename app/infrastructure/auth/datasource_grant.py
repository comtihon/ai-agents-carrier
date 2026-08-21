"""Stateless, HMAC-signed capability grants for ``/mcp/datasources``.

A spawned agent holds no cluster identity and no user credential — the existing
agent callbacks say so out loud (``app.api.middleware.auth``: "Callbacks posted
by spawned agent containers, which hold no user credentials").  So an agent
cannot be *authenticated* into a scope; it has to be *handed* one.

A grant is that hand-off: a signed statement of "this run's agent may call
these operations of these data sources, until this instant".  It is minted when
the agent is spawned (``app.steps.agent_executor._build_agent_config``),
travels as the MCP server entry's ``api_key`` → the agent's ``bearerToken``,
and is verified on every request by the ASGI gate in front of the mount
(``app.api.app._DatasourcesAuthWrapper``).

Three properties matter:

* **It is a capability, not a credential.**  It names operations; it carries no
  data-source secret.  ``DataSourceDefinition.auth`` is resolved inside this
  backend by ``app.infrastructure.datasources.executor.build_auth_headers`` and
  never leaves the process — including for ``service_identity`` sources, whose
  bearer is minted per request from ``SERVICE_AUTH_*`` and is the carrier's own
  identity.  That is exactly why the allow-list has to be enforced server-side:
  a bypass would be a privilege escalation to the backend's own identity, not
  merely "read a source you weren't given".
* **It is stateless.**  No store to read on the hot path, it survives a backend
  restart mid-run, and an off-cluster Docker agent needs nothing but the token.
  The cost is that it cannot be revoked before ``expires_at`` — terminating a
  run does not instantly kill its grant — so the TTL is the revocation window.
* **It fails closed.**  No signing key, a bad signature, a malformed payload, an
  unknown version, or a missing/elapsed expiry all yield ``None``, which every
  caller treats as "not authorized".  An empty operation list grants nothing.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Version-tagged prefix.  Two jobs: the auth gate can tell a grant apart from
# the static MCP_DATASOURCES_API_KEY without attempting a signature check, and
# the payload format can change later without a token of one shape ever being
# read as the other.
GRANT_PREFIX = "dsg1"

# How long a minted grant stays valid when the caller names no lifetime.  Long
# enough for a slow agent run, short enough to bound the un-revocable window.
DEFAULT_TTL_SECONDS = 24 * 60 * 60


class DatasourceGrant(BaseModel):
    """What one run's agent may do on the data sources bridge.

    ``grants`` maps ``source_id`` → the operation names of that source the
    agent may invoke.  A source that is absent, or present with an empty list,
    grants nothing: this mirrors the addon model, where an empty
    ``allowed_operations`` is a deny and not a wildcard.
    """

    version: int = 1
    # Recorded so every tool call can be attributed to a run in the logs.
    run_id: str = ""
    agent_id: str = ""
    grants: dict[str, list[str]] = Field(default_factory=dict)
    # Unix seconds.  0 means "no expiry stated", which verification rejects —
    # an unbounded grant is never what anybody meant to mint.
    expires_at: int = 0

    def is_empty(self) -> bool:
        """True when the grant authorizes no operation at all."""
        return not any(self.grants.values())


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64u_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signature(body: str, signing_key: str) -> str:
    """HMAC over the encoded body, not over re-serialized JSON.

    Signing the exact bytes that travel means verification never has to
    reproduce a canonical JSON form, so a difference in key order or separators
    can never turn a valid token into an invalid one (or vice versa).
    """
    digest = hmac.new(
        signing_key.encode("utf-8"), body.encode("ascii"), hashlib.sha256
    ).digest()
    return _b64u_encode(digest)


def looks_like_grant(token: str) -> bool:
    """Whether *token* is shaped like a grant, without verifying it.

    Lets the auth gate decide which check to run without spending a signature
    computation on a token that is plainly the static API key instead.
    """
    return token.startswith(f"{GRANT_PREFIX}.") and token.count(".") == 2


def mint_grant(
    grant: DatasourceGrant,
    signing_key: str | None,
    *,
    ttl_seconds: int | None = None,
    now: float | None = None,
) -> str | None:
    """Sign *grant* into a bearer token, or ``None`` when it cannot be signed.

    ``None`` is returned — rather than an unsigned token — when there is no
    signing key, so a deployment that never configured one hands the agent
    nothing instead of something the gate will reject anyway.

    ``expires_at`` is filled in from *ttl_seconds* when the grant does not
    already carry one; an explicit value on the grant wins.
    """
    key = (signing_key or "").strip()
    if not key:
        logger.warning(
            "cannot mint a data source grant: neither DATASOURCE_GRANT_SIGNING_KEY "
            "nor MCP_DATASOURCES_API_KEY is set — the agent will get no data "
            "source access at all"
        )
        return None

    payload = grant.model_copy()
    if not payload.expires_at:
        ttl = DEFAULT_TTL_SECONDS if ttl_seconds is None else max(1, int(ttl_seconds))
        payload.expires_at = int((now if now is not None else time.time()) + ttl)

    body = _b64u_encode(
        json.dumps(
            payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return f"{GRANT_PREFIX}.{body}.{_signature(body, key)}"


def verify_grant(
    token: str,
    signing_key: str | None,
    *,
    now: float | None = None,
) -> DatasourceGrant | None:
    """Return the grant *token* carries, or ``None`` if it is not a valid one.

    Every failure mode collapses to ``None`` on purpose: the caller's only
    correct response to "no verified grant" is to refuse, and distinguishing
    "tampered" from "expired" in the response would only help an attacker
    probe.  The reason is logged instead.
    """
    key = (signing_key or "").strip()
    if not key:
        return None
    if not looks_like_grant(token):
        return None

    _, body, signature = token.split(".", 2)
    expected = _signature(body, key)
    # Compare bytes: a token arrives from an HTTP header, which Starlette
    # latin-1 decodes, so `signature` can hold characters that would make
    # compare_digest on str raise TypeError (a 500) instead of failing.
    if not hmac.compare_digest(
        signature.encode("utf-8"), expected.encode("utf-8")
    ):
        logger.warning("data source grant rejected: bad signature")
        return None

    try:
        data = json.loads(_b64u_decode(body))
    except Exception:  # noqa: BLE001 — any malformed payload is simply invalid
        logger.warning("data source grant rejected: payload is not valid JSON")
        return None
    if not isinstance(data, dict):
        logger.warning("data source grant rejected: payload is not an object")
        return None

    try:
        grant = DatasourceGrant.model_validate(data)
    except Exception:  # noqa: BLE001 — a payload we cannot model is invalid
        logger.warning("data source grant rejected: payload does not match the schema")
        return None

    if grant.version != 1:
        logger.warning(
            "data source grant rejected: unsupported version %r", grant.version
        )
        return None
    # A grant with no stated expiry is rejected rather than treated as eternal.
    if not grant.expires_at or grant.expires_at <= int(
        now if now is not None else time.time()
    ):
        logger.warning(
            "data source grant rejected: missing or elapsed expiry (run '%s')",
            grant.run_id,
        )
        return None
    return grant
