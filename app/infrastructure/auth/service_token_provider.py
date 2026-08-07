"""Outbound service identity via the OAuth2 JWT bearer grant (RFC 7523).

The backend authenticates itself to an OAuth2 authorization server with a
signed client assertion (``urn:ietf:params:oauth:grant-type:jwt-bearer``) and
receives an access token it can attach to outbound calls as
``Authorization: Bearer <token>``.

Everything provider-specific — token endpoint, audience, scopes, key material —
arrives through :class:`app.core.config.Settings` at deploy time. Scope strings
are passed through verbatim, so deployments may use plain scopes or
provider-specific scope URNs without code changes.
"""
from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache

import httpx
import jwt

from app.core.config import (
    DEFAULT_SERVICE_IDENTITY,
    ServiceIdentityConfig,
    Settings,
    get_settings,
)

logger = logging.getLogger(__name__)

# RFC 7523 grant type for authenticating with a signed JWT assertion.
JWT_BEARER_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"

# Lifetime of the client assertion we sign (short lived — it is single use).
_ASSERTION_TTL_SECONDS = 300
# Refresh the cached access token this long before it actually expires.
_REFRESH_BUFFER_SECONDS = 60
# Used when the token response carries neither `expires_in` nor a decodable exp.
_DEFAULT_TOKEN_TTL_SECONDS = 3600


class ServiceAuthError(Exception):
    """Raised when a service access token cannot be obtained."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ServiceTokenProvider:
    """Mints and caches the service's own OAuth2 access tokens.

    A deployment may configure several identities (see
    :meth:`app.core.config.Settings.get_service_identities`); tokens are cached
    and refreshed per identity, never shared between them. Each identity gets
    its own :class:`asyncio.Lock`, so concurrent callers trigger at most one
    token request per identity and a slow authorization server for one does not
    block calls using another.
    """

    def __init__(self, settings: Settings, *, timeout_seconds: float = 30.0) -> None:
        self._settings = settings
        self._timeout_seconds = timeout_seconds
        # identity name → (token, monotonic deadline after which to refresh)
        self._tokens: dict[str, tuple[str, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self._settings.service_auth_enabled)

    def validate_configuration(self) -> None:
        """Raise :class:`ServiceAuthError` when enabled but misconfigured.

        Called at startup so a broken deployment fails fast instead of at the
        first outbound call — every configured identity is checked, not just the
        default. A no-op while the feature is disabled.
        """
        if not self.enabled:
            return
        try:
            identities = self._settings.get_service_identities()
        except ValueError as exc:
            raise ServiceAuthError(f"SERVICE_AUTH_IDENTITIES is not valid: {exc}") from exc
        if not identities:
            raise ServiceAuthError(
                "Service authentication is enabled but no identity is configured — "
                "set SERVICE_AUTH_IDENTITIES, or the SERVICE_AUTH_TOKEN_URL / "
                "_CLIENT_ID / _PRIVATE_KEY / _AUDIENCE set."
            )
        for identity in identities:
            self._require_complete(identity)

    def describe(self, name: str | None = None) -> tuple[ServiceIdentityConfig | None, str | None]:
        """Return ``(identity, error)`` for *name* without minting a token.

        ``identity`` is None when no such identity is configured. ``error``
        explains why it is unusable, or is None when it is ready to use.
        """
        if not self.enabled:
            return None, "SERVICE_AUTH_ENABLED is not set on this backend"
        try:
            identity = self._settings.get_service_identity(name)
        except ValueError as exc:
            return None, f"SERVICE_AUTH_IDENTITIES is not valid: {exc}"
        if identity is None:
            return None, f"No service identity named '{name}' is configured"
        try:
            self._require_complete(identity)
        except ServiceAuthError as exc:
            return identity, exc.message
        return identity, None

    async def get_token(self, identity: str | None = None) -> str:
        """Return a valid access token for *identity*, refreshing near expiry."""
        config = self._resolve_identity(identity)
        cached = self._cached_token(config.name)
        if cached is not None:
            return cached
        lock = self._locks.setdefault(config.name, asyncio.Lock())
        async with lock:
            # Another caller may have refreshed while we waited for the lock.
            cached = self._cached_token(config.name)
            if cached is not None:
                return cached
            token, ttl = await self._request_token(config)
            self._tokens[config.name] = (token, time.monotonic() + ttl)
            logger.info(
                "obtained service access token for identity '%s' (ttl=%.0fs)",
                config.name,
                ttl,
            )
            return token

    async def get_auth_header(self, identity: str | None = None) -> dict[str, str]:
        """Return the ``Authorization`` header carrying the service token."""
        return {"Authorization": f"Bearer {await self.get_token(identity)}"}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cached_token(self, name: str) -> str | None:
        entry = self._tokens.get(name)
        if entry is None:
            return None
        token, expires_at = entry
        if expires_at - _REFRESH_BUFFER_SECONDS <= time.monotonic():
            return None
        return token

    def _resolve_identity(self, name: str | None) -> ServiceIdentityConfig:
        """Return the requested identity's config, or raise with the reason."""
        if not self._settings.service_auth_enabled:
            raise ServiceAuthError(
                "Service authentication is disabled — set SERVICE_AUTH_ENABLED=true "
                "to use service identity for outbound calls."
            )
        try:
            identity = self._settings.get_service_identity(name)
        except ValueError as exc:
            raise ServiceAuthError(f"SERVICE_AUTH_IDENTITIES is not valid: {exc}") from exc
        if identity is None:
            configured = [i.name for i in self._settings.get_service_identities()]
            if name:
                raise ServiceAuthError(
                    f"No service identity named '{name}' is configured"
                    + (f" — available: {', '.join(configured)}" if configured else "")
                )
            if len(configured) > 1:
                raise ServiceAuthError(
                    "Several service identities are configured and none is the "
                    f"default — name one of {', '.join(configured)} on the call, or "
                    "set SERVICE_AUTH_DEFAULT_IDENTITY."
                )
            raise ServiceAuthError(
                "Service authentication is enabled but no identity is configured"
            )
        self._require_complete(identity)
        return identity

    def _require_complete(self, identity: ServiceIdentityConfig) -> None:
        """Raise when *identity* is missing a field needed to mint a token."""
        missing = [
            self._field_label(identity, name)
            for name, value in (
                ("token_url", identity.token_url),
                ("client_id", identity.client_id),
                ("audience", identity.audience),
                ("private_key", identity.resolved_private_key()),
            )
            if not value
        ]
        if missing:
            raise ServiceAuthError(
                f"Service identity '{identity.name}' is incomplete — missing: "
                + ", ".join(missing)
            )

    @staticmethod
    def _field_label(identity: ServiceIdentityConfig, field: str) -> str:
        """Name a field the way the operator configured it.

        The default identity may come from the flat ``SERVICE_AUTH_*`` env vars,
        where the env var name is what someone needs in order to fix it; named
        identities come from a JSON object, where the field name is.
        """
        if identity.name == DEFAULT_SERVICE_IDENTITY:
            return f"{field} (SERVICE_AUTH_{field.upper()})"
        return field

    def _build_assertion(self, identity: ServiceIdentityConfig) -> str:
        now = int(time.time())
        payload = {
            "iss": identity.client_id,
            "sub": identity.client_id,
            "aud": identity.audience,
            "iat": now,
            "exp": now + _ASSERTION_TTL_SECONDS,
        }
        headers = {"alg": "RS256"}
        if identity.key_id:
            headers["kid"] = identity.key_id
        private_key = identity.resolved_private_key()
        return jwt.encode(payload, private_key, algorithm="RS256", headers=headers)

    async def _request_token(self, identity: ServiceIdentityConfig) -> tuple[str, float]:
        token_url = identity.token_url
        try:
            assertion = self._build_assertion(identity)
        except Exception as exc:  # invalid / unreadable key material
            raise ServiceAuthError(
                f"Failed to sign the client assertion for identity "
                f"'{identity.name}' — check that "
                f"{self._field_label(identity, 'private_key')} holds a valid PEM "
                f"RSA private key: {exc}"
            ) from exc

        data = {
            "grant_type": JWT_BEARER_GRANT_TYPE,
            "assertion": assertion,
            "scope": identity.scopes,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(
                    token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ServiceAuthError(
                f"Service token request rejected by {token_url} "
                f"(status={exc.response.status_code}): {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceAuthError(
                f"Service token request to {token_url} failed: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ServiceAuthError(
                f"Service token endpoint {token_url} returned a non-JSON response"
            ) from exc
        token = payload.get("access_token")
        if not token or not isinstance(token, str):
            raise ServiceAuthError(
                f"Service token endpoint {token_url} returned no access_token"
            )
        return token, _token_ttl_seconds(payload, token)


def _token_ttl_seconds(payload: dict, token: str) -> float:
    """Derive the token lifetime from `expires_in`, else from its `exp` claim."""
    expires_in = payload.get("expires_in")
    if isinstance(expires_in, bool):
        expires_in = None
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        return float(expires_in)
    if isinstance(expires_in, str):
        try:
            parsed = float(expires_in)
        except ValueError:
            parsed = 0.0
        if parsed > 0:
            return parsed
    try:
        claims = jwt.decode(token, options={"verify_signature": False})
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            remaining = float(exp) - time.time()
            if remaining > 0:
                return remaining
    except Exception:  # noqa: BLE001 — opaque tokens are not decodable
        logger.debug("service token carries no readable exp claim", exc_info=True)
    return float(_DEFAULT_TOKEN_TTL_SECONDS)


@lru_cache(maxsize=1)
def get_service_token_provider() -> ServiceTokenProvider:
    """Process-wide provider built from the settings singleton.

    Used as a fallback where explicit dependency injection is not available;
    the application container injects its own instance everywhere else.
    """
    return ServiceTokenProvider(get_settings())
