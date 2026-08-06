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

from app.core.config import Settings, get_settings

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
    """Mints and caches the service's own OAuth2 access token.

    Thread-safety: a single :class:`asyncio.Lock` guards refreshes so that
    concurrent callers trigger at most one token request.
    """

    def __init__(self, settings: Settings, *, timeout_seconds: float = 30.0) -> None:
        self._settings = settings
        self._timeout_seconds = timeout_seconds
        self._token: str | None = None
        # Monotonic deadline after which the cached token must be refreshed.
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self._settings.service_auth_enabled)

    def validate_configuration(self) -> None:
        """Raise :class:`ServiceAuthError` when enabled but misconfigured.

        Called at startup so a broken deployment fails fast instead of at the
        first outbound call. A no-op while the feature is disabled.
        """
        if not self.enabled:
            return
        self._require_config()

    async def get_token(self) -> str:
        """Return a valid access token, refreshing it when near expiry."""
        cached = self._cached_token()
        if cached is not None:
            return cached
        async with self._lock:
            # Another caller may have refreshed while we waited for the lock.
            cached = self._cached_token()
            if cached is not None:
                return cached
            token, ttl = await self._request_token()
            self._token = token
            self._expires_at = time.monotonic() + ttl
            logger.info("obtained service access token (ttl=%.0fs)", ttl)
            return token

    async def get_auth_header(self) -> dict[str, str]:
        """Return the ``Authorization`` header carrying the service token."""
        return {"Authorization": f"Bearer {await self.get_token()}"}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cached_token(self) -> str | None:
        if self._token is None:
            return None
        if self._expires_at - _REFRESH_BUFFER_SECONDS <= time.monotonic():
            return None
        return self._token

    def _require_config(self) -> tuple[str, str, str, str]:
        """Return (token_url, client_id, private_key, audience) or raise."""
        settings = self._settings
        if not settings.service_auth_enabled:
            raise ServiceAuthError(
                "Service authentication is disabled — set SERVICE_AUTH_ENABLED=true "
                "to use service identity for outbound calls."
            )
        missing = [
            name
            for name, value in (
                ("SERVICE_AUTH_TOKEN_URL", settings.service_auth_token_url),
                ("SERVICE_AUTH_CLIENT_ID", settings.service_auth_client_id),
                ("SERVICE_AUTH_PRIVATE_KEY", settings.service_auth_private_key),
                ("SERVICE_AUTH_AUDIENCE", settings.service_auth_audience),
            )
            if not value
        ]
        if missing:
            raise ServiceAuthError(
                "Service authentication is enabled but incomplete — missing: "
                + ", ".join(missing)
            )
        private_key = settings.resolved_service_auth_private_key()
        assert private_key is not None  # guaranteed by the missing-check above
        return (
            str(settings.service_auth_token_url),
            str(settings.service_auth_client_id),
            private_key,
            str(settings.service_auth_audience),
        )

    def _build_assertion(self, client_id: str, private_key: str, audience: str) -> str:
        now = int(time.time())
        payload = {
            "iss": client_id,
            "sub": client_id,
            "aud": audience,
            "iat": now,
            "exp": now + _ASSERTION_TTL_SECONDS,
        }
        headers = {"alg": "RS256"}
        if self._settings.service_auth_key_id:
            headers["kid"] = self._settings.service_auth_key_id
        return jwt.encode(payload, private_key, algorithm="RS256", headers=headers)

    async def _request_token(self) -> tuple[str, float]:
        token_url, client_id, private_key, audience = self._require_config()
        try:
            assertion = self._build_assertion(client_id, private_key, audience)
        except Exception as exc:  # invalid / unreadable key material
            raise ServiceAuthError(
                "Failed to sign the client assertion — check that "
                f"SERVICE_AUTH_PRIVATE_KEY holds a valid PEM RSA private key: {exc}"
            ) from exc

        data = {
            "grant_type": JWT_BEARER_GRANT_TYPE,
            "assertion": assertion,
            "scope": self._settings.service_auth_scopes,
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
