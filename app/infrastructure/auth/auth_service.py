from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

logger = logging.getLogger(__name__)

# Userinfo response TTL before re-validating with the provider
_USERINFO_CACHE_TTL = timedelta(minutes=5)


class AuthError(Exception):
    def __init__(self, message: str = "Unauthenticated"):
        self.message = message
        super().__init__(message)


class AuthService:
    def __init__(
        self,
        jwks_url: str,
        issuer: str | None = None,
        algorithms: list[str] | None = None,
        audience: str | None = None,
    ):
        self.jwks_url = jwks_url
        self.issuer = issuer
        self.algorithms = algorithms or ["RS256"]
        self.audience = audience
        self._kid_cache: dict[str, dict] = {}
        self._last_updated: datetime | None = None

        # Cache for opaque-token userinfo responses: {token_hash: (claims, expires_at)}
        self._userinfo_cache: dict[str, tuple[dict, datetime]] = {}

        # Derive userinfo URL from issuer (Zitadel: <issuer>/oidc/v1/userinfo)
        self._userinfo_url: str | None = (
            issuer.rstrip("/") + "/oidc/v1/userinfo" if issuer else None
        )

    # ── JWKS helpers ─────────────────────────────────────────────────────────

    async def _fetch_jwks(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.jwks_url)
        if response.status_code != 200:
            logger.error("JWKS endpoint returned status %d", response.status_code)
            raise AuthError("Failed to fetch JWKS")
        for key in response.json()["keys"]:
            self._kid_cache[key["kid"]] = key
        self._last_updated = datetime.now(timezone.utc)

    async def _get_public_key(self, kid: str):
        if (
            self._last_updated is None
            or self._last_updated + timedelta(days=1) < datetime.now(timezone.utc)
        ):
            await self._fetch_jwks()

        key = self._kid_cache.get(kid)
        if not key:
            await self._fetch_jwks()
            key = self._kid_cache.get(kid)
            if not key:
                raise AuthError(f"No matching key found for KID: {kid}")

        return RSAAlgorithm.from_jwk(key)

    # ── Opaque-token fallback ─────────────────────────────────────────────────

    async def _validate_via_userinfo(self, token: str) -> dict:
        """Validate an opaque (non-JWT) token by calling the OIDC userinfo endpoint.

        Zitadel issues opaque access tokens by default when no API audience
        scope is requested.  The userinfo endpoint accepts them and returns
        user claims, effectively acting as token introspection.
        """
        if not self._userinfo_url:
            raise AuthError("Token is not a JWT and userinfo URL cannot be derived (issuer not set)")

        # Short-lived cache keyed on a hash of the token
        cache_key = hashlib.sha256(token.encode()).hexdigest()[:24]
        now = datetime.now(timezone.utc)
        if cache_key in self._userinfo_cache:
            claims, expires_at = self._userinfo_cache[cache_key]
            if now < expires_at:
                return claims

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                self._userinfo_url,
                headers={"Authorization": f"Bearer {token}"},
            )

        if resp.status_code in (401, 403):
            raise AuthError("Token rejected by userinfo endpoint")
        if resp.status_code != 200:
            raise AuthError(f"Userinfo endpoint returned unexpected status {resp.status_code}")

        claims = resp.json()
        self._userinfo_cache[cache_key] = (claims, now + _USERINFO_CACHE_TTL)
        logger.debug("Validated opaque token via userinfo for sub=%s", claims.get("sub", "?"))
        return claims

    # ── Main entry point ──────────────────────────────────────────────────────

    async def validate_token(self, token: str) -> dict:
        # Fast path: try standard JWT validation
        try:
            header = jwt.get_unverified_header(token)
        except jwt.DecodeError:
            # Not a JWT — Zitadel issued an opaque access token.
            # Fall back to userinfo endpoint introspection.
            logger.debug("Token is not a JWT; falling back to userinfo introspection")
            return await self._validate_via_userinfo(token)

        kid = header.get("kid")
        if not kid:
            raise AuthError("KID not found in token header")

        public_key = await self._get_public_key(kid)

        decode_kwargs: dict = {
            "algorithms": self.algorithms,
            "options": {"verify_aud": self.audience is not None},
        }
        if self.issuer:
            decode_kwargs["issuer"] = self.issuer
        if self.audience:
            decode_kwargs["audience"] = self.audience

        try:
            return jwt.decode(token, public_key, **decode_kwargs)
        except jwt.ExpiredSignatureError:
            raise AuthError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthError(f"Invalid token: {e}")
