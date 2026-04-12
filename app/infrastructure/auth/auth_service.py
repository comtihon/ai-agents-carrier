from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

logger = logging.getLogger(__name__)


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
            # KID not in cache — refresh once before giving up
            await self._fetch_jwks()
            key = self._kid_cache.get(kid)
            if not key:
                raise AuthError(f"No matching key found for KID: {kid}")

        return RSAAlgorithm.from_jwk(key)

    async def validate_token(self, token: str) -> dict:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.DecodeError:
            raise AuthError("Invalid JWT header")

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
