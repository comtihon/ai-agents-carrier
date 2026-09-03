"""Outbound Google Workspace tokens via service account impersonation.

Why impersonation and not the ambient credential
------------------------------------------------
On GKE the backend already holds a Google identity through Workload Identity
(``langgraph-backend@…``, no key file).  That identity cannot be used for
Sheets/Drive directly: the token the metadata server issues is
``https://www.googleapis.com/auth/cloud-platform``-scoped, and the Workspace
APIs reject a token that does not carry one of *their* scopes.  There is no way
to ask the metadata server for a different scope set.

So the backend calls ``generateAccessToken`` on a second, keyless service
account it holds ``roles/iam.serviceAccountTokenCreator`` on, and states
``target_scopes`` explicitly.  That second account is the one documents are
shared with, which is what makes access per-document, auditable and revocable
by the document owner.  No credential file exists anywhere in this path.

Which principal may be impersonated
-----------------------------------
Exactly one: ``GOOGLE_IMPERSONATE_SA``.  The target principal arrives inside a
data source definition, and a data source definition is written by an API
caller — so if the name were taken at face value, anyone who can create a data
source could name any service account this backend can impersonate and borrow
its authority.  :func:`resolve_impersonate_subject` therefore ignores the
stored/incoming value unless it equals the configured one, and
:func:`check_impersonate_subject` gives the write paths a message to refuse
with instead of a silent substitution.

Token caching
-------------
``build_auth_headers`` runs once per outbound request and a Google access token
lives about an hour, so a token is cached per
``(target_principal, sorted(scopes))`` and refreshed shortly before expiry —
the same shape as
:class:`app.infrastructure.auth.service_token_provider.ServiceTokenProvider`,
with one :class:`asyncio.Lock` per key so concurrent callers mint at most one.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Refresh this long before the token's own expiry. Google access tokens are
# issued with a ~3600s lifetime, so this lands at ~55 minutes.
_REFRESH_BUFFER_SECONDS = 300
# Used when the minted credential reports no expiry at all.
_DEFAULT_TOKEN_TTL_SECONDS = 3600
# generateAccessToken cannot exceed an hour without an org policy allowing it.
_TOKEN_LIFETIME_SECONDS = 3600

# Scopes a Google Sheets data source needs: the spreadsheet itself, plus
# drive.file so the resolve endpoint can read the metadata of a document that
# was shared with the impersonated account.
SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]


class GoogleAuthError(Exception):
    """Raised when a Google access token cannot be obtained."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


# (target_principal, tuple(sorted(scopes))) → (token, monotonic deadline)
_tokens: dict[tuple[str, tuple[str, ...]], tuple[str, float]] = {}
_locks: dict[tuple[str, tuple[str, ...]], asyncio.Lock] = {}


def configured_subject(settings: Settings | None = None) -> str:
    """The one service account this backend will impersonate, or ``""``."""
    resolved = settings or get_settings()
    return (resolved.google_impersonate_sa or "").strip()


def configured_scopes(settings: Settings | None = None) -> list[str]:
    resolved = settings or get_settings()
    return [s for s in (resolved.google_impersonate_scopes or []) if s] or list(SHEETS_SCOPES)


def check_impersonate_subject(auth: Any, settings: Settings | None = None) -> str | None:
    """Return why *auth* names an unusable principal, or ``None`` when it is fine.

    Called by every write path (REST create/update/try-operation and the
    management MCP tools) so a caller gets told, rather than having its value
    quietly swapped for the configured one.
    """
    if _auth_type(auth) != "google":
        return None
    configured = configured_subject(settings)
    if not configured:
        return (
            "Google auth is not configured on this backend — set "
            "GOOGLE_IMPERSONATE_SA to the service account documents are "
            "shared with."
        )
    requested = (_auth_field(auth, "impersonate_subject") or "").strip()
    if requested and requested != configured:
        return (
            f"auth.impersonate_subject must be '{configured}' (or omitted) — "
            "this backend impersonates only the service account named by "
            "GOOGLE_IMPERSONATE_SA."
        )
    return None


def resolve_impersonate_subject(auth: Any, settings: Settings | None = None) -> str:
    """The principal to impersonate for *auth*, taken from settings.

    Deliberately does not trust the auth block: a value that disagrees with the
    configured one is dropped with a warning rather than used, so a definition
    written before the check existed (or straight into Mongo) still cannot
    reach another service account.
    """
    configured = configured_subject(settings)
    if not configured:
        raise GoogleAuthError(
            "Google auth is not configured on this backend — set "
            "GOOGLE_IMPERSONATE_SA to the service account documents are "
            "shared with."
        )
    requested = (_auth_field(auth, "impersonate_subject") or "").strip()
    if requested and requested != configured:
        logger.warning(
            "google auth block names impersonate_subject '%s', which is not the "
            "configured GOOGLE_IMPERSONATE_SA ('%s') — using the configured one",
            requested, configured,
        )
    return configured


def resolve_scopes(auth: Any, settings: Settings | None = None) -> list[str]:
    """Scopes to mint with: the auth block's, else the deployment default.

    An auth block may narrow the set; it cannot widen it past what the
    deployment allows, because a scope outside the configured list is dropped.
    """
    allowed = configured_scopes(settings)
    requested = [s for s in (_auth_field(auth, "scopes") or []) if s]
    if not requested:
        return allowed
    narrowed = [s for s in requested if s in allowed]
    dropped = [s for s in requested if s not in allowed]
    if dropped:
        logger.warning(
            "google auth block asks for scope(s) %s outside "
            "GOOGLE_IMPERSONATE_SCOPES — ignoring them",
            ", ".join(dropped),
        )
    return narrowed or allowed


async def get_google_token(auth: Any, settings: Settings | None = None) -> str:
    """A valid access token for *auth*, minted by impersonation and cached."""
    subject = resolve_impersonate_subject(auth, settings)
    scopes = resolve_scopes(auth, settings)
    key = (subject, tuple(sorted(scopes)))

    cached = _cached_token(key)
    if cached is not None:
        return cached
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        # Another caller may have refreshed while we waited for the lock.
        cached = _cached_token(key)
        if cached is not None:
            return cached
        token, ttl = await asyncio.to_thread(_mint_token, subject, scopes)
        _tokens[key] = (token, time.monotonic() + ttl)
        logger.info(
            "minted Google access token for '%s' (ttl=%.0fs, scopes=%s)",
            subject, ttl, ", ".join(scopes),
        )
        return token


async def get_google_auth_header(
    auth: Any, settings: Settings | None = None
) -> dict[str, str]:
    """The ``Authorization`` header carrying the impersonated token."""
    return {"Authorization": f"Bearer {await get_google_token(auth, settings)}"}


def reset_token_cache() -> None:
    """Drop every cached token — for tests, and after a settings change."""
    _tokens.clear()
    _locks.clear()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _auth_type(auth: Any) -> str:
    if isinstance(auth, dict):
        return str(auth.get("type") or "none")
    return str(getattr(auth, "type", "none"))


def _auth_field(auth: Any, name: str) -> Any:
    if isinstance(auth, dict):
        return auth.get(name)
    return getattr(auth, name, None)


def _cached_token(key: tuple[str, tuple[str, ...]]) -> str | None:
    entry = _tokens.get(key)
    if entry is None:
        return None
    token, expires_at = entry
    if expires_at - _REFRESH_BUFFER_SECONDS <= time.monotonic():
        return None
    return token


def _mint_token(subject: str, scopes: list[str]) -> tuple[str, float]:
    """Blocking half of the mint — google-auth has no async surface.

    Runs in a worker thread (``asyncio.to_thread``) so the event loop is not
    blocked on the two HTTP round trips this makes (metadata server for the
    source credential, then IAM Credentials for the impersonated one).
    """
    try:
        import google.auth
        import google.auth.transport.requests
        from google.auth import impersonated_credentials
    except ImportError as exc:  # pragma: no cover — dependency is declared
        raise GoogleAuthError(
            "Google auth needs the 'google-auth' package on the backend"
        ) from exc

    try:
        source_credentials, _ = google.auth.default()
    except Exception as exc:
        raise GoogleAuthError(
            "No ambient Google credentials — the backend must run with Workload "
            f"Identity (or ADC) for `google` auth to work: {exc}"
        ) from exc

    try:
        credentials = impersonated_credentials.Credentials(
            source_credentials=source_credentials,
            target_principal=subject,
            target_scopes=list(scopes),
            lifetime=_TOKEN_LIFETIME_SECONDS,
        )
        credentials.refresh(google.auth.transport.requests.Request())
    except Exception as exc:
        raise GoogleAuthError(
            f"Could not impersonate '{subject}': {exc}. The backend's own "
            "service account needs roles/iam.serviceAccountTokenCreator on it."
        ) from exc

    token = getattr(credentials, "token", None)
    if not token or not isinstance(token, str):
        raise GoogleAuthError(
            f"Impersonating '{subject}' returned no access token"
        )
    return token, _credential_ttl_seconds(credentials)


def _credential_ttl_seconds(credentials: Any) -> float:
    """Remaining lifetime of a refreshed credential, from its ``expiry``."""
    expiry = getattr(credentials, "expiry", None)
    if expiry is None:
        return float(_DEFAULT_TOKEN_TTL_SECONDS)
    try:
        from datetime import datetime, timezone

        # google-auth stores a naive UTC datetime.
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
    except Exception:  # noqa: BLE001 — an unreadable expiry is not fatal
        logger.debug("google credential carries no readable expiry", exc_info=True)
        return float(_DEFAULT_TOKEN_TTL_SECONDS)
    return remaining if remaining > 0 else float(_DEFAULT_TOKEN_TTL_SECONDS)
