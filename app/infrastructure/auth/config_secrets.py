"""Resolving a data source's secret from the backend's own config.

An auth block may name a backend config key instead of carrying the secret
itself::

    {"type": "bearer", "from_config": "AFP_SERVICE_TOKEN"}
    {"type": "basic", "username": "svc@example.com", "from_config": "JIRA_API_TOKEN"}

The named key is looked up in the forwardable config (the credential-suffixed
env vars ``Settings.get_forwardable_config`` exposes) and its value replaces
the auth type's single secret field, which is then stored like any pasted one.

This lives here, rather than in the REST route that first needed it, because
the management MCP write paths need the same resolution — and for a stronger
reason than symmetry. A caller that cannot resolve ``from_config`` has to
paste the real secret to create a source at all, which means reading it out of
Secret Manager and carrying it through whatever transcript or log the caller
keeps. The point of ``from_config`` is that the secret never leaves the
backend, so every write path has to honour it or the guarantee is only as good
as the path someone happened to use.
"""
from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings

# Sentinel a response carries in place of a stored secret.
REDACTED_SECRET = "********"

# Secret field(s) per auth type; anything else in the block is not secret.
# ``none``, ``service_identity`` and ``google`` are absent on purpose: they
# store no secret, so there is nothing to redact, to preserve across an update,
# or to resolve from config — a ``from_config`` on one of them is an error.
SECRET_FIELDS: dict[str, tuple[str, ...]] = {
    "bearer": ("token",),
    "basic": ("password",),
    "header": ("value",),
}

FROM_CONFIG_FIELD = "from_config"


class AuthFromConfigError(ValueError):
    """A ``from_config`` reference that cannot be resolved.

    Carries a message meant for the caller: the REST layer turns it into a 422,
    the MCP layer returns it as the tool reply.
    """


def resolve_auth_from_config(auth: Any, settings: Settings | None = None) -> Any:
    """Turn a ``from_config`` reference into the named backend config value.

    Returns *auth* unchanged when it carries no reference, so every write path
    can call this unconditionally.

    An unknown or blank key raises rather than resolving to nothing: storing an
    empty secret would resurface later as an opaque 401 from the target API,
    with the definition looking perfectly well-formed.
    """
    if not isinstance(auth, dict) or FROM_CONFIG_FIELD not in auth:
        return auth
    resolved = dict(auth)
    key = (resolved.pop(FROM_CONFIG_FIELD) or "").strip()
    auth_type = resolved.get("type", "")
    fields = SECRET_FIELDS.get(auth_type, ())
    if not fields:
        raise AuthFromConfigError(
            f"auth type '{auth_type}' has no secret that can come from config"
        )
    if not key:
        raise AuthFromConfigError("auth.from_config must name a config key")
    available = (settings or get_settings()).get_forwardable_config()
    if key not in available:
        raise AuthFromConfigError(f"config key '{key}' is not set on this backend")
    resolved[fields[0]] = available[key]
    return resolved
