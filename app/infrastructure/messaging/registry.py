"""Name -> provider lookup for the messaging abstraction.

A step names its provider (``provider: slack``), so a second provider is a new
:class:`~app.infrastructure.messaging.base.MessagingProvider` subclass plus one
``@register_provider`` decorator — the step, the management tools and the
notification helpers stay untouched.

Instances are cached per name because they are stateless apart from an httpx
client-per-call and their credential, which is read from settings on first use.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.infrastructure.messaging.base import MessagingProvider

_PROVIDERS: dict[str, type["MessagingProvider"]] = {}
_INSTANCES: dict[str, "MessagingProvider"] = {}

DEFAULT_PROVIDER = "slack"


def register_provider(cls: type["MessagingProvider"]) -> type["MessagingProvider"]:
    """Class decorator registering *cls* under its own ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set a non-empty 'name'")
    _PROVIDERS[cls.name] = cls
    return cls


def available_providers() -> list[str]:
    _load_builtins()
    return sorted(_PROVIDERS)


def get_provider(name: str | None = None) -> "MessagingProvider":
    """Return the provider registered under *name* (default ``slack``)."""
    _load_builtins()
    key = (name or DEFAULT_PROVIDER).strip().lower()
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown messaging provider '{key}' "
            f"(available: {', '.join(sorted(_PROVIDERS)) or 'none'})"
        )
    instance = _INSTANCES.get(key)
    if instance is None:
        instance = cls()
        _INSTANCES[key] = instance
    return instance


def reset_providers() -> None:
    """Drop cached instances.  Tests use this; nothing in production does."""
    _INSTANCES.clear()


def _load_builtins() -> None:
    """Import the bundled providers so their decorators have run.

    Done lazily rather than in ``__init__`` so that importing the base classes
    never drags in httpx or the settings object.
    """
    if _PROVIDERS:
        return
    from app.infrastructure.messaging import slack  # noqa: F401
