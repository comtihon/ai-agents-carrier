"""Provider-neutral messaging (Slack today, a second provider tomorrow)."""
from app.infrastructure.messaging.base import (
    Message,
    MessagingError,
    MessagingProvider,
    PostedMessage,
)
from app.infrastructure.messaging.registry import (
    DEFAULT_PROVIDER,
    available_providers,
    get_provider,
    register_provider,
    reset_providers,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "Message",
    "MessagingError",
    "MessagingProvider",
    "PostedMessage",
    "available_providers",
    "get_provider",
    "register_provider",
    "reset_providers",
]
