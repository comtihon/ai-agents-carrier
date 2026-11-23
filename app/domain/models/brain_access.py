"""Access levels to the shared company memory (gbrain).

Both an agent and a workflow carry a ``brain_access`` level.  What an agent
step actually gets is the *lower* of the two: a workflow cannot hand an agent
more than the agent was cleared for, and an agent cleared for writing cannot
write from a workflow that only allows reading.

Anything that is not one of the three known levels — ``None``, a typo, a value
from a future version — is read as ``"none"``.  Unknown means no access.
"""
from __future__ import annotations

from typing import Literal

BrainAccess = Literal["none", "read", "write"]

_ORDER: dict[str, int] = {"none": 0, "read": 1, "write": 2}


def normalize_brain_access(value: object) -> BrainAccess:
    """Return *value* as a known level, or ``"none"`` when it is not one."""
    if isinstance(value, str) and value.strip().lower() in _ORDER:
        return value.strip().lower()  # type: ignore[return-value]
    return "none"


def effective_brain_access(*levels: object) -> BrainAccess:
    """The lowest of *levels* (order none < read < write); ``"none"`` if empty."""
    if not levels:
        return "none"
    return min((normalize_brain_access(level) for level in levels), key=_ORDER.__getitem__)
