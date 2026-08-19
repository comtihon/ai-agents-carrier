"""Authorization gate for unsandboxed `python` workflow steps.

A `python` step with ``sandbox: false`` is ``exec``'d inside the backend process,
next to the Mongo URI, every LLM key, the service-auth private key and the pod's
cloud identity. It is not a data-level privilege — it is code execution in the
backend, so it needs ADMIN rather than WRITE.

The check lives here, apart from any transport, because there is more than one way
to write a workflow definition (the REST API and the management MCP today). A gate
attached to one route would simply be bypassed through the other.
"""
from __future__ import annotations

from typing import Any, Iterable

from app.infrastructure.auth.authorization import Permission


class SandboxNotPermittedError(PermissionError):
    """Raised when a caller without ADMIN submits an unsandboxed python step."""

    def __init__(self, step_ids: list[str]) -> None:
        self.step_ids = step_ids
        listed = ", ".join(step_ids) or "unknown"
        super().__init__(
            f"Unsandboxed python steps require admin permission (steps: {listed}). "
            "A step with 'sandbox: false' runs inside the backend process with access "
            "to its credentials. Remove the flag to run sandboxed, or have an "
            "administrator submit this workflow."
        )


def _steps_of(definition: Any) -> Iterable[dict]:
    """Yield step mappings from a workflow definition of any supported shape.

    Definitions arrive as a raw dict from the API, as a model exposing
    ``to_raw_dict()``, or as a bare list of steps. Anything unrecognised yields
    nothing, which is safe: it cannot silently approve a step this function never
    inspected, because a definition with no readable steps also has no
    ``sandbox: false`` to approve.
    """
    raw: Any = definition
    to_raw = getattr(definition, "to_raw_dict", None)
    if callable(to_raw):
        try:
            raw = to_raw()
        except Exception:
            return

    if isinstance(raw, dict):
        for key in ("steps", "nodes", "graph"):
            value = raw.get(key)
            if isinstance(value, list):
                raw = value
                break
        else:
            # A mapping of step_id -> step is also accepted.
            if all(isinstance(v, dict) for v in raw.values()) and raw:
                raw = list(raw.values())
            else:
                return

    if not isinstance(raw, list):
        return

    for step in raw:
        if isinstance(step, dict):
            yield step


def find_unsandboxed_python_steps(definition: Any) -> list[str]:
    """Ids of `python` steps that ask to run outside the sandbox.

    Only an explicit false counts. ``sandbox`` omitted means sandboxed (the
    documented default), and a truthy value obviously does too.
    """
    offending: list[str] = []
    for step in _steps_of(definition):
        if step.get("type") != "python":
            continue
        if step.get("sandbox") is False:
            offending.append(str(step.get("id") or step.get("name") or "<unnamed>"))
    return offending


def assert_sandbox_allowed(definition: Any, permissions: frozenset[Permission] | set[Permission]) -> None:
    """Raise :class:`SandboxNotPermittedError` unless the caller may disable the sandbox.

    Call this on every path that persists a workflow definition.
    """
    if Permission.ADMIN in permissions:
        return
    offending = find_unsandboxed_python_steps(definition)
    if offending:
        raise SandboxNotPermittedError(offending)
