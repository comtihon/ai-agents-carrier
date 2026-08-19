"""Authorization gate for `python` workflow steps that are not isolated.

A `python` step with ``sandbox: false`` is ``exec``'d inside the backend process,
next to the Mongo URI, every LLM key, the service-auth private key and the pod's
cloud identity. It is not a data-level privilege — it is code execution in the
backend, so it needs ADMIN rather than WRITE.

``sandbox_runtime: local`` — which is also what a step gets when it names no
runtime — is the same privilege wearing a different hat. It runs the script in a
child ``python -I -S`` process of the backend pod: the environment is cleared and
imports are filtered, but the process shares the pod's filesystem, so the script
can simply ``open()`` the mounted Kubernetes service-account token, and the import
filter is a deny-list in the same interpreter as the code it filters (a script
that loads a module through the loader machinery instead of ``import`` is outside
it). It keeps an honest script from touching the backend by accident; it does not
keep a hostile one out. Only the ``docker`` and ``k8s`` runtimes put a kernel
boundary in the way, so those are what WRITE may submit, and anything else needs
ADMIN — otherwise the ADMIN tier would be a formality that any WRITE holder walks
around by leaving one field unset.

The check lives here, apart from any transport, because there is more than one way
to write a workflow definition (the REST API and the management MCP today). A gate
attached to one route would simply be bypassed through the other.
"""
from __future__ import annotations

from typing import Any, Iterable

from app.infrastructure.auth.authorization import Permission

# Sandbox runtimes that isolate the script with something the script itself cannot
# reach around: a container, or a pod with the service-account token unmounted.
ISOLATED_RUNTIMES = frozenset({"docker", "k8s"})


class SandboxNotPermittedError(PermissionError):
    """Raised when a caller without ADMIN submits a non-isolated python step."""

    def __init__(self, step_ids: list[str], reasons: dict[str, str] | None = None) -> None:
        self.step_ids = step_ids
        self.reasons = reasons or {}
        listed = ", ".join(step_ids) or "unknown"
        detail = "; ".join(f"{sid}: {why}" for sid, why in self.reasons.items())
        super().__init__(
            f"Python steps that are not isolated require admin permission "
            f"(steps: {listed}). " + (f"{detail}. " if detail else "")
            + "Such a step runs on the backend pod, with access to its credentials "
            "and its service-account token. Set 'sandbox_runtime: k8s' (or 'docker') "
            "and leave 'sandbox' unset to run isolated, or have an administrator "
            "submit this workflow."
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


def _step_id(step: dict) -> str:
    return str(step.get("id") or step.get("name") or "<unnamed>")


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
            offending.append(_step_id(step))
    return offending


def find_admin_only_python_steps(definition: Any) -> dict[str, str]:
    """``{step_id: reason}`` for python steps that WRITE alone may not submit.

    Two cases, one privilege: the step opts out of the sandbox entirely, or it
    asks for (or defaults to) the ``local`` runtime, which shares the backend
    pod. See the module docstring for why the second is not a lesser case.
    """
    offending: dict[str, str] = {}
    for step in _steps_of(definition):
        if step.get("type") != "python":
            continue
        if step.get("sandbox") is False:
            offending[_step_id(step)] = (
                "'sandbox: false' runs the code inside the backend process"
            )
            continue
        runtime = str(step.get("sandbox_runtime") or "local").strip().lower()
        if runtime not in ISOLATED_RUNTIMES:
            named = "sandbox_runtime" if step.get("sandbox_runtime") else "no runtime"
            offending[_step_id(step)] = (
                f"{named} '{runtime}' runs the code on the backend pod, which is not "
                "an isolation boundary"
            )
    return offending


def assert_sandbox_allowed(definition: Any, permissions: frozenset[Permission] | set[Permission]) -> None:
    """Raise :class:`SandboxNotPermittedError` unless the caller may submit *definition*.

    Call this on every path that persists a workflow definition.
    """
    if Permission.ADMIN in permissions:
        return
    offending = find_admin_only_python_steps(definition)
    if offending:
        raise SandboxNotPermittedError(list(offending), offending)
