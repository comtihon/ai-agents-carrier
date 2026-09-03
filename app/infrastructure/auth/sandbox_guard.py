"""Authorization gate for `python` workflow steps that run unsandboxed.

A `python` step with ``sandbox: false`` is ``exec``'d inside the backend process,
next to the Mongo URI, every LLM key, the service-auth private key and the pod's
cloud identity. It is not a data-level privilege — it is code execution in the
backend, so it needs ADMIN rather than WRITE. That is the whole of what this
gate now covers.

``sandbox_runtime: local`` used to need ADMIN too, and no longer does. It once
shared the backend pod with nothing but an in-interpreter deny-list in the way,
which reflection walks around — a ``str.format()`` chain is enough to reach a
class the deny-list never removed. The sandbox now installs a seccomp-bpf
allow-list before any script runs (see ``script_sandbox._SECCOMP_INSTALL``), and
the kernel refuses ``openat``, ``socket``, ``execve``, ``clone`` and everything
else outside a compute-only set, whatever the interpreter thinks. Denying
``openat`` outright is what closes the case a syscall filter cannot otherwise
reach: a filter cannot inspect a path, so the only way to stop a script reading
the pod's service-account token is to refuse it every file. A script that
reflects its way to ``open`` now gets ``PermissionError`` from the kernel.

The filter is mandatory: if it cannot be installed the sandbox refuses to run
the script at all, so ``local`` cannot silently degrade to the old behaviour.
That is what makes it safe for WRITE.

The check lives here, apart from any transport, because there is more than one way
to write a workflow definition (the REST API and the management MCP today). A gate
attached to one route would simply be bypassed through the other.
"""
from __future__ import annotations

from typing import Any, Iterable

from app.infrastructure.auth.authorization import Permission

# Sandbox runtimes that isolate the script with something it cannot reach around:
# a container, a pod with the service-account token unmounted, or -- since the
# seccomp allow-list -- the kernel refusing the syscalls in-process.
ISOLATED_RUNTIMES = frozenset({"local", "docker", "k8s"})


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
            + "Such a step runs inside the backend process, with access to its "
            "credentials and its service-account token. Leave 'sandbox' unset to "
            "run isolated -- every sandbox runtime, 'local' included, is enforced "
            "by a seccomp allow-list -- or have an administrator submit this "
            "workflow."
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

    One case now: the step opts out of the sandbox entirely. Every sandbox
    runtime — including ``local``, and including the ``local`` a step gets by
    naming no runtime — is a real boundary; see the module docstring.
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
            offending[_step_id(step)] = (
                f"sandbox_runtime '{runtime}' is not a known isolated runtime "
                f"(expected one of: {', '.join(sorted(ISOLATED_RUNTIMES))})"
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


# ---------------------------------------------------------------------------
# Generated code (sheet binding tier 2)
# ---------------------------------------------------------------------------
# A tier-2 sheet binding stores a Python transform an LLM wrote from an
# instruction a user typed, and the backend later runs it. That is code
# execution on the backend, arriving through a surface whose other operations
# are ordinary data edits -- so it is gated at the same tier as an unsandboxed
# `python` workflow step, for the same reason and through this same module.
#
# It is NOT the same *risk*: a tier-2 transform runs inside the seccomp sandbox
# (no filesystem, no network, no processes), whereas `sandbox: false` runs in the
# backend process next to every credential. The gate is nevertheless ADMIN
# rather than WRITE because of who writes the code and how it gets there.
# Holding WRITE means "may edit definitions"; a definition is data a person
# read before saving. Generated code is neither -- nobody has read it when the
# compile call is made, and the instruction that produced it is untrusted input
# that a prompt-injection attempt reaches the model through. Storing executable
# code the caller has not seen is a privileged act, so it needs the privileged
# tier.
#
# Deliberately not gated here: *running* an already-activated binding, reading
# its code, listing it, previewing it. Those are the data-level operations the
# datasource surfaces already govern, and requiring ADMIN to run a binding an
# administrator activated would make the feature unusable by the people it is
# for.

class GeneratedCodeNotPermittedError(PermissionError):
    """Raised when a caller without ADMIN tries to store generated code."""

    def __init__(self, what: str = "a generated transform") -> None:
        super().__init__(
            f"Storing {what} requires admin permission. A tier-2 sheet binding "
            "holds Python that a language model wrote from an instruction, and "
            "which this backend then executes -- sandboxed, but still code that "
            "nobody has read at the moment it is stored. Authoring one is "
            "therefore an administrator's action, not an ordinary definition "
            "edit. A tier-1 binding (a form: no code) needs only WRITE, and "
            "running or previewing a binding an administrator activated needs "
            "no special permission at all."
        )


def generated_code_permitted() -> bool:
    """Whether the ambient caller may store generated code.

    Reads the ambient principal rather than taking one, because the tool cores
    in ``app.application.management_tools`` are called by three surfaces that
    each bind identity differently -- and because an *unbound* principal (no
    authenticating wrapper ran at all: OAuth off, an in-process call) is allowed
    here for the same reason the REST routes fall back to permitting it. That
    is the deployment's pre-RBAC posture, not a hole this function opens.
    """
    from app.infrastructure.auth.authorization import missing_permission

    return missing_permission(Permission.ADMIN) is None


def assert_generated_code_allowed(what: str = "a generated transform") -> None:
    """Raise :class:`GeneratedCodeNotPermittedError` unless the caller has ADMIN.

    Call this on every path that *stores* generated code -- compile, recompile,
    a human edit of the code, and activation -- so that no surface is the easy
    way in. Reads and runs are deliberately not covered; see the note above.
    """
    if not generated_code_permitted():
        raise GeneratedCodeNotPermittedError(what)
