from __future__ import annotations

from app.runtime.base import AgentRuntime


def get_runtime(runtime_type: str) -> AgentRuntime:
    """Return an ``AgentRuntime`` instance for the given *runtime_type*.

    Parameters
    ----------
    runtime_type:
        One of ``"local"``, ``"docker"``, or ``"k8s"``.

    Raises
    ------
    ValueError
        When *runtime_type* is not recognised.
    """
    if runtime_type == "local":
        from app.runtime.local import LocalRuntime
        return LocalRuntime()
    if runtime_type == "docker":
        from app.runtime.docker import DockerRuntime
        return DockerRuntime()
    if runtime_type == "k8s":
        from app.runtime.k8s import K8sRuntime
        return K8sRuntime()
    raise ValueError(
        f"Unknown runtime type '{runtime_type}'. "
        "Valid values are: 'local', 'docker', 'k8s'."
    )
