"""Agent runtime package.

Provides pluggable runtimes for spawning agent processes:
  - LocalRuntime  — subprocess via asyncio.create_subprocess_exec
  - DockerRuntime — Docker container (stub; requires aiodocker)
  - K8sRuntime    — Kubernetes Job (stub; requires kubernetes-asyncio)

Use ``get_runtime(runtime_type)`` from ``factory`` to obtain an instance.
"""
from app.runtime.factory import get_runtime

__all__ = ["get_runtime"]
