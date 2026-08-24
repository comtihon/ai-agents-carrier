"""The k8s sandbox must fail, not hang, when the cluster API does not answer.

Observed in production: a `python` step with sandbox_runtime=k8s sat in `running`
indefinitely, logging nothing after "step running". The step's timeout only ever
bounded the pod-status polling loop -- the create calls that run before it were
outside it, and the kubernetes client applies no timeout of its own, so one
stalled request held the step open forever. A hang is far worse than an error
here: an error names a cause, a hang looks like slow work.
"""
from __future__ import annotations

import asyncio

import pytest

from app.infrastructure.orchestration import script_sandbox


class _StalledCore:
    """A CoreV1Api whose calls never return, like a blackholed API server."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _stall(self, name: str):
        self.calls.append(name)
        # Block the worker thread the way a connect with no timeout would.
        import time
        time.sleep(5)

    def create_namespaced_config_map(self, **kw):
        return self._stall("create_cm")

    def create_namespaced_pod(self, **kw):
        return self._stall("create_pod")

    def read_namespaced_pod(self, **kw):
        return self._stall("read_pod")

    def read_namespaced_pod_log(self, **kw):
        return self._stall("read_log")

    def delete_namespaced_pod(self, **kw):
        return None

    def delete_namespaced_config_map(self, **kw):
        return None


@pytest.fixture()
def stalled_k8s(monkeypatch):
    """Stand in for the kubernetes client module the runner imports lazily."""
    import sys
    import types

    core = _StalledCore()

    client_mod = types.ModuleType("kubernetes.client")
    for attr in (
        "V1ConfigMap", "V1ObjectMeta", "V1Pod", "V1PodSpec", "V1Volume",
        "V1ConfigMapVolumeSource", "V1EmptyDirVolumeSource", "V1Container",
        "V1VolumeMount", "V1ResourceRequirements", "V1SecurityContext",
        "V1Capabilities",
    ):
        setattr(client_mod, attr, lambda *a, **k: None)
    client_mod.CoreV1Api = lambda: core  # type: ignore[attr-defined]

    config_mod = types.ModuleType("kubernetes.config")
    config_mod.load_incluster_config = lambda: None  # type: ignore[attr-defined]
    config_mod.load_kube_config = lambda: None  # type: ignore[attr-defined]

    pkg = types.ModuleType("kubernetes")
    pkg.client = client_mod  # type: ignore[attr-defined]
    pkg.config = config_mod  # type: ignore[attr-defined]

    for name, mod in (
        ("kubernetes", pkg),
        ("kubernetes.client", client_mod),
        ("kubernetes.config", config_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return core


@pytest.mark.asyncio
async def test_a_stalled_api_raises_within_the_step_timeout(stalled_k8s):
    """Deliberately no outer wait_for: an outer guard would supply the very
    TimeoutError being asserted, and the test would pass whether or not the
    runner bounds its own calls. The runner has to raise by itself."""
    loop = asyncio.get_running_loop()
    started = loop.time()

    with pytest.raises((asyncio.TimeoutError, script_sandbox.ScriptSandboxError)):
        await script_sandbox._run_k8s(
            '{"code": "output = 1", "state": {}}',
            timeout=1.0, image="python:3.12-slim", memory_mb=128,
            namespace="langgraph",
        )

    assert loop.time() - started < 10.0, "the call must be bounded by the step timeout"
    assert stalled_k8s.calls[:1] == ["create_cm"], "it must have attempted the create"
