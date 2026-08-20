"""K8sRuntime deploys agent pods under a named ServiceAccount when configured.

Without this, the agent chart's default (`serviceAccount.create: true`) gives
every release a fresh ServiceAccount that carries no Workload Identity
annotation and no RBAC — so an agent pod has no cloud identity of its own, and
the tempting fix is to let it share the backend's. These tests pin the three
behaviours that keep that from happening: the flags are emitted when a service
account is configured, they are absent when it is not, and an explicit
`serviceAccount.*` in an agent's own helm_values wins over the deployment-wide
default.

Only the subprocess boundary and service discovery are patched; the real
argv-construction path in ``deploy`` runs.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models.agent_definition import AgentDefinition
from app.runtime.k8s import K8sRuntime

RUN_ID = "abcdef1234567890"


def _fake_helm_ok(captured: list[list[str]]):
    """Fake ``asyncio.create_subprocess_exec`` that records argv and succeeds."""

    async def _exec(*args, **kwargs):
        captured.append(list(args))
        proc = MagicMock()
        proc.returncode = 0

        async def _communicate():
            return b"", b""

        proc.communicate = _communicate
        return proc

    return _exec


async def _deploy(runtime: K8sRuntime, agent_def: AgentDefinition) -> list[str]:
    """Run spawn() with the outside world stubbed; return the helm argv."""
    captured: list[list[str]] = []
    healthy = MagicMock()
    healthy.status_code = 200
    client = AsyncMock()
    client.get = AsyncMock(return_value=healthy)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.runtime.k8s.asyncio.create_subprocess_exec", side_effect=_fake_helm_ok(captured)), \
         patch.object(K8sRuntime, "_try_oci_registry_login", AsyncMock(return_value=None)), \
         patch.object(K8sRuntime, "_discover_service_url", AsyncMock(return_value="http://agent.test:8000")), \
         patch("app.runtime.k8s.httpx.AsyncClient", return_value=client):
        url = await runtime.spawn(agent_def, {}, RUN_ID, "http://backend.test")

    assert url == "http://agent.test:8000"
    helm_calls = [c for c in captured if c[:3] == ["helm", "upgrade", "--install"]]
    assert len(helm_calls) == 1, captured
    return helm_calls[0]


def _set_pairs(argv: list[str]) -> dict[str, str]:
    """Collapse ``--set k=v`` / ``--set-string k=v`` pairs into a dict."""
    pairs: dict[str, str] = {}
    for flag, value in zip(argv, argv[1:]):
        if flag in ("--set", "--set-string", "--set-json") and "=" in value:
            key, _, val = value.partition("=")
            pairs[key] = val
    return pairs


@pytest.mark.asyncio
async def test_named_service_account_is_passed_to_helm():
    agent_def = AgentDefinition(id="researcher", default_runtime="k8s", helm_chart="oci://example/chart")
    runtime = K8sRuntime(namespace="langgraph", service_account="langgraph-agent")

    pairs = _set_pairs(await _deploy(runtime, agent_def))

    assert pairs["serviceAccount.create"] == "false"
    assert pairs["serviceAccount.name"] == "langgraph-agent"


@pytest.mark.asyncio
async def test_no_service_account_flags_when_unset():
    """Default deployments must not start pinning a ServiceAccount name."""
    agent_def = AgentDefinition(id="researcher", default_runtime="k8s", helm_chart="oci://example/chart")
    runtime = K8sRuntime(namespace="langgraph")

    argv = await _deploy(runtime, agent_def)

    assert not [a for a in argv if a.startswith("serviceAccount.")]


@pytest.mark.asyncio
async def test_agent_helm_values_override_the_default():
    """A per-agent serviceAccount.* is the more specific intent and must win."""
    agent_def = AgentDefinition(
        id="researcher",
        default_runtime="k8s",
        helm_chart="oci://example/chart",
        helm_values={"serviceAccount.name": "special-agent"},
    )
    runtime = K8sRuntime(namespace="langgraph", service_account="langgraph-agent")

    pairs = _set_pairs(await _deploy(runtime, agent_def))

    assert pairs["serviceAccount.name"] == "special-agent"
    assert "serviceAccount.create" not in pairs
