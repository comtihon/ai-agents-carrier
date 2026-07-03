"""Regression tests for K8sRuntime's agent-scoped release lookup.

Prior to this fix, ``has_container_for_run``, ``get_agent_url_for_run``, and
``terminate_by_run_id`` all filtered Helm releases with the broad regex
``agent-.*-{run_id[:8]}`` — matching ANY agent role sharing a run_id, not the
specific role being resolved. This caused a planner step to wrongly resolve
the researcher's still-warm pod (same run_id, different agent) and reuse its
stale output.

These tests construct a real ``K8sRuntime`` and patch only the subprocess
boundary (``asyncio.create_subprocess_exec``) and the service-discovery
network call (``_discover_service_url``), exercising the actual
release-name-matching logic where the bug lived.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models.agent_definition import AgentDefinition
from app.runtime.k8s import K8sRuntime

RUN_ID = "abcdef1234567890"


def _make_defs() -> tuple[AgentDefinition, AgentDefinition]:
    researcher_def = AgentDefinition(id="researcher", default_runtime="k8s")
    planner_def = AgentDefinition(id="planner", default_runtime="k8s")
    return researcher_def, planner_def


def _fake_subprocess_exec_factory(statuses: dict[str, str]):
    """Return an async fake for ``asyncio.create_subprocess_exec``.

    Simulates ``helm status <release_name> -n <ns> -o json``: returns a
    "deployed"/"failed" JSON payload for release names present in *statuses*,
    and a non-zero returncode (release not found) for any other name.
    """

    async def _fake_create_subprocess_exec(*args, **kwargs):
        # cmd = ["helm", "status", release_name, "-n", namespace, "-o", "json"]
        release_name = args[2] if len(args) > 2 else None
        status = statuses.get(release_name)
        proc = MagicMock()
        if status is None:
            proc.returncode = 1

            async def _communicate():
                return b"", b"Error: release: not found"
        else:
            proc.returncode = 0
            payload = json.dumps({"info": {"status": status}}).encode()

            async def _communicate():
                return payload, b""

        proc.communicate = _communicate
        return proc

    return _fake_create_subprocess_exec


@pytest.mark.asyncio
async def test_has_container_for_run_scoped_to_agent():
    """Core regression proof: prior buggy code returned True based on ANY
    matching release; the fix must resolve only the exact agent+run_id release."""
    researcher_def, planner_def = _make_defs()
    runtime = K8sRuntime(namespace="test-ns")
    researcher_release = runtime._release_name(researcher_def, RUN_ID)
    planner_release = runtime._release_name(planner_def, RUN_ID)

    # Both releases exist, healthy — planner's own status governs its result.
    statuses = {researcher_release: "deployed", planner_release: "deployed"}
    with patch(
        "app.runtime.k8s.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess_exec_factory(statuses),
    ):
        assert await runtime.has_container_for_run(planner_def, RUN_ID) is True

    # Planner's release is "failed" — must return False even though the
    # researcher's release (sharing the same run_id) is healthy.
    statuses = {researcher_release: "deployed", planner_release: "failed"}
    with patch(
        "app.runtime.k8s.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess_exec_factory(statuses),
    ):
        assert await runtime.has_container_for_run(planner_def, RUN_ID) is False
        # Researcher's own release is still healthy — proves the two are resolved
        # independently rather than one broad match governing both.
        assert await runtime.has_container_for_run(researcher_def, RUN_ID) is True

    # Planner's release doesn't exist at all — must return False even though
    # the researcher's release (same run_id) exists and is healthy.
    statuses = {researcher_release: "deployed"}
    with patch(
        "app.runtime.k8s.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess_exec_factory(statuses),
    ):
        assert await runtime.has_container_for_run(planner_def, RUN_ID) is False


@pytest.mark.asyncio
async def test_get_agent_url_for_run_scoped_to_agent():
    """Must resolve the planner's own release/URL, never the researcher's,
    even when both releases exist simultaneously under the same run_id."""
    researcher_def, planner_def = _make_defs()
    runtime = K8sRuntime(namespace="test-ns")
    researcher_release = runtime._release_name(researcher_def, RUN_ID)
    planner_release = runtime._release_name(planner_def, RUN_ID)

    statuses = {researcher_release: "deployed", planner_release: "deployed"}

    async def _fake_discover_service_url(self, release_name):
        return f"http://{release_name}.svc/"

    with patch(
        "app.runtime.k8s.asyncio.create_subprocess_exec",
        side_effect=_fake_subprocess_exec_factory(statuses),
    ), patch.object(K8sRuntime, "_discover_service_url", _fake_discover_service_url):
        url = await runtime.get_agent_url_for_run(planner_def, RUN_ID)

    assert url is not None
    assert "agent-planner-" in url
    assert "agent-researcher-" not in url


@pytest.mark.asyncio
async def test_terminate_by_run_id_scoped_to_agent():
    """Must uninstall exactly the planner's release, never a broader match."""
    _researcher_def, planner_def = _make_defs()
    runtime = K8sRuntime(namespace="test-ns")
    planner_release = runtime._release_name(planner_def, RUN_ID)

    with patch.object(runtime, "uninstall_release", AsyncMock()) as mock_uninstall:
        await runtime.terminate_by_run_id(planner_def, RUN_ID)

    mock_uninstall.assert_awaited_once_with(planner_release)
