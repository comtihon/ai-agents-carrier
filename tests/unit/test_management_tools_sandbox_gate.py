"""The management MCP write path must refuse to persist unsandboxed python steps.

There is more than one way to write a workflow definition, so the gate is checked
where definitions are persisted rather than on a single route. These tests drive
the MCP core directly with an ambient principal, which is how the authenticating
ASGI wrapper hands identity to FastMCP tool bodies.
"""
from __future__ import annotations

import json

import pytest

from app.application import management_tools
from app.application.management_tools import ManagementDeps
from app.infrastructure.auth.authorization import (
    AuthorizationPolicy,
    Permission,
    set_current_permissions,
)


class _FakeWorkflowBackend:
    """Records what would have been persisted."""

    def __init__(self) -> None:
        self.created: list[object] = []
        self.updated: list[object] = []

    async def get(self, workflow_id: str):
        return None

    async def create(self, definition) -> None:
        self.created.append(definition)

    async def update(self, workflow_id: str, definition) -> None:
        self.updated.append(definition)


@pytest.fixture
def backend() -> _FakeWorkflowBackend:
    return _FakeWorkflowBackend()


@pytest.fixture
def deps(backend: _FakeWorkflowBackend) -> ManagementDeps:
    return ManagementDeps(registry=None, run_repository=None, workflow_backend=backend)


# Each async test body runs in its own copied context, so a principal bound inside
# one cannot leak into another and needs no explicit reset. Binding from a sync
# fixture would not work at all: the token would belong to a different context than
# the coroutine that reads it.


UNSANDBOXED = [{"id": "danger", "type": "python", "sandbox": False, "code": "print(1)"}]
# "Sandboxed" means isolated by something the script cannot reach around: the
# local runtime shares the backend pod, so it needs ADMIN like `sandbox: false`.
SANDBOXED = [
    {"id": "safe", "type": "python", "sandbox_runtime": "k8s", "code": "print(1)"}
]
DEFAULT_RUNTIME = [{"id": "onpod", "type": "python", "code": "print(1)"}]

API_KEY_PERMISSIONS = AuthorizationPolicy(enforce=True).permissions_for_api_key()
ADMIN_PERMISSIONS = frozenset(Permission)


@pytest.mark.asyncio
async def test_api_key_principal_cannot_create_unsandboxed_step(deps, backend) -> None:
    set_current_permissions(API_KEY_PERMISSIONS)

    result = await management_tools.create_workflow(
        deps, "wf", "WF", "desc", json.dumps(UNSANDBOXED)
    )

    assert "admin" in result.lower()
    assert "danger" in result
    assert backend.created == [], "the definition must not be persisted"


@pytest.mark.asyncio
async def test_api_key_principal_can_create_sandboxed_step(deps, backend) -> None:
    set_current_permissions(API_KEY_PERMISSIONS)

    result = await management_tools.create_workflow(
        deps, "wf", "WF", "desc", json.dumps(SANDBOXED)
    )

    assert "created" in result.lower()
    assert len(backend.created) == 1


@pytest.mark.asyncio
async def test_admin_principal_may_create_unsandboxed_step(deps, backend) -> None:
    set_current_permissions(ADMIN_PERMISSIONS)

    result = await management_tools.create_workflow(
        deps, "wf", "WF", "desc", json.dumps(UNSANDBOXED)
    )

    assert "created" in result.lower()
    assert len(backend.created) == 1


@pytest.mark.asyncio
async def test_unauthenticated_principal_cannot_create_unsandboxed_step(deps, backend) -> None:
    """No ambient principal at all resolves to no permissions, so this is refused."""
    result = await management_tools.create_workflow(
        deps, "wf", "WF", "desc", json.dumps(UNSANDBOXED)
    )

    assert "admin" in result.lower()
    assert backend.created == []


@pytest.mark.asyncio
async def test_refusal_is_returned_as_text_not_raised(deps) -> None:
    """These are MCP tool bodies: the return value is what the model sees, so a
    refusal has to arrive as readable text rather than an exception."""
    set_current_permissions(API_KEY_PERMISSIONS)

    result = await management_tools.create_workflow(
        deps, "wf", "WF", "desc", json.dumps(UNSANDBOXED)
    )

    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_api_key_principal_may_create_a_step_on_the_default_runtime(
    deps, backend
) -> None:
    """A python step that names no runtime gets `local`, which the seccomp
    allow-list makes a real boundary — so the MCP key can now author and edit
    these workflows instead of being locked out of them."""
    set_current_permissions(API_KEY_PERMISSIONS)

    result = await management_tools.create_workflow(
        deps, "wf", "WF", "desc", json.dumps(DEFAULT_RUNTIME)
    )

    assert "admin" not in result.lower()
    assert [w.id for w in backend.created] == ["wf"]
