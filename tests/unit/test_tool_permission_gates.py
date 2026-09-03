"""Every management operation requires its own tier, on every surface.

The REST API derives the required permission from the HTTP method, which the tool
surfaces cannot do: the ``/mcp/management`` tools and the chat agent's platform
tools all arrive as one ``POST``, and the management wrapper checks only the
ACCESS tier. So the gate lives on the shared cores in
``app.application.management_tools`` / ``app.application.run_control`` — the same
functions all three surfaces call — and these tests drive those cores directly
with an ambient principal, which is how the authenticating ASGI wrapper hands
identity to a FastMCP tool body.

The distinction that matters throughout: a principal *bound with nothing* is a
real authenticated caller whose roles grant nothing, and is denied; an *unbound*
principal means no authenticating wrapper ran at all (OAuth disabled, a Slack
callback, a Pub/Sub trigger, an in-process call) and is allowed, which is the
pre-RBAC posture the REST routes fall back to as well.
"""
from __future__ import annotations

import pytest

from app.application import management_tools as core
from app.application import run_control
from app.application.management_tools import ManagementDeps
from app.infrastructure.auth.authorization import (
    Permission,
    set_current_permissions,
)


ACCESS_ONLY = frozenset({Permission.ACCESS})
READER = frozenset({Permission.ACCESS, Permission.READ})
WRITER = frozenset({Permission.ACCESS, Permission.READ, Permission.WRITE})
DELETER = frozenset(
    {Permission.ACCESS, Permission.READ, Permission.WRITE, Permission.DELETE}
)


class _FakeWorkflowBackend:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.created: list[object] = []

    async def get(self, workflow_id: str):
        return None

    async def create(self, definition) -> None:
        self.created.append(definition)

    async def delete(self, workflow_id: str) -> None:
        self.deleted.append(workflow_id)


class _FakeRegistry:
    def list_definitions(self) -> list[dict]:
        return [{"id": "wf", "name": "Workflow"}]


@pytest.fixture
def backend() -> _FakeWorkflowBackend:
    return _FakeWorkflowBackend()


@pytest.fixture
def deps(backend: _FakeWorkflowBackend) -> ManagementDeps:
    return ManagementDeps(
        registry=_FakeRegistry(), run_repository=None, workflow_backend=backend
    )


# ── The map itself ────────────────────────────────────────────────────────────
#
# Snapshot of every tool the management MCP publishes and the tier it needs. It
# is asserted against the live registration, so adding a tool fails here until
# somebody classifies it — the tool equivalent of the method table that covers
# the REST routes.

EXPECTED_TOOL_PERMISSIONS: dict[str, Permission] = {
    # workflows / runs
    "list_workflows": Permission.READ,
    "get_workflow": Permission.READ,
    "run_workflow": Permission.WRITE,
    "list_runs": Permission.READ,
    "get_run": Permission.READ,
    # A run's download manifest and one entry of it. Reads what a `data` step
    # recorded and hands back a URL; the bytes are fetched over HTTP, where the
    # same READ gate applies by method. Exactly as sensitive as get_run.
    "list_run_data": Permission.READ,
    "get_run_data_artifact": Permission.READ,
    # The read side of the approval queue. Deciding a case is WRITE
    # (approve_run / reject_run); reading one before deciding it must not cost
    # more than reading any other record, or the informed decision becomes the
    # privileged one. Matches the GET routes in app.api.routes.approvals, which
    # the method table already puts at READ.
    "list_pending_approvals": Permission.READ,
    "get_approval": Permission.READ,
    "create_workflow": Permission.WRITE,
    "update_workflow": Permission.WRITE,
    "delete_workflow": Permission.DELETE,
    # agent definitions
    "list_agents": Permission.READ,
    "get_agent": Permission.READ,
    "create_agent": Permission.WRITE,
    "update_agent": Permission.WRITE,
    "delete_agent": Permission.DELETE,
    # data sources
    "list_datasources": Permission.READ,
    "get_datasource": Permission.READ,
    "create_datasource": Permission.WRITE,
    "update_datasource": Permission.WRITE,
    "delete_datasource": Permission.DELETE,
    "create_pubsub_datasource": Permission.WRITE,
    "list_pubsub_subscriptions": Permission.READ,
    # Reads Drive metadata as the backend's impersonated service account and
    # writes nothing, so READ — the same tier as get_datasource.
    "resolve_google_file": Permission.READ,
    "create_google_sheets_datasource": Permission.WRITE,
    # Sheet bindings.  Probing and previewing read the spreadsheet as the
    # impersonated service account and write nothing -- a preview plans the
    # write and stops, which is the whole point of it -- so they sit at READ
    # with the other inspection tools.  Saving a binding compiles it into an
    # operation of the data source, so it is a definition change: WRITE.
    "probe_google_sheet": Permission.READ,
    "list_sheet_bindings": Permission.READ,
    "get_sheet_binding": Permission.READ,
    "preview_sheet_binding": Permission.READ,
    "save_sheet_binding": Permission.WRITE,
    "delete_sheet_binding": Permission.DELETE,
    # Tier 2 (generated transforms).  Classified like the rest of the datasource
    # tooling -- reading is READ, mutating is WRITE -- with one extra layer that
    # this table does not express: compile, edit and activate additionally
    # require ADMIN, enforced inside the shared service by
    # auth.sandbox_guard.assert_generated_code_allowed.  WRITE gets a caller to
    # the tool; ADMIN is what lets it store code nobody has read yet.  Marking a
    # binding stale is only WRITE on purpose: switching generated code *off*
    # must never be the privileged direction.
    "get_sheet_binding_code": Permission.READ,
    "compile_sheet_binding_code": Permission.WRITE,
    "edit_sheet_binding_code": Permission.WRITE,
    "activate_sheet_binding_code": Permission.WRITE,
    "retest_sheet_binding_code": Permission.WRITE,
    "mark_sheet_binding_stale": Permission.WRITE,
    # script library — `python` steps reference these by script_id
    "list_scripts": Permission.READ,
    "get_script": Permission.READ,
    "create_script": Permission.WRITE,
    "update_script": Permission.WRITE,
    "delete_script": Permission.DELETE,
    # events (Pub/Sub topics workflows are triggered by)
    "list_events": Permission.READ,
    "get_event": Permission.READ,
    "create_event": Permission.WRITE,
    "update_event": Permission.WRITE,
    "delete_event": Permission.DELETE,
    # Fetches an arbitrary remote URL with optional credentials, so it is a
    # WRITE-tier action even though it only returns a description of a spec.
    "import_datasource_schema": Permission.WRITE,
    "create_datasource_from_schema": Permission.WRITE,
    "add_datasource_operations_from_schema": Permission.WRITE,
    # run control — gated inside app.application.run_control, which raises
    # RunControlError(403) instead of returning a refusal string.
    "terminate_run": Permission.WRITE,
    "retry_run": Permission.WRITE,
    "restart_from_step": Permission.WRITE,
    "approve_run": Permission.WRITE,
    "reject_run": Permission.WRITE,
    # messaging — one shared provider abstraction behind both surfaces.  A
    # delete is a delete even when the thing deleted lives in Slack.
    "post_message": Permission.WRITE,
    "read_messages": Permission.READ,
    "read_thread": Permission.READ,
    "send_direct_message": Permission.WRITE,
    "delete_message": Permission.DELETE,
}

_RUN_CONTROL_TOOLS = {
    "terminate_run", "retry_run", "restart_from_step", "approve_run", "reject_run",
}


def _registered_tool_names() -> list[str]:
    from app.api.mcp.management_server import register_management_tools

    class _Recorder:
        def __init__(self) -> None:
            self.names: list[str] = []

        def add_tool(self, handler, name: str) -> None:
            self.names.append(name)

    recorder = _Recorder()
    register_management_tools(recorder, lambda: None)
    return recorder.names


def test_every_published_tool_is_classified() -> None:
    assert set(_registered_tool_names()) == set(EXPECTED_TOOL_PERMISSIONS)


@pytest.mark.parametrize(
    "name",
    sorted(set(EXPECTED_TOOL_PERMISSIONS) - _RUN_CONTROL_TOOLS),
)
def test_shared_core_declares_the_expected_permission(name: str) -> None:
    """The gate is on the core, so both surfaces inherit it."""
    assert getattr(core, name).required_permission is EXPECTED_TOOL_PERMISSIONS[name]


# ── Denials ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reader_cannot_create_a_workflow(deps, backend) -> None:
    set_current_permissions(READER)
    result = await core.create_workflow(deps, "wf", "WF", "d", "[]")
    assert "Not permitted" in result and "write" in result
    assert backend.created == []


@pytest.mark.asyncio
async def test_reader_cannot_delete_a_workflow(deps, backend) -> None:
    set_current_permissions(READER)
    result = await core.delete_workflow(deps, "wf")
    assert "Not permitted" in result and "delete" in result
    assert backend.deleted == []


@pytest.mark.asyncio
async def test_writer_cannot_delete_a_workflow(deps, backend) -> None:
    """WRITE is enough to POST, so without its own tier DELETE would come free."""
    set_current_permissions(WRITER)
    result = await core.delete_workflow(deps, "wf")
    assert "Not permitted" in result and "delete" in result
    assert backend.deleted == []


@pytest.mark.asyncio
async def test_access_only_principal_cannot_even_list(deps) -> None:
    set_current_permissions(ACCESS_ONLY)
    result = core.list_workflows(deps)
    assert "Not permitted" in result and "read" in result


@pytest.mark.asyncio
async def test_reader_may_list(deps) -> None:
    set_current_permissions(READER)
    assert "Not permitted" not in core.list_workflows(deps)


@pytest.mark.asyncio
async def test_deleter_may_delete(deps, backend) -> None:
    set_current_permissions(DELETER)
    result = await core.delete_workflow(deps, "wf")
    assert "Not permitted" not in result


@pytest.mark.asyncio
async def test_unbound_principal_is_allowed(deps, backend) -> None:
    """No wrapper ran: an internal caller, or a deployment with auth switched off."""
    result = await core.delete_workflow(deps, "wf")
    assert "Not permitted" not in result


# ── Run control ───────────────────────────────────────────────────────────────

class _FakeRunRepository:
    async def get(self, run_id: str):
        return None

    async def claim_for_resume(self, run_id: str):
        return None


class _FakeContainer:
    def __init__(self) -> None:
        self.run_repository = _FakeRunRepository()
        self.live_runners: dict = {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (run_control.terminate_run, ()),
        (run_control.approve_run, ()),
        (run_control.reject_run, ()),
        (run_control.retry_run, ()),
        (run_control.restart_from_step, ("step",)),
    ],
)
async def test_reader_cannot_control_a_run(operation, args) -> None:
    set_current_permissions(READER)
    with pytest.raises(run_control.RunControlError) as exc:
        await operation(_FakeContainer(), "run-1", *args)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_writer_passes_the_run_control_gate() -> None:
    """The 404 proves the gate let it through to the run lookup."""
    set_current_permissions(WRITER)
    with pytest.raises(run_control.RunControlError) as exc:
        await run_control.terminate_run(_FakeContainer(), "missing")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_unbound_principal_passes_the_run_control_gate() -> None:
    """A Slack approval callback carries no roles and must still work."""
    with pytest.raises(run_control.RunControlError) as exc:
        await run_control.terminate_run(_FakeContainer(), "missing")
    assert exc.value.status_code == 404


# ── The REST write path (same gate, different transport) ───────────────────────

UNSANDBOXED = [{"id": "danger", "type": "python", "sandbox": False, "code": "print(1)"}]


def _request(permissions=...):
    from types import SimpleNamespace

    state = SimpleNamespace()
    if permissions is not ...:
        state.permissions = permissions
    return SimpleNamespace(state=state)


def test_rest_write_path_refuses_an_unsandboxed_step_without_admin() -> None:
    """POST/PUT /workflows share this gate; WRITE must not be enough for it."""
    from fastapi import HTTPException

    from app.api.routes.workflows import _guard_sandbox

    with pytest.raises(HTTPException) as exc:
        _guard_sandbox(_request(WRITER), UNSANDBOXED)
    assert exc.value.status_code == 403
    assert "admin" in str(exc.value.detail).lower()


def test_rest_write_path_allows_an_unsandboxed_step_for_admin() -> None:
    from app.api.routes.workflows import _guard_sandbox

    _guard_sandbox(_request(frozenset(Permission)), UNSANDBOXED)


def test_rest_write_path_falls_back_to_full_access_without_middleware() -> None:
    """OAuth disabled: no middleware ran, so behaviour is the pre-RBAC one."""
    from app.api.routes.workflows import _guard_sandbox

    _guard_sandbox(_request(), UNSANDBOXED)
