"""Management tools for the script library.

`python` steps reference reusable scripts by `script_id`, and create_workflow
captures inline bodies into the library automatically -- so the library fills up
on its own, but there was no tool to read, replace or clear it. Dead scripts left
behind by deleted workflows could only be removed through the UI.
"""
from __future__ import annotations

import pytest

from app.application import management_tools as core
from app.application.management_tools import ManagementDeps
from app.domain.models.script_definition import ScriptDefinition
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.auth.authorization import Permission, set_current_permissions


@pytest.fixture(autouse=True)
def _all_permissions():
    """The tools are gated; grant everything so these tests exercise behaviour."""
    token = set_current_permissions({Permission.READ, Permission.WRITE, Permission.DELETE})
    yield
    from app.infrastructure.auth.authorization import reset_current_permissions
    reset_current_permissions(token)


class _Scripts:
    def __init__(self, items=None):
        self.items = {s.id: s for s in (items or [])}

    async def list(self):
        return list(self.items.values())

    async def get(self, script_id):
        return self.items.get(script_id)

    async def get_by_name(self, name):
        return next((s for s in self.items.values() if s.name == name), None)

    async def create(self, defn):
        self.items[defn.id] = defn
        return defn

    async def update(self, script_id, defn):
        self.items[script_id] = defn
        return defn

    async def delete(self, script_id):
        self.items.pop(script_id, None)


class _Workflows:
    def __init__(self, items=None):
        self.items = items or []

    async def list(self):
        return self.items


def _deps(scripts=None, workflows=None):
    return ManagementDeps(
        registry=None,  # type: ignore[arg-type]
        run_repository=None,  # type: ignore[arg-type]
        script_backend=scripts,
        workflow_backend=workflows,
    )


def _script(sid="s1", name="One", code="output = 1"):
    return ScriptDefinition(id=sid, name=name, code=code)


@pytest.mark.asyncio
async def test_list_reports_size_not_the_body():
    """A library of thousand-line scripts must not flood the reply."""
    backend = _Scripts([_script(code="x = 1\ny = 2\noutput = x + y")])

    text = await core.list_scripts(_deps(backend))

    assert "s1" in text and "3 lines" in text
    assert "output = x + y" not in text


@pytest.mark.asyncio
async def test_get_returns_the_code_and_can_omit_it():
    backend = _Scripts([_script(code="output = 42")])

    assert "output = 42" in await core.get_script(_deps(backend), "s1")
    assert "output = 42" not in await core.get_script(_deps(backend), "s1", include_code=False)


@pytest.mark.asyncio
async def test_a_script_is_reachable_by_name_as_well_as_id():
    backend = _Scripts([_script(sid="s1", name="Nice Name")])

    assert "Script: s1" in await core.get_script(_deps(backend), "Nice Name")


@pytest.mark.asyncio
async def test_create_refuses_to_overwrite_silently():
    backend = _Scripts([_script(code="original")])

    msg = await core.create_script(_deps(backend), "s1", "One", "replacement")

    assert "already exists" in msg
    assert backend.items["s1"].code == "original", "the body must be untouched"


@pytest.mark.asyncio
async def test_update_reports_the_size_change():
    """A script that silently shrank to nothing is worth noticing at the edit."""
    backend = _Scripts([_script(code="a" * 100)])

    msg = await core.update_script(_deps(backend), "s1", code="b")

    assert "100 -> 1 chars" in msg
    assert backend.items["s1"].code == "b"


@pytest.mark.asyncio
async def test_update_leaves_omitted_fields_alone():
    backend = _Scripts([_script(code="keep me", name="Keep")])

    await core.update_script(_deps(backend), "s1", description="new words")

    assert backend.items["s1"].code == "keep me"
    assert backend.items["s1"].name == "Keep"


@pytest.mark.asyncio
async def test_delete_is_refused_while_a_step_still_references_it():
    """Deleting under a live step turns it into a confusing runtime failure."""
    backend = _Scripts([_script()])
    workflows = _Workflows([
        WorkflowDefinition(id="wf", steps=[{"id": "run_it", "type": "python", "script_id": "s1"}])
    ])

    msg = await core.delete_script(_deps(backend, workflows), "s1")

    assert "still referenced by wf:run_it" in msg
    assert "s1" in backend.items, "nothing may be deleted"


@pytest.mark.asyncio
async def test_delete_proceeds_when_nothing_references_it():
    backend = _Scripts([_script()])
    workflows = _Workflows([
        WorkflowDefinition(id="wf", steps=[{"id": "other", "type": "python", "script_id": "s2"}])
    ])

    msg = await core.delete_script(_deps(backend, workflows), "s1")

    assert "deleted" in msg
    assert "s1" not in backend.items


@pytest.mark.asyncio
async def test_missing_backend_says_so_rather_than_raising():
    assert "not configured" in await core.list_scripts(_deps(None))
