"""`get_workflow` — the read that was missing.

Workflows were the only resource with no `get`: agents, data sources, scripts and
events all had one. That mattered more here than anywhere else, because
`update_workflow` replaces the *entire* step list, so an update had to be
composed without being able to see what it was replacing.

It also exposes `use_storage`, which was otherwise unobservable — a `storage`
step in a workflow whose flag is off fails at run time, and there was no way to
check the flag first.
"""
from __future__ import annotations

import json

import pytest

from app.application import management_tools as core
from app.application.management_tools import ManagementDeps
from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.auth.authorization import (
    Permission,
    reset_current_permissions,
    set_current_permissions,
)


@pytest.fixture(autouse=True)
def _read_permission():
    token = set_current_permissions({Permission.READ})
    yield
    reset_current_permissions(token)


class _Workflows:
    def __init__(self, items):
        self.items = {w.id: w for w in items}

    async def list(self):
        return list(self.items.values())

    async def get(self, workflow_id):
        return self.items.get(workflow_id)


class _Registry:
    """`_resolve_workflow_id` resolves id-or-name against the registry, so the
    stub has to mirror the definitions the backend holds."""

    def __init__(self, definitions):
        self._defs = [{"id": d.id, "name": d.name} for d in definitions]

    def list_definitions(self):
        return self._defs


def _deps(*definitions):
    return ManagementDeps(
        registry=_Registry(definitions),  # type: ignore[arg-type]
        run_repository=None,  # type: ignore[arg-type]
        workflow_backend=_Workflows(definitions),
    )


def _wf(**kw):
    kw.setdefault("id", "wf")
    kw.setdefault("name", "A workflow")
    kw.setdefault("steps", [{"id": "one", "type": "storage", "action": "get", "key": "k"}])
    return WorkflowDefinition(**kw)


@pytest.mark.asyncio
async def test_the_steps_come_back_whole_so_they_can_be_fed_to_update():
    steps = [
        {"id": "a", "type": "storage", "action": "get", "key": "cfg"},
        {"id": "b", "type": "slack", "action": "post", "channel": "C1", "text": "hi"},
    ]
    text = await core.get_workflow(_deps(_wf(steps=steps)), "wf")

    body = text.split("steps_json:\n", 1)[1]
    assert json.loads(body) == steps, "must round-trip into update_workflow"


@pytest.mark.asyncio
async def test_use_storage_is_visible():
    """The flag that decides whether `storage` steps work at all."""
    on = await core.get_workflow(_deps(_wf(id="wf", use_storage=True)), "wf")
    off = await core.get_workflow(_deps(_wf(id="wf", use_storage=False)), "wf")

    assert "Use storage: True" in on
    assert "Use storage: False" in off


@pytest.mark.asyncio
async def test_enabled_is_visible():
    text = await core.get_workflow(_deps(_wf(enabled=False)), "wf")

    assert "Enabled: False" in text


@pytest.mark.asyncio
async def test_step_ids_and_types_are_summarised():
    text = await core.get_workflow(_deps(_wf()), "wf")

    assert "Steps: 1" in text and "one(storage)" in text


@pytest.mark.asyncio
async def test_include_steps_false_omits_the_body_but_keeps_the_flags():
    text = await core.get_workflow(_deps(_wf()), "wf", include_steps=False)

    assert "steps_json:" not in text
    assert "Use storage:" in text


@pytest.mark.asyncio
async def test_lookup_by_name_works_like_the_other_getters():
    text = await core.get_workflow(_deps(_wf(id="wf", name="Nice Name")), "Nice Name")

    assert "Workflow: wf" in text


@pytest.mark.asyncio
async def test_a_missing_workflow_says_so():
    assert "not found" in await core.get_workflow(_deps(_wf()), "nope")
