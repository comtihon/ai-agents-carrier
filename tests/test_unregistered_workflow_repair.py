"""A workflow whose graph fails to build must stay repairable.

The loader logs and skips a definition it cannot build, so the workflow never
enters the registry — while the definition itself sits untouched in the backing
store. Resolution used to consult only the registry, which made that workflow
simultaneously unreadable by `get_workflow` and unreachable by
`update_workflow`: the two tools you need to diagnose and fix it. The
definition was never lost, only unresolvable, and the only way back was direct
database access.

This is not hypothetical. Removing the `llm_structured` step type took a live
workflow out of the registry, and it could not be inspected or repaired through
any management surface afterwards.
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
def _permissions():
    token = set_current_permissions({Permission.READ, Permission.WRITE})
    yield
    reset_current_permissions(token)


class _Store:
    """The backing store: holds every definition, buildable or not."""

    def __init__(self, items, *, list_raises=False, get_raises=False):
        self.items = {w.id: w for w in items}
        self.list_raises = list_raises
        self.get_raises = get_raises

    async def list(self):
        if self.list_raises:
            raise RuntimeError("store unavailable")
        return list(self.items.values())

    async def get(self, workflow_id):
        if self.get_raises:
            raise RuntimeError("store unavailable")
        return self.items.get(workflow_id)


class _Registry:
    """The registry: holds only what actually built."""

    def __init__(self, definitions):
        self._defs = [{"id": d.id, "name": d.name} for d in definitions]

    def list_definitions(self):
        return self._defs


def _wf(wf_id, name=None, steps=None):
    return WorkflowDefinition(
        id=wf_id,
        name=name or wf_id,
        steps=steps or [{"id": "one", "type": "storage", "action": "get", "key": "k"}],
    )


def _deps(*, registered, stored, **store_kw):
    return ManagementDeps(
        registry=_Registry(registered),  # type: ignore[arg-type]
        run_repository=None,  # type: ignore[arg-type]
        workflow_backend=_Store(stored, **store_kw),
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unregistered_workflow_resolves_from_the_store():
    broken = _wf("organize-hubspot")
    deps = _deps(registered=[], stored=[broken])

    resolved, err = await core._resolve_workflow_id(deps, "organize-hubspot")

    assert err is None
    assert resolved == "organize-hubspot"


@pytest.mark.asyncio
async def test_registry_still_wins_when_it_has_the_workflow():
    """The store is a fallback, not a replacement — no extra lookups when the
    registry already answers."""
    wf = _wf("healthy")
    store = _Store([wf], get_raises=True)  # would blow up if consulted
    deps = ManagementDeps(
        registry=_Registry([wf]),  # type: ignore[arg-type]
        run_repository=None,  # type: ignore[arg-type]
        workflow_backend=store,
    )

    resolved, err = await core._resolve_workflow_id(deps, "healthy")

    assert (resolved, err) == ("healthy", None)


@pytest.mark.asyncio
async def test_store_fallback_is_exact_id_only():
    """Fuzzy-matching a workflow that could not be built would be guessing."""
    deps = _deps(registered=[], stored=[_wf("organize-hubspot")])

    resolved, err = await core._resolve_workflow_id(deps, "hubspot")

    assert resolved is None
    assert err is not None


@pytest.mark.asyncio
async def test_a_genuinely_absent_workflow_still_reports_not_found():
    deps = _deps(registered=[_wf("healthy")], stored=[_wf("healthy")])

    resolved, err = await core._resolve_workflow_id(deps, "no-such-thing")

    assert resolved is None
    assert "not found" in err


@pytest.mark.asyncio
async def test_not_found_message_names_the_unbuildable_workflows():
    """Discovery matters: without this the operator cannot learn the id they
    need in order to use the exact-id fallback at all."""
    deps = _deps(registered=[_wf("healthy")], stored=[_wf("healthy"), _wf("broken-one")])

    _, err = await core._resolve_workflow_id(deps, "typo")

    assert "broken-one" in err
    assert "failed to build" in err
    # A workflow that built fine is not slandered as broken.
    assert "healthy (healthy)" in err.split("Stored but not registered")[0]


@pytest.mark.asyncio
async def test_no_broken_suffix_when_registry_and_store_agree():
    deps = _deps(registered=[_wf("healthy")], stored=[_wf("healthy")])

    _, err = await core._resolve_workflow_id(deps, "typo")

    assert "Stored but not registered" not in err


# ---------------------------------------------------------------------------
# Robustness — the fallback must not make failure worse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_get_failure_degrades_to_plain_not_found():
    deps = _deps(registered=[], stored=[_wf("organize-hubspot")], get_raises=True)

    resolved, err = await core._resolve_workflow_id(deps, "organize-hubspot")

    assert resolved is None
    assert "not found" in err


@pytest.mark.asyncio
async def test_store_list_failure_still_yields_a_usable_message():
    deps = _deps(registered=[], stored=[_wf("x")], list_raises=True, get_raises=True)

    _, err = await core._resolve_workflow_id(deps, "missing")

    assert "not found" in err
    assert "Stored but not registered" not in err


@pytest.mark.asyncio
async def test_resolution_works_without_a_workflow_backend():
    deps = ManagementDeps(
        registry=_Registry([]),  # type: ignore[arg-type]
        run_repository=None,  # type: ignore[arg-type]
        workflow_backend=None,
    )

    resolved, err = await core._resolve_workflow_id(deps, "anything")

    assert resolved is None
    assert "not found" in err


# ---------------------------------------------------------------------------
# The point of it all: read the broken definition, then repair it
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_workflow_can_read_an_unregistered_definition():
    steps = [{"id": "classify", "type": "llm_structured", "system_prompt": "..."}]
    deps = _deps(registered=[], stored=[_wf("organize-hubspot", steps=steps)])

    text = await core.get_workflow(deps, "organize-hubspot")

    assert "organize-hubspot" in text
    # The steps come back whole, which is what makes a repair composable.
    body = text.split("steps_json:\n", 1)[1]
    assert json.loads(body)[0]["type"] == "llm_structured"
