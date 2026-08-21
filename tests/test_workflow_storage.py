"""Per-workflow storage: the isolation is the feature.

A workflow may only reach its own entries. That is not enforced by a check a
future refactor could drop -- ``workflow_id`` is simply never a parameter a step
can set, it comes from the runner that owns the step. These tests pin both
halves: the backend keys by owner, and the node passes its own id and nothing
else.
"""
from __future__ import annotations

import pytest

from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.persistence.workflow_storage import (
    MAX_VALUE_BYTES,
    InMemoryWorkflowStorageBackend,
    check_value_size,
)


class _FakeLLM:
    """Stands in for the chat model: no ``storage`` path ever calls it."""


def _runner(workflow_id: str, steps: list[dict], *, use_storage: bool = True,
            backend=None) -> YamlGraphRunner:
    runner = YamlGraphRunner(
        {"id": workflow_id, "steps": steps, "use_storage": use_storage},
        _FakeLLM(),
        lambda *a, **k: [],
    )
    if backend is not None:
        runner._storage_backend = backend
    return runner


def _node(runner: YamlGraphRunner, step: dict):
    return runner._storage_node(step)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_same_key_in_two_workflows_does_not_collide():
    backend = InMemoryWorkflowStorageBackend()

    await backend.set("wf-a", "alert-state", {"owner": "a"})
    await backend.set("wf-b", "alert-state", {"owner": "b"})

    assert await backend.get("wf-a", "alert-state") == {"owner": "a"}
    assert await backend.get("wf-b", "alert-state") == {"owner": "b"}


@pytest.mark.asyncio
async def test_keys_and_clear_are_scoped_to_one_owner():
    backend = InMemoryWorkflowStorageBackend()
    await backend.set("wf-a", "one", 1)
    await backend.set("wf-a", "two", 2)
    await backend.set("wf-b", "one", 99)

    assert await backend.keys("wf-a") == ["one", "two"]
    assert await backend.keys("wf-b") == ["one"]

    assert await backend.clear("wf-a") == 2
    assert await backend.keys("wf-a") == []
    assert await backend.get("wf-b", "one") == 99, "clearing A must not touch B"


@pytest.mark.asyncio
async def test_absent_key_reads_as_none_not_an_error():
    """A first run has no state yet; that is normal."""
    backend = InMemoryWorkflowStorageBackend()

    assert await backend.get("wf-a", "never-written") is None


def test_oversized_value_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="over the"):
        check_value_size({"blob": "x" * (MAX_VALUE_BYTES + 1)})


def test_unserialisable_value_is_refused():
    """``default=str`` coerces most odd values (datetimes, ObjectIds), which is
    wanted. A circular reference is the case it genuinely cannot encode."""
    circular: dict = {}
    circular["self"] = circular

    with pytest.raises(ValueError, match="not JSON-serialisable"):
        check_value_size(circular)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_step_cannot_reach_another_workflows_entry():
    """The point of the whole design: no key name gets you into wf-b."""
    backend = InMemoryWorkflowStorageBackend()
    await backend.set("wf-b", "secret", "b's data")

    runner = _runner("wf-a", [{"id": "load", "type": "storage"}], backend=backend)
    # Try the composite id, and a traversal-flavoured key, as a step author might.
    for key in ("secret", "wf-b::secret", "../wf-b/secret"):
        node = _node(runner, {"id": "load", "action": "get", "key": key})
        assert (await node({}))["load"] is None, f"{key!r} must not resolve"


@pytest.mark.asyncio
async def test_step_config_cannot_claim_a_different_owner():
    """The owner comes from the runner, never from the step.

    Pinned separately from the key-name test because the tempting regression is
    not a crafted key -- it is someone adding a `workflow_id` (or `graph_id`,
    or `owner`) field to the step config and threading it into the backend call.
    """
    backend = InMemoryWorkflowStorageBackend()
    await backend.set("wf-b", "secret", "b's data")
    runner = _runner("wf-a", [{"id": "load", "type": "storage"}], backend=backend)

    for field in ("workflow_id", "graph_id", "owner", "source"):
        node = _node(runner, {
            "id": "load", "action": "get", "key": "secret", field: "wf-b",
        })
        assert (await node({}))["load"] is None, f"step config {field!r} must be ignored"

    # And a write must land under wf-a regardless of what the config claims.
    await _node(runner, {
        "id": "s", "action": "set", "key": "mine", "value": 1, "workflow_id": "wf-b",
    })({})
    assert await backend.keys("wf-a") == ["mine"]
    assert await backend.keys("wf-b") == ["secret"], "wf-b must be untouched"


@pytest.mark.asyncio
async def test_set_then_get_round_trips_within_one_workflow():
    backend = InMemoryWorkflowStorageBackend()
    runner = _runner("wf-a", [{"id": "s", "type": "storage"}], backend=backend)

    saved = await _node(runner, {
        "id": "s", "action": "set", "key": "alert-state", "value": {"129142": "2026-08-03"},
    })({})
    assert saved["s"] == {"key": "alert-state", "saved": True}

    got = await _node(runner, {"id": "g", "action": "get", "key": "alert-state"})({})
    assert got["g"] == {"129142": "2026-08-03"}


@pytest.mark.asyncio
async def test_key_and_value_are_state_templated():
    backend = InMemoryWorkflowStorageBackend()
    runner = _runner("wf-a", [{"id": "s", "type": "storage"}], backend=backend)

    await _node(runner, {
        "id": "s", "action": "set", "key": "run-{today}", "value": {"at": "{today}"},
    })({"today": "2026-08-21"})

    assert await backend.keys("wf-a") == ["run-2026-08-21"]
    assert await backend.get("wf-a", "run-2026-08-21") == {"at": "2026-08-21"}


@pytest.mark.asyncio
async def test_storage_switched_off_fails_loudly_rather_than_silently():
    """A no-op here would look like "state didn't persist" for weeks."""
    backend = InMemoryWorkflowStorageBackend()
    runner = _runner("wf-a", [{"id": "s", "type": "storage"}],
                     use_storage=False, backend=backend)

    result = await _node(runner, {"id": "s", "action": "set", "key": "k", "value": 1})({})

    assert "switched off" in result["s"]["error"]
    assert result["__failed_step__"] == "s"
    assert await backend.keys("wf-a") == [], "nothing may be written"


@pytest.mark.asyncio
async def test_unknown_action_and_empty_key_are_rejected():
    backend = InMemoryWorkflowStorageBackend()
    runner = _runner("wf-a", [{"id": "s", "type": "storage"}], backend=backend)

    bad_action = await _node(runner, {"id": "s", "action": "increment", "key": "k"})({})
    assert "unknown storage action" in bad_action["s"]["error"]

    no_key = await _node(runner, {"id": "s", "action": "get", "key": "{missing}"})({})
    assert "non-empty 'key'" in no_key["s"]["error"]


@pytest.mark.asyncio
async def test_keys_action_lists_only_this_workflows_keys():
    backend = InMemoryWorkflowStorageBackend()
    await backend.set("wf-a", "mine", 1)
    await backend.set("wf-b", "theirs", 2)
    runner = _runner("wf-a", [{"id": "k", "type": "storage"}], backend=backend)

    assert (await _node(runner, {"id": "k", "action": "keys"})({}))["k"] == ["mine"]


@pytest.mark.asyncio
async def test_delete_removes_only_the_named_entry():
    backend = InMemoryWorkflowStorageBackend()
    await backend.set("wf-a", "gone", 1)
    await backend.set("wf-a", "kept", 2)
    runner = _runner("wf-a", [{"id": "d", "type": "storage"}], backend=backend)

    await _node(runner, {"id": "d", "action": "delete", "key": "gone"})({})

    assert await backend.keys("wf-a") == ["kept"]
