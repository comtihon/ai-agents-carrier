"""``get_run`` must survive every run and show what the run produced.

Two defects motivated these tests. It read ``run.error``, which GraphRun does not
define, so the tool raised AttributeError for *every* run including successful
ones. And it rendered only ``.error`` / ``.status`` sub-keys, so a step whose
output had any other shape appeared as a bare key name -- leaving the tool unable
to answer the one question it exists for.
"""
from __future__ import annotations

import pytest

from app.application.management_tools import ManagementDeps, get_run
from app.domain.models.graph_run import GraphRun


class _Repo:
    def __init__(self, run: GraphRun | None) -> None:
        self._run = run

    async def get(self, run_id: str) -> GraphRun | None:
        return self._run


def _deps(run: GraphRun | None) -> ManagementDeps:
    # `registry` is required but unused by get_run.
    return ManagementDeps(registry=None, run_repository=_Repo(run))  # type: ignore[arg-type]


def _run(**kw) -> GraphRun:
    kw.setdefault("status", "completed")
    return GraphRun(id="r1", graph_id="wf", **kw)


@pytest.mark.asyncio
async def test_a_successful_run_renders_without_raising():
    """GraphRun has no `error` attribute; touching one broke every call."""
    text = await get_run(_deps(_run(state={"report": {"ok": True}})), "r1")

    assert "Run: r1" in text
    assert "completed" in text


@pytest.mark.asyncio
async def test_state_values_are_shown_not_just_key_names():
    state = {"afp_report": {"row_count": 1, "present": {"notes": True}}}

    text = await get_run(_deps(_run(state=state)), "r1")

    assert "row_count" in text and "notes" in text, "the value itself must appear"


@pytest.mark.asyncio
async def test_a_large_value_is_truncated_with_its_size_noted():
    text = await get_run(_deps(_run(state={"blob": "x" * 5000})), "r1")

    assert "chars total]" in text
    assert len(text) < 3000, "the response must stay bounded"


@pytest.mark.asyncio
async def test_internal_keys_are_hidden_but_the_fail_sentinel_is_reported():
    state = {"_private": 1, "__failed_step__": "fetch_afp", "error": "boom"}

    text = await get_run(_deps(_run(status="failed", state=state)), "r1")

    assert "_private" not in text
    assert "Failed step: fetch_afp" in text
    assert "Error: boom" in text


@pytest.mark.asyncio
async def test_key_count_is_capped_and_the_remainder_is_counted():
    state = {f"k{i}": i for i in range(30)}

    text = await get_run(_deps(_run(state=state)), "r1")

    assert "more key(s) not shown" in text


@pytest.mark.asyncio
async def test_a_missing_run_says_so():
    assert "not found" in await get_run(_deps(None), "nope")
