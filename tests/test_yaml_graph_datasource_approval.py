"""The `data_source` step's behaviour when the operation deletes.

The unit of behaviour here is the pause: a delete step must not call the
executor until somebody has approved the case, must resume into the real call
when they do, and must skip the call — without failing the run — when they
don't. The service and the executor preview are stubbed; what is under test is
the node's wiring of the two.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app.domain.models.approval_case import ApprovalCase
from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.datasources.executor import DestructivePlan
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.tools.mcp_client import McpToolsProvider

_SOURCE = DataSourceDefinition.model_validate({
    "id": "files",
    "name": "File store",
    "base_url": "https://files.test",
    "operations": [
        {"name": "stale", "path": "/files?stale=true"},
        {"name": "drop", "method": "DELETE", "path": "/files/{stale.id}"},
        {"name": "list", "path": "/files"},
    ],
})


def _runner(*, executor, service, operation: str = "drop") -> YamlGraphRunner:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="x")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    definition = {"id": "cleanup", "steps": [{
        "id": "purge",
        "type": "data_source",
        "source": "files",
        "operation": operation,
        "output_key": "removed",
    }]}
    runner = YamlGraphRunner(definition, llm=llm, mcp_tools_provider=mcp)
    backend = AsyncMock()
    backend.get = AsyncMock(return_value=_SOURCE)
    runner._data_source_backend = backend
    runner._data_source_executor = executor
    runner._approval_service = service
    runner._current_run = MagicMock(id="run-1")
    return runner


def _case(**overrides) -> ApprovalCase:
    data = dict(
        id="apr_1", status="pending", workflow_id="cleanup", run_id="run-1",
        step_id="purge", datasource_id="files", datasource_name="File store",
        operation="drop", method="DELETE", affected_rows=3,
        affected_sample=["f1", "f2", "f3"],
    )
    data.update(overrides)
    return ApprovalCase(**data)


def _service(case: ApprovalCase, *, existing: ApprovalCase | None = None) -> AsyncMock:
    service = AsyncMock()
    service.find_open_case = AsyncMock(return_value=existing)
    service.open_case = AsyncMock(return_value=case)
    service.decide = AsyncMock(
        side_effect=lambda cid, **kw: _case(
            id=cid,
            status="approved" if kw.get("approved") else "rejected",
            reason=kw.get("reason") or "",
        )
    )
    service.wait_out_veto = AsyncMock(side_effect=lambda c: c)
    return service


def _executor(rows: int = 3) -> AsyncMock:
    executor = AsyncMock()
    executor.preview = AsyncMock(return_value=DestructivePlan(
        affected_rows=rows,
        targets=[f"DELETE https://files.test/files/f{i}" for i in range(rows)],
        sample=[f"f{i}" for i in range(rows)],
    ))
    executor.execute = AsyncMock(return_value={"deleted": rows})
    return executor


_CONFIG = {"configurable": {"thread_id": "t-approval"}}


async def test_a_delete_pauses_before_calling_the_executor():
    executor, service = _executor(), _service(_case())
    runner = _runner(executor=executor, service=service)

    await runner.graph.ainvoke({"request": "clean"}, _CONFIG)

    executor.preview.assert_awaited_once()
    executor.execute.assert_not_awaited()
    snapshot = await runner.graph.aget_state(_CONFIG)
    assert snapshot.next == ("purge",)


async def test_the_interrupt_carries_the_row_count_the_reviewer_needs():
    runner = _runner(executor=_executor(), service=_service(_case()))
    await runner.graph.ainvoke({"request": "clean"}, _CONFIG)

    snapshot = await runner.graph.aget_state(_CONFIG)
    payload = snapshot.interrupts[0].value
    assert payload["type"] == "datasource_approval"
    assert payload["affected_rows"] == 3
    assert payload["case_id"] == "apr_1"
    assert payload["method"] == "DELETE"
    assert payload["affected_sample"] == ["f1", "f2", "f3"]


async def test_approving_runs_the_deletion():
    executor, service = _executor(), _service(_case())
    runner = _runner(executor=executor, service=service)
    await runner.graph.ainvoke({"request": "clean"}, _CONFIG)

    state = await runner.graph.ainvoke(
        Command(resume={"approved": True, "approver_name": "ada",
                        "approver_source": "slack"}),
        _CONFIG,
    )

    executor.execute.assert_awaited_once_with(_SOURCE, "drop", {}, limit=None)
    assert state["removed"] == {"deleted": 3}
    assert state["_approval_case_id"] == "apr_1"
    service.decide.assert_awaited_once()
    assert service.decide.await_args.kwargs["decided_by_name"] == "ada"
    assert service.decide.await_args.kwargs["source"] == "slack"


async def test_rejecting_skips_the_deletion_without_failing_the_run():
    executor, service = _executor(), _service(_case())
    runner = _runner(executor=executor, service=service)
    await runner.graph.ainvoke({"request": "clean"}, _CONFIG)

    state = await runner.graph.ainvoke(
        Command(resume={"approved": False, "reason": "wrong bucket"}), _CONFIG
    )

    executor.execute.assert_not_awaited()
    assert state["removed"]["skipped"] is True
    assert state["removed"]["reason"] == "wrong bucket"
    assert state["removed"]["affected_rows"] == 3


async def test_a_resume_reuses_the_open_case_instead_of_opening_another():
    """The node re-runs from the top on resume — it must not re-open the case.

    Opening a second one would also re-read the whole upstream list, so this is
    the difference between one preview and one per resume.
    """
    executor = _executor()
    case = _case()
    service = _service(case, existing=case)
    runner = _runner(executor=executor, service=service)

    await runner.graph.ainvoke({"request": "clean"}, _CONFIG)
    await runner.graph.ainvoke(Command(resume={"approved": True}), _CONFIG)

    service.open_case.assert_not_awaited()
    executor.preview.assert_not_awaited()


async def test_a_read_operation_is_never_gated():
    executor, service = _executor(), _service(_case())
    runner = _runner(executor=executor, service=service, operation="list")

    state = await runner.graph.ainvoke({"request": "clean"}, _CONFIG)

    service.open_case.assert_not_awaited()
    executor.execute.assert_awaited_once()
    assert state["removed"] == {"deleted": 3}


async def test_a_delete_that_matches_nothing_runs_unattended():
    executor, service = _executor(rows=0), _service(_case())
    runner = _runner(executor=executor, service=service)

    state = await runner.graph.ainvoke({"request": "clean"}, _CONFIG)

    # Asking a human to approve a no-op only teaches them to click Approve.
    service.open_case.assert_not_awaited()
    executor.execute.assert_awaited_once()
    assert state["removed"] == {"deleted": 0}


async def test_no_approval_service_leaves_the_step_ungated():
    executor = _executor()
    runner = _runner(executor=executor, service=None)

    state = await runner.graph.ainvoke({"request": "clean"}, _CONFIG)

    executor.preview.assert_not_awaited()
    executor.execute.assert_awaited_once()
    assert state["removed"] == {"deleted": 3}


async def test_an_autonomous_decision_waits_out_its_veto_window():
    from datetime import datetime, timedelta, timezone

    approved = _case(
        status="approved",
        decision_source="meta_llm",
        veto_deadline=datetime.now(timezone.utc) + timedelta(seconds=30),
    )
    executor = _executor()
    service = _service(approved)
    runner = _runner(executor=executor, service=service)

    state = await runner.graph.ainvoke({"request": "clean"}, _CONFIG)

    service.wait_out_veto.assert_awaited_once()
    executor.execute.assert_awaited_once()
    assert state["removed"] == {"deleted": 3}


async def test_a_veto_during_the_window_stops_the_deletion():
    from datetime import datetime, timedelta, timezone

    approved = _case(
        status="approved",
        decision_source="meta_llm",
        veto_deadline=datetime.now(timezone.utc) + timedelta(seconds=30),
    )
    executor = _executor()
    service = _service(approved)
    service.wait_out_veto = AsyncMock(
        return_value=_case(status="cancelled", reason="Cancelled by ada")
    )
    runner = _runner(executor=executor, service=service)

    state = await runner.graph.ainvoke({"request": "clean"}, _CONFIG)

    executor.execute.assert_not_awaited()
    assert state["removed"]["skipped"] is True
    assert "ada" in state["removed"]["reason"]
