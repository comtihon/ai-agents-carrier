"""The privilege gate in front of destructive data-source operations.

Covers the four things the feature promises: a delete is previewed rather than
run, the preview counts what it would remove, the approval record survives the
decision, and a long enough streak of identical human decisions is what — and
the only thing that — lets the meta-LLM answer alone.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.application.approval_service import ApprovalService
from app.domain.models.approval_case import ApprovalCase, MetaLlmVerdict, history_key_for
from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.datasources import executor as executor_module
from app.infrastructure.datasources.destructive import is_destructive
from app.infrastructure.datasources.executor import DataSourceExecutor
from app.infrastructure.persistence.approval_backend import InMemoryApprovalBackend


def _executor(**kwargs):
    """An executor with a throw-away stream store.

    Every data source result is written to a stream and returned as a
    reference, so an executor needs somewhere to write. Tests that assert on
    records call ``execute_value``, which reads the stream back.
    """
    import tempfile

    from app.infrastructure.datasources.datastream import LocalDiskStreamStore

    kwargs.setdefault("stream_store", LocalDiskStreamStore(tempfile.mkdtemp()))
    return DataSourceExecutor(**kwargs)


# ---------------------------------------------------------------------------
# httpx stub (same shape as tests/test_data_source_executor.py)
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, handler, calls: list[dict]) -> None:
        self._handler = handler
        self._calls = calls

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def request(self, method, url, params=None, headers=None, json=None):
        call = {"method": method, "url": url, "params": dict(params or {}), "json": json}
        self._calls.append(call)
        return FakeResponse(self._handler(call))

    async def post(self, url, json=None, headers=None):
        return await self.request("POST", url, json=json, headers=headers)


@pytest.fixture
def http(monkeypatch):
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.handler = lambda call: {}

    recorder = Recorder()
    monkeypatch.setattr(
        executor_module.httpx,
        "AsyncClient",
        lambda *a, **k: FakeClient(lambda call: recorder.handler(call), recorder.calls),
    )
    return recorder


def _source(**overrides) -> DataSourceDefinition:
    data = {"id": "api", "name": "API", "base_url": "https://api.test", "operations": []}
    data.update(overrides)
    return DataSourceDefinition.model_validate(data)


class _Settings:
    """Only the fields the service reads."""

    slack_bot_token = ""
    slack_approvals_channel = ""
    meta_llm_provider = "anthropic"
    meta_llm_model = "claude-test"
    llm_provider = "anthropic"
    approval_auto_decide_threshold = 10
    approval_veto_window_seconds = 0
    approval_wait_timeout_seconds = 5.0
    approval_poll_interval_seconds = 0.01


class _WorkflowBackend:
    def __init__(self, use_meta_llm: bool = True, name: str = "Nightly cleanup") -> None:
        self._use = use_meta_llm
        self._name = name

    async def get(self, workflow_id: str):
        class _Defn:
            id = workflow_id
            name = self._name
            use_meta_llm = self._use
        return _Defn()


def _service(backend=None, *, use_meta_llm=True, **settings_overrides) -> ApprovalService:
    settings = _Settings()
    for key, value in settings_overrides.items():
        setattr(settings, key, value)
    return ApprovalService(
        backend or InMemoryApprovalBackend(),
        settings,
        workflow_backend=_WorkflowBackend(use_meta_llm),
    )


# ---------------------------------------------------------------------------
# What counts as destructive
# ---------------------------------------------------------------------------

def test_delete_is_destructive_by_verb():
    source = _source(operations=[{"name": "drop", "method": "DELETE", "path": "/x/{params.id}",
                                  "params": [{"name": "id"}]}])
    assert is_destructive(source.get_operation("drop"), source) is True


def test_get_is_not_destructive():
    source = _source(operations=[{"name": "list", "path": "/x"}])
    assert is_destructive(source.get_operation("list"), source) is False


def test_explicit_flag_gates_a_post_and_frees_a_delete():
    source = _source(operations=[
        {"name": "purge", "method": "POST", "path": "/purge", "destructive": True},
        {"name": "clear_cache", "method": "DELETE", "path": "/cache", "destructive": False},
    ])
    assert is_destructive(source.get_operation("purge"), source) is True
    assert is_destructive(source.get_operation("clear_cache"), source) is False


def test_graphql_operations_are_only_destructive_when_stated():
    source = _source(kind="graphql", operations=[{"name": "mutate", "query": "mutation { x }"}])
    assert is_destructive(source.get_operation("mutate"), source) is False


# ---------------------------------------------------------------------------
# Preview: resolve everything except the call itself
# ---------------------------------------------------------------------------

async def test_preview_counts_fanout_rows_and_sends_no_delete(http):
    source = _source(operations=[
        {"name": "stale", "path": "/files?stale=true"},
        {"name": "drop", "method": "DELETE", "path": "/files/{stale.id}"},
    ])
    http.handler = lambda call: [{"id": "f1"}, {"id": "f2"}, {"id": "f3"}]

    plan = await _executor().preview(source, "drop", {})

    assert plan.affected_rows == 3
    assert plan.sample == ["f1", "f2", "f3"]
    assert plan.targets[0] == "DELETE https://api.test/files/f1"
    # The upstream read happened; the deletion did not.
    assert [c["method"] for c in http.calls] == ["GET"]


async def test_preview_of_a_single_targeted_delete_counts_one_row(http):
    source = _source(operations=[
        {"name": "drop", "method": "DELETE", "path": "/files/{params.id}",
         "params": [{"name": "id"}]},
    ])
    plan = await _executor().preview(source, "drop", {"id": "f9"})

    assert plan.affected_rows == 1
    assert plan.targets == ["DELETE https://api.test/files/f9"]
    assert http.calls == []


async def test_preview_of_an_empty_upstream_finds_nothing_to_delete(http):
    source = _source(operations=[
        {"name": "stale", "path": "/files?stale=true"},
        {"name": "drop", "method": "DELETE", "path": "/files/{stale.id}"},
    ])
    http.handler = lambda call: []

    plan = await _executor().preview(source, "drop", {})
    assert plan.affected_rows == 0


# ---------------------------------------------------------------------------
# Opening and deciding a case
# ---------------------------------------------------------------------------

async def _open(service, backend=None, **overrides):
    source = _source()
    kwargs = dict(
        source=source, operation="drop", method="DELETE", params={"id": "f1"},
        affected_rows=3, targets=["DELETE https://api.test/files/f1"], sample=["f1"],
        workflow_id="wf", run_id="run-1", step_id="delete-step",
    )
    kwargs.update(overrides)
    return await service.open_case(**kwargs)


async def test_open_case_records_the_blast_radius():
    backend = InMemoryApprovalBackend()
    case = await _open(_service(backend), backend)

    assert case.status == "pending"
    assert case.affected_rows == 3
    assert case.history_key == history_key_for("wf", "api", "drop")
    assert (await backend.get(case.id)).params == {"id": "f1"}


async def test_first_case_gets_no_recommendation_because_there_is_no_history():
    case = await _open(_service())
    assert case.meta_llm is None


async def test_decide_records_who_and_closes_the_case():
    backend = InMemoryApprovalBackend()
    service = _service(backend)
    case = await _open(service, backend)

    decided = await service.decide(
        case.id, approved=True, source="slack",
        decided_by_name="ada", decided_by_id="U1", reason="expected cleanup",
    )
    assert decided is not None
    assert decided.status == "approved"
    assert decided.decided_by_name == "ada"
    assert decided.decision_source == "slack"
    assert decided.decided_at is not None


async def test_a_second_decision_on_the_same_case_is_refused():
    backend = InMemoryApprovalBackend()
    service = _service(backend)
    case = await _open(service, backend)

    assert await service.decide(case.id, approved=True) is not None
    # The Slack button losing a race with the UI button must not overwrite the
    # answer that already went through.
    assert await service.decide(case.id, approved=False) is None


# ---------------------------------------------------------------------------
# The meta-LLM: recommendation, then autonomy
# ---------------------------------------------------------------------------

def _decided(n: int, status: str, *, source: str = "ui", key: str | None = None) -> list[ApprovalCase]:
    base = datetime.now(timezone.utc)
    return [
        ApprovalCase(
            id=f"apr_{status}_{i}",
            status=status,
            workflow_id="wf",
            datasource_id="api",
            operation="drop",
            history_key=key or history_key_for("wf", "api", "drop"),
            affected_rows=3,
            decision_source=source,
            decided_by_name="ada",
            decided_at=base - timedelta(minutes=i),
        )
        for i in range(n)
    ]


async def _seed(backend, cases):
    for case in cases:
        await backend.create(case)


def _stub_llm(monkeypatch, decision: str, reason: str = "matches prior decisions"):
    """Replace the meta-LLM call with a fixed verdict, recording its prompt."""
    seen: dict = {}

    async def _ask(self, case, history, *, autonomous):
        seen["autonomous"] = autonomous
        seen["history"] = history
        return MetaLlmVerdict(decision=decision, reason=reason, model="claude-test")

    monkeypatch.setattr(ApprovalService, "_ask_meta_llm", _ask)
    return seen


async def test_history_earns_an_advisory_recommendation(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(3, "approved"))
    seen = _stub_llm(monkeypatch, "approve")

    case = await _open(_service(backend), backend)

    assert case.status == "pending", "three prior cases is advice, not authority"
    assert case.meta_llm is not None
    assert case.meta_llm.decision == "approve"
    assert case.meta_llm.autonomous is False
    assert case.meta_llm.history_size == 3
    assert seen["autonomous"] is False


async def test_meta_llm_is_not_consulted_when_the_workflow_disabled_it(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(12, "approved"))
    _stub_llm(monkeypatch, "approve")

    case = await _open(_service(backend, use_meta_llm=False), backend)

    assert case.meta_llm is None
    assert case.status == "pending"


async def test_ten_identical_human_decisions_let_the_meta_llm_decide(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(10, "approved"))
    seen = _stub_llm(monkeypatch, "approve")

    case = await _open(_service(backend), backend)

    assert seen["autonomous"] is True
    assert case.status == "approved"
    assert case.decision_source == "meta_llm"
    assert case.meta_llm is not None and case.meta_llm.autonomous is True


async def test_a_streak_of_rejections_lets_it_reject(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(11, "rejected"))
    _stub_llm(monkeypatch, "reject")

    case = await _open(_service(backend), backend)
    assert case.status == "rejected"


async def test_a_broken_streak_goes_back_to_a_human(monkeypatch):
    backend = InMemoryApprovalBackend()
    history = _decided(10, "approved")
    # One rejection three cases back breaks the run — the streak is 3, not 10.
    history[3].status = "rejected"
    await _seed(backend, history)
    _stub_llm(monkeypatch, "approve")

    case = await _open(_service(backend), backend)
    assert case.status == "pending"


async def test_the_meta_llms_own_decisions_do_not_extend_the_streak(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(12, "approved", source="meta_llm"))
    _stub_llm(monkeypatch, "approve")

    case = await _open(_service(backend), backend)

    # Otherwise one early autonomous mistake would ratchet itself into policy.
    assert case.status == "pending"


async def test_disagreeing_with_the_streak_hands_the_case_back(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(10, "approved"))
    _stub_llm(monkeypatch, "reject", "row count is 200x the historical range")

    case = await _open(_service(backend), backend)

    assert case.status == "pending"
    assert case.meta_llm is not None
    assert case.meta_llm.decision == "reject"
    assert case.meta_llm.autonomous is False


async def test_a_failing_meta_llm_never_blocks_the_case(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(10, "approved"))

    async def _boom(self, case, history, *, autonomous):
        raise RuntimeError("provider down")

    monkeypatch.setattr(ApprovalService, "_ask_meta_llm", _boom)
    case = await _open(_service(backend), backend)

    assert case.status == "pending"
    assert case.meta_llm is None


async def test_history_is_scoped_to_the_workflow(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(
        12, "approved", key=history_key_for("other-wf", "api", "drop")
    ))
    _stub_llm(monkeypatch, "approve")

    case = await _open(_service(backend), backend)

    # Trust earned in one workflow is not trust in another.
    assert case.status == "pending"
    assert case.meta_llm is None


# ---------------------------------------------------------------------------
# The veto window
# ---------------------------------------------------------------------------

async def test_an_autonomous_decision_opens_a_veto_window(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(10, "approved"))
    _stub_llm(monkeypatch, "approve")

    case = await _open(_service(backend, approval_veto_window_seconds=30), backend)

    assert case.status == "approved"
    assert case.veto_deadline is not None


async def test_a_veto_inside_the_window_cancels_the_deletion(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(10, "approved"))
    _stub_llm(monkeypatch, "approve")
    service = _service(backend, approval_veto_window_seconds=30)
    case = await _open(service, backend)

    vetoed = await service.veto(case.id, by="ada")

    assert vetoed is not None
    assert vetoed.status == "cancelled"
    assert vetoed.vetoed_by == "ada"


async def test_a_veto_after_the_window_is_refused(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(10, "approved"))
    _stub_llm(monkeypatch, "approve")
    service = _service(backend, approval_veto_window_seconds=30)
    case = await _open(service, backend)
    case.veto_deadline = datetime.now(timezone.utc) - timedelta(seconds=1)
    await backend.update(case)

    assert await service.veto(case.id, by="ada") is None


async def test_a_human_decided_case_has_no_veto_window():
    backend = InMemoryApprovalBackend()
    service = _service(backend)
    case = await _open(service, backend)
    await service.decide(case.id, approved=True, decided_by_name="ada")

    assert await service.veto(case.id, by="someone else") is None


# ---------------------------------------------------------------------------
# Blocking wait (the agent MCP surface)
# ---------------------------------------------------------------------------

async def test_wait_for_decision_returns_once_somebody_answers():
    import asyncio

    backend = InMemoryApprovalBackend()
    service = _service(backend)
    case = await _open(service, backend)

    async def _answer():
        await asyncio.sleep(0.05)
        await service.decide(case.id, approved=True, decided_by_name="ada")

    asyncio.create_task(_answer())
    resolved = await service.wait_for_decision(case.id, timeout_seconds=2)

    assert resolved is not None and resolved.status == "approved"


async def test_wait_for_decision_expires_a_case_nobody_answers():
    backend = InMemoryApprovalBackend()
    service = _service(backend)
    case = await _open(service, backend)

    resolved = await service.wait_for_decision(case.id, timeout_seconds=0.05)

    assert resolved is not None
    assert resolved.status == "expired"
    assert resolved.decision_source == "timeout"


# ---------------------------------------------------------------------------
# Try run: the approver is already in the room
# ---------------------------------------------------------------------------

async def test_record_confirmed_writes_an_already_approved_case():
    backend = InMemoryApprovalBackend()
    service = _service(backend)

    case = await service.record_confirmed(
        source=_source(), operation="drop", method="DELETE",
        params={"id": "f1"}, affected_rows=4,
        targets=["DELETE https://api.test/files/f1"], sample=["f1"],
        decided_by_name="ada", decided_by_id="U1",
    )

    assert case.status == "approved"
    assert case.surface == "try_run"
    assert case.decided_by_name == "ada"
    assert case.decided_at is not None
    assert case.affected_rows == 4


async def test_record_confirmed_never_consults_the_meta_llm(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _decided(12, "approved"))
    called = {"n": 0}

    async def _ask(self, case, history, *, autonomous):
        called["n"] += 1
        return MetaLlmVerdict(decision="reject", reason="no")

    monkeypatch.setattr(ApprovalService, "_ask_meta_llm", _ask)
    case = await _service(backend).record_confirmed(
        source=_source(), operation="drop", method="DELETE", params={},
        affected_rows=1, targets=[], sample=[],
    )

    # The person clicking has already seen the row count; a model that
    # disagreed could only be ignored or overrule them.
    assert called["n"] == 0
    assert case.status == "approved"


async def test_try_run_confirmations_stay_out_of_the_decision_history():
    backend = InMemoryApprovalBackend()
    service = _service(backend)
    for _ in range(12):
        await service.record_confirmed(
            source=_source(), operation="drop", method="DELETE", params={},
            affected_rows=1, targets=[], sample=[], decided_by_name="ada",
        )

    key = history_key_for("", "api", "drop")
    # Otherwise an author could grant the meta-LLM autonomy over their own
    # delete by clicking Try run ten times.
    assert await backend.history(key) == []
    # They are still in the audit trail.
    assert await backend.count() == 12


# ---------------------------------------------------------------------------
# Tier-2 write probation
# ---------------------------------------------------------------------------
# A generated-code write is held to a person for its first N runs regardless of
# any autonomy streak. The streak means "somebody has waved this operation
# through ten times"; for code a model wrote that is not yet the same claim,
# because a golden fixture over five sample rows is not evidence over five
# hundred real ones -- and the rows the model never saw are exactly where a
# transform goes wrong.

def _tier2_source():
    """A google-sheets source whose `update_project` operation is tier 2."""
    from app.domain.models.sheet_binding import SheetBinding
    from tests.test_sheet_bindings_api import _sheets_source
    from tests.test_sheet_compute import _compute_write_binding

    return _sheets_source().model_copy(update={
        "bindings": [SheetBinding.model_validate(_compute_write_binding())],
    })


def _tier2_decided(n: int, *, source: str = "ui") -> list[ApprovalCase]:
    base = datetime.now(timezone.utc)
    return [
        ApprovalCase(
            id=f"apr_t2_{source}_{i}",
            status="approved",
            workflow_id="wf",
            datasource_id="google-sheets",
            operation="update_project",
            history_key=history_key_for("wf", "google-sheets", "update_project"),
            affected_rows=1,
            decision_source=source,
            decided_by_name="ada",
            decided_at=base - timedelta(minutes=i),
        )
        for i in range(n)
    ]


async def _open_tier2(service, backend):
    return await service.open_case(
        source=_tier2_source(),
        operation="update_project",
        method="POST",
        params={"project_id": "P-1"},
        affected_rows=1,
        targets=["POST https://sheets.googleapis.com/.../values:batchUpdate"],
        sample=["P-1"],
        workflow_id="wf",
        run_id="run-1",
        step_id="write-step",
    )


async def test_a_tier2_write_is_held_for_review_during_probation(monkeypatch):
    """A long streak does not buy a generated write its autonomy yet."""
    backend = InMemoryApprovalBackend()
    # Ten human approvals: enough for any hand-written operation to go autonomous.
    await _seed(backend, _tier2_decided(10))
    seen = _stub_llm(monkeypatch, "approve")

    case = await _open_tier2(
        _service(backend, sheets_compute_write_probation_runs=20), backend
    )

    # The streak still made the meta-LLM answer with authority...
    assert seen["autonomous"] is True
    # ...and probation took it away again.
    assert case.status == "pending"
    assert case.decision_source is None
    assert case.meta_llm is not None
    assert case.meta_llm.autonomous is False
    # The reason says why, so the withholding does not look arbitrary.
    assert "held for review" in case.meta_llm.reason
    assert "10 of 20 human approvals" in case.meta_llm.reason


async def test_a_tier2_write_goes_autonomous_once_probation_is_served(monkeypatch):
    backend = InMemoryApprovalBackend()
    await _seed(backend, _tier2_decided(10))
    _stub_llm(monkeypatch, "approve")

    case = await _open_tier2(
        _service(backend, sheets_compute_write_probation_runs=5), backend
    )

    assert case.status == "approved"
    assert case.decision_source == "meta_llm"
    assert case.meta_llm is not None and case.meta_llm.autonomous is True


async def test_probation_counts_only_human_approvals(monkeypatch):
    """The meta-LLM's own autonomous calls do not shorten probation.

    Otherwise the model reads its own output back as evidence of safety and
    ratchets one early mistake into a standing policy -- the same reasoning
    that makes ``_streak`` ignore its own decisions, applied to the count.

    The ten human approvals here are the *newest*, so the streak is intact and
    the meta-LLM does get authority; probation is then the only thing that
    takes it away, which is what isolates this to the counting rule.
    """
    backend = InMemoryApprovalBackend()
    base = datetime.now(timezone.utc)
    human = [
        ApprovalCase(
            id=f"apr_t2_human_{i}", status="approved", workflow_id="wf",
            datasource_id="google-sheets", operation="update_project",
            history_key=history_key_for("wf", "google-sheets", "update_project"),
            affected_rows=1, decision_source="ui", decided_by_name="ada",
            decided_at=base - timedelta(minutes=i),
        )
        for i in range(10)
    ]
    older_autonomous = [
        ApprovalCase(
            id=f"apr_t2_bot_{i}", status="approved", workflow_id="wf",
            datasource_id="google-sheets", operation="update_project",
            history_key=history_key_for("wf", "google-sheets", "update_project"),
            affected_rows=1, decision_source="meta_llm",
            decided_at=base - timedelta(days=1, minutes=i),
        )
        for i in range(12)
    ]
    await _seed(backend, human + older_autonomous)
    seen = _stub_llm(monkeypatch, "approve")

    case = await _open_tier2(
        _service(backend, sheets_compute_write_probation_runs=15), backend
    )

    assert seen["autonomous"] is True, "the human streak is intact"
    # 22 approvals in total, but only the 10 human ones count.
    assert case.status == "pending"
    assert case.meta_llm is not None and case.meta_llm.autonomous is False
    assert "10 of 15 human approvals" in case.meta_llm.reason


async def test_a_tier2_write_with_no_history_at_all_is_held(monkeypatch):
    """The first run of a generated write always goes to a person."""
    backend = InMemoryApprovalBackend()
    _stub_llm(monkeypatch, "approve")

    case = await _open_tier2(
        _service(backend, sheets_compute_write_probation_runs=5), backend
    )

    assert case.status == "pending"
    # No history means the meta-LLM is not consulted at all, so there is
    # nothing for probation to override -- it is held by the earlier rule, and
    # the outcome is the one that matters.
    assert case.meta_llm is None


async def test_probation_does_not_touch_a_tier1_write(monkeypatch):
    """The gate is about generated code, not about writes in general."""
    from app.domain.models.sheet_binding import SheetBinding
    from tests.test_sheet_bindings_api import _sheets_source, _write_binding

    backend = InMemoryApprovalBackend()
    await _seed(backend, _tier2_decided(10))
    _stub_llm(monkeypatch, "approve")
    tier1 = _sheets_source().model_copy(update={
        "bindings": [SheetBinding.model_validate(_write_binding())],
    })

    case = await _service(
        backend, sheets_compute_write_probation_runs=20
    ).open_case(
        source=tier1,
        operation="update_project",
        method="POST",
        params={"project_id": "P-1"},
        affected_rows=1,
        targets=["POST https://sheets.googleapis.com/"],
        sample=["P-1"],
        workflow_id="wf",
        run_id="run-1",
    )

    assert case.status == "approved"
    assert case.decision_source == "meta_llm"


# ---------------------------------------------------------------------------
# What an approver reads: a write is not a deletion
# ---------------------------------------------------------------------------

def _rendered(case: ApprovalCase, mode: str = "request") -> str:
    from app.infrastructure.notifications.webhook_notifier import _approval_blocks

    import json as _json
    return _json.dumps(_approval_blocks(case, mode=mode))


def test_a_delete_still_reads_as_a_deletion_of_rows():
    """The default wording is unchanged — every pre-existing case is a delete."""
    text = _rendered(ApprovalCase(id="apr_1", affected_rows=12, affected_sample=["42"]))
    assert "Data deletion awaiting approval" in text
    assert "12 rows" in text
    assert "cell" not in text
    # Value-free: the Slack message states the shape, never the data.
    assert "Targets" not in text


def test_a_write_reads_as_a_write_of_cells():
    """Calling an overwrite a deletion tells the approver the data is going away."""
    case = ApprovalCase(
        id="apr_2",
        affected_rows=3,
        affected_sample=["Projects!B2 (status): 'open' -> 'closed'"],
        change_kind="write",
    )
    text = _rendered(case)
    assert "Spreadsheet write awaiting approval" in text
    assert "3 cells" in text
    assert "deletion" not in text
    # Value-free: the before/after cell values stay out of Slack.
    assert "Changes" not in text


def test_one_cell_is_singular():
    case = ApprovalCase(id="apr_3", affected_rows=1, change_kind="write")
    assert "1 cell`" in _rendered(case)


def test_details_reach_the_approver():
    """Which document, and whether a model wrote the values, are shown."""
    case = ApprovalCase(
        id="apr_4",
        affected_rows=1,
        affected_sample=["Projects!B2: 'a' -> 'b'"],
        change_kind="write",
        details={
            "Document": "RC Projects Tracker",
            "Tab": "Projects",
            "Values from": "generated code written by some-model",
        },
    )
    text = _rendered(case)
    # Only the provenance detail survives the trim -- it is a safety signal,
    # not a data value. The document name and the rest of `details` do not.
    assert "generated code written by some-model" in text
    assert "RC Projects Tracker" not in text


def test_the_meta_llm_is_asked_about_a_write_not_a_deletion():
    """Asking the wrong question biases the answer, so the prompt adapts."""
    from app.application.approval_service import _build_prompt

    case = ApprovalCase(
        id="apr_5",
        affected_rows=2,
        affected_sample=["Projects!B2: 'a' -> 'b'"],
        change_kind="write",
        details={"Document": "RC Projects Tracker"},
    )
    prompt = _build_prompt(case, [], autonomous=True)
    assert "whether this write runs" in prompt
    assert "deletion" not in prompt
    assert "Cells affected: 2" in prompt
    assert "RC Projects Tracker" in prompt

    delete = ApprovalCase(id="apr_6", affected_rows=2)
    delete_prompt = _build_prompt(delete, [], autonomous=True)
    assert "whether this deletion runs" in delete_prompt
    assert "Rows affected: 2" in delete_prompt


# ---------------------------------------------------------------------------
# the Slack message must not carry data values
# ---------------------------------------------------------------------------
#
# It used to post `Targets` (rendered request URLs, so record ids), `Input`
# (the operation's params verbatim) and `Changes` (actual before/after cell
# values) into a Slack channel -- CRM fields and spreadsheet contents, to
# answer a question that only needs "how much, and where". The authenticated
# surfaces still show all of it; a channel is a wider audience.

def test_the_slack_message_carries_no_row_values():
    case = ApprovalCase(
        id="apr_leak",
        datasource_id="hubspot-crm",
        datasource_name="HubSpot CRM",
        operation="archive_object",
        method="DELETE",
        workflow_id="cleanup",
        workflow_name="Contact cleanup",
        run_id="r1",
        step_id="s1",
        affected_rows=3,
        endpoint="DELETE https://api.hubapi.com/crm/v3/objects/contacts/8801",
        targets=[
            "DELETE https://api.hubapi.com/crm/v3/objects/contacts/8801",
            "DELETE https://api.hubapi.com/crm/v3/objects/contacts/8802",
        ],
        affected_sample=["ada@example.com", "grace@example.com"],
        params={"objectType": "contacts", "email": "ada@example.com"},
    )

    text = _rendered(case)

    # The shape is there.
    assert "3 rows" in text
    assert "Contact cleanup" in text
    assert "archive_object" in text
    # The data is not -- by value, by id, and by label.
    for leaked in (
        "ada@example.com", "grace@example.com", "8801", "8802",
        "api.hubapi.com", "Targets", "Input", "Endpoint",
    ):
        assert leaked not in text, f"{leaked!r} must not reach Slack"


def test_the_slack_message_names_the_row_without_its_contents():
    case = ApprovalCase(
        id="apr_row",
        datasource_id="google-sheets",
        datasource_name="Google Sheets",
        operation="write_status",
        method="POST",
        workflow_id="tracker",
        workflow_name="Status tracker",
        run_id="r2",
        step_id="s2",
        affected_rows=2,
        change_kind="write",
        affected_rows_label="row 7",
        affected_sample=["B7: 'draft' -> 'delivered'"],
    )

    text = _rendered(case)

    assert "row 7" in text
    assert "2 cells" in text
    assert "delivered" not in text, "the new cell value must not reach Slack"
    assert "draft" not in text


def test_the_meta_llm_is_told_not_to_quote_values_in_its_reason():
    """Its REASON is posted to Slack, and it sees the sample and the inputs.

    Rewriting the message blocks alone would not have closed the leak: a model
    that has been shown the values will happily quote them back in one line of
    free text.
    """
    from app.application import approval_service as svc

    prompt_builder = next(
        getattr(svc, name) for name in dir(svc)
        if name.startswith("_") and "prompt" in name.lower()
        and callable(getattr(svc, name))
    )
    import inspect

    source = inspect.getsource(prompt_builder)

    assert "posted to a Slack channel" in source
    assert "do NOT quote" in source
