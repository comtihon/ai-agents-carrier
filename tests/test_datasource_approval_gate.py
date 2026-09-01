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

    plan = await DataSourceExecutor().preview(source, "drop", {})

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
    plan = await DataSourceExecutor().preview(source, "drop", {"id": "f9"})

    assert plan.affected_rows == 1
    assert plan.targets == ["DELETE https://api.test/files/f9"]
    assert http.calls == []


async def test_preview_of_an_empty_upstream_finds_nothing_to_delete(http):
    source = _source(operations=[
        {"name": "stale", "path": "/files?stale=true"},
        {"name": "drop", "method": "DELETE", "path": "/files/{stale.id}"},
    ])
    http.handler = lambda call: []

    plan = await DataSourceExecutor().preview(source, "drop", {})
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
