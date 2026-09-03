"""Tier 2 over HTTP: the endpoints the escalation panel drives.

The rules themselves are proven against the service in
``test_sheet_compute_lifecycle``; these tests are about the wiring and the
status codes the editor branches on — a refused gate has to arrive as a 422
whose detail is readable, and a caller without ADMIN has to get a 403 rather
than a stored transform.
"""
from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.app import create_app
from app.core.config import Settings
from app.domain.models.sheet_binding import SheetBinding
from app.infrastructure.auth.authorization import (
    Permission,
    reset_current_permissions,
    set_current_permissions,
)
from app.infrastructure.datasources import sheet_compute_generate
from tests.test_datasources_api import InMemoryDataSourceBackend, _build_container
from tests.test_sheet_bindings_api import (
    GOOGLE_SA,
    GRID,
    FakeSheetsExecutor,
    _read_binding,
    _sheets_source,
    _write_binding,
)
from tests.test_sheet_compute import SUM_BY_OWNER, WRITE_STATUS


BASE = "/api/v1/datasources/google-sheets/bindings/read_open_projects"


def _read_form() -> dict:
    binding = _read_binding()
    binding["read"] = {"mode": "rows", "columns": ["project_id", "status", "owner"]}
    binding["output"] = {"key": "totals"}
    return binding


@pytest.fixture(autouse=True)
def _admin():
    token = set_current_permissions(
        {Permission.ACCESS, Permission.READ, Permission.WRITE,
         Permission.DELETE, Permission.ADMIN}
    )
    yield
    reset_current_permissions(token)


@pytest.fixture(autouse=True)
def _model(monkeypatch):
    """One canned reply, and a record of every prompt the route caused."""
    seen: list[tuple[str, str]] = []
    reply = {"code": SUM_BY_OWNER, "rationale": "counts open rows per owner"}

    async def fake_ask(settings, system, user, model, provider):
        seen.append((system, user))
        return json.dumps(reply)

    monkeypatch.setattr(sheet_compute_generate, "_ask_model", fake_ask)
    return seen, reply


@pytest.fixture
async def client(monkeypatch):
    backend = InMemoryDataSourceBackend()
    source = _sheets_source().model_copy(update={
        "bindings": [SheetBinding.model_validate(_read_form())],
    })
    await backend.create(source)
    app = create_app()
    container = _build_container(backend)
    container.settings = Settings(
        GOOGLE_IMPERSONATE_SA=GOOGLE_SA,
        SHEETS_COMPUTE_WRITES_ENABLED=True,
    )
    executor = FakeSheetsExecutor()
    container.data_source_executor = executor
    app.state.container = container
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c, backend, executor


# ─── Compile ─────────────────────────────────────────────────────────────────

async def test_compile_returns_the_code_the_rationale_and_the_output(client):
    c, backend, executor = client

    resp = await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    # The three things the review panel shows before asking for a confirmation.
    assert "def transform" in body["code"]
    assert body["rationale"] == "counts open rows per owner"
    assert body["output"] == [
        {"owner": "ann", "open_rows": 2}, {"owner": "bob", "open_rows": 1},
    ]
    # And it is not running.
    assert body["compute"]["activated"] is False


async def test_compile_returns_ambiguity_questions_as_a_form(client, monkeypatch):
    c, _backend, executor = client

    async def asking(settings, system, user, model, provider):
        return json.dumps({"needs": [
            {"question": "Which date column?", "options": ["created_at", "closed_at"]},
        ]})

    monkeypatch.setattr(sheet_compute_generate, "_ask_model", asking)

    resp = await c.post(f"{BASE}/compile", json={"instruction": "rows this quarter"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "needs"
    # Exactly what the dropdowns are built from.
    assert body["needs"] == [
        {"question": "Which date column?", "options": ["created_at", "closed_at"]},
    ]


async def test_a_failed_compile_is_a_422_carrying_the_checker_message(client, monkeypatch):
    c, _backend, executor = client

    async def bad(settings, system, user, model, provider):
        return json.dumps({"code": "def transform(records, params):\n    return eval('1')"})

    monkeypatch.setattr(sheet_compute_generate, "_ask_model", bad)

    resp = await c.post(f"{BASE}/compile", json={"instruction": "anything"})

    assert resp.status_code == 422
    assert "'eval' is not allowed" in resp.json()["detail"]["error"]


async def test_compile_without_admin_is_a_403(client):
    c, backend, executor = client
    token = set_current_permissions(
        {Permission.ACCESS, Permission.READ, Permission.WRITE}
    )
    try:
        resp = await c.post(f"{BASE}/compile", json={"instruction": "count open rows"})
    finally:
        reset_current_permissions(token)

    assert resp.status_code == 403
    assert "admin permission" in resp.json()["detail"]
    stored = (await backend.get("google-sheets")).get_binding("read_open_projects")
    assert stored.compute is None


# ─── Code viewer ─────────────────────────────────────────────────────────────

async def test_the_code_endpoint_serves_the_viewer(client):
    c, _backend, executor = client
    await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})

    resp = await c.get(f"{BASE}/code")

    assert resp.status_code == 200
    body = resp.json()
    assert "def transform" in body["code"]
    assert body["signature"] == "def transform(records: list[dict], params: dict) -> Any"
    assert body["golden"]["output_hash"]
    assert body["compute"]["edited_by_human"] is False


async def test_a_tier1_binding_has_no_code_endpoint_content(client):
    c, _backend, executor = client
    resp = await c.get(f"{BASE}/code")
    assert resp.status_code == 404
    assert "tier-1" in resp.json()["detail"]


async def test_editing_the_code_sets_edited_by_human_and_stops_regeneration(client):
    c, backend, executor = client
    await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})

    fixed = (
        "def transform(records, params):\n"
        "    return [{'owner': 'ann', 'open_rows': 1}]\n"
    )
    edited = await c.put(f"{BASE}/code", json={"code": fixed})
    assert edited.status_code == 200, edited.text
    assert edited.json()["compute"]["edited_by_human"] is True

    # A later recompile refuses rather than overwriting the fix.
    again = await c.post(f"{BASE}/compile", json={"instruction": "count differently"})
    assert again.status_code == 422
    assert "edited by hand" in again.json()["detail"]
    stored = (await backend.get("google-sheets")).get_binding("read_open_projects")
    assert stored.compute.code == fixed


async def test_a_hand_edit_that_fails_a_gate_is_a_422(client):
    c, _backend, executor = client
    await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})

    resp = await c.put(f"{BASE}/code", json={
        "code": "def transform(records, params):\n    return records.__class__",
    })

    assert resp.status_code == 422
    assert "__class__" in resp.json()["detail"]


# ─── Activate / re-test / stale ──────────────────────────────────────────────

async def test_activation_is_required_before_the_binding_runs(client):
    c, backend, executor = client
    await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})

    # Compiled but off: calling the compiled operation refuses.
    preview = await c.post(f"{BASE}/preview", json={"params": {"assignee": "ann"}})
    assert preview.status_code == 200
    assert "has not been activated" in preview.json()["error"]

    activated = await c.post(f"{BASE}/activate")
    assert activated.status_code == 200
    assert activated.json()["compute"]["activated"] is True

    # And now it runs.
    preview = await c.post(f"{BASE}/preview", json={"params": {"assignee": "ann"}})
    assert preview.json()["status"] == "ok"
    assert preview.json()["output"] == [
        {"owner": "ann", "open_rows": 1}, {"owner": "bob", "open_rows": 1},
    ]


async def test_retest_reports_the_verified_timestamp_the_badge_shows(client):
    c, _backend, executor = client
    await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})
    await c.post(f"{BASE}/activate")

    resp = await c.post(f"{BASE}/retest")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["compute"]["golden"]["verified_days_ago"] == pytest.approx(0, abs=0.01)


async def test_retest_marks_a_drifted_binding_stale(client):
    """A renamed column switches the binding off, loudly and with no fallback."""
    c, backend, executor = client
    await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})
    await c.post(f"{BASE}/activate")
    assert (await c.post(f"{BASE}/retest")).json()["status"] == "ok"

    # Somebody renames a column in the spreadsheet.
    from tests.test_sheet_bindings_api import HEADERS
    executor.grid = [["project_id", "state", *HEADERS[2:]], *GRID[1:]]

    resp = await c.post(f"{BASE}/retest")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "stale"
    assert "header row has changed" in body["error"]
    stored = (await backend.get("google-sheets")).get_binding("read_open_projects")
    assert stored.compute.stale is True
    assert stored.compute.activated is False


async def test_marking_stale_needs_no_admin_and_switches_it_off(client):
    c, backend, executor = client
    await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})
    await c.post(f"{BASE}/activate")

    token = set_current_permissions({Permission.ACCESS, Permission.READ, Permission.WRITE})
    try:
        resp = await c.post(f"{BASE}/stale", json={"reason": "numbers look wrong"})
    finally:
        reset_current_permissions(token)

    assert resp.status_code == 200
    assert resp.json()["compute"]["stale"] is True
    stored = (await backend.get("google-sheets")).get_binding("read_open_projects")
    assert stored.compute.activated is False


# ─── The bindings list ───────────────────────────────────────────────────────

async def test_the_bindings_list_carries_the_tier_for_the_badge(client):
    c, _backend, executor = client
    listed = await c.get("/api/v1/datasources/google-sheets/bindings")
    assert listed.json()[0]["compute_status"]["tier"] == "binding"

    await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})

    listed = await c.get("/api/v1/datasources/google-sheets/bindings")
    status = listed.json()[0]["compute_status"]
    assert status["tier"] == "script"
    assert status["generated"] is True
    assert status["activated"] is False
    assert status["golden"]["verified_days_ago"] == pytest.approx(0, abs=0.01)


async def test_editing_a_tier2_binding_through_the_tier1_form_keeps_its_code(client):
    """Changing the output key must not throw away the transform.

    And the reverse: the tier-1 save path cannot invent a transform either --
    see test_a_tier1_save_cannot_fabricate_llm_provenance.
    """
    c, backend, executor = client
    await c.post(f"{BASE}/compile", json={"instruction": "count open rows per owner"})
    await c.post(f"{BASE}/activate")

    form = _read_form()
    form["output"] = {"key": "renamed"}
    resp = await c.put(
        "/api/v1/datasources/google-sheets/bindings/read_open_projects",
        json={"binding": form},
    )

    assert resp.status_code == 200, resp.text
    stored = (await backend.get("google-sheets")).get_binding("read_open_projects")
    assert stored.output.key == "renamed"
    assert stored.compute is not None
    assert stored.resolution.tier == "script"
    assert stored.resolution.instruction == "count open rows per owner"
