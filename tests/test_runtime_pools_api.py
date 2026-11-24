"""REST routes of runtime pools, and pool agents through the agents API."""
from __future__ import annotations

import pytest

from app.application.runners import get_runner_hub
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.runner import Runner
from app.infrastructure.persistence.runner_backend import InMemoryRunnerBackend
from app.infrastructure.persistence.runtime_pool_backend import InMemoryRuntimePoolBackend
from tests.test_resource_visibility import env  # noqa: F401  (fixture)


class Session:
    closed = False


@pytest.fixture
async def pools(env):  # noqa: F811
    as_, container = env
    container.runtime_pool_backend = InMemoryRuntimePoolBackend()
    hub = get_runner_hub()
    saved = (hub.backend, dict(hub.sessions))
    hub.backend = InMemoryRunnerBackend()
    await hub.backend.save_runner(Runner(id="rnr-ann", name="ann-laptop", agents=["claude"], owner_id="alice"))
    hub.sessions["rnr-ann"] = Session()
    # Alice's private runner agent (created by the runner routes in real life).
    container.agent_backend.inner.docs["ann-claude"] = AgentDefinition(
        id="ann-claude", name="claude on ann-laptop", default_runtime="runner", runner_id="rnr-ann",
        protocol="acp", acp_agent="claude", owner_id="alice", owner_name="Alice", visibility="private",
    )
    yield as_, container
    hub.backend, hub.sessions = saved[0], saved[1]


async def test_pool_lifecycle_join_status_and_pool_agent(pools):
    as_, container = pools
    r = await as_("bob").post("/api/v1/runtime-pools", json={"id": "eng", "name": "Eng", "visibility": "public"})
    assert r.status_code == 201, r.text
    assert r.json()["owner_id"] == "bob" and r.json()["member_count"] == 0

    # Bob cannot add Alice's private agent (he cannot even see it); Alice can.
    assert (await as_("bob").post("/api/v1/runtime-pools/eng/members", json={"agent_id": "ann-claude"})).status_code == 404
    r = await as_("alice").post("/api/v1/runtime-pools/eng/members", json={"agent_id": "ann-claude"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert (body["member_count"], body["online_count"]) == (1, 1)
    assert body["members"][0]["name"] == "claude on ann-laptop"

    # Everyone sees the pool and who is online, never the member's definition.
    seen = (await as_("carol").get("/api/v1/runtime-pools")).json()
    assert [p["id"] for p in seen] == ["eng"] and seen[0]["online_count"] == 1
    assert (await as_("carol").get("/api/v1/agents/ann-claude")).status_code == 404

    # A public agent may run on the public pool.
    r = await as_("bob").post("/api/v1/agents", json={
        "id": "coder", "name": "Coder", "visibility": "public", "default_runtime": "pool", "pool_id": "eng",
    })
    assert r.status_code == 201, r.text
    assert (r.json()["protocol"], r.json()["pool_id"], r.json()["runner_id"]) == ("acp", "eng", None)

    # In use: no delete, no narrowing.
    assert (await as_("bob").delete("/api/v1/runtime-pools/eng")).status_code == 409
    r = await as_("bob").put("/api/v1/runtime-pools/eng", json={"visibility": "private"})
    assert r.status_code == 409

    # Renaming keeps the members.
    r = await as_("bob").put("/api/v1/runtime-pools/eng", json={"name": "Engineering"})
    assert r.status_code == 200 and r.json()["member_count"] == 1

    # Carol may not remove Alice's member; Alice may.
    assert (await as_("carol").delete("/api/v1/runtime-pools/eng/members/ann-claude")).status_code == 403
    r = await as_("alice").delete("/api/v1/runtime-pools/eng/members/ann-claude")
    assert r.status_code == 200 and r.json()["member_count"] == 0


async def test_pool_agent_needs_an_existing_pool(pools):
    as_, _ = pools
    r = await as_("bob").post("/api/v1/agents", json={"id": "coder", "default_runtime": "pool", "pool_id": "nope"})
    assert r.status_code == 400


async def test_deleting_an_agent_drops_it_from_pools(pools):
    as_, container = pools
    await as_("alice").post("/api/v1/runtime-pools", json={"id": "eng", "visibility": "public"})
    await as_("alice").post("/api/v1/runtime-pools/eng/members", json={"agent_id": "ann-claude"})
    assert (await as_("alice").delete("/api/v1/agents/ann-claude")).status_code == 204
    assert (await container.runtime_pool_backend.get("eng")).members == []
