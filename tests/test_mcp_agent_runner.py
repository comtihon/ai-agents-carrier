"""Management MCP tools point an agent at a connected machine (runner_id)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application import management_tools as core
from app.domain.models.runner import Runner
from app.infrastructure.persistence.runner_backend import InMemoryRunnerBackend
from tests.test_agents_api import InMemoryAgentBackend

pytestmark = pytest.mark.asyncio


async def _deps():
    runners = InMemoryRunnerBackend()
    await runners.save_runner(Runner(id="rnr-1", name="box", agents=["codex", "claude"]))
    agents = InMemoryAgentBackend()
    return core.ManagementDeps(registry=MagicMock(), run_repository=AsyncMock(),
                               agent_backend=agents, runner_backend=runners), agents


async def test_create_runner_agent():
    deps, agents = await _deps()
    out = await core.create_agent(deps, "box-codex", "Box", default_runtime="runner",
                                  acp_agent="codex", runner_id="rnr-1", max_concurrency=1)
    assert "created" in out
    a = await agents.get("box-codex")
    assert (a.default_runtime, a.runner_id, a.protocol, a.acp_agent, a.max_concurrency) == (
        "runner", "rnr-1", "acp", "codex", 1)


async def test_create_runner_agent_needs_a_known_runner_and_agent():
    deps, agents = await _deps()
    assert "needs runner_id" in await core.create_agent(deps, "a", "A", default_runtime="runner")
    assert "not found" in await core.create_agent(deps, "a", "A", default_runtime="runner", runner_id="rnr-x")
    assert "does not offer" in await core.create_agent(
        deps, "a", "A", default_runtime="runner", runner_id="rnr-1", acp_agent="pi")
    assert await agents.get("a") is None


async def test_create_non_runner_ignores_runner_id():
    deps, agents = await _deps()
    assert "created" in await core.create_agent(deps, "p", "P", runner_id="rnr-1")
    assert (await agents.get("p")).runner_id is None


async def test_update_moves_agent_onto_runner_and_back():
    deps, agents = await _deps()
    await core.create_agent(deps, "a", "A", protocol="acp", acp_agent="codex")
    assert "updated" in await core.update_agent(deps, "a", runner_id="rnr-1", default_runtime="runner")
    a = await agents.get("a")
    assert (a.default_runtime, a.runner_id, a.protocol) == ("runner", "rnr-1", "acp")
    assert "not found" in await core.update_agent(deps, "a", runner_id="rnr-x")
    assert "needs runner_id" in await core.update_agent(deps, "a", runner_id="")
    assert "updated" in await core.update_agent(deps, "a", runner_id="", default_runtime="local")
    a = await agents.get("a")
    assert (a.default_runtime, a.runner_id) == ("local", None)
