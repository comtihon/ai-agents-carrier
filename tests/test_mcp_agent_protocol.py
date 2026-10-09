"""Management tools set an agent's protocol (http-poll / acp) and ACP agent."""

import pytest

from app.application import management_tools as core
from app.domain.models.agent_definition import AgentDefinition


class _Backend:
    def __init__(self):
        self.items = {}

    async def get(self, agent_id):
        return self.items.get(agent_id)

    async def list(self):
        return list(self.items.values())

    async def create(self, a):
        self.items[a.id] = a

    async def update(self, agent_id, a):
        self.items[agent_id] = a


class _Deps:
    def __init__(self):
        self.agent_backend = _Backend()


@pytest.mark.asyncio
async def test_create_and_switch_protocol():
    deps = _Deps()
    assert "created" in await core.create_agent(deps, "pi", "Pi", default_runtime="k8s", protocol="acp", acp_agent="pi")
    a = deps.agent_backend.items["pi"]
    assert a.protocol == "acp" and a.acp_agent == "pi"

    assert "updated" in await core.update_agent(deps, "pi", protocol="http-poll")
    a = deps.agent_backend.items["pi"]
    assert a.protocol == "http-poll" and a.acp_agent is None

    await core.update_agent(deps, "pi", protocol="acp", acp_agent="claude")
    assert deps.agent_backend.items["pi"].acp_agent == "claude"
    await core.update_agent(deps, "pi", acp_agent="")
    assert deps.agent_backend.items["pi"].acp_agent is None
    # Untouched fields stay.
    assert deps.agent_backend.items["pi"].default_runtime == "k8s"


@pytest.mark.asyncio
async def test_rejects_unknown_protocol():
    deps = _Deps()
    deps.agent_backend.items["x"] = AgentDefinition(id="x")
    assert "Invalid protocol" in await core.update_agent(deps, "x", protocol="grpc")
    assert "Invalid protocol" in await core.create_agent(deps, "y", "Y", protocol="grpc")
