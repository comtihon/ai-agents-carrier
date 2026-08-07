"""`http_call` steps with ``auth: service_identity``.

httpx is stubbed in the yaml_graph module so no network access is needed; the
recorder captures the headers every outbound request carried.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.infrastructure.orchestration import yaml_graph as yaml_graph_module
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.tools.mcp_client import McpToolsProvider


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "ok") -> None:
        self.status_code = status_code
        self.text = text


class FakeClient:
    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def request(self, method, url, headers=None, json=None):
        self._calls.append(
            {"method": method, "url": url, "headers": dict(headers or {}), "json": json}
        )
        return FakeResponse()


@pytest.fixture
def http(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        yaml_graph_module.httpx, "AsyncClient", lambda *a, **kw: FakeClient(calls)
    )
    return calls


class _FakeTokenProvider:
    def __init__(self, token: str = "svc-token") -> None:
        self.token = token
        self.calls = 0
        self.identities: list[str | None] = []

    async def get_auth_header(self, identity: str | None = None) -> dict[str, str]:
        self.calls += 1
        self.identities.append(identity)
        return {"Authorization": f"Bearer {self.token}"}


def _runner() -> YamlGraphRunner:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    mcp.get_tools = MagicMock(return_value=[])
    return YamlGraphRunner(
        {"id": "g", "steps": [{"id": "call", "type": "http_call", "url": "https://svc.test/x"}]},
        llm=llm,
        mcp_tools_provider=mcp,
    )


async def test_service_identity_injects_bearer_header(http):
    runner = _runner()
    provider = _FakeTokenProvider()
    runner._service_token_provider = provider
    node = runner._http_call_node(
        {"id": "call", "url": "https://svc.test/x", "auth": "service_identity"}
    )

    result = await node({})

    assert result["call"]["status"] == 200
    assert http[0]["headers"]["Authorization"] == "Bearer svc-token"
    assert provider.calls == 1


async def test_no_auth_sends_no_authorization_header(http):
    runner = _runner()
    runner._service_token_provider = _FakeTokenProvider()
    node = runner._http_call_node({"id": "call", "url": "https://svc.test/x"})

    await node({})

    assert "Authorization" not in http[0]["headers"]


async def test_explicit_header_wins_over_service_identity(http):
    runner = _runner()
    runner._service_token_provider = _FakeTokenProvider()
    node = runner._http_call_node(
        {
            "id": "call",
            "url": "https://svc.test/x",
            "auth": "service_identity",
            "headers": {"Authorization": "Bearer explicit"},
        }
    )

    await node({})

    assert http[0]["headers"]["Authorization"] == "Bearer explicit"


async def test_unsupported_auth_mode_is_reported(http):
    runner = _runner()
    node = runner._http_call_node(
        {"id": "call", "url": "https://svc.test/x", "auth": "magic"}
    )

    with pytest.raises(ValueError, match="Unsupported auth mode"):
        await node({})
    assert http == []


async def test_token_failure_is_captured_as_step_error(http):
    class FailingProvider:
        async def get_auth_header(self, identity: str | None = None):
            raise RuntimeError("service auth misconfigured")

    runner = _runner()
    runner._service_token_provider = FailingProvider()
    node = runner._http_call_node(
        {"id": "call", "url": "https://svc.test/x", "auth": "service_identity"}
    )

    result = await node({})

    assert "service auth misconfigured" in result["call"]["error"]
    assert http == []


async def test_auth_identity_selects_the_named_identity(http):
    provider = _FakeTokenProvider()
    runner = _runner()
    runner._service_token_provider = provider
    node = runner._http_call_node({
        "id": "call",
        "url": "https://svc.test/x",
        "auth": "service_identity",
        "auth_identity": "afp",
    })

    await node({})

    assert provider.identities == ["afp"]


async def test_blank_auth_identity_falls_back_to_the_default(http):
    provider = _FakeTokenProvider()
    runner = _runner()
    runner._service_token_provider = provider
    node = runner._http_call_node({
        "id": "call",
        "url": "https://svc.test/x",
        "auth": "service_identity",
        "auth_identity": "   ",
    })

    await node({})

    assert provider.identities == [None]
