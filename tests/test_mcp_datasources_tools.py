"""Tests for the /mcp/datasources FastMCP tool surface."""
from __future__ import annotations

import pytest

from app.api.mcp.datasources_server import (
    build_datasources_mcp,
    rebuild_datasource_tools,
    tool_name_for,
)
from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.datasources.executor import DataSourceExecutor
from tests.test_datasources_api import InMemoryDataSourceBackend


def _definition(source_id: str = "github") -> DataSourceDefinition:
    return DataSourceDefinition.model_validate({
        "id": source_id,
        "name": "GitHub",
        "description": "Code host",
        "base_url": "https://api.github.com",
        "operations": [
            {
                "name": "list_repos",
                "path": "/users/{params.owner}/repos",
                "params": [
                    {"name": "owner", "description": "GitHub login"},
                    {"name": "limit", "type": "number", "required": False},
                ],
            },
            # Dependent operation — its upstream must stay out of the schema.
            {"name": "languages", "path": "/repos/{list_repos.name}/languages"},
        ],
    })


class _Container:
    def __init__(self, backend) -> None:
        self.data_source_backend = backend
        self.data_source_executor = DataSourceExecutor()


@pytest.fixture
def mcp():
    return build_datasources_mcp()


async def test_tool_naming_and_params_only_schema(mcp):
    backend = InMemoryDataSourceBackend()
    await backend.create(_definition())
    container = _Container(backend)

    await rebuild_datasource_tools(mcp, backend, lambda: container)
    tools = {t.name: t for t in await mcp.list_tools()}

    assert set(tools) == {"ds_github_list_repos", "ds_github_languages"}
    assert tool_name_for("github", "list_repos") == "ds_github_list_repos"

    schema = tools["ds_github_list_repos"].inputSchema
    assert set(schema["properties"]) == {"owner", "limit"}
    assert schema["required"] == ["owner"]

    # Dependencies are invisible: the dependent tool takes no inputs.
    assert tools["ds_github_languages"].inputSchema.get("properties", {}) == {}


async def test_tool_names_sanitise_dots_and_slashes(mcp):
    backend = InMemoryDataSourceBackend()
    await backend.create(_definition(source_id="my.data source"))
    await rebuild_datasource_tools(mcp, backend, lambda: _Container(backend))
    names = {t.name for t in await mcp.list_tools()}
    assert "ds_my_data_source_list_repos" in names


async def test_rebuild_reflects_crud_changes(mcp):
    backend = InMemoryDataSourceBackend()
    container = _Container(backend)

    await rebuild_datasource_tools(mcp, backend, lambda: container)
    assert await mcp.list_tools() == []

    await backend.create(_definition())
    await rebuild_datasource_tools(mcp, backend, lambda: container)
    assert len(await mcp.list_tools()) == 2

    await backend.delete("github")
    await rebuild_datasource_tools(mcp, backend, lambda: container)
    assert await mcp.list_tools() == []


async def test_tool_call_invokes_executor_at_call_time(mcp, monkeypatch):
    backend = InMemoryDataSourceBackend()
    await backend.create(_definition())
    container = _Container(backend)
    await rebuild_datasource_tools(mcp, backend, lambda: container)

    seen: dict = {}

    async def _fake_execute(source, operation, params):
        seen.update({"source": source.id, "operation": operation, "params": params})
        return {"ok": True}

    monkeypatch.setattr(container.data_source_executor, "execute", _fake_execute)
    await mcp.call_tool("ds_github_list_repos", {"owner": "acme"})

    assert seen == {
        "source": "github",
        "operation": "list_repos",
        "params": {"owner": "acme", "limit": None},
    }


async def test_rebuild_skips_colliding_tool_names_instead_of_overwriting(mcp, caplog):
    """Two sources that sanitize to the same tool name must not silently
    overwrite each other — the second is skipped and logged clearly."""
    backend = InMemoryDataSourceBackend()
    await backend.create(_definition(source_id="my.data"))
    await backend.create(_definition(source_id="my/data"))  # sanitizes to same id
    container = _Container(backend)

    with caplog.at_level("ERROR"):
        await rebuild_datasource_tools(mcp, backend, lambda: container)

    tools = {t.name for t in await mcp.list_tools()}
    # Only the first source's tools survive; the collision is logged, not silent.
    assert tools == {"ds_my_data_list_repos", "ds_my_data_languages"}
    assert any("collision" in rec.message for rec in caplog.records)


async def test_tool_call_reports_missing_source(mcp):
    backend = InMemoryDataSourceBackend()
    await backend.create(_definition())
    container = _Container(backend)
    await rebuild_datasource_tools(mcp, backend, lambda: container)

    await backend.delete("github")
    result = await mcp.call_tool("ds_github_list_repos", {"owner": "acme"})
    assert "not found" in str(result)
