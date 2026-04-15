"""
Unit tests for WorkflowDefinitionBackend implementations.

LocalFilesWorkflowBackend is tested against a real temp directory.
MongoWorkflowBackend tests require MongoDB and are skipped automatically
when it is not available (same pattern as the integration tests).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from app.domain.models.workflow_definition import WorkflowDefinition
from app.infrastructure.persistence.workflow_backend import (
    LocalFilesWorkflowBackend,
    MongoWorkflowBackend,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_DEF = WorkflowDefinition(
    id="my-workflow",
    name="My Workflow",
    description="A test workflow",
    steps=[
        {
            "id": "step1",
            "type": "llm",
            "output_key": "answer",
            "user_template": "{request}",
        }
    ],
)


# ---------------------------------------------------------------------------
# LocalFilesWorkflowBackend
# ---------------------------------------------------------------------------

class TestLocalFilesWorkflowBackend:
    @pytest.mark.asyncio
    async def test_list_empty_when_directory_missing(self, tmp_path) -> None:
        backend = LocalFilesWorkflowBackend(str(tmp_path / "nonexistent"))
        result = await backend.list()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_returns_definitions_from_yaml_files(self, tmp_path) -> None:
        (tmp_path / "wf-a.yaml").write_text(textwrap.dedent("""\
            id: wf-a
            name: Workflow A
            steps:
              - id: s1
                type: llm
                output_key: out
                user_template: "{request}"
        """))
        (tmp_path / "wf-b.yaml").write_text(textwrap.dedent("""\
            id: wf-b
            steps:
              - id: s1
                type: llm
                output_key: out
                user_template: "{request}"
        """))
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        result = await backend.list()
        assert len(result) == 2
        ids = {d.id for d in result}
        assert ids == {"wf-a", "wf-b"}

    @pytest.mark.asyncio
    async def test_list_skips_malformed_yaml(self, tmp_path) -> None:
        (tmp_path / "broken.yaml").write_text("id: broken\nsteps: [{")
        (tmp_path / "good.yaml").write_text(textwrap.dedent("""\
            id: good
            steps:
              - id: s1
                type: llm
                output_key: out
                user_template: "{request}"
        """))
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        result = await backend.list()
        assert len(result) == 1
        assert result[0].id == "good"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, tmp_path) -> None:
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        result = await backend.get("no-such-workflow")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_definition(self, tmp_path) -> None:
        (tmp_path / "my-workflow.yaml").write_text(textwrap.dedent("""\
            id: my-workflow
            name: My Workflow
            description: A test
            steps:
              - id: s1
                type: llm
                output_key: answer
                user_template: "{request}"
        """))
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        result = await backend.get("my-workflow")
        assert result is not None
        assert result.id == "my-workflow"
        assert result.name == "My Workflow"
        assert result.description == "A test"
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_create_writes_yaml_file(self, tmp_path) -> None:
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        saved = await backend.create(_SIMPLE_DEF.model_copy(deep=True))
        assert saved.id == "my-workflow"

        yaml_file = tmp_path / "my-workflow.yaml"
        assert yaml_file.exists()
        raw = yaml.safe_load(yaml_file.read_text())
        assert raw["id"] == "my-workflow"
        assert raw["name"] == "My Workflow"
        assert raw["steps"][0]["id"] == "step1"

    @pytest.mark.asyncio
    async def test_create_creates_directory_if_missing(self, tmp_path) -> None:
        subdir = tmp_path / "graphs" / "nested"
        backend = LocalFilesWorkflowBackend(str(subdir))
        await backend.create(_SIMPLE_DEF.model_copy(deep=True))
        assert (subdir / "my-workflow.yaml").exists()

    @pytest.mark.asyncio
    async def test_update_overwrites_existing_file(self, tmp_path) -> None:
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        await backend.create(_SIMPLE_DEF.model_copy(deep=True))

        updated = _SIMPLE_DEF.model_copy(deep=True)
        updated.name = "Updated Name"
        updated.steps = []
        saved = await backend.update("my-workflow", updated)
        assert saved.name == "Updated Name"

        raw = yaml.safe_load((tmp_path / "my-workflow.yaml").read_text())
        assert raw["name"] == "Updated Name"
        assert raw["steps"] == []

    @pytest.mark.asyncio
    async def test_delete_removes_file(self, tmp_path) -> None:
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        await backend.create(_SIMPLE_DEF.model_copy(deep=True))
        assert (tmp_path / "my-workflow.yaml").exists()

        await backend.delete("my-workflow")
        assert not (tmp_path / "my-workflow.yaml").exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_silent(self, tmp_path) -> None:
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        # Should not raise
        await backend.delete("does-not-exist")

    @pytest.mark.asyncio
    async def test_roundtrip_full_definition(self, tmp_path) -> None:
        """Create → get returns the same definition."""
        backend = LocalFilesWorkflowBackend(str(tmp_path))
        defn = WorkflowDefinition(
            id="rt-test",
            name="RT Test",
            description="Round-trip test",
            steps=[
                {"id": "s1", "type": "llm", "output_key": "out", "user_template": "{request}"},
                {"id": "s2", "type": "human_approval"},
            ],
        )
        await backend.create(defn)
        result = await backend.get("rt-test")
        assert result is not None
        assert result.id == defn.id
        assert result.name == defn.name
        assert result.description == defn.description
        assert len(result.steps) == 2
        assert result.steps[0]["id"] == "s1"
        assert result.steps[1]["id"] == "s2"


# ---------------------------------------------------------------------------
# MongoWorkflowBackend (skipped when MongoDB unavailable)
# ---------------------------------------------------------------------------

_MONGO_URI = "mongodb://localhost:27017"
_TEST_DB = "test_workflow_backend"


def _check_mongo() -> bool:
    try:
        from pymongo import MongoClient
        from pymongo.errors import ServerSelectionTimeoutError
        client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


_mongo_available = _check_mongo()
_skip_mongo = pytest.mark.skipif(not _mongo_available, reason="MongoDB not reachable")


@pytest.fixture()
async def mongo_backend():
    backend = MongoWorkflowBackend(_MONGO_URI, _TEST_DB)
    # Clean slate before each test
    await backend._col.delete_many({})
    yield backend
    await backend._col.delete_many({})
    await backend.close()


class TestMongoWorkflowBackend:
    @_skip_mongo
    @pytest.mark.asyncio
    async def test_list_empty_initially(self, mongo_backend: MongoWorkflowBackend) -> None:
        result = await mongo_backend.list()
        assert result == []

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_create_and_list(self, mongo_backend: MongoWorkflowBackend) -> None:
        await mongo_backend.create(_SIMPLE_DEF.model_copy(deep=True))
        result = await mongo_backend.list()
        assert len(result) == 1
        assert result[0].id == "my-workflow"
        assert result[0].name == "My Workflow"

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self, mongo_backend: MongoWorkflowBackend) -> None:
        result = await mongo_backend.get("no-such")
        assert result is None

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_create_and_get(self, mongo_backend: MongoWorkflowBackend) -> None:
        await mongo_backend.create(_SIMPLE_DEF.model_copy(deep=True))
        result = await mongo_backend.get("my-workflow")
        assert result is not None
        assert result.id == "my-workflow"
        assert result.steps[0]["id"] == "step1"

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_update_replaces_definition(self, mongo_backend: MongoWorkflowBackend) -> None:
        await mongo_backend.create(_SIMPLE_DEF.model_copy(deep=True))
        updated = _SIMPLE_DEF.model_copy(deep=True)
        updated.name = "New Name"
        updated.steps = []
        await mongo_backend.update("my-workflow", updated)
        result = await mongo_backend.get("my-workflow")
        assert result is not None
        assert result.name == "New Name"
        assert result.steps == []

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_delete_removes_definition(self, mongo_backend: MongoWorkflowBackend) -> None:
        await mongo_backend.create(_SIMPLE_DEF.model_copy(deep=True))
        await mongo_backend.delete("my-workflow")
        result = await mongo_backend.get("my-workflow")
        assert result is None

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_silent(self, mongo_backend: MongoWorkflowBackend) -> None:
        await mongo_backend.delete("ghost")  # should not raise

    @_skip_mongo
    @pytest.mark.asyncio
    async def test_create_is_idempotent_upsert(self, mongo_backend: MongoWorkflowBackend) -> None:
        """Creating twice should upsert (last write wins), not raise."""
        await mongo_backend.create(_SIMPLE_DEF.model_copy(deep=True))
        second = _SIMPLE_DEF.model_copy(deep=True)
        second.name = "Second Version"
        await mongo_backend.create(second)
        result = await mongo_backend.get("my-workflow")
        assert result is not None
        assert result.name == "Second Version"
