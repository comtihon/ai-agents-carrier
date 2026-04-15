from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.domain.models.workflow_definition import WorkflowDefinition

logger = logging.getLogger(__name__)


class WorkflowDefinitionBackend(ABC):
    """Persistent storage for workflow definitions.

    Two implementations are provided:
    - LocalFilesWorkflowBackend  — reads/writes YAML files on disk.
    - MongoWorkflowBackend       — reads/writes a MongoDB collection.
    """

    @abstractmethod
    async def list(self) -> list[WorkflowDefinition]: ...

    @abstractmethod
    async def get(self, workflow_id: str) -> WorkflowDefinition | None: ...

    @abstractmethod
    async def create(self, definition: WorkflowDefinition) -> WorkflowDefinition: ...

    @abstractmethod
    async def update(self, workflow_id: str, definition: WorkflowDefinition) -> WorkflowDefinition: ...

    @abstractmethod
    async def delete(self, workflow_id: str) -> None: ...


# ---------------------------------------------------------------------------
# Local-files implementation
# ---------------------------------------------------------------------------

class LocalFilesWorkflowBackend(WorkflowDefinitionBackend):
    """Reads and writes workflow definitions as YAML files in a directory."""

    def __init__(self, directory: str) -> None:
        self._path = Path(directory)

    def _file_path(self, workflow_id: str) -> Path:
        return self._path / f"{workflow_id}.yaml"

    def _load_file(self, path: Path) -> WorkflowDefinition | None:
        try:
            raw: dict[str, Any] = yaml.safe_load(path.read_text())
            stat = path.stat()
            return WorkflowDefinition(
                id=raw["id"],
                name=raw.get("name", ""),
                description=raw.get("description", ""),
                steps=raw.get("steps", []),
                created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        except Exception:
            logger.exception("Failed to load workflow definition from %s", path)
            return None

    async def list(self) -> list[WorkflowDefinition]:
        if not self._path.exists():
            return []
        result: list[WorkflowDefinition] = []
        for f in sorted(self._path.glob("*.yaml")):
            defn = self._load_file(f)
            if defn is not None:
                result.append(defn)
        return result

    async def get(self, workflow_id: str) -> WorkflowDefinition | None:
        path = self._file_path(workflow_id)
        if not path.exists():
            return None
        return self._load_file(path)

    async def create(self, definition: WorkflowDefinition) -> WorkflowDefinition:
        self._path.mkdir(parents=True, exist_ok=True)
        definition.touch()
        self._write(definition)
        return definition

    async def update(self, workflow_id: str, definition: WorkflowDefinition) -> WorkflowDefinition:
        definition.id = workflow_id
        definition.touch()
        self._write(definition)
        return definition

    async def delete(self, workflow_id: str) -> None:
        path = self._file_path(workflow_id)
        if path.exists():
            path.unlink()

    def _write(self, definition: WorkflowDefinition) -> None:
        data: dict[str, Any] = {
            "id": definition.id,
            "name": definition.name,
            "description": definition.description,
            "steps": definition.steps,
        }
        self._file_path(definition.id).write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )


# ---------------------------------------------------------------------------
# MongoDB implementation
# ---------------------------------------------------------------------------

class MongoWorkflowBackend(WorkflowDefinitionBackend):
    """Reads and writes workflow definitions in a MongoDB collection."""

    _COLLECTION = "workflow_definitions"

    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(uri)
        self._col = self._client[database][self._COLLECTION]

    async def list(self) -> list[WorkflowDefinition]:
        docs = await self._col.find({}).to_list(None)
        return [self._from_doc(d) for d in docs]

    async def get(self, workflow_id: str) -> WorkflowDefinition | None:
        doc = await self._col.find_one({"_id": workflow_id})
        return self._from_doc(doc) if doc else None

    async def create(self, definition: WorkflowDefinition) -> WorkflowDefinition:
        definition.touch()
        await self._col.replace_one(
            {"_id": definition.id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def update(self, workflow_id: str, definition: WorkflowDefinition) -> WorkflowDefinition:
        definition.id = workflow_id
        definition.touch()
        await self._col.replace_one(
            {"_id": workflow_id},
            self._to_doc(definition),
            upsert=True,
        )
        return definition

    async def delete(self, workflow_id: str) -> None:
        await self._col.delete_one({"_id": workflow_id})

    async def close(self) -> None:
        self._client.close()

    @staticmethod
    def _to_doc(defn: WorkflowDefinition) -> dict[str, Any]:
        data = defn.model_dump(mode="python")
        data["_id"] = data.pop("id")
        return data

    @staticmethod
    def _from_doc(doc: dict[str, Any]) -> WorkflowDefinition:
        data = dict(doc)
        data["id"] = data.pop("_id")
        return WorkflowDefinition.model_validate(data)
