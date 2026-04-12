from __future__ import annotations

import json
from pathlib import Path

from app.domain.interfaces.workflow_registry import WorkflowDefinitionRegistry
from app.domain.models.workflow_definition import WorkflowDefinition


class InMemoryWorkflowDefinitionRegistry(WorkflowDefinitionRegistry):
    def __init__(self, definitions: dict[str, WorkflowDefinition]) -> None:
        self._definitions = definitions

    def list_definitions(self) -> list[WorkflowDefinition]:
        return list(self._definitions.values())

    def get_definition(self, workflow_id: str) -> WorkflowDefinition:
        try:
            return self._definitions[workflow_id]
        except KeyError as exc:
            raise KeyError(f"Workflow definition '{workflow_id}' is not registered.") from exc


class WorkflowDefinitionLoader:
    def __init__(self, definitions_path: str) -> None:
        self._definitions_path = Path(definitions_path)

    def load(self) -> InMemoryWorkflowDefinitionRegistry:
        if not self._definitions_path.exists():
            raise FileNotFoundError(f"Workflow definitions path does not exist: {self._definitions_path}")

        files = sorted(self._definitions_path.glob("*.json"))
        if not files:
            raise ValueError(f"No workflow definition files found in: {self._definitions_path}")

        definitions: dict[str, WorkflowDefinition] = {}
        for file_path in files:
            raw_data = json.loads(file_path.read_text(encoding="utf-8"))
            definition = WorkflowDefinition.model_validate(raw_data)
            definitions[definition.id] = definition

        if len(definitions) != len(files):
            raise ValueError("Workflow definition ids must be unique across files.")

        return InMemoryWorkflowDefinitionRegistry(definitions)
