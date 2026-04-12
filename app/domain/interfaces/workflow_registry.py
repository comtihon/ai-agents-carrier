from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models.workflow_definition import WorkflowDefinition


class WorkflowDefinitionRegistry(ABC):
    @abstractmethod
    def list_definitions(self) -> list[WorkflowDefinition]:
        raise NotImplementedError

    @abstractmethod
    def get_definition(self, workflow_id: str) -> WorkflowDefinition:
        raise NotImplementedError
