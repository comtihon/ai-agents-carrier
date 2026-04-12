from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models.runtime import WorkflowRun


class WorkflowRunRepository(ABC):
    @abstractmethod
    async def create(self, workflow_run: WorkflowRun) -> WorkflowRun:
        raise NotImplementedError

    @abstractmethod
    async def update(self, workflow_run: WorkflowRun) -> WorkflowRun:
        raise NotImplementedError

    @abstractmethod
    async def get_by_id(self, run_id: str) -> WorkflowRun | None:
        raise NotImplementedError
