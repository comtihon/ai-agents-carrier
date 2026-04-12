from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models.runtime import OpenHandsExecutionResult, RepositoryTask, WorkflowRun


class OpenHandsPort(ABC):
    @abstractmethod
    async def execute_task(self, workflow_run: WorkflowRun, task: RepositoryTask) -> OpenHandsExecutionResult:
        raise NotImplementedError
