from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings
from app.domain.interfaces.openhands import OpenHandsPort
from app.domain.models.runtime import OpenHandsExecutionResult, RepositoryTask, WorkflowRun


class OpenHandsAdapter(OpenHandsPort):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute_task(self, workflow_run: WorkflowRun, task: RepositoryTask) -> OpenHandsExecutionResult:
        if self._settings.openhands_mock_mode:
            return OpenHandsExecutionResult(
                branch=f"feature/{workflow_run.id[:8]}-{task.repo.replace('/', '-')}",
                summary=f"Mock OpenHands execution completed for repository '{task.repo}'.",
                pr_url=None,
                status="success",
                details={"mock": True, "step_id": task.step_id},
            )

        payload = {
            "workflow_run_id": workflow_run.id,
            "workflow_id": workflow_run.workflow_id,
            "repo": task.repo,
            "instructions": task.instructions,
            "context": workflow_run.metadata,
        }
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=self._settings.openhands_timeout_seconds) as client:
            response = await client.post(
                f"{self._settings.openhands_base_url}/sessions/execute",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        return OpenHandsExecutionResult.model_validate(data)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.openhands_api_key:
            headers["Authorization"] = f"Bearer {self._settings.openhands_api_key}"
        return headers
