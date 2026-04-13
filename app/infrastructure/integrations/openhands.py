from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings


class OpenHandsAdapter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute(self, repo: str, instructions: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._settings.openhands_mock_mode:
            return {
                "status": "success",
                "branch": f"feature/openhands-{repo.replace('/', '-')[:20]}",
                "summary": f"Mock execution completed for '{repo}'.",
                "mock": True,
            }

        payload: dict[str, Any] = {"repo": repo, "instructions": instructions}
        if context:
            payload["context"] = context

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.openhands_api_key:
            headers["Authorization"] = f"Bearer {self._settings.openhands_api_key}"

        async with httpx.AsyncClient(timeout=self._settings.openhands_timeout_seconds) as client:
            response = await client.post(
                f"{self._settings.openhands_base_url}/sessions/execute",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
