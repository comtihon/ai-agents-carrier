from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)

_START_TERMINAL = {"READY", "ERROR"}
_EXEC_TERMINAL = {"finished", "error", "stuck"}


def _build_repo_setup_preface(repo: str | None, branch: str | None) -> str:
    """Tell the agent to clone the repo(s) itself instead of relying on
    OpenHands' selected_repository/selected_branch fields.

    Why: when OpenHands V1's start flow performs the clone+checkout it goes
    through the sandbox bash API with the X-Session-API-Key handshake, which
    has been returning 401 for non-default branches. Cloning from inside the
    conversation works because it uses the agent's own GitHub access, so we
    embed the repo+branch as prompt context and start with an empty workspace.
    """
    repos = [r.strip() for r in (repo or "").split(",") if r.strip()]
    if not repos:
        return ""
    if len(repos) == 1:
        line = f"`{repos[0]}`"
        if branch:
            line += f" on branch `{branch}`"
        return (
            f"Before doing anything else, clone repository {line} into a "
            f"sub-directory of the current working directory and `cd` into it."
        )
    bullets = "\n".join(f"- `{r}`" for r in repos)
    branch_note = f"\nUse branch `{branch}` for each repository.\n" if branch else ""
    return (
        "Before doing anything else, clone the following repositories into "
        "sub-directories of the current working directory:\n\n"
        f"{bullets}{branch_note}"
    )


class OpenHandsAdapter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute(
        self,
        repo: str,
        instructions: str,
        context: dict[str, Any] | None = None,
        existing_conv_id: str | None = None,
        conv_id_callback: Callable[[str], Awaitable[None]] | None = None,
        branch: str | None = None,
    ) -> dict[str, Any]:
        if self._settings.openhands_mock_mode:
            return {
                "status": "success",
                "branch": branch or f"feature/openhands-{repo.replace('/', '-')[:20]}",
                "summary": f"Mock execution completed for '{repo}'.",
                "mock": True,
            }

        preface = _build_repo_setup_preface(repo, branch)
        if preface:
            instructions = f"{preface}\n\n{instructions}"
        if context:
            instructions = f"{instructions}\n\nContext:\n{context}"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.openhands_api_key:
            headers["Authorization"] = f"Bearer {self._settings.openhands_api_key}"

        base_url = self._settings.openhands_base_url
        poll_interval = self._settings.openhands_poll_interval_seconds
        deadline = time.monotonic() + self._settings.openhands_task_timeout_seconds

        async with httpx.AsyncClient(timeout=self._settings.openhands_timeout_seconds, headers=headers) as client:
            if existing_conv_id:
                resp = await client.get(f"{base_url}/api/v1/app-conversations", params={"ids": existing_conv_id})
                found = resp.is_success and (resp.json() or [None])[0] is not None
                if found:
                    logger.info("Resuming OpenHands conversation %s", existing_conv_id)
                    conv_id = existing_conv_id
                else:
                    logger.warning("OpenHands conversation %s not found, starting a new one", existing_conv_id)
                    existing_conv_id = None
            if not existing_conv_id:
                # Step 1: start the conversation
                resp = await client.post(
                    f"{base_url}/api/v1/app-conversations",
                    json={
                        "trigger": "openhands_api",
                        "initial_message": {
                            "role": "user",
                            "content": [{"type": "text", "text": instructions}],
                            "run": True,
                        },
                    },
                )
                resp.raise_for_status()
                start_task = resp.json()
                task_id = start_task["id"]

                # Step 2: poll until sandbox is ready
                while start_task.get("status") not in _START_TERMINAL:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"OpenHands start task {task_id} did not reach READY within {self._settings.openhands_task_timeout_seconds}s")
                    await asyncio.sleep(poll_interval)
                    resp = await client.get(
                        f"{base_url}/api/v1/app-conversations/start-tasks",
                        params={"ids": task_id},
                    )
                    resp.raise_for_status()
                    tasks = resp.json()
                    if not tasks or tasks[0] is None:
                        raise RuntimeError(f"OpenHands start task {task_id} disappeared")
                    start_task = tasks[0]

                if start_task.get("status") == "ERROR":
                    raise RuntimeError(f"OpenHands failed to start conversation: {start_task.get('detail')}")

                conv_id = start_task["app_conversation_id"]
                if conv_id_callback:
                    await conv_id_callback(conv_id)

            # Step 3: poll until agent finishes
            conv: dict[str, Any] = {}
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"OpenHands conversation {conv_id} did not finish within {self._settings.openhands_task_timeout_seconds}s")
                await asyncio.sleep(poll_interval)
                resp = await client.get(
                    f"{base_url}/api/v1/app-conversations",
                    params={"ids": conv_id},
                )
                resp.raise_for_status()
                convs = resp.json()
                if not convs or convs[0] is None:
                    raise RuntimeError(f"OpenHands conversation {conv_id} not found")
                conv = convs[0]
                if conv.get("execution_status") in _EXEC_TERMINAL:
                    break

            return {
                "conversation_id": conv_id,
                "execution_status": conv.get("execution_status"),
                "conversation_url": conv.get("conversation_url"),
                "selected_repository": repo,
                "selected_branch": branch,
            }

    async def close_conversation(self, conv_id: str) -> None:
        if self._settings.openhands_mock_mode:
            return
        headers: dict[str, str] = {}
        if self._settings.openhands_api_key:
            headers["Authorization"] = f"Bearer {self._settings.openhands_api_key}"
        async with httpx.AsyncClient(timeout=self._settings.openhands_timeout_seconds, headers=headers) as client:
            resp = await client.delete(f"{self._settings.openhands_base_url}/api/v1/app-conversations/{conv_id}")
            resp.raise_for_status()
