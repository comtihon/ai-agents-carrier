"""BackendClient — used by agent code to communicate back to the backend.

The client is created by the agent server and passed to the agent's ``run``
coroutine.  It wraps the backend's agent-callback HTTP endpoints.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 5.0   # seconds between long-poll retries
_DEFAULT_MAX_WAIT = 3600.0     # 1 hour max wait for an answer


class BackendClient:
    """HTTP client for agent-to-backend callbacks.

    Parameters
    ----------
    callback_url:
        Base URL of the backend, e.g. ``http://localhost:8000``.
    run_id:
        The workflow run ID assigned by the backend.
    poll_interval:
        Seconds to wait between retries when ``ask_question`` long-polls.
    max_wait:
        Maximum total seconds to wait for an answer before raising.
    api_prefix:
        API prefix used by the backend (default ``/api/v1``).
    """

    def __init__(
        self,
        callback_url: str,
        run_id: str,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        max_wait: float = _DEFAULT_MAX_WAIT,
        api_prefix: str = "/api/v1",
    ) -> None:
        self._callback_url = callback_url.rstrip("/")
        self._run_id = run_id
        self._poll_interval = poll_interval
        self._max_wait = max_wait
        self._base = f"{self._callback_url}{api_prefix}/runs/{run_id}/agent"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send_output(self, output: dict[str, Any]) -> None:
        """Signal to the backend that the agent has finished.

        Parameters
        ----------
        output:
            The final output dict.  Keys will be mapped back to workflow state
            according to the step's ``output_mapping`` / ``output_key``.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/output",
                json={"output": output},
                timeout=30.0,
            )
            resp.raise_for_status()
        logger.info("[agent run_id=%s] output delivered to backend", self._run_id)

    async def ask_question(
        self,
        question: str,
        options: list[str] | None = None,
    ) -> str:
        """Pause the agent and ask the user a clarifying question.

        Sends the question to the backend (which forwards it to the frontend),
        then long-polls ``GET /agent/input`` until an answer arrives.

        Parameters
        ----------
        question:
            The question text to show the user.
        options:
            Optional list of allowed answer values shown as choices.

        Returns
        -------
        str
            The user's answer.

        Raises
        ------
        TimeoutError
            If no answer arrives within ``max_wait`` seconds.
        """
        # Step 1: Post the question so the backend/frontend can display it.
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/question",
                json={"question": question, "options": options},
                timeout=10.0,
            )
            resp.raise_for_status()
        logger.info("[agent run_id=%s] question sent: %r", self._run_id, question)

        # Step 2: Long-poll for the answer.
        elapsed = 0.0
        while elapsed < self._max_wait:
            try:
                async with httpx.AsyncClient() as client:
                    # The backend holds the connection open for up to 10 min
                    # then returns 408 if no answer arrived.
                    resp = await client.get(
                        f"{self._base}/input",
                        timeout=620.0,  # slightly longer than the server's 10 min
                    )
                if resp.status_code == 200:
                    answer: str = resp.json().get("answer", "")
                    logger.info("[agent run_id=%s] received answer: %r", self._run_id, answer)
                    return answer
                if resp.status_code == 408:
                    # Server timed out — retry after a short pause
                    logger.debug("[agent run_id=%s] long-poll timed out, retrying", self._run_id)
                    await asyncio.sleep(self._poll_interval)
                    elapsed += self._poll_interval
                    continue
                resp.raise_for_status()
            except httpx.ReadTimeout:
                logger.debug("[agent run_id=%s] read timeout on /input, retrying", self._run_id)
                await asyncio.sleep(self._poll_interval)
                elapsed += self._poll_interval

        raise TimeoutError(
            f"No answer received for run_id={self._run_id} within {self._max_wait:.0f}s"
        )

    async def send_progress(self, message: str) -> None:
        """Send an optional progress update to the backend.

        Parameters
        ----------
        message:
            A human-readable progress message (e.g. "Analysing repository…").
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base}/progress",
                    json={"message": message},
                    timeout=5.0,
                )
                resp.raise_for_status()
        except Exception as exc:
            # Progress is best-effort — never fail the agent because of it.
            logger.warning("[agent run_id=%s] failed to send progress: %s", self._run_id, exc)
