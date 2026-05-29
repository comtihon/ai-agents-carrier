from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.domain.models.agent_definition import AgentDefinition


class AgentRuntime(ABC):
    """Abstract base class for agent runtimes.

    All runtimes share the same three-method interface so that step executors
    can be written against this contract and work with any backend.

    Lifecycle
    ---------
    1. ``spawn()``      — start the agent HTTP server; returns the agent's base URL
    2. ``terminate()``  — call POST {agent_url}/terminate then clean up
    3. ``is_alive()``   — call GET {agent_url}/health

    Output delivery
    ---------------
    Output no longer arrives via ``wait_for_output``.  Instead the agent calls
    back to the backend via ``POST /runs/{run_id}/agent/output`` once it has
    finished.  The backend resumes the paused LangGraph run at that point.
    """

    @abstractmethod
    async def spawn(
        self,
        agent_def: "AgentDefinition",
        step: dict[str, Any],
        run_id: str,
        callback_base_url: str,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        """Start the agent HTTP server and return its base URL.

        Parameters
        ----------
        agent_def:
            The registered agent definition (image, entrypoint, env, …).
        step:
            The raw step dict from the workflow YAML (contains image_override,
            entrypoint_override, timeout_seconds, etc.).
        run_id:
            The workflow run ID passed to the agent as ``RUN_ID`` env var.
        callback_base_url:
            The backend's public base URL passed to the agent as
            ``BACKEND_CALLBACK_URL`` env var.

        Returns
        -------
        str
            The agent's base URL, e.g. ``http://localhost:18042``.  Used by
            ``terminate()`` and ``is_alive()``.
        """
        ...

    @abstractmethod
    async def terminate(self, agent_url: str) -> None:
        """Request graceful shutdown then forcefully stop if necessary.

        Implementations should:
        1. Call ``POST {agent_url}/terminate`` (best-effort).
        2. Wait up to 5 s for the process / container to exit.
        3. Force-kill if it hasn't exited by then.

        Must be idempotent — calling terminate on an already-stopped agent must
        not raise.
        """
        ...

    def rewrite_callback_url(self, callback_base_url: str) -> str:
        """Rewrite the callback URL for use inside the agent's execution environment.

        Default implementation returns the URL unchanged.  Override in runtimes
        where the agent cannot reach ``localhost`` (e.g. Docker containers).
        """
        return callback_base_url

    @abstractmethod
    async def is_alive(self, agent_url: str) -> bool:
        """Return True if the agent server is still healthy.

        Implementations should call ``GET {agent_url}/health`` and return True
        only when the response status is 200.  Any network error returns False.
        """
        ...
