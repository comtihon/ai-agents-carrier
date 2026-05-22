from __future__ import annotations

"""LocalRuntime — no-op stub.

The local runtime no longer spawns any subprocess.  Local agents run inline
inside the backend process via ``app.agents.local_agent.run_local_agent``.

The ``LocalRuntime`` class is kept so that ``get_runtime("local")`` still
returns a valid object, but ``spawn()`` raises ``RuntimeError`` to make it
clear that the executor must handle the local case *before* calling spawn.
"""

import logging
from typing import TYPE_CHECKING, Any

from app.runtime.base import AgentRuntime

if TYPE_CHECKING:
    from app.domain.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)


class LocalRuntime(AgentRuntime):
    """Inline local agent — does NOT spawn a subprocess.

    The executor branches on ``runtime == "local"`` before reaching this class
    and calls ``run_local_agent`` directly.  This class exists only so that
    ``get_runtime("local")`` satisfies the ``AgentRuntime`` ABC and can be used
    in type checks / isinstance tests.
    """

    async def spawn(
        self,
        agent_def: "AgentDefinition",
        step: dict[str, Any],
        run_id: str,
        callback_base_url: str,
    ) -> str:
        raise RuntimeError(
            "LocalRuntime.spawn() should never be called — the executor must "
            "handle the 'local' runtime inline via run_local_agent() before "
            "reaching the spawn() dispatch path."
        )

    async def terminate(self, agent_url: str) -> None:
        # Nothing to terminate for inline agents.
        pass

    async def is_alive(self, agent_url: str) -> bool:
        # Inline agents don't have a URL.
        return False
