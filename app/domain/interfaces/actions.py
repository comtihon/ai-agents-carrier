from __future__ import annotations

from typing import Any, Protocol

from app.domain.models.runtime import WorkflowRun


class ActionHandler(Protocol):
    """
    Protocol for Python action handlers registered in ActionRegistry.

    A handler receives the resolved input dict and the current WorkflowRun,
    and returns a dict that is stored in intermediate_outputs under the step's
    output_key (or step id).

    Example::

        async def notify_slack(handler_input: dict, run: WorkflowRun) -> dict:
            channel = handler_input["channel"]
            ...
            return {"message_ts": "..."}

        registry.register("slack.notify", notify_slack)
    """

    async def __call__(self, handler_input: dict[str, Any], run: WorkflowRun) -> dict[str, Any]: ...
