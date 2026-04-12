from __future__ import annotations

from typing import Any

from app.domain.interfaces.actions import ActionHandler
from app.domain.models.runtime import WorkflowRun
from app.infrastructure.actions.templates import build_template_context, resolve_templates


class ActionRegistry:
    """
    Registry of named Python action handlers.

    Handlers are registered at application startup and invoked by name from
    workflow steps of type ``action``.  Each handler receives a resolved input
    dict (with ``{{ run.* }}`` templates expanded) and the live ``WorkflowRun``,
    and must return a dict that is stored in ``intermediate_outputs``.

    Usage::

        registry = ActionRegistry()

        async def transition_jira(handler_input: dict, run: WorkflowRun) -> dict:
            issue_key = handler_input["issue_key"]
            # ... call internal API ...
            return {"transitioned": True}

        registry.register("jira.transition_issue", transition_jira)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {}

    def register(self, name: str, handler: ActionHandler) -> None:
        self._handlers[name] = handler

    def get(self, name: str) -> ActionHandler | None:
        return self._handlers.get(name)

    def registered_names(self) -> list[str]:
        return list(self._handlers.keys())

    async def execute(self, name: str, handler_input: dict[str, Any], run: WorkflowRun) -> dict[str, Any]:
        handler = self.get(name)
        if handler is None:
            available = self.registered_names()
            raise KeyError(
                f"Action handler '{name}' is not registered. "
                f"Available handlers: {available or '(none)'}"
            )
        context = build_template_context(run)
        resolved_input = resolve_templates(handler_input, context)
        return await handler(resolved_input, run)
