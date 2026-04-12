from __future__ import annotations

from typing import Any

import httpx

from app.domain.models.runtime import WorkflowRun
from app.domain.models.workflow_definition import WorkflowStepDefinition
from app.infrastructure.actions.templates import build_template_context, resolve_templates


class HttpStepExecutor:
    """
    Executes workflow steps of type ``http``.

    Makes an HTTP request to the configured URL, resolves ``{{ run.* }}`` templates
    in the body and headers, and returns the parsed JSON response (or a plain-text
    fallback) as a dict stored in ``intermediate_outputs``.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def execute(self, step: WorkflowStepDefinition, run: WorkflowRun) -> dict[str, Any]:
        context = build_template_context(run)
        url: str = resolve_templates(step.url or "", context)
        body: dict[str, Any] = resolve_templates(step.body, context)
        headers: dict[str, str] = resolve_templates(step.http_headers, context)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.request(
                method=step.method,
                url=url,
                json=body or None,
                headers=headers,
            )
            response.raise_for_status()

        try:
            return response.json()  # type: ignore[no-any-return]
        except Exception:
            return {"status_code": response.status_code, "body": response.text}
