from __future__ import annotations

import re
from typing import Any

from app.domain.models.runtime import WorkflowRun

_TEMPLATE_RE = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")


def build_template_context(run: WorkflowRun) -> dict[str, str]:
    """Return the set of template variables available in http/action step fields."""
    return {
        "run.id": run.id,
        "run.workflow_id": run.workflow_id,
        "run.workflow_name": run.workflow_name,
        "run.user_request": run.user_request,
    }


def resolve_templates(value: Any, context: dict[str, str]) -> Any:
    """
    Recursively replace {{ key }} placeholders in strings, dict values, and list items.
    Unknown keys are left unchanged.
    """
    if isinstance(value, str):
        return _TEMPLATE_RE.sub(lambda m: context.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: resolve_templates(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_templates(v, context) for v in value]
    return value
