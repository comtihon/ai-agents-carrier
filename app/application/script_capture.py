"""Capture inline ``python`` step code into the script library on workflow save.

An agent that writes a workflow puts the Python body straight on the step.  That
body is then invisible to the library: nobody can find it, reuse it in another
workflow, or edit it in one place.  Saving a workflow therefore registers every
inline body as a ``ScriptDefinition`` and points the step at it via
``script_id`` — the same shape the UI's "Save to Library" produces, so the two
paths converge instead of drifting.

Identity
--------
The library id is derived from the workflow and step ids
(``slugify("<workflow_id>-<step_id>")``), not from the step's name.  A
name-derived id would collide with hand-written scripts: two workflows each with
a ``transform`` step would fight over one document, and re-saving would overwrite
a script somebody else owns.  Workflow+step is stable across saves, so repeated
saves update the same entry rather than piling up near-duplicates.

The inline ``code`` stays on the step.  ``script_id`` wins at execution time (see
``YamlGraphRunner._resolve_script_code``), so the copy is what the node shows
when the library is unreachable — never what runs.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.domain.models.script_definition import ScriptDefinition

if TYPE_CHECKING:
    from app.infrastructure.persistence.script_backend import ScriptDefinitionBackend

logger = logging.getLogger(__name__)


def slugify(name: str) -> str:
    """Library-id slug: lowercase, non-alphanumerics collapsed to single dashes."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "script"


async def capture_inline_scripts(
    workflow_id: str,
    steps: Any,
    script_backend: "ScriptDefinitionBackend | None",
) -> list[str]:
    """Register inline ``python`` bodies in the library, in place on *steps*.

    Each ``python`` step that carries ``code`` but no ``script_id`` gets a
    library entry and a ``script_id`` pointing at it.  Steps that already
    reference a script, and steps whose code is blank, are left alone.

    Failures are logged and skipped, never raised: a workflow save must not fail
    because the library is unavailable — the step still runs from its inline
    code.

    Returns
    -------
    list[str]
        Ids of the scripts created or updated, in step order.
    """
    if script_backend is None or not isinstance(steps, list):
        return []

    captured: list[str] = []
    for step in steps:
        if not isinstance(step, dict) or step.get("type") != "python":
            continue
        if step.get("script_id"):
            continue
        code = step.get("code")
        if not isinstance(code, str) or not code.strip():
            continue

        step_id = str(step.get("id") or "").strip()
        if not step_id:
            continue

        script_id = slugify(f"{workflow_id}-{step_id}")
        name = str(step.get("name") or step_id)
        description = step.get("description") or (
            f"Captured from step '{step_id}' of workflow '{workflow_id}'"
        )

        try:
            existing = await script_backend.get(script_id)
            if existing is None:
                await script_backend.create(
                    ScriptDefinition(
                        id=script_id, name=name, description=description, code=code,
                    )
                )
                logger.info(
                    "workflow '%s' step '%s': python body captured as script '%s'",
                    workflow_id, step_id, script_id,
                )
            elif existing.code != code or existing.name != name:
                existing.name = name
                existing.description = description
                existing.code = code
                await script_backend.update(script_id, existing)
                logger.info(
                    "workflow '%s' step '%s': script '%s' updated from the step body",
                    workflow_id, step_id, script_id,
                )
        except Exception as exc:
            logger.warning(
                "workflow '%s' step '%s': could not capture python body into the "
                "library: %s",
                workflow_id, step_id, exc,
            )
            continue

        step["script_id"] = script_id
        captured.append(script_id)

    return captured
