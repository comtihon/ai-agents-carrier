from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from app.infrastructure.actions.registry import ActionRegistry


def load_actions_from_directory(directory: str, registry: ActionRegistry) -> int:
    """
    Scan *directory* for ``*.py`` files and register any handlers declared in
    their ``ACTIONS`` dict.

    Convention — each Python file may define a module-level ``ACTIONS`` dict
    that maps handler name → async callable:

    .. code-block:: python

        # workflows/my_actions.py

        async def send_notification(handler_input: dict, run) -> dict:
            ...
            return {"sent": True}

        ACTIONS = {
            "my_module.send_notification": send_notification,
        }

    Files that do not define an ``ACTIONS`` dict are silently skipped.
    Import errors raise immediately so misconfigured files are visible at startup.

    Returns the number of handlers registered.
    """
    path = Path(directory)
    if not path.is_dir():
        return 0

    registered = 0
    for py_file in sorted(path.glob("*.py")):
        module_name = f"_workflow_actions_{py_file.stem}"

        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        actions = getattr(module, "ACTIONS", None)
        if not isinstance(actions, dict):
            continue

        for name, handler in actions.items():
            if not callable(handler):
                raise TypeError(
                    f"ACTIONS['{name}'] in '{py_file.name}' is not callable."
                )
            registry.register(name, handler)
            registered += 1

    return registered
