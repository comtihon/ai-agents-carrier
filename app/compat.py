"""
Compatibility shims for third-party library version mismatches.

Import this module FIRST in app/api/app.py before any copilotkit or
langgraph imports.

Patches applied
---------------
1. ``langgraph.graph.graph.CompiledGraph``
   copilotkit 0.1.x imports this; it moved to
   ``langgraph.graph.state.CompiledStateGraph`` in langgraph >= 0.4.

2. ``langchain_core.memory.BaseMemory``
   ``langchain 0.3.x`` re-exports ``BaseMemory`` from ``langchain_core.memory``
   but that module was removed in ``langchain_core 1.x``.
"""
import sys
import types
from abc import ABC


def _patch_langgraph() -> None:
    """Make copilotkit's CompiledGraph import work on langgraph >= 0.4."""
    if "langgraph.graph.graph" in sys.modules:
        return
    try:
        from langgraph.graph.state import CompiledStateGraph  # noqa: PLC0415
    except ImportError:
        return
    fake: types.ModuleType = types.ModuleType("langgraph.graph.graph")
    fake.CompiledGraph = CompiledStateGraph  # type: ignore[attr-defined]
    sys.modules["langgraph.graph.graph"] = fake


def _patch_langchain_core_memory() -> None:
    """Provide a stub langchain_core.memory for langchain 0.3.x on langchain-core 1.x."""
    if "langchain_core.memory" in sys.modules:
        return
    module: types.ModuleType = types.ModuleType("langchain_core.memory")

    class BaseMemory(ABC):  # noqa: B024
        """Stub for langchain_core.memory.BaseMemory (removed in langchain-core 1.x)."""

    module.BaseMemory = BaseMemory  # type: ignore[attr-defined]
    sys.modules["langchain_core.memory"] = module


_patch_langchain_core_memory()
_patch_langgraph()
