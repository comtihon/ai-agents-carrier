"""How a workflow uses an agent: where it runs and which addons it carries.

An agent definition is a role (prompt, tools, addons) with a default runtime.
A workflow may run it elsewhere and give it different addons for that use:

* a static agent step: ``runtime`` and ``addons`` on the step;
* a dynamic step: ``runtime`` and ``addons`` on each ``agent_pool`` entry.

``runtime`` (``RuntimeSelection``):

* ``{"kind": "default"}`` (or omitted) — the agent's own ``default_runtime``;
* ``{"kind": "local" | "docker" | "k8s"}`` — that runtime;
* ``{"kind": "runner", "runner_id": ..., "acp_agent": ...}`` — one connected
  machine and the ACP agent on it (``acp_agent`` omitted: the machine's
  default agent);
* ``{"kind": "pool", "pool_id": ...}`` — a runtime pool.

``addons`` override the agent's own addons of the same type for this use; data
source addons are merged by ``source_id`` (the override wins for a source both
name).  Nothing is stored on the agent itself.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, TypeAdapter, model_validator

from app.domain.models.agent_addon import AnyAgentAddon

RuntimeKind = Literal["default", "local", "docker", "k8s", "runner", "pool"]


class RuntimeSelection(BaseModel):
    kind: RuntimeKind = "default"
    runner_id: str | None = None
    acp_agent: str | None = None
    pool_id: str | None = None

    @model_validator(mode="after")
    def _targets(self) -> "RuntimeSelection":
        if self.kind == "runner" and not self.runner_id:
            raise ValueError("runtime kind 'runner' needs runner_id")
        if self.kind == "pool" and not self.pool_id:
            raise ValueError("runtime kind 'pool' needs pool_id")
        return self

    @property
    def is_default(self) -> bool:
        return self.kind == "default"


_ADDONS = TypeAdapter(list[AnyAgentAddon])


def parse_runtime(value: Any, legacy_override: Any = None) -> RuntimeSelection | None:
    """The runtime a use selects: ``value`` (a dict or model), else the legacy
    ``runtime_override`` string (``local``/``docker``/``k8s``), else None."""
    if isinstance(value, RuntimeSelection):
        return None if value.is_default else value
    if isinstance(value, dict) and value:
        sel = RuntimeSelection.model_validate(value)
        return None if sel.is_default else sel
    if isinstance(legacy_override, str) and legacy_override in ("local", "docker", "k8s"):
        return RuntimeSelection(kind=legacy_override)  # type: ignore[arg-type]
    return None


def parse_addons(value: Any) -> list[Any]:
    if not value:
        return []
    return list(_ADDONS.validate_python(value))


def merge_addons(base: list[Any], overrides: list[Any]) -> list[Any]:
    """*base* with *overrides* applied: same type replaced, data sources by source_id."""
    if not overrides:
        return list(base)
    override_types = {a.type for a in overrides if a.type != "datasource"}
    override_sources = {getattr(a, "source_id", None) for a in overrides if a.type == "datasource"}
    kept = [
        a for a in base
        if not (a.type in override_types or (a.type == "datasource" and getattr(a, "source_id", None) in override_sources))
    ]
    return kept + list(overrides)


def apply_usage(agent_def: Any, runtime: Any = None, addons: Any = None, legacy_override: Any = None) -> Any:
    """*agent_def* as this use runs it (a copy; the definition is untouched).

    A legacy ``runtime_override`` string does not move an agent off its
    connected machine or pool (as before); an explicit ``runtime`` does.
    """
    sel = parse_runtime(runtime)
    if sel is None and getattr(agent_def, "default_runtime", None) not in ("runner", "pool"):
        sel = parse_runtime(None, legacy_override)
    extra = parse_addons(addons)
    if sel is None and not extra:
        return agent_def
    update: dict[str, Any] = {}
    if extra:
        update["addons"] = merge_addons(list(agent_def.addons or []), extra)
    if sel is not None:
        if sel.kind == "runner":
            update.update(default_runtime="runner", runner_id=sel.runner_id, pool_id=None,
                          protocol="acp", acp_agent=sel.acp_agent)
        elif sel.kind == "pool":
            update.update(default_runtime="pool", pool_id=sel.pool_id, runner_id=None,
                          protocol="acp", acp_agent=None)
        else:
            update.update(default_runtime=sel.kind, runner_id=None, pool_id=None)
    return agent_def.model_copy(update=update)


def usage_references(runtime: Any, addons: Any) -> list[tuple[str, str]]:
    """``(kind, id)`` a use references: its runner / pool and its addons' data sources."""
    refs: list[tuple[str, str]] = []
    try:
        sel = parse_runtime(runtime)
    except Exception:
        sel = None
    if sel is not None and sel.kind == "runner" and sel.runner_id:
        refs.append(("runner", sel.runner_id))
    if sel is not None and sel.kind == "pool" and sel.pool_id:
        refs.append(("pool", sel.pool_id))
    for addon in addons or []:
        source = addon.get("source_id") if isinstance(addon, dict) else getattr(addon, "source_id", None)
        kind = addon.get("type") if isinstance(addon, dict) else getattr(addon, "type", None)
        if kind == "datasource" and source:
            refs.append(("datasource", str(source)))
    return refs


def usage_datasource_operations(addons: Any) -> dict[str, set[str]]:
    ops: dict[str, set[str]] = {}
    for addon in addons or []:
        get = addon.get if isinstance(addon, dict) else (lambda k, a=addon: getattr(a, k, None))
        if get("type") == "datasource" and get("source_id"):
            ops.setdefault(str(get("source_id")), set()).update(str(o) for o in get("allowed_operations") or [])
    return ops
