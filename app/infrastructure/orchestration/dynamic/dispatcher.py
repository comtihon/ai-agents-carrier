"""Shared helpers of the meta-agent: the roster it picks agents from, the
checks every planned job passes, and the meta-LLM call.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any


from app.domain.models.dynamic import JOB_CATEGORIES, DynamicConfig, Job, JobSpec

logger = logging.getLogger(__name__)

LlmCall = Callable[[str, str], Awaitable[tuple[str, dict[str, int] | None]]]

_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")


class DispatcherError(RuntimeError):
    """The meta-agent could not produce a valid decision."""


ProposedJob = JobSpec


# ─── Roster ──────────────────────────────────────────────────────────────────


def describe_agent(agent_def: Any) -> str:
    """One roster entry: what the agent is for and what it can reach.

    The description is the author's own words about the agent — the only thing
    the dispatcher has to pick by — and the addon summary tells it what the
    agent can actually touch, so it does not hand a datasource task to an agent
    without that datasource.
    """
    parts: list[str] = []
    mcp = getattr(agent_def, "mcp_addon", None)
    if mcp is not None:
        servers = sorted(mcp.enabled_servers())
        if servers:
            parts.append("MCP: " + ", ".join(servers))
    tools = getattr(agent_def, "tools_addon", None)
    if tools is not None:
        enabled = sorted(k for k, v in (tools.tools or {}).items() if v)
        if enabled:
            parts.append("tools: " + ", ".join(enabled))
    for ds in getattr(agent_def, "datasource_addons", []) or []:
        ops = ", ".join(ds.allowed_operations) if ds.allowed_operations else "no operations"
        parts.append(f"datasource {ds.source_id} ({ops})")
    if getattr(agent_def, "s3_addon", None) is not None:
        parts.append("persistent workspace (S3)")
    caps = "; ".join(parts) if parts else "no addons"
    desc = (agent_def.description or "").strip() or "(no description)"
    return f"{desc} [{caps}]"


def build_roster(config: DynamicConfig, agents: dict[str, Any]) -> str:
    lines = []
    for entry in config.agent_pool:
        agent_def = agents.get(entry.agent_id)
        if agent_def is None:
            continue
        cats = ", ".join(entry.categories) if entry.categories else "any"
        name = agent_def.name or entry.agent_id
        lines.append(
            f"- agent_id: {entry.agent_id} | name: {name} | categories: {cats} | "
            f"max parallel instances: {entry.max_instances}\n  {describe_agent(agent_def)}"
        )
    return "\n".join(lines) if lines else "(empty pool)"


def describe_datasources(config: DynamicConfig) -> str:
    """The data sources the meta-agent may grant to jobs."""
    if not config.datasources:
        return "(none: jobs only reach the data sources their agents already have)"
    return "\n".join(
        f"- {d.source_id}: operations {', '.join(d.operations) or '(none)'}"
        + (f"\n  {d.description.strip()}" if d.description.strip() else "")
        for d in config.datasources
    )


# ─── Validation ──────────────────────────────────────────────────────────────


def validate_jobs(
    proposed: list[JobSpec],
    config: DynamicConfig,
    existing: list[Job],
    *,
    allowed_categories: tuple[str, ...] = JOB_CATEGORIES,
) -> list[str]:
    """Return every problem with *proposed* given the jobs that already exist.

    *existing* are the jobs still in the DAG; ids of removed jobs are passed in
    too only to keep ids unique.
    """
    errors: list[str] = []
    existing_ids = {j.id for j in existing}
    new_ids: list[str] = []
    for job in proposed:
        if not _ID_RE.match(job.id):
            errors.append(f"job id {job.id!r} must match {_ID_RE.pattern}")
        if job.id in existing_ids or job.id in new_ids:
            errors.append(f"job id {job.id!r} is already used")
        new_ids.append(job.id)
        if job.category not in allowed_categories:
            errors.append(f"job {job.id!r}: category {job.category!r} is not allowed here (allowed: {list(allowed_categories)})")
        errors += check_assignment(job.id, job.category, job.agent_id, job.datasources, config)
        if not job.prompt.strip():
            errors.append(f"job {job.id!r}: prompt is empty")

    active_ids = {j.id for j in existing if j.active}
    known = active_ids | set(new_ids)
    for job in proposed:
        for dep in job.depends_on:
            if dep == job.id:
                errors.append(f"job {job.id!r} depends on itself")
            elif dep not in known:
                errors.append(f"job {job.id!r} depends on unknown or removed job {dep!r}")

    graph = {j.id: list(j.depends_on) for j in existing if j.active}
    graph.update({j.id: list(j.depends_on) for j in proposed})
    if has_cycle(graph):
        errors.append("depends_on contains a cycle; loop by rewinding, not with cycles")

    for cat in JOB_CATEGORIES:
        count = sum(1 for j in existing if j.active and j.category == cat) + sum(1 for j in proposed if j.category == cat)
        slot = config.jobs[cat]
        if count > slot.max:
            errors.append(f"too many {cat} jobs: {count} > max {slot.max}")
    total = len(existing) + len(proposed)
    if total > config.limits.max_total_jobs:
        errors.append(f"too many jobs in total: {total} > {config.limits.max_total_jobs}")
    return errors


def check_assignment(job_id: str, category: str, agent_id: str, datasources: list[str], config: DynamicConfig) -> list[str]:
    """The agent may fill the category, and every data source is enabled for the step."""
    errors: list[str] = []
    entry = config.pool_entry(agent_id)
    if entry is None:
        errors.append(f"job {job_id!r}: agent {agent_id!r} is not in the pool")
    elif entry.categories and category not in entry.categories:
        errors.append(f"job {job_id!r}: agent {agent_id!r} may only fill {entry.categories}")
    for source_id in datasources:
        if config.datasource(source_id) is None:
            errors.append(f"job {job_id!r}: data source {source_id!r} is not enabled for this step")
    return errors


def has_cycle(graph: dict[str, list[str]]) -> bool:
    state: dict[str, int] = {}

    def _visit(node: str) -> bool:
        state[node] = 1
        for nxt in graph.get(node, []):
            if nxt not in graph:
                continue
            if state.get(nxt) == 1 or (state.get(nxt) is None and _visit(nxt)):
                return True
        state[node] = 2
        return False

    return any(state.get(node) is None and _visit(node) for node in list(graph))


def check_minimums(jobs: list[Job] | list[JobSpec], config: DynamicConfig) -> list[str]:
    errors = []
    for cat in JOB_CATEGORIES:
        count = sum(1 for j in jobs if j.category == cat and getattr(j, "active", True))
        if count < config.jobs[cat].min:
            errors.append(f"at least {config.jobs[cat].min} {cat} job(s) required, plan has {count}")
    return errors


def _extract_json(text: str) -> Any:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            return json.loads(candidate[start : end + 1])
        raise


def _truncate(value: Any, cap: int = 6000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= cap else text[:cap] + f"... [truncated {len(text) - cap} chars]"


def build_llm_call(config: DynamicConfig, settings: Any) -> LlmCall:
    """The production LLM call: META_LLM settings unless the step overrides them."""

    async def call(system: str, user: str) -> tuple[str, dict[str, int] | None]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.core.container import build_llm_native

        provider = config.dispatcher_provider or settings.meta_llm_provider or settings.llm_provider
        model = config.dispatcher_model or settings.meta_llm_model
        llm = build_llm_native(provider, model, settings, max_tokens=8192)
        response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        text = response.content if isinstance(response.content, str) else str(response.content)
        meta = getattr(response, "usage_metadata", None)
        usage = (
            {
                "input_tokens": meta.get("input_tokens", 0),
                "output_tokens": meta.get("output_tokens", 0),
                "total_tokens": meta.get("total_tokens", 0),
            }
            if meta
            else None
        )
        return text, usage

    return call
