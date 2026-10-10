"""The dispatcher: the meta-LLM that turns a request into jobs.

It is the one built-in participant of a dynamic workflow. Planners, coders,
validators are the user's own agents; the dispatcher only decides which of
them fill which job, writes each one a brief, and wires the dependencies.

Every answer is JSON, validated against the pool and the slot limits before
the engine sees it. An invalid answer gets one repair round with the exact
errors; a second failure raises — the engine turns that into a failed run (or
an escalation) rather than executing a plan nobody checked.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.domain.models.dynamic import JOB_CATEGORIES, DynamicConfig, Job, JobSpec

logger = logging.getLogger(__name__)

LlmCall = Callable[[str, str], Awaitable[tuple[str, dict[str, int] | None]]]

_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")


class DispatcherError(RuntimeError):
    pass


ProposedJob = JobSpec


class ProposedPlan(BaseModel):
    summary: str = ""
    jobs: list[ProposedJob] = Field(default_factory=list)


class ProposedDelegate(BaseModel):
    agent_id: str
    title: str = ""
    prompt: str


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


# ─── Validation ──────────────────────────────────────────────────────────────


def validate_jobs(
    proposed: list[ProposedJob],
    config: DynamicConfig,
    existing: list[Job],
    *,
    allowed_categories: tuple[str, ...] = JOB_CATEGORIES,
) -> list[str]:
    """Return every problem with *proposed* given the jobs that already exist."""
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
        entry = config.pool_entry(job.agent_id)
        if entry is None:
            errors.append(f"job {job.id!r}: agent {job.agent_id!r} is not in the pool")
        elif entry.categories and job.category not in entry.categories:
            errors.append(f"job {job.id!r}: agent {job.agent_id!r} may only fill {entry.categories}")
        if not job.prompt.strip():
            errors.append(f"job {job.id!r}: prompt is empty")

    known = existing_ids | set(new_ids)
    for job in proposed:
        for dep in job.depends_on:
            if dep not in known:
                errors.append(f"job {job.id!r} depends on unknown job {dep!r}")
            if dep == job.id:
                errors.append(f"job {job.id!r} depends on itself")

    # Cycle check over the new jobs (existing ones are already acyclic and
    # cannot depend on new ones).
    graph = {j.id: [d for d in j.depends_on if d in set(new_ids)] for j in proposed}
    state: dict[str, int] = {}

    def _visit(node: str) -> bool:
        state[node] = 1
        for nxt in graph.get(node, []):
            if state.get(nxt) == 1 or (state.get(nxt) is None and _visit(nxt)):
                return True
        state[node] = 2
        return False

    for node in graph:
        if state.get(node) is None and _visit(node):
            errors.append("depends_on contains a cycle; iterate with handbacks, not cycles")
            break

    for cat in JOB_CATEGORIES:
        count = sum(1 for j in existing if j.category == cat) + sum(1 for j in proposed if j.category == cat)
        slot = config.jobs[cat]
        if count > slot.max:
            errors.append(f"too many {cat} jobs: {count} > max {slot.max}")
    total = len(existing) + len(proposed)
    if total > config.limits.max_total_jobs:
        errors.append(f"too many jobs in total: {total} > {config.limits.max_total_jobs}")
    return errors


def check_minimums(jobs: list[Job] | list[ProposedJob], config: DynamicConfig) -> list[str]:
    errors = []
    for cat in JOB_CATEGORIES:
        count = sum(1 for j in jobs if j.category == cat)
        if count < config.jobs[cat].min:
            errors.append(f"at least {config.jobs[cat].min} {cat} job(s) required, plan has {count}")
    return errors


# ─── Prompts ─────────────────────────────────────────────────────────────────

_SYSTEM = """You are the dispatcher of a multi-agent workflow. You never do the work yourself.
You read a request and a roster of available agents, and you decide which agents fill which jobs.

Job categories (every one is optional; use only those the request needs and the job slots require):
- planning: research / design / break the work down. A planning chain is fine (e.g. researcher -> planner).
- execution: does the actual work (code, modelling, data changes). Leave it out when nothing has to change
  (e.g. a pure research or review request).
- validation: checks the executors' results (tests, review, geometry checks). Its output carries a verdict;
  a failing verdict sends the work back to the executors it names for another iteration.
- integration: merges the results of parallel executors when they must be combined.

Rules:
- Only use agent_ids from the roster, in the categories the roster allows them.
- Pick agents by their description and capabilities: give a job to an agent that can reach what the job needs.
- Split a big piece of work into parts and give each part its own job: several researchers on different
  questions, several coders on different features, several validators each checking one part. The same agent
  may fill several jobs up to its max parallel instances. A part that fails is retried on its own; the others
  keep their results.
- Parallelise only when the jobs touch disjoint parts. Give every execution job an "owns" list naming what it
  may change (file globs like "src/billing/**", or named parts like "collection:Roof"). If two parts are
  tightly coupled, make them sequential with depends_on instead. A validator of one part depends on that
  part's executor only.
- depends_on must form a DAG. Do not create loops: feedback loops happen at run time through handbacks.
- Each prompt is the complete brief for that agent: goal, scope, inputs it will receive, what to output.
- Keep it lean: the fewest jobs that do the work well.
"""

_FORMAT = """Answer with ONLY a JSON object in a ```json fence:
{
  "summary": "<one paragraph: how the work is split and why>",
  "jobs": [
    {"id": "<slug, e.g. plan-1, code-auth>", "category": "planning|execution|validation|integration",
     "agent_id": "<from roster>", "title": "<short>", "prompt": "<full brief>",
     "depends_on": ["<job id>", ...], "owns": ["<glob or named part>", ...]}
  ]
}"""


def _limits_text(config: DynamicConfig) -> str:
    slots = ", ".join(f"{c}: {config.jobs[c].min}-{config.jobs[c].max}" for c in JOB_CATEGORIES)
    return f"Job slots (min-max): {slots}. Max total jobs: {config.limits.max_total_jobs}."


def _context_text(config: DynamicConfig) -> str:
    parts = []
    if config.repo is not None:
        verify = "; ".join(config.repo.verify) or "none configured"
        parts.append(f"Repository: {config.repo.url} (base branch {config.repo.base_branch}; verify: {verify})")
    if config.shared_volume is not None:
        parts.append(f"All jobs share a volume at {config.shared_volume.mount_point}.")
    if config.dispatcher_instructions.strip():
        parts.append("Author's instructions:\n" + config.dispatcher_instructions.strip())
    return "\n".join(parts)


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


# ─── Dispatcher ──────────────────────────────────────────────────────────────


class Dispatcher:
    def __init__(self, llm_call: LlmCall) -> None:
        self._llm_call = llm_call
        self.usage: dict[str, int] = {}

    async def _ask(self, user: str) -> str:
        text, usage = await self._llm_call(_SYSTEM, user)
        for k, v in (usage or {}).items():
            self.usage[k] = self.usage.get(k, 0) + int(v or 0)
        return text

    async def _ask_validated(self, user: str, parse: Callable[[Any], tuple[Any, list[str]]]) -> Any:
        text = await self._ask(user)
        for attempt in range(2):
            try:
                value, errors = parse(_extract_json(text))
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                value, errors = None, [f"answer is not valid JSON for the schema: {exc}"]
            if not errors:
                return value
            if attempt == 1:
                raise DispatcherError("dispatcher answer still invalid after repair: " + "; ".join(errors))
            logger.info("dispatcher answer invalid, repairing: %s", errors)
            text = await self._ask(
                user
                + "\n\nYour previous answer:\n"
                + _truncate(text, 8000)
                + "\n\nIt was rejected for these reasons:\n- "
                + "\n- ".join(errors)
                + "\n\nAnswer again, fixing every problem."
            )
        raise DispatcherError("unreachable")

    async def plan(
        self,
        request: str,
        config: DynamicConfig,
        roster: str,
        *,
        feedback: str | None = None,
        previous: list[Job] | None = None,
    ) -> ProposedPlan:
        user = (
            f"Request:\n{request}\n\nRoster:\n{roster}\n\n{_limits_text(config)}\n{_context_text(config)}\n\n"
            "Decide the jobs. If the roster has planning agents and the request needs design or research "
            "first, you may return only the planning jobs: once they finish you will be asked to turn their "
            "output into execution and validation jobs. Otherwise return the full set of jobs now.\n"
        )
        if previous:
            user += "\nA human rejected the previous plan:\n" + _truncate([j.model_dump(include={"id", "category", "agent_id", "title", "prompt", "depends_on"}) for j in previous])
        if feedback:
            user += f"\nTheir reason: {feedback}\nAddress it.\n"
        user += "\n" + _FORMAT

        def parse(data: Any) -> tuple[ProposedPlan, list[str]]:
            plan = ProposedPlan.model_validate(data)
            errors = validate_jobs(plan.jobs, config, [])
            if not plan.jobs:
                errors.append("the plan has no jobs")
            only_planning = plan.jobs and all(j.category == "planning" for j in plan.jobs)
            if not only_planning:
                errors += check_minimums(plan.jobs, config)
            return plan, errors

        return await self._ask_validated(user, parse)

    async def expand(
        self,
        request: str,
        config: DynamicConfig,
        roster: str,
        existing: list[Job],
    ) -> ProposedPlan:
        """Turn finished planning jobs' output into the remaining jobs."""
        planning = [
            {"id": j.id, "agent_id": j.agent_id, "title": j.title, "output": j.output}
            for j in existing
            if j.category == "planning"
        ]
        user = (
            f"Request:\n{request}\n\nRoster:\n{roster}\n\n{_limits_text(config)}\n{_context_text(config)}\n\n"
            "The planning jobs have finished. Their output:\n"
            + _truncate(planning, 20000)
            + "\n\nNow create the execution jobs (and validation / integration jobs if useful) that carry the "
            "plan out. New jobs may depend on the planning jobs and on each other. Follow the planners' split "
            "of the work when it is sound; you decide which agents take each part. If the planners' output "
            "already answers the request and nothing needs to be done, return an empty jobs list.\n\n"
            + _FORMAT
        )
        allowed = ("execution", "validation", "integration")

        def parse(data: Any) -> tuple[ProposedPlan, list[str]]:
            plan = ProposedPlan.model_validate(data)
            errors = validate_jobs(plan.jobs, config, existing, allowed_categories=allowed)
            errors += check_minimums(list(existing) + list(plan.jobs), config)
            return plan, errors

        return await self._ask_validated(user, parse)

    async def assign_delegate(
        self,
        request: str,
        config: DynamicConfig,
        roster: str,
        requester: Job,
        ask: str,
        existing: list[Job],
    ) -> ProposedDelegate:
        """Pick the agent that answers *requester*'s delegation."""
        user = (
            f"Overall request:\n{request}\n\nRoster:\n{roster}\n\n"
            f"Job {requester.id!r} ({requester.category}, agent {requester.agent_id}) is asking for help:\n{ask}\n\n"
            "Pick ONE agent from the roster to handle this and write its brief. Its result goes back to the "
            "asking job. Prefer planning-capable agents (e.g. researchers) for information gathering.\n"
            'Answer with ONLY a JSON object in a ```json fence: {"agent_id": "...", "title": "...", "prompt": "..."}'
        )

        def parse(data: Any) -> tuple[ProposedDelegate, list[str]]:
            proposal = ProposedDelegate.model_validate(data)
            errors = []
            if config.pool_entry(proposal.agent_id) is None:
                errors.append(f"agent {proposal.agent_id!r} is not in the pool")
            if not proposal.prompt.strip():
                errors.append("prompt is empty")
            if len(existing) + 1 > config.limits.max_total_jobs:
                errors.append("job budget exhausted")
            return proposal, errors

        return await self._ask_validated(user, parse)


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
