"""The meta-agent: the one participant of a job workflow that sees everything.

Planners, researchers, coders and validators are the user's own agents; the
meta-agent orchestrates them. The job graph asks it to ``decide`` whenever
something happens that fixed rules cannot settle (a ``Situation``), handing it
the request, the agent roster, the data sources it may grant and the whole
DAG. It answers with a ``MetaDecision``: a list of actions the job graph
validates and applies.

``MetaAgent`` is the interface. ``LlmMetaAgent`` is today's implementation —
a meta-LLM called through LangChain, answering in JSON. A separate agent can
implement the same interface later: it gets the same view and returns the same
actions, and the job graph does not change.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from app.domain.models.dynamic import JOB_CATEGORIES, DynamicConfig, DynamicRunState, Job, JobSpec, Situation
from app.infrastructure.orchestration.dynamic.dispatcher import (
    DispatcherError,
    LlmCall,
    _extract_json,
    _truncate,
    describe_datasources,
)

logger = logging.getLogger(__name__)

ActionKind = Literal["add_jobs", "update_job", "rewind", "remove", "answer", "message", "ask_human", "finish", "fail"]


class MetaAction(BaseModel):
    """One step of a decision. Which fields matter depends on ``kind``:

    - add_jobs:   ``jobs`` — new jobs; they may depend on existing ones.
    - update_job: ``job_id`` + ``changes`` (prompt, title, agent_id, depends_on,
                  owns, datasources) — for a job that is not running.
    - rewind:     ``job_ids`` + ``text`` — these jobs run again with ``text`` as a
                  note; every job downstream of them runs again too. A running
                  job is stopped first.
    - remove:     ``job_ids`` + ``text`` — take jobs out of the DAG (running ones
                  are stopped). Jobs that depended on them must be rewired.
    - answer:     ``job_id`` + ``text`` — answer the job's question: live to a
                  running agent that is waiting, or for the next attempt of a job
                  that stopped to ask.
    - message:    ``job_id`` + ``text`` (+ ``restart``) — tell a job something.
                  With ``restart`` a running agent is stopped and the job starts
                  over with the message; without it the agent gets it with its
                  next answer or attempt.
    - ask_human:  ``text`` (+ ``job_id``) — a question only a human can answer;
                  the answer goes to the job.
    - finish:     ``text`` — the work is done; anything not finished is dropped.
    - fail:       ``text`` — the request cannot be done.
    """

    kind: ActionKind
    jobs: list[JobSpec] = Field(default_factory=list)
    job_id: str | None = None
    job_ids: list[str] = Field(default_factory=list)
    text: str = ""
    restart: bool = False
    changes: dict[str, Any] = Field(default_factory=dict)


class MetaDecision(BaseModel):
    summary: str = ""
    actions: list[MetaAction] = Field(default_factory=list)


@dataclass
class MetaView:
    """What the meta-agent is shown."""

    request: str
    config: DynamicConfig
    roster: str
    dag: DynamicRunState
    situations: list[Situation]


Validate = Callable[[MetaDecision], list[str]]


class MetaAgent(Protocol):
    # Token usage since the job graph last collected it.
    usage: dict[str, int]

    async def decide(self, view: MetaView, validate: Validate) -> MetaDecision:
        """Return a decision for *view*'s situations that *validate* accepts.

        Raises ``DispatcherError`` when no valid decision can be produced.
        """
        ...


# ─── The meta-LLM implementation ─────────────────────────────────────────────

_SYSTEM = """You are the meta-agent of a multi-agent workflow. You never do the work yourself: you orchestrate
agents from a roster. You see the request, the roster, the data sources you may grant, and the whole job DAG
with every job's status and output. You are called whenever something needs a decision, and you answer with
actions.

Job categories (every one is optional; use only what the request needs):
- planning: information gathering, research, design, breaking the work down.
- execution: does the actual work (code, modelling, data changes).
- validation: checks other jobs' results. Its output carries a verdict; a failing verdict comes back to you.
- integration: combines the results of parallel executors.

How to orchestrate:
- If you understand the request well enough, plan the work. If you do not, first add information-gathering
  jobs only; you are called again when they finish (phase_done) and plan the next phase from their output.
  Phases may repeat: gather, research more if needed, then coders, then validators.
- Split a big piece of work into parts, one job per part: several researchers on different questions, several
  coders on different features, a validator per part. Parallelise only disjoint parts and give every execution
  job an "owns" list (file globs or named parts). A validator of a part depends on that part only.
- Pick for each job the agent whose description and capabilities fit, and grant it the data sources (from the
  enabled list) it needs, by source id. Do not grant sources a job does not need.
- When a validator requests changes or an agent asks something, think about where the problem really is.
  Answer from what is already known when you can. Otherwise rewind to the job that has to change — the coder,
  or further back to a researcher with a precise new question — and everything after it runs again; rearrange
  what follows when the shape of the plan was wrong (update, remove or add jobs). Rewinding keeps every job's
  notes, so say exactly what must change.
- A running agent that asks a question is waiting for your answer: answer it, or rewind/redirect it.
- You may message or stop running agents when what they do is no longer needed or must change.
- Ask a human only for what no agent and no data can tell you.
- depends_on must form a DAG. Loops happen by rewinding, never with cycles.
- Each job prompt is the complete brief for its agent: goal, scope, inputs it will receive, what to output.
- Keep it lean: the fewest jobs that do the work well. Finish as soon as the request is satisfied.
"""

_ACTIONS = """Actions (a decision is a list of them, applied in order):
- {"kind": "add_jobs", "jobs": [{"id": "<slug>", "category": "...", "agent_id": "<from roster>", "title": "...",
   "prompt": "<full brief>", "depends_on": ["<job id>"], "owns": ["..."], "datasources": ["<source id>"]}]}
- {"kind": "update_job", "job_id": "...", "changes": {"prompt": "...", "agent_id": "...", "depends_on": [...],
   "owns": [...], "datasources": [...], "title": "..."}}            (only for a job that is not running)
- {"kind": "rewind", "job_ids": ["..."], "text": "<what must change and why>"}
- {"kind": "remove", "job_ids": ["..."], "text": "<why>"}
- {"kind": "answer", "job_id": "...", "text": "<the answer>"}
- {"kind": "message", "job_id": "...", "text": "...", "restart": false}
- {"kind": "ask_human", "job_id": "<optional>", "text": "<the question>"}
- {"kind": "finish", "text": "<summary of the result>"}
- {"kind": "fail", "text": "<why the request cannot be done>"}

Answer with ONLY a JSON object in a ```json fence:
{"summary": "<one paragraph: what you decided and why>", "actions": [ ... ]}"""

_SITUATIONS = {
    "start": "A new request. Plan it, or start by gathering information.",
    "phase_done": "Nothing is running or waiting. Plan the next phase from the results, or finish.",
    "change_request": "A validator failed its check.",
    "question": "An agent stopped and asks questions; its job waits for you.",
    "live_question": "A running agent asks a question and is waiting for the answer.",
    "needs_help": "A job asks for information or help; it waits for you.",
    "failure": "A job keeps failing; it waits for you.",
    "rejected": "A human rejected your last decision. Decide again, addressing their reason.",
    "human_answer": "A human answered your question.",
}


def _job_line(job: Job) -> str:
    head = (
        f"- {job.id} [{job.category}] agent={job.agent_id} status={job.status} attempts={len(job.attempts)}"
        + (f" depends_on={job.depends_on}" if job.depends_on else "")
        + (f" datasources={job.datasources}" if job.datasources else "")
        + (f" owns={job.owns}" if job.owns else "")
    )
    parts = [head, f"  brief: {_truncate(job.prompt, 600)}"]
    if job.output is not None:
        parts.append(f"  output: {_truncate(job.output, 1500)}")
    last = job.last
    if last is not None and last.status in ("failed", "needs_input"):
        parts.append(f"  last attempt {last.status}: {_truncate(last.error or last.questions, 600)}")
    for note in job.notes[-3:]:
        parts.append(f"  note ({note.source}): {_truncate(note.text, 400)}")
    return "\n".join(parts)


def render_view(view: MetaView) -> str:
    config = view.config
    jobs = [j for j in view.dag.jobs if j.status != "removed"]
    dag = "\n".join(_job_line(j) for j in jobs) if jobs else "(no jobs yet)"
    removed = [j.id for j in view.dag.jobs if j.status == "removed"]
    slots = ", ".join(f"{c}: {config.jobs[c].min}-{config.jobs[c].max}" for c in JOB_CATEGORIES)
    context = []
    if config.repo is not None:
        context.append(
            f"Repository: {config.repo.url} (base branch {config.repo.base_branch}; "
            f"verify: {'; '.join(config.repo.verify) or 'none configured'})"
        )
    if config.shared_volume is not None:
        context.append(f"All jobs share a volume at {config.shared_volume.mount_point}.")
    if config.dispatcher_instructions.strip():
        context.append("Author's instructions:\n" + config.dispatcher_instructions.strip())
    situations = "\n".join(
        f"- {s.kind}{' (job ' + s.job_id + ')' if s.job_id else ''}: {_SITUATIONS[s.kind]}"
        + (f"\n  {_truncate(s.detail, 3000)}" if s.detail else "")
        for s in view.situations
    )
    limits = config.limits
    return (
        f"Request:\n{view.request}\n\n"
        f"Roster:\n{view.roster}\n\n"
        f"Data sources you may grant:\n{describe_datasources(config)}\n\n"
        f"Job slots (min-max): {slots}. Max total jobs: {limits.max_total_jobs}. "
        f"Max parallel jobs: {limits.max_parallel}. "
        f"Rearrangements used: {view.dag.rearrangements} of {limits.max_rearrangements}.\n"
        + ("\n".join(context) + "\n" if context else "")
        + f"\nJob DAG:\n{dag}\n"
        + (f"Removed jobs (ids may not be reused): {removed}\n" if removed else "")
        + f"\nDecide about:\n{situations}\n\n{_ACTIONS}"
    )


class LlmMetaAgent:
    """The meta-agent as a meta-LLM: one JSON answer per decision, repaired once if invalid."""

    def __init__(self, llm_call: LlmCall) -> None:
        self._llm_call = llm_call
        self.usage: dict[str, int] = {}

    async def _ask(self, user: str) -> str:
        text, usage = await self._llm_call(_SYSTEM, user)
        for k, v in (usage or {}).items():
            self.usage[k] = self.usage.get(k, 0) + int(v or 0)
        return text

    async def decide(self, view: MetaView, validate: Validate) -> MetaDecision:
        user = render_view(view)
        text = await self._ask(user)
        for attempt in range(2):
            try:
                decision = MetaDecision.model_validate(_extract_json(text))
                errors = validate(decision)
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                errors = [f"answer is not valid JSON for the schema: {exc}"]
            if not errors:
                return decision
            if attempt == 1:
                raise DispatcherError("meta-agent answer still invalid after repair: " + "; ".join(errors))
            logger.info("meta-agent answer invalid, repairing: %s", errors)
            text = await self._ask(
                user
                + "\n\nYour previous answer:\n"
                + _truncate(text, 8000)
                + "\n\nIt was rejected for these reasons:\n- "
                + "\n- ".join(errors)
                + "\n\nAnswer again, fixing every problem."
            )
        raise DispatcherError("unreachable")
