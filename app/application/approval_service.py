"""The privilege gate in front of destructive data-source operations.

A workflow author writes a delete step by hand; a workflow *run* executes it
against real data, sometimes with a row count nobody predicted. This service
stands between the two. When a run reaches an operation that deletes, the call
is held, an :class:`ApprovalCase` is written naming the blast radius, and a
person is asked — in Slack when a channel is configured, and always in
copilot_ui.

Three behaviours build on that one record:

* **Recommendation.** Once a workflow has decided this operation before, and
  the workflow has the meta-LLM enabled, the meta-LLM reads that history and
  says what it would do. The recommendation rides on the Slack message and the
  approval panel. It does not decide.
* **Autonomy.** When the last ``approval_auto_decide_threshold`` decided cases
  in the bucket all went the same way, the meta-LLM makes the call itself,
  because at that point asking again is asking a question whose answer is on
  record ten times over.
* **Veto.** An autonomous decision is announced with a countdown, not
  silently. Until the deadline passes anyone can cancel it, so "the model
  decided" never means "nobody could stop it".

Both decisions — the human one and the meta-LLM's — are stored on the same
case, which is what makes the next recommendation better than the last.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.domain.models.approval_case import (
    ApprovalCase,
    CallSurface,
    DecisionSource,
    MetaLlmVerdict,
    history_key_for,
)

logger = logging.getLogger(__name__)

# How many prior cases the meta-LLM is shown. Enough to see the streak that
# grants autonomy, small enough to stay a cheap prompt.
_HISTORY_WINDOW = 25


class ApprovalService:
    """Opens, decides and remembers data-source approval cases."""

    def __init__(
        self,
        backend: Any,
        settings: Any,
        *,
        workflow_backend: Any = None,
        run_repository: Any = None,
    ) -> None:
        self._backend = backend
        self._settings = settings
        self._workflow_backend = workflow_backend
        self._run_repository = run_repository

    # ------------------------------------------------------------------
    # Opening a case
    # ------------------------------------------------------------------

    async def open_case(
        self,
        *,
        source: Any,
        operation: str,
        method: str,
        params: dict[str, Any],
        affected_rows: int,
        targets: list[str],
        sample: list[Any],
        workflow_id: str = "",
        run_id: str = "",
        step_id: str | None = None,
        agent_id: str = "",
        surface: CallSurface = "workflow",
    ) -> ApprovalCase:
        """Write a pending case, consult the meta-LLM, announce it.

        Returns the case in whatever state it ended up in: ``pending`` when a
        person still has to answer, or already ``approved`` / ``rejected`` when
        the streak was long enough for the meta-LLM to answer for them (in
        which case ``veto_deadline`` is set and the caller must wait it out).
        """
        workflow_name = await self._workflow_name(workflow_id)
        case = ApprovalCase(
            id=f"apr_{uuid.uuid4().hex[:16]}",
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            run_id=run_id,
            step_id=step_id,
            agent_id=agent_id,
            surface=surface,
            datasource_id=getattr(source, "id", ""),
            datasource_name=getattr(source, "name", "") or getattr(source, "id", ""),
            operation=operation,
            method=(method or "").upper(),
            endpoint=targets[0] if targets else "",
            targets=targets,
            params=dict(params or {}),
            affected_rows=affected_rows,
            affected_sample=[str(s) for s in sample],
            history_key=history_key_for(workflow_id, getattr(source, "id", ""), operation),
        )

        history = await self._backend.history(case.history_key, limit=_HISTORY_WINDOW)
        verdict = await self._consult_meta_llm(case, history)
        case.meta_llm = verdict

        if verdict is not None and verdict.autonomous:
            probation = await self._tier2_write_probation(source, operation)
            if probation is not None:
                # The streak says "a person has waved this through often enough".
                # For generated code that is not yet the same claim: a golden
                # fixture over five sample rows is not evidence over five
                # hundred real ones, and the rows the model never saw are
                # exactly where a transform goes wrong. So the first N runs of a
                # tier-2 write go to a person regardless of any trusted setting,
                # and the meta-LLM's opinion rides along as advice instead of
                # becoming the decision.
                verdict.autonomous = False
                verdict.reason = (
                    f"{verdict.reason}\n\n[held for review: {probation}]"
                ).strip()
                logger.info(
                    "approval %s: tier-2 write '%s' still on probation (%s) — "
                    "autonomous decision withheld",
                    case.id, operation, probation,
                )
            else:
                self._apply_autonomous_decision(case, verdict)

        await self._backend.create(case)
        await self._announce(case)
        return case

    async def record_confirmed(
        self,
        *,
        source: Any,
        operation: str,
        method: str,
        params: dict[str, Any],
        affected_rows: int,
        targets: list[str],
        sample: list[Any],
        decided_by_name: str = "",
        decided_by_id: str = "",
        surface: CallSurface = "try_run",
    ) -> ApprovalCase:
        """Record a deletion somebody confirmed to their own face, already approved.

        The try-run button in the data-source editor is a person at a keyboard
        who has just been shown the row count and clicked through it. There is
        nobody else to ask, and blocking the editor on a Slack round trip would
        make testing a delete endpoint impossible — so the confirmation *is*
        the decision, and this writes it down.

        No meta-LLM: the answer is already given, and a model that disagreed
        could only either be ignored or overrule a human who is looking
        straight at the consequences. The case is excluded from the decision
        history (see ``ApprovalCaseBackend.history``) so self-approvals cannot
        assemble the streak that grants autonomy.
        """
        now = datetime.now(timezone.utc)
        case = ApprovalCase(
            id=f"apr_{uuid.uuid4().hex[:16]}",
            status="approved",
            surface=surface,
            datasource_id=getattr(source, "id", ""),
            datasource_name=getattr(source, "name", "") or getattr(source, "id", ""),
            operation=operation,
            method=(method or "").upper(),
            endpoint=targets[0] if targets else "",
            targets=targets,
            params=dict(params or {}),
            affected_rows=affected_rows,
            affected_sample=[str(s) for s in sample],
            history_key=history_key_for("", getattr(source, "id", ""), operation),
            decision_source="ui",
            decided_by_name=decided_by_name,
            decided_by_id=decided_by_id,
            decided_at=now,
            reason="Confirmed in the data source editor's try run",
        )
        await self._backend.create(case)
        await self._announce(case)
        logger.info(
            "approval %s: try-run deletion confirmed by %s (%d rows on %s.%s)",
            case.id, decided_by_name or "unknown", affected_rows,
            case.datasource_id, operation,
        )
        return case

    async def find_open_case(
        self, run_id: str, step_id: str | None = None
    ) -> ApprovalCase | None:
        """The case a resuming step already opened, if it has one.

        A LangGraph node re-runs from the top on every resume, so the
        ``data_source`` step calls this first: without it a paused delete would
        write a fresh case — and re-read the upstream list — each time somebody
        loaded the run.
        """
        if not run_id:
            return None
        try:
            return await self._backend.find_pending_for_run(run_id, step_id)
        except Exception:
            logger.warning("approvals: pending-case lookup failed", exc_info=True)
            return None

    def _apply_autonomous_decision(self, case: ApprovalCase, verdict: MetaLlmVerdict) -> None:
        now = datetime.now(timezone.utc)
        case.status = "approved" if verdict.decision == "approve" else "rejected"
        case.decision_source = "meta_llm"
        case.decided_by_name = f"meta-LLM ({verdict.model})" if verdict.model else "meta-LLM"
        case.decided_at = now
        case.reason = verdict.reason
        window = max(0, int(getattr(self._settings, "approval_veto_window_seconds", 60) or 0))
        case.veto_deadline = now + timedelta(seconds=window) if window else None

    async def _workflow_name(self, workflow_id: str) -> str:
        if not workflow_id or self._workflow_backend is None:
            return ""
        try:
            defn = await self._workflow_backend.get(workflow_id)
        except Exception:
            return ""
        return getattr(defn, "name", "") if defn is not None else ""

    async def _meta_llm_enabled(self, workflow_id: str) -> bool:
        """Whether *this workflow* may use the meta-LLM.

        Off by default when there is no workflow to read the flag from — the
        MCP surface reaches here with only a run id, and a recommendation
        attributed to a workflow whose owner turned the meta-LLM off would be
        exactly the surprise the flag exists to prevent.
        """
        if not workflow_id or self._workflow_backend is None:
            return False
        try:
            defn = await self._workflow_backend.get(workflow_id)
        except Exception:
            return False
        return bool(getattr(defn, "use_meta_llm", False)) if defn is not None else False

    # ------------------------------------------------------------------
    # The meta-LLM
    # ------------------------------------------------------------------

    def _streak(self, history: list[ApprovalCase]) -> tuple[str | None, int]:
        """The unbroken run of identical decisions at the head of *history*.

        Only decisions a *person* made count. A streak grown out of the
        meta-LLM's own autonomous calls would be the model reading its own
        output back as evidence, and would ratchet a single early mistake into
        a standing policy.
        """
        streak_value: str | None = None
        length = 0
        for case in history:
            if case.decision_source == "meta_llm":
                break
            if case.status not in ("approved", "rejected"):
                break
            if streak_value is None:
                streak_value = case.status
            elif case.status != streak_value:
                break
            length += 1
        return streak_value, length

    async def _tier2_write_probation(self, source: Any, operation: str) -> str | None:
        """Why this tier-2 write may not be decided autonomously yet, or None.

        Returns a short explanation while the binding is still inside its
        probation window, so the reason can be shown on the case rather than
        the withholding looking arbitrary.

        Only *human* approvals count, and they are counted per data source and
        per operation (not per workflow, as the meta-LLM streak is): probation
        is a property of the generated code, and the same code writing to the
        same sheet has the same risk whichever workflow calls it.
        """
        binding = None
        getter = getattr(source, "get_binding", None)
        if callable(getter):
            binding = getter(operation)
        compute = getattr(binding, "compute", None) if binding is not None else None
        if compute is None or getattr(binding, "operation", "") != "write":
            return None

        required = int(getattr(self._settings, "sheets_compute_write_probation_runs", 5) or 0)
        if required <= 0:
            return None

        try:
            approvals = await self._backend.list(
                status="approved",
                datasource_id=getattr(source, "id", ""),
                operation=operation,
                limit=max(required * 4, 50),
            )
        except TypeError:
            # A backend that predates the `operation` filter: fall back to
            # filtering here rather than skipping probation, which would be the
            # unsafe direction to fail in.
            approvals = [
                c for c in await self._backend.list(
                    status="approved",
                    datasource_id=getattr(source, "id", ""),
                    limit=max(required * 8, 100),
                )
                if c.operation == operation
            ]
        human = sum(1 for c in approvals if c.decision_source != "meta_llm")
        if human >= required:
            return None
        return (
            f"generated-code write, {human} of {required} human approvals so far"
        )

    async def _consult_meta_llm(
        self, case: ApprovalCase, history: list[ApprovalCase]
    ) -> MetaLlmVerdict | None:
        """Ask the meta-LLM what it would do, when there is anything to go on.

        Returns ``None`` when the meta-LLM is not in play at all: the workflow
        turned it off, or this bucket has no decided case yet — with an empty
        history there is nothing to recommend *from*, and a recommendation
        pulled out of the air is worse than none on a message whose whole job
        is to make someone think.
        """
        if not history:
            return None
        if not await self._meta_llm_enabled(case.workflow_id):
            return None

        threshold = int(getattr(self._settings, "approval_auto_decide_threshold", 10) or 10)
        streak_value, streak_len = self._streak(history)
        autonomous = streak_value is not None and streak_len >= threshold

        try:
            verdict = await self._ask_meta_llm(case, history, autonomous=autonomous)
        except Exception as exc:
            # Belt and braces around the provider call: a meta-LLM that cannot
            # answer must cost the case its recommendation, never its
            # existence. The alternative is a deletion that fails to be
            # *reviewed* because the advisor is down.
            logger.warning("approval %s: meta-LLM consult raised: %s", case.id, exc)
            return None
        if verdict is None:
            return None
        verdict.history_size = len(history)

        # Autonomy is granted by the streak, and only for the answer the streak
        # actually contains. A model that disagrees with ten consecutive human
        # decisions has found something the streak did not, so the case goes
        # back to a person rather than either answer winning by default.
        expected = "approve" if streak_value == "approved" else "reject"
        verdict.autonomous = bool(autonomous and verdict.decision == expected)
        if autonomous and not verdict.autonomous:
            logger.info(
                "approval %s: meta-LLM (%s) diverged from a %d-case %s streak — asking a human",
                case.id, verdict.decision, streak_len, streak_value,
            )
        return verdict

    async def _ask_meta_llm(
        self, case: ApprovalCase, history: list[ApprovalCase], *, autonomous: bool
    ) -> MetaLlmVerdict | None:
        try:
            from langchain_core.messages import HumanMessage

            from app.core.container import build_llm_native

            settings = self._settings
            provider = settings.meta_llm_provider or settings.llm_provider
            model = settings.meta_llm_model
            llm = build_llm_native(provider, model, settings, max_tokens=400)

            response = await llm.ainvoke([
                HumanMessage(content=_build_prompt(case, history, autonomous=autonomous))
            ])
            text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            decision, reason = _parse_verdict(text)
            return MetaLlmVerdict(
                decision=decision, reason=reason, model=str(model or provider or "")
            )
        except Exception as exc:
            # Never blocking: a meta-LLM that is down means the case is decided
            # by a person, which is the behaviour without this feature at all.
            logger.warning("approval %s: meta-LLM consult failed: %s", case.id, exc)
            return None

    # ------------------------------------------------------------------
    # Deciding
    # ------------------------------------------------------------------

    async def decide(
        self,
        case_id: str,
        *,
        approved: bool,
        source: DecisionSource = "ui",
        decided_by_name: str = "",
        decided_by_id: str = "",
        reason: str = "",
    ) -> ApprovalCase | None:
        """Record a human answer. Returns None when the case was already closed."""
        case = await self._backend.claim_for_decision(case_id)
        if case is None:
            return None
        case.status = "approved" if approved else "rejected"
        case.decision_source = source
        case.decided_by_name = decided_by_name
        case.decided_by_id = decided_by_id
        case.decided_at = datetime.now(timezone.utc)
        case.reason = reason
        await self._backend.update(case)
        await self._announce_outcome(case)
        logger.info(
            "approval %s: %s by %s via %s (%d rows on %s.%s)",
            case.id, case.status, decided_by_name or "unknown", source,
            case.affected_rows, case.datasource_id, case.operation,
        )
        return case

    async def veto(self, case_id: str, *, by: str = "") -> ApprovalCase | None:
        """Cancel an autonomous decision inside its veto window.

        A veto is a rejection with a name on it, not a return to pending: the
        run is not going to sit waiting for a second opinion after somebody has
        already said stop.
        """
        case = await self._backend.get(case_id)
        if case is None:
            return None
        if case.veto_deadline is None:
            return None
        if datetime.now(timezone.utc) > case.veto_deadline:
            return None
        case.status = "cancelled"
        case.vetoed_by = by
        case.reason = f"Cancelled by {by or 'a reviewer'} during the veto window"
        case.decided_at = datetime.now(timezone.utc)
        await self._backend.update(case)
        await self._announce_outcome(case)
        logger.info("approval %s: vetoed by %s", case.id, by or "unknown")
        return case

    async def expire(self, case_id: str, reason: str = "") -> ApprovalCase | None:
        """Close a case nobody answered in time."""
        case = await self._backend.claim_for_decision(case_id)
        if case is None:
            return None
        case.status = "expired"
        case.decision_source = "timeout"
        case.decided_at = datetime.now(timezone.utc)
        case.reason = reason or "No decision within the approval timeout"
        await self._backend.update(case)
        return case

    # ------------------------------------------------------------------
    # Waiting
    # ------------------------------------------------------------------

    async def wait_out_veto(self, case: ApprovalCase) -> ApprovalCase:
        """Sleep until an autonomous decision's veto window closes.

        Polls rather than sleeping the whole window in one go, so a veto that
        lands in the first second is acted on in the first second.
        """
        if case.veto_deadline is None:
            return case
        interval = float(getattr(self._settings, "approval_poll_interval_seconds", 3) or 3)
        while True:
            now = datetime.now(timezone.utc)
            if now >= case.veto_deadline:
                break
            await asyncio.sleep(min(interval, (case.veto_deadline - now).total_seconds()))
            refreshed = await self._backend.get(case.id)
            if refreshed is not None:
                case = refreshed
                if case.status == "cancelled":
                    return case
        return await self._backend.get(case.id) or case

    async def wait_for_decision(
        self, case_id: str, *, timeout_seconds: float | None = None
    ) -> ApprovalCase | None:
        """Block until a case is decided, or the timeout expires it.

        This is how the surfaces that cannot suspend — an agent's MCP tool call
        is a plain HTTP request, with no checkpoint to interrupt into — wait for
        a person. A workflow ``data_source`` step does not use it; it raises a
        LangGraph interrupt instead, so the run leaves memory while it waits.
        """
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(self._settings, "approval_wait_timeout_seconds", 3600.0)
        )
        interval = float(getattr(self._settings, "approval_poll_interval_seconds", 3) or 3)
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        while True:
            case = await self._backend.get(case_id)
            if case is None:
                return None
            if case.decided and case.veto_deadline is None:
                return case
            if case.decided and case.veto_deadline is not None:
                return await self.wait_out_veto(case)
            if datetime.now(timezone.utc) >= deadline:
                return await self.expire(
                    case_id, f"No decision within {int(timeout)}s"
                )
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Slack
    # ------------------------------------------------------------------

    async def _announce(self, case: ApprovalCase) -> None:
        """Post the case to the configured Slack channel, if there is one.

        No channel configured is not an error: copilot_ui shows every pending
        case regardless, so a deployment without Slack still gets the gate —
        it just gets it in one place instead of two.
        """
        token = getattr(self._settings, "slack_bot_token", "")
        channel = getattr(self._settings, "slack_approvals_channel", "")
        if not (token and channel):
            return
        from app.infrastructure.notifications.webhook_notifier import (
            post_slack_approval_case,
        )
        if case.status == "pending":
            mode = "veto" if case.veto_deadline is not None else "request"
        elif case.veto_deadline is not None:
            mode = "veto"
        else:
            # Already decided when it was announced (a try-run confirmation):
            # a message offering Approve / Reject would be offering an action
            # that no longer exists.
            mode = "notice"
        raw = await post_slack_approval_case(token, channel, case, mode=mode)
        if raw and raw.get("ok"):
            case.slack_ts = str(raw.get("ts") or "")
            case.slack_channel = str(raw.get("channel") or channel)
            await self._backend.update(case)

    async def _announce_outcome(self, case: ApprovalCase) -> None:
        token = getattr(self._settings, "slack_bot_token", "")
        if not (token and case.slack_channel and case.slack_ts):
            return
        from app.infrastructure.notifications.webhook_notifier import (
            post_slack_approval_outcome,
        )
        icon = {"approved": "✅", "rejected": "🚫", "cancelled": "🛑", "expired": "⌛"}.get(
            case.status, "•"
        )
        who = case.vetoed_by or case.decided_by_name or "unknown"
        text = f"{icon} {case.status.capitalize()} by {who}"
        if case.reason:
            text += f" — {case.reason}"
        await post_slack_approval_outcome(token, case.slack_channel, case.slack_ts, text)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _history_lines(history: list[ApprovalCase]) -> str:
    lines = []
    for case in history:
        who = case.decided_by_name or "unknown"
        via = case.decision_source or "?"
        rows = case.affected_rows
        reason = f" — {case.reason}" if case.reason else ""
        lines.append(
            f"- {case.status.upper()} ({rows} rows, by {who} via {via}){reason}"
        )
    return "\n".join(lines)


def _build_prompt(case: ApprovalCase, history: list[ApprovalCase], *, autonomous: bool) -> str:
    import json as _json

    role = (
        "You are deciding, on your own authority, whether this deletion runs. "
        "A long unbroken streak of identical human decisions on this exact "
        "operation has earned you that authority — so follow the streak unless "
        "this case is materially different from the ones in it (a much larger "
        "row count, different inputs, a different target set)."
        if autonomous
        else "You are advising a human reviewer. They decide; you only "
        "recommend, and your recommendation is shown next to the buttons."
    )
    sample = ", ".join(case.affected_sample[:10]) or "(none captured)"
    return (
        f"A workflow is about to run a destructive data-source operation.\n\n"
        f"{role}\n\n"
        f"CURRENT CASE\n"
        f"Workflow: {case.workflow_name or case.workflow_id or '(none)'}\n"
        f"Data source: {case.datasource_name or case.datasource_id}\n"
        f"Operation: {case.operation} [{case.method}]\n"
        f"Endpoint: {case.endpoint}\n"
        f"Rows affected: {case.affected_rows}\n"
        f"Targets (sample): {sample}\n"
        f"Inputs: {_json.dumps(case.params, default=str)[:1500]}\n\n"
        f"PRIOR DECISIONS on this workflow + data source + operation "
        f"(newest first):\n{_history_lines(history)}\n\n"
        "Weigh how closely this case matches the prior ones. A row count far "
        "outside the range of the history is a reason to reject even when every "
        "prior case was approved.\n"
        "Respond with ONLY:\n"
        "DECISION: APPROVE or DECISION: REJECT\n"
        "REASON: <one line>"
    )


def _parse_verdict(text: str) -> tuple[str, str]:
    decision = "abstain"
    reason = ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("DECISION:"):
            raw = line.split(":", 1)[1].strip().upper()
            if raw.startswith("APPROVE"):
                decision = "approve"
            elif raw.startswith("REJECT"):
                decision = "reject"
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return decision, reason
