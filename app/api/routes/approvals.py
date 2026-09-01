"""REST API for data-source deletion approvals.

Every destructive data-source call a run makes leaves a row here, decided or
not. Three things read this API: the copilot_ui approvals panel (the queue and
the history), the approval panel on a paused run (one case, by id), and
whatever decides — a person clicking Approve, or someone cancelling an
autonomous decision inside its veto window.

Deciding a case that belongs to a *workflow* run also resumes that run. The two
have to move together: a case marked approved while its run stays parked at
``waiting_approval`` is a deletion that everybody believes happened and never
did.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.approval_case import ApprovalCase, history_key_for

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/approvals", tags=["approvals"])


class DecisionRequest(BaseModel):
    approved: bool
    reason: str | None = None
    # Who decided, when the caller knows better than the token does (the Slack
    # bridge fills these in; the UI leaves them to the authenticated principal).
    decided_by_name: str | None = None
    decided_by_id: str | None = None


class VetoRequest(BaseModel):
    by: str | None = None


def _require_backend(container: ApplicationContainer) -> None:
    if container.approval_backend is None or container.approval_service is None:
        raise HTTPException(status_code=501, detail="Approvals backend not configured")


def _principal(request: Request) -> tuple[str, str]:
    """``(display name, subject)`` for the caller, or empty strings.

    Read off the validated token the auth middleware parked on the request,
    not off the request body: attribution is the whole point of the decision
    history, and a name a client could write for itself attributes nothing.
    Empty when OAuth is disabled — a deployment with no identities has no name
    to record, and the case still says who *not* to credit.
    """
    claims = getattr(request.state, "jwt_claims", None) or {}
    if not isinstance(claims, dict):
        return "", ""
    name = (
        claims.get("name")
        or claims.get("preferred_username")
        or claims.get("email")
        or claims.get("sub")
        or ""
    )
    return str(name), str(claims.get("sub") or "")


def _case_json(case: ApprovalCase) -> dict:
    data = case.model_dump(mode="json")
    data["summary"] = case.summary_line()
    return data


# ─── Collection ───────────────────────────────────────────────────────────────

@router.get("")
async def list_approvals(
    status: Literal["pending", "approved", "rejected", "expired", "cancelled"] | None = None,
    workflow_id: str | None = None,
    datasource_id: str | None = None,
    run_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    container: ApplicationContainer = Depends(get_container),
):
    """The approval queue and its history, newest first."""
    _require_backend(container)
    assert container.approval_backend is not None
    limit = max(1, min(limit, 200))
    cases = await container.approval_backend.list(
        status=status,
        workflow_id=workflow_id,
        datasource_id=datasource_id,
        run_id=run_id,
        limit=limit,
        offset=offset,
    )
    total = await container.approval_backend.count(
        status=status,
        workflow_id=workflow_id,
        datasource_id=datasource_id,
        run_id=run_id,
    )
    return {
        "items": [_case_json(c) for c in cases],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/pending/count")
async def pending_count(container: ApplicationContainer = Depends(get_container)):
    """Badge count for the dock rail. Cheap enough to poll."""
    if container.approval_backend is None:
        return {"count": 0}
    return {"count": await container.approval_backend.count(status="pending")}


@router.get("/history")
async def decision_history(
    workflow_id: str,
    datasource_id: str,
    operation: str,
    limit: int = 25,
    container: ApplicationContainer = Depends(get_container),
):
    """Past decisions in one bucket — what the meta-LLM reads, shown to people too.

    The streak is computed here rather than in the client so the panel and the
    service can never disagree about whether the next case will be decided
    autonomously.
    """
    _require_backend(container)
    assert container.approval_backend is not None
    key = history_key_for(workflow_id, datasource_id, operation)
    cases = await container.approval_backend.history(key, limit=max(1, min(limit, 100)))

    streak_value: str | None = None
    streak = 0
    for case in cases:
        if case.decision_source == "meta_llm" or case.status not in ("approved", "rejected"):
            break
        if streak_value is None:
            streak_value = case.status
        elif case.status != streak_value:
            break
        streak += 1

    threshold = int(getattr(container.settings, "approval_auto_decide_threshold", 10) or 10)
    return {
        "history_key": key,
        "items": [_case_json(c) for c in cases],
        "streak": streak,
        "streak_decision": streak_value,
        "threshold": threshold,
        "autonomous_next": streak >= threshold,
    }


# ─── One case ─────────────────────────────────────────────────────────────────

@router.get("/{case_id}")
async def get_approval(
    case_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    _require_backend(container)
    assert container.approval_backend is not None
    case = await container.approval_backend.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Approval case not found")
    return _case_json(case)


@router.post("/{case_id}/decide")
async def decide_approval(
    case_id: str,
    body: DecisionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    container: ApplicationContainer = Depends(get_container),
):
    """Approve or reject one case, and resume the run waiting on it.

    A case opened by a workflow step is only half the state: the run is parked
    at ``waiting_approval`` and the node is blocked inside ``interrupt()``. So
    the resume is scheduled here rather than left to the caller — and it is
    scheduled *after* the response, because resuming streams the rest of the
    graph and can take as long as the next step does.
    """
    _require_backend(container)
    assert container.approval_backend is not None
    case = await container.approval_backend.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Approval case not found")
    if case.decided:
        raise HTTPException(
            status_code=409, detail=f"Approval case is already {case.status}"
        )

    name, subject = _principal(request)
    name = body.decided_by_name or name
    subject = body.decided_by_id or subject

    if case.surface == "workflow" and case.run_id:
        # The run's own resume path records the decision: the paused node calls
        # ApprovalService.decide() with the approver carried in the resume
        # payload. Writing it here as well would race that, and the loser would
        # overwrite a correct decision with a duplicate one.
        await _resume_workflow_run(
            container, case, background_tasks,
            approved=body.approved, reason=body.reason or "",
            name=name, subject=subject,
        )
        return {**_case_json(case), "status": "approved" if body.approved else "rejected"}

    decided = await container.approval_service.decide(
        case_id,
        approved=body.approved,
        source="ui",
        decided_by_name=name,
        decided_by_id=subject,
        reason=body.reason or "",
    )
    if decided is None:
        raise HTTPException(status_code=409, detail="Approval case is already decided")
    return _case_json(decided)


@router.post("/{case_id}/veto")
async def veto_approval(
    case_id: str,
    request: Request,
    body: VetoRequest | None = None,
    container: ApplicationContainer = Depends(get_container),
):
    """Cancel an autonomous decision before its veto window closes."""
    _require_backend(container)
    name, _ = _principal(request)
    by = (body.by if body else None) or name or "a reviewer"
    case = await container.approval_service.veto(case_id, by=by)
    if case is None:
        raise HTTPException(
            status_code=409,
            detail="Case is not cancellable — no veto window, or it has closed",
        )
    return _case_json(case)


async def _resume_workflow_run(
    container: ApplicationContainer,
    case: ApprovalCase,
    background_tasks: BackgroundTasks,
    *,
    approved: bool,
    reason: str,
    name: str,
    subject: str,
) -> None:
    """Drive the paused run through the same transition the run routes use."""
    from app.application.run_control import (
        RunControlError,
        _resume_approved,
        _resume_rejected,
        approve_run,
        reject_run,
    )

    try:
        if approved:
            run, runner = await approve_run(container, case.run_id)
            background_tasks.add_task(
                _resume_approved, runner, run, container, None,
                name, subject, "ui",
            )
        else:
            run, runner = await reject_run(container, case.run_id)
            background_tasks.add_task(
                _resume_rejected, runner, run, container, reason or None,
                name, subject, "ui",
            )
    except RunControlError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
