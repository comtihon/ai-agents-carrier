"""Persistent record of one privileged data-source operation put to a human.

A workflow that reaches a destructive data-source operation (a DELETE, or an
operation explicitly flagged ``destructive``) does not run it. It opens an
*approval case* — a row naming the data source, the operation, the resolved
endpoint, the caller-supplied inputs and, above all, **how many rows the call
is about to affect** — and waits for a person to say yes.

The cases are kept after the decision, because the history is the feature: the
meta-LLM reads a workflow's own past decisions on the same operation to
recommend one, and a long enough unbroken streak of identical human decisions
is what lets it decide alone.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

ApprovalStatus = Literal["pending", "approved", "rejected", "expired", "cancelled"]
# "mcp" is the management MCP server's own decision: an agent or an operator
# approving through a tool call rather than the UI, Slack or the REST API. It
# was missing, so every approve_run / reject_run through that server raised a
# pydantic literal_error and the gated call never ran -- the run reported
# "approved; resuming" and then failed with a validation error instead of
# writing. Found by an end-to-end test whose sheet stayed empty.
DecisionSource = Literal["ui", "slack", "meta_llm", "timeout", "api", "mcp"]
CallSurface = Literal["workflow", "mcp", "try_run"]


def history_key_for(workflow_id: str, datasource_id: str, operation: str) -> str:
    """The bucket a case's history is counted in.

    Per workflow, not per data source: a delete operation that a person has
    waved through ten times in one workflow has earned nothing in a workflow
    written by somebody else, which is exactly the accident this feature exists
    to catch.
    """
    return f"{workflow_id or '-'}|{datasource_id}|{operation}"


class MetaLlmVerdict(BaseModel):
    """What the meta-LLM said about a case, recorded next to the human answer.

    ``autonomous`` marks the verdict that *was* the decision — the streak was
    long enough that no human was asked. Everything else is advisory: it rides
    along on the Slack message and the approval panel, and the human still
    decides.
    """

    decision: Literal["approve", "reject", "abstain"] = "abstain"
    reason: str = ""
    confidence: float | None = None
    model: str = ""
    autonomous: bool = False
    # How many prior decided cases the verdict was formed from.
    history_size: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ApprovalCase(BaseModel):
    """One destructive data-source call awaiting (or carrying) a decision."""

    id: str
    status: ApprovalStatus = "pending"

    # ── What is about to happen ───────────────────────────────────────────
    workflow_id: str = ""
    workflow_name: str = ""
    run_id: str = ""
    step_id: str | None = None
    agent_id: str = ""
    surface: CallSurface = "workflow"

    datasource_id: str = ""
    datasource_name: str = ""
    operation: str = ""
    method: str = ""
    # Rendered target(s): the base URL plus path for HTTP, the source id for
    # GraphQL. First entry only — ``targets`` carries the rest.
    endpoint: str = ""
    targets: list[str] = Field(default_factory=list)
    # Caller-supplied operation inputs, exactly as passed.
    params: dict[str, Any] = Field(default_factory=dict)
    # The number this whole feature turns on.
    affected_rows: int = 0
    # A short, human-readable sample of what the operation acts on: the rows
    # being removed for a delete, the before/after of each cell for a write.
    affected_sample: list[Any] = Field(default_factory=list)
    # WHICH rows are affected, stated without their contents ("row 7").
    # The Slack message renders this instead of `affected_sample`, so a
    # channel never carries data values; the authenticated surfaces (the data
    # source editor, the management MCP) keep showing the sample itself.
    affected_rows_label: str = ""
    # Which of those two this is. Decides the wording an approver reads and the
    # question the meta-LLM is asked; "delete" keeps the behaviour every
    # existing case had before writes joined the gate.
    change_kind: Literal["delete", "write", "other"] = "delete"
    # Label -> value context shown beside the operation and given to the
    # meta-LLM: which document a write targets, and whether a model wrote the
    # code that produced its values.
    details: dict[str, str] = Field(default_factory=dict)

    history_key: str = ""

    # ── The decision ──────────────────────────────────────────────────────
    meta_llm: MetaLlmVerdict | None = None
    decision_source: DecisionSource | None = None
    decided_by_name: str = ""
    decided_by_id: str = ""
    decided_at: datetime | None = None
    reason: str = ""
    # Set when the meta-LLM decided alone: until this instant a person can still
    # cancel the call from Slack or the UI.
    veto_deadline: datetime | None = None
    vetoed_by: str = ""

    # ── Where it was announced ────────────────────────────────────────────
    slack_channel: str = ""
    slack_ts: str = ""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    @property
    def approved(self) -> bool:
        return self.status == "approved"

    @property
    def decided(self) -> bool:
        return self.status in ("approved", "rejected", "expired", "cancelled")

    def summary_line(self) -> str:
        """One line naming the operation and its blast radius."""
        rows = "1 row" if self.affected_rows == 1 else f"{self.affected_rows} rows"
        return (
            f"{self.datasource_name or self.datasource_id}.{self.operation} "
            f"[{self.method}] — {rows}"
        )
