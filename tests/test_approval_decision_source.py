import pytest
from app.api.mcp.management_server import APPROVER_SOURCE
from app.domain.models.approval_case import ApprovalCase, DecisionSource
import typing

def test_every_approver_source_the_code_uses_is_a_valid_decision_source():
    """The management MCP records APPROVER_SOURCE as the decision source.

    It was "mcp" while DecisionSource allowed only ui/slack/meta_llm/timeout/
    api, so every approve_run through that server raised a pydantic
    literal_error *after* reporting "approved; resuming" -- the gated data
    source call never ran. An end-to-end test caught it as a silently empty
    spreadsheet.
    """
    allowed = set(typing.get_args(DecisionSource))
    assert APPROVER_SOURCE in allowed, (
        f"management_server records approver_source={APPROVER_SOURCE!r}, "
        f"which ApprovalCase.decision_source rejects (allowed: {sorted(allowed)})"
    )

def test_an_approval_case_accepts_the_mcp_decision_source():
    case = ApprovalCase(
        id="c1", source_id="google-sheets", operation="append_values",
        method="POST", workflow_id="w", run_id="r", step_id="s",
    )
    case.decision_source = "mcp"
    ApprovalCase.model_validate(case.model_dump())
