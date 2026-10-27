"""Runners: machines that dial carrier with acp-web-proxy and offer their ACP agents.

A runner is enrolled by number matching (see app.application.runners): the
machine shows a number, a signed-in user types it into carrier's UI and
approves. The approver owns the runner; its agents become ``AgentDefinition``s
with ``default_runtime: "runner"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.models.ownership import OwnedResource


class AgentLimit(BaseModel):
    """An agent on a runner that ran out of quota, until ``until`` (None: unknown)."""

    until: datetime | None = None
    kind: str = "usage_limit"
    message: str = ""
    updated_at: datetime | None = None


class Runner(OwnedResource):
    id: str
    name: str = ""
    hostname: str = ""
    os: str = ""
    arch: str = ""
    tz: str | None = None
    version: str = ""
    agents: list[str] = Field(default_factory=list)
    default_agent: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    # sha256 of the runner credential's secret; never returned by the API.
    secret_hash: str = ""
    approved_by: str | None = None
    approved_by_name: str | None = None
    created_at: datetime | None = None
    last_seen: datetime | None = None
    connected_at: datetime | None = None
    # Agents (by name on the machine) that hit a usage limit.
    limits: dict[str, AgentLimit] = Field(default_factory=dict)


EnrollmentStatus = Literal["pending", "approved", "denied", "consumed"]


class RunnerEnrollment(BaseModel):
    """A machine asking to become a runner, until approved, denied or expired."""

    id: str
    code_hash: str
    poll_token_hash: str
    status: EnrollmentStatus = "pending"
    hostname: str = ""
    os: str = ""
    arch: str = ""
    tz: str | None = None
    version: str = ""
    agents: list[str] = Field(default_factory=list)
    default_agent: str | None = None
    request_ip: str | None = None
    created_at: datetime
    expires_at: datetime
    # Set on approval: the runner the poll hands out a credential for.
    runner_id: str | None = None
    decided_by: str | None = None
    decided_by_name: str | None = None
