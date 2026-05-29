from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AgentDefinition(BaseModel):
    """Persistent definition of a registered agent.

    Agents are invoked by ``langgraph-agent`` and ``claude-agent`` workflow
    step types.  The ``default_runtime`` field controls how the agent process
    is spawned when no per-step override is provided.

    Fields
    ------
    description:
        Human-readable label shown in the UI.  NOT forwarded to the agent.
    agent_input:
        All runtime configuration sent to the agent on every run.
        Typical keys: ``system_prompt``, ``model``, ``tools``, ``max_tokens``.
        For docker/k8s this populates ``AgentConfig`` fields and ``extra``.
        For the local inline agent this is passed directly to ``run_local_agent``.
    image:
        Docker image (docker runtime only).  The CMD is baked into the image;
        no entrypoint override is needed.
    helm_chart:
        Helm chart reference (k8s runtime only), e.g.
        ``"oci://ghcr.io/org/chart"`` or a local path.
    helm_values:
        Dict of Helm value overrides (k8s runtime only).
    """

    id: str
    name: str = ""
    description: str | None = None          # human label — NOT sent to agent
    default_runtime: Literal["local", "docker", "k8s"] = "local"

    # Sent to the agent on every run:
    agent_input: dict = Field(default_factory=dict)

    # Docker-specific (only relevant when default_runtime == "docker"):
    image: str | None = None
    health_timeout: int = 300  # seconds to wait for /health after container starts

    # K8s-specific (only relevant when default_runtime == "k8s"):
    helm_chart: str | None = None
    helm_values: dict = Field(default_factory=dict)

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def touch(self) -> None:
        from datetime import timezone
        self.updated_at = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = self.updated_at
