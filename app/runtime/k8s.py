from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import httpx

from app.runtime.base import AgentRuntime

if TYPE_CHECKING:
    from app.domain.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)

_AGENT_PORT = 8000
_HEALTH_TIMEOUT = 60.0        # seconds to wait for the Helm release to become healthy
_HEALTH_POLL_INTERVAL = 2.0


class K8sRuntime(AgentRuntime):
    """Spawn the agent via Helm upgrade --install.

    The Helm chart contains the image and all static configuration.
    Only ``agent_def.helm_chart`` (chart reference) and
    ``agent_def.helm_values`` (value overrides) are required.

    Dynamic values injected at deploy time
    ----------------------------------------
    env.AGENT_PORT            — fixed to 8000
    env.BACKEND_CALLBACK_URL  — backend base URL for agent callbacks
    env.RUN_ID                — workflow run identifier

    Service URL convention
    ----------------------
    The release name is ``agent-<agent_id[:20]>-<run_id[:8]>``.
    The chart is expected to create a ClusterIP Service named after the
    release.  The agent URL is therefore::

        http://<release_name>.<namespace>.svc.cluster.local:8000
    """

    def __init__(self, namespace: str = "default") -> None:
        self._namespace = namespace
        # Maps agent_url → release_name
        self._releases: dict[str, str] = {}

    def _release_name(self, agent_def: "AgentDefinition", run_id: str) -> str:
        return f"agent-{agent_def.id[:20]}-{run_id[:8]}"

    async def spawn(
        self,
        agent_def: "AgentDefinition",
        step: dict[str, Any],
        run_id: str,
        callback_base_url: str,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        """Deploy the agent via ``helm upgrade --install`` and return its URL.

        Raises
        ------
        ValueError
            When ``agent_def.helm_chart`` is not set.
        RuntimeError
            When the ``helm`` CLI invocation fails or the agent does not
            become healthy within ``_HEALTH_TIMEOUT`` seconds.
        """
        if not agent_def.helm_chart:
            raise ValueError(
                f"K8sRuntime: agent '{agent_def.id}' has no helm_chart configured. "
                "Set AgentDefinition.helm_chart to the chart OCI ref or local path."
            )

        release_name = self._release_name(agent_def, run_id)
        agent_url = (
            f"http://{release_name}.{self._namespace}.svc.cluster.local:{_AGENT_PORT}"
        )

        # Build --set arguments from helm_values plus runtime overrides.
        set_args: list[str] = []
        for key, value in (agent_def.helm_values or {}).items():
            set_args += ["--set", f"{key}={value}"]
        set_args += ["--set", f"env.AGENT_PORT={_AGENT_PORT}"]
        set_args += ["--set", f"env.BACKEND_CALLBACK_URL={callback_base_url}"]
        set_args += ["--set", f"env.RUN_ID={run_id}"]
        for k, v in (extra_env or {}).items():
            set_args += ["--set-string", f"env.{k}={v}"]

        cmd = [
            "helm", "upgrade", "--install",
            release_name,
            agent_def.helm_chart,
            "--namespace", self._namespace,
            "--wait",          # wait for the rollout to complete
            "--timeout", "120s",
            *set_args,
        ]

        logger.info(
            "K8sRuntime: helm upgrade --install '%s' chart='%s' (run_id=%s)",
            release_name, agent_def.helm_chart, run_id,
        )

        await self._try_oci_registry_login(agent_def.helm_chart)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"K8sRuntime: helm upgrade failed for release '{release_name}':\n"
                f"{stderr.decode()}"
            )

        self._releases[agent_url] = release_name
        logger.info("K8sRuntime: release '%s' deployed at %s", release_name, agent_url)

        # Poll GET /health until the agent service is reachable.
        # Inside the cluster the URL above is directly accessible.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + _HEALTH_TIMEOUT
        async with httpx.AsyncClient() as client:
            while loop.time() < deadline:
                try:
                    resp = await client.get(f"{agent_url}/health", timeout=2.0)
                    if resp.status_code == 200:
                        logger.info(
                            "K8sRuntime: agent at %s is healthy", agent_url
                        )
                        return agent_url
                except Exception:
                    pass
                await asyncio.sleep(_HEALTH_POLL_INTERVAL)

        # Agent did not become healthy — clean up and raise.
        await self.terminate(agent_url)
        raise RuntimeError(
            f"K8sRuntime: agent at {agent_url} did not become healthy "
            f"within {_HEALTH_TIMEOUT}s (run_id={run_id})"
        )

    async def terminate(self, agent_url: str) -> None:
        """Call POST /terminate then uninstall the Helm release."""
        release_name = self._releases.pop(agent_url, None)
        if release_name is None:
            logger.warning(
                "K8sRuntime.terminate: no release found for %s — nothing to uninstall",
                agent_url,
            )
            return

        # Best-effort graceful shutdown via HTTP first.
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{agent_url}/terminate", timeout=3.0)
        except Exception:
            pass

        cmd = [
            "helm", "uninstall", release_name,
            "--namespace", self._namespace,
        ]
        logger.info("K8sRuntime: helm uninstall '%s'", release_name)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                "K8sRuntime: helm uninstall '%s' failed (exit %d):\n%s",
                release_name, proc.returncode, stderr.decode(),
            )
        else:
            logger.info("K8sRuntime: release '%s' uninstalled", release_name)

    async def is_alive(self, agent_url: str) -> bool:
        """Return True if the agent's /health endpoint responds with 200."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{agent_url}/health", timeout=2.0)
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    async def _try_oci_registry_login(chart_ref: str) -> None:
        """Authenticate helm to an OCI registry before pulling the chart.

        Extracts the registry hostname from an ``oci://`` chart reference and
        attempts ``helm registry login`` using an OAuth2 access token obtained
        from the GCP instance metadata server.  This is the standard approach
        for GKE workloads accessing Google Artifact Registry.

        Silently skips when:
        - *chart_ref* is not an OCI reference
        - the metadata server is unreachable (non-GKE environments)
        - the login command fails (will surface later in helm upgrade)
        """
        if not chart_ref.startswith("oci://"):
            return

        registry = chart_ref[len("oci://"):].split("/")[0]

        # Fetch an access token from the GCP metadata server (GKE only).
        try:
            import json
            import urllib.request
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1"
                "/instance/service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                access_token: str = json.loads(resp.read())["access_token"]
        except Exception:
            return  # Not on GKE or metadata server unavailable — skip silently

        proc = await asyncio.create_subprocess_exec(
            "helm", "registry", "login", registry,
            "--username", "oauth2accesstoken",
            "--password", access_token,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info("K8sRuntime: authenticated helm to OCI registry %s", registry)
        else:
            logger.warning(
                "K8sRuntime: helm registry login to %s failed: %s",
                registry, stderr.decode(),
            )
