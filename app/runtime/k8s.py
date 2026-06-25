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

    def __init__(self, namespace: str = "default", callback_override_url: str | None = None) -> None:
        self._namespace = namespace
        self._callback_override_url = callback_override_url
        # Maps agent_url → release_name
        self._releases: dict[str, str] = {}

    def rewrite_callback_url(self, callback_base_url: str) -> str:
        """Return the effective callback URL for the agent.

        When ``callback_override_url`` is set (e.g. an internal cluster URL),
        it takes precedence over the externally-visible ``callback_base_url``.
        """
        return self._callback_override_url if self._callback_override_url else callback_base_url

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

        # Build --set arguments from helm_values plus runtime overrides.
        # Skip dict values — they can't be serialised to --set and indicate a
        # corrupted/nested helm_values document (e.g. from a bad $set path).
        set_args: list[str] = []
        for key, value in (agent_def.helm_values or {}).items():
            if isinstance(value, dict):
                continue
            set_args += ["--set", f"{key}={value}"]
        set_args += ["--set", f"env.AGENT_PORT={_AGENT_PORT}"]
        effective_url = self.rewrite_callback_url(callback_base_url)
        set_args += ["--set", f"env.BACKEND_CALLBACK_URL={effective_url}"]
        set_args += ["--set", f"env.RUN_ID={run_id}"]
        for k, v in (extra_env or {}).items():
            set_args += ["--set-string", f"env.{k}={v}"]

        # PVC addon: create PVC and inject volumes/volumeMounts
        pvc_mount_point = (step or {}).get("pvc_mount_point") if step else None
        if pvc_mount_point and self._is_k8s_available():
            pvc_name = (step or {}).get("pvc_name") or f"pvc-{run_id[:12]}"
            from app.runtime.pvc_manager import PvcManager, is_valid_pvc_name
            if not is_valid_pvc_name(pvc_name):
                pvc_name = f"pvc-{run_id[:12]}"
            await PvcManager(self._namespace).create_pvc(pvc_name)
            import json as _json
            volumes_json = _json.dumps([{"name": "pvc-vol", "persistentVolumeClaim": {"claimName": pvc_name}}])
            mounts_json = _json.dumps([{"name": "pvc-vol", "mountPath": pvc_mount_point}])
            set_args += ["--set-json", f"volumes={volumes_json}"]
            set_args += ["--set-json", f"volumeMounts={mounts_json}"]

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

        _helm_retries = 3
        _helm_retry_delay = 10.0
        for _attempt in range(_helm_retries):
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                break
            stderr_text = stderr.decode()
            if "another operation" in stderr_text and _attempt < _helm_retries - 1:
                logger.warning(
                    "K8sRuntime: helm upgrade for '%s' blocked by concurrent operation, retrying in %.0fs (attempt %d/%d)",
                    release_name, _helm_retry_delay, _attempt + 1, _helm_retries,
                )
                await asyncio.sleep(_helm_retry_delay)
                continue
            raise RuntimeError(
                f"K8sRuntime: helm upgrade failed for release '{release_name}':\n"
                f"{stderr_text}"
            )

        # Discover the Service created by this Helm release.  Helm charts typically
        # name services as "{release_name}-{chart_name}", so we look up by the
        # standard app.kubernetes.io/instance label rather than hard-coding the suffix.
        agent_url = await self._discover_service_url(release_name)
        self._releases[agent_url] = release_name
        logger.info("K8sRuntime: release '%s' deployed at %s", release_name, agent_url)

        # Poll GET /health until the agent service is reachable.
        # Inside the cluster the URL above is directly accessible.
        health_timeout = float(agent_def.health_timeout or _HEALTH_TIMEOUT)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + health_timeout
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
            f"within {health_timeout}s (run_id={run_id})"
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

    async def terminate_by_run_id(self, run_id: str) -> None:
        """Uninstall all Helm releases for the given run_id (best-effort, never raises)."""
        try:
            import json as _json
            prefix = run_id[:8]
            result = await asyncio.create_subprocess_exec(
                "helm", "list", "-n", self._namespace, "--filter", f"agent-.*-{prefix}",
                "-o", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await result.communicate()
            releases = _json.loads(stdout or "[]")
            for rel in releases:
                release_name = rel.get("name", "")
                if not release_name:
                    continue
                try:
                    cmd = [
                        "helm", "uninstall", release_name,
                        "--namespace", self._namespace,
                    ]
                    logger.info("K8sRuntime: helm uninstall '%s' (terminate_by_run_id)", release_name)
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await proc.communicate()
                    # Clean up in-memory tracking
                    self._releases = {k: v for k, v in self._releases.items() if v != release_name}
                except Exception:
                    pass
        except Exception:
            pass

    async def has_container_for_run(self, run_id: str) -> bool:
        """Return True if a Helm release for this run_id already exists in the cluster.

        Used by agent_executor to detect whether a pod was already spawned (e.g. by
        a previous backend instance) so it can resume instead of spawning a duplicate.
        """
        try:
            import json as _json
            prefix = run_id[:8]
            proc = await asyncio.create_subprocess_exec(
                "helm", "list",
                "-n", self._namespace,
                "--filter", f"agent-.*-{prefix}",
                "-o", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            releases = _json.loads(stdout or "[]")
            return bool(releases)
        except Exception:
            return False

    async def is_alive(self, agent_url: str) -> bool:
        """Return True if the agent's /health endpoint responds with 200."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{agent_url}/health", timeout=2.0)
                return resp.status_code == 200
        except Exception:
            return False

    async def _discover_service_url(self, release_name: str) -> str:
        """Return the in-cluster URL for the first Service in the Helm release manifest.

        Helm charts typically name Services as ``{release}-{chart_name}`` rather
        than exactly ``{release}``.  We parse ``helm get manifest`` to discover
        the actual Service name without assuming the chart's naming convention.

        Falls back to ``{release_name}.{namespace}`` if parsing fails.
        """
        proc = await asyncio.create_subprocess_exec(
            "helm", "get", "manifest", release_name,
            "--namespace", self._namespace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            import yaml as _yaml
            try:
                for doc in _yaml.safe_load_all(stdout.decode()):
                    if isinstance(doc, dict) and doc.get("kind") == "Service":
                        svc_name = doc.get("metadata", {}).get("name", "")
                        if svc_name:
                            url = f"http://{svc_name}.{self._namespace}.svc.cluster.local:{_AGENT_PORT}"
                            logger.info(
                                "K8sRuntime: discovered service '%s' for release '%s'",
                                svc_name, release_name,
                            )
                            return url
            except Exception:
                pass
        logger.warning(
            "K8sRuntime: could not discover service for release '%s', using release name as fallback",
            release_name,
        )
        return f"http://{release_name}.{self._namespace}.svc.cluster.local:{_AGENT_PORT}"

    @staticmethod
    def _is_k8s_available() -> bool:
        """Return True if the kubernetes client library is importable."""
        try:
            import kubernetes  # type: ignore  # noqa: F401
            return True
        except ImportError:
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
