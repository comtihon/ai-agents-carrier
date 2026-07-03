from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import TYPE_CHECKING, Any

import aiodocker
import httpx

from app.runtime.base import AgentRuntime

if TYPE_CHECKING:
    from app.domain.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)

_AGENT_PORT = 8000
_HEALTH_TIMEOUT = 300.0       # default seconds to wait for /health to return 200
_HEALTH_POLL_INTERVAL = 0.5   # seconds between health-check attempts


class DockerRuntime(AgentRuntime):
    """Spawn the agent as a Docker container running a FastAPI HTTP server.

    HTTP contract
    -------------
    The container image must expose:
      GET  /health  — returns 200 when ready
      POST /start   — receives {run_id, input, callback_url, agent_config}
      POST /terminate — shuts the server down gracefully

    The CMD is baked into the Docker image — no entrypoint override is needed.

    Environment variables injected into the container
    -------------------------------------------------
    AGENT_PORT            — TCP port the server should listen on (default 8000)
    BACKEND_CALLBACK_URL  — backend base URL for agent-to-backend callbacks
    RUN_ID                — workflow run identifier
    """

    def __init__(
        self,
        registry_username: str | None = None,
        registry_password: str | None = None,
    ) -> None:
        self._client = aiodocker.Docker()
        self._registry_auth: dict[str, str] | None = (
            {"username": registry_username, "password": registry_password}
            if registry_username and registry_password
            else None
        )
        # Maps agent_url → container_id
        self._containers: dict[str, str] = {}

    def rewrite_callback_url(self, callback_base_url: str) -> str:
        return (
            callback_base_url
            .replace("://localhost", "://host.docker.internal")
            .replace("://127.0.0.1", "://host.docker.internal")
        )

    def _fresh_auth(self) -> dict[str, str] | None:
        """Return auth dict, refreshing the token if using gcloud oauth2accesstoken."""
        if self._registry_auth is None:
            return None
        if self._registry_auth.get("username") == "oauth2accesstoken":
            try:
                token = subprocess.check_output(
                    ["gcloud", "auth", "print-access-token"], text=True, timeout=10
                ).strip()
                return {"username": "oauth2accesstoken", "password": token}
            except Exception as exc:
                logger.warning("DockerRuntime: gcloud token refresh failed (%s), using cached token", exc)
        return self._registry_auth

    async def spawn(
        self,
        agent_def: "AgentDefinition",
        step: dict[str, Any],
        run_id: str,
        callback_base_url: str,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        """Create and start a Docker container for the agent, then return its URL.

        The container is started with a random host port bound to ``_AGENT_PORT``
        inside the container.  The published host port is discovered via
        ``container.show()`` after the container starts.  The method then polls
        ``GET /health`` for up to ``_HEALTH_TIMEOUT`` seconds before returning.

        Raises
        ------
        ValueError
            When ``agent_def.image`` is not set and ``step`` contains no
            ``image_override``.
        RuntimeError
            When the agent does not become healthy within ``_HEALTH_TIMEOUT``
            seconds.
        """
        image = step.get("image_override") or step.get("image") or agent_def.image
        health_timeout = float(
            step.get("health_timeout") or agent_def.health_timeout or _HEALTH_TIMEOUT
        )
        if not image:
            raise ValueError(
                f"DockerRuntime: agent '{agent_def.id}' has no image configured. "
                "Set AgentDefinition.image or provide image/image_override in the step."
            )

        # Pull the image before creating the container so that the latest
        # version is always used and creation does not fail with 404 when
        # the image is not present locally (e.g. private Artifact Registry).
        logger.info("DockerRuntime: pulling image %s ...", image)
        try:
            auth = self._fresh_auth()
            await self._client.images.pull(from_image=image, auth=auth)
            logger.info("DockerRuntime: image %s pulled successfully", image)
        except Exception as exc:
            # Re-raise with a clear message so the run error is human-readable.
            # Swallowing the error and falling through to containers.create()
            # produces a cryptic [404] from the Docker daemon instead.
            raise RuntimeError(
                f"DockerRuntime: failed to pull image '{image}': {exc}"
            ) from exc

        port = _AGENT_PORT

        # Rewrite localhost/127.0.0.1 in the callback URL so the container
        # reaches the host, not itself.  host.docker.internal is available on
        # Docker Desktop (Mac/Windows) natively; on Linux we inject the alias
        # via ExtraHosts below.
        container_callback_url = (
            callback_base_url
            .replace("://localhost", "://host.docker.internal")
            .replace("://127.0.0.1", "://host.docker.internal")
        )

        env = {
            **(extra_env or {}),
            "AGENT_PORT": str(port),
            "BACKEND_CALLBACK_URL": container_callback_url,
            "RUN_ID": run_id,
        }

        container = await self._client.containers.create({
            "Image": image,
            "Labels": {"langgraph_run_id": run_id},
            # No "Cmd" override — the CMD is baked into the image.
            "Env": [f"{k}={v}" for k, v in env.items()],
            "ExposedPorts": {f"{port}/tcp": {}},
            "HostConfig": {
                "PortBindings": {f"{port}/tcp": [{"HostPort": "0"}]},  # random host port
                # Lets the container resolve host.docker.internal → host gateway on Linux
                "ExtraHosts": ["host.docker.internal:host-gateway"],
            },
        })
        await container.start()

        # Find the published host port from the container inspect data.
        info = await container.show()
        host_port = info["NetworkSettings"]["Ports"][f"{port}/tcp"][0]["HostPort"]
        agent_url = f"http://localhost:{host_port}"

        self._containers[agent_url] = container.id
        logger.info(
            "DockerRuntime: container %s started at %s (run_id=%s)",
            container.id[:12], agent_url, run_id,
        )

        # Poll GET /health until the server is ready (health_timeout seconds).
        # Every 5 seconds also check the container state so we fail fast when
        # the process crashed instead of waiting out the full timeout.
        loop = asyncio.get_event_loop()
        deadline = loop.time() + health_timeout
        _last_state_check = loop.time()
        _STATE_CHECK_INTERVAL = 5.0
        async with httpx.AsyncClient() as client:
            while loop.time() < deadline:
                try:
                    resp = await client.get(f"{agent_url}/health", timeout=2.0)
                    if resp.status_code == 200:
                        logger.info(
                            "DockerRuntime: agent at %s is healthy", agent_url
                        )
                        return agent_url
                except Exception:
                    pass

                now = loop.time()
                if now - _last_state_check >= _STATE_CHECK_INTERVAL:
                    _last_state_check = now
                    state, tail = await self._get_container_state(container.id)
                    if state in ("exited", "dead"):
                        logger.error(
                            "DockerRuntime: container %s exited early (state=%r) logs:\n%s",
                            container.id[:12], state, tail,
                        )
                        await self.terminate(agent_url)
                        raise RuntimeError(
                            f"DockerRuntime: container exited before becoming healthy "
                            f"(state={state!r}, run_id={run_id}). "
                            f"Last logs:\n{tail}"
                        )
                    logger.debug(
                        "DockerRuntime: container %s still starting (state=%r)",
                        container.id[:12], state,
                    )

                await asyncio.sleep(_HEALTH_POLL_INTERVAL)

        # Agent did not become healthy within timeout — collect logs and raise.
        _, tail = await self._get_container_state(container.id)
        await self.terminate(agent_url)
        raise RuntimeError(
            f"DockerRuntime: agent at {agent_url} did not become healthy "
            f"within {health_timeout}s (run_id={run_id}). "
            f"Last logs:\n{tail}"
        )

    async def _get_container_state(self, container_id: str) -> tuple[str, str]:
        """Return (state, last_logs) for a container. Never raises."""
        try:
            container = await self._client.containers.get(container_id)
            info = await container.show()
            state = info.get("State", {}).get("Status", "unknown")
            logs = await container.log(stdout=True, stderr=True, tail=30)
            tail = "".join(logs).strip()
            return state, tail
        except Exception as exc:
            return "unknown", f"(could not retrieve logs: {exc})"

    async def terminate(self, agent_url: str) -> None:
        """Call POST /terminate then docker stop + docker rm.

        The HTTP call is best-effort — any error is swallowed so that the
        container is always removed even if the agent is unresponsive.
        """
        # Best-effort graceful shutdown via HTTP first.
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{agent_url}/terminate", timeout=3.0)
        except Exception:
            pass

        container_id = self._containers.pop(agent_url, None)
        if container_id:
            try:
                container = await self._client.containers.get(container_id)
                await container.stop()
                await container.delete()
                logger.info(
                    "DockerRuntime: container %s removed (url=%s)",
                    container_id[:12], agent_url,
                )
            except Exception as exc:
                logger.warning(
                    "DockerRuntime: error removing container %s: %s",
                    container_id[:12], exc,
                )
        else:
            logger.debug(
                "DockerRuntime.terminate: no container tracked for %s", agent_url
            )

    async def has_container_for_run(self, agent_def: "AgentDefinition", run_id: str) -> bool:
        """Return True if a running container labelled with this run_id exists.

        ``agent_def`` is accepted for signature parity with ``K8sRuntime`` (whose
        equivalent method is now agent-scoped) but is currently unused — Docker's
        run_id-only scoping is a documented follow-up, out of scope for this fix.
        """
        try:
            containers = await self._client.containers.list(
                filters={"label": [f"langgraph_run_id={run_id}"], "status": ["running"]}
            )
            return len(containers) > 0
        except Exception:
            return False

    async def terminate_by_run_id(self, agent_def: "AgentDefinition | None", run_id: str) -> None:
        """Find and terminate all containers labeled langgraph_run_id=<run_id>.

        ``agent_def`` is accepted for signature parity with ``K8sRuntime`` (whose
        equivalent method is now agent-scoped) but is currently unused — Docker's
        run_id-only scoping is a documented follow-up, out of scope for this fix.
        """
        try:
            containers = await self._client.containers.list(
                filters={"label": [f"langgraph_run_id={run_id}"]}
            )
            for container in containers:
                try:
                    await container.stop()
                    await container.delete()
                    logger.info(
                        "DockerRuntime: container %s for run %s removed",
                        container.id[:12], run_id,
                    )
                    # Keep _containers dict consistent
                    self._containers = {
                        url: cid for url, cid in self._containers.items()
                        if cid != container.id
                    }
                except Exception as exc:
                    logger.warning(
                        "DockerRuntime: error removing container for run %s: %s", run_id, exc
                    )
        except Exception as exc:
            logger.warning("DockerRuntime: terminate_by_run_id failed for run %s: %s", run_id, exc)

    async def is_alive(self, agent_url: str) -> bool:
        """Call GET {agent_url}/health and return True if status is 200."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{agent_url}/health", timeout=2.0)
                return resp.status_code == 200
        except Exception:
            return False
