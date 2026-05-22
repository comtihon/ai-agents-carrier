from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.runtime.base import AgentRuntime

if TYPE_CHECKING:
    from app.domain.models.agent_definition import AgentDefinition

logger = logging.getLogger(__name__)


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

    Requirements
    ------------
    ``aiodocker`` must be installed::

        pip install aiodocker

    TODO: full implementation
    -------------------------
    The stubs below define the interface.  Replace the ``raise NotImplementedError``
    calls with the real aiodocker API calls once the dependency is available.
    """

    def __init__(self) -> None:
        # TODO: initialise aiodocker.Docker() client
        # self._client = aiodocker.Docker()
        # Maps agent_url → container_id
        self._containers: dict[str, str] = {}

    async def spawn(
        self,
        agent_def: "AgentDefinition",
        step: dict[str, Any],
        run_id: str,
        callback_base_url: str,
    ) -> str:
        """Create and start a Docker container for the agent, then return its URL.

        TODO: implement with aiodocker
        --------------------------------
        import aiodocker
        client = aiodocker.Docker()
        image = step.get("image_override") or agent_def.image
        port = 8000  # fixed; CMD in the image binds to 8000

        env = {
            "AGENT_PORT": str(port),
            "BACKEND_CALLBACK_URL": callback_base_url,
            "RUN_ID": run_id,
        }

        container = await client.containers.create({
            "Image": image,
            # No "Cmd" override — the CMD is baked into the image.
            "Env": [f"{k}={v}" for k, v in env.items()],
            "ExposedPorts": {f"{port}/tcp": {}},
            "HostConfig": {
                "PortBindings": {f"{port}/tcp": [{"HostPort": "0"}]},  # random host port
            },
        })
        await container.start()

        # TODO: find the published host port from the container inspect data.
        # info = await container.show()
        # host_port = info["NetworkSettings"]["Ports"][f"{port}/tcp"][0]["HostPort"]
        # agent_url = f"http://localhost:{host_port}"

        # Poll GET /health until the server is ready (10s timeout).
        # ...

        agent_url = f"http://localhost:{host_port}"
        self._containers[agent_url] = container.id
        return agent_url
        """
        raise NotImplementedError(
            "DockerRuntime.spawn: install aiodocker and implement this method. "
            "See the TODO comments in app/runtime/docker.py."
        )

    async def terminate(self, agent_url: str) -> None:
        """Call POST /terminate then docker stop + docker rm.

        TODO: implement with aiodocker
        --------------------------------
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{agent_url}/terminate", timeout=3.0)
        except Exception:
            pass

        container_id = self._containers.pop(agent_url, None)
        if container_id:
            container = await self._client.containers.get(container_id)
            await container.stop()
            await container.delete()
        """
        raise NotImplementedError(
            "DockerRuntime.terminate: install aiodocker and implement this method."
        )

    async def is_alive(self, agent_url: str) -> bool:
        """Call GET {agent_url}/health and return True if status is 200.

        TODO: implement with aiodocker (or httpx alone)
        --------------------------------
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{agent_url}/health", timeout=2.0)
                return resp.status_code == 200
        except Exception:
            return False
        """
        raise NotImplementedError(
            "DockerRuntime.is_alive: install aiodocker and implement this method."
        )
