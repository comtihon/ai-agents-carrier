"""FastAPI server for agent containers.

Exposes four endpoints that the backend uses to control the agent:

  GET  /health     — readiness probe
  GET  /status     — returns current run state
  POST /start      — backend sends {run_id, input, callback_url} to begin execution
  POST /terminate  — backend requests graceful shutdown

Agent authors write an async ``run(input_data, client)`` coroutine and register
it via ``create_app(agent_fn=...)``.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable, Coroutine
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_server.backend_client import BackendClient

logger = logging.getLogger(__name__)

RunState = Literal["idle", "running", "done", "failed"]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server passed from the backend."""

    name: str
    url: str | None = None
    command: list[str] | None = None  # for local stdio MCP servers
    env: dict[str, str] = {}
    transport: str = "sse"  # sse | stdio | streamable_http


class AgentConfig(BaseModel):
    """Agent configuration forwarded from the backend on every ``/start`` call.

    The backend resolves these values from ``AgentDefinition`` fields and the
    platform MCP / LLM settings so that the agent does not need to read its own
    environment variables for service discovery.
    """

    system_prompt: str | None = None
    model: str | None = None
    tools: list[str] | None = None          # which tools / MCP servers are enabled
    mcp_servers: list[MCPServerConfig] = [] # full MCP server configs
    credentials: dict[str, str] = {}        # API keys, tokens — passed as resolved values
    extra: dict = {}                        # AgentDefinition.config passthrough
    expected_output_fields: list[str] = [] # keys the agent MUST include in its output dict


class StartRequest(BaseModel):
    run_id: str
    input: dict[str, Any]
    callback_url: str
    agent_config: AgentConfig = AgentConfig()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    agent_fn: Callable[..., Coroutine[Any, Any, None]] | None = None,
) -> FastAPI:
    """Create and return the agent FastAPI application.

    Parameters
    ----------
    agent_fn:
        An async callable with signature
        ``async def run(input_data, config, client)`` where *config* is an
        :class:`AgentConfig` instance.
        If None, the server starts in idle mode and ``POST /start`` will return
        500 until an agent function is registered.
    """
    app = FastAPI(title="Agent Server", docs_url=None, redoc_url=None)

    # Shared mutable state (safe because asyncio is single-threaded)
    state: dict[str, Any] = {
        "status": "idle",
        "run_id": None,
        "task": None,         # asyncio.Task for the running agent
        "agent_fn": agent_fn,
    }

    @app.get("/health")
    async def health():
        """Readiness probe — always returns 200 once the server is up."""
        return {"status": "ok"}

    @app.get("/status")
    async def status():
        """Return the current run state."""
        return {
            "status": state["status"],
            "run_id": state["run_id"],
        }

    @app.post("/start", status_code=202)
    async def start(body: StartRequest):
        """Start the agent.  Returns 202 immediately; execution runs in the background."""
        if state["status"] == "running":
            return JSONResponse(
                status_code=409,
                content={"detail": "Agent is already running"},
            )

        fn = state["agent_fn"]
        if fn is None:
            return JSONResponse(
                status_code=500,
                content={"detail": "No agent function registered"},
            )

        client = BackendClient(
            callback_url=body.callback_url,
            run_id=body.run_id,
        )
        state["status"] = "running"
        state["run_id"] = body.run_id

        async def _run_agent():
            try:
                await fn(body.input, body.agent_config, client)
                state["status"] = "done"
                logger.info("[agent] run_id=%s finished", body.run_id)
            except asyncio.CancelledError:
                state["status"] = "idle"
                logger.info("[agent] run_id=%s cancelled", body.run_id)
            except Exception as exc:
                state["status"] = "failed"
                logger.exception("[agent] run_id=%s failed: %s", body.run_id, exc)
                # Best-effort: try to notify the backend of the failure
                try:
                    await client.send_output({"error": str(exc)})
                except Exception:
                    pass

        task = asyncio.create_task(_run_agent())
        state["task"] = task
        return {"run_id": body.run_id, "status": "started"}

    @app.post("/terminate")
    async def terminate():
        """Request graceful shutdown.  Cancels the running task then exits."""
        task: asyncio.Task | None = state["task"]
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        logger.info("[agent] terminate requested — exiting")

        # Schedule sys.exit after a short drain so the HTTP response can be sent.
        async def _exit():
            await asyncio.sleep(0.1)
            sys.exit(0)

        asyncio.create_task(_exit())
        return {"status": "terminating"}

    return app
