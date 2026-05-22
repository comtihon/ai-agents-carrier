"""agent_server — reusable FastAPI server for langgraph-backend agents.

Usage
-----
Embed this package in your agent container and start it with::

    python -m agent_server --port 8000 --agent my_agent:run

The ``--agent`` argument specifies the Python import path to your async
``run(input_data: dict, client: BackendClient) -> None`` coroutine.

Environment variables
---------------------
AGENT_PORT            TCP port to listen on (overrides --port)
BACKEND_CALLBACK_URL  Backend base URL (set automatically by the runtime)
RUN_ID                Workflow run ID (set automatically by the runtime)
"""
from agent_server.backend_client import BackendClient
from agent_server.server import AgentConfig, MCPServerConfig, create_app

__all__ = ["AgentConfig", "BackendClient", "MCPServerConfig", "create_app"]
