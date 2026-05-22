"""Entry point: python -m agent_server --port 8000 --agent agent:run

Command-line arguments
----------------------
--port     TCP port to listen on (default: value of AGENT_PORT env var or 8000)
--agent    Python import path to the agent coroutine, e.g. ``my_module:run``

Environment variables
---------------------
AGENT_PORT            Overrides --port
BACKEND_CALLBACK_URL  Injected by the backend runtime
RUN_ID                Injected by the backend runtime
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

import uvicorn

from agent_server.server import create_app


def _load_agent_fn(agent_path: str):
    """Import and return the agent coroutine from a ``module:attr`` path."""
    if ":" not in agent_path:
        raise ValueError(
            f"--agent must be in 'module:attr' format, got: {agent_path!r}"
        )
    module_path, attr = agent_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, attr, None)
    if fn is None:
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr}'")
    return fn


def main() -> None:
    parser = argparse.ArgumentParser(description="langgraph-backend agent server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AGENT_PORT", "8000")),
        help="TCP port to listen on (default: AGENT_PORT env var or 8000)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent coroutine import path, e.g. 'my_module:run'",
    )
    args = parser.parse_args()

    agent_fn = None
    if args.agent:
        try:
            agent_fn = _load_agent_fn(args.agent)
        except Exception as exc:
            print(f"Failed to load agent function '{args.agent}': {exc}", file=sys.stderr)
            sys.exit(1)

    app = create_app(agent_fn=agent_fn)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
