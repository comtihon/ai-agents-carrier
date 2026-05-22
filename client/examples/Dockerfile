# Sample Dockerfile for a langgraph-backend compatible agent (HTTP protocol).
#
# Build:
#   docker build -t my-agent .
#
# The backend runtime injects these environment variables when spawning the container:
#   AGENT_PORT            — TCP port the server should listen on
#   BACKEND_CALLBACK_URL  — backend base URL for callbacks
#   RUN_ID                — workflow run identifier
#
# The image must satisfy the runtime contract:
#   1. Expose a FastAPI server (via agent_server) on AGENT_PORT.
#   2. GET  /health   returns 200 when ready.
#   3. POST /start    accepts {run_id, input, callback_url} and runs the agent.
#   4. POST /terminate requests graceful shutdown.

FROM python:3.12-slim

WORKDIR /app

# Install the agent_server package and agent dependencies.
# Replace with your actual requirements.
RUN pip install --no-cache-dir fastapi uvicorn httpx

# (Optional) Install LangGraph or Anthropic SDK depending on your agent type:
# RUN pip install langgraph langchain-openai
# RUN pip install anthropic

# Copy the reusable agent_server package.
COPY agent_server/ ./agent_server/

# Copy your agent source.
# Adjust the source path to match your project layout.
COPY examples/langgraph_agent/agent.py ./agent.py

# Start the agent server on the port injected by the backend runtime.
# The --agent flag points to the run() coroutine in your agent module.
CMD ["python", "-m", "agent_server", "--agent", "agent:run"]
