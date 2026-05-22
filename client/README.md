# langgraph-backend Agent Client

This directory contains the reusable `agent_server` package and examples showing
how to write agents compatible with the `langgraph-backend` HTTP agent protocol.

## Overview

The backend supports two agent step types in workflow YAML definitions:

- `langgraph-agent` — runs a LangGraph-based agent
- `claude-agent` — runs a Claude SDK-based agent

Both step types use the same **HTTP protocol** and the same runtime system
(`local`, `docker`, or `k8s`).

---

## HTTP Protocol

Instead of stdin/stdout, agents now communicate with the backend over HTTP:

```
Backend                         Agent Server
   │                                  │
   │  1. spawn (start subprocess)     │
   │─────────────────────────────────>│
   │                                  │ (server starts, binds port)
   │  2. GET /health (poll)           │
   │─────────────────────────────────>│
   │  200 OK {"status": "ok"}        │
   │<─────────────────────────────────│
   │                                  │
   │  3. POST /start                  │
   │  {run_id, input, callback_url}   │
   │─────────────────────────────────>│
   │  202 Accepted                    │
   │<─────────────────────────────────│ (agent runs in background)
   │                                  │
   │  (optional) POST /agent/question │
   │<─────────────────────────────────│
   │  (optional) GET /agent/input     │
   │<─────────────────────────────────│ (agent long-polls for answer)
   │  {"answer": "..."}               │
   │─────────────────────────────────>│
   │                                  │
   │  POST /runs/{run_id}/agent/output│
   │<─────────────────────────────────│
   │  202 Accepted                    │
   │─────────────────────────────────>│
```

### Agent Server Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness probe — 200 when server is up |
| `GET` | `/status` | Returns `{"status": "idle|running|done|failed"}` |
| `POST` | `/start` | Start the agent (`{run_id, input, callback_url}`) |
| `POST` | `/terminate` | Graceful shutdown request |

### Backend Callback Endpoints

The agent uses `BackendClient` to call these:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/runs/{run_id}/agent/output` | Agent done — resume the run |
| `POST` | `/api/v1/runs/{run_id}/agent/question` | Agent asks a question |
| `GET` | `/api/v1/runs/{run_id}/agent/input` | Long-poll for answer (10 min timeout) |
| `POST` | `/api/v1/runs/{run_id}/agent/reply` | Frontend submits answer |
| `POST` | `/api/v1/runs/{run_id}/agent/progress` | Optional progress update |

---

## Environment Variables

The backend runtime injects these into every agent container / subprocess:

| Variable | Description |
|----------|-------------|
| `AGENT_PORT` | TCP port the server should listen on |
| `BACKEND_CALLBACK_URL` | Backend base URL (e.g. `http://localhost:8000`) |
| `RUN_ID` | Workflow run identifier |

---

## Agent Configuration (`AgentConfig`)

Every `POST /start` call now includes an `agent_config` object that is passed
directly to your agent function as the second positional argument.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `system_prompt` | `str \| None` | System message for LLM calls (set in `AgentDefinition.system_prompt`) |
| `model` | `str \| None` | LLM model identifier, e.g. `"claude-opus-4-7"` (set in `AgentDefinition.model`) |
| `tools` | `list[str] \| None` | Enabled tool / MCP server names; `None` = all enabled ones |
| `mcp_servers` | `list[MCPServerConfig]` | Full MCP server configs resolved from backend settings |
| `credentials` | `dict[str, str]` | Resolved API keys forwarded from the backend |
| `extra` | `dict` | Passthrough of `AgentDefinition.config` (arbitrary extras) |

### Using credentials

```python
from agent_server import AgentConfig

async def run(input_data, config: AgentConfig, client):
    api_key = config.credentials.get("ANTHROPIC_API_KEY")
    # Falls back to env var if not in credentials
    import anthropic
    claude = anthropic.Anthropic(api_key=api_key)
```

### Using MCP servers

```python
async def run(input_data, config: AgentConfig, client):
    for mcp in config.mcp_servers:
        print(f"MCP server '{mcp.name}': transport={mcp.transport}, url={mcp.url}")
```

### Security note

Credentials are passed as resolved string values in the HTTP body of the
`POST /start` call.  The call is made over the local cluster network
(loopback or private VPC), so this is acceptable for single-cluster
deployments.  For stricter security in production Kubernetes, mount credentials
as K8s Secrets and inject them as environment variables in the agent's pod spec
instead of relying on this in-memory forwarding.

---

## Writing an Agent

### 1. Implement the run coroutine

```python
# my_agent.py
from agent_server import AgentConfig, BackendClient

async def run(input_data: dict, config: AgentConfig, client: BackendClient) -> None:
    # Optional: report progress
    await client.send_progress("Starting analysis…")

    # Optional: ask a clarifying question
    scope = await client.ask_question(
        "What scope should I search?",
        options=["broad", "narrow"],
    )

    # Do your work
    result = do_work(input_data, scope)

    # Send the output back to the backend
    await client.send_output({"result": result})
```

### 2. Start the server

```bash
AGENT_PORT=18001 \
BACKEND_CALLBACK_URL=http://localhost:8000 \
RUN_ID=test-run-1 \
python -m agent_server --port 18001 --agent my_agent:run
```

### 3. Docker

Use `examples/Dockerfile` as a starting point:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install fastapi uvicorn httpx langgraph langchain-openai
COPY agent_server/ ./agent_server/
COPY my_agent.py ./
CMD ["python", "-m", "agent_server", "--agent", "my_agent:run"]
```

---

## Examples

### LangGraph Agent

See [`examples/langgraph_agent/agent.py`](examples/langgraph_agent/agent.py).

```bash
pip install langgraph langchain-openai httpx fastapi uvicorn
AGENT_PORT=18001 BACKEND_CALLBACK_URL=http://localhost:8000 RUN_ID=test \
  OPENAI_API_KEY=sk-... \
  python -m agent_server --port 18001 --agent examples/langgraph_agent/agent:run
```

### Claude SDK Agent

See [`examples/claude_agent/agent.py`](examples/claude_agent/agent.py).

```bash
pip install anthropic httpx fastapi uvicorn
AGENT_PORT=18001 BACKEND_CALLBACK_URL=http://localhost:8000 RUN_ID=test \
  ANTHROPIC_API_KEY=sk-ant-... \
  python -m agent_server --port 18001 --agent examples/claude_agent/agent:run
```

---

## Registering an Agent

Before using a step with `agent_id: my-agent`, register the agent definition:

```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "id": "my-agent",
    "name": "My LangGraph Agent",
    "type": "langgraph",
    "default_runtime": "local",
    "entrypoint": ["python", "-m", "agent_server", "--agent", "agent:run"],
    "agent_server_port": 8000,
    "env": {
      "OPENAI_API_KEY": "sk-..."
    }
  }'
```

For Docker:

```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -d '{
    "id": "my-docker-agent",
    "type": "claude",
    "default_runtime": "docker",
    "image": "my-org/my-claude-agent:latest",
    "agent_server_port": 8000,
    "env": {
      "ANTHROPIC_API_KEY": "sk-ant-..."
    }
  }'
```

---

## Workflow Step Configuration

```yaml
steps:
  - id: my_agent_step
    type: langgraph-agent        # or claude-agent
    agent_id: my-agent           # references a registered AgentDefinition

    # Map workflow state keys → agent input keys.
    input_mapping:
      request: request
      plan: task_description

    # Map agent output keys → workflow state keys.
    output_mapping:
      answer: final_answer

    # Fallback: store the whole agent output under this state key.
    output_key: agent_result

    # Per-step overrides (optional):
    runtime_override: docker     # local | docker | k8s
    image_override: my-org/my-agent:v2
```

---

## Agent Definitions API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/agents` | List all agent definitions |
| `POST` | `/api/v1/agents` | Create a new agent definition |
| `GET` | `/api/v1/agents/{id}` | Get an agent definition |
| `PUT` | `/api/v1/agents/{id}` | Update an agent definition |
| `DELETE` | `/api/v1/agents/{id}` | Delete an agent definition |
| `GET` | `/api/v1/agents/types` | List supported types and runtimes |
