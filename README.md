# LangGraph Backend

Workflow orchestration backend for AI-assisted software delivery. Workflows are defined in YAML, run as LangGraph graphs, and can involve LLM reasoning, MCP tool calls, human approval gates, cron/webhook triggers, and autonomous code execution via OpenHands.

---

## How to run

```bash
docker-compose up -d          # start MongoDB
pip install -e ".[dev]"       # install deps
cp .env.example .env          # configure (see below)
uvicorn app.main:app --reload
```

API at `http://localhost:8000`. Health check: `GET /health`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | — | `anthropic` or `openai` |
| `LLM_MODEL` | provider default | Model name override |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `GOOGLE_API_KEY` | — | Required when `LLM_PROVIDER=google` |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DATABASE` | `langgraph_backend` | Database name |
| `WORKFLOW_BACKEND` | `localfiles` | `localfiles` or `mongodb` |
| `GRAPH_DEFINITIONS_PATH` | `graphs` | Directory of YAML workflow files (localfiles backend) |
| `BASE_URL` | `http://localhost:8000` | Public URL — used to build approval callback links |
| `WEBHOOK_SECRET` | — | HMAC-SHA256 secret for incoming webhook signatures |
| `OAUTH_ENABLED` | `false` | Enable JWT Bearer auth on all endpoints |
| `OAUTH_JWKS_URL` | — | JWKS endpoint for token validation |
| `OPENHANDS_BASE_URL` | `http://openhands:3000` | OpenHands service URL |
| `OPENHANDS_API_KEY` | — | OpenHands auth token |
| `OPENHANDS_MOCK_MODE` | `true` | Return stub results instead of calling OpenHands |
| `DOCKER_REGISTRY_USERNAME` | — | Registry username for pulling private images (DockerRuntime) |
| `DOCKER_REGISTRY_PASSWORD` | — | Registry password / token for pulling private images |
| `META_LLM_PROVIDER` | — | LLM provider for post-agent analysis (`anthropic` or `openai`; defaults to `LLM_PROVIDER`) |
| `META_LLM_MODEL` | `claude-haiku-4-5-20251001` | Model for post-agent meta-analysis (haiku recommended for cost/speed) |

### MCP integrations

Each integration is configured with three env vars: `MCP_<NAME>_ENABLED=true`, `MCP_<NAME>_URL`, `MCP_<NAME>_API_KEY`. Supported names: `FIGMA`, `JIRA`, `MIRO`, `NOTION`, `GITHUB`.

Jira also supports a stdio transport via `uvx mcp-atlassian` — set `MCP_JIRA_TRANSPORT=stdio` and provide `MCP_JIRA_JIRA_URL`, `MCP_JIRA_USERNAME`, `MCP_JIRA_API_TOKEN`.

### Docker runtime — private registry auth

When a workflow step uses `runtime: docker`, the backend pulls the agent image via the Docker daemon. For private registries, set credentials via env vars:

| Registry | `DOCKER_REGISTRY_USERNAME` | `DOCKER_REGISTRY_PASSWORD` |
|---|---|---|
| Google Artifact Registry | `oauth2accesstoken` | `$(gcloud auth print-access-token)` |
| AWS ECR | `AWS` | `$(aws ecr get-login-password --region <region>)` |
| Docker Hub / other | your username | password or personal access token |

No credentials set → pull proceeds without auth (public images, or if the Docker daemon already has credentials configured via `docker login`).

**Local `.env`:**
```bash
DOCKER_REGISTRY_USERNAME=oauth2accesstoken
DOCKER_REGISTRY_PASSWORD=ya29.your-token-here
```

**Kubernetes secret:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: registry-credentials
stringData:
  DOCKER_REGISTRY_USERNAME: oauth2accesstoken
  DOCKER_REGISTRY_PASSWORD: <token>
```
Then reference with `envFrom: - secretRef: name: registry-credentials` in the deployment.

### Agent meta-analysis

After each `langgraph-agent` or `claude-agent` step completes, the backend runs a lightweight internal LLM call to decide how to proceed:

- **PROCEED** — agent answered the request; workflow continues normally
- **ASK_CLARIFICATION** — agent was blocked or needs more info; the UI shows a question form before the workflow continues
- **ASK_APPROVAL** — output should be reviewed by a human (falls through to the next `human_approval` step)

Configure with a fast, cheap model to minimise cost:

```env
META_LLM_PROVIDER=anthropic
META_LLM_MODEL=claude-haiku-4-5-20251001
```

When `META_LLM_PROVIDER` is not set, the main `LLM_PROVIDER` value is used.

---

## Workflow definitions

Workflows are YAML files in `graphs/` (or the path set by `GRAPH_DEFINITIONS_PATH`). They are loaded at startup and can also be managed via the REST API (`GET/POST/PUT/DELETE /api/v1/workflows`).  
Workflows can be stored in MongoDB if the backend is configured.

```yaml
id: my-workflow
name: My Workflow
description: "..."
steps:
  - id: step-one
    type: llm_structured
    ...
  - id: step-two
    type: human_approval
    ...
```

Steps run sequentially. Each step can be skipped with `when: <state-key>` — the step is skipped if `state[key]` is falsy.

---

## Step types

### `llm_structured` — agentic LLM with structured output

Runs a tool-calling loop until the LLM emits a `submit_output` call with all required fields.

```yaml
- id: gather_context
  type: llm_structured
  system_prompt: "..."
  user_template: "Ticket: {ticket_id}"   # {key} resolved from state
  bind_mcp_tools: true                   # expose MCP tools to LLM (default true)
  max_iterations: 25
  fail_if_false:                         # fail run if any listed bool field is false
    - success
  output:
    - name: context
      type: str
      description: "..."
    - name: needs_jira
      type: bool
      description: "..."
```

### `llm` — single LLM call

One-shot call, no tool loop. Result stored as a string.

```yaml
- id: plan
  type: llm
  system_prompt: "You are a planning assistant."
  user_template: "Context: {context}"
  output_key: plan
```

### `mcp` — call an MCP tool directly

```yaml
- id: fetch_board
  type: mcp
  tool: miro_get_board
  tool_input:
    board_id: "{board_id}"
  output_key: board_data
```

### `human_approval` — pause for human review

Pauses the run (`status: waiting_approval`). Resume via `POST /api/v1/workflows/runs/{id}/approve` or `/reject`. The `approved` and `reject_reason` keys are written to state automatically.

```yaml
- id: approve
  type: human_approval
  interrupt_payload:
    plan: "{plan}"
  notify:                               # optional — send an HTTP notification
    url: "https://hooks.example.com/approval"
    auth:
      type: bearer                      # bearer | basic (optional)
      token: "my-token"
    payload:
      text: "Approval needed: {plan}"
      approve_url: "{approve_url}"      # auto-injected callback URL
      reject_url: "{reject_url}"        # auto-injected callback URL
```

Callback endpoints (no auth required — the UUID is the secret):

```
POST /api/v1/callbacks/{run_id}/approve
POST /api/v1/callbacks/{run_id}/reject   body: {"reason": "..."}
```

### `execute` — run code via OpenHands

```yaml
- id: implement
  type: execute
  when: approved
  repo_template: "{repo}"
  instructions_template: "Implement {ticket_id} per the plan:\n{plan}"
  output_key: implementation
```

### `http_call` — outbound HTTP request

```yaml
- id: create_ticket
  type: http_call
  url: "https://api.example.com/issues"
  method: POST
  headers:
    Authorization: "Bearer {token}"
  body:
    title: "{request}"
  output_key: ticket
```

### `workflow` — spawn a child workflow

```yaml
- id: spawn_child
  type: workflow
  workflow_id: another-workflow
  input_template: "{request}"
  output_key: child_result
```

### `python` — inline Python

```yaml
- id: transform
  type: python
  code: |
    output = state["items"][0]["value"]
  output_key: result
```

### `cron` — scheduled trigger

Entry-point step. Registers a cron job; each firing creates a new run.

```yaml
- id: trigger
  type: cron
  schedule: "0 9 * * 1-5"             # 5-field UTC cron
  request_template: "Daily run on {date}"
```

### `http` — webhook trigger

Entry-point step. Listens at `POST /api/v1/webhooks/{workflow-id}`. The request body is stored under `output_key`.

```yaml
- id: trigger
  type: http
  output_key: webhook_data
```

Incoming requests must include an `X-Webhook-Signature` header (HMAC-SHA256 of the body, keyed with `WEBHOOK_SECRET`).

---

## API

```
# Workflows
GET    /api/v1/workflows                      list workflows
POST   /api/v1/workflows                      create workflow
GET    /api/v1/workflows/{id}                 get workflow
PUT    /api/v1/workflows/{id}                 update workflow
DELETE /api/v1/workflows/{id}                 delete workflow

# Runs
POST   /api/v1/workflows/runs                 start a run
GET    /api/v1/workflows/runs/{id}            get run status
POST   /api/v1/workflows/runs/{id}/approve    approve a paused run
POST   /api/v1/workflows/runs/{id}/reject     reject a paused run

# Triggers
POST   /api/v1/webhooks/{workflow-id}         HTTP webhook trigger

# Approval callbacks (no auth)
POST   /api/v1/callbacks/{run-id}/approve
POST   /api/v1/callbacks/{run-id}/reject
```
