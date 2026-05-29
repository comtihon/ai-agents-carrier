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

### `langgraph-agent` / `claude-agent` — autonomous agent step

Spawns a registered agent, sends the task, suspends until the agent calls back with its result. Supports `local` (in-process), `docker`, and `k8s` runtimes.

```yaml
- id: researcher
  type: langgraph-agent          # or claude-agent
  agent_id: my-researcher        # must exist in /api/v1/agents
  runtime_override: docker       # local | docker | k8s (defaults to agent's default_runtime)
  image: myregistry/my-agent:1.0 # Docker image override
  output_key: agent_result       # stores the agent's text result
  compression_level: full        # none | lite | full | ultra — caveman-compress the agent's responses
  env_vars:                      # additional env vars forwarded to the container
    - name: GOOGLE_APPLICATION_CREDENTIALS_JSON
      from_config: GOOGLE_APPLICATION_CREDENTIALS_JSON   # from backend config
    - name: MY_VAR
      value: custom-value        # literal value
  output_mapping:                # map individual agent output keys → state keys (optional)
    result: agent_result
```

**After the agent completes**, a meta-LLM call analyzes the output and decides:
- **PROCEED** — result is good, workflow continues
- **ASK_CLARIFICATION** — agent was blocked; UI shows a question form before proceeding
- **ASK_APPROVAL** — output needs human sign-off (falls through to the next `human_approval` step)

Configure the meta-LLM via `META_LLM_PROVIDER` / `META_LLM_MODEL` (default: haiku).

### `parallel` — fan-out to concurrent branches

Starts multiple branches in parallel. Each target step runs concurrently; edges define which steps are in the parallel group.

```yaml
- id: fan_out
  type: parallel
  max_parallel: 3      # max concurrent branches (default: unlimited)
  targets:
    - branch_a
    - branch_b
    - branch_c
```

### `join` — wait for all parallel branches

Waits for all incoming branches to complete before continuing.

```yaml
- id: fan_in
  type: join
  max_timeout: 300     # fail if branches don't finish within N seconds (default: unlimited)
```

### `switch` — conditional routing

Routes to one of several targets based on a condition expression. Conditions are evaluated in order; the first truthy condition wins. `when: null` is an unconditional default.

```yaml
- id: router
  type: switch
  routes:
    - when: "score > 4 and status != 'skip'"   # Python expression; state vars in scope
      next: high_priority
    - when: approved                            # simple bool state key
      next: standard_path
    - when: null                               # default fallback
      next: low_priority
```

**Expression syntax**: any Python expression using state variables. `&&` / `||` / `===` / `!==` are accepted as JS aliases and rewritten to Python equivalents. Available builtins: `len`, `str`, `int`, `float`, `bool`, `abs`, `min`, `max`, `sum`, `round`, `any`, `all`, `sorted`, `isinstance`.

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

## Agents

Agents are registered persistent definitions that `langgraph-agent` / `claude-agent` steps look up by `agent_id`. Each definition stores the runtime type, Docker image, and the `agent_input` dict (system prompt, model, tools, etc.) forwarded to the agent on every run.

```yaml
# Example agent definition (managed via API or copilot_ui)
id: researcher
name: Researcher
default_runtime: docker
image: europe-west4-docker.pkg.dev/myorg/registry/langgraph-agent:0.1.6
agent_input:
  system_prompt: "You are a research agent with access to bash and code-search tools."
  model: claude-opus-4-7
  max_tokens: 8096
health_timeout: 300     # seconds to wait for /health after container starts
```

**Agent HTTP protocol** — the backend calls the agent container:

```
POST /start     {run_id, input, callback_url, agent_config}  → 202 Accepted
GET  /health    → 200 when ready
POST /terminate → graceful shutdown
```

The agent calls back to the backend:

```
POST {callback_url}/api/v1/runs/{run_id}/agent/output    {output: {...}}
POST {callback_url}/api/v1/runs/{run_id}/agent/progress  {message: str}
POST {callback_url}/api/v1/runs/{run_id}/agent/question  {question, options?}
GET  {callback_url}/api/v1/runs/{run_id}/agent/input     (long-poll for answer)
```

Progress messages starting with `__token__:` carry live token counts: `__token__:{"input_tokens":N,"output_tokens":N,"total_tokens":N}` — the backend stores these in `_live_token_usage` and surfaces them in the run response for real-time display.

**Credential forwarding** — any env var in the backend matching a credential suffix (`_API_KEY`, `_TOKEN`, `_JSON`, `_SECRET`, `_CREDENTIALS`) is automatically available to forward to agent containers via the `env_vars` step config. The list is exposed at `GET /api/v1/llm/config/keys` (names only, no values).

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
GET    /api/v1/workflows/runs/{id}/trace      get LangSmith / token trace
POST   /api/v1/workflows/runs/{id}/approve    approve a paused run
POST   /api/v1/workflows/runs/{id}/reject     reject a paused run

# Agents
GET    /api/v1/agents                         list registered agents
POST   /api/v1/agents                         register an agent
GET    /api/v1/agents/{id}                    get agent definition
PUT    /api/v1/agents/{id}                    update agent definition
DELETE /api/v1/agents/{id}                    delete agent definition

# Agent callbacks (called by running agent containers)
POST   /api/v1/runs/{id}/agent/output         deliver result, resume run
POST   /api/v1/runs/{id}/agent/progress       send progress / token update
POST   /api/v1/runs/{id}/agent/question       ask a clarifying question
GET    /api/v1/runs/{id}/agent/input          long-poll for answer to question
POST   /api/v1/runs/{id}/agent/reply          submit answer (from UI)

# Config
GET    /api/v1/llm/config/keys                list forwardable credential key names
GET    /api/v1/llm/providers                  list configured LLM providers

# Triggers
POST   /api/v1/webhooks/{workflow-id}         HTTP webhook trigger

# Approval callbacks (no auth)
POST   /api/v1/callbacks/{run-id}/approve
POST   /api/v1/callbacks/{run-id}/reject
```

---

## Add-ons

### Slack approvals

Set `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, and `SLACK_APPROVALS_CHANNEL` to send approval requests to a Slack channel. The `human_approval` step's `notify` block targets Slack via webhook or the bot token.

### LangSmith tracing

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to send all LLM calls to LangSmith. The LangSmith run URL is included in the run trace response (`GET /runs/{id}/trace`).

### OpenHands code execution

Set `OPENHANDS_BASE_URL` and `OPENHANDS_API_KEY` (set `OPENHANDS_MOCK_MODE=false`) to enable the `execute` step type, which delegates coding tasks to an OpenHands instance.

### Custom LLM providers

`LLM_INTEGRATIONS` accepts a JSON array of OpenAI-compatible endpoints:

```bash
LLM_INTEGRATIONS='[{"name":"ollama","base_url":"http://localhost:11434/v1","default_model":"llama3","api_key_env":"OLLAMA_API_KEY"}]'
LLM_PROVIDER=ollama
```

Any entry can be referenced by name in workflow steps via `llm_provider: ollama`.
