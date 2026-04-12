# LangGraph Backend

Workflow orchestration backend for AI-assisted software delivery. Receives a user request, coordinates MCP tool calls, LLM reasoning, human approval gates, and autonomous repository execution (via OpenHands) into structured, observable, and resumable workflows.

Sits between the Copilot UI and your external systems. The UI submits a request and polls for results; this service decides what to fetch, what to reason about, what to ask a human, and what code to write.

---

## Table of Contents

- [How to run](#how-to-run)
- [Configuration](#configuration)
  - [LLM](#llm)
  - [MCP integrations](#mcp-integrations)
  - [OpenHands](#openhands)
  - [Full environment variable reference](#full-environment-variable-reference)
- [Workflow definitions](#workflow-definitions)
- [Step type reference](#step-type-reference)
- [Example workflow: Miro → LLM → Jira + Figma → Code](#example-workflow)
- [Current limitations](#current-limitations)

---

## How to run

**1. Start MongoDB**

```bash
docker-compose up -d
```

**2. Install dependencies**

```bash
pip install -e ".[dev]"
```

**3. Configure (see [Configuration](#configuration) below)**

```bash
cp .env.example .env   # then fill in your values
```

**4. Start the server**

```bash
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`. Health check: `GET /health`.

---

## Configuration

### LLM

The `llm` step type drives an agentic tool-calling loop. Configure the provider, model, and API key through environment variables:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `anthropic` or `openai` |
| `LLM_MODEL` | Model name — overrides the provider default (optional) |
| `ANTHROPIC_API_KEY` | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | Required when `LLM_PROVIDER=openai` |

**Anthropic** (default model: `claude-opus-4-6`)

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
# LLM_MODEL=claude-sonnet-4-6   # optional override
```

**OpenAI** (default model: `gpt-4o`)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# LLM_MODEL=gpt-4o-mini   # optional override
```

When `LLM_PROVIDER` is not set the service starts with a no-op stub. Workflows without any `llm` steps run normally; any run that reaches an `llm` step will fail with a clear error message.

### MCP integrations

MCP servers are configured through environment variables and loaded at startup via `McpToolsProvider`. Set one group per integration:

| Variable | Description |
|---|---|
| `MCP_MIRO_ENABLED` | `true` to enable |
| `MCP_MIRO_URL` | MCP server URL (e.g. `https://mcp.miro.com/v1`) |
| `MCP_MIRO_API_KEY` | Bearer token / OAuth token |
| `MCP_MIRO_TRANSPORT` | `streamable_http` (default) or `sse` |
| `MCP_JIRA_ENABLED` | `true` to enable |
| `MCP_JIRA_URL` | e.g. `https://mcp.atlassian.com/v1/sse` |
| `MCP_JIRA_API_KEY` | Atlassian API token |
| `MCP_JIRA_TRANSPORT` | `streamable_http` or `sse` |
| `MCP_FIGMA_ENABLED` | `true` to enable |
| `MCP_FIGMA_URL` | e.g. `https://www.figma.com/api/mcp/v1` |
| `MCP_FIGMA_API_KEY` | Figma personal access token |
| `MCP_FIGMA_TRANSPORT` | `streamable_http` or `sse` |
| `MCP_NOTION_ENABLED` | `true` to enable |
| `MCP_NOTION_URL` | Notion MCP URL |
| `MCP_NOTION_API_KEY` | Notion integration token |
| `MCP_GITHUB_ENABLED` | `true` to enable |
| `MCP_GITHUB_URL` | GitHub MCP URL |
| `MCP_GITHUB_API_KEY` | GitHub personal access token |

Tools exposed by enabled MCP servers are automatically available to `fetch` steps (by tool name) and bound to the LLM in `llm` steps.

### OpenHands

OpenHands is the repository-level execution agent. It receives instructions, opens branches, writes code, runs tests, and creates PRs.

| Variable | Default | Description |
|---|---|---|
| `OPENHANDS_BASE_URL` | `http://openhands:3000` | Base URL of the OpenHands service |
| `OPENHANDS_API_KEY` | — | API key if auth is enabled |
| `OPENHANDS_TIMEOUT_SECONDS` | `60` | Per-task timeout |
| `OPENHANDS_MOCK_MODE` | `true` | If `true`, returns stub results without calling OpenHands (useful for local dev) |

### Full environment variable reference

| Variable | Default | Description |
|---|---|---|
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DATABASE` | `langgraph_backend` | Database name |
| `WORKFLOW_DEFINITIONS_PATH` | `workflows` | Directory containing workflow JSON files |
| `HTTP_ACTION_TIMEOUT_SECONDS` | `30` | Timeout for `http` step requests |
| `LLM_PROVIDER` | — | `anthropic` or `openai` |
| `LLM_MODEL` | provider default | Model name override |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENHANDS_BASE_URL` | `http://openhands:3000` | OpenHands service URL |
| `OPENHANDS_API_KEY` | — | OpenHands auth token |
| `OPENHANDS_TIMEOUT_SECONDS` | `60` | Per-task timeout |
| `OPENHANDS_MOCK_MODE` | `true` | Use stub OpenHands responses |

---

## Workflow definitions

Workflows are JSON files stored in `workflows/` (or the path set by `WORKFLOW_DEFINITIONS_PATH`). They are loaded and validated at startup.

```
workflows/
  feature_flow.json
  mcp_llm_http_flow.json
  llm_tool_call_flow.json
  jira_actions.py        # Python action handlers — auto-loaded at startup
```

The fixed graph topology is:

```
request → fetch_context → llm_agent → plan → approval → run_actions → execute → result
```

Each node runs the steps of its matching type. Nodes with no matching steps in the workflow definition are skipped automatically.

### Resumable runs

When a workflow hits an `approval` step it pauses with `status: waiting_approval`. The API endpoints to drive it forward:

```
POST /api/v1/workflows/runs                    # submit
GET  /api/v1/workflows/runs/{id}               # poll status
POST /api/v1/workflows/runs/{id}/approve       # approve → resumes execution
POST /api/v1/workflows/runs/{id}/reject        # reject → marks run as failed
```

---

## Step type reference

### `fetch` — read data via MCP

Calls a named MCP tool and stores the result in `intermediate_outputs`. Runs in the `fetch_context` node, before the LLM and before any approval gate.

```json
{
  "id": "fetch_miro_board",
  "name": "Fetch Miro Board",
  "type": "fetch",
  "tool": "miro_get_board",
  "tool_input": { "board_id": "uXjVIpExample=" },
  "output_key": "miro_board"
}
```

| Field | Required | Description |
|---|---|---|
| `tool` | yes | MCP tool name (must be exposed by an enabled MCP server) |
| `tool_input` | no | Static input payload forwarded to the tool |
| `output_key` | no | Key under `intermediate_outputs` (defaults to step `id`) |

---

### `llm` — agentic tool-calling loop

Sends the user request to the configured LLM. If the LLM requests a tool call, the tool is executed and the result is fed back. This loop repeats until the LLM returns a final text answer. All MCP tools from enabled integrations are bound to the LLM automatically.

```json
{
  "id": "ask_llm",
  "name": "Ask LLM with Tools",
  "type": "llm",
  "output_key": "agent_result"
}
```

The output stored under `output_key` has the shape:

```json
{
  "response": "Here is the plan...",
  "tool_calls_made": [
    { "name": "miro_get_board", "args": { "board_id": "..." }, "result": { ... } }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `output_key` | no | Key under `intermediate_outputs` (defaults to step `id`) |

---

### `plan` — LLM task decomposition

Produces a `PlanResult` that drives the `execute` node: which repositories to touch and in what order. If the workflow has no `execute` steps, a single default task is created pointing at `metadata.default_repo`.

```json
{
  "id": "plan",
  "name": "Plan Work",
  "type": "plan"
}
```

---

### `approval` — human review gate

Pauses the run (`status: waiting_approval`, `approval_status: pending`) until the client calls `/approve` or `/reject`. Multiple `approval` steps in one workflow are treated as a single gate.

```json
{
  "id": "review_plan",
  "name": "Review Plan",
  "type": "approval",
  "metadata": {
    "description": "Review the generated plan before implementation begins."
  }
}
```

---

### `http` — outbound HTTP request

Sends an HTTP request to any URL. Supports `{{ run.* }}` template variables in `url`, `body`, and `http_headers`. Runs after the approval gate in the `run_actions` node.

```json
{
  "id": "create_jira_ticket",
  "name": "Create Jira Ticket",
  "type": "http",
  "url": "https://your-org.atlassian.net/rest/api/3/issue",
  "method": "POST",
  "http_headers": {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
  },
  "body": {
    "summary": "{{ run.user_request }}",
    "run_id": "{{ run.id }}"
  },
  "output_key": "jira_ticket"
}
```

Available template variables: `{{ run.id }}`, `{{ run.workflow_id }}`, `{{ run.workflow_name }}`, `{{ run.user_request }}`.

| Field | Required | Description |
|---|---|---|
| `url` | yes | Target URL (supports templates) |
| `method` | no | `GET`, `POST`, `PUT`, `PATCH`, or `DELETE` (default: `POST`) |
| `http_headers` | no | Request headers (supports templates) |
| `body` | no | JSON body (supports templates) |
| `output_key` | no | Key under `intermediate_outputs` (defaults to step `id`) |

---

### `action` — Python handler loaded from file

Calls a named async Python function defined in a `.py` file placed alongside your workflow JSON files. Useful for logic that is too complex for a raw HTTP call. Runs in the `run_actions` node (after approval).

At startup the service scans the workflow definitions directory for `*.py` files, imports them, and registers every entry in their `ACTIONS` dict — no application code changes required.

**Step definition** (`workflows/idea_to_code.json`):

```json
{
  "id": "update_ticket_status",
  "name": "Move Ticket to In Progress",
  "type": "action",
  "handler": "jira.transition_issue",
  "handler_input": {
    "issue_key": "PLAT-42",
    "transition": "In Progress"
  },
  "output_key": "jira_transition"
}
```

**Handler file** (`workflows/jira_actions.py`):

```python
async def transition_issue(handler_input: dict, run) -> dict:
    issue_key = handler_input["issue_key"]
    transition = handler_input["transition"]
    # call your internal Jira client here
    return {"transitioned": True, "issue_key": issue_key}

ACTIONS = {
    "jira.transition_issue": transition_issue,
}
```

Drop the file into the same directory as your JSON workflows and restart the server — the handler is picked up automatically.

| Field | Required | Description |
|---|---|---|
| `handler` | yes | Handler name matching a key in an `ACTIONS` dict |
| `handler_input` | no | Static input dict forwarded to the handler (supports `{{ run.* }}` templates) |
| `output_key` | no | Key under `intermediate_outputs` (defaults to step `id`) |

---

### `execute` — repository implementation via OpenHands

Delegates implementation to OpenHands for a specific repository. OpenHands opens a branch, writes or modifies code, runs tests, and creates a PR. One `execute` step corresponds to one repository. Multiple execute steps run sequentially.

```json
{
  "id": "execute_backend",
  "name": "Implement Backend Changes",
  "type": "execute",
  "repo": "your-org/backend",
  "instructions": "Implement the changes described in the plan. Reference the Figma design for UI guidance."
}
```

| Field | Required | Description |
|---|---|---|
| `repo` | yes | Repository in `org/repo` format |
| `instructions` | no | Additional instructions passed to OpenHands (supplements the user request) |

---

### `result` — complete the run

Marks the workflow run as `completed` and finalises the response. Every workflow must have exactly one `result` step.

```json
{
  "id": "result",
  "name": "Produce Result",
  "type": "result"
}
```

---

## Example workflow

The following workflow takes a user idea, reads the Miro board for context, sends it to the LLM for analysis, waits for a human to approve the plan, then creates a Jira ticket and a Figma design file, and finally has OpenHands implement the code.

### Execution graph

```
POST /runs  ─────────────────────────────────────────────────────────────┐
                                                                          │
  [fetch_context]                                                         │
    fetch_miro_board  ──── reads the idea board from Miro via MCP        │
                                                                          │
  [llm_agent]                                                             │
    ask_llm  ──────────── LLM receives Miro data + user request          │
                          may call MCP tools for additional context       │
                          returns structured analysis                     │
                                                                          │
  [approval]                                                              │
    review_analysis  ──── run paused: status = waiting_approval ─────────┘

POST /runs/{id}/approve  ────────────────────────────────────────────────┐
                                                                          │
  [run_actions]  (sequential — see Limitations for parallel support)     │
    create_jira_ticket  ── HTTP POST → Jira REST API                     │
    create_figma_design ── HTTP POST → Figma API                         │
    update_jira_status  ── Python action handler                         │
                                                                          │
  [execute]  (sequential per task)                                        │
    execute_backend  ───── OpenHands implements backend changes           │
    execute_frontend ───── OpenHands implements frontend changes          │
                                                                          │
  status = completed ──────────────────────────────────────────────────── ┘
```

### Workflow definition

```json
{
  "id": "idea_to_code",
  "name": "Idea to Code",
  "description": "Read a Miro board, reason with an LLM, get human approval, then implement.",
  "entrypoint": "fetch_miro_board",
  "metadata": {
    "default_repo": "your-org/backend",
    "outputs_required": ["backend_pr", "frontend_pr"]
  },
  "steps": [
    {
      "id": "fetch_miro_board",
      "name": "Fetch Miro Board",
      "type": "fetch",
      "tool": "miro_get_board",
      "tool_input": { "board_id": "uXjVIpExample=" },
      "output_key": "miro_board"
    },
    {
      "id": "ask_llm",
      "name": "Analyse with LLM",
      "type": "llm",
      "output_key": "analysis",
      "requires": ["fetch_miro_board"]
    },
    {
      "id": "review_analysis",
      "name": "Review Analysis",
      "type": "approval",
      "requires": ["ask_llm"],
      "metadata": {
        "description": "Review the LLM analysis before creating tickets and designs."
      }
    },
    {
      "id": "create_jira_ticket",
      "name": "Create Jira Ticket",
      "type": "http",
      "url": "https://your-org.atlassian.net/rest/api/3/issue",
      "method": "POST",
      "http_headers": { "Authorization": "Bearer <token>", "Content-Type": "application/json" },
      "body": { "summary": "{{ run.user_request }}", "run_id": "{{ run.id }}" },
      "output_key": "jira_ticket",
      "requires": ["review_analysis"]
    },
    {
      "id": "create_figma_design",
      "name": "Create Figma Design File",
      "type": "http",
      "url": "https://api.figma.com/v1/files",
      "method": "POST",
      "http_headers": { "X-Figma-Token": "<token>" },
      "body": { "name": "{{ run.user_request }}", "run_id": "{{ run.id }}" },
      "output_key": "figma_file",
      "requires": ["review_analysis"]
    },
    {
      "id": "update_jira_status",
      "name": "Move Ticket to In Progress",
      "type": "action",
      "handler": "jira.transition_issue",
      "handler_input": { "transition": "In Progress" },
      "output_key": "jira_transition",
      "requires": ["create_jira_ticket"]
    },
    {
      "id": "execute_backend",
      "name": "Implement Backend",
      "type": "execute",
      "repo": "your-org/backend",
      "instructions": "Implement the backend changes from the plan. Use the Figma design file for reference.",
      "requires": ["update_jira_status", "create_figma_design"]
    },
    {
      "id": "execute_frontend",
      "name": "Implement Frontend",
      "type": "execute",
      "repo": "your-org/frontend",
      "instructions": "Implement the frontend matching the Figma design.",
      "requires": ["execute_backend"]
    },
    {
      "id": "result",
      "name": "Produce Result",
      "type": "result",
      "requires": ["execute_frontend"]
    }
  ]
}
```

### What each step does at runtime

| Step | Node | Description |
|---|---|---|
| `fetch_miro_board` | `fetch_context` | Calls the Miro MCP tool and stores board data before planning |
| `ask_llm` | `llm_agent` | Sends user request + Miro data to the LLM; LLM may call further MCP tools |
| `review_analysis` | `approval` | Pauses the run; client calls `/approve` to continue or `/reject` to abort |
| `create_jira_ticket` | `run_actions` | HTTP POST to Jira REST API to create a ticket |
| `create_figma_design` | `run_actions` | HTTP POST to Figma API to create a design file |
| `update_jira_status` | `run_actions` | Calls Python handler from `workflows/jira_actions.py` to transition the ticket |
| `execute_backend` | `execute` | OpenHands implements backend changes, opens branch + PR |
| `execute_frontend` | `execute` | OpenHands implements frontend changes, opens branch + PR |
| `result` | `result` | Aggregates outputs and marks run as `completed` |

---

## Current limitations

### No parallel step execution

The internal LangGraph graph is a linear chain:

```
request → fetch_context → llm_agent → plan → approval → run_actions → execute → result
```

Steps of the same type (e.g. two `fetch` steps, two `http` steps) always run **sequentially** within their node. The `requires` field in step definitions is stored as metadata for documentation purposes but does not change execution order at runtime.

LangGraph itself supports parallel node execution via fan-out edges and the `Send` API — this is a planned addition to this service.

### Single approval gate per run

Each workflow run supports **one** approval pause. If a workflow defines multiple `approval` steps they are all treated as a single gate: the run pauses once, and a single `/approve` or `/reject` call resumes or cancels it.

Workflows requiring separate approval decisions at different stages (e.g. "approve the design" then later "approve the implementation") should be split into multiple sequential workflow runs, where the output of the first becomes the input context of the next.
