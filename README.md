# AI Development Orchestration System

## Overview

This project is an AI-powered software delivery orchestration backend.

It provides a system where:

- users interact through a Copilot-style UI
- workflows are orchestrated with LangGraph
- tools and integrations are exposed through LangChain abstractions
- repository-level execution is delegated to OpenHands

The backend is designed for multi-step, multi-repository workflows with persistent runtime state and optional human approval checkpoints.

### Example Use Case: Jira Task to Implementation

```text
1. User asks: "Implement Jira task PROJ-123"
   ↓
2. Backend fetches Jira ticket details via Jira MCP
   ↓
3. LLM analyzes requirements and checks if Figma design exists
   ↓
4. If design required → Figma MCP creates/fetches design spec
   ↓
5. LLM creates high-level solution design and implementation plan
   ↓
6. ⏸️  APPROVAL CHECKPOINT - Execution pauses
   - Plan is shown to user via CopilotKit UI or Slack
   - User reviews the plan and design
   - User clicks "Approve" or "Reject"
   ↓
7. If approved → Continue execution
   - OpenHands implements the code changes
   - Creates PR(s) in the target repository(ies)
   - Updates Jira ticket status
   ↓
8. Results returned to user with PR links
```

This workflow combines:
- **External integrations** (Jira, Figma) via MCP
- **AI planning** with LLM-powered solution design
- **Human approval** for quality control
- **Automated execution** via OpenHands

## Core Stack

- FastAPI for the HTTP API
- LangServe for exposing runnable workflows
- LangGraph for orchestration
- LangChain for tool abstraction
- OpenHands for repository execution
- MongoDB for runtime workflow state
- Helm for deployment and workflow-definition delivery

## Architecture

```text
Copilot UI / Client
        |
        v
FastAPI + LangServe
        |
        v
Application Services
        |
        v
LangGraph Orchestrator
        |
        v
OpenHands Adapter + Tool Adapters
        |
        v
External Systems
```

## Workflow Model

### Workflow Definitions

Workflow definitions are static deployment artifacts.

- Stored in the repository under `workflows/`
- Versioned with Git
- Packaged by Helm into a ConfigMap
- Mounted into the application container filesystem
- Loaded and validated at application startup

Example repository layout:

```text
workflows/
  feature_flow.json
  multi_repo_delivery.json
```

### Runtime Workflow State

Runtime state is stored in MongoDB and includes:

- workflow run id
- current node / current step
- intermediate outputs
- approval status
- execution status
- timestamps
- errors
- linked session, user, and task metadata

MongoDB is used for live execution state and history only. It is not the source of truth for workflow definitions.

## Expected Graph Format

The backend currently expects workflow definitions as JSON documents with this shape:

```json
{
  "id": "multi_repo_delivery",
  "name": "Multi Repository Delivery",
  "description": "Plan and execute coordinated work across repositories.",
  "entrypoint": "plan",
  "metadata": {
    "default_repo": "acme/backend",
    "outputs_required": ["backend_pr", "frontend_pr"]
  },
  "steps": [
    {
      "id": "plan",
      "name": "Plan Work",
      "type": "plan"
    },
    {
      "id": "execute_backend",
      "name": "Implement Backend Changes",
      "type": "execute",
      "repo": "acme/backend",
      "instructions": "Implement backend changes required by the request.",
      "requires": ["plan"]
    },
    {
      "id": "execute_frontend",
      "name": "Implement Frontend Changes",
      "type": "execute",
      "repo": "acme/frontend",
      "instructions": "Implement frontend changes after backend outputs are available.",
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

Supported step types:

- `plan` - LLM-based planning step
- `execute` - Repository-level execution via OpenHands
- `approval` - Human review checkpoint (pauses execution until approved/rejected)
- `result` - Final aggregation step
- `fetch` - Retrieve data via MCP tools
- `http` - Execute HTTP calls
- `action` - Run registered Python handlers

Validation rules:

- `id` must be unique across workflow files
- each step `id` must be unique within a workflow
- `entrypoint` must reference an existing step
- each `requires` dependency must reference an existing step

### Expected Execution Shape

The MVP graph executes in this order:

```text
request -> plan -> execute -> result
```

For a multi-repository workflow, the logical dependency graph is expected to look like:

```text
plan
  |
  v
execute_backend
  |
  v
execute_frontend
  |
  v
result
```

The current MVP executes repository tasks sequentially.

## Responsibilities

### LangGraph

- workflow orchestration
- state transitions
- task decomposition
- approval checkpoints
- execution routing

### OpenHands

- step-level execution agent
- repository interaction
- code generation and modification
- test execution
- branch and PR creation

### LangChain Tools

- GitHub integration
- Jira integration stub
- Figma integration stub
- future infrastructure tooling

### LangServe

- exposes workflows as runnable endpoints
- integrates LangGraph into FastAPI
- supports task submission and result retrieval

## Repository Structure

```text
/
|-- app/
|-- workflows/
|-- tests/
|-- .helm/
|-- README.md
|-- AGENTS.md
```

## Deployment

Deployment is managed with Helm.

- Chart location: `.helm/`
- Workflow files are packaged into a ConfigMap
- Workflow ConfigMap is mounted into the application container
- The service loads definitions from `WORKFLOW_DEFINITIONS_PATH`

### Runtime Configuration

Primary settings are provided through environment variables:

- `WORKFLOW_DEFINITIONS_PATH`
- `MONGODB_URI`
- `MONGODB_DATABASE`
- `OPENHANDS_BASE_URL`
- `OPENHANDS_API_KEY`
- `OPENHANDS_MOCK_MODE`

## Execution Flow

1. User submits a request through the client.
2. FastAPI or LangServe receives the request.
3. The backend loads the selected workflow definition.
4. LangGraph orchestrates the workflow:
   - **fetch_context**: Retrieves external data via MCP tools
   - **plan**: LLM creates execution plan
   - **approval_check**: If workflow has an approval step, execution pauses for human review
   - **run_actions**: Executes HTTP calls and Python action handlers
   - **execute**: OpenHands implements code changes in repositories
   - **result**: Aggregates outputs and finalizes the workflow
5. Runtime state is persisted in MongoDB after each step.
6. The final result is returned to the client.

### Human-in-the-Loop Approval

When a workflow definition includes a step of type `approval`, the execution will pause after the planning phase and wait for explicit user approval before proceeding to execution.

**Workflow behavior:**
- After the plan is generated, the workflow status changes to `waiting_approval`
- The approval status is set to `pending`
- The workflow run is saved to MongoDB with the current state
- An error message is returned indicating that approval is required

**User actions:**
- **Approve**: `POST /workflows/runs/{run_id}/approve` - Continues execution
- **Reject**: `POST /workflows/runs/{run_id}/reject` - Marks workflow as failed and stops execution

**Example approval flow:**

```bash
# 1. Submit workflow with approval step
curl -X POST http://localhost:8000/workflows/runs \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "design_first_flow",
    "user_request": "Implement user profile page"
  }'

# Response includes run_id and status "waiting_approval"
# {
#   "run": {
#     "id": "abc-123",
#     "status": "waiting_approval",
#     "approval_status": "pending",
#     "intermediate_outputs": {
#       "plan_summary": "...",
#       "approval_message": "Please review the plan..."
#     }
#   }
# }

# 2. Review the plan via GET endpoint
curl http://localhost:8000/workflows/runs/abc-123

# 3a. Approve the plan to continue execution
curl -X POST http://localhost:8000/workflows/runs/abc-123/approve

# OR

# 3b. Reject the plan with optional reason
curl -X POST http://localhost:8000/workflows/runs/abc-123/reject \
  -H "Content-Type: application/json" \
  -d '{"reason": "Design needs revision"}'
```

## MCP-Powered Workflow Example

The following example shows an end-to-end workflow that pulls context from Miro and Jira,
creates a Figma design spec, decomposes work into Jira tickets, then implements them
using OpenHands — all driven by a single workflow definition.

### Graph

```text
┌─────────────────────────────────────────────────────────────────────┐
│  fetch_context  (type: fetch — reads data from MCP integrations)    │
│                                                                      │
│   fetch_miro_board ─────────────────────────────────────────────┐   │
│                                                                  │   │
│   fetch_jira_epic  ─────────────────────────────────────────────┤   │
└──────────────────────────────────────────────────────────────── ┼ ──┘
                                                                  │
                                                                  ▼
                                                       create_figma_design  (type: fetch)
                                                                  │
                                                                  ▼
                                                              plan
                                                         (LLM decomposes work,
                                                          creates Jira tickets)
                                                                  │
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  run_actions  (type: http / action — side-effects after planning)   │
│                                                                      │
│   notify_internal  (http POST → internal notification service)      │
│                                                                      │
│   transition_jira  (action → Python handler updates ticket state)   │
└──────────────────────────────────────────────────────────────────── ┘
                                                                  │
                                                                  ▼
                                                   ┌──── execute_backend ────┐
                                                   │                         │
                                                   └──── execute_frontend ───┘
                                                                  │
                                                                  ▼
                                                              result
```

The full LangGraph topology is:

```text
request → fetch_context → plan → approval_check → run_actions → execute → result
                                       ↓
                                  (conditional)
                                       ↓
                            waiting_approval (pause)
                                       ↓
                            user approve/reject
                                       ↓
                            continue or fail
```

| Phase | Step types | Description |
|---|---|---|
| `fetch_context` | `fetch` | Reads external data via MCP tools before planning |
| `plan` | `plan` | LLM decomposes work; result is stored in the run |
| `approval_check` | `approval` | Human review checkpoint - pauses if approval step exists in workflow |
| `run_actions` | `http`, `action` | Side-effects after planning: HTTP calls, Python handlers |
| `execute` | `execute` | OpenHands implements code in each repository |

### Workflow Definition

```json
{
  "id": "idea_to_implementation",
  "name": "Idea to Implementation",
  "description": "Turn a Miro board idea into implemented code via Figma and Jira.",
  "entrypoint": "plan",
  "metadata": {
    "default_repo": "acme/backend",
    "outputs_required": ["backend_pr", "frontend_pr"]
  },
  "steps": [
    {
      "id": "fetch_miro_board",
      "name": "Fetch Miro Board",
      "type": "fetch",
      "tool": "miro_get_board",
      "tool_input": { "board_id": "uXjVIpExample=" },
      "output_key": "miro_board",
      "requires": []
    },
    {
      "id": "fetch_jira_epic",
      "name": "Fetch Jira Epic",
      "type": "fetch",
      "tool": "jira_get_issue",
      "tool_input": { "issue_key": "PLAT-42" },
      "output_key": "jira_epic",
      "requires": []
    },
    {
      "id": "create_figma_design",
      "name": "Create Figma Design Spec",
      "type": "fetch",
      "tool": "figma_create_file",
      "tool_input": { "project_id": "123456789" },
      "output_key": "figma_design",
      "requires": ["fetch_miro_board"]
    },
    {
      "id": "plan",
      "name": "Plan Work and Create Jira Tickets",
      "type": "plan",
      "requires": ["fetch_miro_board", "fetch_jira_epic", "create_figma_design"]
    },
    {
      "id": "notify_internal",
      "name": "Notify Internal Service",
      "type": "http",
      "url": "https://internal.example.com/delivery/started",
      "method": "POST",
      "body": {
        "run_id": "{{ run.id }}",
        "workflow": "{{ run.workflow_name }}",
        "request": "{{ run.user_request }}"
      },
      "output_key": "notification_result",
      "requires": ["plan"]
    },
    {
      "id": "transition_jira",
      "name": "Move Jira Epic to In Progress",
      "type": "action",
      "handler": "jira.transition_issue",
      "handler_input": {
        "issue_key": "PLAT-42",
        "transition": "In Progress",
        "run_id": "{{ run.id }}"
      },
      "output_key": "jira_transition_result",
      "requires": ["plan"]
    },
    {
      "id": "execute_backend",
      "name": "Implement Backend Changes",
      "type": "execute",
      "repo": "acme/backend",
      "instructions": "Implement backend changes from the plan. Reference the Figma design and Jira epic for context.",
      "requires": ["plan"]
    },
    {
      "id": "execute_frontend",
      "name": "Implement Frontend Changes",
      "type": "execute",
      "repo": "acme/frontend",
      "instructions": "Implement frontend changes matching the Figma design spec.",
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

### What each node does

| Node | Type | Integration | Description |
|---|---|---|---|
| `fetch_miro_board` | `fetch` | Miro MCP | Reads the idea board — shapes, stickies, and structure become context for the planner |
| `fetch_jira_epic` | `fetch` | Jira MCP | Reads the parent epic — acceptance criteria and labels flow into planning |
| `create_figma_design` | `fetch` | Figma MCP | Creates a design file in the target project using the Miro board content as input |
| `plan` | `plan` | LLM | Decomposes work into repository tasks and creates Jira sub-tickets under the epic |
| `notify_internal` | `http` | Internal API | POST to the internal notification service with `{{ run.id }}` and plan summary |
| `transition_jira` | `action` | Python handler | Calls registered handler `jira.transition_issue` to move the epic to In Progress |
| `execute_backend` | `execute` | OpenHands | Implements backend changes, opens a branch and PR |
| `execute_frontend` | `execute` | OpenHands | Implements frontend changes against the Figma spec, opens a branch and PR |
| `result` | `result` | — | Aggregates PR URLs and a summary into the workflow run response |

### Step type reference

#### `http` — outbound HTTP call

Sends a request to any URL. Supports `{{ run.* }}` templates in `body`, `http_headers`, and `url`.

```json
{
  "id": "notify",
  "type": "http",
  "url": "https://internal.example.com/notify",
  "method": "POST",
  "body": { "run_id": "{{ run.id }}", "workflow": "{{ run.workflow_name }}" },
  "http_headers": { "X-Source": "acme" },
  "output_key": "notification_result"
}
```

Available template variables: `{{ run.id }}`, `{{ run.workflow_id }}`, `{{ run.workflow_name }}`, `{{ run.user_request }}`.

#### `action` — registered Python handler

Calls a named Python function registered in `ActionRegistry` at startup. The handler receives the resolved `handler_input` dict and the live `WorkflowRun`.

```json
{
  "id": "transition_jira",
  "type": "action",
  "handler": "jira.transition_issue",
  "handler_input": { "issue_key": "PLAT-42", "transition": "In Progress" },
  "output_key": "jira_transition_result"
}
```

Register the handler in application startup (e.g. in `app/api/app.py`):

```python
from app.infrastructure.actions.registry import ActionRegistry
from app.domain.models.runtime import WorkflowRun

async def transition_jira_issue(handler_input: dict, run: WorkflowRun) -> dict:
    issue_key = handler_input["issue_key"]
    transition = handler_input["transition"]
    # ... call your internal Jira client ...
    return {"transitioned": True, "issue_key": issue_key}

container.action_registry.register("jira.transition_issue", transition_jira_issue)
```

#### `approval` — human review checkpoint

Pauses workflow execution after the planning phase and requires explicit user approval before proceeding to execution. This enables human-in-the-loop workflows where plans can be reviewed and approved before code changes are implemented.

```json
{
  "id": "review_plan",
  "name": "Review and Approve Plan",
  "type": "approval",
  "requires": ["plan"],
  "metadata": {
    "description": "Human review checkpoint - user must approve the plan before execution continues"
  }
}
```

**Behavior:**
- When an `approval` step is present in the workflow, execution pauses after planning
- Workflow status changes to `waiting_approval`
- Approval status is set to `pending`
- User must call `POST /workflows/runs/{run_id}/approve` to continue
- Or call `POST /workflows/runs/{run_id}/reject` to cancel execution

**Integration points:**
- **CopilotKit UI**: Display the plan and show approve/reject buttons
- **Slack notifications**: Send message with plan summary and action buttons
- **Custom webhooks**: Notify external systems when approval is required

**Example workflow with approval:**

```json
{
  "id": "jira_task_flow",
  "name": "Jira Task Implementation",
  "steps": [
    {
      "id": "fetch_jira",
      "type": "fetch",
      "tool": "jira_get_issue",
      "tool_input": { "issue_key": "PROJ-123" }
    },
    {
      "id": "plan",
      "type": "plan",
      "requires": ["fetch_jira"]
    },
    {
      "id": "approval",
      "type": "approval",
      "requires": ["plan"]
    },
    {
      "id": "execute",
      "type": "execute",
      "repo": "myorg/myrepo",
      "requires": ["approval"]
    },
    {
      "id": "result",
      "type": "result",
      "requires": ["execute"]
    }
  ]
}
```

### Required environment configuration

Enable the integrations in your Helm values and point to the right secret:

```yaml
# values-prod.yaml
app:
  existingSecret: "langgraph-backend-secrets-prod"

mcp:
  miro:
    enabled: "true"
    url: "https://mcp.miro.com/v1"
  jira:
    enabled: "true"
    url: "https://mcp.atlassian.com/v1/sse"
  figma:
    enabled: "true"
    url: "https://www.figma.com/api/mcp/v1"
```

The corresponding k8s Secret must contain:

```
MCP_MIRO_API_KEY=<miro-oauth-token>
MCP_JIRA_API_KEY=<atlassian-api-token>
MCP_FIGMA_API_KEY=<figma-personal-access-token>
OPENHANDS_API_KEY=<openhands-api-key>
```

## API Reference

### Workflow Endpoints

#### Submit a workflow

```
POST /workflows/runs
```

**Request body:**
```json
{
  "workflow_id": "design_first_flow",
  "user_request": "Implement user authentication",
  "session_id": "optional-session-id",
  "user_id": "optional-user-id",
  "context": {}
}
```

**Response:**
```json
{
  "run": {
    "id": "abc-123",
    "workflow_id": "design_first_flow",
    "status": "running|waiting_approval|completed|failed",
    "approval_status": "not_required|pending|approved|rejected",
    "plan": { ... },
    "execution_results": [],
    ...
  }
}
```

#### Get workflow run status

```
GET /workflows/runs/{run_id}
```

Returns the current state of a workflow run, including status, plan, and any intermediate outputs.

#### Approve a workflow run

```
POST /workflows/runs/{run_id}/approve
```

Approves a workflow that is in `waiting_approval` status and resumes execution.

**Response:** Returns the updated workflow run that will continue execution.

#### Reject a workflow run

```
POST /workflows/runs/{run_id}/reject
```

**Request body (optional):**
```json
{
  "reason": "Design needs revision before implementation"
}
```

Rejects a workflow in `waiting_approval` status and marks it as failed.

#### Resume a workflow run

```
POST /workflows/runs/{run_id}/resume
```

Resumes a previously paused or failed workflow run. Generally used for manual recovery scenarios.

## Current Status

Current MVP goals:

- workflow loading from mounted files
- LangGraph orchestration with human-in-the-loop approval
- OpenHands integration
- MongoDB runtime persistence
- Copilot-compatible API surface

## Vision

Build a system where:

- AI plans and executes development work
- humans review and approve when needed
- workflows remain structured, observable, and reproducible
