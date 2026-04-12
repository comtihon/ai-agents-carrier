# AI Development Orchestration System

## Overview

This project is an AI-powered software delivery orchestration backend.

It provides a system where:

- users interact through a Copilot-style UI
- workflows are orchestrated with LangGraph
- tools and integrations are exposed through LangChain abstractions
- repository-level execution is delegated to OpenHands

The backend is designed for multi-step, multi-repository workflows with persistent runtime state and optional human approval checkpoints.

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

- `plan`
- `execute`
- `approval`
- `result`

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
4. LangGraph plans the work.
5. Execution steps call OpenHands through the adapter.
6. Runtime state is persisted in MongoDB after each step.
7. The final result is returned to the client.

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

The full LangGraph topology is always:

```text
request → fetch_context → plan → run_actions → execute → result
```

| Phase | Step types | Description |
|---|---|---|
| `fetch_context` | `fetch` | Reads external data via MCP tools before planning |
| `plan` | `plan` | LLM decomposes work; result is stored in the run |
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

## Current Status

Current MVP goals:

- workflow loading from mounted files
- LangGraph orchestration
- OpenHands integration
- MongoDB runtime persistence
- Copilot-compatible API surface

## Vision

Build a system where:

- AI plans and executes development work
- humans review and approve when needed
- workflows remain structured, observable, and reproducible
