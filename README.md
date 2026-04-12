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
