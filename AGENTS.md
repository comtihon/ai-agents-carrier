# Backend Build Instructions

## Objective

Build a production-ready backend system based on:

- FastAPI
- LangServe
- LangGraph
- LangChain tools
- OpenHands integration
- Helm deployment
- MongoDB for runtime workflow state

The system must follow clean architecture principles and remain extensible for future AI-driven workflows.

## High-Level Requirements

The system must include:

- FastAPI backend
- LangServe integration layer
- LangGraph orchestration engine
- LangChain-based tool abstraction layer
- OpenHands adapter
- Helm chart in `.helm/`
- workflow definitions delivered through a Helm ConfigMap
- MongoDB persistence for runtime workflow state
- clean, maintainable project structure
- production-quality code

## Architecture Requirements

Follow strict layered architecture:

- API layer for FastAPI routes
- Application layer for use cases and orchestration services
- Domain layer for models, interfaces, and rules
- Infrastructure layer for LangGraph, tools, integrations, persistence, and config loading

### Rules

- Do not mix responsibilities across layers.
- Do not place business logic in API routes.
- Do not directly call external services from the domain layer.
- All integrations must go through adapters.
- Workflow definitions are static deployment artifacts, not runtime-edited entities.
- Runtime workflow state must be persisted in MongoDB.

## Workflow Definition Strategy

Workflow definitions must be stored in the repository and deployed through Helm as ConfigMap-mounted files.

### Source Of Truth

Workflow definition files must live in the repository, for example:

```text
workflows/
  feature_flow.json
  design_first_flow.json
  multi_repo_delivery.json
```

### Delivery Mechanism

Helm must package these workflow definition files into a ConfigMap and mount them into the application container.

### Runtime Behavior

At application startup, the backend must:

- read workflow definition files from the mounted path
- validate file structure
- deserialize them into internal workflow definition models
- register definitions in memory for runtime usage
- fail fast if definitions are invalid

Expected configuration:

```text
WORKFLOW_DEFINITIONS_PATH=/app/workflows
```

### Constraints

- Do not implement an API for editing workflow definitions.
- Do not pull workflow definitions directly from Git at runtime.
- Do not store workflow definitions in MongoDB as the primary source of truth.
- Do not rely on PVC for workflow definition storage.

## Runtime State Strategy

MongoDB must be used for runtime workflow state.

Runtime state includes:

- workflow run id
- current node / current step
- intermediate outputs
- approval status
- execution status
- timestamps
- errors
- linked session, user, and task metadata

MongoDB must not be used as the source of truth for workflow definitions. It is only for live execution state and history.

## Implementation Plan

### Step 1: Project Setup

- Initialize the Python project with `pyproject.toml`
- Add dependencies:
  - `fastapi`
  - `uvicorn`
  - `langgraph`
  - `langchain`
  - `langserve`
  - `pydantic`
  - `pydantic-settings`
  - `pymongo` or `motor`

Create this base structure:

```text
app/
  api/
  application/
  domain/
  infrastructure/
  core/
workflows/
.helm/
```

### Step 2: FastAPI Application

- Implement the application factory pattern
- Add endpoints:
  - `/health`
  - `/ready`
  - `/api/v1/...`
- Use dependency injection
- Keep controllers thin

The FastAPI layer must not contain workflow orchestration logic.

### Step 3: Workflow Definition Loader

Implement a workflow definition loader component.

Responsibilities:

- read JSON workflow files from the mounted path
- validate file structure
- deserialize into internal workflow definition models
- register definitions for runtime access
- expose loaded workflows to application services

If definitions are invalid, startup should fail.

### Step 4: LangGraph Setup

- Define a typed runtime state object
- Implement a minimal graph execution flow

Required flow:

```text
request -> plan -> execute -> result
```

Requirements:

- explicit state
- no hidden memory
- testable nodes
- ability to bind execution to a loaded workflow definition

LangGraph should execute based on loaded workflow definitions and runtime state.

### Step 5: LangServe Integration

Expose LangGraph via LangServe.

Provide endpoints for:

- submitting tasks
- retrieving results
- resuming approval-driven workflows if needed

Ensure compatibility with a Copilot-style frontend.

Do not expose workflow-definition management APIs.

### Step 6: Planning Node

Planning logic input:

- user request
- selected workflow definition
- available context

Planning logic output:

- list of repositories
- tasks per repository
- execution order
- step outputs required by later stages

Use LLM reasoning where needed and return structured output.

### Step 7: Tool Layer

Implement tool abstractions for:

- GitHub
- Jira stub
- Figma stub

Requirements:

- tools must be decoupled
- tools must be replaceable
- no direct API calls inside orchestration nodes without abstraction

If OpenHands is used as the step-level agent, tool access may happen through OpenHands-facing integrations where appropriate, but workflow control remains in the backend.

### Step 8: OpenHands Adapter

Create an adapter class for OpenHands.

Responsibilities:

- start execution session
- send task
- poll for result
- return structured output

Expected output format:

```json
{
  "branch": "...",
  "summary": "...",
  "pr_url": "...",
  "status": "success | failed"
}
```

The adapter must be isolated behind an interface.

### Step 9: Execution Node

Implement a LangGraph node that:

- takes a repository task or workflow step
- calls the OpenHands adapter
- collects the result
- updates runtime state

Support sequential execution for multiple repositories in the MVP.

### Step 10: MongoDB Persistence

Implement a MongoDB persistence layer for runtime state.

Responsibilities:

- create workflow runs
- update workflow state after each step
- store approval state
- store execution outputs
- retrieve workflow run by id
- support workflow resume

Suggested abstractions:

- `WorkflowRunRepository`
- `ApprovalRepository` or approval fields inside workflow run documents
- `ExecutionStateRepository` if separated

Persistence must be behind interfaces and not directly coupled to LangGraph nodes.

### Step 11: MVP Workflow

Implement this minimal working flow:

```text
request
  -> load selected workflow definition
  -> planning
  -> execution (OpenHands)
  -> persist state in MongoDB
  -> return result
```

### Step 12: Helm Chart

Create a Helm chart in `.helm/`.

It must include:

- Deployment
- Service
- ConfigMap for workflow definitions
- environment configuration
- Secret references
- MongoDB connection settings
- probes

Required files:

- `Chart.yaml`
- `values.yaml`
- `values-dev.yaml`
- `values-prod.yaml`
- `templates/`

Helm must package workflow files into a ConfigMap and mount them into the container filesystem. The application must read those definitions from the mounted path at startup.

## Coding Standards

- Use Python type hints everywhere.
- Use Pydantic for schemas and validated config.
- Keep functions small.
- Use clear naming.
- Avoid hardcoded values.
- Use environment-based configuration.
- Separate workflow definition models from runtime workflow state models.

## Design Patterns

Use where appropriate:

- Adapter for external integrations and OpenHands
- Service layer for application logic
- Factory for graph and tool creation
- Strategy for planning logic
- Repository pattern for MongoDB persistence
- DTOs for API contracts

Do not overuse patterns unnecessarily.

## Constraints

- Do not implement everything in one file.
- Do not couple LangGraph directly to external APIs.
- Do not put business logic in FastAPI routes.
- Do not skip abstraction layers.
- Do not fetch workflow definitions from Git at runtime.
- Do not expose workflow-definition CRUD APIs.
- Do not use filesystem persistence as the source of truth for runtime workflow state.

## Definition Of Done

The MVP is complete when:

- the FastAPI server runs
- workflow definitions are loaded from mounted ConfigMap files at startup
- a LangGraph workflow executes
- the planning node produces structured output
- the execution node calls the OpenHands adapter
- runtime workflow state is persisted in MongoDB
- the result is returned via API
- the Helm chart deploys the service, including mounted workflow definitions

## Future Extensions

Do not implement yet:

- workflow editing UI
- workflow definition CRUD API
- parallel multi-repo execution
- advanced approval UI
- full Jira and Figma integration
- deployment automation
- version registry for workflow definitions in the database

## Notes

Focus on:

- correctness
- clean architecture
- extensibility
- strict separation between workflow definition storage and runtime workflow state

Avoid over-engineering beyond the MVP.
