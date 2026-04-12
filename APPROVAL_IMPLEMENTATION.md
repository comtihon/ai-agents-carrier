# Approval Step Implementation Summary

## Overview

This document describes the human-in-the-loop approval functionality that has been implemented in the LangGraph backend.

## What Was Implemented

### 1. Graph-Level Changes (`app/infrastructure/orchestration/graph.py`)

- Added `approval_check` node to the LangGraph workflow
- Implemented conditional routing after the approval check
- Added logic to pause workflow execution when approval is required
- Workflow now follows: `request → fetch_context → plan → approval_check → run_actions → execute → result`

**Key methods:**
- `_approval_check_node()`: Checks if workflow requires approval and pauses if needed
- `_route_after_approval()`: Routes workflow based on approval status (approved → continue, rejected → end)

### 2. Service Layer Changes (`app/application/services/orchestration_service.py`)

Added two new service methods:

- `approve(run_id)`: Approves a workflow in `waiting_approval` status and resumes execution
- `reject(run_id, reason)`: Rejects a workflow and marks it as failed

### 3. API Endpoints (`app/api/routes/workflows.py`)

Added two new REST endpoints:

- `POST /workflows/runs/{run_id}/approve`: Approve and resume workflow
- `POST /workflows/runs/{run_id}/reject`: Reject workflow with optional reason

### 4. Example Workflows

Updated and created workflows demonstrating approval functionality:

- `workflows/design_first_flow.json`: Updated with approval step
- `workflows/jira_task_with_approval.json`: Complete Jira-to-implementation workflow with approval

## How It Works

### Workflow Execution Flow

```
1. User submits workflow request
   ↓
2. Backend executes: fetch_context → plan
   ↓
3. Approval check node:
   - If no approval step in workflow → Continue to run_actions
   - If approval step exists and not yet approved:
     * Set status to "waiting_approval"
     * Set approval_status to "pending"
     * Save state to MongoDB
     * Raise error to pause execution
   ↓
4. User reviews plan via GET /workflows/runs/{run_id}
   ↓
5. User decides:
   a) POST /workflows/runs/{run_id}/approve
      → Resume workflow → run_actions → execute → result
   b) POST /workflows/runs/{run_id}/reject
      → Mark as failed → end
```

### State Management

The `WorkflowRun` model already had the necessary fields:

- `status`: Can be "pending", "running", "waiting_approval", "completed", "failed"
- `approval_status`: Can be "not_required", "pending", "approved", "rejected"

When approval is required:
1. Status changes from "running" to "waiting_approval"
2. Approval status changes to "pending"
3. Approval message is stored in `intermediate_outputs.approval_message`
4. Workflow state is persisted to MongoDB

### Approval Step Definition

In a workflow JSON file:

```json
{
  "id": "review_plan",
  "name": "Review and Approve Plan",
  "type": "approval",
  "requires": ["plan"],
  "metadata": {
    "description": "Human review checkpoint"
  }
}
```

## Integration Points

### CopilotKit UI Integration

The UI should:

1. Poll or listen for workflow status changes
2. When `status === "waiting_approval"`:
   - Display the plan from `workflow_run.plan`
   - Show the approval message from `workflow_run.intermediate_outputs.approval_message`
   - Present "Approve" and "Reject" buttons
3. On approve: Call `POST /workflows/runs/{run_id}/approve`
4. On reject: Call `POST /workflows/runs/{run_id}/reject` with optional reason

### Slack Integration

Can be implemented via webhook or action handler:

```python
# Example: In app/api/app.py or similar
async def notify_slack_for_approval(handler_input: dict, run: WorkflowRun) -> dict:
    # Send Slack message with approval buttons
    # Buttons link back to API endpoints or trigger approval via Slack actions
    pass

# Register the handler
container.action_registry.register("slack.notify_approval", notify_slack_for_approval)
```

Then in workflow definition:

```json
{
  "id": "notify_slack",
  "type": "action",
  "handler": "slack.notify_approval",
  "requires": ["plan"]
}
```

## API Examples

### Submit Workflow with Approval

```bash
curl -X POST http://localhost:8000/workflows/runs \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "design_first_flow",
    "user_request": "Implement user profile page"
  }'
```

Response:
```json
{
  "run": {
    "id": "abc-123",
    "status": "waiting_approval",
    "approval_status": "pending",
    "plan": {
      "summary": "...",
      "tasks": [...]
    },
    "intermediate_outputs": {
      "plan_summary": "...",
      "approval_message": "Please review the plan and approve to continue..."
    }
  }
}
```

### Check Status

```bash
curl http://localhost:8000/workflows/runs/abc-123
```

### Approve

```bash
curl -X POST http://localhost:8000/workflows/runs/abc-123/approve
```

### Reject

```bash
curl -X POST http://localhost:8000/workflows/runs/abc-123/reject \
  -H "Content-Type: application/json" \
  -d '{"reason": "Design needs revision"}'
```

## Testing

All existing tests pass:
- 23 tests in test suite
- No breaking changes to existing functionality
- Workflows without approval steps continue to work as before

## Future Enhancements

Potential improvements:

1. **Multi-step approval**: Support multiple approval points in a workflow
2. **Approval delegation**: Assign approvers based on workflow metadata
3. **Timeout handling**: Auto-reject workflows if not approved within X hours
4. **Approval history**: Track who approved/rejected and when
5. **Conditional approval**: Require approval only if certain conditions are met (e.g., budget > $X)
6. **Plan modification**: Allow users to modify the plan before approval

## Files Modified

1. `app/infrastructure/orchestration/graph.py` - Added approval check node and routing
2. `app/application/services/orchestration_service.py` - Added approve/reject methods
3. `app/api/routes/workflows.py` - Added approve/reject endpoints
4. `workflows/design_first_flow.json` - Added approval step
5. `workflows/jira_task_with_approval.json` - New complete example workflow
6. `README.md` - Updated documentation with approval workflow details
