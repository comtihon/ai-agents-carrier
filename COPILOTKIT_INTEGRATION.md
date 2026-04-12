# CopilotKit Integration Guide for Approval Workflow

## Overview

This guide shows how to integrate the approval workflow with CopilotKit UI for a seamless user experience.

## User Flow

```
1. User types: "Implement Jira task PROJ-123"
   ↓
2. Backend fetches Jira details, creates plan
   ↓
3. UI receives workflow status: "waiting_approval"
   ↓
4. CopilotKit displays:
   - Plan summary
   - Figma design preview (if available)
   - Estimated changes
   - [Approve] [Reject] buttons
   ↓
5. User clicks [Approve]
   ↓
6. Backend continues execution
   ↓
7. UI shows: "Implementing changes..."
   ↓
8. UI displays PR link when complete
```

## API Integration

### 1. Submit Workflow

```typescript
// In your CopilotKit backend integration
async function submitWorkflow(userRequest: string) {
  const response = await fetch('http://backend/workflows/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      workflow_id: 'jira_task_with_approval',
      user_request: userRequest,
      session_id: getCurrentSessionId(),
      user_id: getCurrentUserId(),
    })
  });
  
  const data = await response.json();
  return data.run;
}
```

### 2. Poll for Status Updates

```typescript
async function pollWorkflowStatus(runId: string) {
  const response = await fetch(`http://backend/workflows/runs/${runId}`);
  const data = await response.json();
  return data.run;
}

// Poll every 2 seconds
const interval = setInterval(async () => {
  const run = await pollWorkflowStatus(runId);
  
  if (run.status === 'waiting_approval') {
    // Show approval UI
    showApprovalDialog(run);
    clearInterval(interval);
  } else if (run.status === 'completed') {
    // Show results
    showResults(run);
    clearInterval(interval);
  } else if (run.status === 'failed') {
    // Show error
    showError(run);
    clearInterval(interval);
  }
}, 2000);
```

### 3. Approve Workflow

```typescript
async function approveWorkflow(runId: string) {
  const response = await fetch(`http://backend/workflows/runs/${runId}/approve`, {
    method: 'POST'
  });
  
  const data = await response.json();
  // Resume polling or wait for completion
  return data.run;
}
```

### 4. Reject Workflow

```typescript
async function rejectWorkflow(runId: string, reason?: string) {
  const response = await fetch(`http://backend/workflows/runs/${runId}/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason })
  });
  
  const data = await response.json();
  return data.run;
}
```

## UI Components

### Approval Dialog Component (React)

```tsx
import { useState } from 'react';

interface ApprovalDialogProps {
  workflowRun: WorkflowRun;
  onApprove: () => void;
  onReject: (reason: string) => void;
}

function ApprovalDialog({ workflowRun, onApprove, onReject }: ApprovalDialogProps) {
  const [showRejectReason, setShowRejectReason] = useState(false);
  const [rejectReason, setRejectReason] = useState('');

  const plan = workflowRun.plan;
  const approvalMessage = workflowRun.intermediate_outputs?.approval_message;

  return (
    <div className="approval-dialog">
      <h2>Review Implementation Plan</h2>
      
      <div className="approval-message">
        {approvalMessage}
      </div>

      <div className="plan-summary">
        <h3>Summary</h3>
        <p>{plan?.summary}</p>
      </div>

      {plan?.tasks && (
        <div className="tasks">
          <h3>Tasks ({plan.tasks.length})</h3>
          <ul>
            {plan.tasks.map((task, idx) => (
              <li key={idx}>
                <strong>{task.repo}</strong>: {task.instructions}
              </li>
            ))}
          </ul>
        </div>
      )}

      {workflowRun.intermediate_outputs?.figma_design && (
        <div className="figma-preview">
          <h3>Design Reference</h3>
          <a href={workflowRun.intermediate_outputs.figma_design.url} target="_blank">
            View in Figma
          </a>
        </div>
      )}

      <div className="actions">
        {!showRejectReason ? (
          <>
            <button 
              onClick={onApprove}
              className="approve-btn"
            >
              ✓ Approve & Continue
            </button>
            <button 
              onClick={() => setShowRejectReason(true)}
              className="reject-btn"
            >
              ✗ Reject
            </button>
          </>
        ) : (
          <div className="reject-form">
            <textarea
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              placeholder="Reason for rejection (optional)"
              rows={3}
            />
            <button onClick={() => onReject(rejectReason)}>
              Submit Rejection
            </button>
            <button onClick={() => setShowRejectReason(false)}>
              Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
```

### Integration with CopilotKit Chat

```tsx
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import { useState } from "react";

function App() {
  const [currentWorkflowRun, setCurrentWorkflowRun] = useState(null);
  const [showApprovalDialog, setShowApprovalDialog] = useState(false);

  const handleApprove = async () => {
    await approveWorkflow(currentWorkflowRun.id);
    setShowApprovalDialog(false);
    // Resume polling
  };

  const handleReject = async (reason: string) => {
    await rejectWorkflow(currentWorkflowRun.id, reason);
    setShowApprovalDialog(false);
  };

  return (
    <CopilotKit>
      <div className="app">
        <CopilotChat
          labels={{
            title: "AI Development Assistant",
            initial: "Hi! Ask me to implement any Jira task.",
          }}
          onMessage={async (message) => {
            // Submit workflow
            const run = await submitWorkflow(message);
            setCurrentWorkflowRun(run);
            
            // Start polling
            const interval = setInterval(async () => {
              const updatedRun = await pollWorkflowStatus(run.id);
              setCurrentWorkflowRun(updatedRun);
              
              if (updatedRun.status === 'waiting_approval') {
                setShowApprovalDialog(true);
                clearInterval(interval);
              } else if (updatedRun.status === 'completed') {
                clearInterval(interval);
              }
            }, 2000);
          }}
        />

        {showApprovalDialog && currentWorkflowRun && (
          <ApprovalDialog
            workflowRun={currentWorkflowRun}
            onApprove={handleApprove}
            onReject={handleReject}
          />
        )}
      </div>
    </CopilotKit>
  );
}
```

## Webhook Alternative (Server-Sent Events)

For real-time updates without polling:

### Backend (FastAPI)

```python
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
import asyncio

router = APIRouter()

@router.get("/workflows/runs/{run_id}/events")
async def workflow_events(run_id: str):
    async def event_generator():
        while True:
            run = await workflow_run_repository.get_by_id(run_id)
            
            yield {
                "event": "workflow_update",
                "data": run.model_dump_json()
            }
            
            if run.status in ["completed", "failed", "waiting_approval"]:
                break
                
            await asyncio.sleep(2)
    
    return EventSourceResponse(event_generator())
```

### Frontend

```typescript
function useWorkflowEvents(runId: string) {
  const [run, setRun] = useState(null);

  useEffect(() => {
    const eventSource = new EventSource(
      `http://backend/workflows/runs/${runId}/events`
    );

    eventSource.addEventListener('workflow_update', (event) => {
      const run = JSON.parse(event.data);
      setRun(run);
    });

    return () => eventSource.close();
  }, [runId]);

  return run;
}
```

## Slack Integration

For Slack notifications when approval is required:

```python
# In app/api/app.py or similar

from slack_sdk.webhook import WebhookClient

async def notify_slack_approval(handler_input: dict, run: WorkflowRun) -> dict:
    webhook = WebhookClient(os.getenv("SLACK_WEBHOOK_URL"))
    
    response = webhook.send(
        text=f"Workflow {run.id} needs approval",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Approval Required*\n\n{run.intermediate_outputs.get('approval_message', '')}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Plan Summary*\n{run.plan.summary if run.plan else 'N/A'}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "url": f"{os.getenv('FRONTEND_URL')}/workflows/{run.id}/approve"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "url": f"{os.getenv('FRONTEND_URL')}/workflows/{run.id}/reject"
                    }
                ]
            }
        ]
    )
    
    return {"notified": True, "channel": response}

# Register the handler
action_registry.register("slack.notify_approval", notify_slack_approval)
```

Then in your workflow definition:

```json
{
  "id": "notify_slack_approval",
  "name": "Notify via Slack",
  "type": "action",
  "handler": "slack.notify_approval",
  "requires": ["plan"]
}
```

## Best Practices

1. **Error Handling**: Always handle network errors and show user-friendly messages
2. **Loading States**: Show spinners while waiting for backend responses
3. **Timeout**: Set reasonable timeouts for approval (e.g., 24 hours)
4. **Persistence**: Store workflow run ID in localStorage to recover state on page reload
5. **Notifications**: Consider browser notifications when approval is required
6. **Accessibility**: Ensure approval dialogs are keyboard-navigable and screen-reader friendly

## Example User Experience

```
User: "Implement JIRA-123"
Agent: "Fetching JIRA-123 details..."
       [2 seconds later]
Agent: "I've analyzed the ticket and created a design. Here's the implementation plan:
       
       **Summary**: Implement user profile page with avatar upload
       
       **Changes**:
       - Backend: Add /api/users/profile endpoint
       - Backend: Add avatar upload to S3
       - Frontend: Create ProfilePage component
       - Frontend: Add image upload widget
       
       **Estimated time**: ~3 hours
       **Files changed**: ~8 files
       
       [Approve] [Reject]"

User: [clicks Approve]
Agent: "Great! Starting implementation..."
       [5 minutes later]
Agent: "✓ Implementation complete!
       
       **Pull Requests**:
       - Backend: https://github.com/myorg/backend/pull/456
       - Frontend: https://github.com/myorg/frontend/pull/789
       
       **Next Steps**:
       - Review the PRs
       - Run tests in CI
       - Deploy to staging
       
       Would you like me to notify the team?"
```