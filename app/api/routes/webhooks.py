from __future__ import annotations

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.api.dependencies import get_container
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.infrastructure.config.graph_loader import build_runner_from_definition
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
from app.infrastructure.triggers.webhook_validator import validate_webhook_signature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

_SIGNATURE_HEADER = "X-Webhook-Signature"


async def _execute_webhook_run(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
    trigger_payload: dict,
) -> None:
    run.step_statuses = {s["id"]: "pending" for s in runner.steps}
    initial_state = {"request": run.user_request, "trigger_payload": trigger_payload}
    await stream_graph_to_pause(runner, run, container.run_repository, initial_state)
    if run.status in ("completed", "failed", "cancelled"):
        container.live_runners.pop(run.id, None)


@router.post("/{workflow_id}", status_code=202)
async def receive_webhook(
    workflow_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    container: ApplicationContainer = Depends(get_container),
):
    """Receive an HMAC-authenticated HTTP trigger and start a workflow run.

    The caller must include a ``X-Webhook-Signature`` header whose value is the
    HMAC-SHA256 hex digest of the raw request body, keyed with ``WEBHOOK_SECRET``.

    A field named ``request`` in the JSON body is used as the workflow's initial
    request string; if absent, the entire body is JSON-serialised as the request.
    """
    secret = container.settings.webhook_secret
    if not secret:
        raise HTTPException(status_code=503, detail="Webhook secret not configured (set WEBHOOK_SECRET)")

    body = await request.body()
    signature = request.headers.get(_SIGNATURE_HEADER, "")
    if not signature or not validate_webhook_signature(body, signature, secret):
        logger.warning("Webhook signature validation failed for workflow '%s'", workflow_id)
        raise HTTPException(status_code=403, detail="Invalid or missing webhook signature")

    try:
        payload: dict = json.loads(body)
    except Exception:
        payload = {"raw": body.decode("utf-8", errors="replace")}

    if container.workflow_backend is not None:
        defn = await container.workflow_backend.get(workflow_id)
        if defn is None:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
        runner = build_runner_from_definition(
            defn,
            llm=container.llm,
            mcp_tools_provider=container.mcp_tools_provider,
            registry=container.yaml_graph_registry,
            run_repository=container.run_repository,
            openhands=container.openhands,
        )
        definition_snapshot: dict | None = defn.to_raw_dict()
    else:
        runner = container.yaml_graph_registry.get(workflow_id)
        if runner is None:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
        definition_snapshot = None

    user_request: str = payload.get("request", "") if isinstance(payload, dict) else ""
    if not user_request:
        user_request = json.dumps(payload)

    thread_id = str(uuid4())
    container.live_runners[thread_id] = runner

    run = GraphRun(
        id=thread_id,
        graph_id=workflow_id,
        user_request=user_request,
        status="running",
        workflow_definition=definition_snapshot,
    )
    await container.run_repository.create(run)
    background_tasks.add_task(_execute_webhook_run, runner, run, container, payload)

    logger.info("HTTP trigger started run %s for workflow '%s'", thread_id, workflow_id)
    return {"run_id": thread_id, "workflow_id": workflow_id, "status": "running"}
