from __future__ import annotations

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from app.api.dependencies import get_container
from app.application.run_control import WorkflowDisabledError, ensure_workflow_enabled
from app.core.container import ApplicationContainer
from app.domain.models.graph_run import GraphRun
from app.infrastructure.config.graph_loader import build_runner_from_definition
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
from app.infrastructure.triggers.webhook_validator import valid_bearer, valid_signature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# http trigger auth (``auth_mode`` on the workflow's ``http`` step):
#   "hmac"   — HMAC-SHA256 of the raw body with ``webhook_secret`` (else the
#              global WEBHOOK_SECRET) in the ``signature_header`` header
#              (default X-Webhook-Signature), bare hex or "sha256=<hex>";
#   "bearer" — ``Authorization: Bearer <bearer_token>``;
#   "any"    — either of the two.
_AUTH_MODES = ("hmac", "bearer", "any")


async def _execute_webhook_run(
    runner: YamlGraphRunner,
    run: GraphRun,
    container: ApplicationContainer,
    trigger_payload: dict,
) -> None:
    run.step_statuses = {s["id"]: "pending" for s in runner.steps}
    initial_state = {"request": run.user_request, "trigger_payload": trigger_payload}
    await stream_graph_to_pause(runner, run, container.run_repository, initial_state, base_url=container.settings.base_url)
    if run.status in ("completed", "failed", "cancelled", "rejected"):
        container.live_runners.pop(run.id, None)


@router.post("/{workflow_id}", status_code=202)
async def receive_webhook(
    workflow_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    container: ApplicationContainer = Depends(get_container),
):
    """Receive an authenticated HTTP trigger and start a workflow run.

    How the caller authenticates is the ``http`` trigger step's ``auth_mode``
    (see ``_AUTH_MODES``): an HMAC-SHA256 signature of the raw body in the
    step's ``signature_header`` (default ``X-Webhook-Signature``; bare hex or
    ``sha256=<hex>``; secret from the step's ``webhook_secret``, else the
    global ``WEBHOOK_SECRET``), a bearer token (``Authorization: Bearer
    <bearer_token>``), or either (``any``).

    A field named ``request`` in the JSON body is used as the workflow's initial
    request string; if absent, the entire body is JSON-serialised as the request.
    """
    # Load the workflow first so we can read its per-workflow secret.
    if container.workflow_backend is not None:
        defn = await container.workflow_backend.get(workflow_id)
        if defn is None:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
        runner = build_runner_from_definition(
            defn,
            llm=container.llm,
            llm_factory=container.llm_factory,
            mcp_tools_provider=container.mcp_tools_provider,
            registry=container.yaml_graph_registry,
            run_repository=container.run_repository,
            openhands=container.openhands,
            checkpointer=container.checkpointer,
        )
        runner._callback_base_url = container.settings.agent_callback_url or container.settings.base_url
        definition_snapshot: dict | None = defn.to_raw_dict()
        steps = defn.steps
    else:
        runner = container.yaml_graph_registry.get(workflow_id)
        if runner is None:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
        definition_snapshot = None
        steps = runner.steps

    http_step = next((s for s in steps if s.get("type") == "http"), None)
    auth_mode = (http_step or {}).get("auth_mode", "hmac")

    body = await request.body()

    if auth_mode not in _AUTH_MODES:
        raise HTTPException(status_code=503, detail=f"Unknown auth_mode '{auth_mode}' on the http trigger step")
    token = (http_step or {}).get("bearer_token") or ""
    secret = (http_step or {}).get("webhook_secret") or container.settings.webhook_secret or ""
    use_bearer = auth_mode in ("bearer", "any")
    use_hmac = auth_mode in ("hmac", "any")
    if use_bearer and not use_hmac and not token:
        raise HTTPException(status_code=503, detail="Bearer token not configured on the http trigger step")
    if use_hmac and not use_bearer and not secret:
        raise HTTPException(status_code=503, detail="Webhook secret not configured (set WEBHOOK_SECRET or per-workflow webhook_secret)")
    if auth_mode == "any" and not token and not secret:
        raise HTTPException(status_code=503, detail="Neither a bearer token nor a webhook secret is configured on the http trigger step")
    ok = (use_bearer and valid_bearer(request.headers, token)) or (
        use_hmac and bool(secret)
        and valid_signature(body, request.headers, secret, (http_step or {}).get("signature_header"))
    )
    if not ok:
        logger.warning("Webhook auth (%s) failed for workflow '%s'", auth_mode, workflow_id)
        detail = {
            "bearer": "Invalid or missing Bearer token",
            "hmac": "Invalid or missing webhook signature",
            "any": "Invalid or missing Bearer token or webhook signature",
        }[auth_mode]
        raise HTTPException(status_code=403, detail=detail)

    # Checked only after the signature/token has been verified: an unauthenticated
    # caller must not be able to probe which workflows exist and which are off.
    try:
        ensure_workflow_enabled(runner)
    except WorkflowDisabledError as exc:
        logger.info("HTTP trigger refused: %s", exc.detail)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    try:
        payload: dict = json.loads(body)
    except Exception:
        payload = {"raw": body.decode("utf-8", errors="replace")}

    user_request: str = payload.get("request", "") if isinstance(payload, dict) else ""
    if not user_request:
        user_request = json.dumps(payload)

    thread_id = str(uuid4())
    container.live_runners[thread_id] = runner
    container._inject_runner_dependencies(runner)

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
