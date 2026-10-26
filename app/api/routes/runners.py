"""Runners: machines connected with acp-web-proxy (see app.application.runners).

Public (no user token — a machine has no credential yet, or carries its own):

    GET    /acp/runners/info                      discovery for `acp-web-proxy install`
    GET    /acp/runners/install.md                instructions for an AI agent
    POST   /acp/runners/enrollments               → {enrollment_id, poll_token, code, …}
    GET    /acp/runners/enrollments/{id}/poll     Bearer <poll_token>
    DELETE /acp/runners/self                      Bearer <runner token>
    WS     /acp/runners/connect                   control socket, Bearer <runner token>
    WS     /acp/runners/connect/connections/{id}  data socket, Bearer <runner token>

Signed-in users:

    POST   /acp/runners/enrollments/lookup        {code} → the waiting machine
    POST   /acp/runners/enrollments/{id}/approve  {code, name?, visibility?, team_id?, agents?}
    POST   /acp/runners/enrollments/{id}/deny     {code}
    GET    /acp/runners                           runners the caller can see
    PATCH  /acp/runners/{id}                      {name?, visibility?, team_id?}
    DELETE /acp/runners/{id}                      revoke
    POST   /acp/runners/{id}/agents               {acp_agent, id?, name?} → agent definition
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from app.api.dependencies import get_container
from app.application.runners import (
    EnrollmentError,
    EnrollmentService,
    InboundSocket,
    RunnerSession,
    authenticate_runner,
    clean_agents,
    get_runner_hub,
    utcnow,
)
from app.application.visibility import (
    can_change_visibility,
    can_edit,
    can_manage,
    can_see,
    ownership_fields,
    unwrap,
    visibility_kwargs,
)
from app.core.config import get_settings
from app.core.container import ApplicationContainer
from app.domain.models.agent_definition import AgentDefinition
from app.domain.models.runner import Runner
from app.infrastructure.auth.authorization import get_current_subject

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/acp/runners", tags=["runners"])

_services: dict[int, EnrollmentService] = {}


def _backend(container: ApplicationContainer) -> Any:
    backend = getattr(container, "runner_backend", None)
    if backend is None or not get_settings().acp_runners_enabled:
        raise HTTPException(status_code=404, detail="Runners are not enabled")
    return backend


def _service(container: ApplicationContainer) -> EnrollmentService:
    backend = _backend(container)
    svc = _services.get(id(backend))
    if svc is None:
        svc = _services[id(backend)] = EnrollmentService(backend, ttl_s=get_settings().acp_runner_enrollment_ttl_s)
    return svc


def _http(err: EnrollmentError) -> HTTPException:
    return HTTPException(status_code=err.status, detail=err.detail)


def _client_ip(request: Request) -> str | None:
    """The address the approver is shown and enrollments are rate-limited by.

    The last X-Forwarded-For hop is the one our own ingress added; earlier
    entries come from the client and could name any address.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else None


def _user_key(request: Request) -> str:
    subject = get_current_subject()
    return f"user:{subject.id}" if subject is not None and subject.id else f"ip:{_client_ip(request)}"


def _require_enroller() -> None:
    """WRITE (checked by the middleware) plus, when configured, an enroll role."""
    roles = get_settings().acp_runner_enroll_roles
    subject = get_current_subject()
    if not roles or subject is None or subject.is_super_admin:
        return
    if not (set(subject.roles or ()) & set(roles)):
        raise HTTPException(status_code=403, detail="You may not connect machines to carrier")


def _check_scope(visibility: str, team_id: str | None) -> str | None:
    """The team a runner may be shared with by the caller (None unless "team")."""
    if visibility != "team":
        return None
    subject = get_current_subject()
    if subject is None:
        return team_id
    team = team_id or subject.team_id
    if not team:
        raise HTTPException(status_code=400, detail="You are not in a team")
    if team != subject.team_id and not subject.is_super_admin:
        raise HTTPException(status_code=403, detail="You may only share with your own team")
    return team


def _present_enrollment(e: Any) -> dict[str, Any]:
    return {
        "id": e.id,
        "hostname": e.hostname,
        "os": e.os,
        "arch": e.arch,
        "tz": e.tz,
        "version": e.version,
        "agents": e.agents,
        "default_agent": e.default_agent,
        "request_ip": e.request_ip,
        "created_at": e.created_at.isoformat(),
        "expires_at": e.expires_at.isoformat(),
    }


async def _runner_agents(container: ApplicationContainer, runner_id: str) -> list[AgentDefinition]:
    if container.agent_backend is None:
        return []
    return [a for a in await container.agent_backend.list() if a.runner_id == runner_id]


async def _present(container: ApplicationContainer, runner: Runner) -> dict[str, Any]:
    hub = get_runner_hub()
    data = runner.model_dump(mode="json", exclude={"secret_hash", "limits", "can_edit", "team_name"})
    now = utcnow()
    data["limits"] = {
        agent: limit.model_dump(mode="json")
        for agent, limit in runner.limits.items()
        if limit.until is None or limit.until > now
    }
    data["online"] = hub.is_online(runner.id)
    data.update(ownership_fields(runner))
    data["agent_definitions"] = [
        {"id": a.id, "name": a.name, "acp_agent": a.acp_agent} for a in await _runner_agents(container, runner.id)
    ]
    return data


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "runner"


async def _create_runner_agent(
    container: ApplicationContainer, runner: Runner, agent: str, agent_id: str | None = None, name: str | None = None
) -> dict[str, Any]:
    if container.agent_backend is None:
        raise HTTPException(status_code=501, detail="Agent backend not configured")
    raw = unwrap(container.agent_backend)
    base = agent_id or f"{_slug(runner.name)}-{_slug(agent)}"
    candidate, n = base, 2
    # Checked on the unscoped store: an id taken by someone else's private agent is still taken.
    while await raw.get(candidate) is not None:
        if agent_id:
            raise HTTPException(status_code=409, detail=f"Agent '{agent_id}' already exists")
        candidate, n = f"{base}-{n}", n + 1
    defn = AgentDefinition(
        id=candidate,
        name=name or f"{agent} on {runner.name}",
        description=f"{agent} on {runner.hostname or runner.name}, connected with acp-web-proxy",
        default_runtime="runner",
        runner_id=runner.id,
        protocol="acp",
        acp_agent=agent,
        **visibility_kwargs(runner.visibility, runner.team_id),
    )
    saved = await container.agent_backend.create(defn)
    return saved.model_dump(mode="json")


async def _visible_runner(container: ApplicationContainer, runner_id: str) -> Runner:
    runner = await _backend(container).get_runner(runner_id)
    if runner is None or not can_see(runner):
        raise HTTPException(status_code=404, detail=f"Runner '{runner_id}' not found")
    return runner


# ─── public ───────────────────────────────────────────────────────────────────


@router.get("/info")
async def runners_info():
    settings = get_settings()
    return {"api": "ai-agents-carrier", "runners": settings.acp_runners_enabled, "ui_url": settings.ui_base_url}


INSTALL_MD = """# Connect this machine to carrier

Instructions for an AI coding agent asked to "install the web proxy" for carrier.

1. Check `node --version` is 20 or newer. If Node is missing, install the LTS with
   the platform's package manager (ask the user before installing system packages).
2. At least one coding agent should be installed and logged in: `claude`, `codex`
   or `gemini`.
3. Run:

       npx -y github:comtihon/acp-web-proxy install {carrier}

   It returns at once and prints an approval number.
4. Tell the user the number and where to enter it ({where}). Only a signed-in
   carrier user may approve it. Never approve it yourself.
5. When the user says it is approved, run `acp-web-proxy status`: `connected`
   means done. `expired` or `denied`: run step 3 again for a new number.

Logs: `acp-web-proxy logs -n 50`. Remove: `acp-web-proxy uninstall --purge`.
"""


@router.get("/install.md", response_class=PlainTextResponse)
async def install_instructions():
    settings = get_settings()
    where = f"{settings.ui_base_url.rstrip('/')}/#approve-runner" if settings.ui_base_url else "carrier UI → Runners → Approve device"
    return PlainTextResponse(INSTALL_MD.format(carrier=settings.base_url.rstrip("/"), where=where), media_type="text/markdown")


@router.post("/enrollments", status_code=201)
async def create_enrollment(request: Request, container: ApplicationContainer = Depends(get_container)):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="JSON body required")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    try:
        e, code, poll_token = await _service(container).create(body, _client_ip(request))
    except EnrollmentError as err:
        raise _http(err)
    ui = get_settings().ui_base_url
    return {
        "enrollment_id": e.id,
        "poll_token": poll_token,
        "code": code,
        "expires_at": e.expires_at.isoformat(),
        "interval_s": 3,
        "approve_url": f"{ui.rstrip('/')}/#approve-runner" if ui else None,
    }


@router.get("/enrollments/{enrollment_id}/poll")
async def poll_enrollment(enrollment_id: str, request: Request, container: ApplicationContainer = Depends(get_container)):
    auth = request.headers.get("authorization", "")
    token = auth.split(" ", 1)[1].strip() if auth.lower().startswith("bearer ") else None
    try:
        return await _service(container).poll(enrollment_id, token)
    except EnrollmentError as err:
        raise _http(err)


@router.delete("/self", status_code=204)
async def delete_self(request: Request, container: ApplicationContainer = Depends(get_container)):
    backend = _backend(container)
    runner = await authenticate_runner(backend, request.headers.get("authorization"))
    if runner is None:
        raise HTTPException(status_code=401, detail="Invalid runner credential")
    await backend.delete_runner(runner.id)
    await get_runner_hub().drop(runner.id)
    return Response(status_code=204)


# ─── signed-in users ─────────────────────────────────────────────────────────


class LookupRequest(BaseModel):
    code: str


class DecideRequest(BaseModel):
    code: str
    name: str | None = None
    visibility: Literal["private", "team", "public"] | None = None
    team_id: str | None = None
    # Agents of the machine to add as carrier agents; None = all of them.
    agents: list[str] | None = None


@router.post("/enrollments/lookup")
async def lookup_enrollment(body: LookupRequest, request: Request, container: ApplicationContainer = Depends(get_container)):
    _require_enroller()
    try:
        e = await _service(container).lookup(body.code, _user_key(request))
    except EnrollmentError as err:
        raise _http(err)
    return _present_enrollment(e)


@router.post("/enrollments/{enrollment_id}/approve")
async def approve_enrollment(
    enrollment_id: str, body: DecideRequest, request: Request, container: ApplicationContainer = Depends(get_container)
):
    _require_enroller()
    visibility = body.visibility or "private"
    team_id = _check_scope(visibility, body.team_id)
    try:
        runner = await _service(container).approve(
            enrollment_id, body.code,
            user_key=_user_key(request), subject=get_current_subject(),
            name=body.name, visibility=visibility, team_id=team_id,
        )
    except EnrollmentError as err:
        raise _http(err)
    wanted = runner.agents if body.agents is None else [a for a in runner.agents if a in body.agents]
    agents = [await _create_runner_agent(container, runner, a) for a in wanted]
    logger.info("runner %s (%s) approved by %s with agents %s", runner.id, runner.hostname, runner.approved_by, wanted)
    return {"runner": await _present(container, runner), "agents": agents}


@router.post("/enrollments/{enrollment_id}/deny")
async def deny_enrollment(
    enrollment_id: str, body: DecideRequest, request: Request, container: ApplicationContainer = Depends(get_container)
):
    _require_enroller()
    try:
        await _service(container).deny(enrollment_id, body.code, user_key=_user_key(request), subject=get_current_subject())
    except EnrollmentError as err:
        raise _http(err)
    return {"status": "denied"}


@router.get("")
async def list_runners(container: ApplicationContainer = Depends(get_container)):
    runners = [r for r in await _backend(container).list_runners() if can_see(r)]
    runners.sort(key=lambda r: (r.name or r.id).lower())
    return [await _present(container, r) for r in runners]


@router.get("/{runner_id}")
async def get_runner(runner_id: str, container: ApplicationContainer = Depends(get_container)):
    return await _present(container, await _visible_runner(container, runner_id))


class RunnerUpdate(BaseModel):
    name: str | None = None
    visibility: Literal["private", "team", "public"] | None = None
    team_id: str | None = None


@router.patch("/{runner_id}")
async def update_runner(runner_id: str, body: RunnerUpdate, container: ApplicationContainer = Depends(get_container)):
    backend = _backend(container)
    runner = await _visible_runner(container, runner_id)
    fields: dict[str, Any] = {}
    if body.name is not None and body.name.strip() != runner.name:
        if not can_edit(runner):
            raise HTTPException(status_code=403, detail="You may not rename this runner")
        fields["name"] = body.name.strip()[:80]
    if body.visibility is not None and (body.visibility != runner.visibility or body.team_id not in (None, runner.team_id)):
        if not can_change_visibility(runner):
            raise HTTPException(status_code=403, detail="You may not change who can use this runner")
        fields["visibility"] = body.visibility
        fields["team_id"] = _check_scope(body.visibility, body.team_id)
    if fields:
        await backend.update_runner(runner_id, fields)
        runner = await backend.get_runner(runner_id)
    return await _present(container, runner)


@router.delete("/{runner_id}", status_code=204)
async def delete_runner(runner_id: str, container: ApplicationContainer = Depends(get_container)):
    runner = await _visible_runner(container, runner_id)
    if not can_manage(runner):
        raise HTTPException(status_code=403, detail="Only the runner's owner may disconnect it")
    await _backend(container).delete_runner(runner_id)
    await get_runner_hub().drop(runner_id)
    logger.info("runner %s revoked", runner_id)
    return Response(status_code=204)


class RunnerAgentRequest(BaseModel):
    acp_agent: str
    id: str | None = None
    name: str | None = None


@router.post("/{runner_id}/agents", status_code=201)
async def add_runner_agent(runner_id: str, body: RunnerAgentRequest, container: ApplicationContainer = Depends(get_container)):
    runner = await _visible_runner(container, runner_id)
    if body.acp_agent not in runner.agents:
        raise HTTPException(status_code=400, detail=f"Runner '{runner.name}' does not offer agent '{body.acp_agent}'")
    return await _create_runner_agent(container, runner, body.acp_agent, body.id, body.name)


# ─── sockets ─────────────────────────────────────────────────────────────────


async def _socket_runner(ws: WebSocket) -> Runner | None:
    """The runner behind a socket's credential; closes the socket when there is none."""
    container = ws.app.state.container
    backend = getattr(container, "runner_backend", None)
    if backend is None or not get_settings().acp_runners_enabled:
        await ws.close(code=1008)
        return None
    try:
        runner = await authenticate_runner(backend, ws.headers.get("authorization"))
    except Exception:
        # A storage hiccup is not a revocation: let the runner retry.
        logger.exception("runner authentication failed")
        await ws.accept()
        await ws.close(code=1011)
        return None
    if runner is None:
        # Before accept: answered with HTTP 403, which the runner reads as revoked.
        await ws.close(code=4403)
        return None
    return runner


@router.websocket("/connect")
async def runner_control(ws: WebSocket):
    runner = await _socket_runner(ws)
    if runner is None:
        return
    backend = ws.app.state.container.runner_backend
    hub = get_runner_hub()
    await ws.accept()
    session = RunnerSession(runner.id, ws)
    await hub.attach(session)
    now = utcnow()
    await backend.update_runner(runner.id, {"connected_at": now, "last_seen": now})
    logger.info("runner %s (%s) connected", runner.id, runner.name)
    last_seen = now
    try:
        while True:
            text = await ws.receive_text()
            try:
                msg = json.loads(text)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            method = msg.get("method")
            if method == "_proxy/register":
                await _register(backend, runner, msg.get("params") or {})
                if msg.get("id") is not None:
                    await session.send({"jsonrpc": "2.0", "id": msg["id"], "result": {"runner_id": runner.id, "name": runner.name}})
            elif method is None:
                session.resolve(msg)
            elif msg.get("id") is not None:
                await session.send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": f"unknown method {method}"}})
            if (utcnow() - last_seen).total_seconds() > 60:
                last_seen = utcnow()
                await backend.update_runner(runner.id, {"last_seen": last_seen})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.detach(session)
        try:
            await backend.update_runner(runner.id, {"last_seen": utcnow()})
        except Exception:
            pass
        logger.info("runner %s disconnected", runner.id)


async def _register(backend: Any, runner: Runner, params: dict[str, Any]) -> None:
    agents = clean_agents(params.get("agents"))
    labels = params.get("labels") if isinstance(params.get("labels"), dict) else {}
    default_agent = params.get("default_agent")
    fields: dict[str, Any] = {
        "agents": agents,
        "default_agent": default_agent if default_agent in agents else (agents[0] if agents else None),
        "labels": {str(k)[:64]: str(v)[:256] for k, v in list(labels.items())[:20]},
        "last_seen": utcnow(),
    }
    for key, limit in (("version", 32), ("hostname", 128), ("os", 32), ("arch", 32), ("tz", 64)):
        if isinstance(params.get(key), str) and params[key]:
            fields[key] = params[key][:limit]
    if isinstance(params.get("workspace"), str):
        fields["labels"]["workspace"] = params["workspace"][:512]
    await backend.update_runner(runner.id, fields)


@router.websocket("/connect/connections/{connection_id}")
async def runner_data(ws: WebSocket, connection_id: str):
    runner = await _socket_runner(ws)
    if runner is None:
        return
    await ws.accept()
    sock = InboundSocket(ws, connection_id)
    if not get_runner_hub().offer_data_socket(runner.id, connection_id, sock):
        await ws.close(code=4404)
        return
    await sock.done.wait()
