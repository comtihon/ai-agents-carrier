"""Runtime pools: which runner agent a pool agent's step runs on.

A pool (app.domain.models.runtime_pool) is shared like any other resource
(app.application.visibility): usually with a team, whose workflows then use
it through *pool agents* — agents with ``default_runtime: "pool"``. The pool
itself has no prompt or tools; those stay on the pool agent. The pool only
says which machine runs it.

Membership
----------
A member is a runner agent (``default_runtime: "runner"``, ACP) of any
visibility. Only the agent's **owner** may add it, and only to a pool they can
see: that is their consent to run the pool's workflows on their machine, with
whatever that machine can reach. The owner may take it out again any time;
whoever may edit the pool, or an admin of its team, may remove any member but
never add someone else's.

Choosing a member
-----------------
:func:`assign_member` decides, per workflow run:

* a run that already has a member keeps it while that member is online (or
  only briefly offline) — every step of the run shares one machine, so they
  share its workspace;
* a run without one takes the next member in round-robin order that is online
  and not out of quota (an online member out of quota is the fallback: the
  step then waits for its quota as any runner step does);
* a run whose member went offline moves to the next one, recorded in the
  assignment's ``previous``.

The choice is stored (``acp_runtime_pool_assignments``) with compare-and-set,
so parallel steps of one run agree on a member and a carrier restart re-attaches
to it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from app.application.visibility import (
    NotOwner,
    ResourceNotFound,
    VisibilityError,
    can_change_visibility,
    can_edit,
    can_see,
    ownership_fields,
)
from app.domain.models.runtime_pool import PoolAssignment, PoolMember, RuntimePool
from app.infrastructure.auth.authorization import get_current_subject

logger = logging.getLogger(__name__)

# How long a pool step waits for any member to come online before failing.
# None (or <= 0): wait until one does — the default, so a run started while
# every member is offline simply starts when the first comes back.
DEFAULT_MEMBER_WAIT_S: float | None = None
# How long a run's member may be offline before the run moves on.
DEFAULT_GRACE_S = 120.0
_POLL_S = 5.0


class PoolError(VisibilityError):
    status_code = 400


class NoMemberAvailable(RuntimeError):
    """No member of the pool came online in time."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─── Members ──────────────────────────────────────────────────────────────────

def check_member_agent(agent: Any) -> None:
    """Refuse an agent that cannot be a pool member."""
    if getattr(agent, "default_runtime", None) != "runner" or not getattr(agent, "runner_id", None):
        raise PoolError(
            f"Agent '{agent.id}' does not run on a connected machine; only runner agents "
            "(default_runtime 'runner') can join a runtime pool"
        )
    if getattr(agent, "protocol", None) != "acp":
        raise PoolError(f"Agent '{agent.id}' must speak ACP to join a runtime pool")


async def add_member(pool_backend: Any, pool: RuntimePool, agent: Any) -> RuntimePool:
    """The caller puts their own runner agent into *pool*."""
    subject = get_current_subject()
    if not can_see(pool):
        raise ResourceNotFound(f"Runtime pool '{pool.id}' not found")
    check_member_agent(agent)
    if subject is not None and agent.owner_id != subject.id:
        # Not even a super admin: a member runs other people's workflows on
        # its owner's machine, which only the owner may agree to.
        raise NotOwner(f"Only the owner of agent '{agent.id}' may add it to a runtime pool")
    if agent.id in pool.member_ids():
        return pool
    members = list(pool.members) + [PoolMember(
        agent_id=agent.id,
        added_by=getattr(subject, "id", None),
        added_by_name=getattr(subject, "name", None),
        added_at=_utcnow(),
    )]
    await pool_backend.set_members(pool.id, members)
    pool.members = members
    logger.info("agent %s joined runtime pool %s", agent.id, pool.id)
    return pool


async def remove_member(pool_backend: Any, pool: RuntimePool, agent_id: str, agent: Any = None) -> RuntimePool:
    """The agent's owner takes it out, or whoever may edit the pool removes it."""
    subject = get_current_subject()
    if not can_see(pool):
        raise ResourceNotFound(f"Runtime pool '{pool.id}' not found")
    if agent_id not in pool.member_ids():
        raise ResourceNotFound(f"Agent '{agent_id}' is not a member of runtime pool '{pool.id}'")
    is_owner = subject is None or (agent is not None and agent.owner_id == subject.id)
    if not (is_owner or can_edit(pool) or can_change_visibility(pool)):
        raise NotOwner(
            f"Only the owner of agent '{agent_id}', whoever may edit runtime pool '{pool.id}', "
            "or an admin of its team may remove it"
        )
    members = [m for m in pool.members if m.agent_id != agent_id]
    await pool_backend.set_members(pool.id, members)
    pool.members = members
    logger.info("agent %s left runtime pool %s", agent_id, pool.id)
    return pool


async def drop_agent_everywhere(pool_backend: Any, agent_id: str) -> None:
    """Forget a deleted agent in every pool it was a member of."""
    if pool_backend is None:
        return
    for pool in await pool_backend.list():
        if agent_id in pool.member_ids():
            await pool_backend.set_members(pool.id, [m for m in pool.members if m.agent_id != agent_id])


# ─── Status ───────────────────────────────────────────────────────────────────

class MemberStatus:
    """One member as a pool step sees it."""

    def __init__(self, agent_id: str, agent: Any, runner: Any, online: bool, limit: Any) -> None:
        self.agent_id = agent_id
        self.agent = agent
        self.runner = runner
        self.online = online
        self.limit = limit

    @property
    def usable(self) -> bool:
        return self.agent is not None and self.runner is not None

    @property
    def acp_agent(self) -> str | None:
        if self.agent is None:
            return None
        return self.agent.acp_agent or (self.runner.default_agent if self.runner is not None else None)


async def member_status(agent_id: str, *, agent_backend: Any, hub: Any) -> MemberStatus:
    from app.application.visibility import unwrap
    agent = await unwrap(agent_backend).get(agent_id) if agent_backend is not None else None
    if agent is not None and (agent.default_runtime != "runner" or not agent.runner_id):
        agent = None
    runner = None
    if agent is not None and hub.backend is not None:
        runner = await hub.backend.get_runner(agent.runner_id)
    online = runner is not None and hub.is_online(runner.id)
    status = MemberStatus(agent_id, agent, runner, online, None)
    if online and status.acp_agent:
        status.limit = await hub.active_limit(runner.id, status.acp_agent)
    return status


async def present(
    pool: RuntimePool, *, agent_backend: Any, hub: Any, pool_backend: Any, run_repository: Any = None,
) -> dict[str, Any]:
    """A pool as the API shows it: members, who is online, which runs hold whom."""
    data = pool.model_dump(mode="json", exclude={"members", "cursor", "can_edit", "team_name"})
    data.update(ownership_fields(pool))
    holders: dict[str, list[dict[str, Any]]] = {}
    if pool_backend is not None:
        since = _utcnow() - timedelta(days=2)
        for a in await pool_backend.list_assignments(pool.id, since=since):
            status = None
            if run_repository is not None:
                try:
                    run = await run_repository.get(a.run_id)
                    status = getattr(run, "status", None) if run is not None else None
                except Exception:
                    logger.debug("could not load run %s", a.run_id, exc_info=True)
            if status not in ("completed", "failed", "cancelled", "rejected"):
                holders.setdefault(a.agent_id, []).append(
                    {"run_id": a.run_id, "assigned_at": a.assigned_at.isoformat(), "run_status": status}
                )
    members: list[dict[str, Any]] = []
    statuses: list[MemberStatus] = []
    for m in pool.members:
        s = await member_status(m.agent_id, agent_backend=agent_backend, hub=hub)
        statuses.append(s)
        members.append({
            "agent_id": m.agent_id,
            # Name and owner only: a private member's definition stays its owner's.
            "name": getattr(s.agent, "name", None),
            "owner_name": getattr(s.agent, "owner_name", None),
            "visibility": getattr(s.agent, "visibility", None),
            "runner_name": getattr(s.runner, "name", None),
            "platform": getattr(s.runner, "os", None),
            "acp_agent": s.acp_agent,
            "online": s.online,
            "usable": s.usable,
            "limited_until": (s.limit.until.isoformat() if s.limit is not None and s.limit.until else
                              ("unknown" if s.limit is not None else None)),
            "added_by_name": m.added_by_name,
            "added_at": m.added_at.isoformat() if m.added_at else None,
            "runs": holders.get(m.agent_id, []),
        })
    data["members"] = members
    data["member_count"] = len(members)
    # What the pool offers: members per platform (their runners' OS) and per
    # ACP agent, counting usable members (e.g. {"linux": 2, "win32": 1},
    # {"claude": 3, "codex": 1}), and how many of them are online now.
    platforms: dict[str, int] = {}
    agents: dict[str, int] = {}
    agents_online: dict[str, int] = {}
    for m, status in zip(members, statuses):
        if not m["usable"]:
            continue
        os_name = getattr(status.runner, "os", None) or "unknown"
        platforms[os_name] = platforms.get(os_name, 0) + 1
        name = m["acp_agent"] or "default"
        agents[name] = agents.get(name, 0) + 1
        if m["online"]:
            agents_online[name] = agents_online.get(name, 0) + 1
    data["platforms"] = platforms
    data["agents"] = agents
    data["agents_online"] = agents_online
    data["online_count"] = sum(1 for m in members if m["online"] and m["usable"])
    data["available_count"] = sum(1 for m in members if m["online"] and m["usable"] and not m["limited_until"])
    return data


# ─── Assignment ───────────────────────────────────────────────────────────────

async def _pick(
    pool: RuntimePool, *, pool_backend: Any, agent_backend: Any, hub: Any, exclude: set[str],
) -> MemberStatus | None:
    """The next member in round-robin order: online and in quota first."""
    ids = [i for i in pool.member_ids() if i not in exclude]
    if not ids:
        return None
    start = await pool_backend.next_cursor(pool.id) % len(ids)
    ordered = ids[start:] + ids[:start]
    fallback: MemberStatus | None = None
    for agent_id in ordered:
        s = await member_status(agent_id, agent_backend=agent_backend, hub=hub)
        if not (s.usable and s.online):
            continue
        if s.limit is None:
            return s
        fallback = fallback or s
    return fallback


async def assign_member(
    pool_id: str,
    run_id: str,
    *,
    pool_backend: Any,
    agent_backend: Any,
    hub: Any,
    failed: str | None = None,
    exclude: set[str] | None = None,
    grace_s: float = DEFAULT_GRACE_S,
    wait_s: float | None = DEFAULT_MEMBER_WAIT_S,
    on_wait: Any = None,
) -> tuple[MemberStatus, PoolAssignment]:
    """The member run *run_id* uses on pool *pool_id* (see module docstring).

    *failed*: the member the run just lost; the run moves on from it to
    another ONLINE member (unless a parallel step already moved it).
    *exclude*: members not to pick on the first pass.

    When no member is online the call waits: *wait_s* seconds, or until one
    comes online when *wait_s* is None or <= 0. While waiting every member is
    eligible again, the run's own member (including *failed*) first — it holds
    the run's workspace — then the others in round-robin order. *on_wait* is
    awaited once, when the waiting starts. Cancelling the caller stops the wait.
    """
    exclude = set(exclude or ())
    if failed:
        exclude.add(failed)
    deadline = time.monotonic() + wait_s if wait_s is not None and wait_s > 0 else None
    waited = False
    while True:
        pool = await pool_backend.get(pool_id)
        if pool is None:
            raise RuntimeError(f"runtime pool '{pool_id}' no longer exists")
        if not pool.members:
            raise RuntimeError(f"runtime pool '{pool.name or pool_id}' has no members")
        current = await pool_backend.get_assignment(pool_id, run_id)
        if current is not None and current.agent_id != failed and current.agent_id in pool.member_ids():
            s = await member_status(current.agent_id, agent_backend=agent_backend, hub=hub)
            if s.usable and current.agent_id not in exclude:
                # The grace applies once; while waiting, the run's member is
                # simply checked on every round, like the others.
                if s.online or (
                    not waited and s.runner is not None and await hub.wait_online(s.runner.id, grace_s)
                ):
                    return s, current
                if not waited:
                    logger.info("pool %s: run %s lost member %s (offline)", pool_id, run_id, current.agent_id)
            if not waited:
                exclude.add(current.agent_id)
        picked = await _pick(pool, pool_backend=pool_backend, agent_backend=agent_backend, hub=hub, exclude=exclude)
        if picked is not None:
            replacing = current.agent_id if current is not None else None
            won = await pool_backend.claim_assignment(pool_id, run_id, picked.agent_id, replacing)
            if won.agent_id != picked.agent_id:
                # A parallel step of the run chose first: use its member.
                s = await member_status(won.agent_id, agent_backend=agent_backend, hub=hub)
                if s.usable and s.online:
                    return s, won
                continue
            return picked, won
        if deadline is not None and time.monotonic() >= deadline:
            raise NoMemberAvailable(
                f"no member of runtime pool '{pool.name or pool_id}' came online within {int(wait_s or 0)}s "
                f"({len(pool.members)} member(s))"
            )
        if not waited:
            waited = True
            if on_wait is not None:
                await on_wait(pool)
        # Nobody else is online: every member may serve the run again — the
        # one it just lost too, and that one first if it comes back first.
        failed = None
        exclude = set()
        await asyncio.sleep(_POLL_S)
