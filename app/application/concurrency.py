"""Global concurrency limits on agent runtimes.

An agent runtime is what actually executes a step:

* ``runner:<runner_id>:<acp_agent>`` — an ACP agent on a connected machine
  (the definition's ``acp_agent``, else the runner's ``default_agent``);
* ``agent:<agent_definition_id>`` — any other agent (pod, docker, local).

Its limit is the lowest of everything configured for it — the runner's
``agent_concurrency[acp_agent]`` and ``max_concurrency`` of every agent
definition using it (runner keys), or the definition's ``max_concurrency``
(agent keys). Nothing configured: no gating at all.

The limit holds across every workflow of every user (private ones included)
and every carrier replica: slots are leases in MongoDB
(app.infrastructure.persistence.concurrency_backend). Runs over the limit queue
FIFO; they do not fail.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from app.infrastructure.persistence.concurrency_backend import (
    ConcurrencyBackend,
    InMemoryConcurrencyBackend,
)

logger = logging.getLogger(__name__)

DEFAULT_TTL_S = 120.0
DEFAULT_POLL_S = 2.0

_backend: ConcurrencyBackend | None = None


def get_concurrency_backend() -> ConcurrencyBackend:
    """The process-wide lease store (in-memory until the container sets Mongo's)."""
    global _backend
    if _backend is None:
        _backend = InMemoryConcurrencyBackend()
    return _backend


def set_concurrency_backend(backend: ConcurrencyBackend | None) -> None:
    global _backend
    _backend = backend


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ─── keys and limits ─────────────────────────────────────────────────────────


def runner_key(runner_id: str, acp_agent: str | None) -> str:
    return f"runner:{runner_id}:{acp_agent or ''}"


def agent_key(agent_id: str) -> str:
    return f"agent:{agent_id}"


def parse_key(key: str) -> dict[str, Any]:
    """``{kind, runner_id, acp_agent, agent_id}`` of a runtime key."""
    if key.startswith("runner:"):
        _, rest = key.split(":", 1)
        runner_id, _, acp_agent = rest.partition(":")
        return {"kind": "runner", "runner_id": runner_id, "acp_agent": acp_agent or None, "agent_id": None}
    return {"kind": "agent", "runner_id": None, "acp_agent": None, "agent_id": key.split(":", 1)[-1]}


def _min_limit(values: list[Any]) -> int | None:
    found = [int(v) for v in values if isinstance(v, int) and not isinstance(v, bool) and v >= 1]
    return min(found) if found else None


def runner_agent_of(agent_def: Any, runner: Any) -> str | None:
    return getattr(agent_def, "acp_agent", None) or (getattr(runner, "default_agent", None) if runner else None)


async def _all_definitions(agent_backend: Any) -> list[Any]:
    if agent_backend is None:
        return []
    from app.application.visibility import unwrap
    try:
        # Unscoped: another user's private definition still limits a shared runtime.
        return list(await unwrap(agent_backend).list())
    except Exception:
        logger.warning("could not list agent definitions for concurrency limits", exc_info=True)
        return []


async def runtime_for(
    agent_def: Any, runner: Any = None, agent_backend: Any = None,
) -> tuple[str, int | None]:
    """The runtime key of *agent_def* and its effective limit (None: unlimited)."""
    if runner is not None:
        acp_agent = runner_agent_of(agent_def, runner)
        values: list[Any] = [
            (getattr(runner, "agent_concurrency", None) or {}).get(acp_agent or ""),
            getattr(agent_def, "max_concurrency", None),
        ]
        for d in await _all_definitions(agent_backend):
            if (
                getattr(d, "default_runtime", None) == "runner"
                and getattr(d, "runner_id", None) == runner.id
                and runner_agent_of(d, runner) == acp_agent
            ):
                values.append(getattr(d, "max_concurrency", None))
        return runner_key(runner.id, acp_agent), _min_limit(values)
    return agent_key(agent_def.id), _min_limit([getattr(agent_def, "max_concurrency", None)])


async def configured_limits(agent_backend: Any, runner_backend: Any) -> dict[str, int]:
    """Every runtime key with a limit configured, and that limit."""
    runners = {}
    if runner_backend is not None:
        try:
            runners = {r.id: r for r in await runner_backend.list_runners()}
        except Exception:
            logger.warning("could not list runners for concurrency limits", exc_info=True)
    values: dict[str, list[Any]] = {}
    for r in runners.values():
        for agent, cap in (r.agent_concurrency or {}).items():
            values.setdefault(runner_key(r.id, agent), []).append(cap)
    for d in await _all_definitions(agent_backend):
        if d.max_concurrency is None:
            continue
        if d.default_runtime == "runner":
            runner = runners.get(d.runner_id or "")
            if runner is None:
                continue
            values.setdefault(runner_key(runner.id, runner_agent_of(d, runner)), []).append(d.max_concurrency)
        else:
            values.setdefault(agent_key(d.id), []).append(d.max_concurrency)
    out = {}
    for key, vals in values.items():
        limit = _min_limit(vals)
        if limit is not None:
            out[key] = limit
    return out


def merge_agent_concurrency(current: dict[str, int], changes: dict[str, int | None]) -> dict[str, int]:
    """*current* with *changes* applied (None/0 removes); ValueError on a bad entry."""
    merged = dict(current or {})
    for agent, cap in changes.items():
        agent = (agent or "").strip()
        if not agent or len(agent) > 128:
            raise ValueError("agent_concurrency needs agent names (1-128 characters)")
        if cap is None or cap == 0:
            merged.pop(agent, None)
        elif not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
            raise ValueError(f"agent_concurrency['{agent}'] must be a positive integer, 0 or null")
        else:
            merged[agent] = cap
    return merged



# ─── the gate ────────────────────────────────────────────────────────────────


WaitCallback = Callable[[int, int], Awaitable[None]]


class ConcurrencySlot:
    """One step's claim on a runtime: ``acquire()`` waits its turn, ``release()`` frees it.

    ``limit`` None makes both no-ops. While held, a background task keeps the
    lease alive; if the lease was lost anyway (the event loop stalled past the
    TTL) the step — which is running regardless — takes its slot back.
    Re-acquiring with the same ``entry_id`` (the step's task key) after a
    restart finds the queue place or slot it had.
    """

    def __init__(
        self,
        key: str,
        limit: int | None,
        entry: dict[str, Any],
        *,
        backend: ConcurrencyBackend | None = None,
        ttl_s: float = DEFAULT_TTL_S,
        poll_s: float = DEFAULT_POLL_S,
        on_wait: WaitCallback | None = None,
        on_acquired: Callable[[bool], Awaitable[None]] | None = None,
    ) -> None:
        self.key = key
        self.limit = limit
        self.entry = dict(entry)
        self.backend = backend or get_concurrency_backend()
        self.ttl_s = max(float(ttl_s), 1.0)
        self.poll_s = max(float(poll_s), 0.01)
        self.on_wait = on_wait
        self.on_acquired = on_acquired
        self.held = False
        self._heartbeat: asyncio.Task | None = None

    @property
    def gated(self) -> bool:
        return self.limit is not None

    async def acquire(self, force: bool = False) -> None:
        """Hold a slot, waiting in the queue as long as it takes.

        ``force``: the step is already executing (re-attached after a restart);
        take the slot back even if that puts the runtime over its limit.
        """
        if not self.gated or self.held:
            return
        self.entry.setdefault("enqueued_at", _utcnow())
        waited = False
        last: tuple[int, int] | None = None
        try:
            while True:
                res = await self.backend.try_acquire(
                    self.key, self.entry, self.limit, _utcnow(), self.ttl_s, force=force,  # type: ignore[arg-type]
                )
                if res.acquired:
                    break
                waited = True
                if self.on_wait is not None and (res.position, res.holders) != last:
                    last = (res.position, res.holders)
                    try:
                        await self.on_wait(res.position, res.holders)
                    except Exception:
                        logger.debug("queue wait report failed", exc_info=True)
                await asyncio.sleep(self.poll_s)
        except BaseException:
            # Cancelled or failed while queued: leave the queue.
            await self._release_quietly()
            raise
        self.held = True
        self._heartbeat = asyncio.ensure_future(self._beat())
        if self.on_acquired is not None:
            try:
                await self.on_acquired(waited)
            except Exception:
                logger.debug("slot acquired report failed", exc_info=True)

    async def _beat(self) -> None:
        interval = max(self.ttl_s / 4, 0.01)
        while True:
            await asyncio.sleep(interval)
            try:
                if not await self.backend.heartbeat(self.key, self.entry["id"], _utcnow(), self.ttl_s):
                    logger.warning("lease on %s for %s was lost; taking it back", self.key, self.entry["id"])
                    await self.backend.try_acquire(self.key, self.entry, self.limit or 1, _utcnow(), self.ttl_s, force=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("lease heartbeat on %s failed", self.key, exc_info=True)

    async def _release_quietly(self) -> None:
        try:
            await asyncio.shield(self.backend.release(self.key, self.entry["id"]))
        except BaseException:
            logger.warning("could not release lease on %s for %s", self.key, self.entry["id"], exc_info=True)

    async def release(self) -> None:
        if not self.gated:
            return
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            self._heartbeat = None
        self.held = False
        await self._release_quietly()

    async def __aenter__(self) -> "ConcurrencySlot":
        await self.acquire()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.release()


def queue_reporter(sink: Any, step_id: str, key: str, limit: int | None, agent_id: str) -> tuple[WaitCallback, Callable[[bool], Awaitable[None]]]:
    """``on_wait`` / ``on_acquired`` writing ``_queue_wait_<step_id>`` and progress lines."""
    state_key = f"_queue_wait_{step_id}"
    since: list[str] = []

    async def alive() -> None:
        # Waiting in the queue (or holding the slot again) after a restart is not a crash loop: reset the run's
        # startup-recovery counter, or a run queued behind a long holder is quarantined (and its workflow
        # disabled) after run_recovery_max_attempts ordinary restarts.
        reset = getattr(sink, "reset_recovery_attempts", None)
        if reset is not None:
            await reset()

    async def on_wait(position: int, holders: int) -> None:
        first = not since
        if first:
            since.append(_utcnow().isoformat())
            sink.add(f"⏳ queued for {key} (position {position}, limit {limit})")
            await sink.flush()
        await sink.set_state(state_key, {
            "key": key, "limit": limit, "position": position, "holders": holders,
            "since": since[0], "agent_id": agent_id,
        })
        if first:
            await alive()

    async def on_acquired(waited: bool) -> None:
        if waited:
            await sink.set_state(state_key, None)
            sink.add("▶ slot acquired")
        await alive()

    return on_wait, on_acquired


async def clear_queue_state(sink: Any, step_id: str) -> None:
    await sink.set_state(f"_queue_wait_{step_id}", None)


# ─── the live view ───────────────────────────────────────────────────────────


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


async def describe(
    *,
    agent_backend: Any,
    runner_backend: Any,
    key: str | None = None,
    key_visible: Callable[[dict[str, Any]], Awaitable[bool]] | None = None,
    workflow_visible: Callable[[str | None], Awaitable[bool]] | None = None,
    backend: ConcurrencyBackend | None = None,
) -> list[dict[str, Any]]:
    """Holders and queue per runtime key, as GET /agent-runtimes/concurrency shows them."""
    store = backend or get_concurrency_backend()
    live = {d["key"]: d for d in await store.list_keys(_utcnow())}
    limits = await configured_limits(agent_backend, runner_backend)
    keys = sorted(set(live) | set(limits))
    if key is not None:
        keys = [k for k in keys if k == key]
    seen: dict[str | None, bool] = {}

    async def visible_wf(workflow_id: str | None) -> bool:
        if workflow_visible is None:
            return True
        if workflow_id not in seen:
            try:
                seen[workflow_id] = bool(await workflow_visible(workflow_id))
            except Exception:
                seen[workflow_id] = False
        return seen[workflow_id]

    async def present(e: dict[str, Any], position: int | None = None) -> dict[str, Any]:
        shown = await visible_wf(e.get("workflow_id"))
        out = {
            "id": e.get("id") if shown else None,
            "run_id": e.get("run_id") if shown else None,
            "step_id": e.get("step_id") if shown else None,
            "agent_id": e.get("agent_id") if shown else None,
            "workflow_id": e.get("workflow_id") if shown else None,
            "hidden": not shown,
        }
        if position is None:
            out["acquired_at"] = _iso(e.get("acquired_at"))
        else:
            out["position"] = position
            out["enqueued_at"] = _iso(e.get("enqueued_at"))
        out["heartbeat_at"] = _iso(e.get("heartbeat_at"))
        out["expires_at"] = _iso(e.get("expires_at"))
        return out

    result = []
    for k in keys:
        info = {"key": k, **parse_key(k)}
        if key_visible is not None and not await key_visible(info):
            continue
        doc = live.get(k) or {"holders": [], "queue": [], "limit": None}
        info["limit"] = limits.get(k, doc.get("limit") if doc.get("holders") or doc.get("queue") else None)
        info["holders"] = [await present(h) for h in doc["holders"]]
        info["queue"] = [await present(q, i + 1) for i, q in enumerate(doc["queue"])]
        result.append(info)
    return result


async def describe_for_caller(
    *, agent_backend: Any, runner_backend: Any, workflow_backend: Any, key: str | None = None,
) -> list[dict[str, Any]]:
    """``describe`` limited to what the current caller may see.

    A key is listed when the caller can see its runner (runner keys) or agent
    definition (agent keys). Entries of runs in workflows the caller cannot see
    still count, but carry no ids (``hidden: true``).
    """
    from app.application.visibility import can_see, workflow_visible

    async def key_visible(info: dict[str, Any]) -> bool:
        if info["kind"] == "runner":
            if runner_backend is None:
                return False
            runner = await runner_backend.get_runner(info["runner_id"])
            return runner is not None and can_see(runner)
        if agent_backend is None:
            return False
        return await agent_backend.get(info["agent_id"]) is not None

    async def wf_visible(workflow_id: str | None) -> bool:
        return await workflow_visible(workflow_backend, workflow_id)

    return await describe(
        agent_backend=agent_backend, runner_backend=runner_backend, key=key,
        key_visible=key_visible, workflow_visible=wf_visible,
    )
