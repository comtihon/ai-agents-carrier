"""Cron schedules must be able to name their own timezone.

A schedule like "every working day at 09:00 Berlin" has no fixed-UTC
equivalent: 09:00 Berlin is 07:00 UTC in summer and 08:00 UTC in winter, so a
hardcoded UTC hour is wrong for half the year. These tests pin the two things
that matter: the named zone reaches APScheduler, and a bad zone name degrades
to UTC instead of dropping the trigger on the floor.
"""
from __future__ import annotations

import asyncio

from app.infrastructure.triggers.cron_scheduler import CronScheduler


async def _noop() -> None:
    return None


def _tz_name(scheduler: CronScheduler, job_key: str) -> str:
    job = scheduler._scheduler.get_job(job_key)
    assert job is not None, f"job {job_key!r} was not registered"
    return str(job.trigger.timezone)


def test_named_timezone_is_used() -> None:
    sched = CronScheduler()
    sched.register("wf", "trigger", "0 9 * * 1-5", _noop, timezone="Europe/Berlin")
    assert _tz_name(sched, "wf:trigger") == "Europe/Berlin"


def test_timezone_defaults_to_utc() -> None:
    sched = CronScheduler()
    sched.register("wf", "trigger", "0 9 * * 1-5", _noop)
    assert _tz_name(sched, "wf:trigger") == "UTC"


def test_unknown_timezone_falls_back_to_utc() -> None:
    """A typo in the zone must not silently cost you the whole schedule."""
    sched = CronScheduler()
    sched.register("wf", "trigger", "0 9 * * 1-5", _noop, timezone="Europe/Berlim")
    assert _tz_name(sched, "wf:trigger") == "UTC"


def test_berlin_schedule_survives_dst() -> None:
    """The point of naming a zone: 09:00 local stays 09:00 across the switch.

    Fire times are computed on both sides of the European DST boundary; in UTC
    terms they differ by an hour, which is exactly the drift a fixed UTC cron
    would have baked in.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from apscheduler.triggers.cron import CronTrigger

    berlin = ZoneInfo("Europe/Berlin")
    trigger = CronTrigger.from_crontab("0 9 * * 1-5", timezone="Europe/Berlin")

    # Summer (CEST, UTC+2) and winter (CET, UTC+1) reference points.
    summer_prev = datetime(2026, 7, 1, 0, 0, tzinfo=berlin)
    winter_prev = datetime(2026, 12, 1, 0, 0, tzinfo=berlin)

    summer_next = trigger.get_next_fire_time(None, summer_prev)
    winter_next = trigger.get_next_fire_time(None, winter_prev)

    # Local wall-clock hour is stable...
    assert summer_next.astimezone(berlin).hour == 9
    assert winter_next.astimezone(berlin).hour == 9
    # ...precisely because the UTC hour is not.
    assert summer_next.utcoffset() != winter_next.utcoffset()


def test_register_passes_step_timezone_through_container() -> None:
    """The container must forward a cron step's ``timezone`` to the scheduler."""
    import app.core.container as container_mod

    recorded: dict[str, object] = {}

    class _FakeScheduler:
        def register(self, workflow_id, step_id, schedule, callback, timezone="UTC"):
            recorded.update(
                workflow_id=workflow_id, step_id=step_id,
                schedule=schedule, timezone=timezone,
            )

    class _FakeRunner:
        id = "csm"
        steps = [
            {"id": "trigger", "type": "cron",
             "schedule": "0 9 * * 1-5", "timezone": "Europe/Berlin"},
        ]

    container = object.__new__(container_mod.ApplicationContainer)
    container.cron_scheduler = _FakeScheduler()  # type: ignore[assignment]
    container._make_cron_job = lambda wf, tmpl: _noop  # type: ignore[assignment]

    container_mod.ApplicationContainer._register_cron_steps(container, _FakeRunner())  # type: ignore[arg-type]

    assert recorded["timezone"] == "Europe/Berlin"
    assert recorded["schedule"] == "0 9 * * 1-5"


def test_asyncio_available() -> None:
    """Guard against the module-level import being dropped as unused."""
    assert asyncio.iscoroutinefunction(_noop)
