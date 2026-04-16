from __future__ import annotations

import logging
from typing import Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


class CronScheduler:
    """Manages cron-based workflow trigger jobs using APScheduler."""

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        # Maps "workflow_id:step_id" -> APScheduler job id
        self._jobs: dict[str, str] = {}

    def start(self) -> None:
        self._scheduler.start()
        logger.info("Cron scheduler started")

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Cron scheduler stopped")

    def register(
        self,
        workflow_id: str,
        step_id: str,
        cron_expr: str,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        """Register (or replace) a cron job for a workflow step.

        *cron_expr* is a standard 5-field cron expression, e.g. ``"0 9 * * 1-5"``.
        """
        job_key = f"{workflow_id}:{step_id}"
        self._remove_job(job_key)
        try:
            trigger = CronTrigger.from_crontab(cron_expr, timezone="UTC")
            job = self._scheduler.add_job(callback, trigger, id=job_key, replace_existing=True)
            self._jobs[job_key] = job.id
            logger.info(
                "Cron job registered: workflow=%s step=%s schedule=%r",
                workflow_id, step_id, cron_expr,
            )
        except Exception:
            logger.exception(
                "Failed to register cron job for workflow=%s step=%s", workflow_id, step_id
            )

    def unregister(self, workflow_id: str, step_id: str) -> None:
        self._remove_job(f"{workflow_id}:{step_id}")

    def unregister_workflow(self, workflow_id: str) -> None:
        keys = [k for k in list(self._jobs) if k.startswith(f"{workflow_id}:")]
        for key in keys:
            self._remove_job(key)

    def _remove_job(self, job_key: str) -> None:
        if job_key in self._jobs:
            try:
                self._scheduler.remove_job(job_key)
            except Exception:
                pass
            self._jobs.pop(job_key, None)
