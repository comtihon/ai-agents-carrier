"""Storage for runners and their enrollments (MongoDB, plus an in-memory twin for tests)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from app.domain.models.ownership import COMPUTED_FIELDS
from app.domain.models.runner import AgentLimit, Runner, RunnerEnrollment


class RunnerBackend(ABC):
    @abstractmethod
    async def list_runners(self) -> list[Runner]: ...

    @abstractmethod
    async def get_runner(self, runner_id: str) -> Runner | None: ...

    @abstractmethod
    async def save_runner(self, runner: Runner) -> Runner: ...

    @abstractmethod
    async def update_runner(self, runner_id: str, fields: dict[str, Any]) -> None:
        """Set fields of a runner without rewriting it (heartbeat, limits)."""

    @abstractmethod
    async def delete_runner(self, runner_id: str) -> None: ...

    @abstractmethod
    async def set_limit(self, runner_id: str, agent: str, limit: AgentLimit | None) -> None:
        """Record (or with None clear) a usage limit of one agent on a runner."""

    @abstractmethod
    async def create_enrollment(self, enrollment: RunnerEnrollment) -> None: ...

    @abstractmethod
    async def get_enrollment(self, enrollment_id: str) -> RunnerEnrollment | None: ...

    @abstractmethod
    async def find_pending_by_code(self, code_hash: str, now: datetime) -> RunnerEnrollment | None: ...

    @abstractmethod
    async def transition_enrollment(
        self, enrollment_id: str, from_status: str, fields: dict[str, Any]
    ) -> bool:
        """Set *fields* only if the enrollment is still *from_status*; whether it was."""

    @abstractmethod
    async def delete_expired_enrollments(self, before: datetime) -> None: ...


def _runner_doc(runner: Runner) -> dict[str, Any]:
    doc = runner.model_dump(mode="python", exclude=set(COMPUTED_FIELDS))
    doc["_id"] = doc.pop("id")
    return doc


def _runner_from(doc: dict[str, Any]) -> Runner:
    doc = dict(doc)
    doc["id"] = doc.pop("_id")
    for key in COMPUTED_FIELDS:
        doc.pop(key, None)
    return Runner.model_validate(doc)


def _enrollment_doc(e: RunnerEnrollment) -> dict[str, Any]:
    doc = e.model_dump(mode="python")
    doc["_id"] = doc.pop("id")
    return doc


def _enrollment_from(doc: dict[str, Any]) -> RunnerEnrollment:
    doc = dict(doc)
    doc["id"] = doc.pop("_id")
    return RunnerEnrollment.model_validate(doc)


class MongoRunnerBackend(RunnerBackend):
    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient

        self._client = AsyncIOMotorClient(uri, tz_aware=True)
        db = self._client[database]
        self._runners = db["acp_runners"]
        self._enrollments = db["acp_runner_enrollments"]

    async def list_runners(self) -> list[Runner]:
        return [_runner_from(d) for d in await self._runners.find({}).to_list(None)]

    async def get_runner(self, runner_id: str) -> Runner | None:
        doc = await self._runners.find_one({"_id": runner_id})
        return _runner_from(doc) if doc else None

    async def save_runner(self, runner: Runner) -> Runner:
        await self._runners.replace_one({"_id": runner.id}, _runner_doc(runner), upsert=True)
        return runner

    async def update_runner(self, runner_id: str, fields: dict[str, Any]) -> None:
        await self._runners.update_one({"_id": runner_id}, {"$set": fields})

    async def delete_runner(self, runner_id: str) -> None:
        await self._runners.delete_one({"_id": runner_id})

    async def set_limit(self, runner_id: str, agent: str, limit: AgentLimit | None) -> None:
        if limit is None:
            await self._runners.update_one({"_id": runner_id}, {"$unset": {f"limits.{agent}": ""}})
        else:
            await self._runners.update_one({"_id": runner_id}, {"$set": {f"limits.{agent}": limit.model_dump()}})

    async def create_enrollment(self, enrollment: RunnerEnrollment) -> None:
        await self._enrollments.insert_one(_enrollment_doc(enrollment))

    async def get_enrollment(self, enrollment_id: str) -> RunnerEnrollment | None:
        doc = await self._enrollments.find_one({"_id": enrollment_id})
        return _enrollment_from(doc) if doc else None

    async def find_pending_by_code(self, code_hash: str, now: datetime) -> RunnerEnrollment | None:
        doc = await self._enrollments.find_one(
            {"code_hash": code_hash, "status": "pending", "expires_at": {"$gt": now}}
        )
        return _enrollment_from(doc) if doc else None

    async def transition_enrollment(self, enrollment_id: str, from_status: str, fields: dict[str, Any]) -> bool:
        res = await self._enrollments.update_one({"_id": enrollment_id, "status": from_status}, {"$set": fields})
        return res.modified_count == 1

    async def delete_expired_enrollments(self, before: datetime) -> None:
        await self._enrollments.delete_many({"expires_at": {"$lt": before}})

    async def close(self) -> None:
        self._client.close()


class InMemoryRunnerBackend(RunnerBackend):
    def __init__(self) -> None:
        self.runners: dict[str, Runner] = {}
        self.enrollments: dict[str, RunnerEnrollment] = {}

    async def list_runners(self) -> list[Runner]:
        return [r.model_copy(deep=True) for r in self.runners.values()]

    async def get_runner(self, runner_id: str) -> Runner | None:
        r = self.runners.get(runner_id)
        return r.model_copy(deep=True) if r else None

    async def save_runner(self, runner: Runner) -> Runner:
        self.runners[runner.id] = runner.model_copy(deep=True)
        return runner

    async def update_runner(self, runner_id: str, fields: dict[str, Any]) -> None:
        r = self.runners.get(runner_id)
        if r is None:
            return
        self.runners[runner_id] = r.model_copy(update=fields)

    async def delete_runner(self, runner_id: str) -> None:
        self.runners.pop(runner_id, None)

    async def set_limit(self, runner_id: str, agent: str, limit: AgentLimit | None) -> None:
        r = self.runners.get(runner_id)
        if r is None:
            return
        if limit is None:
            r.limits.pop(agent, None)
        else:
            r.limits[agent] = limit.model_copy()

    async def create_enrollment(self, enrollment: RunnerEnrollment) -> None:
        self.enrollments[enrollment.id] = enrollment.model_copy(deep=True)

    async def get_enrollment(self, enrollment_id: str) -> RunnerEnrollment | None:
        e = self.enrollments.get(enrollment_id)
        return e.model_copy(deep=True) if e else None

    async def find_pending_by_code(self, code_hash: str, now: datetime) -> RunnerEnrollment | None:
        for e in self.enrollments.values():
            if e.code_hash == code_hash and e.status == "pending" and e.expires_at > now:
                return e.model_copy(deep=True)
        return None

    async def transition_enrollment(self, enrollment_id: str, from_status: str, fields: dict[str, Any]) -> bool:
        e = self.enrollments.get(enrollment_id)
        if e is None or e.status != from_status:
            return False
        self.enrollments[enrollment_id] = e.model_copy(update=fields)
        return True

    async def delete_expired_enrollments(self, before: datetime) -> None:
        for key in [k for k, e in self.enrollments.items() if e.expires_at < before]:
            del self.enrollments[key]
