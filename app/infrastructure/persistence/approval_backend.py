"""Persistent storage for data-source approval cases.

One collection, ``approvals``. Two access patterns matter and both are indexed:
the pending queue (status + created_at) that the UI panel and the Slack veto
window read, and the per-bucket decision history (history_key + decided_at)
that the meta-LLM reads to form a recommendation and that the streak check
counts.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.domain.models.approval_case import ApprovalCase

logger = logging.getLogger(__name__)


class ApprovalCaseBackend(ABC):
    """Persistent storage for :class:`ApprovalCase` rows."""

    @abstractmethod
    async def create(self, case: ApprovalCase) -> ApprovalCase: ...

    @abstractmethod
    async def get(self, case_id: str) -> ApprovalCase | None: ...

    @abstractmethod
    async def update(self, case: ApprovalCase) -> ApprovalCase: ...

    @abstractmethod
    async def claim_for_decision(self, case_id: str) -> ApprovalCase | None:
        """Atomically take a pending case out of the queue, or return None."""

    @abstractmethod
    async def list(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        datasource_id: str | None = None,
        run_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ApprovalCase]: ...

    @abstractmethod
    async def count(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        datasource_id: str | None = None,
        run_id: str | None = None,
        operation: str | None = None,
    ) -> int: ...

    @abstractmethod
    async def history(self, history_key: str, limit: int = 20) -> list[ApprovalCase]:
        """Decided cases in one bucket, newest first.

        Try-run confirmations are excluded. They are real approvals and they
        stay in the audit trail, but the person who confirmed one is the person
        who wrote the operation, and a streak assembled out of self-approvals
        would let an author grant the meta-LLM autonomy over their own delete
        by clicking Try run ten times.
        """

    @abstractmethod
    async def find_pending_for_run(
        self, run_id: str, step_id: str | None = None
    ) -> ApprovalCase | None:
        """The still-open case for a run (optionally for one step of it).

        A LangGraph node re-runs from the top every time it resumes from an
        ``interrupt``, so the ``data_source`` step asks this before opening
        anything: finding its own pending case is what stops one paused delete
        from writing a new row — and re-reading the upstream list — on every
        resume.
        """


# ---------------------------------------------------------------------------
# MongoDB implementation
# ---------------------------------------------------------------------------

class MongoApprovalBackend(ApprovalCaseBackend):
    """Reads and writes approval cases in the ``approvals`` collection."""

    _COLLECTION = "approvals"
    # Statuses that count as "a person (or the meta-LLM) answered", which is
    # what the history and the streak are about. An expired or cancelled case
    # answers nothing and must not extend a streak.
    _DECIDED = ("approved", "rejected")
    # Surfaces whose decisions carry weight in the history. Try-run is absent:
    # see ``ApprovalCaseBackend.history``.
    _HISTORY_SURFACES = ("workflow", "mcp")

    def __init__(self, uri: str, database: str) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(uri)
        self._col = self._client[database][self._COLLECTION]
        self._indexes_ready = False

    async def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        try:
            await self._col.create_index([("status", 1), ("created_at", -1)])
            await self._col.create_index([("history_key", 1), ("decided_at", -1)])
            await self._col.create_index([("run_id", 1), ("status", 1)])
            self._indexes_ready = True
        except Exception:
            # An index we could not create costs latency, never correctness.
            logger.warning("approvals: index creation failed", exc_info=True)

    async def create(self, case: ApprovalCase) -> ApprovalCase:
        await self.ensure_indexes()
        case.touch()
        await self._col.insert_one(self._to_doc(case))
        return case

    async def get(self, case_id: str) -> ApprovalCase | None:
        doc = await self._col.find_one({"_id": case_id})
        return self._from_doc(doc) if doc else None

    async def update(self, case: ApprovalCase) -> ApprovalCase:
        case.touch()
        await self._col.replace_one({"_id": case.id}, self._to_doc(case), upsert=True)
        return case

    async def claim_for_decision(self, case_id: str) -> ApprovalCase | None:
        """Flip pending → pending, once, for exactly one caller.

        Same TOCTOU that ``claim_for_resume`` closes on runs: two people
        clicking Approve and Reject within the same second must not both write
        a decision, and a Slack button must not race the UI. The claim marks
        the row ``deciding`` so the loser sees a decided case and stops.
        """
        from pymongo import ReturnDocument
        from datetime import datetime, timezone
        doc = await self._col.find_one_and_update(
            {"_id": case_id, "status": "pending"},
            {"$set": {"status": "deciding", "updated_at": datetime.now(timezone.utc)}},
            return_document=ReturnDocument.AFTER,
        )
        return self._from_doc(doc) if doc else None

    def _query(
        self,
        status: str | None,
        workflow_id: str | None,
        datasource_id: str | None,
        run_id: str | None,
        operation: str | None = None,
    ) -> dict:
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        if workflow_id:
            query["workflow_id"] = workflow_id
        if datasource_id:
            query["datasource_id"] = datasource_id
        if run_id:
            query["run_id"] = run_id
        if operation:
            query["operation"] = operation
        return query

    async def list(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        datasource_id: str | None = None,
        run_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ApprovalCase]:
        cursor = (
            self._col.find(self._query(status, workflow_id, datasource_id, run_id))
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )
        return [self._from_doc(d) for d in await cursor.to_list(length=None)]

    async def count(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        datasource_id: str | None = None,
        run_id: str | None = None,
        operation: str | None = None,
    ) -> int:
        return await self._col.count_documents(
            self._query(status, workflow_id, datasource_id, run_id, operation)
        )

    async def history(self, history_key: str, limit: int = 20) -> list[ApprovalCase]:
        cursor = (
            self._col.find({
                "history_key": history_key,
                "status": {"$in": list(self._DECIDED)},
                "surface": {"$in": list(self._HISTORY_SURFACES)},
            })
            .sort("decided_at", -1)
            .limit(limit)
        )
        return [self._from_doc(d) for d in await cursor.to_list(length=None)]

    async def find_pending_for_run(
        self, run_id: str, step_id: str | None = None
    ) -> ApprovalCase | None:
        query: dict[str, Any] = {
            "run_id": run_id,
            "status": {"$in": ["pending", "deciding"]},
        }
        if step_id is not None:
            query["step_id"] = step_id
        doc = await self._col.find_one(query, sort=[("created_at", -1)])
        return self._from_doc(doc) if doc else None

    async def close(self) -> None:
        self._client.close()

    @staticmethod
    def _to_doc(case: ApprovalCase) -> dict[str, Any]:
        data = case.model_dump(mode="python")
        data["_id"] = data.pop("id")
        return data

    @staticmethod
    def _from_doc(doc: dict[str, Any]) -> ApprovalCase:
        data = dict(doc)
        data["id"] = data.pop("_id")
        # ``deciding`` is an internal claim marker, not a status the domain
        # model knows: a claimed-but-unwritten case is still pending.
        if data.get("status") == "deciding":
            data["status"] = "pending"
        return ApprovalCase.model_validate(data)


class InMemoryApprovalBackend(ApprovalCaseBackend):
    """Non-persistent backend used by tests and by Mongo-less deployments."""

    def __init__(self) -> None:
        self._cases: dict[str, ApprovalCase] = {}
        self._claimed: set[str] = set()

    async def create(self, case: ApprovalCase) -> ApprovalCase:
        self._cases[case.id] = case
        return case

    async def get(self, case_id: str) -> ApprovalCase | None:
        return self._cases.get(case_id)

    async def update(self, case: ApprovalCase) -> ApprovalCase:
        case.touch()
        self._cases[case.id] = case
        return case

    async def claim_for_decision(self, case_id: str) -> ApprovalCase | None:
        case = self._cases.get(case_id)
        if case is None or case.status != "pending" or case_id in self._claimed:
            return None
        self._claimed.add(case_id)
        return case

    def _match(
        self, case: ApprovalCase, status, workflow_id, datasource_id, run_id
    ) -> bool:
        return (
            (status is None or case.status == status)
            and (workflow_id is None or case.workflow_id == workflow_id)
            and (datasource_id is None or case.datasource_id == datasource_id)
            and (run_id is None or case.run_id == run_id)
        )

    async def list(
        self, *, status=None, workflow_id=None, datasource_id=None, run_id=None,
        limit: int = 50, offset: int = 0,
    ) -> list[ApprovalCase]:
        rows = [
            c for c in self._cases.values()
            if self._match(c, status, workflow_id, datasource_id, run_id)
        ]
        rows.sort(key=lambda c: c.created_at, reverse=True)
        return rows[offset: offset + limit]

    async def count(
        self, *, status=None, workflow_id=None, datasource_id=None, run_id=None,
        operation=None,
    ) -> int:
        return sum(
            1 for c in self._cases.values()
            if self._match(c, status, workflow_id, datasource_id, run_id)
            and (not operation or c.operation == operation)
        )

    async def history(self, history_key: str, limit: int = 20) -> list[ApprovalCase]:
        rows = [
            c for c in self._cases.values()
            if c.history_key == history_key
            and c.status in ("approved", "rejected")
            and c.surface != "try_run"
        ]
        rows.sort(key=lambda c: c.decided_at or c.created_at, reverse=True)
        return rows[:limit]

    async def find_pending_for_run(
        self, run_id: str, step_id: str | None = None
    ) -> ApprovalCase | None:
        rows = [
            c for c in self._cases.values()
            if c.run_id == run_id and c.status == "pending"
            and (step_id is None or c.step_id == step_id)
        ]
        rows.sort(key=lambda c: c.created_at, reverse=True)
        return rows[0] if rows else None
