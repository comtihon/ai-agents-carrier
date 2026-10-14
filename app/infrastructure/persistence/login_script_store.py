"""The login script as stored in MongoDB, editable at runtime by a super admin.

One document, ``{_id: "login_script", script, version, updated_at,
updated_by}`` in the ``auth_settings`` collection. ``version`` only ever grows
— disabling the script sets ``script`` to ``None`` and still bumps it — so a
replica can tell "changed since I last looked" from one cheap read of that
field, without fetching or compiling the script.

A script set through ``AUTH_LOGIN_SCRIPT`` / ``AUTH_LOGIN_SCRIPT_PATH`` wins
over this one; see app.application.login_admin.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_DOC_ID = "login_script"


@dataclass(frozen=True)
class StoredLoginScript:
    script: str | None
    version: int
    updated_at: datetime | None
    updated_by: str | None


class LoginScriptStore(ABC):
    @abstractmethod
    async def get(self) -> StoredLoginScript | None:
        """The stored document, or ``None`` when nothing was ever saved."""

    @abstractmethod
    async def version(self) -> int:
        """The current version (0 when nothing was ever saved). Cheap."""

    @abstractmethod
    async def save(self, script: str | None, updated_by: str | None) -> StoredLoginScript:
        """Store *script* (``None``: disabled) and bump the version."""


class InMemoryLoginScriptStore(LoginScriptStore):
    """For tests and for running without MongoDB."""

    def __init__(self) -> None:
        self.doc: StoredLoginScript | None = None

    async def get(self) -> StoredLoginScript | None:
        return self.doc

    async def version(self) -> int:
        return self.doc.version if self.doc else 0

    async def save(self, script, updated_by) -> StoredLoginScript:
        self.doc = StoredLoginScript(
            script=script,
            version=(self.doc.version if self.doc else 0) + 1,
            updated_at=datetime.now(timezone.utc),
            updated_by=updated_by,
        )
        return self.doc


class MongoLoginScriptStore(LoginScriptStore):
    def __init__(self, database: Any) -> None:
        self._collection = database["auth_settings"]

    @staticmethod
    def _doc(doc: dict[str, Any]) -> StoredLoginScript:
        updated_at = doc.get("updated_at")
        if isinstance(updated_at, datetime) and updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return StoredLoginScript(
            script=doc.get("script"),
            version=int(doc.get("version") or 0),
            updated_at=updated_at,
            updated_by=doc.get("updated_by"),
        )

    async def get(self) -> StoredLoginScript | None:
        doc = await self._collection.find_one({"_id": _DOC_ID})
        return self._doc(doc) if doc else None

    async def version(self) -> int:
        doc = await self._collection.find_one({"_id": _DOC_ID}, {"version": 1})
        return int(doc.get("version") or 0) if doc else 0

    async def save(self, script, updated_by) -> StoredLoginScript:
        from pymongo import ReturnDocument

        doc = await self._collection.find_one_and_update(
            {"_id": _DOC_ID},
            {
                "$set": {
                    "script": script,
                    "updated_at": datetime.now(timezone.utc),
                    "updated_by": updated_by,
                },
                "$inc": {"version": 1},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return self._doc(doc)
