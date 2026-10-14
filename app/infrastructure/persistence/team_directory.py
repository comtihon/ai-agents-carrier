"""Carrier's own users and teams.

Two collections:

* ``users`` — one document per authenticated caller, keyed by the token's
  ``sub``: display name, email, when they were last seen, and their team
  membership (``team_id`` + ``team_role``). Membership lives here rather than
  on the team so "at most one team per user" holds by construction.
* ``teams`` — ``{_id, name, created_at}``. Names are unique.

What a membership *means*, and who may change one, is app.application.teams.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# A caller is recorded at most this often; membership is still read every time.
_TOUCH_INTERVAL_SECONDS = 60


@dataclass
class UserRecord:
    sub: str
    name: str | None = None
    email: str | None = None
    team_id: str | None = None
    team_role: str | None = None
    last_seen: datetime | None = None


@dataclass
class TeamRecord:
    id: str
    name: str
    created_at: datetime


class TeamNameTaken(Exception):
    pass


class TeamDirectory(ABC):
    def __init__(self) -> None:
        self._touched: dict[str, float] = {}

    # ── users ──
    async def touch_user(self, sub: str, name: str | None, email: str | None) -> None:
        """Record that *sub* called, at most once per interval per process."""
        now = time.monotonic()
        last = self._touched.get(sub)
        if last is not None and now - last < _TOUCH_INTERVAL_SECONDS:
            return
        self._touched[sub] = now
        await self._upsert_seen(sub, name, email, datetime.now(timezone.utc))

    async def record_profile(self, sub: str, name: str | None, email: str | None) -> None:
        """Record *sub* now, bypassing the touch interval (a login just ran)."""
        self._touched[sub] = time.monotonic()
        await self._upsert_seen(sub, name, email, datetime.now(timezone.utc))

    @abstractmethod
    async def _upsert_seen(
        self, sub: str, name: str | None, email: str | None, when: datetime
    ) -> None: ...

    @abstractmethod
    async def get_user(self, sub: str) -> UserRecord | None: ...

    @abstractmethod
    async def list_users(self) -> list[UserRecord]: ...

    @abstractmethod
    async def set_membership(self, sub: str, team_id: str | None, role: str | None) -> None:
        """Put *sub* in *team_id* with *role* (None, None: in no team)."""

    @abstractmethod
    async def clear_team(self, team_id: str) -> None:
        """Remove every member of *team_id*."""

    # ── teams ──
    @abstractmethod
    async def list_teams(self) -> list[TeamRecord]: ...

    @abstractmethod
    async def get_team(self, team_id: str) -> TeamRecord | None: ...

    @abstractmethod
    async def create_team(self, name: str) -> TeamRecord: ...

    @abstractmethod
    async def rename_team(self, team_id: str, name: str) -> TeamRecord | None: ...

    @abstractmethod
    async def delete_team(self, team_id: str) -> None: ...

    async def members(self, team_id: str) -> list[UserRecord]:
        return [u for u in await self.list_users() if u.team_id == team_id]


def _new_team_id() -> str:
    return uuid.uuid4().hex[:12]


class InMemoryTeamDirectory(TeamDirectory):
    """For tests and for running without MongoDB."""

    def __init__(self) -> None:
        super().__init__()
        self.users: dict[str, UserRecord] = {}
        self.teams: dict[str, TeamRecord] = {}

    async def _upsert_seen(self, sub, name, email, when) -> None:
        user = self.users.setdefault(sub, UserRecord(sub=sub))
        user.name = name or user.name
        user.email = email or user.email
        user.last_seen = when

    async def get_user(self, sub: str) -> UserRecord | None:
        return self.users.get(sub)

    async def list_users(self) -> list[UserRecord]:
        return list(self.users.values())

    async def set_membership(self, sub, team_id, role) -> None:
        user = self.users.setdefault(sub, UserRecord(sub=sub))
        user.team_id, user.team_role = team_id, role

    async def clear_team(self, team_id: str) -> None:
        for user in self.users.values():
            if user.team_id == team_id:
                user.team_id = user.team_role = None

    async def list_teams(self) -> list[TeamRecord]:
        return sorted(self.teams.values(), key=lambda t: t.name.lower())

    async def get_team(self, team_id: str) -> TeamRecord | None:
        return self.teams.get(team_id)

    async def create_team(self, name: str) -> TeamRecord:
        if any(t.name == name for t in self.teams.values()):
            raise TeamNameTaken(name)
        team = TeamRecord(id=_new_team_id(), name=name, created_at=datetime.now(timezone.utc))
        self.teams[team.id] = team
        return team

    async def rename_team(self, team_id: str, name: str) -> TeamRecord | None:
        team = self.teams.get(team_id)
        if team is None:
            return None
        if any(t.name == name and t.id != team_id for t in self.teams.values()):
            raise TeamNameTaken(name)
        team.name = name
        return team

    async def delete_team(self, team_id: str) -> None:
        self.teams.pop(team_id, None)


class MongoTeamDirectory(TeamDirectory):
    def __init__(self, uri: str, database: str) -> None:
        super().__init__()
        from motor.motor_asyncio import AsyncIOMotorClient
        self._client = AsyncIOMotorClient(uri)
        db = self._client[database]
        self._users = db["users"]
        self._teams = db["teams"]

    async def ensure_indexes(self) -> None:
        await self._teams.create_index("name", unique=True)
        await self._users.create_index("team_id")

    @staticmethod
    def _user(doc: dict[str, Any]) -> UserRecord:
        return UserRecord(
            sub=doc["_id"],
            name=doc.get("name"),
            email=doc.get("email"),
            team_id=doc.get("team_id"),
            team_role=doc.get("team_role"),
            last_seen=doc.get("last_seen"),
        )

    @staticmethod
    def _team(doc: dict[str, Any]) -> TeamRecord:
        return TeamRecord(id=doc["_id"], name=doc["name"], created_at=doc["created_at"])

    async def _upsert_seen(self, sub, name, email, when) -> None:
        fields: dict[str, Any] = {"last_seen": when}
        if name:
            fields["name"] = name
        if email:
            fields["email"] = email
        await self._users.update_one({"_id": sub}, {"$set": fields}, upsert=True)

    async def get_user(self, sub: str) -> UserRecord | None:
        doc = await self._users.find_one({"_id": sub})
        return self._user(doc) if doc else None

    async def list_users(self) -> list[UserRecord]:
        return [self._user(d) for d in await self._users.find({}).to_list(None)]

    async def members(self, team_id: str) -> list[UserRecord]:
        return [self._user(d) for d in await self._users.find({"team_id": team_id}).to_list(None)]

    async def set_membership(self, sub, team_id, role) -> None:
        await self._users.update_one(
            {"_id": sub}, {"$set": {"team_id": team_id, "team_role": role}}, upsert=True
        )

    async def clear_team(self, team_id: str) -> None:
        await self._users.update_many(
            {"team_id": team_id}, {"$set": {"team_id": None, "team_role": None}}
        )

    async def list_teams(self) -> list[TeamRecord]:
        docs = await self._teams.find({}).sort("name", 1).to_list(None)
        return [self._team(d) for d in docs]

    async def get_team(self, team_id: str) -> TeamRecord | None:
        doc = await self._teams.find_one({"_id": team_id})
        return self._team(doc) if doc else None

    async def create_team(self, name: str) -> TeamRecord:
        from pymongo.errors import DuplicateKeyError
        if await self._teams.find_one({"name": name}):
            raise TeamNameTaken(name)
        team = TeamRecord(id=_new_team_id(), name=name, created_at=datetime.now(timezone.utc))
        try:
            await self._teams.insert_one(
                {"_id": team.id, "name": team.name, "created_at": team.created_at}
            )
        except DuplicateKeyError as exc:
            raise TeamNameTaken(name) from exc
        return team

    async def rename_team(self, team_id: str, name: str) -> TeamRecord | None:
        from pymongo.errors import DuplicateKeyError
        clash = await self._teams.find_one({"name": name, "_id": {"$ne": team_id}})
        if clash:
            raise TeamNameTaken(name)
        try:
            result = await self._teams.update_one({"_id": team_id}, {"$set": {"name": name}})
        except DuplicateKeyError as exc:
            raise TeamNameTaken(name) from exc
        if result.matched_count == 0:
            return None
        return await self.get_team(team_id)

    async def delete_team(self, team_id: str) -> None:
        await self._teams.delete_one({"_id": team_id})

    async def close(self) -> None:
        self._client.close()
