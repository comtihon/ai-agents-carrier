"""Carrier's own users and teams.

Two collections:

* ``users`` — one document per authenticated caller, keyed by the token's
  ``sub``: display name, email, when they were last seen, the roles and
  super-admin flag they last called with, and their team membership
  (``team_id`` + ``team_role``). Membership lives here rather than
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
from typing import Any, Iterable

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
    # Effective roles (after the login script) and super-admin flag of the
    # caller's most recent recorded request. Informational: what a request
    # may do is decided from its own token, never from these.
    roles: list[str] | None = None
    is_super_admin: bool | None = None


@dataclass
class TeamRecord:
    id: str
    name: str
    created_at: datetime


class TeamNameTaken(Exception):
    pass


class TeamDirectory(ABC):
    def __init__(self) -> None:
        # sub -> (when last recorded, the roles/super-admin flag recorded then)
        self._touched: dict[str, tuple[float, Any]] = {}

    # ── users ──
    async def touch_user(
        self,
        sub: str,
        name: str | None,
        email: str | None,
        roles: Iterable[str] | None = None,
        is_super_admin: bool | None = None,
    ) -> None:
        """Record that *sub* called, at most once per interval per process.

        A change of roles or super-admin flag is recorded at once, interval or
        not, so the stored roles follow a login that changed them.
        """
        role_list = sorted(set(roles)) if roles is not None else None
        access = (tuple(role_list) if role_list is not None else None, is_super_admin)
        now = time.monotonic()
        last = self._touched.get(sub)
        if last is not None and now - last[0] < _TOUCH_INTERVAL_SECONDS and last[1] == access:
            return
        self._touched[sub] = (now, access)
        await self._upsert_seen(
            sub, name, email, datetime.now(timezone.utc), role_list, is_super_admin
        )

    async def record_profile(self, sub: str, name: str | None, email: str | None) -> None:
        """Record *sub* now, bypassing the touch interval (a login just ran).

        Roles are not known here; the request's own touch records them.
        """
        self._touched.pop(sub, None)
        await self._upsert_seen(sub, name, email, datetime.now(timezone.utc), None, None)

    def forget_touch(self, sub: str) -> None:
        """Make the next touch of *sub* write, whatever the interval."""
        self._touched.pop(sub, None)

    @abstractmethod
    async def _upsert_seen(
        self,
        sub: str,
        name: str | None,
        email: str | None,
        when: datetime,
        roles: list[str] | None,
        is_super_admin: bool | None,
    ) -> None:
        """Upsert *sub*. ``None`` for a field leaves the stored value alone."""

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

    async def delete_user(self, sub: str) -> bool:
        """Forget *sub* (membership included). False when there was nothing."""
        self.forget_touch(sub)
        return await self._delete_user(sub)

    @abstractmethod
    async def _delete_user(self, sub: str) -> bool: ...

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


def _aware(value: Any) -> Any:
    """MongoDB hands datetimes back naive; they were stored as UTC."""
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _new_team_id() -> str:
    return uuid.uuid4().hex[:12]


class InMemoryTeamDirectory(TeamDirectory):
    """For tests and for running without MongoDB."""

    def __init__(self) -> None:
        super().__init__()
        self.users: dict[str, UserRecord] = {}
        self.teams: dict[str, TeamRecord] = {}

    async def _upsert_seen(self, sub, name, email, when, roles, is_super_admin) -> None:
        user = self.users.setdefault(sub, UserRecord(sub=sub))
        user.name = name or user.name
        user.email = email or user.email
        user.last_seen = when
        if roles is not None:
            user.roles = list(roles)
        if is_super_admin is not None:
            user.is_super_admin = is_super_admin

    async def _delete_user(self, sub: str) -> bool:
        return self.users.pop(sub, None) is not None

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
        # Shared with the other small auth stores (login_script_store).
        self.database = db
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
            last_seen=_aware(doc.get("last_seen")),
            roles=doc.get("roles"),
            is_super_admin=doc.get("is_super_admin"),
        )

    @staticmethod
    def _team(doc: dict[str, Any]) -> TeamRecord:
        return TeamRecord(id=doc["_id"], name=doc["name"], created_at=doc["created_at"])

    async def _upsert_seen(self, sub, name, email, when, roles, is_super_admin) -> None:
        fields: dict[str, Any] = {"last_seen": when}
        if name:
            fields["name"] = name
        if email:
            fields["email"] = email
        if roles is not None:
            fields["roles"] = list(roles)
        if is_super_admin is not None:
            fields["is_super_admin"] = is_super_admin
        await self._users.update_one({"_id": sub}, {"$set": fields}, upsert=True)

    async def _delete_user(self, sub: str) -> bool:
        result = await self._users.delete_one({"_id": sub})
        return result.deleted_count > 0

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
