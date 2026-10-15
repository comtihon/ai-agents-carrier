"""Teams: who is in which, who may change that, and the caller's team.

A user is in at most one team, or none, with a role in it: ``member`` or
``admin``. Teams live in carrier's own store (see
app.infrastructure.persistence.team_directory), so they work with any OIDC
provider.

Two kinds of administrator:

* **Super admin** — a caller holding a role mapped to ADMIN (evaluated
  strictly; see ``Subject.is_super_admin``). System-wide: creates, renames and deletes
  teams, moves users between teams, and sees and manages every resource and run
  regardless of any team they happen to be in.
* **Team admin** — ``team_role == "admin"`` in one team, a status only a super
  admin assigns (here, or through the login script a super admin maintains).
  May add users who are in *no* team to their team as members and remove its
  (non-admin) members. May not make anyone — themselves included — a team admin,
  nor demote or remove one: who administers a team is the super admin's call.
  Nothing outside their team. Over the team's resources one extra right: they
  may change the visibility of a resource shared with their team (see
  app.application.visibility.can_change_visibility).
* **Everyone else** — no user or team management at all.

With nothing bound (authentication off, a test, the platform itself) every
check here passes, as everywhere else in visibility.

The caller's team is looked up from the directory on every request (see
:func:`resolve_subject`), so a membership change applies on the next request,
on every replica, with nothing to invalidate.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Iterable

from app.application.visibility import NotOwner, ResourceNotFound, VisibilityError
from app.infrastructure.auth.authorization import Subject, get_current_subject

logger = logging.getLogger(__name__)

ROLES = ("member", "admin")


class TeamConflict(VisibilityError):
    status_code = 409


class TeamInvalid(VisibilityError):
    status_code = 422


# ─── Team names ───────────────────────────────────────────────────────────────
# Resources store only ``team_id``; the name shown next to one is looked up
# here so a rename shows everywhere at once. Refreshed from the directory at
# most every few seconds per process (see resolve_subject) and on every change
# made through this process.

_NAMES: dict[str, str] = {}
_NAMES_LOADED_AT: float = 0.0
_NAMES_TTL_SECONDS = 30.0


def team_name(team_id: str | None) -> str | None:
    return _NAMES.get(team_id) if team_id else None


def remember_teams(teams: Iterable[Any], *, complete: bool = False) -> None:
    global _NAMES_LOADED_AT
    if complete:
        _NAMES.clear()
        _NAMES_LOADED_AT = time.monotonic()
    for team in teams:
        _NAMES[team.id] = team.name


def forget_team(team_id: str) -> None:
    _NAMES.pop(team_id, None)


async def _refresh_names(directory: Any) -> None:
    if time.monotonic() - _NAMES_LOADED_AT < _NAMES_TTL_SECONDS:
        return
    remember_teams(await directory.list_teams(), complete=True)


# ─── The caller ───────────────────────────────────────────────────────────────

async def resolve_subject(directory: Any, subject: Subject) -> Subject:
    """*subject* with its team filled in, recording the caller as seen.

    Fails closed: if the directory cannot be read, the caller is treated as in
    no team (they see less, never more).
    """
    if directory is None or not subject.id:
        return subject
    try:
        await directory.touch_user(
            subject.id, subject.name, subject.email, subject.roles, subject.is_super_admin
        )
        user = await directory.get_user(subject.id)
        await _refresh_names(directory)
    except Exception:
        logger.exception("team lookup failed for %s; treating as in no team", subject.id)
        return subject
    if user is None or not user.team_id:
        return subject
    return replace(subject, team_id=user.team_id, team_role=user.team_role or "member")


def directory_of(holder: Any) -> Any:
    """The team directory of a container (or app state holding one), if any."""
    container = getattr(holder, "container", holder)
    return getattr(container, "team_directory", None)


def is_super_admin(subject: Subject | None = None) -> bool:
    subject = subject if subject is not None else get_current_subject()
    return subject is None or subject.is_super_admin


def is_team_admin(team_id: str, subject: Subject | None = None) -> bool:
    subject = subject if subject is not None else get_current_subject()
    return subject is not None and subject.team_id == team_id and subject.team_role == "admin"


def require_super_admin(action: str) -> None:
    if not is_super_admin():
        raise NotOwner(f"Only a super admin may {action}")


def require_team_manager(team_id: str) -> None:
    if not (is_super_admin() or is_team_admin(team_id)):
        raise NotOwner("Only a super admin or an admin of this team may manage its members")


# ─── Team management ──────────────────────────────────────────────────────────

def _user_dict(user: Any) -> dict[str, Any]:
    team = None
    if user.team_id:
        team = {
            "id": user.team_id,
            "name": team_name(user.team_id),
            "role": user.team_role or "member",
        }
    return {
        "sub": user.sub,
        "name": user.name,
        "email": user.email,
        "team": team,
        # As of the user's most recent recorded request (after the login script).
        "is_super_admin": bool(user.is_super_admin),
        "roles": sorted(user.roles or []),
        "last_seen": user.last_seen.isoformat() if user.last_seen else None,
        # Flat forms of `team`, kept for existing clients.
        "team_id": user.team_id,
        "team_name": team_name(user.team_id),
        "team_role": user.team_role,
    }


async def team_dict(directory: Any, team: Any) -> dict[str, Any]:
    members = await directory.members(team.id)
    return {
        "id": team.id,
        "name": team.name,
        "created_at": team.created_at,
        "members": [
            {"sub": m.sub, "name": m.name, "email": m.email, "role": m.team_role or "member"}
            for m in sorted(members, key=lambda m: (m.name or m.sub).lower())
        ],
    }


async def list_teams(directory: Any) -> list[dict[str, Any]]:
    """Every team for a super admin; the caller's own team otherwise."""
    teams = await directory.list_teams()
    remember_teams(teams, complete=True)
    if not is_super_admin():
        subject = get_current_subject()
        own = subject.team_id if subject is not None else None
        teams = [t for t in teams if t.id == own]
    return [await team_dict(directory, t) for t in teams]


async def _team_or_404(directory: Any, team_id: str) -> Any:
    team = await directory.get_team(team_id)
    if team is None:
        raise ResourceNotFound(f"Team '{team_id}' not found")
    return team


def _clean_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise TeamInvalid("A team needs a name")
    return name


async def create_team(directory: Any, name: str) -> dict[str, Any]:
    from app.infrastructure.persistence.team_directory import TeamNameTaken
    require_super_admin("create teams")
    name = _clean_name(name)
    try:
        team = await directory.create_team(name)
    except TeamNameTaken:
        raise TeamConflict(f"A team named '{name}' already exists")
    remember_teams([team])
    return await team_dict(directory, team)


async def rename_team(directory: Any, team_id: str, name: str) -> dict[str, Any]:
    from app.infrastructure.persistence.team_directory import TeamNameTaken
    require_super_admin("rename teams")
    name = _clean_name(name)
    try:
        team = await directory.rename_team(team_id, name)
    except TeamNameTaken:
        raise TeamConflict(f"A team named '{name}' already exists")
    if team is None:
        raise ResourceNotFound(f"Team '{team_id}' not found")
    remember_teams([team])
    return await team_dict(directory, team)


async def delete_team(directory: Any, team_id: str, backends: Iterable[Any]) -> None:
    """Refused while any agent, data source, script or workflow belongs to it."""
    from app.application.visibility import unwrap
    require_super_admin("delete teams")
    await _team_or_404(directory, team_id)
    owned = 0
    for backend in backends:
        if backend is None:
            continue
        owned += sum(
            1 for r in await unwrap(backend).list() if getattr(r, "team_id", None) == team_id
        )
    if owned:
        raise TeamConflict(
            f"Team '{team_id}' still has {owned} resource(s) shared with it; "
            f"change their visibility or delete them first"
        )
    await directory.clear_team(team_id)
    await directory.delete_team(team_id)
    forget_team(team_id)


async def set_member(directory: Any, team_id: str, sub: str, role: str | None) -> dict[str, Any]:
    """Add *sub* to the team, or change their role in it.

    A team admin may add only users in no team, and only as members; moving a
    user out of another team, and making or unmaking a team admin, is a super
    admin's call.
    """
    require_team_manager(team_id)
    team = await _team_or_404(directory, team_id)
    role = role or "member"
    if role not in ROLES:
        raise TeamInvalid(f"Unknown role '{role}'; expected one of {', '.join(ROLES)}")
    user = await directory.get_user(sub)
    current_team = user.team_id if user is not None else None
    if current_team and current_team != team_id and not is_super_admin():
        raise TeamConflict(
            "User is in another team; only a super admin can move users between teams"
        )
    if not is_super_admin():
        # Team admin status is assigned (and taken away) by a super admin only.
        if role == "admin":
            raise NotOwner("Only a super admin may make someone a team admin")
        if current_team == team_id and user.team_role == "admin":
            raise NotOwner("Only a super admin may change a team admin's role")
    await directory.set_membership(sub, team_id, role)
    return await team_dict(directory, team)


async def remove_member(directory: Any, team_id: str, sub: str) -> dict[str, Any]:
    require_team_manager(team_id)
    team = await _team_or_404(directory, team_id)
    user = await directory.get_user(sub)
    if user is None or user.team_id != team_id:
        raise ResourceNotFound(f"User '{sub}' is not a member of team '{team_id}'")
    if user.team_role == "admin" and not is_super_admin():
        raise NotOwner("Only a super admin may remove a team admin")
    await directory.set_membership(sub, None, None)
    return await team_dict(directory, team)


async def list_users(directory: Any) -> list[dict[str, Any]]:
    """All users for a super admin; for a team admin, users in no team plus
    their own team's members (whom they can add or manage)."""
    subject = get_current_subject()
    users = await directory.list_users()
    await _refresh_names(directory)
    if not is_super_admin():
        if subject is None or subject.team_role != "admin" or not subject.team_id:
            raise NotOwner("Only a super admin or a team admin may list users")
        users = [u for u in users if not u.team_id or u.team_id == subject.team_id]
    users.sort(key=lambda u: (u.name or u.email or u.sub).lower())
    return [_user_dict(u) for u in users]


async def _user_or_404(directory: Any, sub: str) -> Any:
    user = await directory.get_user(sub)
    if user is None:
        raise ResourceNotFound(f"User '{sub}' not found")
    return user


async def set_user_team(
    directory: Any, sub: str, team_id: str | None, role: str | None
) -> dict[str, Any]:
    """Super admin: put *sub* in *team_id* with *role*, or in no team (None).

    Not bound by the team-admin rules (a team keeps an admin): a super admin
    may leave a team without one. A login script that assigns teams overrides
    this on that user's next login.
    """
    require_super_admin("change a user's team")
    await _user_or_404(directory, sub)
    if team_id:
        await _team_or_404(directory, team_id)
        role = role or "member"
        if role not in ROLES:
            raise TeamInvalid(f"Unknown role '{role}'; expected one of {', '.join(ROLES)}")
        await directory.set_membership(sub, team_id, role)
    else:
        await directory.set_membership(sub, None, None)
    await _refresh_names(directory)
    return _user_dict(await directory.get_user(sub))


async def delete_user(directory: Any, sub: str) -> None:
    """Super admin: forget *sub*, membership included.

    Their resources keep their owner id. A user who calls again is recorded
    again, in no team (unless a login script assigns one).
    """
    require_super_admin("delete users")
    if not await directory.delete_user(sub):
        raise ResourceNotFound(f"User '{sub}' not found")


async def team_of(directory: Any, sub: str | None) -> str | None:
    """The team *sub* is in now (None when unknown or no directory)."""
    if directory is None or not sub:
        return None
    user = await directory.get_user(sub)
    return user.team_id if user is not None else None
