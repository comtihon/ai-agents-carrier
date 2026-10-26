"""Who may see and change an agent, data source, script or workflow.

Every one of the four is ``public`` (visible to and usable by every user),
``team`` (visible to and usable by the members of one team — see
app.application.teams) or ``private`` (its owner's alone). To anyone outside a
resource's scope it does not exist: absent from lists, 404 by id, never offered
in a picker or to the chat agent.

Changing a resource is its owner's, with one widening: members of a *team
workflow*'s team may edit it too (``OwnedResource.team_editable``). Deleting a
resource stays with the owner; changing its visibility or team with the owner
and, for a resource shared with a team, that team's admins
(:func:`can_change_visibility`). A super admin sees and manages everything,
whatever team they are in.

Enforcement lives in one place, :class:`VisibilityScopedBackend`, which the
container wraps around each of the four backends. Every surface — the REST
routes, the management MCP tools, the chat agent's tools, the dynamic
dispatcher's agent roster, run-time lookups — reads through the same backend,
so none of them can disagree about what a caller may see.

The caller is the ambient subject (see
``app.infrastructure.auth.authorization``). When none is bound — OAuth off, a
test, or the platform itself running a cron/Pub/Sub/webhook trigger — every
check here is a no-op, so a deployment without authentication behaves exactly
as it did before visibility existed, and a triggered run of a private workflow
can still resolve its owner's private resources.

References are checked on save. Scopes rank public > team > private, and a
resource may only use what is (a) visible to the caller and (b) at least as
widely visible as itself — otherwise someone who can see (and run) the
referrer would run something they cannot see:

* a public resource uses public ones only;
* a team resource uses public ones and its own team's;
* a private resource uses public ones, its owner's team's, and its owner's
  private ones.

The converse is checked when a resource narrows (public → team → private, or
moves to another team): refused while a resource that uses it would break the
rule.

Grants (data sources only) open a source to someone its scope leaves out:

* a **user grant** lets that user see the source and use it in resources
  private to them — not in a team or public one, whose other viewers were
  never granted anything;
* a **workflow grant** lets that one workflow use it, whatever the workflow's
  scope. Whoever edits the workflow still cannot see the source anywhere else,
  so they cannot put it in another workflow.

Either carries an access level: ``read`` permits only operations that do not
write (``app.infrastructure.datasources.destructive.operation_writes``),
``read_write`` all of them. A grant never lets the grantee change, delete or
re-share the source. Grants are checked where scope is — on save, as part of
the reference rules, and at run time (:func:`workflow_grant_access`) — so
revoking one stops the next run that relied on it.

Legacy documents, stored before these fields existed, load as public with no
owner. An ownerless resource stays editable by any caller with WRITE, as it
always was; narrowing one makes the caller its owner, since a private resource
nobody owns would be visible to super admins only.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Literal

from app.infrastructure.auth.authorization import get_current_subject

Kind = Literal["agent", "datasource", "script", "workflow"]

_LABEL: dict[str, str] = {
    "agent": "agent",
    "datasource": "data source",
    "script": "script",
    "workflow": "workflow",
}


# ─── Errors ───────────────────────────────────────────────────────────────────
# Mapped onto HTTP responses by an exception handler in app.api.app; the
# management tools let them surface as tool errors with the same message.

class VisibilityError(Exception):
    status_code = 400

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ResourceNotFound(VisibilityError):
    status_code = 404


class NotOwner(VisibilityError):
    status_code = 403


class InvalidReference(VisibilityError):
    status_code = 422


class StillReferenced(VisibilityError):
    status_code = 409


class IdTaken(VisibilityError):
    status_code = 409


# ─── Predicates ───────────────────────────────────────────────────────────────

def _field(resource: Any, name: str) -> Any:
    # Registry listings are dicts; everything else is a model.
    if isinstance(resource, dict):
        return resource.get(name)
    return getattr(resource, name, None)


def _visibility(resource: Any) -> str:
    return _field(resource, "visibility") or "public"


def _owner(resource: Any) -> str | None:
    return _field(resource, "owner_id")


def _team(resource: Any) -> str | None:
    return _field(resource, "team_id") if _visibility(resource) == "team" else None


_RANK = {"private": 0, "team": 1, "public": 2}


def _rank(visibility: str) -> int:
    return _RANK.get(visibility, 2)


def _in_team(subject: Any, resource: Any) -> bool:
    team = _team(resource)
    return team is not None and subject.team_id == team


# ─── Grants ───────────────────────────────────────────────────────────────────

_ACCESS_RANK = {"read": 0, "read_write": 1}


def grants_of(resource: Any) -> list[Any]:
    return list(_field(resource, "grants") or [])


def _grant_field(grant: Any, name: str) -> Any:
    return grant.get(name) if isinstance(grant, dict) else getattr(grant, name, None)


def _grant(resource: Any, grantee_type: str, grantee_id: str | None) -> str | None:
    """The access *grantee* holds on *resource*, or None."""
    if not grantee_id:
        return None
    for grant in grants_of(resource):
        if (
            _grant_field(grant, "grantee_type") == grantee_type
            and _grant_field(grant, "grantee_id") == grantee_id
        ):
            return _grant_field(grant, "access") or "read"
    return None


def user_grant(resource: Any, user_id: str | None) -> str | None:
    return _grant(resource, "user", user_id)


def workflow_grant(resource: Any, workflow_id: str | None) -> str | None:
    return _grant(resource, "workflow", workflow_id)


def _best(*levels: str | None) -> str | None:
    present = [lvl for lvl in levels if lvl]
    return max(present, key=lambda lvl: _ACCESS_RANK.get(lvl, 0)) if present else None


def caller_grant(resource: Any) -> str | None:
    """The caller's own user grant on *resource* (None with nobody bound)."""
    subject = get_current_subject()
    return user_grant(resource, subject.id) if subject is not None else None


def grant_access(user: Any, user_kind: str, used: Any) -> str | None:
    """The access *user* (a workflow or agent) has to *used* through a grant.

    A workflow grant counts for that workflow only; a user grant for whatever
    is private to the grantee.
    """
    if not grants_of(used):
        return None
    via_workflow = workflow_grant(used, _field(user, "id")) if user_kind == "workflow" else None
    via_user = (
        user_grant(used, _owner(user)) if _visibility(user) == "private" else None
    )
    return _best(via_workflow, via_user)


def workflow_grant_access(source: Any, workflow: Any) -> str | None:
    """At run time: the grant *workflow* uses *source* through, or None when
    it needs none (its scope covers the source) or has none.

    *workflow* is the runner (it carries the definition's owner fields). The
    scope test is the save-time one minus the directory lookup a private
    workflow's owner's team would need; the caller's own team stands in for it.
    A source reached through scope alone is never restricted here.
    """
    if workflow is None or not grants_of(source):
        return None
    vis = _visibility(source)
    if vis == "public":
        return None
    if _owner(source) is not None and _owner(source) == _owner(workflow):
        return None
    if vis == "team":
        if _team(workflow) == _team(source):
            return None
        subject = get_current_subject()
        if (
            _visibility(workflow) == "private"
            and subject is not None
            and subject.id == _owner(workflow)
            and subject.team_id == _team(source)
        ):
            return None
    return grant_access(workflow, "workflow", source)


def operation_denied(source: Any, operation: str, access: str | None) -> str | None:
    """Why *access* does not cover *operation* of *source*, or None."""
    if access != "read":
        return None
    op = source.get_operation(operation) if hasattr(source, "get_operation") else None
    if op is None:
        return None
    from app.infrastructure.datasources.destructive import operation_writes
    if not operation_writes(op, source):
        return None
    return (
        f"Operation '{operation}' of data source '{getattr(source, 'id', '')}' writes, "
        f"and only read access to it was granted"
    )


def can_see(resource: Any) -> bool:
    subject = get_current_subject()
    if subject is None:
        return True
    if _visibility(resource) == "public" or subject.is_super_admin:
        return True
    if subject.id is not None and _owner(resource) == subject.id:
        return True
    if subject.id is not None and user_grant(resource, subject.id):
        return True
    return _in_team(subject, resource)


def can_see_natively(resource: Any) -> bool:
    """Visible without counting the caller's own grant."""
    subject = get_current_subject()
    if subject is None or not subject.id or not user_grant(resource, subject.id):
        return can_see(resource)
    if _visibility(resource) == "public" or subject.is_super_admin:
        return True
    return _owner(resource) == subject.id or _in_team(subject, resource)


def can_manage(resource: Any) -> bool:
    """Delete it or change its visibility/team: owner or super admin.

    A legacy, ownerless public resource: anyone with WRITE, as before.
    """
    subject = get_current_subject()
    if subject is None or subject.is_super_admin:
        return True
    owner = _owner(resource)
    if owner is None:
        # An ownerless non-public resource is admin-only (it cannot normally
        # arise: narrowing a resource assigns an owner).
        return _visibility(resource) == "public"
    return subject.id is not None and owner == subject.id


def can_edit(
    resource: Any, team_editable: bool | None = None, grant_edits: bool | None = None,
) -> bool:
    """Change its content: whoever may manage it, plus — for a kind whose team
    members may edit (workflows) — the members of its team, plus — for a kind
    a grant can open to editing (agents, workflows) — a user holding a
    ``read_write`` grant."""
    if can_manage(resource):
        return True
    if team_editable is None:
        team_editable = bool(getattr(resource, "team_editable", False))
    if grant_edits is None:
        grant_edits = bool(getattr(resource, "grant_edits", False))
    subject = get_current_subject()
    if subject is None:
        return False
    if team_editable and _in_team(subject, resource):
        return True
    return grant_edits and user_grant(resource, subject.id) == "read_write"


def can_change_visibility(resource: Any) -> bool:
    """Change who can see it (visibility / team): whoever may manage it, plus
    an admin of the team it is shared with. That team admin may not delete it
    or change its content beyond what can_edit allows — only who sees it."""
    if can_manage(resource):
        return True
    subject = get_current_subject()
    return (
        subject is not None
        and subject.team_role == "admin"
        and _in_team(subject, resource)
    )


# Fields a visibility-only change may differ in (owner fields are restamped
# from the stored resource, timestamps by the backend, the rest is computed).
_SCOPE_FIELDS = frozenset(
    {"visibility", "team_id", "owner_id", "owner_name", "grants", "created_at", "updated_at"}
)


def _only_scope_changed(existing: Any, resource: Any) -> bool:
    try:
        exclude = set(_SCOPE_FIELDS) | set(getattr(type(existing), "model_computed_fields", {}) or {})
        return existing.model_dump(exclude=exclude) == resource.model_dump(exclude=exclude)
    except AttributeError:  # not a model: nothing to compare, so not "only scope"
        return False


def default_visibility() -> str:
    """New resources are private when someone in particular creates them."""
    subject = get_current_subject()
    return "private" if subject is not None and subject.id else "public"


# ─── References ───────────────────────────────────────────────────────────────

def references(kind: str, resource: Any) -> list[tuple[str, str]]:
    """Every ``(kind, id)`` the resource uses."""
    refs: list[tuple[str, str]] = []
    if kind == "workflow":
        for step in getattr(resource, "steps", None) or []:
            if not isinstance(step, dict):
                continue
            t = step.get("type")
            if t in ("langgraph-agent", "claude-agent") and step.get("agent_id"):
                refs.append(("agent", str(step["agent_id"])))
            elif t == "data_source" and step.get("source"):
                refs.append(("datasource", str(step["source"])))
            elif t == "python" and step.get("script_id"):
                refs.append(("script", str(step["script_id"])))
            elif t == "workflow" and step.get("workflow_id"):
                refs.append(("workflow", str(step["workflow_id"])))
            elif t == "combined_datasource":
                from app.infrastructure.orchestration.combined_datasource import iter_parts
                for _path, part in iter_parts(step.get("parts")):
                    if part.get("type") == "data_source" and part.get("source"):
                        refs.append(("datasource", str(part["source"])))
                    elif part.get("type") == "python" and part.get("script_id"):
                        refs.append(("script", str(part["script_id"])))
            elif t == "dynamic":
                for entry in step.get("agent_pool") or []:
                    if isinstance(entry, dict) and entry.get("agent_id"):
                        refs.append(("agent", str(entry["agent_id"])))
    elif kind == "agent":
        for addon in getattr(resource, "addons", None) or []:
            if getattr(addon, "type", None) == "datasource" and getattr(addon, "source_id", None):
                refs.append(("datasource", str(addon.source_id)))
    return refs


def datasource_operations(kind: str, resource: Any) -> dict[str, set[str]]:
    """Which operations of each data source the resource calls."""
    ops: dict[str, set[str]] = {}

    def add(source: Any, operation: Any) -> None:
        if source and operation:
            ops.setdefault(str(source), set()).add(str(operation))

    if kind == "workflow":
        for step in getattr(resource, "steps", None) or []:
            if not isinstance(step, dict):
                continue
            t = step.get("type")
            if t == "data_source":
                add(step.get("source"), step.get("operation"))
            elif t == "combined_datasource":
                from app.infrastructure.orchestration.combined_datasource import iter_parts
                for _path, part in iter_parts(step.get("parts")):
                    if part.get("type") == "data_source":
                        add(part.get("source"), part.get("operation"))
    elif kind == "agent":
        for addon in getattr(resource, "addons", None) or []:
            if getattr(addon, "type", None) == "datasource":
                for operation in getattr(addon, "allowed_operations", None) or []:
                    add(getattr(addon, "source_id", None), operation)
    return ops


class ReferenceGuard:
    """Checks references across the four (unscoped) backends.

    Holds the *raw* backends: it has to see a private resource to say "that
    one is private", and to find referrers the caller cannot see.
    """

    def __init__(
        self,
        resolve: Callable[[str], Any],
        directory: Callable[[], Any] | None = None,
    ) -> None:
        # kind -> raw backend (or None), looked up per call so a backend
        # swapped in after construction is still the one consulted.
        self._resolve = resolve
        self._directory = directory or (lambda: None)

    async def _owner_team(self, resource: Any) -> str | None:
        from app.application.teams import team_of
        return await team_of(self._directory(), _owner(resource))

    async def _violation(self, user: Any, user_kind: str, used: Any, used_kind: str) -> str | None:
        """Why *user* may not use *used*, or None. Scope rules, then grants."""
        problem = await self._scope_violation(user, user_kind, used, used_kind)
        if problem and grant_access(user, user_kind, used):
            return None
        return problem

    async def access_through_grant(
        self, user: Any, user_kind: str, used: Any, used_kind: str = "datasource",
    ) -> str | None:
        """The grant *user* needs to use *used*: None when its scope suffices."""
        if await self._scope_violation(user, user_kind, used, used_kind) is None:
            return None
        return grant_access(user, user_kind, used)

    async def _scope_violation(self, user: Any, user_kind: str, used: Any, used_kind: str) -> str | None:
        """Why *user* may not use *used*, or None. Scope rules only."""
        own, label = _LABEL[user_kind], _LABEL[used_kind]
        uid = getattr(used, "id", None)
        user_vis, used_vis = _visibility(user), _visibility(used)
        if _rank(used_vis) < _rank(user_vis):
            return (
                f"A {user_vis} {own} cannot use {used_vis} {label} '{uid}'. "
                f"Widen the {label}, or narrow this {own}."
            )
        if used_vis == "team":
            if user_vis == "team":
                if _team(user) != _team(used):
                    return f"This {own} cannot use {label} '{uid}': it belongs to another team."
            elif user_vis == "private":
                if await self._owner_team(user) != _team(used):
                    return (
                        f"This {own} cannot use {label} '{uid}': it belongs to a team "
                        f"its owner is not in."
                    )
        elif used_vis == "private":
            if _owner(used) != _owner(user):
                return f"This {own} cannot use {label} '{uid}': it is private to another user."
        return None

    async def check_references(self, kind: str, resource: Any, existing: Any = None) -> None:
        # What the stored version already used stays usable by whoever may
        # edit it — a team member, or a read_write grantee who cannot see
        # everything its owner can.
        kept = set(references(kind, existing)) if existing is not None else set()
        for ref_kind, ref_id in dict.fromkeys(references(kind, resource)):
            if ref_kind == kind and ref_id == getattr(resource, "id", None):
                continue
            backend = self._resolve(ref_kind)
            if backend is None:
                continue
            target = await backend.get(ref_id)
            if target is None:
                # Dangling references are not this module's concern: they fail
                # at run time exactly as they did before.
                continue
            if (
                not can_see(target)
                and not grant_access(resource, kind, target)
                and (ref_kind, ref_id) not in kept
            ):
                raise InvalidReference(f"{_LABEL[ref_kind].capitalize()} '{ref_id}' not found")
            problem = await self._violation(resource, kind, target, ref_kind)
            if problem:
                raise InvalidReference(problem)
            if ref_kind == "datasource":
                access = await self.access_through_grant(resource, kind, target)
                for operation in sorted(datasource_operations(kind, resource).get(ref_id, ())):
                    denied = operation_denied(target, operation, access)
                    if denied:
                        raise InvalidReference(denied)

    async def drop_workflow_grants(self, workflow_id: str) -> None:
        """Forget grants to a deleted workflow, so a new one reusing its id
        does not inherit them."""
        for kind in ("datasource", "agent", "workflow"):
            await self._drop_workflow_grants(kind, workflow_id)

    async def _drop_workflow_grants(self, kind: str, workflow_id: str) -> None:
        backend = self._resolve(kind)
        if backend is None:
            return
        for source in await backend.list():
            kept = [
                g for g in grants_of(source)
                if not (
                    _grant_field(g, "grantee_type") == "workflow"
                    and _grant_field(g, "grantee_id") == workflow_id
                )
            ]
            if len(kept) != len(grants_of(source)):
                source.grants = kept
                await backend.update(source.id, source)

    async def check_narrowing(self, kind: str, resource: Any) -> None:
        """Refuse to narrow a resource that something wider still uses."""
        rid = getattr(resource, "id", None)
        referrer_kinds = ["workflow"] + (["agent"] if kind == "datasource" else [])
        for referrer_kind in referrer_kinds:
            backend = self._resolve(referrer_kind)
            if backend is None:
                continue
            for referrer in await backend.list():
                if referrer_kind == kind and referrer.id == rid:
                    continue
                if (kind, rid) not in references(referrer_kind, referrer):
                    continue
                if await self._violation(referrer, referrer_kind, resource, kind) is None:
                    continue
                label = _LABEL[referrer_kind]
                who = (
                    f"{_visibility(referrer)} {label} '{referrer.id}'"
                    if can_see(referrer)
                    else f"a {label} you cannot see"
                )
                raise StillReferenced(
                    f"Cannot make {_LABEL[kind]} '{rid}' {_visibility(resource)}: {who} uses it."
                )


# ─── Scoped backend ───────────────────────────────────────────────────────────

class VisibilityScopedBackend:
    """Wraps one of the four definition backends with visibility and ownership.

    Reads hide what the caller cannot see; writes stamp the owner on create,
    preserve it on update (a PUT that rebuilds the model cannot change or drop
    it), refuse callers who are not the owner, and run the reference checks.
    Anything else is forwarded to the wrapped backend untouched.
    """

    def __init__(self, inner: Any, kind: Kind, guard: ReferenceGuard) -> None:
        self.inner = inner
        self.kind = kind
        self._guard = guard

    _OWN = frozenset({"inner", "kind", "_guard"})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Patching a method on the scoped backend (as tests do) patches the
        # backend it wraps, so the scope still applies on top of the patch.
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    @property
    def _label(self) -> str:
        return _LABEL[self.kind]

    # ── reads ──
    async def list(self) -> list[Any]:
        return [r for r in await self.inner.list() if can_see(r)]

    async def get(self, resource_id: str) -> Any:
        resource = await self.inner.get(resource_id)
        return resource if resource is not None and can_see(resource) else None

    async def get_by_name(self, name: str) -> Any:
        # Scripts only. Names are unique within what one caller can see, so a
        # private script of someone else neither blocks a name nor reveals it.
        resource = await self.inner.get_by_name(name)
        if resource is None or can_see(resource):
            return resource
        for candidate in await self.inner.list():
            if getattr(candidate, "name", None) == name and can_see(candidate):
                return candidate
        return None

    # ── writes ──
    async def _resolve_team(self, resource: Any, existing: Any | None) -> str | None:
        """The team a resource saved as "team" belongs to.

        A requested ``team_id`` equal to the stored one is no change. Otherwise
        a super admin may name any existing team (defaulting to their own, if
        they are in one); anyone else gets their own team, and naming another
        is refused.
        """
        from app.application.teams import TeamInvalid
        subject = get_current_subject()
        requested = resource.team_id if "team_id" in resource.model_fields_set else None
        stored = _team(existing) if existing is not None else None
        if requested and requested == stored:
            return stored
        if subject is None:
            return requested or stored
        if subject.is_super_admin:
            team = requested or stored or subject.team_id
            if not team:
                raise TeamInvalid('team_id required: say which team to share with')
            directory = self._guard._directory()
            if directory is not None and await directory.get_team(team) is None:
                raise TeamInvalid(f"Team '{team}' not found")
            return team
        if requested and requested != subject.team_id:
            raise NotOwner("You can only share with your own team")
        if stored and not requested:
            # Re-saving a team resource (e.g. a member editing a team workflow)
            # keeps its team; moving it is the owner's, below.
            return stored
        if not subject.team_id:
            raise TeamInvalid("You are not in a team, so you cannot share with one")
        return subject.team_id

    async def _prepare(self, resource: Any, existing: Any | None) -> None:
        subject = get_current_subject()
        if existing is None:
            if "visibility" not in resource.model_fields_set:
                resource.visibility = default_visibility()
            if subject is not None and subject.id:
                resource.owner_id = subject.id
                resource.owner_name = subject.name
            else:
                resource.owner_id = None
                resource.owner_name = None
            if "grants" in type(resource).model_fields:
                resource.grants = []
        else:
            if "grants" in type(resource).model_fields:
                # Only the grant routes change grants; they write past this
                # backend. Anything saved through it keeps the stored ones.
                resource.grants = list(grants_of(existing))
            resource.owner_id = getattr(existing, "owner_id", None)
            resource.owner_name = getattr(existing, "owner_name", None)
            if "visibility" not in resource.model_fields_set:
                resource.visibility = _visibility(existing)
            if (
                resource.visibility != "public"
                and resource.owner_id is None
                and subject is not None
                and subject.id
            ):
                resource.owner_id = subject.id
                resource.owner_name = subject.name
        resource.team_id = (
            await self._resolve_team(resource, existing) if resource.visibility == "team" else None
        )
        if existing is not None and (
            resource.visibility != _visibility(existing) or resource.team_id != _team(existing)
        ):
            if not can_change_visibility(existing):
                raise NotOwner(
                    f"Only the owner of {self._label} '{resource.id}', an admin of its "
                    f"team, or a super admin may change who can see it"
                )
        if subject is None:
            return
        await self._guard.check_references(self.kind, resource, existing)
        narrowed = existing is None and resource.visibility != "public" or (
            existing is not None
            and (
                _rank(resource.visibility) < _rank(_visibility(existing))
                or (resource.visibility == "team" and resource.team_id != _team(existing))
            )
        )
        if narrowed:
            await self._guard.check_narrowing(self.kind, resource)

    def _check_editable(self, existing: Any, resource_id: str) -> None:
        if not can_see(existing):
            raise ResourceNotFound(f"{self._label.capitalize()} '{resource_id}' not found")
        if not can_edit(existing):
            raise NotOwner(
                f"Only the owner of {self._label} '{resource_id}' (or a super admin) may change it"
            )

    async def create(self, resource: Any) -> Any:
        existing = await self.inner.get(resource.id)
        if existing is not None:
            # create() upserts. Over a resource the caller cannot see this would
            # be a silent takeover, so it is refused without saying whose it is.
            if not can_see(existing):
                raise IdTaken(
                    f"The id '{resource.id}' is already taken; choose another"
                )
            self._check_editable(existing, resource.id)
        await self._prepare(resource, existing)
        return await self.inner.create(resource)

    async def update(self, resource_id: str, resource: Any) -> Any:
        existing = await self.inner.get(resource_id)
        if existing is not None:
            if not (
                can_see(existing)
                and not can_edit(existing)
                and can_change_visibility(existing)
                and _only_scope_changed(existing, resource)
            ):
                # Anything but a team admin re-scoping their team's resource.
                self._check_editable(existing, resource_id)
        await self._prepare(resource, existing)
        return await self.inner.update(resource_id, resource)

    async def delete(self, resource_id: str) -> None:
        existing = await self.inner.get(resource_id)
        if existing is not None:
            if not can_see(existing):
                raise ResourceNotFound(f"{self._label.capitalize()} '{resource_id}' not found")
            if not can_manage(existing):
                raise NotOwner(
                    f"Only the owner of {self._label} '{resource_id}' (or a super admin) "
                    f"may delete it"
                )
        await self.inner.delete(resource_id)
        if self.kind == "workflow" and existing is not None:
            await self._guard.drop_workflow_grants(resource_id)


class WorkflowSources:
    """A definition backend (data sources, agents, workflows) as one workflow
    sees it: what the caller can see, plus what is granted to that workflow
    (which the caller may not see anywhere else). Anything but ``get`` and
    ``granted`` goes to the wrapped backend."""

    def __init__(self, backend: Any, workflow_id: str | None) -> None:
        self._backend = backend
        self._workflow_id = workflow_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    async def get(self, source_id: str) -> Any:
        source = await self._backend.get(source_id)
        if source is not None or not self._workflow_id:
            return source
        raw = await unwrap(self._backend).get(source_id)
        if raw is not None and workflow_grant(raw, self._workflow_id):
            return raw
        return None

    async def granted(self) -> list[Any]:
        """Sources granted to the workflow that the caller cannot otherwise see."""
        if not self._workflow_id:
            return []
        return [
            s for s in await unwrap(self._backend).list()
            if workflow_grant(s, self._workflow_id) and not can_see(s)
        ]


def sources_for_workflow(backend: Any, workflow_id: str | None) -> Any:
    return None if backend is None else WorkflowSources(backend, workflow_id)


# Same view, for agents and workflows.
granted_for_workflow = sources_for_workflow


def unwrap(backend: Any) -> Any:
    """The raw backend behind a scoped one (identity for anything else)."""
    return getattr(backend, "inner", backend) if isinstance(backend, VisibilityScopedBackend) else backend


# ─── Runs ─────────────────────────────────────────────────────────────────────
# A run belongs to its workflow: a private workflow's runs, traces, data and
# approvals are its owner's (and admins') only.

async def hidden_ids(backend: Any) -> list[str]:
    """Ids the caller may not see in *backend*; empty when nothing is bound."""
    if backend is None or get_current_subject() is None:
        return []
    return [r.id for r in await unwrap(backend).list() if not can_see(r)]


async def hidden_workflow_ids(workflow_backend: Any) -> list[str]:
    return await hidden_ids(workflow_backend)


async def approval_exclusions(container: Any) -> dict[str, list[str]]:
    """Approval-list filters hiding cases of invisible workflows / data sources.

    Empty (no kwargs at all) when nothing is hidden, so backends that predate
    the filters keep working unchanged.
    """
    filters: dict[str, list[str]] = {}
    workflows = await hidden_ids(getattr(container, "workflow_backend", None))
    if workflows:
        filters["exclude_workflow_ids"] = workflows
    sources = await hidden_ids(getattr(container, "data_source_backend", None))
    if sources:
        filters["exclude_datasource_ids"] = sources
    return filters


async def approval_case_visible(container: Any, case: Any) -> bool:
    """A case is as visible as the workflow and the data source it concerns."""
    if get_current_subject() is None or case is None:
        return True
    if not await workflow_visible(getattr(container, "workflow_backend", None), getattr(case, "workflow_id", None)):
        return False
    backend = getattr(container, "data_source_backend", None)
    source_id = getattr(case, "datasource_id", None)
    if backend is not None and source_id:
        source = await unwrap(backend).get(source_id)
        if source is not None and not can_see(source):
            return False
    return True


async def workflow_visible(workflow_backend: Any, workflow_id: str | None) -> bool:
    """Whether runs of *workflow_id* are visible. A deleted workflow's are."""
    if workflow_backend is None or not workflow_id or get_current_subject() is None:
        return True
    workflow = await unwrap(workflow_backend).get(workflow_id)
    return workflow is None or can_see(workflow)


async def ensure_run_visible(workflow_backend: Any, run: Any) -> None:
    if run is not None and not await workflow_visible(workflow_backend, getattr(run, "graph_id", None)):
        raise ResourceNotFound("Run not found")


def ownership_fields(
    resource: Any, team_editable: bool | None = None, grant_edits: bool | None = None,
) -> dict[str, Any]:
    """The fields every list/summary entry carries."""
    from app.application.teams import team_name
    team_id = _team(resource)
    fields = {
        "visibility": _visibility(resource),
        "owner_id": _owner(resource),
        "owner_name": _field(resource, "owner_name"),
        "team_id": team_id,
        "team_name": team_name(team_id),
        "can_edit": can_edit(resource, team_editable, grant_edits),
    }
    if _field(resource, "grants") is not None:
        fields["granted_access"] = caller_grant(resource)
        fields["can_share"] = can_change_visibility(resource)
    return fields


def visibility_kwargs(visibility: str | None, team_id: str | None = None) -> dict[str, Any]:
    """Model kwargs for an optional requested visibility (and team).

    Omitted (None) must stay *unset* on the model rather than become a value:
    the scoped backend tells "keep the stored one" / "use the default" apart
    from an explicit choice by whether the field was set.
    """
    kwargs: dict[str, Any] = {}
    if visibility is not None:
        kwargs["visibility"] = visibility
    if team_id is not None:
        kwargs["team_id"] = team_id
    return kwargs


def filter_visible(resources: Iterable[Any]) -> list[Any]:
    return [r for r in resources if can_see(r)]
