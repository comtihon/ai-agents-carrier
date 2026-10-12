"""Who may see and change an agent, data source, script or workflow.

Every one of the four is ``public`` (visible to and usable by every user) or
``private`` (visible to its owner and to admins only — to anyone else it does
not exist: absent from lists, 404 by id, never offered in a picker or to the
chat agent). Only the owner or an admin may change or delete one, public or not.

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

References follow two rules, checked on save:

* you can only reference what you can see;
* a public resource may only use public ones, and a private one only public
  ones or its own owner's private ones — otherwise a user who can see (and
  run) the referrer would run something they cannot see.

The converse is checked when a resource turns private: refused while a public
resource, or another user's resource, still uses it.

Legacy documents, stored before these fields existed, load as public with no
owner. An ownerless resource stays editable by any caller with WRITE, as it
always was; turning one private makes the caller its owner, since a private
resource nobody owns would be visible to admins only.
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


def can_see(resource: Any) -> bool:
    subject = get_current_subject()
    if subject is None:
        return True
    if _visibility(resource) == "public" or subject.is_admin:
        return True
    return subject.id is not None and _owner(resource) == subject.id


def can_edit(resource: Any) -> bool:
    """Owner or admin; anyone with WRITE for a legacy, ownerless resource."""
    subject = get_current_subject()
    if subject is None or subject.is_admin:
        return True
    owner = _owner(resource)
    if owner is None:
        # An ownerless private resource is admin-only (it cannot normally
        # arise: turning a resource private assigns an owner).
        return _visibility(resource) == "public"
    return subject.id is not None and owner == subject.id


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
            elif t == "dynamic":
                for entry in step.get("agent_pool") or []:
                    if isinstance(entry, dict) and entry.get("agent_id"):
                        refs.append(("agent", str(entry["agent_id"])))
    elif kind == "agent":
        for addon in getattr(resource, "addons", None) or []:
            if getattr(addon, "type", None) == "datasource" and getattr(addon, "source_id", None):
                refs.append(("datasource", str(addon.source_id)))
    return refs


class ReferenceGuard:
    """Checks references across the four (unscoped) backends.

    Holds the *raw* backends: it has to see a private resource to say "that
    one is private", and to find referrers the caller cannot see.
    """

    def __init__(self, resolve: Callable[[str], Any]) -> None:
        # kind -> raw backend (or None), looked up per call so a backend
        # swapped in after construction is still the one consulted.
        self._resolve = resolve

    async def check_references(self, kind: str, resource: Any) -> None:
        own_label = _LABEL[kind]
        is_public = _visibility(resource) == "public"
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
            label = _LABEL[ref_kind]
            if not can_see(target):
                raise InvalidReference(f"{label.capitalize()} '{ref_id}' not found")
            if _visibility(target) != "private":
                continue
            if is_public:
                raise InvalidReference(
                    f"A public {own_label} cannot use private {label} '{ref_id}'. "
                    f"Make the {label} public, or keep this {own_label} private."
                )
            if _owner(target) != _owner(resource):
                raise InvalidReference(
                    f"This {own_label} cannot use {label} '{ref_id}': it is private "
                    f"to another user."
                )

    async def check_turning_private(self, kind: str, resource: Any) -> None:
        """Refuse to hide a resource that public or foreign resources use."""
        rid = getattr(resource, "id", None)
        owner = _owner(resource)
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
                if _visibility(referrer) == "private" and _owner(referrer) == owner:
                    continue
                label = _LABEL[referrer_kind]
                who = (
                    f"public {label} '{referrer.id}'"
                    if _visibility(referrer) == "public"
                    else f"a {label} of another user"
                )
                raise StillReferenced(
                    f"Cannot make {_LABEL[kind]} '{rid}' private: {who} uses it."
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
        else:
            resource.owner_id = getattr(existing, "owner_id", None)
            resource.owner_name = getattr(existing, "owner_name", None)
            if "visibility" not in resource.model_fields_set:
                resource.visibility = _visibility(existing)
            if (
                resource.visibility == "private"
                and resource.owner_id is None
                and subject is not None
                and subject.id
            ):
                resource.owner_id = subject.id
                resource.owner_name = subject.name
        if subject is None:
            return
        await self._guard.check_references(self.kind, resource)
        if resource.visibility == "private" and (
            existing is None or _visibility(existing) != "private"
        ):
            await self._guard.check_turning_private(self.kind, resource)

    def _check_editable(self, existing: Any, resource_id: str) -> None:
        if not can_see(existing):
            raise ResourceNotFound(f"{self._label.capitalize()} '{resource_id}' not found")
        if not can_edit(existing):
            raise NotOwner(
                f"Only the owner of {self._label} '{resource_id}' (or an admin) may change it"
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
            self._check_editable(existing, resource_id)
        await self._prepare(resource, existing)
        return await self.inner.update(resource_id, resource)

    async def delete(self, resource_id: str) -> None:
        existing = await self.inner.get(resource_id)
        if existing is not None:
            self._check_editable(existing, resource_id)
        await self.inner.delete(resource_id)


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


def ownership_fields(resource: Any) -> dict[str, Any]:
    """The four fields every list/summary entry carries."""
    return {
        "visibility": _visibility(resource),
        "owner_id": _owner(resource),
        "owner_name": _field(resource, "owner_name"),
        "can_edit": can_edit(resource),
    }


def visibility_kwargs(visibility: str | None) -> dict[str, Any]:
    """Model kwargs for an optional requested visibility.

    Omitted (None) must stay *unset* on the model rather than become a value:
    the scoped backend tells "keep the stored one" / "use the default" apart
    from an explicit choice by whether the field was set.
    """
    return {"visibility": visibility} if visibility is not None else {}


def filter_visible(resources: Iterable[Any]) -> list[Any]:
    return [r for r in resources if can_see(r)]
