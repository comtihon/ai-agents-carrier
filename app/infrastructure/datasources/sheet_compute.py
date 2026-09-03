"""Tier 2 of a sheet binding: a generated pure transform, and its gates.

A tier-1 :class:`~app.domain.models.sheet_binding.SheetBinding` is a form —
every column picked from a dropdown, every value tagged as a literal or a state
path, nothing to ``eval``.  That covers "read the open rows", "set status on the
row whose id matches".  It does not cover "sum hours by owner for open rows this
quarter", because grouping and aggregation across rows is computation and a form
is not a programming language.

Tier 2 is the smallest possible escape hatch for exactly that: **a model writes
the computation, and nothing else.**

    [ds]      get_values                        fixed, holds the credential
    [sandbox] sheets_rows_to_records            tier-1, hand-written
    [sandbox] transform(records, params)        THE ONLY GENERATED PART
    [sandbox] sheets_build_write                tier-1, hand-written, computes A1
    [ds]      batch_update_values               fixed, approval gate

Why that sandwich is the whole security story
---------------------------------------------
Generated code produces **values, never addresses.**  It replaces the
computation half of a binding and inherits the addressing half unchanged: the
binding still declares ``document``, ``sheet``, ``key_column`` and the
``columns`` whitelist, and tier-1's ``sheets_build_write`` is still the only
thing that turns a column name into an A1 range.

So a generated transform *cannot*, by construction rather than by good
behaviour:

* write to a column the binding does not name — the whitelist is re-checked
  here as a hard runtime error (:func:`check_write_output`), not merely asked
  for in the prompt, and ``sheets_build_write`` would ignore it regardless;
* choose a different document or tab — it never sees a file id and has no way
  to make a call;
* compute a range — it returns ``{column: value}``, and A1 arithmetic lives in
  tier-1 code no model wrote;
* see a credential — it runs in the seccomp sandbox, where the kernel refuses
  ``openat`` and ``socket``, so neither the pod's service-account token nor the
  network is reachable even by a script that reflects its way around the
  interpreter's own deny-list.

The gates, and what each one is actually for
--------------------------------------------
``validate_transform_source``  (static)
    An AST allow-list, in the spirit of the route-condition check in
    ``yaml_graph`` and the python-step gate in ``auth.sandbox_guard``.  This is
    not the security boundary — the sandbox is.  It is a *fast, legible*
    boundary: it fails at compile time with a message that says which line was
    the problem, so a model's third attempt is informed rather than random, and
    a reviewer reading the stored code can see it contains no imports and no
    reflection without having to trust the sandbox to have caught it.

``run_transform``  (dynamic)
    The same seccomp sandbox and the same ``run_script`` path as a user-authored
    python step.  There is deliberately **no** separate "our own codegen is
    trusted" route and no bare ``exec`` anywhere in this module: code a model
    wrote from an instruction a user typed is the least trusted code in the
    system, not the most.

``assert_deterministic``  (double-run)
    Two runs, compared.  Catches ``random``, a wall-clock read, and iteration
    over a ``set`` — none of which are malicious, all of which turn a binding
    into something that silently writes a different answer every night.  A
    non-deterministic transform also makes the golden fixture meaningless, so
    this gate has to pass before a fixture is worth freezing.

``golden fixture``  (regression)
    ``sample_rows -> output``, frozen on the binding and re-run on every
    recompile and on any schema change.  The synthetic rows
    (:func:`adversarial_rows`) matter more than the real ones: a fully empty
    row, a row with missing trailing cells, a number stored as text and a
    duplicate key are the four shapes real spreadsheets produce constantly and
    sample data almost never contains.

None of this makes generated code as safe as a form.  It makes it *reviewable,
reproducible and bounded*, which is the most that can honestly be claimed — and
why tier 2 is an outcome the backend escalates to, never a mode a user picks.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The signature
# ---------------------------------------------------------------------------
# Bumped when the entry point's shape changes.  Pinned into every binding
# (``compute.signature_version``) so a stored script from an older shape is
# refused rather than called with arguments it never expected -- a silent
# mis-call is far worse than a loud "this needs recompiling".
SIGNATURE_VERSION = 1

TRANSFORM_NAME = "transform"
TRANSFORM_PARAMS = ("records", "params")

# Handed to the model verbatim, and quoted back in every rejection so the
# feedback loop and the prompt cannot drift apart.
TRANSFORM_SIGNATURE = (
    "def transform(records: list[dict], params: dict) -> Any"
)

# Appended to the generated source to make it a runnable sandbox program.  This
# is the only call site of the generated function, and it is fixed text: the
# model never writes the call, only the callee.
_EPILOGUE = f'output = {TRANSFORM_NAME}(state["records"], state["params"])'


# ---------------------------------------------------------------------------
# Static gate: AST allow-list
# ---------------------------------------------------------------------------

# A pure transform over rows of a spreadsheet needs dates, arithmetic and
# pattern matching.  It does not need anything else, and every module outside
# this set is either useless here or a way out.
ALLOWED_IMPORTS = frozenset({"datetime", "math", "re"})

# Names that are reflection or code execution.  ``getattr`` is on the list with
# the rest because ``getattr(x, "__cl" + "ass__")`` defeats any check on
# attribute *syntax*, which is what the dunder rule below inspects.
DENIED_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "open", "input", "breakpoint",
    "memoryview", "exit", "quit", "help", "dir", "super", "type",
})

# Non-dunder attributes that reach a frame, and through it the enclosing
# module's globals.  The dunder rule below does not cover these because they
# are not spelled with underscores.
DENIED_ATTRIBUTES = frozenset({
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code",
    "gi_frame", "cr_frame", "ag_frame", "tb_frame", "func_globals",
})

# Node types with no place in a pure transform.  ``Global``/``Nonlocal`` would
# let a helper mutate module state between the two determinism runs; the async
# and generator nodes would let the function return before it has computed,
# which makes the output-shape check inspect a generator object instead of a
# result.
_DENIED_NODES: tuple[type[ast.AST], ...] = (
    ast.Global, ast.Nonlocal,
    ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await,
    ast.Yield, ast.YieldFrom,
)

# Caps.  A transform that needs more than this is not a transform, and both
# limits also bound how much work the AST walk and the sandbox parse do on
# input a model produced.
MAX_SOURCE_CHARS = 20_000
MAX_AST_NODES = 2_500


class ComputeValidationError(ValueError):
    """Generated (or hand-authored) transform source that must not be stored.

    The message is written to be read twice: once by the person reviewing an
    escalation in the editor, and once by the model on its next attempt.
    """


def validate_transform_source(code: str) -> ast.Module:
    """Raise :class:`ComputeValidationError` unless *code* is a safe transform.

    Returns the parsed tree, so a caller that wants to inspect it further does
    not parse twice.

    This runs *before* the sandbox, on every store and every load — a script
    already in the database is re-validated when it is used, so tightening this
    function retroactively rejects code that an earlier, looser version let
    through.  A gate that only ran at authoring time would leave whatever it
    once accepted running forever.
    """
    if not code or not code.strip():
        raise ComputeValidationError("the transform is empty")
    if len(code) > MAX_SOURCE_CHARS:
        raise ComputeValidationError(
            f"the transform is {len(code)} characters, over the "
            f"{MAX_SOURCE_CHARS} limit — a binding's computation should be "
            "small enough to read in one screen"
        )

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ComputeValidationError(
            f"the transform does not parse: {exc.msg} (line {exc.lineno})"
        ) from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ComputeValidationError(
            f"the transform has {len(nodes)} AST nodes, over the "
            f"{MAX_AST_NODES} limit"
        )

    _check_entry_point(tree)
    for node in nodes:
        _check_node(node)
    return tree


def _where(node: ast.AST) -> str:
    line = getattr(node, "lineno", None)
    return f" (line {line})" if line else ""


def _check_entry_point(tree: ast.Module) -> None:
    """Exactly one top-level ``def transform(records, params)``, and no work beside it.

    Top-level statements other than the entry point and its helpers are refused
    because they run at import time, before any gate has seen a *value*: the
    determinism and shape checks inspect what ``transform`` returns, and code
    that did its damage while the module was still being exec'd never passes
    through them.
    """
    entry: ast.FunctionDef | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == TRANSFORM_NAME:
            if entry is not None:
                raise ComputeValidationError(
                    f"'{TRANSFORM_NAME}' is defined more than once"
                )
            entry = stmt
        elif isinstance(stmt, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            # Helper defs, helper classes and (allow-listed) imports are fine.
            continue
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            # A module-level constant table is a normal way to write a
            # transform; it is data, and it is covered by the node checks.
            continue
        else:
            raise ComputeValidationError(
                f"top-level {type(stmt).__name__} is not allowed{_where(stmt)}: "
                f"put the work inside '{TRANSFORM_NAME}', which is the only "
                "thing that is called"
            )

    if entry is None:
        raise ComputeValidationError(
            f"the transform must define an entry point with the exact "
            f"signature '{TRANSFORM_SIGNATURE}'; no '{TRANSFORM_NAME}' "
            "function was found"
        )

    args = entry.args
    names = tuple(a.arg for a in args.posonlyargs + args.args)
    if names != TRANSFORM_PARAMS:
        raise ComputeValidationError(
            f"'{TRANSFORM_NAME}' must take exactly {list(TRANSFORM_PARAMS)}, "
            f"got {list(names)} — the signature is fixed: "
            f"'{TRANSFORM_SIGNATURE}'"
        )
    if args.vararg or args.kwarg or args.kwonlyargs:
        raise ComputeValidationError(
            f"'{TRANSFORM_NAME}' must not take *args or **kwargs — the "
            f"signature is fixed: '{TRANSFORM_SIGNATURE}'"
        )


def _check_node(node: ast.AST) -> None:
    if isinstance(node, _DENIED_NODES):
        raise ComputeValidationError(
            f"{type(node).__name__} is not allowed in a transform{_where(node)}"
        )

    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                raise ComputeValidationError(
                    f"import of '{alias.name}' is not allowed{_where(node)}: a "
                    f"transform may import only {', '.join(sorted(ALLOWED_IMPORTS))}"
                )
        return

    if isinstance(node, ast.ImportFrom):
        root = (node.module or "").split(".")[0]
        if node.level or root not in ALLOWED_IMPORTS:
            raise ComputeValidationError(
                f"import from '{node.module or '.'}' is not allowed"
                f"{_where(node)}: a transform may import only "
                f"{', '.join(sorted(ALLOWED_IMPORTS))}"
            )
        return

    if isinstance(node, ast.Attribute):
        name = node.attr
        if name.startswith("__") and name.endswith("__"):
            raise ComputeValidationError(
                f"attribute '{name}' is not allowed{_where(node)}: dunder "
                "access is how sandboxed code reaches the interpreter"
            )
        if name in DENIED_ATTRIBUTES:
            raise ComputeValidationError(
                f"attribute '{name}' is not allowed{_where(node)}: it reaches "
                "a stack frame, and through it the enclosing module"
            )
        return

    if isinstance(node, ast.Name):
        if node.id in DENIED_NAMES:
            raise ComputeValidationError(
                f"'{node.id}' is not allowed in a transform{_where(node)}"
            )
        if node.id.startswith("__") and node.id.endswith("__"):
            raise ComputeValidationError(
                f"'{node.id}' is not allowed in a transform{_where(node)}"
            )
        return

    if isinstance(node, ast.While):
        # An unbounded loop is refused here rather than left to the sandbox
        # timeout: the timeout is a backstop that costs a wall-clock stall on
        # every attempt, and `while True` in a pure transform over a finite
        # list of records is a mistake in every case, never a style.
        test = node.test
        if isinstance(test, ast.Constant) and bool(test.value):
            raise ComputeValidationError(
                f"'while {test.value!r}' never terminates on its own"
                f"{_where(node)}: iterate over 'records' with a for loop"
            )
        return


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def content_hash(code: str) -> str:
    """``sha256:…`` over the exact stored source.

    The hash is what makes a run auditable: it is recorded per run, so a
    six-month-old binding's log line identifies the code that produced the
    values, even if the binding has been recompiled since.
    """
    return "sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest()


def script_id_for(code: str) -> str:
    """A stable, readable id for a transform: ``sheets_tx_<8 hex>``.

    Derived from the content rather than random, so recompiling to byte-identical
    code keeps the id and the binding shows no spurious change.
    """
    return "sheets_tx_" + hashlib.sha256(code.encode("utf-8")).hexdigest()[:8]


def cache_key(
    *,
    instruction: str,
    answers: dict[str, Any] | None,
    schema_fingerprint: str,
    model_id: str,
    signature_version: int = SIGNATURE_VERSION,
) -> str:
    """Identity of a compile request: change any input, get a different key.

    The four inputs are exactly the things that make previously generated code
    wrong rather than merely old — a reworded instruction, a new answer to an
    ambiguity question, a reordered header row, a different model, a changed
    signature.  ``answers`` is folded in sorted so a dict that arrived in a
    different order is the same key.
    """
    material = json.dumps(
        {
            "instruction": (instruction or "").strip(),
            "answers": answers or {},
            "schema_fingerprint": schema_fingerprint or "",
            "model_id": model_id or "",
            "signature_version": signature_version,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Dynamic gate: the sandbox
# ---------------------------------------------------------------------------

class ComputeRuntimeError(RuntimeError):
    """A transform that failed to run, timed out, or returned the wrong shape."""


def build_sandbox_program(code: str) -> str:
    """The generated source plus the fixed call that runs it.

    ``run_script`` execs its payload with ``state`` bound and reads back
    whatever the program assigned to ``output`` (see
    ``orchestration.script_sandbox``), so the epilogue is the whole adapter.
    It is fixed text appended after validation: the model supplies the callee,
    never the call.
    """
    return f"{code.rstrip()}\n\n{_EPILOGUE}\n"


async def run_transform(
    code: str,
    records: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    *,
    runtime: str = "local",
    timeout: float = 10.0,
    memory_mb: int = 256,
    validate: bool = True,
) -> Any:
    """Run *code*'s ``transform`` over *records* in the sandbox.

    ``validate`` re-runs the static gate first and defaults to true on purpose:
    every path that executes a transform goes through the AST allow-list on the
    way, including a script loaded from storage that was validated when it was
    written.  Passing false is for a caller that has just validated the same
    source itself.

    The timeout is much shorter than a workflow python step's (10s vs 60s): a
    pure transform over a probe's worth of rows either finishes immediately or
    is looping.
    """
    if validate:
        validate_transform_source(code)

    from app.infrastructure.orchestration.script_sandbox import (
        ScriptSandboxError,
        run_script,
    )

    program = build_sandbox_program(code)
    # Only rows and the caller's params cross into the sandbox. No settings, no
    # credentials, no binding -- in particular no file id, which is what keeps
    # "produces values, never addresses" true at the transport level and not
    # only by convention.
    state = {"records": list(records or []), "params": dict(params or {})}
    try:
        return await run_script(
            program, state, runtime=runtime, timeout=timeout, memory_mb=memory_mb,
        )
    except ScriptSandboxError as exc:
        raise ComputeRuntimeError(
            f"the transform did not produce a result: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the compile loop verbatim
        raise ComputeRuntimeError(f"the transform failed to run: {exc}") from exc


# ---------------------------------------------------------------------------
# Determinism gate
# ---------------------------------------------------------------------------

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))


async def assert_deterministic(
    code: str,
    records: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    **run_kwargs: Any,
) -> Any:
    """Run *code* twice and return its output, or raise on a difference.

    Two runs is enough to catch what actually happens in practice — ``random``,
    ``datetime.now()`` reaching the output, iteration order of a ``set`` — and
    it is the only check that can catch them at all, since each is perfectly
    legal Python that the AST gate has no business refusing (a transform may
    legitimately *import* datetime; it may not put the clock in its answer).

    It is also a precondition of the golden fixture: freezing
    ``sample_rows -> output`` for a transform whose output moves would produce a
    fixture that fails on its first re-run.
    """
    first = await run_transform(code, records, params, **run_kwargs)
    # The second run re-validates for free but does not need to; the source has
    # not changed between the two.
    second = await run_transform(code, records, params, validate=False, **{
        k: v for k, v in run_kwargs.items() if k != "validate"
    })
    if _canonical(first) != _canonical(second):
        raise ComputeValidationError(
            "the transform is not deterministic: two runs over the same rows "
            "returned different results. A binding runs unattended, so its "
            "computation must not depend on the clock, on random, or on the "
            "iteration order of a set — sort before returning, and take any "
            "'today' it needs from params rather than reading the clock."
        )
    return first


# ---------------------------------------------------------------------------
# Output shape + the column whitelist
# ---------------------------------------------------------------------------

# What a cell can hold.  A dict or a list in a column value means the transform
# returned structure where a cell was wanted, which is a bug worth naming
# rather than str()-ing into the sheet.
_SCALARS = (str, int, float, bool, type(None))


def check_read_output(output: Any, shape: str = "records") -> Any:
    """Validate a read transform's output against its declared *shape*.

    ``records``
        A list of flat dicts — what a read binding normally publishes, and what
        a downstream step or an LLM prompt can consume.
    ``record``
        A single flat dict (a one-row answer, e.g. a set of totals).
    ``value``
        A bare scalar (one number or string).

    The shape is declared on the binding rather than inferred from the first
    run, so a transform that happens to return one row today does not quietly
    change the binding's contract when the sheet grows.
    """
    if shape == "value":
        if not isinstance(output, _SCALARS):
            raise ComputeValidationError(
                f"this binding declares output shape 'value', so the transform "
                f"must return a single number or string, not "
                f"{type(output).__name__}"
            )
        return output

    if shape == "record":
        if not isinstance(output, dict):
            raise ComputeValidationError(
                f"this binding declares output shape 'record', so the transform "
                f"must return one dict, not {type(output).__name__}"
            )
        _check_flat(output, "the returned record")
        return output

    if shape == "records":
        if not isinstance(output, list):
            raise ComputeValidationError(
                f"this binding declares output shape 'records', so the "
                f"transform must return a list of dicts, not "
                f"{type(output).__name__}"
            )
        for index, row in enumerate(output):
            if not isinstance(row, dict):
                raise ComputeValidationError(
                    f"the transform returned {type(row).__name__} at index "
                    f"{index}; output shape 'records' means a list of dicts"
                )
            _check_flat(row, f"the record at index {index}")
        return output

    raise ComputeValidationError(f"unknown output shape '{shape}'")


def _check_flat(row: dict[str, Any], where: str) -> None:
    for key, value in row.items():
        if not isinstance(key, str):
            raise ComputeValidationError(
                f"{where} has a non-string key {key!r}"
            )
        if not isinstance(value, _SCALARS):
            raise ComputeValidationError(
                f"{where} puts {type(value).__name__} in '{key}'; a cell holds "
                "a single value, so flatten it before returning"
            )


def check_write_output(
    output: Any,
    allowed_columns: Iterable[str],
) -> dict[str, Any]:
    """Validate a write transform's output and return it as ``{column: value}``.

    **This is the hard enforcement of the column whitelist**, and the reason it
    is a runtime check and not just a line in the prompt: a prompt instruction
    is a request, and the whole claim of tier 2 is that generated code *cannot*
    write outside the columns the binding declares.  A model that returns an
    extra key here is not silently trimmed either — trimming would make a
    mis-generated transform look like it worked while dropping half the
    author's intent.  It is rejected, loudly, with the allowed list quoted.

    ``list[dict]`` of length one is accepted and unwrapped, because a model
    told "return column→value maps" reasonably often returns a list of one.  A
    longer list is refused: tier-1's ``sheets_build_write`` composes a write to
    exactly one resolved row, so a multi-row answer has nowhere to go — the
    message points at the tier-1 read + separate step composition instead.
    """
    allowed = set(allowed_columns or ())

    if isinstance(output, list):
        if len(output) == 1 and isinstance(output[0], dict):
            output = output[0]
        elif not output:
            raise ComputeValidationError(
                "the transform returned an empty list; a write transform "
                "returns one {column: value} map"
            )
        else:
            raise ComputeValidationError(
                f"the transform returned {len(output)} rows. A write binding "
                "writes one resolved row, so a multi-row result cannot be "
                "addressed — read with a tier-1 binding and write the rows "
                "from a separate step instead."
            )

    if not isinstance(output, dict):
        raise ComputeValidationError(
            f"a write transform must return a {{column: value}} map, not "
            f"{type(output).__name__}"
        )

    unknown = sorted(k for k in output if k not in allowed)
    if unknown:
        raise ComputeValidationError(
            f"the transform returned value(s) for column(s) this binding does "
            f"not declare: {', '.join(repr(u) for u in unknown)}. A write only "
            f"ever touches the columns named in its column map "
            f"({', '.join(repr(a) for a in sorted(allowed)) or 'none'}), so a "
            "value for anything else is refused rather than dropped. Add the "
            "column to the binding, or stop returning it."
        )
    _check_flat(output, "the returned write map")
    return dict(output)


# ---------------------------------------------------------------------------
# Golden fixture
# ---------------------------------------------------------------------------

# Marker written into synthetic rows that need a recognisable value, so a
# fixture diff reads clearly instead of showing an unexplained string.
_SYNTHETIC_TEXT_NUMBER = "42"


def adversarial_rows(headers: list[str], sample_rows: list[list[Any]]) -> list[list[Any]]:
    """The probe's real rows, plus the four shapes real sheets always produce.

    Sample data is the happy path by definition — it is whatever the first few
    rows of the author's sheet happen to be — so a fixture built from it alone
    proves the transform works on rows nobody has trouble with.  These four are
    where generated code actually breaks:

    * **a fully empty row** — someone pressed enter at the bottom of the table;
      a transform that indexes or float()s without a guard dies here;
    * **a row with missing trailing cells** — the Sheets API truncates trailing
      empties, so this is not an edge case but the normal encoding of a blank
      last column (``sheets_rows_to_records`` pads it, and this proves the
      transform survives the padded ``""``);
    * **a number stored as text** — a cell formatted as text, or typed with a
      thousands separator; ``"1,234"`` is what a sum has to cope with;
    * **a duplicate key** — two rows with the same key column value, which is
      what makes "the row where id matches" ambiguous and a group-by
      interesting.

    Appended after the real rows so a fixture's first entries still look like
    the author's own data when they read it back.
    """
    width = len(headers)
    if not width:
        return [list(r) for r in sample_rows or []]

    rows: list[list[Any]] = [
        [row[i] if i < len(row) else "" for i in range(width)]
        for row in (sample_rows or [])
    ]

    # A fully empty row.
    rows.append([""] * width)
    # Missing trailing cells: only the first cell present, exactly as the API
    # would hand back a row whose remaining columns are blank.
    rows.append([_SYNTHETIC_TEXT_NUMBER])
    # A number stored as text, in every column (the transform's own column
    # choice decides which one matters, and this way none is missed).
    rows.append(["1,234"] * width)
    # A duplicate of the first real row, so some key column repeats.  With no
    # real rows to duplicate, a constant row twice does the same job.
    template = rows[0] if sample_rows else [_SYNTHETIC_TEXT_NUMBER] * width
    rows.append(list(template))
    return rows


def fixture_hash(value: Any) -> str:
    """``sha256:…`` over a canonicalised fixture side."""
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def compare_golden(expected: Any, actual: Any) -> None:
    """Raise unless *actual* matches the frozen *expected* output.

    Called on every recompile and whenever the schema fingerprint changes,
    which is the point of freezing it: "the model regenerated the code and it
    still parses" is not evidence that it still computes the same answer, and a
    changed header row can change what a transform reads without changing a
    line of it.
    """
    if _canonical(expected) == _canonical(actual):
        return
    raise ComputeValidationError(
        "this binding's golden fixture no longer reproduces: the transform "
        "returns a different result for the same sample rows than it did when "
        "it was approved. Review the new output before activating it — the "
        "computation has changed, whether or not the instruction did.\n"
        f"expected: {_canonical(expected)[:800]}\n"
        f"actual:   {_canonical(actual)[:800]}"
    )
