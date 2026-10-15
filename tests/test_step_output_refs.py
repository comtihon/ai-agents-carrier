"""Step outputs in object storage, references in state (feature: step-output refs).

The production failure this guards against: a run document holding
``step_inputs`` (a full state snapshot per step) + ``state`` + ``step_outputs``
grew to steps x state, crossed Mongo's 16 MB limit (DocumentTooLarge) and the
backend crash-looped on resume.  These tests pin down the contract:

* every step output is written to ``runs/<run_id>/<step>.md`` and state keeps a
  small reference (offload on write);
* the engine resolves the references a step needs before templates, ``when``
  guards, switch conditions and a python step's ``state`` see them;
* a 170-step workflow with a ~200 KB state keeps its run document small;
* old run documents with inline values still load and resume;
* a run paused across a restart resumes from the stored files.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import bson
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.domain.models.graph_run import GraphRun
from app.infrastructure.datasources.datastream import LocalDiskStreamStore, validate_object_key
from app.infrastructure.orchestration.step_outputs import (
    StepInputTooLarge,
    StepOutputStore,
    is_output_ref,
    object_key,
    parse_markdown,
    referenced_keys,
    render_markdown,
    step_input_snapshot,
)
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
from app.infrastructure.persistence.mongo import MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider

BIG = "x" * 5000


class _Repo:
    """In-memory run repository that measures every write like Mongo would."""

    def __init__(self) -> None:
        self.runs: dict[str, GraphRun] = {}
        self.max_doc_bytes = 0

    async def create(self, run: GraphRun) -> None:
        await self.update(run)

    async def update(self, run: GraphRun) -> None:
        doc = MongoGraphRunRepository._to_doc(run)
        size = len(bson.encode(doc))
        self.max_doc_bytes = max(self.max_doc_bytes, size)
        if size > 16 * 1024 * 1024:
            raise RuntimeError(f"DocumentTooLarge: {size}")
        self.runs[run.id] = run.model_copy(deep=True)

    async def get(self, run_id: str) -> GraphRun | None:
        run = self.runs.get(run_id)
        return run.model_copy(deep=True) if run else None


class _CaptureLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        return AIMessage(content="ok")


def _runner(steps, *, store=None, llm=None, checkpointer=None, inline_max=256) -> YamlGraphRunner:
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner(
        {"id": "g", "steps": steps},
        llm=llm or _CaptureLLM(),
        mcp_tools_provider=mcp,
        checkpointer=checkpointer,
    )
    if store is not None:
        runner._stream_store = store
        runner._step_output_store = StepOutputStore(store, inline_max_bytes=inline_max)
    return runner


def _run(run_id: str, runner: YamlGraphRunner) -> GraphRun:
    return GraphRun(
        id=run_id, graph_id=runner.id, user_request="go", status="running",
        step_statuses={s["id"]: "pending" for s in runner.steps},
    )


@pytest.fixture
def store(tmp_path) -> LocalDiskStreamStore:
    return LocalDiskStreamStore(tmp_path / "streams")


def _py(step_id: str, code: str, **extra) -> dict[str, Any]:
    return {"id": step_id, "type": "python", "sandbox": False, "code": code, **extra}


# ---------------------------------------------------------------------------
# File format and key shape
# ---------------------------------------------------------------------------

def test_markdown_round_trips_and_carries_a_header():
    values = {"plan": {"items": [1, 2, 3], "note": "has ``` fences\n inside"}, "n": 3}
    data = render_markdown(run_id="r1", step_id="plan", step_type="python",
                           iteration=2, values=values)
    text = data.decode()
    assert text.startswith("# Step output: plan")
    assert "- type: `python`" in text and "- run: `r1`" in text and "written_at" in text
    assert "```json" in text
    assert parse_markdown(data) == values


def test_object_key_is_per_run_folder_with_iteration_suffix():
    assert object_key("run-1", "fetch") == "runs/run-1/fetch.md"
    assert object_key("run-1", "fetch", 3) == "runs/run-1/fetch-3.md"
    assert object_key("r", "a/../b") == "runs/r/a_.._b.md"
    validate_object_key(object_key("r", "a/../b"))
    for bad in ("../x", "runs/../x", "runs/.hidden", "/runs/x", "runs/a\\b"):
        with pytest.raises(ValueError):
            validate_object_key(bad)


def test_referenced_keys_scans_templates_and_conditions_not_structure():
    step = {"id": "fetch", "type": "llm", "next": "plan", "output_key": "summary",
            "user_template": "Summarise {plan[items]} for {customer}", "when": "ready and score > 3"}
    keys = referenced_keys(step)
    assert {"plan", "customer", "ready", "score"} <= keys
    assert "summary" not in keys and "fetch" not in keys
    assert referenced_keys({"id": "p", "type": "python", "code": "output = 1"}) is None


# ---------------------------------------------------------------------------
# Offload on write
# ---------------------------------------------------------------------------

async def test_outputs_are_written_to_the_run_folder_and_state_keeps_refs(store):
    steps = [
        _py("produce", "output = {'blob': 'x' * 5000, 'items': list(range(50))}", output_key="big"),
        _py("flag", "output = True", output_key="ok"),
    ]
    runner = _runner(steps, store=store)
    repo = _Repo()
    run = _run("run-offload", runner)
    await repo.create(run)
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})

    saved = await repo.get("run-offload")
    assert saved.status == "completed"
    ref = saved.state["big"]
    assert is_output_ref(ref)
    assert ref["key"] == "runs/run-offload/produce.md"
    assert ref["bytes"] > 5000 and len(ref["sha256"]) == 64 and ref["preview"]
    assert len(json.dumps(ref)) < 700
    # Small values stay inline (no bigger than their ref) -- but are still in the file.
    assert saved.state["ok"] is True
    assert parse_markdown(await store.get_object("runs/run-offload/flag.md")) == {"ok": True}
    # The checkpoint holds the ref, not the value.
    snap = await runner.graph.aget_state({"configurable": {"thread_id": "run-offload"}})
    assert is_output_ref(snap.values["big"])
    # step_outputs too.
    assert is_output_ref(saved.step_outputs["produce"]["big"])
    stored = parse_markdown(await store.get_object(ref["key"]))
    assert stored["big"]["items"] == list(range(50))


async def test_without_a_store_outputs_stay_inline(store):
    runner = _runner([_py("produce", "output = 'x' * 5000", output_key="big")])
    repo = _Repo()
    run = _run("run-inline", runner)
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})
    assert (await repo.get("run-inline")).state["big"] == BIG


# ---------------------------------------------------------------------------
# Hydration before the step runs
# ---------------------------------------------------------------------------

async def test_templates_see_values_not_refs(store):
    llm = _CaptureLLM()
    steps = [
        _py("produce", "output = {'title': 'Quarterly', 'body': 'y' * 3000}", output_key="doc"),
        {"id": "summarise", "type": "llm", "user_template": "Title={doc[title]} len={doc[body]}",
         "output_key": "summary"},
    ]
    runner = _runner(steps, store=store, llm=llm)
    run = _run("run-tpl", runner)
    await stream_graph_to_pause(runner, run, _Repo(), {"request": "go"})
    assert llm.prompts and llm.prompts[0].startswith("Title=Quarterly len=yyy")
    assert "__ref__" not in llm.prompts[0]


async def test_switch_conditions_and_when_guards_see_values(store):
    steps = [
        _py("produce", "output = list(range(300))", output_key="rows", next="route"),
        {"id": "route", "type": "switch", "routes": [
            {"when": "len(rows) > 100", "next": "many"},
            {"next": "few"},
        ]},
        _py("many", "output = 'many:' + str(len(state['rows']))", output_key="verdict", next="guarded"),
        _py("few", "output = 'few'", output_key="verdict", next="guarded"),
        {**_py("guarded", "output = 'ran'", output_key="guard_out"), "when": "rows"},
    ]
    runner = _runner(steps, store=store)
    run = _run("run-switch", runner)
    repo = _Repo()
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})
    saved = await repo.get("run-switch")
    assert is_output_ref(saved.state["rows"])
    assert saved.state["verdict"] == "many:300"
    assert saved.state["guard_out"] == "ran"


async def test_python_state_is_hydrated_and_capped(store):
    steps = [
        _py("produce", "output = {'n': 7, 'pad': 'z' * 2000}", output_key="data"),
        _py("consume", "output = state['data']['n'] * 6", output_key="answer"),
    ]
    runner = _runner(steps, store=store)
    repo = _Repo()
    run = _run("run-py", runner)
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})
    assert (await repo.get("run-py")).state["answer"] == 42

    capped = StepOutputStore(store, hydrate_max_bytes=100)
    state = {"data": (await repo.get("run-py")).state["data"]}
    with pytest.raises(StepInputTooLarge, match="STEP_INPUT_HYDRATE_MAX_BYTES"):
        await capped.hydrate(state, None, step_id="consume")


async def test_datastream_refs_are_left_alone(store):
    writer = await store.open_writer(source_id="crm", operation="list")
    await writer.append_many([{"id": i} for i in range(10)])
    ref = (await writer.close()).to_state()
    steps = [
        _py("fetch", f"output = {ref!r}", output_key="contacts"),
        _py("count", "output = sum(1 for _ in records())", stream="contacts", output_key="n"),
    ]
    runner = _runner(steps, store=store)
    repo = _Repo()
    run = _run("run-ds", runner)
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})
    saved = await repo.get("run-ds")
    assert saved.state["n"] == 10
    assert saved.state["contacts"]["__stream__"] == 1


# ---------------------------------------------------------------------------
# Regression: the run document stays small
# ---------------------------------------------------------------------------

async def test_170_steps_with_200kb_state_keep_the_run_document_small(store):
    n = 170
    steps = [
        _py(f"slot_{i:03d}", f"output = {{'slot': {i}, 'rows': ['r' * 60] * 20}}", output_key=f"slot_{i:03d}")
        for i in range(n)
    ]
    runner = _runner(steps, store=store)
    repo = _Repo()
    run = _run("run-big", runner)
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})
    saved = await repo.get("run-big")
    assert saved.status == "completed"

    total_state = 0
    for i in range(n):
        ref = saved.state[f"slot_{i:03d}"]
        total_state += len(json.dumps(parse_markdown(await store.get_object(ref["key"]))))
    assert total_state > 200_000  # the state really is ~200 KB+
    # Before: step_inputs alone was sum over steps of the state so far (~17 MB).
    assert repo.max_doc_bytes < 400_000, repo.max_doc_bytes
    for i in range(n):
        assert len(json.dumps(saved.step_inputs.get(f"slot_{i:03d}", {}))) < 2_000


async def test_step_inputs_hold_only_referenced_keys():
    state = {f"k{i}": "v" * 3000 for i in range(50)}
    state["plan"] = {"x": 1}
    snap = step_input_snapshot({"id": "s", "type": "llm", "user_template": "{plan} {k3}"}, state)
    assert set(snap) == {"plan", "k3"}
    assert snap["k3"]["__truncated__"] is True and snap["k3"]["bytes"] > 3000


# ---------------------------------------------------------------------------
# Backward compatibility and resume
# ---------------------------------------------------------------------------

async def test_old_runs_with_inline_values_hydrate_as_no_op(store):
    outputs = StepOutputStore(store)
    state = {"plan": {"items": [1, 2]}, "big": BIG}
    assert await outputs.hydrate(state, None) is state


async def test_resume_after_restart_reads_refs_from_the_store(store):
    """Pause at an approval, drop every in-memory object, resume on fresh ones."""
    steps = [
        _py("produce", "output = {'ids': list(range(500))}", output_key="cands", next="approve"),
        {"id": "approve", "type": "human_approval", "next": "consume"},
        _py("consume", "output = len(state['cands']['ids'])", output_key="count"),
    ]
    checkpointer = MemorySaver()  # stands in for the Mongo checkpointer
    repo = _Repo()
    runner = _runner(steps, store=store, checkpointer=checkpointer)
    run = _run("run-resume", runner)
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})
    paused = await repo.get("run-resume")
    assert paused.status == "waiting_approval"
    assert is_output_ref(paused.state["cands"])

    # "Restart": a new runner and a new output store (empty cache) over the
    # same durable store and checkpointer.
    fresh_store = LocalDiskStreamStore(store._dir)
    runner2 = _runner(steps, store=fresh_store, checkpointer=checkpointer)
    resumed = await repo.get("run-resume")
    resumed.status = "running"
    await stream_graph_to_pause(runner2, resumed, repo, Command(resume={"approved": True}))
    done = await repo.get("run-resume")
    assert done.status == "completed"
    assert done.state["count"] == 500


async def test_resume_of_a_legacy_checkpoint_with_inline_values(store):
    """A run that paused before this change resumes with offload switched on."""
    steps = [
        _py("produce", "output = {'ids': list(range(500))}", output_key="cands", next="approve"),
        {"id": "approve", "type": "human_approval", "next": "consume"},
        _py("consume", "output = len(state['cands']['ids'])", output_key="count"),
    ]
    checkpointer = MemorySaver()
    repo = _Repo()
    legacy = _runner(steps, checkpointer=checkpointer)  # no store: inline values
    run = _run("run-legacy", legacy)
    await stream_graph_to_pause(legacy, run, repo, {"request": "go"})
    assert isinstance((await repo.get("run-legacy")).state["cands"], dict)
    assert not is_output_ref((await repo.get("run-legacy")).state["cands"])

    upgraded = _runner(steps, store=store, checkpointer=checkpointer)
    resumed = await repo.get("run-legacy")
    resumed.status = "running"
    await stream_graph_to_pause(upgraded, resumed, repo, Command(resume={"approved": True}))
    done = await repo.get("run-legacy")
    assert done.status == "completed" and done.state["count"] == 500


async def test_missing_file_fails_the_step_with_a_clear_error(store):
    steps = [
        _py("produce", "output = 'p' * 4000", output_key="blob", next="consume"),
        {"id": "approve", "type": "human_approval", "next": "consume"},
        _py("consume", "output = len(state['blob'])", output_key="n"),
    ]
    steps[0]["next"] = "approve"
    checkpointer = MemorySaver()
    repo = _Repo()
    runner = _runner(steps, store=store, checkpointer=checkpointer)
    run = _run("run-gone", runner)
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})
    await store.delete_objects("runs/run-gone/")
    runner2 = _runner(steps, store=LocalDiskStreamStore(store._dir), checkpointer=checkpointer)
    resumed = await repo.get("run-gone")
    resumed.status = "running"
    await stream_graph_to_pause(runner2, resumed, repo, Command(resume={"approved": True}))
    failed = await repo.get("run-gone")
    assert failed.status == "failed"
    assert "runs/run-gone/produce.md" in failed.state["error"]
    assert failed.step_statuses["consume"] == "failed"


# ---------------------------------------------------------------------------
# Run API: refs resolved for display, files downloadable
# ---------------------------------------------------------------------------

async def test_run_response_resolves_refs_and_links_the_rest(store):
    from app.api.routes.workflows import _run_response

    steps = [
        _py("small", "output = {'a': 'q' * 400}", output_key="s"),
        _py("huge", "output = 'h' * 50000", output_key="h"),
    ]
    runner = _runner(steps, store=store)
    repo = _Repo()
    run = _run("run-api", runner)
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})
    saved = await repo.get("run-api")
    body = await _run_response(saved, runner, output_store=runner._step_output_store,
                               display_budget_bytes=10_000)
    by_id = {s["id"]: s for s in body["steps"]}
    assert by_id["small"]["output"]["s"] == {"a": "q" * 400}
    huge = by_id["huge"]["output"]["h"]
    assert is_output_ref(huge) and huge["download_url"] == "/runs/run-api/outputs/huge.md"
    assert body["intermediate_outputs"]["s"] == {"a": "q" * 400}


async def test_download_route_serves_markdown_and_single_values(store, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.routes import run_data

    steps = [_py("produce", "output = {'k': 'v' * 1000}", output_key="doc")]
    runner = _runner(steps, store=store)
    repo = _Repo()
    run = _run("run-dl", runner)
    await stream_graph_to_pause(runner, run, repo, {"request": "go"})

    container = MagicMock()
    container.run_repository = repo
    container.workflow_backend = None
    container.step_output_store = runner._step_output_store
    app = FastAPI()
    app.include_router(run_data.router)
    app.state.container = container
    client = TestClient(app)

    listing = client.get("/runs/run-dl/outputs").json()
    assert listing["outputs"][0]["name"] == "produce.md"
    md = client.get("/runs/run-dl/outputs/produce.md")
    assert md.status_code == 200 and md.text.startswith("# Step output: produce")
    assert client.get("/runs/run-dl/outputs/produce.md", params={"field": "doc"}).json() == {"k": "v" * 1000}
    # Only files the run references are reachable.
    assert client.get("/runs/run-dl/outputs/other.md").status_code == 404
