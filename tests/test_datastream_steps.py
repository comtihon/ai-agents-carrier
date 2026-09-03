"""Steps consuming a data source result, which is always a file.

Covers the three shapes a consumer can take: a `python` step reading records
off the descriptor, an `llm` step that cannot read a descriptor at all and so
is given a bounded selection, and `result_mode: ram` as the escape hatch for a
workflow that needs the value inline.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.domain.models.data_source_definition import DataSourceDefinition
from app.domain.models.datastream import DataRef, as_data_ref, is_data_ref
from app.infrastructure.datasources.datastream import LocalDiskStreamStore
from app.infrastructure.orchestration.yaml_graph import (
    YamlGraphRunner,
    _stream_safe_state,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider

_SOURCE = DataSourceDefinition.model_validate({
    "id": "crm", "base_url": "https://api.test",
    "operations": [{"name": "list_contacts", "path": "/contacts"}],
})

ROWS = [{"id": i, "amount": i * 10, "region": "eu" if i % 2 else "us"}
        for i in range(1, 1001)]


@pytest.fixture
async def streamed(tmp_path):
    """A store holding ROWS, plus the ref pointing at them."""
    store = LocalDiskStreamStore(tmp_path / "streams")
    writer = await store.open_writer(source_id="crm", operation="list_contacts")
    await writer.append_many(ROWS)
    ref = await writer.close()
    return store, ref


class _CaptureLLM:
    """Records the prompts it is asked, so a test can assert on them.

    A plain object rather than a fake chat model: the fakes are pydantic
    models and refuse an assigned `ainvoke`.
    """

    def __init__(self, reply=lambda n: f"part{n}") -> None:
        self.prompts: list[str] = []
        self._reply = reply

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self._reply(len(self.prompts)))


def _runner(steps) -> YamlGraphRunner:
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="answer")])
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    return YamlGraphRunner({"id": "g", "steps": steps}, llm=llm, mcp_tools_provider=mcp)


# ---------------------------------------------------------------------------
# python step reads the descriptor
# ---------------------------------------------------------------------------

async def test_a_script_reads_every_record_off_the_stream(streamed):
    store, ref = streamed
    step = {
        "id": "sum", "type": "python", "stream": "contacts",
        "sandbox": False, "output_key": "totals",
        "code": (
            "n = 0\n"
            "total = 0\n"
            "for row in records():\n"
            "    n += 1\n"
            "    total += row['amount']\n"
            "output = {'count': n, 'amount': total, 'expected': stream_records}\n"
        ),
    }
    runner = _runner([step])
    runner._stream_store = store

    out = await (runner._python_node(step))({"contacts": ref.to_state()})

    assert out["totals"]["count"] == len(ROWS)
    assert out["totals"]["amount"] == sum(r["amount"] for r in ROWS)
    # The count is known up front, so a script can check it read everything.
    assert out["totals"]["expected"] == len(ROWS)


async def test_records_can_be_iterated_more_than_once(streamed):
    """It seeks back, so a two-pass script needs no second fetch."""
    store, ref = streamed
    step = {
        "id": "twopass", "type": "python", "stream": "contacts",
        "sandbox": False, "output_key": "out",
        "code": (
            "total = sum(r['amount'] for r in records())\n"
            "mean = total / stream_records\n"
            "above = sum(1 for r in records() if r['amount'] > mean)\n"
            "output = {'mean': mean, 'above': above}\n"
        ),
    }
    runner = _runner([step])
    runner._stream_store = store

    out = await (runner._python_node(step))({"contacts": ref.to_state()})

    assert out["out"]["mean"] == sum(r["amount"] for r in ROWS) / len(ROWS)
    assert out["out"]["above"] == 500


async def test_a_script_with_no_stream_declared_gets_a_clear_error(streamed):
    store, ref = streamed
    step = {
        "id": "s", "type": "python", "sandbox": False, "output_key": "out",
        "code": "output = sum(1 for _ in records())",
    }
    runner = _runner([step])
    runner._stream_store = store

    out = await (runner._python_node(step))({"contacts": ref.to_state()})

    assert "no data stream is attached" in out["out"]["error"]


async def test_stream_naming_a_key_that_is_not_a_result_is_an_error(streamed):
    store, _ = streamed
    step = {
        "id": "s", "type": "python", "stream": "nope", "sandbox": False,
        "code": "output = 1", "output_key": "out",
    }
    runner = _runner([step])
    runner._stream_store = store

    out = await (runner._python_node(step))({"nope": "a string"})

    assert "does not hold a data source result reference" in out["out"]["error"]


async def test_the_sandbox_is_handed_a_path_not_the_data(streamed, monkeypatch):
    """The delivery contract: local/docker get a path, never a value."""
    store, ref = streamed
    captured: dict = {}

    async def _fake_run_script(code, state, **kwargs):
        captured.update(kwargs)
        captured["state"] = state
        return {"ok": True}

    import app.infrastructure.orchestration.script_sandbox as sandbox_mod
    monkeypatch.setattr(sandbox_mod, "run_script", _fake_run_script)

    step = {
        "id": "s", "type": "python", "stream": "contacts",
        "sandbox": True, "sandbox_runtime": "local",
        "code": "output = 1", "output_key": "out",
    }
    runner = _runner([step])
    runner._stream_store = store

    await (runner._python_node(step))({"contacts": ref.to_state()})

    assert captured["stream_path"].endswith(f"{ref.id}.jsonl")
    assert captured["stream_records"] == len(ROWS)
    assert "stream_copy" not in captured
    # State carries the ref, not the rows.
    assert is_data_ref(captured["state"]["contacts"])


async def test_a_k8s_sandbox_is_handed_a_copy_callable(streamed, monkeypatch):
    """Another pod: the bytes have to be transferred, so a copier is passed."""
    store, ref = streamed
    captured: dict = {}

    async def _fake_run_script(code, state, **kwargs):
        captured.update(kwargs)
        return None

    import app.infrastructure.orchestration.script_sandbox as sandbox_mod
    monkeypatch.setattr(sandbox_mod, "run_script", _fake_run_script)

    step = {
        "id": "s", "type": "python", "stream": "contacts",
        "sandbox": True, "sandbox_runtime": "k8s",
        "code": "output = 1", "output_key": "out",
    }
    runner = _runner([step])
    runner._stream_store = store

    await (runner._python_node(step))({"contacts": ref.to_state()})

    assert "stream_path" not in captured
    assert callable(captured["stream_copy"])

    # And the copier really writes the bytes into whatever sink it is given.
    class _Sink:
        def __init__(self): self.blocks = []
        def write(self, b): self.blocks.append(b); return len(b)

    sink = _Sink()
    written = await captured["stream_copy"](sink)
    assert written == ref.bytes
    joined = b"".join(sink.blocks).decode()
    assert json.loads(joined.splitlines()[0]) == ROWS[0]
    assert len(joined.splitlines()) == len(ROWS)


# ---------------------------------------------------------------------------
# llm step — cannot read a descriptor
# ---------------------------------------------------------------------------

async def test_an_llm_step_gets_a_stated_sample_read_off_the_file(streamed):
    store, ref = streamed
    step = {
        "id": "ask", "type": "llm", "stream": "contacts",
        "sample_records": 4,
        "user_template": "Describe the data.", "output_key": "answer",
    }
    runner = _runner([step])
    runner._stream_store = store
    llm = _CaptureLLM(reply=lambda n: "ok")
    runner._get_llm_for_step = lambda s: llm

    out = await (runner._llm_node(step))({"contacts": ref.to_state()})

    assert out["answer"] == "ok"
    prompt = llm.prompts[0]
    assert "1000 records are available" in prompt
    assert "NOT included here" in prompt
    assert "first 4 records" in prompt
    # Four records, not a thousand.
    assert prompt.count('"amount"') == 4
    assert len(prompt) < 4000


async def test_llm_map_reduce_reads_every_chunk_then_combines(streamed):
    store, ref = streamed
    step = {
        "id": "ask", "type": "llm", "stream": "contacts",
        "stream_mode": "map_reduce", "chunk_items": 250,
        "user_template": "Total the amounts.", "output_key": "answer",
    }
    runner = _runner([step])
    runner._stream_store = store
    llm = _CaptureLLM()
    runner._get_llm_for_step = lambda s: llm

    out = await (runner._llm_node(step))({"contacts": ref.to_state()})

    # 4 map calls (1000 / 250) + 1 combining call.
    assert len(llm.prompts) == 5
    assert "part1" in llm.prompts[-1] and "part4" in llm.prompts[-1]
    assert out["answer"] == "part5"


async def test_unknown_stream_mode_is_rejected(streamed):
    store, ref = streamed
    step = {
        "id": "ask", "type": "llm", "stream": "contacts",
        "stream_mode": "inline_everything", "output_key": "answer",
    }
    runner = _runner([step])
    runner._stream_store = store

    out = await (runner._llm_node(step))({"contacts": ref.to_state()})

    assert "unknown stream_mode" in out["answer"]["error"]


# ---------------------------------------------------------------------------
# data_source step
# ---------------------------------------------------------------------------

async def test_the_data_source_step_passes_the_reference_through(streamed):
    store, ref = streamed
    backend, executor = AsyncMock(), AsyncMock()
    backend.get.return_value = _SOURCE
    executor.execute.return_value = ref.to_state()

    step = {
        "id": "fetch", "type": "data_source", "source": "crm",
        "operation": "list_contacts", "output_key": "contacts",
    }
    runner = _runner([step])
    runner._data_source_backend = backend
    runner._data_source_executor = executor
    runner._stream_store = store

    out = await (runner._data_source_node(step))({})

    assert is_data_ref(out["contacts"])
    assert as_data_ref(out["contacts"]).items == len(ROWS)


async def test_result_mode_ram_loads_it_back_for_workflows_that_need_it(streamed):
    """The escape hatch: route conditions and http_call bodies need values."""
    store, ref = streamed
    backend, executor = AsyncMock(), AsyncMock()
    backend.get.return_value = _SOURCE
    executor.execute.return_value = ref.to_state()

    step = {
        "id": "fetch", "type": "data_source", "source": "crm",
        "operation": "list_contacts", "output_key": "contacts",
        "result_mode": "ram",
    }
    runner = _runner([step])
    runner._data_source_backend = backend
    runner._data_source_executor = executor
    runner._stream_store = store

    out = await (runner._data_source_node(step))({})

    assert isinstance(out["contacts"], list)
    assert len(out["contacts"]) == len(ROWS)


async def test_result_mode_ram_refuses_what_will_not_fit(streamed, monkeypatch):
    store, ref = streamed
    backend, executor = AsyncMock(), AsyncMock()
    backend.get.return_value = _SOURCE
    executor.execute.return_value = ref.to_state()

    from app.core import config as config_mod
    monkeypatch.setattr(
        config_mod.get_settings(), "stream_read_all_max_bytes", 1024, raising=False
    )

    step = {
        "id": "fetch", "type": "data_source", "source": "crm",
        "operation": "list_contacts", "output_key": "contacts",
        "result_mode": "ram",
    }
    runner = _runner([step])
    runner._data_source_backend = backend
    runner._data_source_executor = executor
    runner._stream_store = store

    out = await (runner._data_source_node(step))({})

    assert "over the" in out["contacts"]["error"]


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------

def test_a_reference_renders_as_a_summary_not_as_a_dict():
    """`{key}` in any template must never paste the ref's guts."""
    ref = DataRef(id="ds_abc", items=1200, bytes=5_000_000,
                  source_id="crm", operation="list_contacts")
    state = {"contacts": ref.to_state(), "request": "hi"}

    rendered = YamlGraphRunner._render("got {contacts}", state)

    assert "1200 items" in rendered
    assert "__stream__" not in rendered and "ds_abc" not in rendered


def test_a_template_can_still_read_the_count():
    ref = DataRef(id="ds_abc", items=1200, bytes=99)
    state = {"contacts": ref.to_state()}

    assert YamlGraphRunner._render("{contacts[items]}", state) == "1200"


def test_stream_safe_state_leaves_ordinary_state_untouched():
    state = {"a": 1, "b": [1, 2]}
    assert _stream_safe_state(state) is state
