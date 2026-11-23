"""The ``pubsub_publish`` workflow step."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core import config
from app.infrastructure.orchestration import pubsub_publish as pp
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner


class _Future:
    def __init__(self, message_id: str) -> None:
        self._id = message_id

    def result(self, timeout=None):
        return self._id


class FakePublisher:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.fail: Exception | None = None

    def publish(self, topic, data, **attrs):
        if self.fail:
            raise self.fail
        self.calls.append({"topic": topic, "data": data, "attrs": attrs})
        return _Future(f"m-{len(self.calls)}")


@pytest.fixture
def publisher(monkeypatch):
    fake = FakePublisher()
    pp.reset_client()
    monkeypatch.setattr(pp, "client_factory", lambda: fake)
    yield fake
    pp.reset_client()


def _settings(monkeypatch, allowed: str, project: str | None = "proj-a"):
    monkeypatch.setattr(
        config, "get_settings",
        lambda: SimpleNamespace(
            pubsub_publish_allowed_topics=allowed, pubsub_project_id=project
        ),
    )


def _node(step: dict):
    step = {"id": "notify", "type": "pubsub_publish", **step}
    runner = YamlGraphRunner({"id": "wf", "steps": [step]}, object(), lambda *a, **k: [])
    return runner._pubsub_publish_node(step)


STATE = {
    "event": {"projectId": "p-42"},
    "result": {"artifact_url": "gs://b/x.glb", "status": "done"},
    "env_name": "staging",
}


@pytest.mark.asyncio
async def test_publishes_templated_json_and_attributes(monkeypatch, publisher):
    _settings(monkeypatch, "service.modelapi.prediction")
    node = _node({
        "topic": "service.modelapi.prediction",
        "data": {"projectId": "{event.projectId}", "artifactUrl": "{result[artifact_url]}"},
        "attributes": {"source": "agent", "agentStatus": "{result[status]}"},
        "output_key": "published",
    })

    out = await node(STATE)

    assert len(publisher.calls) == 1
    call = publisher.calls[0]
    assert call["topic"] == "projects/proj-a/topics/service.modelapi.prediction"
    assert json.loads(call["data"]) == {"projectId": "p-42", "artifactUrl": "gs://b/x.glb"}
    assert call["attrs"] == {"source": "agent", "agentStatus": "done"}
    assert out == {"published": {
        "message_id": "m-1",
        "topic": "projects/proj-a/topics/service.modelapi.prediction",
    }}


@pytest.mark.asyncio
async def test_string_data_and_templated_topic_and_project(monkeypatch, publisher):
    _settings(monkeypatch, "projects/proj-b/topics/staging-*")
    node = _node({
        "topic": "{env_name}-service.modelapi.prediction",
        "project_id": "proj-b",
        "data": "done {event[projectId]}",
    })

    out = await node(STATE)

    assert publisher.calls[0]["topic"] == "projects/proj-b/topics/staging-service.modelapi.prediction"
    assert publisher.calls[0]["data"] == b"done p-42"
    assert publisher.calls[0]["attrs"] == {}
    assert out["notify"]["message_id"] == "m-1"  # output_key defaults to step id


@pytest.mark.asyncio
async def test_full_path_topic_matched_by_short_name_glob(monkeypatch, publisher):
    _settings(monkeypatch, "other, service.modelapi.*", project=None)
    out = await _node({"topic": "projects/x/topics/service.modelapi.prediction"})(STATE)
    assert out["notify"]["topic"] == "projects/x/topics/service.modelapi.prediction"


@pytest.mark.asyncio
async def test_topic_not_in_allowlist_fails_step(monkeypatch, publisher):
    _settings(monkeypatch, "service.modelapi.prediction")
    out = await _node({"topic": "billing.events", "data": {"a": 1}})(STATE)
    assert out["__failed_step__"] == "notify"
    assert "not in PUBSUB_PUBLISH_ALLOWED_TOPICS" in out["notify"]["error"]
    assert publisher.calls == []


@pytest.mark.asyncio
async def test_empty_allowlist_disables_publishing(monkeypatch, publisher):
    _settings(monkeypatch, "")
    out = await _node({"topic": "service.modelapi.prediction"})(STATE)
    assert out["__failed_step__"] == "notify"
    assert "disabled" in out["notify"]["error"]
    assert publisher.calls == []


@pytest.mark.asyncio
async def test_short_name_without_project_fails(monkeypatch, publisher):
    _settings(monkeypatch, "*", project=None)
    out = await _node({"topic": "service.modelapi.prediction"})(STATE)
    assert out["__failed_step__"] == "notify"
    assert "PUBSUB_PROJECT_ID" in out["notify"]["error"]


@pytest.mark.asyncio
async def test_publisher_error_fails_step_unless_ignored(monkeypatch, publisher):
    _settings(monkeypatch, "*")
    publisher.fail = RuntimeError("boom")
    out = await _node({"topic": "t"})(STATE)
    assert out["__failed_step__"] == "notify"
    assert "boom" in out["notify"]["error"]

    out = await _node({"topic": "t", "ignore_errors": True})(STATE)
    assert "__failed_step__" not in out
    assert out["notify"] == {"error": "boom"}


def test_check_allowed_globs():
    pp.check_allowed("projects/p/topics/service.modelapi.prediction", ["service.modelapi.prediction"])
    pp.check_allowed("projects/p/topics/staging-x", ["projects/p/topics/staging-*"])
    with pytest.raises(PermissionError):
        pp.check_allowed("projects/q/topics/staging-x", ["projects/p/topics/staging-*"])
    with pytest.raises(PermissionError):
        pp.check_allowed("projects/p/topics/foo", [])


def test_client_is_cached(monkeypatch):
    made = []
    pp.reset_client()
    monkeypatch.setattr(pp, "client_factory", lambda: made.append(1) or FakePublisher())
    assert pp._get_client() is pp._get_client()
    assert made == [1]
    pp.reset_client()


def test_validation():
    ok = {"id": "a", "type": "pubsub_publish", "topic": "{t}", "data": {"x": 1},
          "attributes": {"s": "v"}}
    assert pp.validate_pubsub_publish_steps([ok, {"id": "b", "type": "llm"}]) == []
    errors = pp.validate_pubsub_publish_steps([
        {"id": "a", "type": "pubsub_publish"},
        {"id": "b", "type": "pubsub_publish", "topic": "projects/p/subs/x"},
        {"id": "c", "type": "pubsub_publish", "topic": "t", "data": 5,
         "attributes": {"k": {"n": 1}}, "timeout_seconds": 0},
    ])
    joined = "; ".join(errors)
    assert "step 'a': 'topic' is required" in joined
    assert "step 'b': invalid Pub/Sub topic path" in joined
    assert "step 'c': 'data'" in joined
    assert "attribute 'k'" in joined
    assert "timeout_seconds" in joined


def test_make_node_dispatches_pubsub_publish():
    runner = YamlGraphRunner(
        {"id": "wf", "steps": [{"id": "n", "type": "pubsub_publish", "topic": "t"}]},
        object(), lambda *a, **k: [],
    )
    assert callable(runner._make_node({"id": "n", "type": "pubsub_publish", "topic": "t"}))


def test_render_dot_access_reads_dict_keys():
    r = YamlGraphRunner._render
    assert r("{event.projectId}/{event.missing}/{event[projectId]}", STATE) == "p-42//p-42"
