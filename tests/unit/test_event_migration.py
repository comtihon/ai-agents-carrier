"""The one-off script that moves kind="pubsub" data sources into the events.

The script talks to Mongo through pymongo directly (that is the house style for
``scripts/migrations``), so the collections are faked here — what is under test
is the flattening, the idempotency and the dry-run guard, not motor.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts" / "migrations" / "2026-08-19_move_pubsub_datasources_to_events.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("move_pubsub_to_events", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


migration = _load_script()


class FakeCollection:
    def __init__(self, docs: list[dict] | None = None) -> None:
        self.docs = {d["_id"]: dict(d) for d in (docs or [])}

    def find(self, query: dict):
        return [
            dict(d) for d in self.docs.values()
            if all(d.get(k) == v for k, v in query.items())
        ]

    def find_one(self, query: dict):
        return next(iter(self.find(query)), None)

    def insert_one(self, doc: dict) -> None:
        self.docs[doc["_id"]] = dict(doc)

    def delete_one(self, query: dict) -> None:
        self.docs.pop(query["_id"], None)


class FakeClient:
    def __init__(self, sources: FakeCollection, events: FakeCollection) -> None:
        self._db = {
            migration.SOURCE_COLLECTION: sources,
            migration.TARGET_COLLECTION: events,
        }

    def __getitem__(self, _database: str) -> dict:
        return self._db


def _pubsub_source(**overrides) -> dict:
    doc = {
        "_id": "pubsub-orders",
        "name": "Order events",
        "description": "Shop orders",
        "kind": "pubsub",
        "base_url": "",
        "operations": [],
        "pubsub": {
            "topic": "orders",
            "subscription": "projects/p/subscriptions/aac-orders",
            "project_id": "p",
            "event_schema": {"type": "object"},
        },
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def run(monkeypatch):
    """Run the script's main() against fake collections; returns them."""
    def _run(sources: list[dict], events: list[dict] | None = None, *, argv: list[str]):
        src = FakeCollection(sources)
        dst = FakeCollection(events)
        import pymongo
        monkeypatch.setattr(pymongo, "MongoClient", lambda uri: FakeClient(src, dst))
        monkeypatch.setattr(sys, "argv", ["migration", *argv])
        assert migration.main() == 0
        return src, dst
    return _run


def test_the_pubsub_block_is_flattened_onto_the_event():
    doc = migration.to_event_doc(_pubsub_source())

    assert doc["_id"] == "pubsub-orders"
    assert doc["topic"] == "orders"
    assert doc["subscription"] == "projects/p/subscriptions/aac-orders"
    assert doc["project_id"] == "p"
    assert doc["event_schema"] == {"type": "object"}
    # None of the data source's own machinery comes along.
    assert "kind" not in doc and "operations" not in doc and "base_url" not in doc


def test_a_dry_run_writes_nothing(run):
    src, dst = run([_pubsub_source()], argv=[])

    assert list(dst.docs) == []
    assert list(src.docs) == ["pubsub-orders"]


def test_applying_moves_the_document(run):
    src, dst = run([_pubsub_source()], argv=["--apply"])

    assert dst.docs["pubsub-orders"]["topic"] == "orders"
    assert src.docs == {}


def test_http_data_sources_are_left_alone(run):
    http = {"_id": "github", "kind": "http", "base_url": "https://api.github.com"}
    src, dst = run([http], argv=["--apply"])

    assert list(src.docs) == ["github"]
    assert dst.docs == {}


def test_rerunning_over_an_already_migrated_event_only_clears_the_source(run):
    existing_event = {"_id": "pubsub-orders", "topic": "orders", "subscription": "kept"}
    src, dst = run([_pubsub_source()], [existing_event], argv=["--apply"])

    # The event that is already there wins — it may have been edited since.
    assert dst.docs["pubsub-orders"]["subscription"] == "kept"
    assert src.docs == {}
