#!/usr/bin/env python3
"""Migration: move ``kind="pubsub"`` data sources into the new ``events``
collection, where Pub/Sub topics now live.

Before (``data_source_definitions``):
    {"_id": "pubsub-orders", "name": "Orders", "kind": "pubsub",
     "base_url": "", "auth": {...}, "operations": [],
     "pubsub": {"topic": "orders", "subscription": "projects/p/subscriptions/s",
                "project_id": "p", "event_schema": {...}}}

After (``event_definitions``):
    {"_id": "pubsub-orders", "name": "Orders",
     "topic": "orders", "subscription": "projects/p/subscriptions/s",
     "project_id": "p", "event_schema": {...}}
    (the data_source_definitions document is deleted)

The ``_id`` is preserved, so a workflow step that still says
``datasource: pubsub-orders`` keeps resolving — the backend reads that key as
the pre-events spelling of ``event`` and looks in the events first.

Safety
------
- Dry-run by default. Pass --apply to actually write.
- Idempotent: a topic already present in event_definitions is not copied again,
  and the source document is only deleted once the event exists.
- DEPLOYMENT ORDERING: apply this ONLY AFTER the backend code that serves
  /api/v1/events is deployed. Applying it earlier leaves the pubsub triggers
  unresolvable until the new code lands. NEVER run --apply from an agent
  session.
- MONGODB_URI is read from the environment (default: localhost) — always point
  it at a local/scratch Mongo when testing.

Usage
-----
    python3 scripts/migrations/2026-08-19_move_pubsub_datasources_to_events.py [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

SOURCE_COLLECTION = "data_source_definitions"
TARGET_COLLECTION = "event_definitions"


def to_event_doc(source: dict) -> dict:
    """Flatten a pubsub data source document into an event document."""
    pubsub = source.get("pubsub") or {}
    return {
        "_id": source["_id"],
        "name": source.get("name", ""),
        "description": source.get("description"),
        "topic": pubsub.get("topic", ""),
        "subscription": pubsub.get("subscription", ""),
        "project_id": pubsub.get("project_id", ""),
        "event_schema": pubsub.get("event_schema"),
        "created_at": source.get("created_at"),
        "updated_at": datetime.now(timezone.utc),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually write the change (default: dry-run)")
    args = parser.parse_args()

    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_database = os.environ.get("MONGODB_DATABASE", "langgraph_backend")

    from pymongo import MongoClient

    client = MongoClient(mongodb_uri)
    db = client[mongodb_database]
    sources = db[SOURCE_COLLECTION]
    events = db[TARGET_COLLECTION]

    docs = list(sources.find({"kind": "pubsub"}))
    if not docs:
        print(f"No {SOURCE_COLLECTION} documents with kind='pubsub'. Nothing to do.")
        return 0

    print(f"Found {len(docs)} pubsub data source(s) to move:")
    moved = 0
    for source in docs:
        event_doc = to_event_doc(source)
        already = events.find_one({"_id": event_doc["_id"]})
        print(f"  {event_doc['_id']}: topic={event_doc['topic']!r}"
              f"{' (event already exists — only deleting the source)' if already else ''}")
        if not args.apply:
            print("    " + json.dumps(event_doc, default=str))
            continue
        if already is None:
            events.insert_one(event_doc)
        sources.delete_one({"_id": source["_id"]})
        moved += 1

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to migrate.")
        return 0

    print(f"\nMigrated {moved} event(s) into {TARGET_COLLECTION}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
