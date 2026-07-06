#!/usr/bin/env python3
"""Migration: grandfather existing agents onto the new "tools" addon.

The "tools" addon gates bash-level credentials/binaries (github/jira/graphify)
per agent. New semantics are STRICT: an agent with no tools addon gets ALL
tools disabled. To preserve existing behaviour, every agent_definitions doc
that does NOT already carry a ``tools`` addon gets one appended with all tools
ENABLED::

    {"type": "tools", "hidden": false,
     "tools": {"github": true, "jira": true, "graphify": true}}

Safety
------
- Dry-run by default. Pass --apply to actually write.
- NEVER run this against a real/remote MongoDB from an agent session.
  MONGODB_URI is read from the environment (default: localhost) — always
  point it at a local/scratch Mongo when testing.
- Idempotent: docs that already have a tools addon are skipped, so re-running
  after --apply reports Modified=0.

Usage
-----
    python3 scripts/migrations/2026-07-06_add_tools_addon_to_agents.py [--apply]
    python3 scripts/migrations/2026-07-06_add_tools_addon_to_agents.py --uri mongodb://localhost:27017 --db langgraph_backend --apply
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

TOOLS_ADDON = {
    "type": "tools",
    "hidden": False,
    "tools": {"github": True, "jira": True, "graphify": True},
}


def _has_tools_addon(doc: dict) -> bool:
    for addon in doc.get("addons") or []:
        if isinstance(addon, dict) and addon.get("type") == "tools":
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually write changes (default: dry-run)")
    parser.add_argument("--uri", default=None, help="override MONGODB_URI")
    parser.add_argument("--db", default=None, help="override MONGODB_DATABASE")
    args = parser.parse_args()

    mongodb_uri = args.uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_database = args.db or os.environ.get("MONGODB_DATABASE", "langgraph_backend")

    from pymongo import MongoClient

    client = MongoClient(mongodb_uri)
    col = client[mongodb_database]["agent_definitions"]

    matched = 0
    modified = 0
    for doc in col.find({}):
        matched += 1
        agent_id = doc.get("_id")
        if _has_tools_addon(doc):
            print(f"skip   {agent_id!r}: already has a tools addon")
            continue

        print(f"append {agent_id!r}: adding tools addon (all enabled)")
        if not args.apply:
            continue

        result = col.update_one(
            {"_id": agent_id},
            {
                "$push": {"addons": TOOLS_ADDON},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )
        modified += result.modified_count

    print(f"\nMatched={matched} Modified={modified}")
    if not args.apply:
        print("Dry-run only (no --apply passed). No writes performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
