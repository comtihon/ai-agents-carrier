#!/usr/bin/env python3
"""Migration: remove the "questions != []" route from the develop-a-ticket
workflow so unresolved planner questions no longer route to a re-run of the
researcher step (node_fc70fbf7f73f).

Route removed (wherever it lives):
    {"next": "node_fc70fbf7f73f", "when": "questions != []"}

Keeps the "plan" route (-> node_plan_approval) intact.

Handles BOTH known shapes, since the live/run-time copy has been observed
in shape (b):
  (a) ``routes`` directly on the planner step node (node_ce35772b6cfa).
  (b) ``routes`` on a separate switch step (node_27a8395841fd) that follows
      the planner step.

IMPORTANT — behavioural consequence of this migration (documented, not
solved here): once the "questions != []" route is gone, a planner run that
emits questions with no plan must signal that via the ask_context interrupt
inside the planner step itself (``context_sufficient: false``), not via
workflow routing. If the planner instead returns neither ``plan`` nor
``context_sufficient: false``, the switch/router will have no matching
route and will loud-fail — this is considered acceptable (fail loud rather
than silently loop), and is a deliberate consequence of this change.

Safety
------
- Dry-run by default. Pass --apply to actually write.
- NEVER run this against a real/remote MongoDB from an agent session.
  MONGODB_URI is read from the environment (default: localhost) — always
  point it at a local/scratch Mongo when testing.

Usage
-----
    python3 scripts/migrations/2026-07-06_update_develop_ticket_planner_routes.py [--workflow-id develop-a-ticket] [--apply]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime, timezone

_REMOVED_ROUTE = {"next": "node_fc70fbf7f73f", "when": "questions != []"}


def strip_questions_route(steps: list[dict]) -> tuple[list[dict], list[str]]:
    """Remove ``_REMOVED_ROUTE`` from every step's ``routes`` list, wherever
    it's found (planner step shape (a), separate switch step shape (b), or
    both). Returns (new_steps, ids_of_changed_steps)."""
    new_steps = copy.deepcopy(steps)
    changed_ids: list[str] = []
    for step in new_steps:
        routes = step.get("routes")
        if not routes:
            continue
        filtered = [r for r in routes if r != _REMOVED_ROUTE]
        if len(filtered) != len(routes):
            step["routes"] = filtered
            changed_ids.append(step.get("id", "<unknown>"))
    return new_steps, changed_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-id", default="develop-a-ticket")
    parser.add_argument("--apply", action="store_true", help="actually write the change (default: dry-run)")
    args = parser.parse_args()

    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_database = os.environ.get("MONGODB_DATABASE", "langgraph_backend")

    from pymongo import MongoClient

    client = MongoClient(mongodb_uri)
    col = client[mongodb_database]["workflow_definitions"]

    doc = col.find_one({"_id": args.workflow_id})
    if doc is None:
        print(f"No workflow_definitions doc found for _id={args.workflow_id!r}. Nothing to do.")
        return 1

    steps = doc.get("steps") or []
    for step in steps:
        if step.get("routes"):
            print(f"Found routes on step_id={step.get('id')!r}: {step['routes']}")

    new_steps, changed_ids = strip_questions_route(steps)
    print(f"Matched=1 (workflow_definitions._id={args.workflow_id!r})")
    if not changed_ids:
        print(
            f"Route {_REMOVED_ROUTE} not found on any step — nothing to change. "
            "Nothing to apply."
        )
        return 0

    print(f"Planned change: remove {_REMOVED_ROUTE} from step(s): {changed_ids}")
    for step in new_steps:
        if step.get("id") in changed_ids:
            print(f"  step_id={step['id']!r} routes AFTER: {step['routes']}")

    if not args.apply:
        print("\nDry-run only (no --apply passed). Modified=0.")
        return 0

    result = col.update_one(
        {"_id": args.workflow_id},
        {"$set": {"steps": new_steps, "updated_at": datetime.now(timezone.utc)}},
    )
    print(f"Modified={result.modified_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
