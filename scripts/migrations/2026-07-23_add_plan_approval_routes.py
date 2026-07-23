#!/usr/bin/env python3
"""Migration: convert the develop-a-ticket ``node_plan_approval`` human_approval
step from a flat unconditional ``next`` edge into a conditional ``routes`` pair
so the approval step becomes a real goalkeeper.

Before (flat, unconditional — the bug):
    node_plan_approval.next = "node_26283bfddd5b"   # continues regardless of decision

After (conditional route):
    node_plan_approval.routes = [
        {"when": "plan_approved", "next": "node_26283bfddd5b"},  # approve -> next step
        {"next": "END"},                                        # reject  -> END (rejected)
    ]
    (node_plan_approval.next removed)

The ``when`` guard uses the step's ``output_key`` (``plan_approved``), which the
approval node writes as a bool. Approve routes to the current next step; reject
falls through to END, and the run terminates with status "rejected".

Live values confirmed against the UAT ``develop-a-ticket`` definition
(2026-07-23): node_plan_approval.output_key == "plan_approved",
node_plan_approval.next == "node_26283bfddd5b".

Safety
------
- Dry-run by default. Pass --apply to actually write.
- Idempotent: if the step already has a ``routes`` field, it is left untouched.
- DEPLOYMENT ORDERING: apply this ONLY AFTER the new backend code (which adds
  "human_approval" to _MULTI_OUTPUT_TYPES) is deployed. Applying it against a
  backend that still treats human_approval as single-output would fail to build
  the graph. NEVER run --apply from an agent session.
- MONGODB_URI is read from the environment (default: localhost) — always point
  it at a local/scratch Mongo when testing.

Usage
-----
    python3 scripts/migrations/2026-07-23_add_plan_approval_routes.py [--workflow-id develop-a-ticket] [--step-id node_plan_approval] [--apply]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone


def build_routes(step: dict) -> list[dict]:
    """Build the approve->next / reject->END conditional routes from the step's
    current flat ``next`` value and its ``output_key`` guard."""
    guard = step.get("output_key", "approved")
    current_next = step.get("next")
    return [
        {"when": guard, "next": current_next},
        {"next": "END"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-id", default="develop-a-ticket")
    parser.add_argument("--step-id", default="node_plan_approval")
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
    target = next((s for s in steps if s.get("id") == args.step_id), None)
    if target is None:
        print(f"No step with id={args.step_id!r} found in workflow {args.workflow_id!r}. Nothing to do.")
        return 1

    print(f"Matched=1 (workflow_definitions._id={args.workflow_id!r}, step_id={args.step_id!r})")
    print(f"Step BEFORE:\n{json.dumps(target, indent=2, default=str)}")

    if target.get("routes"):
        print(
            f"\nStep {args.step_id!r} already has a 'routes' field — nothing to change "
            "(idempotent). Nothing to apply."
        )
        return 0

    if target.get("next") is None:
        print(
            f"\nStep {args.step_id!r} has no 'next' value to convert — cannot build "
            "approve route. Nothing to apply."
        )
        return 1

    new_steps = copy.deepcopy(steps)
    new_target = next(s for s in new_steps if s.get("id") == args.step_id)
    new_target["routes"] = build_routes(new_target)
    del new_target["next"]

    print(f"\nStep AFTER:\n{json.dumps(new_target, indent=2, default=str)}")

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
