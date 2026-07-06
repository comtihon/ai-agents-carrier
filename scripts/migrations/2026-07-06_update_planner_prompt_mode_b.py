#!/usr/bin/env python3
"""Migration: rewrite the planner agent's "Mode B" system_prompt framing.

Old framing (removed): unresolved questions "loop back to researcher".
New framing: the planner analyzes ``previous_task`` (previous input/output)
plus clarification answers; if resolved it continues and produces the plan;
if not resolved it returns ``context_sufficient: false`` with the unanswered
questions — which go to a HUMAN via Slack, not back to another agent.

Safety
------
- Dry-run by default. Pass --apply to actually write.
- NEVER run this against a real/remote MongoDB from an agent session.
  MONGODB_URI is read from the environment (default: localhost) — always
  point it at a local/scratch Mongo when testing.
- Does targeted string surgery on the existing ``agent_input.system_prompt``
  found on the doc at runtime (we don't hardcode the full prompt text here
  since it may already differ from what this migration was authored
  against). If the "Mode B" section can't be located, nothing is written
  and a clear error is printed — a human must inspect the prompt by hand.

Usage
-----
    python3 scripts/migrations/2026-07-06_update_planner_prompt_mode_b.py [--agent-id planner] [--apply]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone

NEW_MODE_B_BODY = (
    "Analyze previous_task (previous input/output) and clarification answers; "
    "if resolved, continue and produce the plan; if not resolved, return "
    "context_sufficient: false plus unanswered questions (these go to a human "
    "via Slack, not to another agent)."
)

# Matches a "Mode B" section header (e.g. "**Mode B**", "### Mode B", "Mode B:")
# up to (but not including) the next header-like line or end of string.
_MODE_B_SECTION_RE = re.compile(
    r"(?P<header>(?:^|\n)\s{0,4}(?:#{1,6}\s*)?\**Mode B\**:?\s*\n?)"
    r"(?P<body>.*?)"
    r"(?=\n\s{0,4}(?:#{1,6}\s|\**Mode [A-Z]\**)|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Fallback: any lingering sentence framing unanswered questions as looping
# back to the researcher, wherever it appears in the prompt.
_LOOP_BACK_RE = re.compile(
    r"[^.\n]*quest(?:ion)?s?[^.\n]*(?:loop(?:s|ed)? back|route[sd]?|return[sed]*|go(?:es)? back)"
    r"[^.\n]*research(?:er)?[^.\n]*[.\n]",
    re.IGNORECASE,
)


def rewrite_mode_b(prompt: str) -> tuple[str, bool]:
    """Return (new_prompt, changed). Targeted surgery, no full replacement."""
    match = _MODE_B_SECTION_RE.search(prompt)
    if not match:
        # Nothing to safely rewrite — leave untouched, caller decides to abort.
        return prompt, False

    header = match.group("header")
    new_section = f"{header}{NEW_MODE_B_BODY}\n"
    new_prompt = prompt[: match.start()] + new_section + prompt[match.end() :]

    # Also strip any stray "loop back to researcher" framing elsewhere.
    new_prompt = _LOOP_BACK_RE.sub("", new_prompt)

    # Collapse any run of 3+ newlines left behind by the surgery above back
    # to a single blank line, so repeated runs of this migration are stable
    # (idempotent) instead of accumulating blank lines each time.
    new_prompt = re.sub(r"\n{3,}", "\n\n", new_prompt)

    return new_prompt, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", default="planner", help="agent_definitions _id to update (default: planner)")
    parser.add_argument("--apply", action="store_true", help="actually write the change (default: dry-run)")
    args = parser.parse_args()

    mongodb_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_database = os.environ.get("MONGODB_DATABASE", "langgraph_backend")

    from pymongo import MongoClient

    client = MongoClient(mongodb_uri)
    col = client[mongodb_database]["agent_definitions"]

    doc = col.find_one({"_id": args.agent_id})
    if doc is None:
        print(f"No agent_definitions doc found for _id={args.agent_id!r}. Nothing to do.")
        return 1

    agent_input = doc.get("agent_input") or {}
    current_prompt = agent_input.get("system_prompt")
    if not current_prompt:
        print(f"Doc {args.agent_id!r} has no agent_input.system_prompt. Nothing to do.")
        return 1

    new_prompt, changed = rewrite_mode_b(current_prompt)
    if not changed:
        print(
            "Could not locate a 'Mode B' section in the current system_prompt — "
            "aborting without writing. Inspect the prompt manually:\n"
            "--- current system_prompt ---\n" + current_prompt
        )
        return 1

    print(f"Matched=1 (agent_definitions._id={args.agent_id!r})")
    print("--- BEFORE ---")
    print(current_prompt)
    print("--- AFTER ---")
    print(new_prompt)

    if new_prompt == current_prompt:
        print("No textual change produced — nothing to apply.")
        return 0

    if not args.apply:
        print("\nDry-run only (no --apply passed). Modified=0.")
        return 0

    result = col.update_one(
        {"_id": args.agent_id},
        {
            "$set": {
                "agent_input.system_prompt": new_prompt,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
    print(f"Modified={result.modified_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
