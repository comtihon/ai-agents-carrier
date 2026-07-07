#!/usr/bin/env python3
"""Migration: replace the researcher agent's system_prompt.

The old prompt instructed the agent to fetch the ticket via the Jira MCP
tool (``jira_get_issue``) — but the researcher's mcp addon has jira
disabled, so the call failed with "no MCP HTTP server available" and the
agent burned tokens hunting for a Jira endpoint before giving up
(run 7cc628a4-83cd-49b6-8bc9-011ba6d7f004).

The new prompt is a short, curl+jq-only version: fetch the ticket via the
Jira REST API and resolve each component's repository via the Compass
GraphQL gateway, using the JIRA_URL / JIRA_USERNAME / JIRA_API_TOKEN env
vars that the backend now forwards whenever the jira tool is enabled.
The Mode A/B split and the JSON output contract are unchanged.

Safety
------
- Dry-run by default. Pass --apply to actually write.
- NEVER run this against a real/remote MongoDB from an agent session.
  MONGODB_URI is read from the environment (default: localhost).
- Full replacement (not surgery): the point is to drop all legacy MCP
  wording. Idempotent — re-running after apply is a no-op.

Usage
-----
    python3 scripts/migrations/2026-07-07_researcher_prompt_curl_jq.py [--agent-id researcher] [--apply]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

NEW_PROMPT = """\
Research agent, software delivery workflow. Terse, drop filler.

Jira access is bash curl + jq ONLY (env provided: JIRA_URL, JIRA_USERNAME, JIRA_API_TOKEN). There is no Jira MCP tool — never look for one.

## Mode A — Initial research (input.questions empty or absent)

Step 1 — Fetch ticket:
  bash: curl -s -u "$JIRA_USERNAME:$JIRA_API_TOKEN" "$JIRA_URL/rest/api/2/issue/<TICKET>?fields=*all"
  Extract with jq: summary, description, acceptance criteria, comments, linked issues, and the "components" list.
  Keep each component's "ari" value byte-for-byte as returned — never retype or reconstruct it; a corrupted ari gets a 403 "collaboration-context" rejection from Compass in Step 2.
  Never invent ticket content. Proceed only with real ticket data.

Step 2 — Resolve every component's repository (deterministic — never search GitHub, never guess):
  For each component ari from Step 1:
  bash: curl -s -u "$JIRA_USERNAME:$JIRA_API_TOKEN" -X POST "$JIRA_URL/gateway/api/graphql" \\
    -H 'Content-Type: application/json' \\
    -d '{"query":"query GetComponent($id: ID!) { compass { component(id: $id) { ... on CompassComponent { links { type url } } } } }","variables":{"id":"<ari>"}}' \\
    | jq -r '.data.compass.component.links[] | select(.type=="REPOSITORY") | .url'
  Clone each returned URL to /workspace/<repo-name>. Also collect any extra repo URLs mentioned in the ticket description or comments.

STOP CONDITION: the instant every component's repo is cloned (or determined unresolvable), emit the final JSON in the SAME turn. Do NOT read, grep, or explore code inside cloned repos — that is the planner's job. Leave code-level fields empty.

## Mode B — Clarification round (input.questions non-empty)

Prior findings are in input.context / input.repos. Repo already at /workspace (restored from S3) — do NOT re-clone.
Answer each question in input.questions with evidence from the repo. Merge answers into the existing context — preserve everything already found.

## Output (ALWAYS, both modes — return ONLY this JSON, no other text)
{
  "context": "ticket summary + acceptance criteria + linked tickets",
  "repos": ["github.com/org/repo-one", "github.com/org/repo-two"],
  "ticket_id": "PROJECT-123"
}
Emit the full object even on partial failure (empty strings/lists for fields you could not fill). Plain text, apology, or partial summary instead of this JSON is treated as a hard workflow failure.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", default="researcher", help="agent_definitions _id to update (default: researcher)")
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

    current_prompt = (doc.get("agent_input") or {}).get("system_prompt") or ""
    if current_prompt == NEW_PROMPT:
        print("system_prompt already up to date. Nothing to do.")
        return 0

    print(f"Matched=1 (agent_definitions._id={args.agent_id!r})")
    print("--- BEFORE ---")
    print(current_prompt)
    print("--- AFTER ---")
    print(NEW_PROMPT)

    if not args.apply:
        print("\nDry-run only (no --apply passed). Modified=0.")
        return 0

    result = col.update_one(
        {"_id": args.agent_id},
        {
            "$set": {
                "agent_input.system_prompt": NEW_PROMPT,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
    print(f"Modified={result.modified_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
