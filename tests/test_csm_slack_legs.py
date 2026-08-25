"""The CSM deadline watcher's Slack legs, expressed with the ``slack`` step.

The watcher (workflow ``csm-deadline-watcher``, deliberately disabled and
trigger-less) has three Slack legs in its spec, and until now they were the only
part of it that had to be written with ``http_call``:

  1. read ~26h of ``#csm-deadline-tracker-agent`` and hand every message
     verbatim to the library script ``csm-deadline-parse-overrides``,
  2. confirm each accepted override in its own thread, skipping any thread that
     already carries the confirmation,
  3. post the digest to the channel — but when ``export_ok`` is false, post
     **nothing** to the channel and DM instead.

These are the step definitions for those legs, and the tests below hold them to
the three properties that actually matter: the read leg's output is the shape the
parser consumes, the confirm leg is idempotent, and a bad export can reach the
DM but never the channel.  A false all-clear in that channel is the one outcome
the design cannot have, because nobody notices it.
"""
from __future__ import annotations

import pytest

from app.infrastructure.messaging import register_provider, reset_providers
from app.infrastructure.messaging import registry as provider_registry
# The legs run on the Slack provider in production, so the fixtures below use
# its message type: `ts`/`user` are Slack's own field names and the library
# parser reads exactly those.
from app.infrastructure.messaging.slack import SlackMessage
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from tests.test_yaml_graph_messaging import FAKE, FakeProvider

CHANNEL = "C0BLDDSEB1D"  # #csm-deadline-tracker-agent

CSM_SLACK_LEGS: list[dict] = [
    # 1. Read the override channel.  ``ignore_errors`` on purpose: the deadline
    #    check must still run when Slack is unreachable — the stored overrides
    #    then stand.  ``oldest`` is templated so the 26-hour window can be
    #    narrowed later without touching the step; unset, it reads the last
    #    ``limit`` messages, which the idempotency guard below makes safe.
    {
        "id": "read_deadline_commands",
        "type": "slack",
        "action": "history",
        "channel": CHANNEL,
        "oldest": "{slack_oldest}",
        "limit": 200,
        "output_key": "slack_messages",
        "ignore_errors": True,
        "next": "parse_deadline_commands",
    },
    # 2. The deterministic parser, unchanged.  It reads state["slack_messages"]
    #    and never interprets message text itself.
    {
        "id": "parse_deadline_commands",
        "type": "python",
        "script_id": "csm-deadline-parse-overrides",
        "output_key": "overrides_parsed",
        "sandbox": True,
        "sandbox_runtime": "k8s",
        "timeout_seconds": 120,
        "next": "confirm_overrides",
    },
    # 3. One thread confirmation per accepted override, skipping the ones
    #    already confirmed on an earlier run over the overlapping window.
    {
        "id": "confirm_overrides",
        "type": "slack",
        "action": "reply",
        "channel": CHANNEL,
        "items": "overrides_parsed.confirmations",
        "skip_if_replied": True,
        "output_key": "confirmations_posted",
        "ignore_errors": True,
        "next": "gate_export",
    },
    # 4. The failure path is routing, not a flag: there is no edge from here to
    #    the channel post when the export is unusable.
    {
        "id": "gate_export",
        "type": "switch",
        "routes": [
            {"when": "export_ok", "next": "post_digest"},
            {"when": "!export_ok", "next": "dm_owner"},
        ],
    },
    # 5. The digest, posted unchanged — it already carries the owner's <@id>
    #    mention, which is why it must not be reformatted here.
    {
        "id": "post_digest",
        "type": "slack",
        "action": "post",
        "channel": CHANNEL,
        "text": "{digest}",
        "output_key": "digest_posted",
    },
    # 6. No usable export: DM instead.  Not ignore_errors — a DM that cannot be
    #    delivered has to fail the run loudly, since the alternative is silence.
    {
        "id": "dm_owner",
        "type": "slack",
        "action": "dm",
        "user_id": "{fallback_owner_slack_id}",
        "text": "CSM-Deadline-Watcher: kein brauchbarer Export, keine Meldung im "
                "Kanal.\n\n{board_warning}",
        "output_key": "digest_dm",
    },
]


class _FakeLLM:
    """No Slack leg calls the chat model."""


@pytest.fixture
def provider():
    register_provider(FakeProvider)
    reset_providers()
    yield provider_registry.get_provider(FAKE)
    provider_registry._PROVIDERS.pop(FAKE, None)
    reset_providers()


def _runner(steps: list[dict] | None = None) -> YamlGraphRunner:
    return YamlGraphRunner(
        {"id": "csm-deadline-watcher", "steps": steps or CSM_SLACK_LEGS},
        _FakeLLM(),
        lambda *a, **k: [],
    )


def _leg(step_id: str) -> dict:
    step = next(s for s in CSM_SLACK_LEGS if s["id"] == step_id)
    return {**step, "provider": FAKE}


# ---------------------------------------------------------------------------
# The legs are valid platform configuration
# ---------------------------------------------------------------------------

def test_the_legs_compile_into_a_graph():
    assert _runner().graph is not None


def test_no_leg_uses_http_call_or_carries_a_credential():
    """The point of the migration: no hand-rolled Slack HTTP, no token in the
    definition.  A credential here would be persisted in the workflow record."""
    for step in CSM_SLACK_LEGS:
        assert step["type"] != "http_call"
        rendered = repr(step)
        assert "xoxb" not in rendered
        for forbidden in ("token", "Authorization", "headers", "url"):
            assert forbidden not in step


def test_the_digest_leg_posts_to_the_tracker_channel_and_nothing_else():
    channels = {s.get("channel") for s in CSM_SLACK_LEGS if s.get("channel")}
    assert channels == {CHANNEL}
    # SLACK_APPROVALS_CHANNEL is a different channel with a different purpose.
    assert "C0ATMLKBJ02" not in repr(CSM_SLACK_LEGS)


# ---------------------------------------------------------------------------
# Leg 1: what the parser gets
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_read_leg_hands_the_parser_the_fields_it_reads(provider):
    """csm-deadline-parse-overrides pulls ``text``, ``ts`` and ``user`` off each
    entry of state["slack_messages"].  Those three are the whole contract."""
    provider.history = [
        SlackMessage(id="1787600001.1", channel=CHANNEL, author="U07JVE4DEH1",
                     text="FRIST 128498: 3"),
    ]
    node = _runner()._messaging_node(_leg("read_deadline_commands"))

    out = await node({"slack_oldest": "1787513601"})

    assert provider.last_history == {"channel": CHANNEL, "oldest": "1787513601",
                                     "limit": 200}
    entry = out["slack_messages"][0]
    assert entry["text"] == "FRIST 128498: 3"
    assert entry["ts"] == "1787600001.1"
    assert entry["user"] == "U07JVE4DEH1"


@pytest.mark.asyncio
async def test_an_unset_window_reads_the_last_page(provider):
    node = _runner()._messaging_node(_leg("read_deadline_commands"))
    await node({})
    assert provider.last_history["oldest"] is None


@pytest.mark.asyncio
async def test_slack_being_unreachable_does_not_stop_the_deadline_check(provider):
    from app.infrastructure.messaging import MessagingError

    provider.fail_with = MessagingError("not_in_channel", code="not_in_channel")
    node = _runner()._messaging_node(_leg("read_deadline_commands"))

    out = await node({})

    assert out == {"slack_messages": []}
    assert "__failed_step__" not in out


# ---------------------------------------------------------------------------
# Leg 2: confirmations, exactly once
# ---------------------------------------------------------------------------

def _parsed() -> dict:
    """The shape csm-deadline-parse-overrides emits."""
    return {
        "overrides_parsed": {
            "confirmations": [
                {"thread_ts": "1787600001.1",
                 "text": "Frist fuer Projekt 128498 uebernommen: 3 Arbeitstage"},
                {"thread_ts": "1787600002.2",
                 "text": "Frist fuer Projekt 133170 uebernommen: fix 2026-08-15"},
            ]
        }
    }


@pytest.mark.asyncio
async def test_each_accepted_override_is_confirmed_in_its_own_thread(provider):
    node = _runner()._messaging_node(_leg("confirm_overrides"))

    out = await node(_parsed())

    assert [p["thread_id"] for p in provider.posted] == ["1787600001.1",
                                                        "1787600002.2"]
    assert out["confirmations_posted"]["posted_count"] == 2


@pytest.mark.asyncio
async def test_an_already_confirmed_override_is_not_confirmed_again(provider):
    """Tomorrow's run sees the same message again — the window overlaps by
    design.  The thread, not a stored marker, is the source of truth."""
    provider.threads["1787600001.1"] = [
        SlackMessage(id="1787600001.1", channel=CHANNEL, author="U07JVE4DEH1",
                     text="FRIST 128498: 3"),
        SlackMessage(id="1787600055.5", channel=CHANNEL, author="U0ATR9G06MA",
                     thread_id="1787600001.1",
                     text="Frist fuer Projekt 128498 uebernommen: 3 Arbeitstage"),
    ]
    node = _runner()._messaging_node(_leg("confirm_overrides"))

    out = await node(_parsed())

    assert [p["thread_id"] for p in provider.posted] == ["1787600002.2"]
    assert out["confirmations_posted"]["skipped"] == [
        {"thread_id": "1787600001.1", "reason": "already_replied"}
    ]


@pytest.mark.asyncio
async def test_no_accepted_override_means_no_reply_at_all(provider):
    node = _runner()._messaging_node(_leg("confirm_overrides"))
    out = await node({"overrides_parsed": {"confirmations": []}})
    assert provider.posted == []
    assert out["confirmations_posted"]["posted_count"] == 0


# ---------------------------------------------------------------------------
# Leg 3: the failure path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_usable_export_routes_to_the_channel(provider):
    router = _runner()._make_router_fn("gate_export", _leg("gate_export")["routes"])
    assert await router({"export_ok": True}) == "post_digest"


@pytest.mark.asyncio
async def test_an_unusable_export_routes_to_the_dm_and_not_the_channel(provider):
    router = _runner()._make_router_fn("gate_export", _leg("gate_export")["routes"])
    assert await router({"export_ok": False}) == "dm_owner"


@pytest.mark.asyncio
async def test_a_missing_export_ok_is_treated_as_unusable(provider):
    """Absent is not "fine" — the compute step failing to say so must land on
    the DM, never on an all-clear in the channel."""
    router = _runner()._make_router_fn("gate_export", _leg("gate_export")["routes"])
    assert await router({}) == "dm_owner"


def test_the_channel_post_is_only_reachable_through_the_gate():
    """Structural, not behavioural: if some later edit gave post_digest another
    inbound edge, a bad export could reach the channel again."""
    inbound = [
        s["id"] for s in CSM_SLACK_LEGS
        if s.get("next") == "post_digest"
        or any(r.get("next") == "post_digest" for r in s.get("routes", []))
    ]
    assert inbound == ["gate_export"]


@pytest.mark.asyncio
async def test_the_dm_leg_dms_the_owner(provider):
    node = _runner()._messaging_node(_leg("dm_owner"))

    out = await node({"fallback_owner_slack_id": "U07JVE4DEH1",
                      "board_warning": "Export ist 3 Arbeitstage alt"})

    assert provider.opened == ["U07JVE4DEH1"]
    assert provider.posted[0]["channel"] == "D-U07JVE4DEH1"
    assert "Export ist 3 Arbeitstage alt" in provider.posted[0]["text"]
    assert out["digest_dm"]["user_id"] == "U07JVE4DEH1"


@pytest.mark.asyncio
async def test_a_dm_that_cannot_be_delivered_fails_the_run(provider):
    """Not ignore_errors: silence is the failure mode this leg exists to
    prevent, so it has to be visible as a failed run."""
    from app.infrastructure.messaging import MessagingError

    provider.fail_with = MessagingError(
        "Slack conversations.open returned error 'missing_scope'",
        code="missing_scope",
    )
    node = _runner()._messaging_node(_leg("dm_owner"))

    out = await node({"fallback_owner_slack_id": "U07JVE4DEH1"})

    assert out["__failed_step__"] == "dm_owner"
    assert provider.posted == []
