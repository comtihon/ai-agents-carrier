"""An agent step that declares both `output_key` and `output_mapping` must get
both into workflow state.

The two are independent: `output_key` stores the agent's whole payload,
`output_mapping` lifts individual keys out of it under names the rest of the
graph routes on. Declaring both is the natural thing to write when you want the
raw result kept *and* a couple of fields to branch on.

The state schema used to declare the mapped names only when `output_key` was
absent, so that exact combination silently lost them: LangGraph drops any key
its schema does not know, the step still reports "finished", and the run
completes with the fields simply missing. Nothing fails — which is what makes
it expensive to find. It cost a live debugging round on a real workflow.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.orchestration.yaml_graph import (
    YamlGraphRunner,
    _build_state_schema,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider


def _agent_step(**extra):
    return {
        "id": "classify",
        "type": "claude-agent",
        "agent_id": "some-agent",
        **extra,
    }


def _schema_keys(step):
    return set(_build_state_schema([step]).__annotations__)


# ---------------------------------------------------------------------------
# Schema declaration
# ---------------------------------------------------------------------------

def test_mapping_declared_when_output_key_is_also_present():
    """The regression: both declared, mapped names must survive."""
    keys = _schema_keys(_agent_step(
        output_key="classifier_raw",
        output_mapping={"pipeline": "pipeline", "confidence": "confidence"},
    ))

    assert {"pipeline", "confidence"} <= keys
    assert "classifier_raw" in keys


def test_mapping_declared_when_output_key_is_absent():
    keys = _schema_keys(_agent_step(output_mapping={"pipeline": "pipeline"}))

    assert "pipeline" in keys


def test_mapping_can_rename():
    keys = _schema_keys(_agent_step(
        output_key="raw",
        output_mapping={"pipeline": "hs_pipeline"},
    ))

    assert "hs_pipeline" in keys
    # The agent-side name is not a state key; only the workflow-side name is.
    assert "pipeline" not in keys


def test_output_key_alone_still_declared():
    assert "agent_out" in _schema_keys(_agent_step(output_key="agent_out"))


@pytest.mark.parametrize("agent_type", ["claude-agent", "langgraph-agent"])
def test_both_agent_types_get_the_same_treatment(agent_type):
    keys = _schema_keys({
        "id": "s", "type": agent_type, "agent_id": "a",
        "output_key": "raw", "output_mapping": {"verdict": "verdict"},
    })

    assert "verdict" in keys


# ---------------------------------------------------------------------------
# End to end: the values actually reach state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mapped_fields_reach_state_alongside_output_key():
    definition = {
        "id": "mapping-graph",
        "steps": [_agent_step(
            output_key="classifier_raw",
            output_mapping={"pipeline": "pipeline", "confidence": "confidence"},
        )],
    }
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner(definition, llm=MagicMock(), mcp_tools_provider=mcp)
    runner._agent_backend = MagicMock()

    agent_result = {
        "classifier_raw": {"pipeline": "Finance", "confidence": 91},
        "pipeline": "Finance",
        "confidence": 91,
    }
    with patch(
        "app.steps.agent_executor.execute_agent_step",
        new=AsyncMock(return_value=agent_result),
    ):
        state = await runner.graph.ainvoke(
            {"request": "go"}, {"configurable": {"thread_id": "mapping-1"}}
        )

    assert state.get("pipeline") == "Finance"
    assert state.get("confidence") == 91
    assert state.get("classifier_raw") == {"pipeline": "Finance", "confidence": 91}


@pytest.mark.asyncio
async def test_a_route_can_branch_on_a_mapped_field_with_output_key_set():
    """Branching on a mapped field is the whole reason to declare both."""
    definition = {
        "id": "mapping-route-graph",
        "steps": [
            _agent_step(
                output_key="raw",
                output_mapping={"spam": "spam"},
                routes=[
                    {"when": "spam", "next": "drop"},
                    {"next": "keep"},
                ],
            ),
            {"id": "drop", "type": "python", "sandbox": False,
             "code": "output = 'dropped'", "output_key": "drop_out", "next": "__end__"},
            {"id": "keep", "type": "python", "sandbox": False,
             "code": "output = 'kept'", "output_key": "keep_out", "next": "__end__"},
        ],
    }
    mcp = MagicMock(spec=McpToolsProvider)
    mcp.get_tool = MagicMock(return_value=None)
    runner = YamlGraphRunner(definition, llm=MagicMock(), mcp_tools_provider=mcp)
    runner._agent_backend = MagicMock()

    with patch(
        "app.steps.agent_executor.execute_agent_step",
        new=AsyncMock(return_value={"raw": {"spam": True}, "spam": True}),
    ):
        state = await runner.graph.ainvoke(
            {"request": "go"}, {"configurable": {"thread_id": "mapping-route-1"}}
        )

    assert state.get("drop_out") == "dropped"
    assert "keep_out" not in state
