"""
Integration test: Python action handlers loaded from the workflows directory.

Flow under test
───────────────
Startup  →  ActionFileLoader scans workflows/*.py
         →  finds test_actions.py, registers "test_actions.greet" and "test_actions.echo"

POST /runs  →  action step calls "test_actions.greet" with handler_input
            →  handler returns {"message": "Hello, ...", "run_id": ...}
            →  status: completed, intermediate_outputs["greeting"] populated
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.application.services.planning_service import PlanningService
from app.domain.models.runtime import PlanResult

_EMPTY_PLAN = PlanResult(summary="no tasks", tasks=[], execution_order=[], outputs_required=[])


@pytest.mark.asyncio
async def test_python_handler_loaded_from_file_and_executed(client) -> None:
    """
    The 'test_actions.greet' handler is defined in workflows/test_actions.py.
    No app-code registration — the loader discovers and registers it at startup.
    """
    with patch.object(PlanningService, "create_plan", AsyncMock(return_value=_EMPTY_PLAN)):
        response = await client.post(
            "/api/v1/workflows/runs",
            json={"workflow_id": "python_action_flow", "user_request": "Alice"},
        )

    assert response.status_code == 201, response.text
    run = response.json()["run"]

    assert run["status"] == "completed"

    greeting = run["intermediate_outputs"]["greeting"]
    assert greeting["message"] == "Hello, Alice!"
    assert greeting["run_id"] == run["id"]

    action_results = run["action_results"]
    assert len(action_results) == 1
    assert action_results[0]["step_id"] == "run_action"
    assert action_results[0]["status"] == "success"
