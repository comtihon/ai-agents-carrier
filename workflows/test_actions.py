"""
Test Python action handlers — loaded automatically from the workflows directory.

This file is used by the integration test for the Python action loader.
It demonstrates the convention: define an ACTIONS dict mapping handler name
to an async callable.  No imports from the application are required.
"""
from __future__ import annotations


async def greet(handler_input: dict, run) -> dict:
    name = handler_input.get("name", "world")
    return {"message": f"Hello, {name}!", "run_id": run.id}


async def echo(handler_input: dict, run) -> dict:
    return {"echoed": handler_input}


ACTIONS = {
    "test_actions.greet": greet,
    "test_actions.echo": echo,
}
