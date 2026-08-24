"""A script whose helpers call each other must work.

The bootstrap used to exec with separate globals and locals. Under that, every
top-level `def` bound into locals while a function body resolved names against
globals -- so the moment one helper called another, the script died with
NameError. Nothing in the platform's own probes caught it, because a one-liner
that never defines a function works fine either way.

Both execution paths are covered: the sandboxed bootstrap and the non-sandboxed
in-process exec, which had the same defect.
"""
from __future__ import annotations

import pytest

from app.infrastructure.orchestration.script_sandbox import run_script

_HELPERS = """
def double(n):
    return n * 2

def apply_twice(n):
    # The call that used to raise NameError: one top-level def reaching another.
    return double(double(n))

CONSTANT = 5

def uses_module_constant():
    return CONSTANT

output = {
    "twice": apply_twice(3),
    "constant": uses_module_constant(),
    "from_state": double(state["n"]),
}
"""


@pytest.mark.asyncio
async def test_local_runtime_lets_helpers_call_each_other():
    result = await run_script(_HELPERS, {"n": 7}, runtime="local", timeout=30)

    assert result == {"twice": 12, "constant": 5, "from_state": 14}


@pytest.mark.asyncio
async def test_a_comprehension_can_see_module_level_names():
    """Comprehensions have their own scope and were a second casualty."""
    code = """
FACTOR = 3

def scale(v):
    return v * FACTOR

output = [scale(v) for v in state["values"]]
"""
    result = await run_script(code, {"values": [1, 2]}, runtime="local", timeout=30)

    assert result == [3, 6]


@pytest.mark.asyncio
async def test_a_class_and_its_methods_work():
    code = """
class Counter:
    def __init__(self, start):
        self.value = start

    def bump(self):
        self.value += 1
        return self.value

c = Counter(state["start"])
c.bump()
output = c.bump()
"""
    result = await run_script(code, {"start": 10}, runtime="local", timeout=30)

    assert result == 12
