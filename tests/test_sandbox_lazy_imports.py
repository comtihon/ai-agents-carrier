"""Standard-library calls that import lazily must work in the sandbox.

The bootstrap pre-imports the allowed modules and then clears the import
machinery, so anything a module reaches for later cannot be loaded. That is
invisible until the call happens: `import datetime` succeeds, and then
`datetime.strptime` raises ImportError for `_strptime` the first time it is
called. Every sandboxed script parsing a date hit this.
"""
from __future__ import annotations

import pytest

from app.infrastructure.orchestration.script_sandbox import run_script


@pytest.mark.asyncio
async def test_strptime_works():
    code = """
from datetime import datetime
dt = datetime.strptime(state["s"], "%a %Y-%m-%d %H:%M:%S")
output = {"iso": dt.isoformat(), "hour": dt.hour}
"""
    result = await run_script(
        code, {"s": "Thu 2026-07-30 10:12:44"}, runtime="local", timeout=30
    )

    assert result == {"iso": "2026-07-30T10:12:44", "hour": 10}


@pytest.mark.asyncio
async def test_strftime_and_isoformat_round_trip():
    code = """
from datetime import date, datetime
d = date.fromisoformat(state["d"])
output = {
    "fmt": d.strftime("%d.%m."),
    "parsed": datetime.strptime("2026-08-03", "%Y-%m-%d").date().isoformat(),
}
"""
    result = await run_script(code, {"d": "2026-08-04"}, runtime="local", timeout=30)

    assert result == {"fmt": "04.08.", "parsed": "2026-08-03"}


@pytest.mark.asyncio
async def test_decimal_still_works():
    """decimal also has a private fallback module; check it did not regress."""
    code = """
from decimal import Decimal
output = str(Decimal("1.10") + Decimal("2.20"))
"""
    assert await run_script(code, {}, runtime="local", timeout=30) == "3.30"
