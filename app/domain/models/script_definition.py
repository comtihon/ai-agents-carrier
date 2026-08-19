from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ScriptDefinition(BaseModel):
    """A reusable Python script stored in the script library.

    Scripts are referenced by ``python`` workflow steps through ``script_id``.
    A step may still carry inline ``code`` instead; the library is an
    alternative source, not a replacement.

    Fields
    ------
    id:
        Stable identifier.  Derived from the name when the UI creates a script
        (slugified), so "save by name" overwrites the same document.
    name:
        Human-readable label, unique within the library.
    description:
        What the script does — shown in the library list and the node picker.
    code:
        The Python source.  Executed with a ``state`` dict in scope; the script
        assigns ``output`` to return a value.
    """

    id: str
    name: str = ""
    description: str | None = None
    code: str = ""

    created_at: datetime | None = None
    updated_at: datetime | None = None

    def touch(self) -> None:
        from datetime import timezone
        self.updated_at = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = self.updated_at
