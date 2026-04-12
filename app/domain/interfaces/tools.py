from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExternalTool(ABC):
    name: str

    @abstractmethod
    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
