"""Base class for all backend checkers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .runtime_types import CheckerRuntimePayload

logger = logging.getLogger("app.checkers")


class BaseChecker(ABC):
    """Every checker receives its config dict at construction time and
    implements ``check(payload) -> bool``."""

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def check(self, payload: CheckerRuntimePayload) -> bool: ...

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        """Return (passed, detail_dict). Subclasses override for richer info."""
        passed = self.check(payload)
        return passed, {"checker_type": type(self).__name__, "passed": passed}


class AlwaysFalseChecker(BaseChecker):
    """Fallback for unknown / unsupported checker types."""

    def __init__(self, config: dict, reason: str = "") -> None:
        super().__init__(config)
        self.reason = reason

    def check(self, payload: CheckerRuntimePayload) -> bool:
        logger.warning("AlwaysFalseChecker triggered: %s", self.reason)
        return False
