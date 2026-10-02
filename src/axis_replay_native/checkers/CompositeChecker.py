"""CompositeChecker: combines multiple checkers with AND/OR logic."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload

logger = logging.getLogger("app.checkers")


class CompositeChecker(BaseChecker):
    """Combine sub-checkers with AND or OR operator."""

    def __init__(self, config: dict, checkers: list[BaseChecker] | None = None) -> None:
        super().__init__(config)
        self.operator: str = config.get("operator", "AND").upper()
        self.checkers: list[BaseChecker] = checkers or []

    def check(self, payload: CheckerRuntimePayload) -> bool:
        if not self.checkers:
            return False

        results: list[bool] = []
        for checker in self.checkers:
            try:
                results.append(checker.check(payload))
            except Exception:
                logger.exception("CompositeChecker: sub-checker error")
                results.append(False)

        if self.operator == "OR":
            return any(results)
        # Default to AND
        return all(results)

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        if not self.checkers:
            return False, {"checker_type": "CompositeChecker", "passed": False, "operator": self.operator, "sub_results": []}

        sub_results: list[dict[str, Any]] = []
        passed_list: list[bool] = []
        for checker in self.checkers:
            try:
                sub_passed, sub_detail = checker.check_detail(payload)
                passed_list.append(sub_passed)
                sub_results.append(sub_detail)
            except Exception:
                logger.exception("CompositeChecker: sub-checker error")
                passed_list.append(False)
                sub_results.append({"checker_type": type(checker).__name__, "passed": False})

        if self.operator == "OR":
            overall = any(passed_list)
        else:
            overall = all(passed_list)

        return overall, {
            "checker_type": "CompositeChecker",
            "passed": overall,
            "operator": self.operator,
            "sub_results": sub_results,
        }
