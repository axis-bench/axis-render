"""CheckerManager: builds a checker tree from task checker_config."""

from __future__ import annotations

import logging
from typing import Any

from .base import AlwaysFalseChecker, BaseChecker
from .CompositeChecker import CompositeChecker
from .registry import get_checker_class

logger = logging.getLogger("app.checkers")


class CheckerManager:
    """Build and evaluate a checker tree from a task's ``checker_config``."""

    def __init__(self, checker_config: dict | None) -> None:
        self.checker: BaseChecker | None = None
        if checker_config:
            self._build(checker_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, payload) -> bool:
        if self.checker is None:
            return False
        return self.checker.check(payload)

    def check_with_details(self, payload) -> tuple[bool, list[dict[str, Any]]]:
        """Run checker and return detailed results."""
        if self.checker is None:
            return False, []
        passed, detail = self.checker.check_detail(payload)
        # If top-level is CompositeChecker, flatten sub_results
        if detail.get("sub_results"):
            return passed, detail["sub_results"]
        return passed, [detail]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self, checker_config: dict) -> None:
        try:
            if "checker" in checker_config:
                self.checker = self._create_checker(checker_config["checker"])
            else:
                # Legacy format: each key is a checker type in camelCase
                self.checker = self._build_legacy(checker_config)
        except Exception:
            logger.exception("CheckerManager: failed to build checker tree")
            self.checker = None

    def _create_checker(self, config: dict) -> BaseChecker:
        checker_type = config.get("type", "")

        # CompositeChecker
        if checker_type in ("CompositeChecker", "composite"):
            sub_configs = config.get("checkers", [])
            if not sub_configs:
                return AlwaysFalseChecker(config, reason="CompositeChecker with no sub-checkers")
            sub_checkers = [self._create_checker(sc) for sc in sub_configs]
            return CompositeChecker(config, checkers=sub_checkers)

        # Single checker
        cls = get_checker_class(checker_type)
        if cls is None:
            logger.warning("Unknown checker type: %s", checker_type)
            return AlwaysFalseChecker(config, reason=f"Unknown type: {checker_type}")

        return cls(config)

    def _build_legacy(self, checker_config: dict) -> BaseChecker:
        """Legacy format: keys like 'gripperOpenChecker' -> GripperOpenChecker."""
        checkers: list[BaseChecker] = []
        for key, config in checker_config.items():
            if not isinstance(config, dict):
                continue
            if config.get("enabled") is False:
                continue
            # camelCase key -> PascalCase type name
            type_name = key[0].upper() + key[1:]
            cls = get_checker_class(type_name)
            if cls is None:
                logger.warning("Legacy: unknown checker type from key '%s' -> '%s'", key, type_name)
                checkers.append(AlwaysFalseChecker(config, reason=f"Unknown legacy key: {key}"))
                continue
            checkers.append(cls(config))

        if not checkers:
            return AlwaysFalseChecker({}, reason="Legacy config produced no checkers")

        if len(checkers) == 1:
            return checkers[0]

        # Multiple legacy checkers -> AND composite (mirrors frontend)
        return CompositeChecker({"operator": "AND"}, checkers=checkers)
