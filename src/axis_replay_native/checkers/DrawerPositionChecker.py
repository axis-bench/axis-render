"""DrawerPositionChecker: check if any drawer joint is below threshold."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_joint

logger = logging.getLogger("app.checkers")


class DrawerPositionChecker(BaseChecker):
    """Check if any drawer joint position is below threshold (drawer opened)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.threshold: float = cfg(config, "threshold", default=-0.1)
        self.drawer_joint_names: list[str] = cfg(
            config, "drawerJointNames", "drawer_joint_names",
            default=[
                "wooden_cabinet/top_level",
                "wooden_cabinet/middle_level",
                "wooden_cabinet/bottom_level",
            ],
        )

    def check(self, payload: CheckerRuntimePayload) -> bool:
        if payload.joint_positions is None:
            logger.warning("DrawerPositionChecker: joint_positions not provided")
            return False

        joint_vals = {}
        for name in self.drawer_joint_names:
            val = resolve_joint(name, payload.joint_positions)
            if val is not None:
                joint_vals[name] = val
                if val < self.threshold:
                    return True

        details = "  ".join(
            f"{name}={val:.6f}{'  ** OUT' if val >= self.threshold else ''}"
            for name, val in joint_vals.items()
        )
        logger.info(
            "DrawerPositionChecker FAIL: none below threshold=%.6f  %s",
            self.threshold, details,
        )
        return False

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "DrawerPositionChecker", "passed": False}
        if payload.joint_positions is None:
            detail["threshold"] = self.threshold
            detail["joint_values"] = {}
            return False, detail

        joint_vals: dict[str, float] = {}
        passed = False
        for name in self.drawer_joint_names:
            val = resolve_joint(name, payload.joint_positions)
            if val is not None:
                joint_vals[name] = val
                if val < self.threshold:
                    passed = True

        detail.update(passed=passed, threshold=self.threshold, joint_values=joint_vals)
        return passed, detail
