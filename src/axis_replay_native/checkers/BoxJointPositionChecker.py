"""BoxJointPositionChecker: check if box joint is below threshold (lid closed)."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_joint

logger = logging.getLogger("app.checkers")


class BoxJointPositionChecker(BaseChecker):
    """Check if box joint position is below threshold (lid closed)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.threshold: float = cfg(config, "threshold", default=-0.244346)
        self.joint_name: str = cfg(config, "jointName", "joint_name", default="box_base/box_joint")

    def check(self, payload: CheckerRuntimePayload) -> bool:
        if payload.joint_positions is None:
            logger.warning("BoxJointPositionChecker: joint_positions not provided")
            return False

        val = resolve_joint(self.joint_name, payload.joint_positions)
        if val is None:
            logger.warning("BoxJointPositionChecker: joint not found: %s", self.joint_name)
            return False

        passed = val < self.threshold
        if not passed:
            logger.info(
                "BoxJointPositionChecker FAIL: joint=%s val=%.6f threshold=%.6f (expected val < threshold)",
                self.joint_name, val, self.threshold,
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "BoxJointPositionChecker", "passed": False}
        if payload.joint_positions is None:
            return False, detail
        val = resolve_joint(self.joint_name, payload.joint_positions)
        if val is None:
            return False, detail

        passed = val < self.threshold
        detail.update(
            passed=passed,
            joint_name=self.joint_name,
            value=val,
            threshold=self.threshold,
        )
        return passed, detail
