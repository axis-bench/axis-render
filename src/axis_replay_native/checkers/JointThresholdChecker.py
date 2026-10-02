"""JointThresholdChecker: generic joint comparison with configurable mode."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_joint

logger = logging.getLogger("app.checkers")


class JointThresholdChecker(BaseChecker):
    """Generic joint comparison with configurable mode (gt/ge/lt/le)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        obj_name: str = cfg(config, "objName", "obj_name", default="")
        joint_name: str = cfg(config, "jointName", "joint_name", default="")
        if joint_name and "/" not in joint_name and obj_name:
            joint_name = f"{obj_name}/{joint_name}"
        self.joint_name: str = joint_name
        self.joint_index: int | None = cfg(config, "jointIndex", "joint_index")
        self.mode: str = cfg(config, "mode", default="ge")
        self.threshold: float = cfg(config, "threshold", default=0.0)

    def check(self, payload: CheckerRuntimePayload) -> bool:
        if payload.joint_positions is None:
            logger.warning("JointThresholdChecker: joint_positions not provided")
            return False

        val = resolve_joint(self.joint_name, payload.joint_positions)
        if val is None:
            logger.warning("JointThresholdChecker: joint not found: %s", self.joint_name)
            return False

        passed = False
        if self.mode == "gt":
            passed = val > self.threshold
        elif self.mode == "ge":
            passed = val >= self.threshold
        elif self.mode == "lt":
            passed = val < self.threshold
        elif self.mode == "le":
            passed = val <= self.threshold
        else:
            logger.warning("JointThresholdChecker: unknown mode %s", self.mode)
            return False

        if not passed:
            logger.info(
                "JointThresholdChecker FAIL: joint=%s val=%.6f %s threshold=%.6f",
                self.joint_name, val, self.mode, self.threshold,
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "JointThresholdChecker", "passed": False}
        if payload.joint_positions is None:
            return False, detail
        val = resolve_joint(self.joint_name, payload.joint_positions)
        if val is None:
            return False, detail

        passed = False
        if self.mode == "gt":
            passed = val > self.threshold
        elif self.mode == "ge":
            passed = val >= self.threshold
        elif self.mode == "lt":
            passed = val < self.threshold
        elif self.mode == "le":
            passed = val <= self.threshold

        detail.update(
            passed=passed,
            joint_name=self.joint_name,
            value=val,
            mode=self.mode,
            threshold=self.threshold,
        )
        return passed, detail
