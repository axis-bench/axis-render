"""GripperOpenChecker: check if any finger joint exceeds open threshold."""

from __future__ import annotations

import logging
from math import isclose
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_joint

logger = logging.getLogger("app.checkers")

_FULL_OPEN_THRESHOLD = 0.04
_COMPAT_THRESHOLD = 0.02


class GripperOpenChecker(BaseChecker):
    """Check if any finger joint exceeds open threshold."""

    _DEFAULT_FINGER_JOINT_NAMES = [
        "franka/panda_finger_joint1",
        "franka/panda_finger_joint2",
    ]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        threshold = float(cfg(config, "threshold", default=0.2))
        # Backend normalizes the Franka "fully open" threshold because this checker
        # uses a strict ">" comparison. Mirror that behavior for parity.
        if isclose(threshold, _FULL_OPEN_THRESHOLD, rel_tol=0.0, abs_tol=1e-9):
            logger.info(
                "GripperOpenChecker threshold %.6f normalized to %.6f for compatibility",
                threshold,
                _COMPAT_THRESHOLD,
            )
            threshold = _COMPAT_THRESHOLD
        self.threshold = threshold
        self.finger_joint_names: list[str] = cfg(
            config, "fingerJointNames", "finger_joint_names",
            "gripperJointNames", "gripper_joint_names",
            default=self._DEFAULT_FINGER_JOINT_NAMES,
        )

    def check(self, payload: CheckerRuntimePayload) -> bool:
        if payload.joint_positions is None or not self.finger_joint_names:
            logger.warning("GripperOpenChecker: joint_positions or finger names missing")
            return False

        joint_vals = {}
        for name in self.finger_joint_names:
            val = resolve_joint(name, payload.joint_positions)
            if val is not None:
                joint_vals[name] = val
                if val > self.threshold:
                    return True

        details = "  ".join(
            f"{name}={val:.6f}{'  ** OUT' if val <= self.threshold else ''}"
            for name, val in joint_vals.items()
        )
        logger.info(
            "GripperOpenChecker FAIL: none above threshold=%.6f  %s",
            self.threshold, details,
        )
        return False

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "GripperOpenChecker", "passed": False}
        if payload.joint_positions is None or not self.finger_joint_names:
            detail["threshold"] = self.threshold
            detail["joint_values"] = {}
            return False, detail

        joint_vals: dict[str, float] = {}
        passed = False
        for name in self.finger_joint_names:
            val = resolve_joint(name, payload.joint_positions)
            if val is not None:
                joint_vals[name] = val
                if val > self.threshold:
                    passed = True

        detail.update(passed=passed, threshold=self.threshold, joint_values=joint_vals)
        return passed, detail
