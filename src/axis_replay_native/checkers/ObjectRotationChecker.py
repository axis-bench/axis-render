"""ObjectRotationChecker: quaternion angle diff check."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, quat_angle_deg, resolve_orientation

logger = logging.getLogger("app.checkers")


class ObjectRotationChecker(BaseChecker):
    """Check if object has rotated beyond threshold from initial rotation."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.body_name: str = cfg(config, "bodyName", "body_name", default="sample/")
        self.threshold_deg: float = cfg(config, "rotationThresholdDeg", "rotation_threshold_deg", default=30.0)
        self.initial_rotation: list[float] | None = cfg(config, "initialRotation", "initial_rotation")

    def check(self, payload: CheckerRuntimePayload) -> bool:
        if payload.object_orientations is None:
            logger.warning("ObjectRotationChecker: object_orientations not provided")
            return False

        current = resolve_orientation(self.body_name, payload.object_orientations)
        if current is None:
            logger.warning("ObjectRotationChecker: orientation not found for %s", self.body_name)
            return False

        initial = self.initial_rotation
        if initial is None:
            if payload.initial_object_orientations is not None:
                initial = resolve_orientation(self.body_name, payload.initial_object_orientations)
            if initial is None:
                logger.warning("ObjectRotationChecker: no initial rotation available")
                return False

        angle = quat_angle_deg(initial, current)
        passed = angle >= self.threshold_deg
        if not passed:
            logger.info(
                "ObjectRotationChecker FAIL: body=%s angle=%.4f° threshold=%.4f° (expected angle >= threshold)",
                self.body_name, angle, self.threshold_deg,
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "ObjectRotationChecker", "passed": False, "body_name": self.body_name}
        if payload.object_orientations is None:
            return False, detail
        current = resolve_orientation(self.body_name, payload.object_orientations)
        if current is None:
            return False, detail

        initial = self.initial_rotation
        if initial is None:
            if payload.initial_object_orientations is not None:
                initial = resolve_orientation(self.body_name, payload.initial_object_orientations)
            if initial is None:
                return False, detail

        angle = quat_angle_deg(initial, current)
        passed = angle >= self.threshold_deg
        detail.update(passed=passed, angle_deg=angle, threshold_deg=self.threshold_deg)
        return passed, detail
