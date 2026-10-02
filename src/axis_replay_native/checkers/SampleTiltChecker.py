"""SampleTiltChecker: quaternion rotation angle check (simpler version)."""

from __future__ import annotations

import logging
import math
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, quat_inverse, quat_multiply, resolve_orientation

logger = logging.getLogger("app.checkers")


class SampleTiltChecker(BaseChecker):
    """Check rotation angle relative to initial quaternion (simpler version)."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.body_name: str = cfg(config, "sampleBodyName", "sample_body_name", default="sample/")
        self.threshold_deg: float = cfg(config, "tiltThresholdDegrees", "tilt_threshold_degrees", default=30.0)
        self.initial_rotation: list[float] | None = cfg(config, "initialRotation", "initial_rotation")

    def check(self, payload: CheckerRuntimePayload) -> bool:
        if payload.object_orientations is None:
            logger.warning("SampleTiltChecker: object_orientations not provided")
            return False

        current = resolve_orientation(self.body_name, payload.object_orientations)
        if current is None:
            logger.warning("SampleTiltChecker: orientation not found for %s", self.body_name)
            return False

        initial = self.initial_rotation
        if initial is None:
            if payload.initial_object_orientations is not None:
                initial = resolve_orientation(self.body_name, payload.initial_object_orientations)
            if initial is None:
                initial = [0.0, 0.0, 0.0, 1.0]

        diff = quat_multiply(quat_inverse(initial), current)
        w = abs(diff[3])
        w = min(1.0, max(-1.0, w))
        angle_deg = math.degrees(2.0 * math.acos(w))
        passed = angle_deg >= self.threshold_deg
        if not passed:
            logger.info(
                "SampleTiltChecker FAIL: body=%s angle=%.4f° threshold=%.4f° (expected angle >= threshold)",
                self.body_name, angle_deg, self.threshold_deg,
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "SampleTiltChecker", "passed": False, "body_name": self.body_name}
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
                initial = [0.0, 0.0, 0.0, 1.0]

        diff = quat_multiply(quat_inverse(initial), current)
        w = abs(diff[3])
        w = min(1.0, max(-1.0, w))
        angle_deg = math.degrees(2.0 * math.acos(w))
        passed = angle_deg >= self.threshold_deg
        detail.update(passed=passed, angle_deg=angle_deg, threshold_deg=self.threshold_deg)
        return passed, detail
