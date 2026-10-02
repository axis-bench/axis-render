"""SampleRotationChecker: tilt/rotation check with optional directional tilt."""

from __future__ import annotations

import logging
import math
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import (
    cfg,
    quat_angle_deg,
    quat_normalize,
    quat_rotate_vec,
    resolve_orientation,
    vec3_dot,
    vec3_length,
    vec3_scale,
    vec3_sub,
)

logger = logging.getLogger("app.checkers")


class SampleRotationChecker(BaseChecker):
    """Check sample tilt/rotation with optional directional tilt."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.body_name: str = cfg(config, "sampleBodyName", "sample_body_name", default="sample/")
        self.threshold_deg: float = cfg(config, "tipAngleThreshold", "tip_angle_threshold", default=30.0)
        self.tilt_only: bool = cfg(config, "tiltOnly", "tilt_only", default=True)
        self.tilt_world_direction: list[float] | None = cfg(
            config, "tiltWorldDirection", "tilt_world_direction"
        )
        self.initial_rotation: list[float] | None = cfg(config, "initialRotation", "initial_rotation")

    def check(self, payload: CheckerRuntimePayload) -> bool:
        if payload.object_orientations is None:
            logger.warning("SampleRotationChecker: object_orientations not provided")
            return False

        current_quat = resolve_orientation(self.body_name, payload.object_orientations)
        if current_quat is None:
            logger.warning("SampleRotationChecker: orientation not found for %s", self.body_name)
            return False

        initial_quat = self.initial_rotation
        if initial_quat is None:
            if payload.initial_object_orientations is not None:
                initial_quat = resolve_orientation(self.body_name, payload.initial_object_orientations)
            if initial_quat is None:
                initial_quat = [0.0, 0.0, 0.0, 1.0]

        current_quat = quat_normalize(current_quat)
        initial_quat = quat_normalize(initial_quat)

        if self.tilt_only:
            tip_angle = self._compute_tilt(current_quat, initial_quat)
        else:
            tip_angle = quat_angle_deg(initial_quat, current_quat)

        passed = tip_angle >= self.threshold_deg
        if not passed:
            logger.info(
                "SampleRotationChecker FAIL: body=%s tilt_only=%s angle=%.4f° threshold=%.4f° (expected angle >= threshold)",
                self.body_name, self.tilt_only, tip_angle, self.threshold_deg,
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "SampleRotationChecker", "passed": False, "body_name": self.body_name}
        if payload.object_orientations is None:
            return False, detail
        current_quat = resolve_orientation(self.body_name, payload.object_orientations)
        if current_quat is None:
            return False, detail

        initial_quat = self.initial_rotation
        if initial_quat is None:
            if payload.initial_object_orientations is not None:
                initial_quat = resolve_orientation(self.body_name, payload.initial_object_orientations)
            if initial_quat is None:
                initial_quat = [0.0, 0.0, 0.0, 1.0]

        current_quat = quat_normalize(current_quat)
        initial_quat = quat_normalize(initial_quat)

        if self.tilt_only:
            tip_angle = self._compute_tilt(current_quat, initial_quat)
        else:
            tip_angle = quat_angle_deg(initial_quat, current_quat)

        passed = tip_angle >= self.threshold_deg
        detail.update(passed=passed, tilt_only=self.tilt_only, angle_deg=tip_angle, threshold_deg=self.threshold_deg)
        return passed, detail

    def _compute_tilt(self, current_q: list[float], initial_q: list[float]) -> float:
        """Compute tilt angle (ignoring yaw) between current and initial."""
        up = [0.0, 0.0, 1.0]
        current_up = quat_rotate_vec(current_q, up)
        initial_up = quat_rotate_vec(initial_q, up)

        if self.tilt_world_direction is not None:
            direction = list(self.tilt_world_direction)
            proj = vec3_dot(initial_up, direction)
            dir_in_plane = vec3_sub(direction, vec3_scale(initial_up, proj))
            length = vec3_length(dir_in_plane)

            if length < 1e-8:
                dot = max(-1.0, min(1.0, vec3_dot(current_up, initial_up)))
                return math.degrees(math.acos(dot))
            else:
                dir_in_plane = vec3_scale(dir_in_plane, 1.0 / length)
                up_dot = max(-1.0, min(1.0, vec3_dot(current_up, initial_up)))
                dir_dot = vec3_dot(current_up, dir_in_plane)
                angle_rad = math.atan2(dir_dot, up_dot)
                return math.degrees(angle_rad)
        else:
            dot = max(-1.0, min(1.0, vec3_dot(current_up, initial_up)))
            return math.degrees(math.acos(dot))
