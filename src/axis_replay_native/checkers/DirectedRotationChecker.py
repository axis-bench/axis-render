"""DirectedRotationChecker: rotate body by ~targetAngleDeg around any axis
while keeping tilt below maxTiltDeg.

  - rotation_angle = total quaternion angle diff from initial (any axis)
  - tilt = angle between the body's up-axis in initial vs current orientation
    (i.e., how far the object's local +Z has swung from where it started)

Success iff |rotation_angle - targetAngleDeg| <= angleToleranceDeg AND
tilt <= maxTiltDeg. This matches the pattern seen in "Rotate the frying pan"
tasks (targetAngleDeg=90, maxTiltDeg=15): 90 degrees of yaw while staying
mostly upright.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import (
    cfg,
    quat_angle_deg,
    quat_rotate_vec,
    resolve_orientation,
    vec3_dot,
    vec3_length,
)

logger = logging.getLogger("app.checkers")

_BODY_UP = [0.0, 0.0, 1.0]


def _tilt_deg(q_init: list[float], q_curr: list[float]) -> float:
    """Angle in degrees between the body's up axis in initial vs current
    orientation, measured in world frame."""
    up_init = quat_rotate_vec(q_init, _BODY_UP)
    up_curr = quat_rotate_vec(q_curr, _BODY_UP)
    n_init = vec3_length(up_init)
    n_curr = vec3_length(up_curr)
    if n_init < 1e-9 or n_curr < 1e-9:
        return 0.0
    cos = vec3_dot(up_init, up_curr) / (n_init * n_curr)
    cos = min(1.0, max(-1.0, cos))
    return math.degrees(math.acos(cos))


class DirectedRotationChecker(BaseChecker):
    """Check if a body has rotated by (targetAngleDeg ± angleToleranceDeg)
    while its tilt from initial-up stays under maxTiltDeg.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.body_name: str = cfg(
            config, "bodyName", "body_name", default="sample/"
        )
        self.target_angle_deg: float = float(cfg(
            config, "targetAngleDeg", "target_angle_deg", default=90.0
        ))
        self.angle_tol_deg: float = float(cfg(
            config, "angleToleranceDeg", "angle_tolerance_deg", default=10.0
        ))
        self.max_tilt_deg: float = float(cfg(
            config, "maxTiltDeg", "max_tilt_deg", default=15.0
        ))
        self.initial_rotation: list[float] | None = cfg(
            config, "initialRotation", "initial_rotation"
        )

    def _resolve(self, payload: CheckerRuntimePayload
                 ) -> tuple[list[float], list[float]] | None:
        if payload.object_orientations is None:
            return None
        current = resolve_orientation(self.body_name, payload.object_orientations)
        if current is None:
            return None
        initial = self.initial_rotation
        if initial is None and payload.initial_object_orientations is not None:
            initial = resolve_orientation(
                self.body_name, payload.initial_object_orientations)
        if initial is None:
            return None
        return initial, current

    def check(self, payload: CheckerRuntimePayload) -> bool:
        got = self._resolve(payload)
        if got is None:
            logger.warning(
                "DirectedRotationChecker: orientation not resolvable for %s",
                self.body_name)
            return False
        initial, current = got
        angle = quat_angle_deg(initial, current)
        tilt = _tilt_deg(initial, current)
        passed = (abs(angle - self.target_angle_deg) <= self.angle_tol_deg
                  and tilt <= self.max_tilt_deg)
        if not passed:
            logger.info(
                "DirectedRotationChecker FAIL: body=%s angle=%.2f° "
                "(target=%.2f±%.2f) tilt=%.2f° (max=%.2f)",
                self.body_name, angle, self.target_angle_deg,
                self.angle_tol_deg, tilt, self.max_tilt_deg,
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload
                     ) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {
            "checker_type": "DirectedRotationChecker",
            "passed": False,
            "body_name": self.body_name,
            "target_angle_deg": self.target_angle_deg,
            "angle_tol_deg": self.angle_tol_deg,
            "max_tilt_deg": self.max_tilt_deg,
        }
        got = self._resolve(payload)
        if got is None:
            return False, detail
        initial, current = got
        angle = quat_angle_deg(initial, current)
        tilt = _tilt_deg(initial, current)
        passed = (abs(angle - self.target_angle_deg) <= self.angle_tol_deg
                  and tilt <= self.max_tilt_deg)
        detail.update(passed=passed, angle_deg=angle, tilt_deg=tilt)
        return passed, detail
