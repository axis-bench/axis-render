"""BowlPositionChecker: check bowl position against bounds or distance threshold."""

from __future__ import annotations

import logging
import math
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_position

logger = logging.getLogger("app.checkers")


class BowlPositionChecker(BaseChecker):
    """Check bowl position against bounds or distance threshold."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.bowl_body_name: str = cfg(config, "bowlBodyName", "bowl_body_name", default="akita_black_bowl/")
        self.position_threshold: float | None = cfg(config, "positionThreshold", "position_threshold")
        self.min_bounds: list[float] | None = cfg(config, "minBounds", "min_bounds")
        self.max_bounds: list[float] | None = cfg(config, "maxBounds", "max_bounds")

    @staticmethod
    def _mujoco_to_threejs(pos: list[float]) -> list[float]:
        """Convert MuJoCo (Z-up) position to Three.js (Y-up) position.

        Three.js swizzle: (x, y, z) = (mj_x, mj_z, -mj_y)
        """
        return [pos[0], pos[2], -pos[1]]

    def check(self, payload: CheckerRuntimePayload) -> bool:
        pos = resolve_position(self.bowl_body_name, payload.object_positions)
        if pos is None:
            logger.warning("BowlPositionChecker: position not found for %s", self.bowl_body_name)
            return False
        # Bounds are defined in frontend Three.js coordinate system,
        # so convert MuJoCo position before comparison.
        pos = self._mujoco_to_threejs(pos)

        if self.min_bounds is not None and self.max_bounds is not None:
            passed = all(
                self.min_bounds[i] <= pos[i] <= self.max_bounds[i]
                for i in range(3)
            )
            if not passed:
                axes = ['x', 'y', 'z']
                details = "  ".join(
                    f"{axes[i]}: pos={pos[i]:+.5f} bounds=[{self.min_bounds[i]:.4f}, {self.max_bounds[i]:.4f}]"
                    f"{'  ** OUT' if not (self.min_bounds[i] <= pos[i] <= self.max_bounds[i]) else ''}"
                    for i in range(3)
                )
                logger.info("BowlPositionChecker FAIL: bowl=%s  %s", self.bowl_body_name, details)
            return passed

        if self.position_threshold is not None:
            dist = math.sqrt(sum(c * c for c in pos))
            passed = dist <= self.position_threshold
            if not passed:
                logger.info(
                    "BowlPositionChecker FAIL: bowl=%s dist=%.5f (threshold=%.5f)",
                    self.bowl_body_name, dist, self.position_threshold,
                )
            return passed

        return True

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "BowlPositionChecker", "passed": False}
        pos = resolve_position(self.bowl_body_name, payload.object_positions)
        if pos is None:
            detail["body_name"] = self.bowl_body_name
            return False, detail

        pos = self._mujoco_to_threejs(pos)
        detail["body_name"] = self.bowl_body_name
        detail["position"] = list(pos)

        if self.min_bounds is not None and self.max_bounds is not None:
            axes_labels = ["x", "y", "z"]
            axes_in_range = {
                axes_labels[i]: self.min_bounds[i] <= pos[i] <= self.max_bounds[i]
                for i in range(3)
            }
            passed = all(axes_in_range.values())
            detail.update(
                passed=passed,
                min_bounds=list(self.min_bounds),
                max_bounds=list(self.max_bounds),
                axes_in_range=axes_in_range,
            )
            return passed, detail

        if self.position_threshold is not None:
            dist = math.sqrt(sum(c * c for c in pos))
            passed = dist <= self.position_threshold
            detail.update(
                passed=passed,
                distance=dist,
                position_threshold=self.position_threshold,
            )
            return passed, detail

        detail["passed"] = True
        return True, detail
