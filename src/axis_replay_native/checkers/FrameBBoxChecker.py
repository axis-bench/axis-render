"""FrameBBoxChecker: check if object is inside a bbox in a frame's local coords."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import (
    cfg,
    mat3_transpose_mul_vec,
    resolve_position,
    resolve_rot_mat,
    resolve_site_position,
    vec3_sub,
)

logger = logging.getLogger("app.checkers")


class FrameBBoxChecker(BaseChecker):
    """Check if object is inside a bounding box in a frame's local coords."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.obj_name: str = cfg(config, "objName", "obj_name", default="")
        self.frame_name: str = cfg(config, "frameName", "frame_name", default="")
        self.frame_type: str = cfg(config, "frameType", "frame_type", default="site")
        self.lower: list[float] = cfg(config, "lower", default=[0.0, 0.0, 0.0])
        self.upper: list[float] = cfg(config, "upper", default=[0.0, 0.0, 0.0])

    def check(self, payload: CheckerRuntimePayload) -> bool:
        obj_pos = resolve_position(self.obj_name, payload.object_positions)
        if obj_pos is None:
            logger.warning("FrameBBoxChecker: obj position not found: %s", self.obj_name)
            return False

        if self.frame_type == "site":
            if payload.site_positions is None or payload.site_rot_mats is None:
                logger.warning("FrameBBoxChecker: site_positions/site_rot_mats not provided")
                return False
            frame_pos = resolve_site_position(self.frame_name, payload.site_positions)
            frame_rot = resolve_rot_mat(self.frame_name, payload.site_rot_mats)
        else:
            frame_pos = resolve_position(self.frame_name, payload.object_positions)
            if payload.body_rot_mats is None:
                logger.warning("FrameBBoxChecker: body_rot_mats not provided")
                return False
            frame_rot = resolve_rot_mat(self.frame_name, payload.body_rot_mats)

        if frame_pos is None or frame_rot is None:
            logger.warning("FrameBBoxChecker: frame pos/rot not found: %s", self.frame_name)
            return False

        rel = vec3_sub(obj_pos, frame_pos)
        local = mat3_transpose_mul_vec(frame_rot, rel)

        passed = all(
            self.lower[i] <= local[i] <= self.upper[i]
            for i in range(3)
        )
        if not passed:
            axes = ['x', 'y', 'z']
            details = "  ".join(
                f"{axes[i]}: local={local[i]:+.5f} bounds=[{self.lower[i]:.4f}, {self.upper[i]:.4f}]"
                f"{'  ** OUT' if not (self.lower[i] <= local[i] <= self.upper[i]) else ''}"
                for i in range(3)
            )
            logger.info(
                "FrameBBoxChecker FAIL: obj=%s frame=%s  %s",
                self.obj_name, self.frame_name, details,
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "FrameBBoxChecker", "passed": False}
        obj_pos = resolve_position(self.obj_name, payload.object_positions)
        if obj_pos is None:
            detail.update(obj_name=self.obj_name, frame_name=self.frame_name)
            return False, detail

        if self.frame_type == "site":
            if payload.site_positions is None or payload.site_rot_mats is None:
                detail.update(obj_name=self.obj_name, frame_name=self.frame_name)
                return False, detail
            frame_pos = resolve_site_position(self.frame_name, payload.site_positions)
            frame_rot = resolve_rot_mat(self.frame_name, payload.site_rot_mats)
        else:
            frame_pos = resolve_position(self.frame_name, payload.object_positions)
            if payload.body_rot_mats is None:
                detail.update(obj_name=self.obj_name, frame_name=self.frame_name)
                return False, detail
            frame_rot = resolve_rot_mat(self.frame_name, payload.body_rot_mats)

        if frame_pos is None or frame_rot is None:
            detail.update(obj_name=self.obj_name, frame_name=self.frame_name)
            return False, detail

        rel = vec3_sub(obj_pos, frame_pos)
        local = mat3_transpose_mul_vec(frame_rot, rel)

        axes_labels = ["x", "y", "z"]
        axes_in_range = {
            axes_labels[i]: self.lower[i] <= local[i] <= self.upper[i]
            for i in range(3)
        }
        passed = all(axes_in_range.values())
        detail.update(
            passed=passed,
            obj_name=self.obj_name,
            frame_name=self.frame_name,
            local_coords=list(local),
            lower=list(self.lower),
            upper=list(self.upper),
            axes_in_range=axes_in_range,
        )
        return passed, detail
