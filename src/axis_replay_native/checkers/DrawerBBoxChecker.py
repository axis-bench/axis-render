"""DrawerBBoxChecker: check if object is inside a moving bbox attached to drawer."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import (
    cfg,
    mat3_mul_mat,
    mat3_mul_vec,
    mat3_transpose_mul_vec,
    quat_to_mat3,
    resolve_joint,
    resolve_position,
    resolve_rot_mat,
    vec3_add,
    vec3_sub,
)

logger = logging.getLogger("app.checkers")


class DrawerBBoxChecker(BaseChecker):
    """Check if object is inside a moving bbox attached to a drawer joint."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.obj_name: str = cfg(config, "objName", "obj_name", default="")
        self.cabinet_name: str = cfg(config, "cabinetName", "cabinet_name", default="")
        self.joint_name: str = cfg(config, "jointName", "joint_name", default="")
        self.base_offset: list[float] = list(cfg(config, "baseOffset", "base_offset", default=[0.0, 0.0, 0.0]))
        self.displacement_axis: int = cfg(config, "displacementAxis", "displacement_axis", default=1)
        self.relative_quat: list[float] = cfg(config, "relativeQuat", "relative_quat", default=[1.0, 0.0, 0.0, 0.0])
        self.half_size: list[float] = cfg(config, "halfSize", "half_size", default=[0.0, 0.0, 0.0])

    def check(self, payload: CheckerRuntimePayload) -> bool:
        obj_pos = resolve_position(self.obj_name, payload.object_positions)
        cab_pos = resolve_position(self.cabinet_name, payload.object_positions)
        if obj_pos is None or cab_pos is None:
            logger.warning("DrawerBBoxChecker: missing positions obj=%s cab=%s", self.obj_name, self.cabinet_name)
            return False

        if payload.body_rot_mats is None:
            logger.warning("DrawerBBoxChecker: body_rot_mats not provided")
            return False
        cab_rot = resolve_rot_mat(self.cabinet_name, payload.body_rot_mats)
        if cab_rot is None:
            logger.warning("DrawerBBoxChecker: rotation not found for %s", self.cabinet_name)
            return False

        if payload.joint_positions is None:
            logger.warning("DrawerBBoxChecker: joint_positions not provided")
            return False
        joint_pos = resolve_joint(self.joint_name, payload.joint_positions)
        if joint_pos is None:
            logger.warning("DrawerBBoxChecker: joint not found: %s", self.joint_name)
            return False

        base = list(self.base_offset)
        base[self.displacement_axis] += joint_pos

        rel_rot = quat_to_mat3(self.relative_quat)
        world_rot = mat3_mul_mat(cab_rot, rel_rot)
        world_t = vec3_add(mat3_mul_vec(cab_rot, base), cab_pos)

        rel = vec3_sub(obj_pos, world_t)
        local = mat3_transpose_mul_vec(world_rot, rel)

        eps = 1e-6
        passed = all(
            abs(local[i]) <= self.half_size[i] + eps
            for i in range(3)
        )
        if not passed:
            axes = ['x', 'y', 'z']
            details = "  ".join(
                f"{axes[i]}: local={local[i]:+.5f} halfSize={self.half_size[i]:.5f}"
                f"{'  ** OUT' if abs(local[i]) > self.half_size[i] + eps else ''}"
                for i in range(3)
            )
            logger.info(
                "DrawerBBoxChecker FAIL: obj=%s  %s",
                self.obj_name, details,
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "DrawerBBoxChecker", "passed": False}
        obj_pos = resolve_position(self.obj_name, payload.object_positions)
        cab_pos = resolve_position(self.cabinet_name, payload.object_positions)
        if obj_pos is None or cab_pos is None:
            detail["obj_name"] = self.obj_name
            return False, detail

        if payload.body_rot_mats is None:
            detail["obj_name"] = self.obj_name
            return False, detail
        cab_rot = resolve_rot_mat(self.cabinet_name, payload.body_rot_mats)
        if cab_rot is None:
            detail["obj_name"] = self.obj_name
            return False, detail

        if payload.joint_positions is None:
            detail["obj_name"] = self.obj_name
            return False, detail
        joint_pos = resolve_joint(self.joint_name, payload.joint_positions)
        if joint_pos is None:
            detail["obj_name"] = self.obj_name
            return False, detail

        base = list(self.base_offset)
        base[self.displacement_axis] += joint_pos

        rel_rot = quat_to_mat3(self.relative_quat)
        world_rot = mat3_mul_mat(cab_rot, rel_rot)
        world_t = vec3_add(mat3_mul_vec(cab_rot, base), cab_pos)

        rel = vec3_sub(obj_pos, world_t)
        local = mat3_transpose_mul_vec(world_rot, rel)

        eps = 1e-6
        axes_labels = ["x", "y", "z"]
        axes_in_range = {
            axes_labels[i]: abs(local[i]) <= self.half_size[i] + eps
            for i in range(3)
        }
        passed = all(axes_in_range.values())
        detail.update(
            passed=passed,
            obj_name=self.obj_name,
            local_coords=list(local),
            half_size=list(self.half_size),
            axes_in_range=axes_in_range,
        )
        return passed, detail
