"""BowlInDrawerChecker: check if bowl is inside the drawer region."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_joint, resolve_position, vec3_sub

logger = logging.getLogger("app.checkers")


class BowlInDrawerChecker(BaseChecker):
    """Check if bowl is inside the drawer region (hardcoded geometry)."""

    DRAWER_REGION_CENTER = [0.00328, 0.01128, 0.18563]
    DRAWER_REGION_HALF_SIZE = [0.02993, 0.07561, 0.10224]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.bowl_body_name: str = cfg(config, "bowlBodyName", "bowl_body_name", default="akita_black_bowl/")
        self.drawer_body_name: str = cfg(config, "drawerBodyName", "drawer_body_name", default="wooden_cabinet/cabinet_top")
        self.drawer_joint_name: str = cfg(config, "drawerJointName", "drawer_joint_name", default="wooden_cabinet/top_level")
        self.cabinet_base_body_name: str = cfg(config, "cabinetBaseBodyName", "cabinet_base_body_name", default="wooden_cabinet/")

    def check(self, payload: CheckerRuntimePayload) -> bool:
        bowl_pos = resolve_position(self.bowl_body_name, payload.object_positions)
        if bowl_pos is None:
            logger.warning("BowlInDrawerChecker: bowl position not found: %s", self.bowl_body_name)
            return False

        cab_base_pos = resolve_position(self.cabinet_base_body_name, payload.object_positions)
        if cab_base_pos is None:
            cab_base_pos = [0.5, -0.3, 0.0]

        drawer_body_pos = resolve_position(self.drawer_body_name, payload.object_positions)
        if drawer_body_pos is None:
            logger.warning("BowlInDrawerChecker: drawer body not found: %s", self.drawer_body_name)
            return False

        joint_pos = 0.0
        if payload.joint_positions is not None:
            val = resolve_joint(self.drawer_joint_name, payload.joint_positions)
            if val is not None:
                joint_pos = val

        center = [
            cab_base_pos[0] + drawer_body_pos[0] + self.DRAWER_REGION_CENTER[0],
            cab_base_pos[1] + drawer_body_pos[1] + joint_pos + self.DRAWER_REGION_CENTER[1],
            cab_base_pos[2] + drawer_body_pos[2] + self.DRAWER_REGION_CENTER[2],
        ]

        half = self.DRAWER_REGION_HALF_SIZE
        diff = vec3_sub(bowl_pos, center)
        passed = all(abs(diff[i]) <= half[i] for i in range(3))
        if not passed:
            axes = ['x', 'y', 'z']
            details = "  ".join(
                f"{axes[i]}: diff={diff[i]:+.5f} half={half[i]:.5f}"
                f"{'  ** OUT' if abs(diff[i]) > half[i] else ''}"
                for i in range(3)
            )
            logger.info(
                "BowlInDrawerChecker FAIL: bowl=%s center=[%.4f,%.4f,%.4f]  %s",
                self.bowl_body_name, center[0], center[1], center[2], details,
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "BowlInDrawerChecker", "passed": False}
        bowl_pos = resolve_position(self.bowl_body_name, payload.object_positions)
        if bowl_pos is None:
            detail["bowl_body"] = self.bowl_body_name
            return False, detail

        cab_base_pos = resolve_position(self.cabinet_base_body_name, payload.object_positions)
        if cab_base_pos is None:
            cab_base_pos = [0.5, -0.3, 0.0]

        drawer_body_pos = resolve_position(self.drawer_body_name, payload.object_positions)
        if drawer_body_pos is None:
            detail["bowl_body"] = self.bowl_body_name
            return False, detail

        joint_pos = 0.0
        if payload.joint_positions is not None:
            val = resolve_joint(self.drawer_joint_name, payload.joint_positions)
            if val is not None:
                joint_pos = val

        center = [
            cab_base_pos[0] + drawer_body_pos[0] + self.DRAWER_REGION_CENTER[0],
            cab_base_pos[1] + drawer_body_pos[1] + joint_pos + self.DRAWER_REGION_CENTER[1],
            cab_base_pos[2] + drawer_body_pos[2] + self.DRAWER_REGION_CENTER[2],
        ]

        half = self.DRAWER_REGION_HALF_SIZE
        diff = vec3_sub(bowl_pos, center)
        axes_labels = ["x", "y", "z"]
        axes_in_range = {
            axes_labels[i]: abs(diff[i]) <= half[i]
            for i in range(3)
        }
        passed = all(axes_in_range.values())
        detail.update(
            passed=passed,
            bowl_body=self.bowl_body_name,
            bowl_position=list(bowl_pos),
            region_center=center,
            diff=list(diff),
            half_size=list(half),
            axes_in_range=axes_in_range,
        )
        return passed, detail
