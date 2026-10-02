"""RelativeCylinderChecker: xy-radius + z-height check around a reference."""

from __future__ import annotations

import logging
import math
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_position, resolve_site_position

logger = logging.getLogger("app.checkers")


class RelativeCylinderChecker(BaseChecker):
    """Check if object is within a cylinder around a reference point."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.obj_name: str = cfg(config, "objName", "obj_name", default="")
        self.ref_name: str = cfg(config, "refName", "ref_name", default="")
        self.ref_type: str = cfg(config, "refType", "ref_type", default="object")
        self.xy_radius: float = cfg(config, "xyRadius", "xy_radius", default=0.06)
        self.height_min: float = cfg(config, "heightMin", "height_min", default=0.0)
        self.height_max: float = cfg(config, "heightMax", "height_max", default=0.03)

    def check(self, payload: CheckerRuntimePayload) -> bool:
        obj_pos = resolve_position(self.obj_name, payload.object_positions)
        if obj_pos is None:
            logger.warning("RelativeCylinderChecker: obj position not found for %s", self.obj_name)
            return False

        if self.ref_type == "site":
            if payload.site_positions is None:
                logger.warning("RelativeCylinderChecker: site_positions not provided")
                return False
            ref_pos = resolve_site_position(self.ref_name, payload.site_positions)
        else:
            ref_pos = resolve_position(self.ref_name, payload.object_positions)

        if ref_pos is None:
            logger.warning("RelativeCylinderChecker: ref position not found for %s", self.ref_name)
            return False

        dx = obj_pos[0] - ref_pos[0]
        dy = obj_pos[1] - ref_pos[1]
        dz = obj_pos[2] - ref_pos[2]

        xy_dist = math.sqrt(dx * dx + dy * dy)
        xy_ok = xy_dist < self.xy_radius
        dz_ok = self.height_min < dz < self.height_max
        passed = xy_ok and dz_ok
        if not passed:
            logger.info(
                "RelativeCylinderChecker FAIL: obj=%s ref=%s  "
                "xy_dist=%.5f (limit=%.5f)%s  dz=%.5f (range=[%.5f, %.5f])%s",
                self.obj_name, self.ref_name,
                xy_dist, self.xy_radius, "  ** OUT" if not xy_ok else "",
                dz, self.height_min, self.height_max, "  ** OUT" if not dz_ok else "",
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "RelativeCylinderChecker", "passed": False}
        obj_pos = resolve_position(self.obj_name, payload.object_positions)
        if obj_pos is None:
            detail.update(obj_name=self.obj_name, ref_name=self.ref_name)
            return False, detail

        if self.ref_type == "site":
            if payload.site_positions is None:
                detail.update(obj_name=self.obj_name, ref_name=self.ref_name)
                return False, detail
            ref_pos = resolve_site_position(self.ref_name, payload.site_positions)
        else:
            ref_pos = resolve_position(self.ref_name, payload.object_positions)

        if ref_pos is None:
            detail.update(obj_name=self.obj_name, ref_name=self.ref_name)
            return False, detail

        dx = obj_pos[0] - ref_pos[0]
        dy = obj_pos[1] - ref_pos[1]
        dz = obj_pos[2] - ref_pos[2]

        xy_dist = math.sqrt(dx * dx + dy * dy)
        xy_ok = xy_dist < self.xy_radius
        dz_ok = self.height_min < dz < self.height_max
        passed = xy_ok and dz_ok

        detail.update(
            passed=passed,
            obj_name=self.obj_name,
            ref_name=self.ref_name,
            xy_distance=xy_dist,
            xy_radius=self.xy_radius,
            xy_in_range=xy_ok,
            dz=dz,
            height_min=self.height_min,
            height_max=self.height_max,
            dz_in_range=dz_ok,
        )
        return passed, detail
