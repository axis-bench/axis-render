"""RelativePositionBoundsChecker: per-axis range check on delta between obj and ref."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_position

logger = logging.getLogger("app.checkers")


class RelativePositionBoundsChecker(BaseChecker):
    """Check object position relative to reference within axis ranges."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.obj_name: str = cfg(config, "objName", "obj_name", default="")
        self.ref_name: str = cfg(config, "refName", "ref_name", default="")
        self.x_range: list[float] | None = cfg(config, "xRange", "x_range")
        self.y_range: list[float] | None = cfg(config, "yRange", "y_range")
        self.z_range: list[float] | None = cfg(config, "zRange", "z_range")

    def check(self, payload: CheckerRuntimePayload) -> bool:
        obj_pos = resolve_position(self.obj_name, payload.object_positions)
        ref_pos = resolve_position(self.ref_name, payload.object_positions)
        if obj_pos is None or ref_pos is None:
            logger.warning(
                "RelativePositionBoundsChecker: missing positions obj=%s ref=%s",
                self.obj_name, self.ref_name,
            )
            return False

        dx = obj_pos[0] - ref_pos[0]
        dy = obj_pos[1] - ref_pos[1]
        dz = obj_pos[2] - ref_pos[2]

        checks: list[bool] = []
        axis_details: list[str] = []
        if self.x_range is not None:
            ok = self.x_range[0] < dx < self.x_range[1]
            checks.append(ok)
            axis_details.append(
                f"x: delta={dx:+.5f} range=({self.x_range[0]:.5f}, {self.x_range[1]:.5f})"
                f"{'  ** OUT' if not ok else ''}"
            )
        if self.y_range is not None:
            ok = self.y_range[0] < dy < self.y_range[1]
            checks.append(ok)
            axis_details.append(
                f"y: delta={dy:+.5f} range=({self.y_range[0]:.5f}, {self.y_range[1]:.5f})"
                f"{'  ** OUT' if not ok else ''}"
            )
        if self.z_range is not None:
            ok = self.z_range[0] < dz < self.z_range[1]
            checks.append(ok)
            axis_details.append(
                f"z: delta={dz:+.5f} range=({self.z_range[0]:.5f}, {self.z_range[1]:.5f})"
                f"{'  ** OUT' if not ok else ''}"
            )

        if not checks:
            return False
        passed = all(checks)
        if not passed:
            logger.info(
                "RelativePositionBoundsChecker FAIL: obj=%s ref=%s  %s",
                self.obj_name, self.ref_name, "  ".join(axis_details),
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "RelativePositionBoundsChecker", "passed": False}
        obj_pos = resolve_position(self.obj_name, payload.object_positions)
        ref_pos = resolve_position(self.ref_name, payload.object_positions)
        if obj_pos is None or ref_pos is None:
            detail.update(obj_name=self.obj_name, ref_name=self.ref_name)
            return False, detail

        delta = [obj_pos[i] - ref_pos[i] for i in range(3)]
        ranges = {"x": self.x_range, "y": self.y_range, "z": self.z_range}
        delta_map = {"x": delta[0], "y": delta[1], "z": delta[2]}

        axes_detail: dict[str, dict[str, Any]] = {}
        checks: list[bool] = []
        for axis in ("x", "y", "z"):
            r = ranges[axis]
            if r is not None:
                d = delta_map[axis]
                ok = r[0] < d < r[1]
                checks.append(ok)
                axes_detail[axis] = {"delta": d, "range": list(r), "in_range": ok}

        passed = bool(checks) and all(checks)
        detail.update(
            passed=passed,
            obj_name=self.obj_name,
            ref_name=self.ref_name,
            obj_position=list(obj_pos),
            ref_position=list(ref_pos),
            delta=delta,
            axes=axes_detail,
        )
        return passed, detail
