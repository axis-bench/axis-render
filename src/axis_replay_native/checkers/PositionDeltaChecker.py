"""PositionDeltaChecker: per-axis delta min/max check from initial position."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_position

logger = logging.getLogger("app.checkers")

AXIS_MAP = {"x": 0, "y": 1, "z": 2}


class PositionDeltaChecker(BaseChecker):
    """Check if object has moved within delta ranges from initial position."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.body_name: str = cfg(config, "sampleBodyName", "sample_body_name", default="sample/")
        self.initial_position: list[float] | None = cfg(config, "initialPosition", "initial_position")
        self.min_delta_x: float | None = cfg(config, "minDeltaX", "min_delta_x")
        self.max_delta_x: float | None = cfg(config, "maxDeltaX", "max_delta_x")
        self.min_delta_y: float | None = cfg(config, "minDeltaY", "min_delta_y")
        self.max_delta_y: float | None = cfg(config, "maxDeltaY", "max_delta_y")
        self.min_delta_z: float | None = cfg(config, "minDeltaZ", "min_delta_z")
        self.max_delta_z: float | None = cfg(config, "maxDeltaZ", "max_delta_z")
        axes_raw = cfg(config, "axes", default=["x", "z"])
        if isinstance(axes_raw, str):
            axes_raw = [axes_raw]
        self.axes: list[str] = axes_raw

    def check(self, payload: CheckerRuntimePayload) -> bool:
        current_pos = resolve_position(self.body_name, payload.object_positions)
        if current_pos is None:
            logger.warning("PositionDeltaChecker: position not found for %s", self.body_name)
            return False

        initial = self.initial_position
        if initial is None:
            if payload.initial_object_positions is not None:
                initial = resolve_position(self.body_name, payload.initial_object_positions)
            if initial is None:
                logger.warning("PositionDeltaChecker: no initial position available")
                return False

        delta = [current_pos[i] - initial[i] for i in range(3)]

        thresholds = {
            "x": (self.min_delta_x, self.max_delta_x),
            "y": (self.min_delta_y, self.max_delta_y),
            "z": (self.min_delta_z, self.max_delta_z),
        }

        failed = False
        fail_details = []
        for axis in self.axes:
            idx = AXIS_MAP.get(axis)
            if idx is None:
                continue
            min_val, max_val = thresholds[axis]
            value = delta[idx]
            out = False
            if min_val is not None and value < min_val:
                out = True
                failed = True
            if max_val is not None and value > max_val:
                out = True
                failed = True
            min_s = f"{min_val:.5f}" if min_val is not None else "None"
            max_s = f"{max_val:.5f}" if max_val is not None else "None"
            fail_details.append(
                f"{axis}: delta={value:+.5f} bounds=[{min_s}, {max_s}]"
                f"{'  ** OUT' if out else ''}"
            )

        if failed:
            logger.info(
                "PositionDeltaChecker FAIL: body=%s  %s",
                self.body_name, "  ".join(fail_details),
            )
            return False

        return True

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": type(self).__name__, "passed": False}
        current_pos = resolve_position(self.body_name, payload.object_positions)
        if current_pos is None:
            detail["body_name"] = self.body_name
            return False, detail

        initial = self.initial_position
        if initial is None:
            if payload.initial_object_positions is not None:
                initial = resolve_position(self.body_name, payload.initial_object_positions)
            if initial is None:
                detail["body_name"] = self.body_name
                return False, detail

        delta = [current_pos[i] - initial[i] for i in range(3)]

        thresholds = {
            "x": (self.min_delta_x, self.max_delta_x),
            "y": (self.min_delta_y, self.max_delta_y),
            "z": (self.min_delta_z, self.max_delta_z),
        }

        axes_detail: dict[str, dict[str, Any]] = {}
        failed = False
        for axis in self.axes:
            idx = AXIS_MAP.get(axis)
            if idx is None:
                continue
            min_val, max_val = thresholds[axis]
            value = delta[idx]
            in_range = True
            if min_val is not None and value < min_val:
                in_range = False
                failed = True
            if max_val is not None and value > max_val:
                in_range = False
                failed = True
            axes_detail[axis] = {"delta": value, "min": min_val, "max": max_val, "in_range": in_range}

        passed = not failed
        detail.update(
            passed=passed,
            body_name=self.body_name,
            current_position=list(current_pos),
            initial_position=list(initial),
            delta=delta,
            axes=axes_detail,
        )
        return passed, detail
