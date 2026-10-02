"""RelativeAxisRangeChecker: axis delta range check between target and reference."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg, resolve_position

logger = logging.getLogger("app.checkers")


class RelativeAxisRangeChecker(BaseChecker):
    """Check if target axis value is within range relative to reference."""

    AXIS_MAP = {"x": 0, "y": 1, "z": 2}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.target_body_name: str = cfg(config, "targetBodyName", "target_body_name", default="sample/")
        self.reference_body_name: str = cfg(config, "referenceBodyName", "reference_body_name", default="target/")
        axis_str: str = cfg(config, "axis", default="y")
        self.axis_index: int = self.AXIS_MAP.get(axis_str, 1)
        self.min_offset: float = cfg(config, "minOffset", "min_offset", default=float("-inf"))
        self.max_offset: float = cfg(config, "maxOffset", "max_offset", default=float("inf"))

    def check(self, payload: CheckerRuntimePayload) -> bool:
        target_pos = resolve_position(self.target_body_name, payload.object_positions)
        ref_pos = resolve_position(self.reference_body_name, payload.object_positions)
        if target_pos is None or ref_pos is None:
            logger.warning(
                "RelativeAxisRangeChecker: missing positions target=%s ref=%s",
                self.target_body_name, self.reference_body_name,
            )
            return False

        delta = target_pos[self.axis_index] - ref_pos[self.axis_index]
        passed = self.min_offset <= delta <= self.max_offset
        if not passed:
            axis_names = {0: 'x', 1: 'y', 2: 'z'}
            axis_label = axis_names.get(self.axis_index, '?')
            logger.info(
                "RelativeAxisRangeChecker FAIL: target=%s ref=%s axis=%s delta=%.5f bounds=[%.5f, %.5f]"
                "%s%s",
                self.target_body_name, self.reference_body_name, axis_label, delta,
                self.min_offset, self.max_offset,
                "  ** OUT (below min)" if delta < self.min_offset else "",
                "  ** OUT (above max)" if delta > self.max_offset else "",
            )
        return passed

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "RelativeAxisRangeChecker", "passed": False}
        target_pos = resolve_position(self.target_body_name, payload.object_positions)
        ref_pos = resolve_position(self.reference_body_name, payload.object_positions)
        if target_pos is None or ref_pos is None:
            detail.update(target_body=self.target_body_name, reference_body=self.reference_body_name)
            return False, detail

        axis_names = {0: "x", 1: "y", 2: "z"}
        delta = target_pos[self.axis_index] - ref_pos[self.axis_index]
        passed = self.min_offset <= delta <= self.max_offset
        detail.update(
            passed=passed,
            target_body=self.target_body_name,
            reference_body=self.reference_body_name,
            axis=axis_names.get(self.axis_index, "?"),
            delta=delta,
            min_offset=self.min_offset,
            max_offset=self.max_offset,
        )
        return passed, detail
