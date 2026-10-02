"""Checker type registry: maps type name strings to checker classes."""

from __future__ import annotations

from .BowlPositionChecker import BowlPositionChecker
from .RelativePositionBoundsChecker import RelativePositionBoundsChecker
from .RelativeCylinderChecker import RelativeCylinderChecker
from .RelativeAxisRangeChecker import RelativeAxisRangeChecker
from .ObjectRotationChecker import ObjectRotationChecker
from .SampleRotationChecker import SampleRotationChecker
from .SampleTiltChecker import SampleTiltChecker
from .DirectedRotationChecker import DirectedRotationChecker
from .PositionDeltaChecker import PositionDeltaChecker
from .SamplePositionDeltaChecker import SamplePositionDeltaChecker
from .GripperOpenChecker import GripperOpenChecker
from .DrawerPositionChecker import DrawerPositionChecker
from .BoxJointPositionChecker import BoxJointPositionChecker
from .JointThresholdChecker import JointThresholdChecker
from .FrameBBoxChecker import FrameBBoxChecker
from .DrawerBBoxChecker import DrawerBBoxChecker
from .BowlInDrawerChecker import BowlInDrawerChecker
from .KeyPressSetChecker import KeyPressSetChecker

CHECKER_REGISTRY: dict[str, type] = {
    "BowlPositionChecker": BowlPositionChecker,
    "RelativePositionBoundsChecker": RelativePositionBoundsChecker,
    "RelativeCylinderChecker": RelativeCylinderChecker,
    "RelativeAxisRangeChecker": RelativeAxisRangeChecker,
    "ObjectRotationChecker": ObjectRotationChecker,
    "SampleRotationChecker": SampleRotationChecker,
    "SampleTiltChecker": SampleTiltChecker,
    "DirectedRotationChecker": DirectedRotationChecker,
    "PositionDeltaChecker": PositionDeltaChecker,
    "SamplePositionDeltaChecker": SamplePositionDeltaChecker,
    "GripperOpenChecker": GripperOpenChecker,
    "DrawerPositionChecker": DrawerPositionChecker,
    "BoxJointPositionChecker": BoxJointPositionChecker,
    "JointThresholdChecker": JointThresholdChecker,
    "FrameBBoxChecker": FrameBBoxChecker,
    "DrawerBBoxChecker": DrawerBBoxChecker,
    "BowlInDrawerChecker": BowlInDrawerChecker,
    "KeyPressSetChecker": KeyPressSetChecker,
}


def get_checker_class(type_name: str) -> type | None:
    return CHECKER_REGISTRY.get(type_name)
