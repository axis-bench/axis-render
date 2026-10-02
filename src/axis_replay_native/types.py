"""Replay unified input/output structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecodedStep:
    sim_time: float
    ctrl_by_name: dict[str, float]
    keyboard_down_indices: list[int] = field(default_factory=list)


@dataclass
class DecodedTrajectory:
    format: str
    steps: list[DecodedStep]
    control_names: list[str]
    keyboard_names: list[str] = field(default_factory=list)
    object_state_start: dict[str, Any] | None = None
    object_state_end: dict[str, Any] | None = None
    initial_snapshot: dict[str, Any] | None = None
    task_id: int | None = None
    raw_step_count: int = 0
    removed_zero_dt_steps: int = 0


@dataclass
class BodyState:
    name: str
    pos: list[float]
    quat: list[float]


@dataclass
class ReplayResult:
    success: bool
    final_body_states: dict[str, BodyState] = field(default_factory=dict)
    error_report: list[str] = field(default_factory=list)
    position_errors: dict[str, float] = field(default_factory=dict)
    rotation_errors_deg: dict[str, float] = field(default_factory=dict)
    total_steps_executed: int = 0
    final_sim_time: float = 0.0
    timing_summary: dict[str, Any] = field(default_factory=dict)
