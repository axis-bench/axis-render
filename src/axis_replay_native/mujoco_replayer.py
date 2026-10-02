"""Execute MuJoCo trajectory replay loop."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import mujoco
import numpy as np

from .mujoco_model_loader import build_actuator_id_map, get_body_name
from .replay_trace import JsonlTraceWriter, collect_body_pose_map, collect_joint_qpos_map
from .types import BodyState, DecodedTrajectory, ReplayResult

logger = logging.getLogger(__name__)
ZERO_DT_EPS = 1e-9


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _collect_body_states(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, BodyState]:
    states: dict[str, BodyState] = {}
    for body_id in range(model.nbody):
        name = get_body_name(model, body_id)
        if not name:
            continue
        states[name] = BodyState(
            name=name,
            pos=data.xpos[body_id].tolist(),
            quat=data.xquat[body_id].tolist(),
        )
    return states


def _compare_with_expected(
    final_states: dict[str, BodyState],
    expected_end: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    if not expected_end:
        return {}, {}, []

    expected_positions = expected_end.get("object_positions", {})
    expected_orientations = expected_end.get("object_orientations", {})
    position_errors: dict[str, float] = {}
    rotation_errors: dict[str, float] = {}
    messages: list[str] = []

    for body_name in set(expected_positions.keys()) | set(expected_orientations.keys()):
        if body_name == "world":
            continue
        actual = final_states.get(body_name)
        if actual is None:
            continue

        exp_pos = expected_positions.get(body_name)
        if exp_pos is not None:
            diff = np.array(actual.pos) - np.array(exp_pos[:3])
            err = float(np.linalg.norm(diff))
            position_errors[body_name] = err
            if err > 0.05:
                messages.append(f"[POS] {body_name}: error={err:.4f}m (>0.05m threshold)")

        exp_quat = expected_orientations.get(body_name)
        if exp_quat is not None:
            q_actual = np.array(actual.quat)
            q_expected = np.array(exp_quat[:4])
            q_actual = q_actual / (np.linalg.norm(q_actual) + 1e-12)
            q_expected = q_expected / (np.linalg.norm(q_expected) + 1e-12)
            angle_deg = _quat_angle_deg(q_actual, q_expected)
            rotation_errors[body_name] = angle_deg
            if angle_deg > 10.0:
                messages.append(f"[ROT] {body_name}: error={angle_deg:.2f}° (>10.0° threshold)")

    return position_errors, rotation_errors, messages


def build_step_schedule(
    trajectory: DecodedTrajectory,
    timestep: float,
    *,
    fallback_ctrl_dt: float,
) -> tuple[list[int], dict[str, Any]]:
    """Translate cleaned sim_time samples into MuJoCo step counts.

    The schedule is generated cumulatively to avoid drift from repeatedly rounding
    each fractional segment independently. The final control is held for one more
    interval using the last observed positive dt; if the trajectory does not
    provide enough timing signal, we fall back to the nominal control interval.
    """
    if timestep <= 0:
        raise ValueError(f"MuJoCo timestep must be positive, got {timestep}")

    steps = trajectory.steps
    if not steps:
        return [], {
            "timing_mode": "trajectory_sim_time",
            "raw_step_count": trajectory.raw_step_count,
            "cleaned_step_count": 0,
            "removed_zero_dt_steps": trajectory.removed_zero_dt_steps,
            "fallback_ctrl_dt": float(fallback_ctrl_dt),
            "final_hold_dt": 0.0,
            "positive_dt_count": 0,
            "positive_dt_min": None,
            "positive_dt_max": None,
        }

    positive_dts: list[float] = []
    for prev, cur in zip(steps, steps[1:], strict=False):
        dt = float(cur.sim_time) - float(prev.sim_time)
        if dt < -ZERO_DT_EPS:
            raise ValueError(f"Trajectory sim_time is not monotonic: prev={prev.sim_time}, current={cur.sim_time}")
        if dt > ZERO_DT_EPS:
            positive_dts.append(dt)

    final_hold_dt = positive_dts[-1] if positive_dts else max(0.0, float(fallback_ctrl_dt))

    t0 = float(steps[0].sim_time)
    total_steps_executed = 0
    schedule: list[int] = []
    for index, step in enumerate(steps):
        if index + 1 < len(steps):
            segment_end_time = float(steps[index + 1].sim_time)
        else:
            segment_end_time = float(step.sim_time) + final_hold_dt

        rel_end_time = max(0.0, segment_end_time - t0)
        target_total_steps = int(round(rel_end_time / timestep))
        n_steps = max(0, target_total_steps - total_steps_executed)
        schedule.append(n_steps)
        total_steps_executed += n_steps

    return schedule, {
        "timing_mode": "trajectory_sim_time",
        "raw_step_count": trajectory.raw_step_count,
        "cleaned_step_count": len(steps),
        "removed_zero_dt_steps": trajectory.removed_zero_dt_steps,
        "fallback_ctrl_dt": float(fallback_ctrl_dt),
        "final_hold_dt": final_hold_dt,
        "positive_dt_count": len(positive_dts),
        "positive_dt_min": min(positive_dts) if positive_dts else None,
        "positive_dt_max": max(positive_dts) if positive_dts else None,
    }


def replay(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: DecodedTrajectory,
    *,
    fallback_ctrl_dt: float,
    trace_writer: JsonlTraceWriter | None = None,
) -> ReplayResult:
    actuator_map = build_actuator_id_map(model, trajectory.control_names)
    if not actuator_map:
        return ReplayResult(
            success=False,
            error_report=["No actuators could be mapped from control_names."],
        )

    unmapped = [name for name in trajectory.control_names if name not in actuator_map]
    if unmapped:
        logger.warning("Unmapped control names: %s", unmapped)

    timestep = model.opt.timestep
    steps = trajectory.steps
    step_schedule, timing_summary = build_step_schedule(trajectory, timestep, fallback_ctrl_dt=fallback_ctrl_dt)
    total_mj_steps = 0

    replay_start = time.perf_counter()
    logger.info(
        "Starting replay: raw_steps=%d cleaned_steps=%d timestep=%.6f zero_dt_removed=%d final_hold_dt=%.6f",
        trajectory.raw_step_count,
        len(steps),
        timestep,
        trajectory.removed_zero_dt_steps,
        timing_summary["final_hold_dt"],
    )

    for index, (step, n_steps) in enumerate(zip(steps, step_schedule, strict=True)):
        sim_time_before = float(data.time)
        for ctrl_name, ctrl_value in step.ctrl_by_name.items():
            actuator_id = actuator_map.get(ctrl_name)
            if actuator_id is not None:
                data.ctrl[actuator_id] = ctrl_value

        for _ in range(n_steps):
            mujoco.mj_step(model, data)
            total_mj_steps += 1

        if trace_writer is not None:
            trace_writer.write(
                "step",
                {
                    "source": "replay_native",
                    "step_index": index,
                    "traj_sim_time": step.sim_time,
                    "sim_time_before": sim_time_before,
                    "sim_time_after": float(data.time),
                    "physics_steps": n_steps,
                    "runout_after": index + 1 >= len(steps),
                    "control_targets": step.ctrl_by_name,
                    "joint_qpos": collect_joint_qpos_map(model, data),
                    "body_pose": collect_body_pose_map(model, data),
                },
            )

        if (index + 1) % 20 == 0 or index == len(steps) - 1:
            logger.debug(
                "Replay progress: step %d/%d, sim_time=%.3f",
                index + 1,
                len(steps),
                data.time,
            )

    final_states = _collect_body_states(model, data)
    pos_errors, rot_errors, error_msgs = _compare_with_expected(final_states, trajectory.object_state_end)
    replay_elapsed = time.perf_counter() - replay_start
    logger.info(
        "Replay complete: %d mj_steps, final_time=%.3f, replay_wall_seconds=%.3f",
        total_mj_steps,
        data.time,
        replay_elapsed,
    )
    if error_msgs:
        logger.warning("Replay drift detected:\n  %s", "\n  ".join(error_msgs))

    if trace_writer is not None:
        trace_writer.write(
            "final",
            {
                "source": "replay_native",
                "final_sim_time": float(data.time),
                "total_mj_steps": total_mj_steps,
                "position_errors": pos_errors,
                "rotation_errors_deg": rot_errors,
                "error_report": error_msgs,
            },
        )

    return ReplayResult(
        success=True,
        final_body_states=final_states,
        error_report=error_msgs,
        position_errors=pos_errors,
        rotation_errors_deg=rot_errors,
        total_steps_executed=total_mj_steps,
        final_sim_time=data.time,
        timing_summary=timing_summary,
    )
