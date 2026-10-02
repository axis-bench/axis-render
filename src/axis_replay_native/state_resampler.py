"""Joint-aware fixed-rate resampling for packed MuJoCo state trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from .packed_state_decoder import PackedStateTrajectory


STATE_RESAMPLING_SCHEMA = "joint_aware_state_resampling_v1"
_TIME_EPS = 1e-9
_QUAT_EPS = 1e-12
_VELOCITY_RELIABILITY_P90_MAX = 0.5
_HERMITE_ENVELOPE_FRACTION = 0.05


@dataclass(frozen=True)
class ResampledStateTimeline:
    trajectory: PackedStateTrajectory
    state_times: np.ndarray
    metadata: dict[str, Any]


def _validated_source_row_times(
    decoded: PackedStateTrajectory,
    state_times: list[float],
    time_summary: dict[str, Any],
    *,
    expected_duration_s: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    times = np.asarray(state_times, dtype=np.float64)
    if times.shape != (decoded.n_steps,):
        raise ValueError(
            f"state_times shape {times.shape} does not match decoded.n_steps={decoded.n_steps}."
        )
    if not np.all(np.isfinite(times)):
        raise ValueError("state_times must be finite.")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("state_times must be strictly increasing for state resampling.")

    summary = dict(time_summary)
    summary["source_time_header_corrected"] = False
    summary["source_time_header_correction"] = None
    start_time = float(decoded.initial_snapshot.get("time", 0.0) or 0.0)
    initial_qpos = np.asarray(decoded.initial_snapshot["qpos"], dtype=np.float64)
    initial_qvel = np.asarray(decoded.initial_snapshot["qvel"], dtype=np.float64)
    first_row_is_initial = bool(
        initial_qpos.shape == decoded.qpos[0].shape
        and initial_qvel.shape == decoded.qvel[0].shape
        and np.allclose(initial_qpos, decoded.qpos[0], atol=1e-5, rtol=1e-5)
        and np.allclose(initial_qvel, decoded.qvel[0], atol=1e-5, rtol=1e-5)
    )

    if decoded.step_counts is not None:
        between_row_offsets = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(np.asarray(decoded.step_counts, dtype=np.float64))
                * float(decoded.timestep),
            )
        )
        if expected_duration_s is not None and float(expected_duration_s) > 0.0:
            expected_end = start_time + float(expected_duration_s)
            leading_dt = expected_end - start_time - float(between_row_offsets[-1])
            tolerance = max(1e-6, 0.5 * float(decoded.timestep))
            if first_row_is_initial:
                if abs(leading_dt) > tolerance:
                    raise ValueError(
                        "step_counts timing with an initial-state first row disagrees with "
                        f"simulation duration by {leading_dt:.9g}s."
                    )
                leading_dt = 0.0
            elif leading_dt <= tolerance:
                raise ValueError(
                    "step_counts timing requires a positive initial-to-first-row interval, "
                    f"got {leading_dt:.9g}s."
                )
            step_ratio = leading_dt / float(decoded.timestep)
            if not np.isclose(step_ratio, round(step_ratio), rtol=0.0, atol=5e-4):
                raise ValueError(
                    "Authoritative duration implies a non-integral initial step count: "
                    f"initial_dt={leading_dt:.9g}s, timestep={decoded.timestep:.9g}s."
                )
            correction_policy = "authoritative_duration_plus_step_counts"
        elif first_row_is_initial:
            leading_dt = 0.0
            correction_policy = "initial_state_plus_step_counts"
        else:
            raise ValueError(
                "Post-step step_counts timing requires authoritative simulation duration "
                "to place the first source row without guessing."
            )

        corrected = start_time + float(leading_dt) + between_row_offsets
        changed = not np.allclose(corrected, times, rtol=1e-7, atol=1e-9)
        positive_intervals = np.diff(corrected)
        summary.update(
            {
                "state_time_mode": (
                    "step_counts_initial_state"
                    if first_row_is_initial
                    else "step_counts_post_step"
                ),
                "state_time_start": float(corrected[0]),
                "state_time_end": float(corrected[-1]),
                "state_dt_median": (
                    float(np.median(positive_intervals))
                    if positive_intervals.size
                    else None
                ),
                "source_time_header_corrected": changed,
                "source_time_header_correction": (
                    {
                        "policy": correction_policy,
                        "original_state_time_start": float(times[0]),
                        "original_state_time_end": float(times[-1]),
                        "corrected_state_time_start": float(corrected[0]),
                        "corrected_state_time_end": float(corrected[-1]),
                        "initial_to_first_row_dt": float(leading_dt),
                    }
                    if changed
                    else None
                ),
            }
        )
        return corrected, summary

    if expected_duration_s is None or float(expected_duration_s) <= 0.0:
        return times, summary

    expected_end = start_time + float(expected_duration_s)
    if expected_end <= start_time:
        raise ValueError(
            f"expected_duration_s={expected_end} must be greater than initial time {start_time}."
        )
    positive_intervals = np.diff(times)
    median_interval = float(np.median(positive_intervals)) if positive_intervals.size else 0.0
    end_drift = expected_end - float(times[-1])
    snap_tolerance = max(1e-6, min(1e-4, 1e-3 * median_interval))
    if abs(end_drift) <= snap_tolerance:
        summary["authoritative_end_time_drift_seconds"] = float(end_drift)
        summary["target_end_snapped_to_source_support"] = bool(abs(end_drift) > _TIME_EPS)
        return times, summary

    mode = str(summary.get("state_time_mode") or "")
    if mode == "steps_per_ctrl_post_step":
        interval_count = decoded.n_steps
        offset = 1
    elif mode == "steps_per_ctrl_initial_state":
        interval_count = decoded.n_steps - 1
        offset = 0
    else:
        raise ValueError(
            "Cannot repair regular source timing for unsupported state_time_mode "
            f"{mode!r}."
        )
    if interval_count < 1:
        raise ValueError("A single source state cannot cover a non-zero episode duration.")

    corrected_dt = (expected_end - start_time) / float(interval_count)
    if not np.isfinite(corrected_dt) or corrected_dt <= 0.0:
        raise ValueError(f"Invalid corrected source dt {corrected_dt!r}.")
    corrected = start_time + (np.arange(decoded.n_steps, dtype=np.float64) + offset) * corrected_dt
    summary.update(
        {
            "state_time_start": float(corrected[0]),
            "state_time_end": float(corrected[-1]),
            "state_dt_median": float(corrected_dt),
            "source_time_header_corrected": True,
            "source_time_header_correction": {
                "policy": "authoritative_duration_regular_grid",
                "original_state_time_end": float(times[-1]),
                "corrected_state_time_end": float(corrected[-1]),
                "corrected_state_dt": float(corrected_dt),
                "expected_duration_s": float(expected_duration_s),
                "authoritative_expected_end_time": expected_end,
                "original_end_drift_seconds": float(end_drift),
            },
        }
    )
    return corrected, summary


def _source_knots(
    decoded: PackedStateTrajectory,
    row_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    initial_time = float(decoded.initial_snapshot.get("time", 0.0) or 0.0)
    initial_qpos = np.asarray(decoded.initial_snapshot["qpos"], dtype=np.float64)
    initial_qvel = np.asarray(decoded.initial_snapshot["qvel"], dtype=np.float64)
    if initial_qpos.shape != (decoded.nq,) or initial_qvel.shape != (decoded.nv,):
        raise ValueError("initial_snapshot dimensions do not match the packed state trajectory.")

    prepend_initial = bool(float(row_times[0]) > initial_time + _TIME_EPS)
    if float(row_times[0]) < initial_time - _TIME_EPS:
        raise ValueError(
            f"First source row time {row_times[0]} precedes initial snapshot time {initial_time}."
        )

    if decoded.nu:
        snapshot_ctrl = decoded.initial_snapshot.get("ctrl")
        if snapshot_ctrl is None:
            if prepend_initial:
                raise ValueError(
                    "A post-step trajectory needs initial_snapshot.ctrl to resample controls "
                    "before the first packed row."
                )
            initial_ctrl = np.asarray(decoded.ctrl[0], dtype=np.float64)
            initial_ctrl_source = "first_source_row_same_time"
        else:
            initial_ctrl = np.asarray(snapshot_ctrl, dtype=np.float64)
            initial_ctrl_source = "initial_snapshot"
        if initial_ctrl.shape != (decoded.nu,):
            raise ValueError("Initial ctrl dimensions do not match decoded.nu.")
    else:
        initial_ctrl = np.zeros((0,), dtype=np.float64)
        initial_ctrl_source = "empty"

    if prepend_initial:
        times = np.concatenate(([initial_time], row_times))
        qpos = np.vstack((initial_qpos, decoded.qpos))
        qvel = np.vstack((initial_qvel, decoded.qvel))
        ctrl = np.vstack((initial_ctrl, decoded.ctrl))
    else:
        times = row_times.copy()
        qpos = np.asarray(decoded.qpos, dtype=np.float64).copy()
        qvel = np.asarray(decoded.qvel, dtype=np.float64).copy()
        ctrl = np.asarray(decoded.ctrl, dtype=np.float64).copy()

    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("State interpolation knots must be strictly increasing.")
    if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)) and np.all(np.isfinite(ctrl))):
        raise ValueError("State interpolation knots contain non-finite values.")
    return (
        times,
        qpos,
        qvel,
        ctrl,
        {
            "initial_snapshot_used_as_knot": prepend_initial,
            "initial_ctrl_source": initial_ctrl_source,
        },
    )


def _target_times(
    *,
    start_time: float,
    end_time: float,
    fps: int,
) -> np.ndarray:
    if int(fps) <= 0:
        raise ValueError(f"Target fps must be positive, got {fps!r}.")
    duration = float(end_time) - float(start_time)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"State resampling requires a positive duration, got {duration!r}.")
    # A duration-D episode contains round(D * fps) right-endpoint samples. This
    # matches the encoded 15 Hz episode duration while avoiding a duplicated
    # t=0 / t=D pair (0.8 s -> 12 frames, not 13).
    frame_count = max(1, int(np.floor(duration * float(fps) + 0.5 + 1e-9)))
    target_dt = 1.0 / float(fps)
    # End alignment preserves the terminal state without appending an irregular
    # frame or extrapolating beyond the interpolation support.
    times = float(end_time) - np.arange(frame_count - 1, -1, -1, dtype=np.float64) * target_dt
    if float(times[0]) < float(start_time) - _TIME_EPS:
        raise AssertionError("Internal error: target timeline escaped interpolation support.")
    return times


def _interpolate_euclidean(
    source_times: np.ndarray,
    values: np.ndarray,
    velocities: np.ndarray | None,
    target_times: np.ndarray,
    *,
    bounds: tuple[float, float] | None = None,
    speed_floor: float,
) -> tuple[np.ndarray, dict[str, int]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if len(source_times) != len(values):
        raise ValueError("Interpolation values do not align with source_times.")
    if len(source_times) < 2:
        raise ValueError("At least two knots are required for continuous state interpolation.")

    velocities_array: np.ndarray | None = None
    if velocities is not None:
        velocities_array = np.asarray(velocities, dtype=np.float64)
        if velocities_array.ndim == 1:
            velocities_array = velocities_array[:, None]
        if velocities_array.shape != values.shape:
            velocities_array = None

    result = np.empty((len(target_times), values.shape[1]), dtype=np.float64)
    methods: Counter[str] = Counter()
    for coordinate in range(values.shape[1]):
        coordinate_values = values[:, coordinate]
        if len(source_times) == 2:
            result[:, coordinate] = np.interp(
                target_times,
                source_times,
                coordinate_values,
            )
            methods["linear_two_knots"] += 1
            continue

        velocity_reliable = False
        coordinate_velocity: np.ndarray | None = None
        if velocities_array is not None:
            coordinate_velocity = velocities_array[:, coordinate]
            if np.all(np.isfinite(coordinate_velocity)):
                reference = np.gradient(coordinate_values, source_times, edge_order=2)
                scale = max(float(np.quantile(np.abs(reference), 0.9)), float(speed_floor))
                normalized_error = np.abs(coordinate_velocity - reference) / (
                    np.abs(reference) + scale
                )
                moving = np.abs(reference) > float(speed_floor)
                if np.any(moving):
                    magnitude_ratio = np.abs(coordinate_velocity[moving]) / (
                        np.abs(reference[moving]) + float(speed_floor)
                    )
                    magnitude_ok = bool(
                        np.quantile(magnitude_ratio, 0.5) >= 0.2
                        and np.quantile(magnitude_ratio, 0.9) <= 5.0
                    )
                    direction_ok = bool(
                        np.mean(
                            coordinate_velocity[moving] * reference[moving] >= 0.0
                        )
                        >= 0.9
                    )
                else:
                    magnitude_ok = True
                    direction_ok = True
                velocity_reliable = bool(
                    np.quantile(normalized_error, 0.9)
                    <= _VELOCITY_RELIABILITY_P90_MAX
                    and magnitude_ok
                    and direction_ok
                )

        fallback_reason = "missing_or_unreliable_velocity"
        if velocity_reliable and coordinate_velocity is not None:
            hermite = CubicHermiteSpline(
                source_times,
                coordinate_values,
                coordinate_velocity,
                extrapolate=False,
            )
            fractions = np.asarray([0.25, 0.5, 0.75], dtype=np.float64)
            probe_times = (
                source_times[:-1, None]
                + np.diff(source_times)[:, None] * fractions[None, :]
            )
            probe_values = np.asarray(
                hermite(probe_times.reshape(-1)),
                dtype=np.float64,
            ).reshape(len(source_times) - 1, len(fractions))
            endpoints_min = np.minimum(coordinate_values[:-1], coordinate_values[1:])
            endpoints_max = np.maximum(coordinate_values[:-1], coordinate_values[1:])
            envelope = np.maximum(
                np.abs(coordinate_values[1:] - coordinate_values[:-1])
                * _HERMITE_ENVELOPE_FRACTION,
                1e-8,
            )
            envelope_ok = bool(
                np.all(probe_values >= endpoints_min[:, None] - envelope[:, None])
                and np.all(probe_values <= endpoints_max[:, None] + envelope[:, None])
            )
            bounds_ok = bool(
                bounds is None
                or (
                    np.all(probe_values >= float(bounds[0]) - 1e-7)
                    and np.all(probe_values <= float(bounds[1]) + 1e-7)
                )
            )
            if envelope_ok and bounds_ok:
                result[:, coordinate] = np.asarray(hermite(target_times), dtype=np.float64)
                methods["cubic_hermite_reliable_qvel"] += 1
                continue
            fallback_reason = "hermite_overshoot_guard"

        pchip = PchipInterpolator(
            source_times,
            coordinate_values,
            extrapolate=False,
        )
        result[:, coordinate] = np.asarray(pchip(target_times), dtype=np.float64)
        methods[f"pchip_{fallback_reason}"] += 1

    if not np.all(np.isfinite(result)):
        raise ValueError("State interpolation produced non-finite values.")
    if bounds is not None and (
        np.any(result < float(bounds[0]) - 1e-6)
        or np.any(result > float(bounds[1]) + 1e-6)
    ):
        raise ValueError(
            f"Interpolated joint position violates range [{bounds[0]}, {bounds[1]}]."
        )
    return result, dict(sorted(methods.items()))


def _slerp_quaternions(
    source_times: np.ndarray,
    quaternions_wxyz: np.ndarray,
    target_times: np.ndarray,
) -> tuple[np.ndarray, int]:
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64).copy()
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= _QUAT_EPS):
        raise ValueError("Quaternion interpolation received an invalid quaternion.")
    quaternions /= norms[:, None]
    sign_flips = 0
    for index in range(1, len(quaternions)):
        if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
            sign_flips += 1
    if len(source_times) == 1:
        return np.repeat(quaternions, len(target_times), axis=0), sign_flips

    quaternions_xyzw = np.concatenate((quaternions[:, 1:], quaternions[:, :1]), axis=1)
    rotations = Rotation.from_quat(quaternions_xyzw)
    interpolated_xyzw = Slerp(source_times, rotations)(target_times).as_quat()
    interpolated_wxyz = np.concatenate(
        (interpolated_xyzw[:, 3:], interpolated_xyzw[:, :3]), axis=1
    )
    interpolated_wxyz /= np.linalg.norm(interpolated_wxyz, axis=1, keepdims=True)
    return interpolated_wxyz, sign_flips


def _interpolate_qpos(
    model: mujoco.MjModel,
    source_times: np.ndarray,
    source_qpos: np.ndarray,
    source_qvel: np.ndarray,
    target_times: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if source_qpos.shape != (len(source_times), model.nq):
        raise ValueError("source_qpos dimensions do not match the MuJoCo model.")
    if source_qvel.shape != (len(source_times), model.nv):
        raise ValueError("source_qvel dimensions do not match the MuJoCo model.")

    output = np.empty((len(target_times), model.nq), dtype=np.float64)
    methods: Counter[str] = Counter()
    quaternion_sign_flips = 0
    source_joint_limit_violations: list[dict[str, Any]] = []
    covered = np.zeros(model.nq, dtype=bool)

    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            position, position_methods = _interpolate_euclidean(
                source_times,
                source_qpos[:, qpos_adr : qpos_adr + 3],
                source_qvel[:, dof_adr : dof_adr + 3],
                target_times,
                speed_floor=1e-4,
            )
            quaternion, flips = _slerp_quaternions(
                source_times,
                source_qpos[:, qpos_adr + 3 : qpos_adr + 7],
                target_times,
            )
            output[:, qpos_adr : qpos_adr + 3] = position
            output[:, qpos_adr + 3 : qpos_adr + 7] = quaternion
            covered[qpos_adr : qpos_adr + 7] = True
            for method, count in position_methods.items():
                methods[f"free_position:{method}"] += count
            methods["free_rotation:slerp"] += 1
            quaternion_sign_flips += flips
            continue

        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            quaternion, flips = _slerp_quaternions(
                source_times,
                source_qpos[:, qpos_adr : qpos_adr + 4],
                target_times,
            )
            output[:, qpos_adr : qpos_adr + 4] = quaternion
            covered[qpos_adr : qpos_adr + 4] = True
            methods["ball_rotation:slerp"] += 1
            quaternion_sign_flips += flips
            continue

        if joint_type not in {
            int(mujoco.mjtJoint.mjJNT_SLIDE),
            int(mujoco.mjtJoint.mjJNT_HINGE),
        }:
            raise ValueError(f"Unsupported MuJoCo joint type {joint_type} for joint {joint_id}.")

        values = source_qpos[:, qpos_adr].copy()
        joint_label = "slide" if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE) else "hinge"
        if joint_label == "hinge" and not bool(model.jnt_limited[joint_id]):
            values = np.unwrap(values, period=2.0 * np.pi)
            methods["hinge:unwrap"] += 1
        bounds = None
        if bool(model.jnt_limited[joint_id]):
            model_lower = float(model.jnt_range[joint_id, 0])
            model_upper = float(model.jnt_range[joint_id, 1])
            source_lower = float(np.min(values))
            source_upper = float(np.max(values))
            lower_violation = max(0.0, model_lower - source_lower)
            upper_violation = max(0.0, source_upper - model_upper)
            if lower_violation > 1e-7 or upper_violation > 1e-7:
                joint_name = (
                    mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        joint_id,
                    )
                    or f"joint_{joint_id}"
                )
                source_joint_limit_violations.append(
                    {
                        "joint_id": int(joint_id),
                        "joint_name": joint_name,
                        "model_range": [model_lower, model_upper],
                        "source_range": [source_lower, source_upper],
                        "lower_violation": float(lower_violation),
                        "upper_violation": float(upper_violation),
                    }
                )
                methods[f"{joint_label}:source_range_expanded"] += 1
            # Recorded state replay must preserve the source knots even when the
            # model's nominal range is slightly tighter (common for fingers).
            # The guard still forbids interpolation from moving any farther out.
            bounds = (
                min(model_lower, source_lower),
                max(model_upper, source_upper),
            )
        interpolated, scalar_methods = _interpolate_euclidean(
            source_times,
            values,
            source_qvel[:, dof_adr],
            target_times,
            bounds=bounds,
            speed_floor=1e-4 if joint_label == "slide" else 1e-3,
        )
        output[:, qpos_adr] = interpolated[:, 0]
        covered[qpos_adr] = True
        for method, count in scalar_methods.items():
            methods[f"{joint_label}:{method}"] += count

    if not np.all(covered):
        missing = np.flatnonzero(~covered).tolist()
        raise ValueError(f"State interpolation did not populate qpos indices {missing}.")
    return output, {
        "joint_interpolation_methods": dict(sorted(methods.items())),
        "quaternion_source_sign_flips": int(quaternion_sign_flips),
        "source_joint_limit_violations": source_joint_limit_violations,
    }


def _differentiate_interpolated_qpos(
    model: mujoco.MjModel,
    target_times: np.ndarray,
    *,
    source_times: np.ndarray,
    source_qpos: np.ndarray,
    source_qvel: np.ndarray,
    target_fps: int,
) -> tuple[np.ndarray, float]:
    count = len(target_times)
    output = np.empty((count, model.nv), dtype=np.float64)
    minimum_source_dt = float(np.min(np.diff(source_times)))
    derivative_epsilon = max(
        1e-7,
        min(1e-4, 0.01 * minimum_source_dt, 1e-3 / float(target_fps)),
    )
    left_times = np.maximum(target_times - derivative_epsilon, float(source_times[0]))
    right_times = np.minimum(target_times + derivative_epsilon, float(source_times[-1]))
    left_qpos, _ = _interpolate_qpos(
        model,
        source_times,
        source_qpos,
        source_qvel,
        left_times,
    )
    right_qpos, _ = _interpolate_qpos(
        model,
        source_times,
        source_qpos,
        source_qvel,
        right_times,
    )

    for index in range(count):
        dt = float(right_times[index] - left_times[index])
        if dt <= 0.0:
            raise ValueError("State interpolation derivative probe has a non-positive interval.")
        mujoco.mj_differentiatePos(
            model,
            output[index],
            dt,
            np.asarray(left_qpos[index], dtype=np.float64),
            np.asarray(right_qpos[index], dtype=np.float64),
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Differentiating interpolated qpos produced non-finite qvel.")
    return output, derivative_epsilon


def _causal_ctrl_resample(
    source_times: np.ndarray,
    source_ctrl: np.ndarray,
    target_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if source_ctrl.shape[0] != len(source_times):
        raise ValueError("source_ctrl does not align with source_times.")
    indexes = np.searchsorted(source_times, target_times, side="right") - 1
    indexes = np.clip(indexes, 0, len(source_times) - 1)
    return (
        np.asarray(source_ctrl[indexes], dtype=np.float64).copy(),
        indexes.astype(np.int64, copy=False),
    )


def resample_packed_state_to_fps(
    model: mujoco.MjModel,
    decoded: PackedStateTrajectory,
    state_times: list[float],
    time_summary: dict[str, Any],
    *,
    fps: int,
    expected_duration_s: float | None,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> ResampledStateTimeline:
    """Resample full MuJoCo state without stepping physics or repeating state rows."""

    if model.nq != decoded.nq or model.nv != decoded.nv or model.nu != decoded.nu:
        raise ValueError("Decoded state dimensions do not match the MuJoCo model.")
    if int(frame_stride) < 1:
        raise ValueError("frame_stride must be >= 1.")

    row_times, corrected_time_summary = _validated_source_row_times(
        decoded,
        state_times,
        time_summary,
        expected_duration_s=expected_duration_s,
    )
    source_times, source_qpos, source_qvel, source_ctrl, knot_metadata = _source_knots(
        decoded,
        row_times,
    )
    start_time = float(decoded.initial_snapshot.get("time", 0.0) or 0.0)
    # expected_duration_s is used above to validate or repair the source knots.
    # The target grid is then snapped to the validated interpolation support so
    # harmless float drift in stored timestamps can never trigger extrapolation or lose terminal.
    end_time = float(source_times[-1])
    full_target_times = _target_times(start_time=start_time, end_time=end_time, fps=int(fps))
    if len(source_times) < 2:
        raise ValueError("At least two state knots are required for 15 Hz state resampling.")

    full_qpos, interpolation_metadata = _interpolate_qpos(
        model,
        source_times,
        source_qpos,
        source_qvel,
        full_target_times,
    )
    full_qvel, derivative_epsilon = _differentiate_interpolated_qpos(
        model,
        full_target_times,
        source_times=source_times,
        source_qpos=source_qpos,
        source_qvel=source_qvel,
        target_fps=int(fps),
    )
    full_ctrl, full_ctrl_source_indices = _causal_ctrl_resample(
        source_times,
        source_ctrl,
        full_target_times,
    )

    selected = np.arange(len(full_target_times), dtype=np.int64)[:: int(frame_stride)]
    if max_frames is not None:
        selected = selected[: max(0, int(max_frames))]
    if selected.size == 0:
        raise ValueError("No frames selected after 15 Hz state resampling.")

    target_times = full_target_times[selected]
    qpos = full_qpos[selected]
    qvel = full_qvel[selected]
    ctrl = full_ctrl[selected]
    right = np.searchsorted(source_times, target_times, side="right")
    right = np.clip(right, 1, len(source_times) - 1)
    left = right - 1
    denominators = source_times[right] - source_times[left]
    alpha = (target_times - source_times[left]) / denominators
    alpha = np.clip(alpha, 0.0, 1.0)
    exact_source = np.isclose(alpha, 0.0, atol=1e-8) | np.isclose(alpha, 1.0, atol=1e-8)

    source_intervals = np.diff(row_times)
    source_dt = float(np.median(source_intervals)) if source_intervals.size else None
    source_hz = 1.0 / source_dt if source_dt is not None and source_dt > 0.0 else None
    if source_hz is None:
        mode = "single_source_state"
    elif source_hz > float(fps) * (1.0 + 1e-6):
        mode = "downsample"
    elif source_hz < float(fps) * (1.0 - 1e-6):
        mode = "upsample_interpolate"
    else:
        mode = "target_rate_resample"

    resampled = PackedStateTrajectory(
        task_id=decoded.task_id,
        timestep=decoded.timestep,
        steps_per_ctrl=None,
        step_counts=None,
        n_steps=len(target_times),
        nq=decoded.nq,
        nv=decoded.nv,
        nu=decoded.nu,
        initial_snapshot=dict(decoded.initial_snapshot),
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
    )
    metadata = {
        "schema": STATE_RESAMPLING_SCHEMA,
        "timeline_mode": mode,
        "physics_rollout": False,
        "mj_step_calls": 0,
        "target_fps": int(fps),
        "target_frame_period_seconds": 1.0 / float(fps),
        "target_alignment": "terminal_aligned_without_extrapolation",
        "requested_expected_duration_seconds": expected_duration_s,
        "source_timing": corrected_time_summary,
        "source_frame_count": int(decoded.n_steps),
        "source_knot_count": int(len(source_times)),
        "source_state_time_start_seconds": float(source_times[0]),
        "source_state_time_end_seconds": float(source_times[-1]),
        "source_nominal_frame_period_seconds": source_dt,
        "source_nominal_fps": source_hz,
        "source_is_regular": bool(
            len(source_intervals) <= 1
            or np.allclose(source_intervals, source_dt, rtol=1e-6, atol=1e-9)
        ),
        "target_frame_count_before_stride_and_cap": int(len(full_target_times)),
        "output_frame_count": int(len(target_times)),
        "output_time_start_seconds": float(target_times[0]),
        "output_time_end_seconds": float(target_times[-1]),
        "output_duration_seconds": float(len(target_times)) / float(fps),
        "frame_stride_applied_after_resampling": int(frame_stride),
        "max_frames_applied_after_resampling": max_frames,
        "terminal_source_time_included": bool(
            np.isclose(float(target_times[-1]), float(source_times[-1]), atol=1e-8)
        ),
        "interpolated_output_frame_count": int((~exact_source).sum()),
        "exact_source_time_frame_count": int(exact_source.sum()),
        "source_left_knot_indices": left.astype(int).tolist(),
        "source_right_knot_indices": right.astype(int).tolist(),
        "source_knot_index_semantics": (
            "indexes include prepended initial_snapshot when initial_snapshot_used_as_knot=true"
        ),
        "source_interpolation_alpha": alpha.astype(float).tolist(),
        "qvel_method": "mujoco_mj_differentiatePos_of_continuous_qpos",
        "qvel_derivative_probe_seconds": float(derivative_epsilon),
        "ctrl_method": "causal_zero_order_hold",
        "ctrl_source_knot_indices": full_ctrl_source_indices[selected].astype(int).tolist(),
        **knot_metadata,
        **interpolation_metadata,
    }
    return ResampledStateTimeline(
        trajectory=resampled,
        state_times=np.asarray(target_times, dtype=np.float64),
        metadata=metadata,
    )
