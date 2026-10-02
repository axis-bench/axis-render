"""Direct MuJoCo state replay for packed_state_v1 trajectories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .checker_payload_extractor import (
    build_checker_payload,
    collect_pressed_keys,
    extract_initial_orientations,
    extract_initial_positions,
)
from .checkers.CheckerManager import CheckerManager
from .mujoco_replayer import _collect_body_states, _compare_with_expected
from .packed_state_decoder import PackedStateTrajectory, decode_packed_state_v1
from .replay_trace import JsonlTraceWriter, collect_body_pose_map, collect_joint_qpos_map
from .types import ReplayResult

logger = logging.getLogger(__name__)


def _validate_dimensions(model: mujoco.MjModel, decoded: PackedStateTrajectory) -> None:
    if model.nq != decoded.nq:
        raise ValueError(f"Model nq={model.nq} does not match trajectory nq={decoded.nq}.")
    if model.nv != decoded.nv:
        raise ValueError(f"Model nv={model.nv} does not match trajectory nv={decoded.nv}.")
    if model.nu != decoded.nu:
        raise ValueError(f"Model nu={model.nu} does not match trajectory nu={decoded.nu}.")


def _set_row(data: mujoco.MjData, decoded: PackedStateTrajectory, index: int, state_time: float) -> None:
    data.qpos[:] = decoded.qpos[index]
    data.qvel[:] = decoded.qvel[index]
    if decoded.nu:
        data.ctrl[:] = decoded.ctrl[index]
    data.time = float(state_time)


def _initial_snapshot_matches_first_row(decoded: PackedStateTrajectory) -> bool:
    qpos0 = np.asarray(decoded.initial_snapshot.get("qpos") or [], dtype=np.float64)
    qvel0 = np.asarray(decoded.initial_snapshot.get("qvel") or [], dtype=np.float64)
    if qpos0.shape != decoded.qpos[0].shape or qvel0.shape != decoded.qvel[0].shape:
        return False
    return bool(np.allclose(qpos0, decoded.qpos[0], atol=1e-5, rtol=1e-5) and np.allclose(qvel0, decoded.qvel[0], atol=1e-5, rtol=1e-5))


def _build_state_times(
    decoded: PackedStateTrajectory,
    *,
    expected_duration_s: float | None = None,
) -> tuple[list[float], dict[str, Any]]:
    start_time = float(decoded.initial_snapshot.get("time", 0.0) or 0.0)
    if decoded.step_counts is not None:
        times = [start_time]
        running_time = start_time
        for count in decoded.step_counts:
            running_time += float(count) * float(decoded.timestep)
            times.append(running_time)
        mode = "step_counts"
    else:
        state_dt = float(decoded.timestep) * float(decoded.steps_per_ctrl or 1)
        offset = 0
        mode = "steps_per_ctrl_initial_state"
        if expected_duration_s is not None:
            target_duration = max(0.0, float(expected_duration_s) - start_time)
            duration_with_initial = max(0, decoded.n_steps - 1) * state_dt
            duration_post_step = decoded.n_steps * state_dt
            tolerance = max(1e-6, 0.5 * state_dt)
            if abs(target_duration - duration_post_step) <= tolerance:
                offset = 1
                mode = "steps_per_ctrl_post_step"
            elif abs(target_duration - duration_with_initial) <= tolerance:
                offset = 0
                mode = "steps_per_ctrl_initial_state"
            elif not _initial_snapshot_matches_first_row(decoded):
                offset = 1
                mode = "steps_per_ctrl_post_step"
        elif not _initial_snapshot_matches_first_row(decoded):
            offset = 1
            mode = "steps_per_ctrl_post_step"
        times = [start_time + float(index + offset) * state_dt for index in range(decoded.n_steps)]

    intervals = np.diff(np.asarray(times, dtype=np.float64)) if len(times) > 1 else np.asarray([], dtype=np.float64)
    positive_intervals = intervals[intervals > 1e-12]
    median_dt = float(np.median(positive_intervals)) if positive_intervals.size else None
    summary = {
        "state_time_mode": mode,
        "state_time_start": float(times[0]),
        "state_time_end": float(times[-1]),
        "state_dt_median": median_dt,
        "expected_duration_s": expected_duration_s,
    }
    return times, summary


def _infer_video_fps(state_times: list[float], requested_fps: float | None) -> float:
    if requested_fps is not None:
        return float(requested_fps)
    return 30.0


def _build_video_schedule(
    state_times: list[float],
    *,
    fps: float,
    mode: str,
    expected_duration_s: float | None = None,
) -> tuple[list[int], float]:
    if fps <= 0:
        raise ValueError(f"video fps must be positive, got {fps!r}.")
    if not state_times:
        return [], 0.0
    if mode == "fast":
        return list(range(len(state_times))), float(len(state_times)) / float(fps)
    if mode != "realtime":
        raise ValueError(f"Unsupported video playback mode {mode!r}; expected 'fast' or 'realtime'.")

    state_array = np.asarray(state_times, dtype=np.float64)
    playback_start = 0.0 if state_array[0] > 0.0 else float(state_array[0])
    if expected_duration_s is not None and expected_duration_s > 0:
        duration = float(expected_duration_s)
    else:
        duration = max(0.0, float(state_array[-1]) - playback_start)
    frame_count = max(1, int(round(duration * fps)))
    frame_times = playback_start + np.arange(frame_count, dtype=np.float64) / float(fps)

    right = np.searchsorted(state_array, frame_times, side="left")
    right = np.clip(right, 0, len(state_array) - 1)
    left = np.clip(right - 1, 0, len(state_array) - 1)
    choose_right = np.abs(state_array[right] - frame_times) < np.abs(frame_times - state_array[left])
    indexes = np.where(choose_right, right, left).astype(int)
    return indexes.tolist(), duration


def _robot_body_name(name: str) -> bool:
    lower = name.lower()
    return (
        lower.startswith("franka")
        or lower.startswith("panda")
        or "/panda_" in lower
        or "panda_" in lower
    )


def _focus_body_ids(model: mujoco.MjModel) -> list[int]:
    object_ids: list[int] = []
    end_effector_ids: list[int] = []
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name or ""
        lower = name.lower()
        if "panda_hand" in lower or "finger" in lower:
            end_effector_ids.append(body_id)
        elif not _robot_body_name(name):
            object_ids.append(body_id)
    return object_ids + end_effector_ids or list(range(1, model.nbody))


def build_closeup_camera(model, data, decoded, state_times):
    """Public wrapper: frame a close-up from a model's contents, so a caller can reuse it."""
    return _build_closeup_camera(model, data, decoded, state_times)


def _build_closeup_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    decoded: PackedStateTrajectory,
    state_times: list[float],
) -> mujoco.MjvCamera:
    body_ids = _focus_body_ids(model)
    if decoded.n_steps <= 80:
        sample_indexes = list(range(decoded.n_steps))
    else:
        sample_indexes = sorted({int(value) for value in np.linspace(0, decoded.n_steps - 1, 80)})

    positions: list[np.ndarray] = []
    for index in sample_indexes:
        _set_row(data, decoded, index, state_times[index])
        mujoco.mj_forward(model, data)
        positions.append(np.asarray(data.xpos[body_ids], dtype=np.float64).copy())

    if positions:
        all_positions = np.concatenate(positions, axis=0)
    else:
        all_positions = np.asarray([[0.5, 0.0, 0.05]], dtype=np.float64)
    bbox_min = np.min(all_positions, axis=0)
    bbox_max = np.max(all_positions, axis=0)
    center = (bbox_min + bbox_max) * 0.5
    center[2] = max(float(center[2]), 0.06)

    span = np.maximum(bbox_max - bbox_min, 1e-6)
    horizontal_span = float(max(span[0], span[1]))
    vertical_span = float(span[2])
    distance = max(0.55, min(1.25, max(horizontal_span * 1.7, vertical_span * 3.5, 0.5)))

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center.tolist()
    camera.distance = float(distance)
    camera.azimuth = 135.0
    camera.elevation = -35.0
    return camera


class _VideoSink:
    def __init__(self, video_path: str, *, fps: float) -> None:
        self.out_path = Path(video_path).expanduser()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.frame_count = 0
        self._writer = None
        self._frames: list[np.ndarray] | None = None
        try:
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(str(self.out_path), fps=self.fps, macro_block_size=1)
        except ImportError:
            self._frames = []

    def append(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.append_data(frame)
        else:
            assert self._frames is not None
            self._frames.append(frame.copy())
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        elif self._frames is not None:
            try:
                import mediapy
            except ImportError as exc:
                raise RuntimeError(
                    "Video output requires imageio or mediapy."
                ) from exc
            mediapy.write_video(str(self.out_path), self._frames, fps=self.fps)
        logger.info("State replay video saved: %s (%d frames, %.3g fps)", self.out_path.resolve(), self.frame_count, self.fps)


def replay_packed_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory_data: dict[str, Any],
    *,
    trace_writer: JsonlTraceWriter | None = None,
    video_path: str | None = None,
    video_fps: float | None = None,
    video_playback_mode: str = "fast",
    video_width: int = 960,
    video_height: int = 540,
    expected_duration_s: float | None = None,
    camera: mujoco.MjvCamera | None = None,
) -> tuple[ReplayResult, PackedStateTrajectory]:
    decoded = decode_packed_state_v1(trajectory_data)
    _validate_dimensions(model, decoded)
    state_times, time_summary = _build_state_times(decoded, expected_duration_s=expected_duration_s)

    renderer = None
    # A caller-supplied camera survives; otherwise one is framed from this model's own contents
    # below. Sharing a camera is what lets a recording and a retarget of it be compared frame by
    # frame, since the two are different models and would otherwise be framed differently.
    video_state_indexes: list[int] = []
    video_duration_s = 0.0
    video_frame_count = 0
    output_video_fps = _infer_video_fps(state_times, video_fps)
    if video_path:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, video_width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, video_height)
        renderer = mujoco.Renderer(model, height=video_height, width=video_width)
        if camera is None:
            camera = _build_closeup_camera(model, data, decoded, state_times)
        video_state_indexes, video_duration_s = _build_video_schedule(
            state_times,
            fps=output_video_fps,
            mode=video_playback_mode,
            expected_duration_s=expected_duration_s,
        )

    for index in range(decoded.n_steps):
        _set_row(data, decoded, index, state_times[index])
        mujoco.mj_forward(model, data)

        if trace_writer is not None:
            trace_writer.write(
                "state",
                {
                    "source": "packed_state_v1_direct",
                    "step_index": index,
                    "sim_time": float(data.time),
                    "joint_qpos": collect_joint_qpos_map(model, data),
                    "body_pose": collect_body_pose_map(model, data),
                },
            )

    if renderer is not None:
        sink = _VideoSink(str(video_path), fps=output_video_fps)
        try:
            for index in video_state_indexes:
                _set_row(data, decoded, index, state_times[index])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera if camera is not None else -1)
                sink.append(renderer.render())
            video_frame_count = sink.frame_count
        finally:
            sink.close()
            renderer.close()

    _set_row(data, decoded, decoded.n_steps - 1, state_times[-1])
    mujoco.mj_forward(model, data)

    final_states = _collect_body_states(model, data)
    object_state = trajectory_data.get("object_state", {}) or {}
    pos_errors, rot_errors, error_msgs = _compare_with_expected(final_states, object_state.get("end"))
    if trace_writer is not None:
        trace_writer.write(
            "final",
            {
                "source": "packed_state_v1_direct",
                "final_sim_time": float(data.time),
                "total_state_frames": decoded.n_steps,
                "state_time_mode": time_summary["state_time_mode"],
                "position_errors": pos_errors,
                "rotation_errors_deg": rot_errors,
                "error_report": error_msgs,
            },
        )

    return (
        ReplayResult(
            success=True,
            final_body_states=final_states,
            error_report=error_msgs,
            position_errors=pos_errors,
            rotation_errors_deg=rot_errors,
            total_steps_executed=int(decoded.n_steps),
            final_sim_time=float(data.time),
            timing_summary={
                "timing_mode": "packed_state_v1_direct",
                "raw_step_count": int(decoded.n_steps),
                "cleaned_step_count": int(decoded.n_steps),
                "removed_zero_dt_steps": 0,
                "mj_dt": float(decoded.timestep),
                "steps_per_ctrl": decoded.steps_per_ctrl,
                "step_counts_present": decoded.step_counts is not None,
                "state_time_mode": time_summary["state_time_mode"],
                "state_time_start": time_summary["state_time_start"],
                "state_time_end": time_summary["state_time_end"],
                "state_dt_median": time_summary["state_dt_median"],
                "expected_duration_s": time_summary["expected_duration_s"],
                "video_frame_count": video_frame_count,
                "video_fps": output_video_fps if video_path else None,
                "video_duration_s": video_duration_s if video_path else None,
                "video_playback_mode": video_playback_mode if video_path else None,
            },
        ),
        decoded,
    )


def run_checker_on_final_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    decoded: PackedStateTrajectory,
    trajectory_data: dict[str, Any],
    checker_config: dict[str, Any] | None,
    *,
    initial_state: dict[str, Any] | None = None,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    if not checker_config:
        return False, [], {"warning": "missing checker_config"}

    object_state = trajectory_data.get("object_state", {}) or {}
    compat_trajectory = type(
        "StateReplayCheckerTrajectory",
        (),
        {
            "object_state_start": object_state.get("start"),
            "keyboard_names": [],
            "steps": [],
        },
    )()
    pressed_keys = collect_pressed_keys(compat_trajectory)
    initial_positions = extract_initial_positions(model, data, object_state.get("start"), initial_state)
    initial_orientations = extract_initial_orientations(model, data, object_state.get("start"), initial_state)
    payload = build_checker_payload(
        model,
        data,
        compat_trajectory,
        initial_state=initial_state,
        initial_positions=initial_positions,
        initial_orientations=initial_orientations,
        pressed_keys=pressed_keys,
    )
    manager = CheckerManager(checker_config)
    passed, details = manager.check_with_details(payload)
    diagnostics = {
        "object_count": len(payload.object_positions),
        "joint_count": len(payload.joint_positions or {}),
        "site_count": len(payload.site_positions or {}),
        "pressed_keys_count": len(pressed_keys),
    }
    return passed, details, diagnostics
