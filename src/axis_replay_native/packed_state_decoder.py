"""Decode compact packed_state_v1 trajectories."""

from __future__ import annotations

import base64
import gzip
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PackedStateTrajectory:
    task_id: int | None
    timestep: float
    steps_per_ctrl: int | None
    step_counts: list[int] | None
    n_steps: int
    nq: int
    nv: int
    nu: int
    initial_snapshot: dict[str, Any]
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray

    @property
    def row_width(self) -> int:
        return self.nq + self.nv + self.nu


def _decode_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise ValueError("initial_snapshot must be an object.")

    decoded: dict[str, Any] = {}
    for key in (
        "qpos",
        "qvel",
        "ctrl",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "userdata",
        "mocap_pos",
        "mocap_quat",
    ):
        value = snapshot.get(key)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"initial_snapshot.{key} must be an array when present.")
        decoded[key] = [float(item) for item in value]

    time_value = snapshot.get("time")
    if time_value is not None:
        decoded["time"] = float(time_value)

    if "qpos" not in decoded or "qvel" not in decoded:
        raise ValueError("initial_snapshot must include qpos and qvel.")

    return decoded


def _decode_payload_bytes(encoded: str) -> bytes:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("data_b64_gzip must be a non-empty base64 string.")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("data_b64_gzip is not valid base64.") from exc

    try:
        return gzip.decompress(compressed)
    except Exception as exc:
        raise ValueError("data_b64_gzip is not valid gzip data.") from exc


def decode_packed_state_v1(trajectory_data: dict[str, Any]) -> PackedStateTrajectory:
    fmt = trajectory_data.get("format")
    if fmt != "packed_state_v1":
        raise ValueError(f"Unsupported format: {fmt!r}. Expected 'packed_state_v1'.")

    try:
        n_steps = int(trajectory_data.get("n_steps"))
        nq = int(trajectory_data.get("nq"))
        nv = int(trajectory_data.get("nv"))
        nu = int(trajectory_data.get("nu"))
        timestep = float(trajectory_data.get("timestep"))
    except (TypeError, ValueError) as exc:
        raise ValueError("packed_state_v1 must include numeric timestep, n_steps, nq, nv, and nu.") from exc

    if n_steps < 1 or nq < 1 or nv < 1 or nu < 0:
        raise ValueError("packed_state_v1 dimensions are invalid.")

    steps_per_ctrl_raw = trajectory_data.get("steps_per_ctrl")
    steps_per_ctrl: int | None = None
    if steps_per_ctrl_raw is not None:
        try:
            steps_per_ctrl = int(steps_per_ctrl_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"steps_per_ctrl must be an integer, got {steps_per_ctrl_raw!r}") from exc
        if steps_per_ctrl < 1:
            raise ValueError("steps_per_ctrl must be >= 1 when present.")

    step_counts_raw = trajectory_data.get("step_counts")
    step_counts: list[int] | None = None
    if step_counts_raw is not None:
        if not isinstance(step_counts_raw, list):
            raise ValueError("step_counts must be an array when present.")
        step_counts = []
        for index, item in enumerate(step_counts_raw):
            try:
                count = int(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"step_counts[{index}] is not an integer.") from exc
            if count < 1:
                raise ValueError(f"step_counts[{index}] must be >= 1.")
            step_counts.append(count)
        if len(step_counts) != max(0, n_steps - 1):
            raise ValueError(
                f"step_counts length ({len(step_counts)}) must equal n_steps - 1 ({max(0, n_steps - 1)})."
            )

    initial_snapshot = _decode_snapshot(trajectory_data.get("initial_snapshot"))

    payload = np.frombuffer(
        _decode_payload_bytes(trajectory_data.get("data_b64_gzip")),
        dtype=np.float32,
    )
    row_width = nq + nv + nu
    expected_values = n_steps * row_width
    if payload.size != expected_values:
        raise ValueError(f"packed_state_v1 payload has {payload.size} float32 values; expected {expected_values}.")

    rows = payload.reshape(n_steps, row_width)
    qpos = rows[:, :nq].astype(np.float64, copy=True)
    qvel = rows[:, nq : nq + nv].astype(np.float64, copy=True)
    ctrl = rows[:, nq + nv :].astype(np.float64, copy=True) if nu else np.zeros((n_steps, 0), dtype=np.float64)

    if len(initial_snapshot["qpos"]) != nq or len(initial_snapshot["qvel"]) != nv:
        raise ValueError("initial_snapshot qpos/qvel dimensions do not match packed_state_v1 header.")
    initial_ctrl = initial_snapshot.get("ctrl")
    if initial_ctrl is not None and len(initial_ctrl) != nu:
        raise ValueError("initial_snapshot ctrl dimension does not match packed_state_v1 header.")

    task_id_raw = trajectory_data.get("task_id")
    task_id = int(task_id_raw) if task_id_raw is not None else None

    return PackedStateTrajectory(
        task_id=task_id,
        timestep=timestep,
        steps_per_ctrl=steps_per_ctrl,
        step_counts=step_counts,
        n_steps=n_steps,
        nq=nq,
        nv=nv,
        nu=nu,
        initial_snapshot=initial_snapshot,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
    )
