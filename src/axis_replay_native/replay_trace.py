"""Optional structured replay traces for cross-implementation diffing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


class JsonlTraceWriter:
    def __init__(self, path: str | None):
        self.path = path
        self._file = None
        if path:
            trace_path = Path(path).expanduser()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = trace_path.open("w", encoding="utf-8")

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if self._file is None:
            return
        record = {"event": event, **_to_jsonable(payload)}
        json.dump(record, self._file, ensure_ascii=True)
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def _joint_qpos_size(joint_type: int) -> int:
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def collect_joint_qpos_map(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float | list[float]]:
    joint_qpos: dict[str, float | list[float]] = {}
    for joint_id in range(model.njnt):
        joint = model.joint(joint_id)
        if not joint.name:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos_size = _joint_qpos_size(int(model.jnt_type[joint_id]))
        qpos_slice = data.qpos[qpos_adr : qpos_adr + qpos_size].tolist()
        joint_qpos[joint.name] = qpos_slice[0] if qpos_size == 1 else qpos_slice
    return joint_qpos


def collect_body_pose_map(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, dict[str, list[float]]]:
    body_pose: dict[str, dict[str, list[float]]] = {}
    for body_id in range(model.nbody):
        body = model.body(body_id)
        if not body.name:
            continue
        body_pose[body.name] = {
            "pos": data.xpos[body_id].tolist(),
            "quat": data.xquat[body_id].tolist(),
        }
    return body_pose
