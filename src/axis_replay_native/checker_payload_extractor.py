"""Extract CheckerRuntimePayload fields from MuJoCo simulation state.

Bridges the replay module to the checker module by collecting all
fields defined in CheckerRuntimePayload from MjModel/MjData and
DecodedTrajectory.
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

from .asset_materializer import SCOPE_TOKEN
from .checkers.runtime_types import CheckerRuntimePayload
from .mujoco_model_loader import get_body_name
from .types import DecodedTrajectory

logger = logging.getLogger(__name__)
_LOGGED_MISSING_SIGNATURES: set[tuple[str, ...]] = set()


def _wxyz_to_xyzw(q: list[float] | np.ndarray) -> list[float]:
    """Convert quaternion from MuJoCo [w,x,y,z] to checker [x,y,z,w]."""
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _with_desanitized_aliases(mapping: dict[str, Any]) -> dict[str, Any]:
    """Expose sanitized generated-task names under their original slash form too."""
    if not mapping:
        return mapping

    result = dict(mapping)
    for key, value in list(mapping.items()):
        alias = key.replace(SCOPE_TOKEN, "/")
        if alias != key and alias not in result:
            result[alias] = value
    return result


def _get_site_name(model: mujoco.MjModel, site_id: int) -> str:
    """Get site name by ID."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""


def _get_joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    """Get joint name by ID."""
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""


def extract_object_positions(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict[str, list[float]]:
    """Extract body centre-of-mass positions: {body_name: [x, y, z]}."""
    positions: dict[str, list[float]] = {}
    for i in range(model.nbody):
        name = get_body_name(model, i)
        if name and name != "world":
            positions[name] = data.xpos[i].tolist()
    return _with_desanitized_aliases(positions)


def extract_object_orientations(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict[str, list[float]]:
    """Extract body orientations as [x,y,z,w] quaternions."""
    orientations: dict[str, list[float]] = {}
    for i in range(model.nbody):
        name = get_body_name(model, i)
        if name and name != "world":
            orientations[name] = _wxyz_to_xyzw(data.xquat[i])
    return _with_desanitized_aliases(orientations)


def extract_joint_positions(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict[str, float]:
    """Extract joint positions: {joint_name: qpos_value}."""
    joints: dict[str, float] = {}
    for j in range(model.njnt):
        name = _get_joint_name(model, j)
        if not name:
            continue
        # Only scalar joints (hinge, slide); skip free joints (7 DOF)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        addr = model.jnt_qposadr[j]
        joints[name] = float(data.qpos[addr])
    return _with_desanitized_aliases(joints)


def extract_site_positions(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict[str, list[float]]:
    """Extract site positions: {site_name: [x, y, z]}."""
    sites: dict[str, list[float]] = {}
    for i in range(model.nsite):
        name = _get_site_name(model, i)
        if name:
            sites[name] = data.site_xpos[i].tolist()
    return _with_desanitized_aliases(sites)


def extract_site_rot_mats(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict[str, list[float]]:
    """Extract site rotation matrices (3x3 flattened, row-major)."""
    mats: dict[str, list[float]] = {}
    for i in range(model.nsite):
        name = _get_site_name(model, i)
        if name:
            mats[name] = data.site_xmat[i].tolist()
    return _with_desanitized_aliases(mats)


def extract_body_rot_mats(
    model: mujoco.MjModel, data: mujoco.MjData,
) -> dict[str, list[float]]:
    """Extract body rotation matrices (3x3 flattened, row-major)."""
    mats: dict[str, list[float]] = {}
    for i in range(model.nbody):
        name = get_body_name(model, i)
        if name and name != "world":
            mats[name] = data.xmat[i].tolist()
    return _with_desanitized_aliases(mats)


def collect_pressed_keys(trajectory: DecodedTrajectory) -> set[str]:
    """Accumulate all pressed keys across the entire trajectory.

    For each step, keyboard_down_indices maps to keyboard_names.
    The union of all mapped key names is returned.
    """
    keyboard_names = trajectory.keyboard_names
    if not keyboard_names:
        return set()

    pressed: set[str] = set()
    for step in trajectory.steps:
        for idx in step.keyboard_down_indices:
            if 0 <= idx < len(keyboard_names):
                pressed.add(keyboard_names[idx])
    return pressed


def extract_initial_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_state_start: dict[str, Any] | None,
    initial_state: dict[str, Any] | None,
) -> dict[str, list[float]]:
    """Extract initial object positions.

    Priority: object_state_start > initial_state > current data snapshot.
    """
    # Priority 1: object_state_start
    if object_state_start:
        positions = object_state_start.get("object_positions")
        if positions:
            return {k: list(v[:3]) for k, v in positions.items() if k != "world"}

    # Priority 2: initial_state
    if initial_state:
        positions: dict[str, list[float]] = {}
        for section in ("objects", "robots"):
            entities = initial_state.get(section, {})
            for entity_key, cfg in entities.items():
                if not cfg:
                    continue
                pos = cfg.get("pos")
                if pos and len(pos) >= 3:
                    # Resolve body name using same candidates as state_restore
                    body_name = _resolve_body_name(model, entity_key)
                    if body_name:
                        positions[body_name] = list(pos[:3])
        if positions:
            return positions

    # Fallback: current data (after restore_scene_state, before replay)
    return extract_object_positions(model, data)


def extract_initial_orientations(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_state_start: dict[str, Any] | None,
    initial_state: dict[str, Any] | None,
) -> dict[str, list[float]]:
    """Extract initial object orientations as [x,y,z,w].

    Priority: object_state_start > initial_state > current data snapshot.
    """
    # Priority 1: object_state_start (already in xyzw per object_state convention)
    if object_state_start:
        orientations = object_state_start.get("object_orientations")
        if orientations:
            return {k: list(v[:4]) for k, v in orientations.items() if k != "world"}

    # Priority 2: initial_state (rot is in MuJoCo wxyz, needs conversion)
    if initial_state:
        orientations: dict[str, list[float]] = {}
        for section in ("objects", "robots"):
            entities = initial_state.get(section, {})
            for entity_key, cfg in entities.items():
                if not cfg:
                    continue
                rot = cfg.get("rot")
                if rot and len(rot) >= 4:
                    body_name = _resolve_body_name(model, entity_key)
                    if body_name:
                        # initial_state stores MuJoCo wxyz; convert to xyzw
                        orientations[body_name] = _wxyz_to_xyzw(rot)
        if orientations:
            return orientations

    # Fallback: current data (already converted to xyzw by extract_object_orientations)
    return extract_object_orientations(model, data)


def _resolve_body_name(model: mujoco.MjModel, entity_key: str) -> str | None:
    """Resolve entity_key to an actual MuJoCo body name.

    Uses the same candidate strategy as state_restore.
    """
    candidates = [f"{entity_key}/", entity_key]
    if entity_key == "franka":
        candidates.insert(0, "franka/")

    for name in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return name
    return None


def build_checker_payload(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: DecodedTrajectory,
    initial_state: dict[str, Any] | None = None,
    initial_positions: dict[str, list[float]] | None = None,
    initial_orientations: dict[str, list[float]] | None = None,
    pressed_keys: set[str] | None = None,
) -> CheckerRuntimePayload:
    """Build a complete CheckerRuntimePayload from final simulation state.

    Args:
        model: MuJoCo model after replay.
        data: MuJoCo data after replay (final frame).
        trajectory: Decoded trajectory (used for pressed_keys if not provided).
        initial_state: task.initial_state (used for initial positions/orientations
                       if not pre-computed).
        initial_positions: Pre-computed initial positions (overrides extraction).
        initial_orientations: Pre-computed initial orientations (overrides extraction).
        pressed_keys: Pre-computed pressed keys (overrides extraction from trajectory).
    """
    # Runtime state from final simulation frame
    object_positions = extract_object_positions(model, data)
    object_orientations = extract_object_orientations(model, data)
    joint_positions = extract_joint_positions(model, data)
    site_positions = extract_site_positions(model, data)
    site_rot_mats = extract_site_rot_mats(model, data)
    body_rot_mats = extract_body_rot_mats(model, data)

    # Pressed keys
    if pressed_keys is None:
        pressed_keys = collect_pressed_keys(trajectory)

    # Initial state fields
    if initial_positions is None:
        initial_positions = extract_initial_positions(
            model, data, trajectory.object_state_start, initial_state,
        )
    if initial_orientations is None:
        initial_orientations = extract_initial_orientations(
            model, data, trajectory.object_state_start, initial_state,
        )

    payload = CheckerRuntimePayload(
        object_positions=object_positions,
        object_orientations=object_orientations,
        joint_positions=joint_positions,
        site_positions=site_positions,
        site_rot_mats=site_rot_mats,
        body_rot_mats=body_rot_mats,
        pressed_keys=pressed_keys,
        initial_object_positions=initial_positions,
        initial_object_orientations=initial_orientations,
    )

    # Log diagnostics
    _log_payload_diagnostics(payload)

    return payload


def build_checker_payload_from_wasm(
    wasm_result: Any,
    trajectory: DecodedTrajectory,
    initial_state: dict[str, Any] | None = None,
    initial_positions: dict[str, list[float]] | None = None,
    initial_orientations: dict[str, list[float]] | None = None,
    pressed_keys: set[str] | None = None,
) -> CheckerRuntimePayload:
    """Build CheckerRuntimePayload from WASM replay result (no MjModel/MjData).

    Args:
        wasm_result: WasmReplayResult with body_states, joint_states, site_states, body_rot_mats.
        trajectory: Decoded trajectory (for pressed_keys extraction).
        initial_state: task.initial_state.
        initial_positions: Pre-computed initial positions.
        initial_orientations: Pre-computed initial orientations.
        pressed_keys: Pre-computed pressed keys.
    """
    # Extract positions and orientations from WASM body_states
    # body_states: {name: {pos: [x,y,z], quat: [w,x,y,z]}}
    object_positions: dict[str, list[float]] = {}
    object_orientations: dict[str, list[float]] = {}
    for name, state in wasm_result.body_states.items():
        if name == "world":
            continue
        object_positions[name] = state["pos"]
        # Convert wxyz → xyzw for checker convention
        q = state["quat"]
        object_orientations[name] = [q[1], q[2], q[3], q[0]]

    # Joint positions
    joint_positions = dict(wasm_result.joint_states) if wasm_result.joint_states else {}

    # Site positions and rotation matrices
    site_positions: dict[str, list[float]] = {}
    site_rot_mats: dict[str, list[float]] = {}
    for name, state in (wasm_result.site_states or {}).items():
        site_positions[name] = state["pos"]
        site_rot_mats[name] = state["xmat"]

    # Body rotation matrices
    body_rot_mats = dict(wasm_result.body_rot_mats) if wasm_result.body_rot_mats else {}

    # Pressed keys
    if pressed_keys is None:
        pressed_keys = collect_pressed_keys(trajectory)

    # Initial state fields
    if initial_positions is None:
        obj_start = trajectory.object_state_start
        if obj_start:
            pos_data = obj_start.get("object_positions")
            if pos_data:
                initial_positions = {
                    k: list(v[:3]) for k, v in pos_data.items() if k != "world"
                }
        if not initial_positions:
            initial_positions = object_positions

    if initial_orientations is None:
        obj_start = trajectory.object_state_start
        if obj_start:
            ori_data = obj_start.get("object_orientations")
            if ori_data:
                initial_orientations = {
                    k: list(v[:4]) for k, v in ori_data.items() if k != "world"
                }
        if not initial_orientations:
            initial_orientations = object_orientations

    payload = CheckerRuntimePayload(
        object_positions=object_positions,
        object_orientations=object_orientations,
        joint_positions=joint_positions,
        site_positions=site_positions,
        site_rot_mats=site_rot_mats,
        body_rot_mats=body_rot_mats,
        pressed_keys=pressed_keys,
        initial_object_positions=initial_positions,
        initial_object_orientations=initial_orientations,
    )

    _log_payload_diagnostics(payload)
    return payload


def _as_env_array(value: Any, env_id: int) -> np.ndarray:
    """Convert a tensor-like value to a numpy array for one env."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim > 1:
        arr = arr[env_id]
    return arr


def _quat_wxyz_to_rot_mat_flat(q: list[float] | np.ndarray) -> list[float]:
    """Convert quaternion [w,x,y,z] to a flattened row-major rotation matrix."""
    w, x, y, z = [float(v) for v in q[:4]]
    n = w * w + x * x + y * y + z * z
    if n <= 0.0:
        return np.eye(3, dtype=float).reshape(-1).tolist()
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return [
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy),
    ]


def _initial_from_tensor_context(
    trajectory: DecodedTrajectory,
    initial_state: dict[str, Any] | None,
    current_positions: dict[str, list[float]],
    current_orientations: dict[str, list[float]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    positions: dict[str, list[float]] = {}
    orientations: dict[str, list[float]] = {}

    obj_start = trajectory.object_state_start
    if obj_start:
        pos_data = obj_start.get("object_positions")
        if pos_data:
            positions = {k: list(v[:3]) for k, v in pos_data.items() if k != "world"}
        ori_data = obj_start.get("object_orientations")
        if ori_data:
            orientations = {k: list(v[:4]) for k, v in ori_data.items() if k != "world"}

    if initial_state:
        for section in ("objects", "robots"):
            for entity_key, cfg in (initial_state.get(section, {}) or {}).items():
                if not cfg:
                    continue
                if not positions and (pos := cfg.get("pos")) and len(pos) >= 3:
                    positions[entity_key] = list(pos[:3])
                if not orientations and (rot := cfg.get("rot")) and len(rot) >= 4:
                    orientations[entity_key] = _wxyz_to_xyzw(rot)

    if not positions:
        positions = current_positions
    if not orientations:
        orientations = current_orientations
    return _with_desanitized_aliases(positions), _with_desanitized_aliases(orientations)


def build_checker_payload_from_tensor_state(
    env_states: Any,
    trajectory: DecodedTrajectory,
    *,
    handler: Any | None = None,
    env_id: int = 0,
    initial_state: dict[str, Any] | None = None,
    pressed_keys: set[str] | None = None,
) -> CheckerRuntimePayload:
    """Build CheckerRuntimePayload from MetaSim TensorState.

    Generated wrappers import this helper unconditionally. MuJoCo wrappers use
    ``build_checker_payload`` at runtime, while Isaac-based wrappers pass a
    TensorState here.
    """
    object_positions: dict[str, list[float]] = {}
    object_orientations: dict[str, list[float]] = {}
    body_rot_mats: dict[str, list[float]] = {}
    joint_positions: dict[str, float] = {}

    def add_root(name: str, root_state: Any) -> None:
        root = _as_env_array(root_state, env_id)
        if root.shape[0] < 7:
            return
        pos = root[:3].astype(float).tolist()
        quat = root[3:7].astype(float).tolist()
        object_positions[name] = pos
        object_orientations[name] = _wxyz_to_xyzw(quat)
        body_rot_mats[name] = _quat_wxyz_to_rot_mat_flat(quat)

    def add_body_states(prefix: str, state: Any) -> None:
        body_state = getattr(state, "body_state", None)
        body_names = getattr(state, "body_names", None)
        if body_state is None or not body_names:
            return
        bodies = _as_env_array(body_state, env_id)
        for idx, body_name in enumerate(body_names):
            if idx >= len(bodies):
                break
            body = np.asarray(bodies[idx])
            if body.shape[0] < 7:
                continue
            name = str(body_name or f"{prefix}/body_{idx}")
            pos = body[:3].astype(float).tolist()
            quat = body[3:7].astype(float).tolist()
            object_positions[name] = pos
            object_orientations[name] = _wxyz_to_xyzw(quat)
            body_rot_mats[name] = _quat_wxyz_to_rot_mat_flat(quat)

    def add_joints(entity_name: str, state: Any) -> None:
        joint_pos = getattr(state, "joint_pos", None)
        if joint_pos is None:
            return
        values = _as_env_array(joint_pos, env_id).reshape(-1)
        joint_names: list[str] | None = None
        if handler is not None:
            get_joint_names = getattr(handler, "_get_joint_names", None)
            if callable(get_joint_names):
                try:
                    joint_names = list(get_joint_names(entity_name, sort=True))
                except Exception:
                    joint_names = None
        for idx, value in enumerate(values):
            name = joint_names[idx] if joint_names and idx < len(joint_names) else f"{entity_name}/joint_{idx}"
            joint_positions[str(name)] = float(value)

    for name, state in getattr(env_states, "objects", {}).items():
        add_root(str(name), getattr(state, "root_state", None))
        add_body_states(str(name), state)
        add_joints(str(name), state)

    for name, state in getattr(env_states, "robots", {}).items():
        add_root(str(name), getattr(state, "root_state", None))
        add_body_states(str(name), state)
        add_joints(str(name), state)

    object_positions = _with_desanitized_aliases(object_positions)
    object_orientations = _with_desanitized_aliases(object_orientations)
    body_rot_mats = _with_desanitized_aliases(body_rot_mats)
    joint_positions = _with_desanitized_aliases(joint_positions)

    if pressed_keys is None:
        pressed_keys = collect_pressed_keys(trajectory)
    initial_positions, initial_orientations = _initial_from_tensor_context(
        trajectory,
        initial_state,
        object_positions,
        object_orientations,
    )

    payload = CheckerRuntimePayload(
        object_positions=object_positions,
        object_orientations=object_orientations,
        joint_positions=joint_positions,
        site_positions={},
        site_rot_mats={},
        body_rot_mats=body_rot_mats,
        pressed_keys=pressed_keys,
        initial_object_positions=initial_positions,
        initial_object_orientations=initial_orientations,
    )
    _log_payload_diagnostics(payload)
    return payload


def _log_payload_diagnostics(payload: CheckerRuntimePayload) -> None:
    """Log summary of payload fields for debugging."""
    fields = {
        "object_positions": len(payload.object_positions),
        "object_orientations": len(payload.object_orientations or {}),
        "joint_positions": len(payload.joint_positions or {}),
        "site_positions": len(payload.site_positions or {}),
        "site_rot_mats": len(payload.site_rot_mats or {}),
        "body_rot_mats": len(payload.body_rot_mats or {}),
        "pressed_keys": len(payload.pressed_keys or set()),
        "initial_object_positions": len(payload.initial_object_positions or {}),
        "initial_object_orientations": len(payload.initial_object_orientations or {}),
    }

    missing = [k for k, v in fields.items() if v == 0]
    if missing:
        signature = tuple(sorted(missing))
        if signature not in _LOGGED_MISSING_SIGNATURES:
            _LOGGED_MISSING_SIGNATURES.add(signature)
            logger.warning("Payload fields with zero entries: %s", missing)

    logger.debug(
        "CheckerRuntimePayload built: %s",
        ", ".join(f"{k}={v}" for k, v in fields.items()),
    )
