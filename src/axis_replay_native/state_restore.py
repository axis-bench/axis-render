"""Apply initial_state and object_state.start to MuJoCo model."""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


def _find_body_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _find_joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _find_free_joint_for_body(model: mujoco.MjModel, body_id: int) -> int | None:
    for joint_id in range(model.njnt):
        if model.jnt_bodyid[joint_id] == body_id and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            return joint_id
    return None


def _set_free_joint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    pos: list[float] | None,
    quat: list[float] | None,
) -> bool:
    joint_id = _find_free_joint_for_body(model, body_id)
    if joint_id is None:
        return False

    addr = model.jnt_qposadr[joint_id]
    if pos is not None and len(pos) >= 3:
        data.qpos[addr:addr + 3] = pos[:3]
    if quat is not None and len(quat) >= 4:
        data.qpos[addr + 3:addr + 7] = quat[:4]
    return True


def _set_joint_scalar_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_id: int, value: float) -> bool:
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
        return False
    addr = model.jnt_qposadr[joint_id]
    data.qpos[addr] = value
    return True


def _zero_dynamic_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0
    data.ctrl[:] = 0.0
    if model.na > 0:
        data.act[:] = 0.0


def _resolve_joint_name_candidates(entity_key: str, joint_key: str) -> list[str]:
    candidates = []
    if "/" in joint_key:
        candidates.append(joint_key)
        return candidates
    if entity_key == "franka":
        candidates.append(f"franka/{joint_key}")
    candidates.append(f"{entity_key}/{joint_key}")
    candidates.append(f"{entity_key}_base/{joint_key}")
    candidates.append(joint_key)
    return candidates


def apply_initial_state(model: mujoco.MjModel, data: mujoco.MjData, initial_state: dict[str, Any]) -> None:
    if not initial_state:
        return

    def apply_entity(entity_key: str, cfg: dict[str, Any]) -> None:
        if not cfg:
            return

        pos = cfg.get("pos")
        quat = cfg.get("rot")
        if pos or quat:
            body_candidates = [f"{entity_key}/", entity_key]
            if entity_key == "franka":
                body_candidates.insert(0, "franka/")
            for body_name in body_candidates:
                body_id = _find_body_id(model, body_name)
                if body_id >= 0:
                    _set_free_joint_pose(model, data, body_id, pos, quat)
                    break

        dof_pos = cfg.get("dof_pos") or cfg.get("dofPos")
        if dof_pos and isinstance(dof_pos, dict):
            for joint_key, value in dof_pos.items():
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue

                candidates = _resolve_joint_name_candidates(entity_key, joint_key)
                applied = False
                for name in candidates:
                    joint_id = _find_joint_id(model, name)
                    if joint_id >= 0:
                        _set_joint_scalar_qpos(model, data, joint_id, numeric_value)
                        applied = True
                        break
                if not applied:
                    logger.warning(
                        "Could not resolve joint %r for entity %r. Tried: %s",
                        joint_key,
                        entity_key,
                        candidates,
                    )

    for key, cfg in (initial_state.get("objects", {}) or {}).items():
        apply_entity(key, cfg)
    for key, cfg in (initial_state.get("robots", {}) or {}).items():
        apply_entity(key, cfg)

    mujoco.mj_forward(model, data)


def apply_object_state(model: mujoco.MjModel, data: mujoco.MjData, object_state: dict[str, Any]) -> None:
    if not object_state:
        return

    positions = object_state.get("object_positions", {})
    orientations = object_state.get("object_orientations", {})
    all_names = set(positions.keys()) | set(orientations.keys())

    for body_name in all_names:
        if body_name == "world":
            continue
        body_id = _find_body_id(model, body_name)
        if body_id < 0:
            continue
        _set_free_joint_pose(
            model,
            data,
            body_id,
            positions.get(body_name),
            orientations.get(body_name),
        )

    mujoco.mj_forward(model, data)


def apply_state_snapshot(model: mujoco.MjModel, data: mujoco.MjData, snapshot: dict[str, Any]) -> bool:
    if not snapshot:
        return False

    qpos = snapshot.get("qpos")
    qvel = snapshot.get("qvel")
    if qpos is None or qvel is None:
        logger.warning("Initial snapshot missing qpos/qvel; falling back to initial_state restoration.")
        return False

    if len(qpos) != model.nq or len(qvel) != model.nv:
        logger.warning(
            "Initial snapshot shape mismatch (qpos=%s expected %d, qvel=%s expected %d); "
            "falling back to initial_state restoration.",
            len(qpos),
            model.nq,
            len(qvel),
            model.nv,
        )
        return False

    data.qpos[:] = np.asarray(qpos, dtype=float)
    data.qvel[:] = np.asarray(qvel, dtype=float)

    ctrl = snapshot.get("ctrl")
    if ctrl is not None:
        if len(ctrl) == model.nu:
            data.ctrl[:] = np.asarray(ctrl, dtype=float)
        else:
            logger.warning(
                "Initial snapshot ctrl length mismatch (%d expected %d); ignoring ctrl field.",
                len(ctrl),
                model.nu,
            )

    act = snapshot.get("act")
    if act is not None:
        if len(act) == model.na:
            data.act[:] = np.asarray(act, dtype=float)
        else:
            logger.warning(
                "Initial snapshot act length mismatch (%d expected %d); ignoring act field.",
                len(act),
                model.na,
            )

    qacc_warmstart = snapshot.get("qacc_warmstart")
    if qacc_warmstart is not None:
        if len(qacc_warmstart) == model.nv:
            data.qacc_warmstart[:] = np.asarray(qacc_warmstart, dtype=float)
        else:
            logger.warning(
                "Initial snapshot qacc_warmstart length mismatch (%d expected %d); ignoring qacc_warmstart.",
                len(qacc_warmstart),
                model.nv,
            )

    qfrc_applied = snapshot.get("qfrc_applied")
    if qfrc_applied is not None:
        if len(qfrc_applied) == model.nv:
            data.qfrc_applied[:] = np.asarray(qfrc_applied, dtype=float)
        else:
            logger.warning(
                "Initial snapshot qfrc_applied length mismatch (%d expected %d); ignoring qfrc_applied.",
                len(qfrc_applied),
                model.nv,
            )

    xfrc_applied = snapshot.get("xfrc_applied")
    if xfrc_applied is not None:
        expected = model.nbody * 6
        if len(xfrc_applied) == expected:
            data.xfrc_applied[:] = np.asarray(xfrc_applied, dtype=float).reshape(model.nbody, 6)
        else:
            logger.warning(
                "Initial snapshot xfrc_applied length mismatch (%d expected %d); ignoring xfrc_applied.",
                len(xfrc_applied),
                expected,
            )

    userdata = snapshot.get("userdata")
    if userdata is not None:
        if len(userdata) == model.nuserdata:
            data.userdata[:] = np.asarray(userdata, dtype=float)
        else:
            logger.warning(
                "Initial snapshot userdata length mismatch (%d expected %d); ignoring userdata.",
                len(userdata),
                model.nuserdata,
            )

    mocap_pos = snapshot.get("mocap_pos")
    if mocap_pos is not None:
        expected = model.nmocap * 3
        if len(mocap_pos) == expected:
            data.mocap_pos[:] = np.asarray(mocap_pos, dtype=float).reshape(model.nmocap, 3)
        else:
            logger.warning(
                "Initial snapshot mocap_pos length mismatch (%d expected %d); ignoring mocap_pos.",
                len(mocap_pos),
                expected,
            )

    mocap_quat = snapshot.get("mocap_quat")
    if mocap_quat is not None:
        expected = model.nmocap * 4
        if len(mocap_quat) == expected:
            data.mocap_quat[:] = np.asarray(mocap_quat, dtype=float).reshape(model.nmocap, 4)
        else:
            logger.warning(
                "Initial snapshot mocap_quat length mismatch (%d expected %d); ignoring mocap_quat.",
                len(mocap_quat),
                expected,
            )

    if "time" in snapshot:
        try:
            data.time = float(snapshot["time"])
        except (TypeError, ValueError):
            logger.warning("Initial snapshot time is invalid; ignoring time field.")

    mujoco.mj_forward(model, data)
    return True


def restore_scene_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_state: dict[str, Any] | None,
    object_state_start: dict[str, Any] | None,
    initial_snapshot: dict[str, Any] | None = None,
) -> None:
    if apply_state_snapshot(model, data, initial_snapshot or {}):
        return
    if initial_state:
        apply_initial_state(model, data, initial_state)
    if object_state_start:
        apply_object_state(model, data, object_state_start)
    _zero_dynamic_state(model, data)
    mujoco.mj_forward(model, data)
