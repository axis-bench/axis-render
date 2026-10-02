"""Load MuJoCo model from materialized assets."""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco

logger = logging.getLogger(__name__)

FRANKA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]


def load_model(work_dir: str, entry_xml_filename: str = "scene.xml") -> tuple[mujoco.MjModel, mujoco.MjData]:
    entry_path = Path(work_dir) / entry_xml_filename
    if not entry_path.exists():
        raise FileNotFoundError(f"Entry XML not found: {entry_path}")

    logger.info("Loading MuJoCo model from %s", entry_path)
    model = mujoco.MjModel.from_xml_path(str(entry_path))
    data = mujoco.MjData(model)
    logger.info(
        "Model loaded: %d bodies, %d joints, %d actuators, timestep=%.6f",
        model.nbody,
        model.njnt,
        model.nu,
        model.opt.timestep,
    )
    return model, data


def get_body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""


def get_actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id) or ""


def _normalize_name_for_match(name: str) -> str:
    return str(name).replace("__axis_scope__", "/").rstrip("/").split("/")[-1]


def _remap_legacy_control_names(control_names: list[str]) -> list[str]:
    if (
        control_names
        and all(str(name).startswith("joint_") for name in control_names)
        and len(control_names) <= len(FRANKA_JOINT_NAMES)
    ):
        return FRANKA_JOINT_NAMES[: len(control_names)]
    return [str(name) for name in control_names]


def build_actuator_id_map(model: mujoco.MjModel, control_names: list[str]) -> dict[str, int]:
    resolved_control_names = _remap_legacy_control_names(control_names)
    name_to_id: dict[str, int] = {}
    suffix_to_ids: dict[str, list[int]] = {}
    for i in range(model.nu):
        name = get_actuator_name(model, i)
        if name:
            name_to_id[name] = i
            suffix_to_ids.setdefault(_normalize_name_for_match(name), []).append(i)

    result: dict[str, int] = {}
    for original_name, ctrl_name in zip(control_names, resolved_control_names):
        if ctrl_name in name_to_id:
            result[original_name] = name_to_id[ctrl_name]
            continue
        matches = suffix_to_ids.get(_normalize_name_for_match(ctrl_name), [])
        if len(matches) == 1:
            result[original_name] = matches[0]
            continue
        logger.warning("Actuator not found for control name: %s", original_name)
    return result
