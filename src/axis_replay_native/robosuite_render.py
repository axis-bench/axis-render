"""Robosuite / MuJoCo state replay renderer for AXIS packed_state_v1 data."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
import traceback
import urllib.parse
import xml.etree.ElementTree as ET
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from robosuite.environments.base import MujocoEnv
from robosuite.models.arenas import TableArena
from robosuite.utils import binding_utils as robosuite_binding_utils
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.mjmod import CameraModder, LightingModder, TextureModder

from .asset_materializer import materialize_scene
from .asset_store import HttpAssetStore, LocalAssetStore
from .lerobot_atomic import (
    ATOMIC_COMMIT_SCHEMA,
    EpisodeIdentity,
    StreamingEpisodeShardWriter,
    atomic_write_json,
    commit_episode_shard,
    commit_episode_shards,
    load_committed_attempt,
    prepare_episode_shard,
)
from .packed_state_decoder import PackedStateTrajectory, decode_packed_state_v1
from .robosuite_visual_scene import apply_visual_scene, resolve_visual_scene_profile
from .state_mujoco_replayer import (
    _build_state_times,
    _set_row,
    _validate_dimensions,
)
from .state_resampler import STATE_RESAMPLING_SCHEMA, resample_packed_state_to_fps
from .task_runtime import runtime_metadata_from_bundle

logger = logging.getLogger(__name__)

ARENA_PREFIX = "robosuite_arena_"
# MuJoCo stores znear / zfar as fractions of model extent; these meter values
# are resolved after compilation and applied to every embodiment. Legacy
# per-embodiment clipping fields stay in config for fingerprint compatibility
# but do not override these defaults.
DEFAULT_CAMERA_CLIPPING_NEAR_M = 0.005
DEFAULT_CAMERA_CLIPPING_FAR_M = 100.0
RENDER_FINGERPRINT_SCHEMA_VERSION = (
    "robosuite_render_profile_v8_joint_aware_state_resampling_visual_only_rgb"
)
NORMALIZED_RENDER_FPS = 15
DEFAULT_RENDER_WIDTH = 640
DEFAULT_RENDER_HEIGHT = 360
LEROBOT_POLICY_SCHEMA = "reference_policy_fields_v2"
CAMERA_CAPTURE_MULTIPLEX_SCHEMA = "fixed_plus_randomized_camera_pairs_v1"
CAMERA_CAPTURE_MANIFEST_SCHEMA = "axis_camera_capture_manifest_v1"
PROXY_ENV_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
XML_FILE_ATTR_TAGS = {"include", "mesh", "texture", "hfield"}
DEFAULT_TABLE_TOP_Z = 0.48
DEFAULT_TABLE_CENTER_XY = (0.45, -0.1)
DEFAULT_TABLE_FULL_SIZE = (1.4, 1.2, 0.05)
TABLE_TOP_MARGIN = 0.25
OBJECT_BODY_NAME_NEEDLES = (
    "apple",
    "banana",
    "bowl",
    "grape",
    "lemon",
    "orange",
    "pear",
    "walnut",
)
ARM_ROBOT_BODY_NEEDLES = ("franka", "panda", "ur10", "ur10e", "ur5", "ur5e", "ur_")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_TEXTURE_ASSET_DIR = PROJECT_ROOT / "assets" / "table_textures" / "polyhaven"
# The bank itself is downloaded, not vendored, and Polyhaven's catalogue moves: a fresh download
# names its files by position in that catalogue, so a manifest recorded earlier can stop resolving
# against a bank fetched later. Overriding the manifest path lets a machine describe the bank it
# actually has without editing the one the repo ships.
TABLE_TEXTURE_MANIFEST_PATH = Path(
    os.environ.get("AXIS_TABLE_TEXTURE_MANIFEST") or (TABLE_TEXTURE_ASSET_DIR / "manifest.json")
)
CURATED_TABLE_TEXTURE_ASSET_DIR = PROJECT_ROOT / "assets" / "table_textures" / "tabletop_1k"
CURATED_TABLE_TEXTURE_MANIFEST_PATH = CURATED_TABLE_TEXTURE_ASSET_DIR / "manifest.json"
CAMERA_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "cameras"
MIN_TABLE_TEXTURE_BANK_SIZE = 1000
MIN_CURATED_TABLE_TEXTURE_BANK_SIZE = 33
TABLE_MANIFEST_REQUIRED_KEYS = (
    "material_family",
    "texture_variation",
    "texture_file",
    "source_asset_id",
    "source_url",
    "license",
    "material",
)
SCENE_TEXTURE_SURFACES = ("table", "floor", "wall")
FLOOR_MATERIAL_KEYWORDS = {
    "asphalt",
    "brick",
    "ceramic",
    "concrete",
    "floor",
    "ground",
    "marble",
    "pavement",
    "road",
    "rock",
    "stone",
    "terrain",
    "tile",
    "tiles",
    "wood",
}
WALL_MATERIAL_KEYWORDS = {
    "brick",
    "ceramic",
    "concrete",
    "fabric",
    "marble",
    "metal",
    "plaster",
    "rock",
    "stone",
    "tile",
    "tiles",
    "wall",
    "wood",
}
TABLE_PROCEDURAL_MATERIAL_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "material_family": "wood",
        "texture_variation": "gradient",
        "rgb1": (92, 52, 26),
        "rgb2": (199, 143, 77),
        "material": (0.08, 0.32, 0.18),
    },
    {
        "material_family": "stone",
        "texture_variation": "noise",
        "rgb1": (87, 90, 92),
        "rgb2": (184, 184, 174),
        "fraction": 0.34,
        "material": (0.18, 0.22, 0.12),
    },
    {
        "material_family": "marble",
        "texture_variation": "noise",
        "rgb1": (210, 204, 189),
        "rgb2": (95, 105, 122),
        "fraction": 0.18,
        "material": (0.22, 0.48, 0.28),
    },
    {
        "material_family": "ceramic",
        "texture_variation": "checker",
        "rgb1": (219, 219, 199),
        "rgb2": (92, 128, 148),
        "material": (0.12, 0.58, 0.35),
    },
    {
        "material_family": "metal",
        "texture_variation": "gradient",
        "rgb1": (90, 100, 110),
        "rgb2": (180, 188, 192),
        "material": (0.45, 0.80, 0.75),
    },
    {
        "material_family": "checker",
        "texture_variation": "checker",
        "rgb1": (70, 75, 80),
        "rgb2": (220, 174, 67),
        "material": (0.10, 0.35, 0.20),
    },
    {
        "material_family": "noise",
        "texture_variation": "noise",
        "rgb1": (60, 112, 95),
        "rgb2": (194, 209, 132),
        "fraction": 0.48,
        "material": (0.16, 0.28, 0.16),
    },
    {
        "material_family": "gradient",
        "texture_variation": "gradient",
        "rgb1": (80, 92, 180),
        "rgb2": (210, 112, 118),
        "material": (0.14, 0.42, 0.26),
    },
)
def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_curated_tabletop_entry(
    entry: dict[str, Any],
    *,
    entry_index: int,
    texture_path: Path,
    expected_texture_bank_id: str,
) -> None:
    field_prefix = f"Curated tabletop manifest entry {entry_index}"
    source_asset_id = str(entry.get("source_asset_id") or "").strip()
    if not source_asset_id:
        raise ValueError(f"{field_prefix} has an empty source_asset_id.")
    if entry.get("texture_bank_id") != expected_texture_bank_id:
        raise ValueError(
            f"{field_prefix} has texture_bank_id={entry.get('texture_bank_id')!r}; "
            f"expected {expected_texture_bank_id!r}."
        )
    if entry.get("license") != "CC0" or entry.get("source_asset_type") != "textures":
        raise ValueError(f"{field_prefix} must identify a CC0 Poly Haven texture asset.")
    if entry.get("source_resolution") != "1k":
        raise ValueError(f"{field_prefix} must identify the 1k source map.")

    source_page = urllib.parse.urlparse(str(entry.get("source_url") or ""))
    if (
        source_page.scheme != "https"
        or source_page.hostname != "polyhaven.com"
        or source_page.path != f"/a/{source_asset_id}"
        or source_page.query
        or source_page.fragment
    ):
        raise ValueError(
            f"{field_prefix} has an invalid Poly Haven source URL: {source_page.geturl()!r}"
        )
    direct_url = urllib.parse.urlparse(str(entry.get("direct_download_url") or ""))
    if (
        direct_url.scheme != "https"
        or direct_url.hostname != "dl.polyhaven.org"
        or f"/{source_asset_id}/" not in direct_url.path
        or not direct_url.path.endswith("_1k.jpg")
        or direct_url.query
        or direct_url.fragment
    ):
        raise ValueError(
            f"{field_prefix} has an invalid Poly Haven download URL: {direct_url.geturl()!r}"
        )

    expected_bytes = entry.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ValueError(f"{field_prefix}.bytes must be a positive integer.")
    actual_bytes = texture_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{field_prefix} byte count mismatch for {texture_path}: "
            f"expected={expected_bytes}, actual={actual_bytes}."
        )
    expected_sha256 = str(entry.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"{field_prefix}.sha256 must be a lowercase SHA-256 digest.")
    actual_sha256 = _file_digest(texture_path, "sha256")
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{field_prefix} SHA-256 mismatch for {texture_path}: "
            f"expected={expected_sha256}, actual={actual_sha256}."
        )

    pixel_width = entry.get("pixel_width")
    pixel_height = entry.get("pixel_height")
    if (pixel_width, pixel_height) != (1024, 1024):
        raise ValueError(
            f"{field_prefix} must declare pixel_width=pixel_height=1024, "
            f"got {(pixel_width, pixel_height)}."
        )
    if entry.get("image_mode") != "RGB" or entry.get("image_format") != "JPEG":
        raise ValueError(f"{field_prefix} must declare an RGB JPEG image.")
    from PIL import Image

    with Image.open(texture_path) as image:
        image.load()
        actual_image = (image.size, image.mode, image.format)
    expected_image = ((1024, 1024), "RGB", "JPEG")
    if actual_image != expected_image:
        raise ValueError(
            f"{field_prefix} decoded image mismatch for {texture_path}: "
            f"expected={expected_image}, actual={actual_image}."
        )


def _load_table_image_material_families(
    manifest_path: Path,
    *,
    asset_dir: Path,
    minimum_count: int,
    texture_bank_id: str,
    verify_integrity: bool = False,
) -> tuple[dict[str, Any], ...]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Table texture manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Table texture manifest must be a list: {manifest_path}")

    resolved_asset_dir = asset_dir.expanduser().resolve()
    families: list[dict[str, Any]] = []
    seen_source_asset_ids: set[str] = set()
    seen_texture_files: set[str] = set()
    for index, raw_entry in enumerate(payload):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Table texture manifest entry {index} is not an object.")
        missing = [key for key in TABLE_MANIFEST_REQUIRED_KEYS if key not in raw_entry]
        if missing:
            raise ValueError(f"Table texture manifest entry {index} is missing keys: {missing}")
        entry = dict(raw_entry)
        if str(entry["texture_variation"]) != "image":
            raise ValueError(f"Table texture manifest entry {index} is not an image texture: {entry!r}")
        material = entry["material"]
        if not isinstance(material, list | tuple) or len(material) != 3:
            raise ValueError(f"Table texture manifest entry {index} has invalid material: {material!r}")
        entry["material"] = tuple(float(value) for value in material)
        texture_file = str(entry["texture_file"])
        texture_path = (resolved_asset_dir / texture_file).resolve()
        try:
            texture_path.relative_to(resolved_asset_dir)
        except ValueError as exc:
            raise ValueError(f"Table texture path escapes its asset directory: {texture_file!r}") from exc
        if not texture_path.is_file():
            raise FileNotFoundError(f"Table texture image is missing: {texture_path}")
        if verify_integrity:
            source_asset_id = str(entry["source_asset_id"])
            if source_asset_id in seen_source_asset_ids:
                raise ValueError(
                    "Curated tabletop manifest has duplicate source_asset_id "
                    f"{source_asset_id!r}."
                )
            if texture_file in seen_texture_files:
                raise ValueError(f"Curated tabletop manifest has duplicate texture_file {texture_file!r}.")
            seen_source_asset_ids.add(source_asset_id)
            seen_texture_files.add(texture_file)
            _verify_curated_tabletop_entry(
                entry,
                entry_index=index,
                texture_path=texture_path,
                expected_texture_bank_id=texture_bank_id,
            )
        entry["texture_bank_id"] = str(entry.get("texture_bank_id") or texture_bank_id)
        entry["_texture_asset_dir"] = str(resolved_asset_dir)
        families.append(entry)

    if len(families) < minimum_count:
        raise ValueError(
            f"Table texture manifest has {len(families)} entries; expected at least "
            f"{minimum_count}: {manifest_path}"
        )
    return tuple(families)


TABLE_IMAGE_MATERIAL_FAMILIES: tuple[dict[str, Any], ...] = _load_table_image_material_families(
    TABLE_TEXTURE_MANIFEST_PATH,
    asset_dir=TABLE_TEXTURE_ASSET_DIR,
    minimum_count=MIN_TABLE_TEXTURE_BANK_SIZE,
    texture_bank_id="polyhaven_prepared_256_v1",
)
TABLE_MATERIAL_FAMILIES: tuple[dict[str, Any], ...] = TABLE_IMAGE_MATERIAL_FAMILIES
CURATED_TABLE_MATERIAL_FAMILIES: tuple[dict[str, Any], ...] = _load_table_image_material_families(
    CURATED_TABLE_TEXTURE_MANIFEST_PATH,
    asset_dir=CURATED_TABLE_TEXTURE_ASSET_DIR,
    minimum_count=MIN_CURATED_TABLE_TEXTURE_BANK_SIZE,
    texture_bank_id="axis_curated_tabletop_1k_v1",
    verify_integrity=True,
)


def _material_entry_tokens(entry: dict[str, Any]) -> set[str]:
    values: list[str] = []
    for key in ("material_family", "source_asset_id", "asset_id", "label", "source_asset_type"):
        value = entry.get(key)
        if value is not None:
            values.append(str(value))
    for key in ("categories", "tags"):
        value = entry.get(key) or []
        if isinstance(value, list | tuple):
            values.extend(str(item) for item in value)
    tokens: set[str] = set()
    for value in values:
        tokens.update(token for token in re.split(r"[^a-z0-9]+", value.lower()) if token)
    return tokens


def _filter_material_candidates(
    candidates: tuple[tuple[int, dict[str, Any]], ...],
    keywords: set[str],
    *,
    source_asset_types: set[str] | None = None,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    selected = tuple(
        (index, family)
        for index, family in candidates
        if (source_asset_types is None or str(family.get("source_asset_type")) in source_asset_types)
        and _material_entry_tokens(family) & keywords
    )
    if not selected:
        raise ValueError(f"No PolyHaven material candidates matched keywords: {sorted(keywords)}")
    return selected


ALL_PREPARED_MATERIAL_CANDIDATES: tuple[tuple[int, dict[str, Any]], ...] = tuple(
    enumerate(TABLE_MATERIAL_FAMILIES)
)
TABLE_MATERIAL_CANDIDATES: tuple[tuple[int, dict[str, Any]], ...] = tuple(
    enumerate(CURATED_TABLE_MATERIAL_FAMILIES)
)
FLOOR_MATERIAL_CANDIDATES: tuple[tuple[int, dict[str, Any]], ...] = _filter_material_candidates(
    ALL_PREPARED_MATERIAL_CANDIDATES,
    FLOOR_MATERIAL_KEYWORDS,
    source_asset_types={"textures"},
)
WALL_MATERIAL_CANDIDATES: tuple[tuple[int, dict[str, Any]], ...] = _filter_material_candidates(
    ALL_PREPARED_MATERIAL_CANDIDATES,
    WALL_MATERIAL_KEYWORDS,
    source_asset_types={"textures"},
)
SURFACE_MATERIAL_CANDIDATES: dict[str, tuple[tuple[int, dict[str, Any]], ...]] = {
    "table": TABLE_MATERIAL_CANDIDATES,
    "floor": FLOOR_MATERIAL_CANDIDATES,
    "wall": WALL_MATERIAL_CANDIDATES,
}


def _configured_surface_material_candidates(
    texture_profile: dict[str, Any],
) -> tuple[
    dict[str, tuple[tuple[int, dict[str, Any]], ...]],
    dict[str, dict[str, Any]],
]:
    configured = texture_profile.get("asset_ids_by_surface") or {}
    if not isinstance(configured, dict):
        raise ValueError("render_profile.randomization.texture.asset_ids_by_surface must be an object.")
    unknown_surfaces = sorted(set(str(name) for name in configured) - set(SCENE_TEXTURE_SURFACES))
    if unknown_surfaces:
        raise ValueError(f"Unknown material candidate surfaces: {unknown_surfaces}")

    resolved: dict[str, tuple[tuple[int, dict[str, Any]], ...]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for surface_name, base_candidates in SURFACE_MATERIAL_CANDIDATES.items():
        requested_raw = configured.get(surface_name)
        if requested_raw is None:
            resolved[surface_name] = base_candidates
            metadata[surface_name] = {
                "mode": "default_surface_bank",
                "asset_ids": None,
                "candidate_count": len(base_candidates),
            }
            continue
        if not isinstance(requested_raw, list) or not requested_raw:
            raise ValueError(f"Configured material asset ids for {surface_name} must be a non-empty list.")
        requested = [str(asset_id).strip() for asset_id in requested_raw]
        if any(not asset_id for asset_id in requested) or len(requested) != len(set(requested)):
            raise ValueError(f"Configured material asset ids for {surface_name} must be unique and non-empty.")
        by_asset_id = {
            str(family.get("source_asset_id")): (family_index, family)
            for family_index, family in base_candidates
        }
        missing = [asset_id for asset_id in requested if asset_id not in by_asset_id]
        if missing:
            raise ValueError(
                f"Configured material assets are not available in the {surface_name} bank: {missing}"
            )
        selected = tuple(by_asset_id[asset_id] for asset_id in requested)
        resolved[surface_name] = selected
        metadata[surface_name] = {
            "mode": "explicit_scene_asset_ids",
            "asset_ids": requested,
            "candidate_count": len(selected),
        }
    return resolved, metadata


@dataclass(frozen=True)
class RenderCandidate:
    task_id: int
    task_name: str
    attempt_id: int
    embodiment_id: str
    attempt_pass_verify: bool | None
    trajectory_pass_verify: bool | None


@dataclass(frozen=True)
class RenderJobSpec:
    attempt_id: int
    output_dir: str
    cache_dir: str
    config_path: str | None
    asset_dir: str | None
    asset_base_url: str | None
    cameras: tuple[str, ...]
    width: int
    height: int
    frame_stride: int
    max_frames: int | None
    seed: int
    variant_id: int
    save_preview_video: bool
    save_contact_sheet: bool
    randomization_enabled: bool
    resume: bool
    gpu_id: int | None
    lerobot_root: str | None
    lerobot_repo_id: str | None
    lerobot_fps: int
    lerobot_robot_type: str
    lerobot_vcodec: str | None
    lerobot_image_writer_threads: int
    lerobot_image_writer_processes: int
    lerobot_encoder_queue_maxsize: int = 128
    lerobot_encoder_threads: int | None = 1
    #: The JSON bundle to render: the (task.mjcf_xml, task_trajectory.trajectory_data) pair of
    #: one attempt. render_attempt_job reads it from this file.
    local_bundle_path: str | None = None
    #: Override the embodiment id, which decides which config/cameras/<id>.json applies.
    embodiment_id_override: str | None = None



@dataclass
class RenderRecord:
    status: str
    task_id: int | None = None
    task_name: str | None = None
    attempt_id: int | None = None
    embodiment_id: str | None = None
    model_id: str | None = None
    variant_id: int | None = None
    seed: int | None = None
    num_frames: int = 0
    camera_names: list[str] | None = None
    render_time_s: float = 0.0
    output_path: str | None = None
    metadata_path: str | None = None
    lerobot_episode: dict[str, Any] | None = None
    lerobot_episodes: list[dict[str, Any]] | None = None
    camera_dataset_outputs: list[dict[str, Any]] | None = None
    capture_manifest_path: str | None = None
    num_lerobot_episodes: int = 0
    num_state_frames_replayed: int = 0
    num_camera_frames_rendered: int = 0
    num_lerobot_frames_written: int = 0
    preview_videos: dict[str, str] | None = None
    contact_sheet: str | None = None
    randomization_config: dict[str, Any] | None = None
    error_message: str | None = None
    traceback: str | None = None


@dataclass(frozen=True)
class CameraCapturePair:
    capture_id: str
    camera_group: str | None
    camera_replica_id: int | None
    camera_name_map: tuple[tuple[str, str], ...]

    @property
    def physical_camera_names(self) -> tuple[str, ...]:
        return tuple(physical_name for _, physical_name in self.camera_name_map)

    @property
    def logical_camera_names(self) -> tuple[str, ...]:
        return tuple(logical_name for logical_name, _ in self.camera_name_map)


@dataclass(frozen=True)
class ScenePlacement:
    policy: str
    table_center_xy: tuple[float, float]
    table_top_z: float
    table_full_size: tuple[float, float, float]
    preserve_source_table: bool
    arm_scene_z_offset: float
    free_joint_qpos_z_offset: float
    table_surface_geom_names: tuple[str, ...]
    metadata: dict[str, Any]
    #: Bodies to lift onto the table besides the ones ARM_ROBOT_BODY_NEEDLES recognises. Those
    #: needles spell arm robots by vendor name, which an assembled embodiment's root body need not
    #: carry; leaving it unrecognised lifts the objects onto the table and leaves the arm at floor
    #: height, so the arm ends up inside the table rather than reaching over it. Defaulted so every
    #: existing construction of this record keeps working unchanged.
    arm_root_body_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _MJCFDocument:
    """An entry MJCF document or one of its recursively included documents."""

    root: ET.Element
    path: Path | None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def stable_seed(global_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256()
    digest.update(str(int(global_seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False) % (2**32)


def _camera_randomization_seed(
    base_camera_seed: int,
    camera_name: str,
    camera_entry: dict[str, Any],
) -> tuple[int, int | None, str]:
    capture_replica = camera_entry.get("capture_replica") or {}
    if str(capture_replica.get("camera_group") or "") != "randomized":
        return stable_seed(base_camera_seed, camera_name), None, camera_name
    replica_id = int(capture_replica["camera_replica_id"])
    logical_camera_name = str(capture_replica["logical_camera_name"])
    pair_seed = stable_seed(base_camera_seed, "randomized_pair", replica_id)
    return stable_seed(pair_seed, logical_camera_name), pair_seed, logical_camera_name


def disable_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def infer_embodiment_id(task_name: str, mjcf_xml: str | None = None) -> str:
    return "franka"


def _camera_config_id(embodiment_id: str, mjcf_xml: str | None = None) -> str:
    return embodiment_id


def _load_embodiment_camera_config(
    embodiment_id: str,
    requested_camera_names: tuple[str, ...],
    *,
    config_id: str | None = None,
) -> dict[str, Any]:
    resolved_config_id = str(config_id or embodiment_id)
    config_path = CAMERA_CONFIG_DIR / f"{resolved_config_id}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing camera config for embodiment {embodiment_id!r}: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Camera config must be a JSON object: {config_path}")
    if str(payload.get("embodiment_id")) != embodiment_id:
        raise ValueError(
            f"Camera config {config_path} has embodiment_id={payload.get('embodiment_id')!r}; "
            f"expected {embodiment_id!r}."
        )
    declared_config_id = str(payload.get("config_id") or resolved_config_id)
    if declared_config_id != resolved_config_id:
        raise ValueError(
            f"Camera config {config_path} has config_id={declared_config_id!r}; "
            f"expected {resolved_config_id!r}."
        )
    cameras = payload.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"Camera config must define a non-empty cameras list: {config_path}")
    camera_by_name: dict[str, dict[str, Any]] = {}
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            raise ValueError(f"Camera config entry {index} is not an object: {config_path}")
        name = str(camera.get("name") or "").strip()
        if not name:
            raise ValueError(f"Camera config entry {index} has an empty name: {config_path}")
        if name in camera_by_name:
            raise ValueError(f"Duplicate camera name {name!r} in {config_path}")
        camera_by_name[name] = camera
    if int(payload.get("camera_count") or len(cameras)) != len(cameras):
        raise ValueError(f"camera_count does not match cameras length in {config_path}")

    default_camera_names = tuple(str(name) for name in (payload.get("default_camera_names") or camera_by_name.keys()))
    selected_names = requested_camera_names or default_camera_names
    missing = [name for name in selected_names if name not in camera_by_name]
    if missing:
        raise ValueError(
            f"Requested camera(s) {missing} are not defined for embodiment {embodiment_id!r}. "
            f"Available cameras: {sorted(camera_by_name)}"
        )
    selected_cameras = [deepcopy(camera_by_name[name]) for name in selected_names]
    resolved = deepcopy(payload)
    resolved.update(
        {
            "config_path": str(config_path),
            "config_id": resolved_config_id,
            "embodiment_id": embodiment_id,
            "camera_count": len(cameras),
            "default_camera_names": list(default_camera_names),
            "selected_camera_names": list(selected_names),
            "cameras": selected_cameras,
            "all_camera_names": list(camera_by_name),
        }
    )
    return resolved


def _camera_names_from_config(camera_config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(camera["name"]) for camera in camera_config["cameras"])


def _expand_camera_capture_plan(
    camera_config: dict[str, Any],
    *,
    randomization_enabled: bool,
) -> tuple[dict[str, Any], tuple[CameraCapturePair, ...]]:
    """Expand logical cameras into one fixed pair and N randomized physical pairs."""

    logical_config = deepcopy(camera_config)
    logical_cameras = logical_config.get("cameras") or []
    if not logical_cameras:
        raise ValueError("Camera capture expansion requires at least one selected logical camera.")
    logical_names = tuple(str(camera["name"]) for camera in logical_cameras)
    if len(logical_names) != len(set(logical_names)):
        raise ValueError(f"Logical camera names must be unique: {logical_names!r}.")

    raw_profile = logical_config.get("capture_multiplexing") or {}
    if not isinstance(raw_profile, dict):
        raise ValueError("capture_multiplexing must be a JSON object.")
    if not bool(raw_profile.get("enabled", False)):
        return (
            logical_config,
            (
                CameraCapturePair(
                    capture_id="legacy",
                    camera_group=None,
                    camera_replica_id=None,
                    camera_name_map=tuple((name, name) for name in logical_names),
                ),
            ),
        )

    schema = str(raw_profile.get("schema") or "")
    if schema != CAMERA_CAPTURE_MULTIPLEX_SCHEMA:
        raise ValueError(
            f"Unsupported capture_multiplexing schema {schema!r}; "
            f"expected {CAMERA_CAPTURE_MULTIPLEX_SCHEMA!r}."
        )
    fixed_replicas = int(raw_profile.get("fixed_replicas", 1))
    randomized_replicas = int(raw_profile.get("randomized_replicas", 0))
    if fixed_replicas != 1:
        raise ValueError("fixed_plus_randomized_camera_pairs_v1 requires fixed_replicas=1.")
    if randomized_replicas < 0:
        raise ValueError("capture_multiplexing.randomized_replicas must be non-negative.")
    effective_randomized_replicas = randomized_replicas if randomization_enabled else 0

    fixed_entries: list[dict[str, Any]] = []
    fixed_mapping: list[tuple[str, str]] = []
    for prototype in logical_cameras:
        logical_name = str(prototype["name"])
        fixed = deepcopy(prototype)
        fixed_randomization = deepcopy(fixed.get("randomization") or {})
        fixed_randomization["enabled"] = False
        fixed["randomization"] = fixed_randomization
        fixed["capture_replica"] = {
            "schema": schema,
            "camera_group": "fixed",
            "camera_replica_id": 0,
            "logical_camera_name": logical_name,
        }
        fixed_entries.append(fixed)
        fixed_mapping.append((logical_name, logical_name))

    physical_entries = list(fixed_entries)
    captures: list[CameraCapturePair] = [
        CameraCapturePair(
            capture_id="camera_fixed_replica_000",
            camera_group="fixed",
            camera_replica_id=0,
            camera_name_map=tuple(fixed_mapping),
        )
    ]
    for replica_id in range(effective_randomized_replicas):
        random_mapping: list[tuple[str, str]] = []
        for prototype in logical_cameras:
            logical_name = str(prototype["name"])
            randomization = deepcopy(prototype.get("randomization") or {})
            if not bool(randomization.get("enabled", False)):
                raise ValueError(
                    f"Multiplexed random camera {logical_name!r} must enable its randomization profile."
                )
            physical_name = f"{logical_name}_random_{replica_id:02d}"
            clone = deepcopy(prototype)
            clone["name"] = physical_name
            clone["capture_replica"] = {
                "schema": schema,
                "camera_group": "randomized",
                "camera_replica_id": int(replica_id),
                "logical_camera_name": logical_name,
            }
            physical_entries.append(clone)
            random_mapping.append((logical_name, physical_name))
        captures.append(
            CameraCapturePair(
                capture_id=f"camera_randomized_replica_{replica_id:03d}",
                camera_group="randomized",
                camera_replica_id=int(replica_id),
                camera_name_map=tuple(random_mapping),
            )
        )

    physical_names = [str(camera["name"]) for camera in physical_entries]
    if len(physical_names) != len(set(physical_names)):
        raise ValueError(f"Expanded physical camera names are not unique: {physical_names!r}.")
    runtime_config = deepcopy(logical_config)
    runtime_config["camera_count"] = len(physical_entries)
    runtime_config["default_camera_names"] = physical_names
    runtime_config["selected_camera_names"] = physical_names
    runtime_config["all_camera_names"] = physical_names
    runtime_config["logical_camera_names"] = list(logical_names)
    runtime_config["cameras"] = physical_entries
    runtime_config["capture_multiplexing"] = {
        **deepcopy(raw_profile),
        "enabled": True,
        "schema": schema,
        "fixed_replicas": 1,
        "configured_randomized_replicas": randomized_replicas,
        "effective_randomized_replicas": effective_randomized_replicas,
        "randomization_suppressed_by_runtime": bool(
            randomized_replicas and not randomization_enabled
        ),
        "pairing": "same_replica_index_across_logical_cameras",
        "seed_contract": "pair_seed_then_logical_camera_child_seed_count_independent",
    }
    return runtime_config, tuple(captures)


def _camera_config_for_capture_pair(
    runtime_camera_config: dict[str, Any],
    capture: CameraCapturePair,
) -> dict[str, Any]:
    by_name = {
        str(camera["name"]): camera
        for camera in runtime_camera_config.get("cameras") or []
    }
    missing = [name for name in capture.physical_camera_names if name not in by_name]
    if missing:
        raise ValueError(f"Capture {capture.capture_id!r} references unknown cameras: {missing!r}.")
    cameras = [deepcopy(by_name[name]) for name in capture.physical_camera_names]
    pair_config = deepcopy(runtime_camera_config)
    pair_config.pop("capture_multiplexing", None)
    pair_config["camera_count"] = len(cameras)
    pair_config["default_camera_names"] = list(capture.physical_camera_names)
    pair_config["selected_camera_names"] = list(capture.physical_camera_names)
    pair_config["all_camera_names"] = list(capture.physical_camera_names)
    pair_config["logical_camera_names"] = list(capture.logical_camera_names)
    pair_config["cameras"] = cameras
    pair_config["capture_episode"] = {
        "schema": CAMERA_CAPTURE_MULTIPLEX_SCHEMA,
        "capture_id": capture.capture_id,
        "camera_group": capture.camera_group,
        "camera_replica_id": capture.camera_replica_id,
        "camera_name_map": [
            {"logical": logical_name, "physical": physical_name}
            for logical_name, physical_name in capture.camera_name_map
        ],
    }
    return pair_config


def _validate_append_only_capture_manifest(
    existing_manifest: dict[str, Any],
    current_capture_ids: set[str],
) -> set[str]:
    prior_capture_key = (
        "captures" if bool(existing_manifest.get("complete")) else "expected_captures"
    )
    prior_capture_ids = {
        str(capture.get("capture_id") or "")
        for capture in existing_manifest.get(prior_capture_key) or []
    }
    if not prior_capture_ids:
        raise ValueError(
            f"Existing {CAMERA_CAPTURE_MANIFEST_SCHEMA} manifest does not declare "
            f"{prior_capture_key}; refusing to modify non-empty camera datasets."
        )
    if not prior_capture_ids.issubset(current_capture_ids):
        raise ValueError(
            "Changing camera multiplexing may append replicas but cannot silently remove "
            f"committed or in-progress captures: existing={sorted(prior_capture_ids)!r}, "
            f"current={sorted(current_capture_ids)!r}. Use a new output root."
        )
    return prior_capture_ids


def _validate_camera_render_aspect_ratio(
    camera_config: dict[str, Any],
    *,
    width: int,
    height: int,
) -> None:
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError(f"Render width and height must be positive, got {width}x{height}.")
    camera_randomization = _camera_randomization_by_name(camera_config)
    custom_modes = {
        str(profile.get("mode") or "")
        for profile in camera_randomization.values()
        if str(profile.get("mode") or "")
        in {WRIST_CAMERA_RANDOMIZATION_MODE, FRONT_CAMERA_RANDOMIZATION_MODE}
    }
    if not custom_modes:
        return
    required_ratios: dict[str, tuple[int, int]] = {}
    for camera in camera_config["cameras"]:
        camera_name = str(camera["name"])
        raw_ratio = str((camera.get("intrinsics") or {}).get("required_aspect_ratio") or "").strip()
        match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", raw_ratio)
        if match is None:
            raise ValueError(
                f"Custom-randomized camera {camera_name!r} must declare required_aspect_ratio as W:H."
            )
        required_ratios[camera_name] = (int(match.group(1)), int(match.group(2)))
    unique_ratios = sorted(set(required_ratios.values()))
    if len(unique_ratios) != 1:
        raise ValueError(f"Selected custom camera profiles disagree on aspect ratio: {required_ratios}.")
    required_width, required_height = unique_ratios[0]
    if int(width) * required_height != int(height) * required_width:
        raise ValueError(
            "Franka semantic camera randomization requires the configured camera aspect ratio: "
            f"requested={width}x{height}, required={required_width}:{required_height}."
        )


def _fixed_surface_texture_asset(camera_config: dict[str, Any]) -> dict[str, Any] | None:
    render_profile = camera_config.get("render_profile") or {}
    surface_profile = render_profile.get("surface_appearance") or {}
    if not surface_profile:
        return None
    if not isinstance(surface_profile, dict):
        raise ValueError("render_profile.surface_appearance must be a JSON object.")
    mode = str(surface_profile.get("mode") or "").strip()
    if mode != "fixed_texture_material":
        raise ValueError(f"Unsupported surface appearance mode {mode!r}.")
    table_profile = surface_profile.get("table") or {}
    if not isinstance(table_profile, dict):
        raise ValueError("render_profile.surface_appearance.table must be a JSON object.")
    texture_file = str(table_profile.get("texture_file") or "").strip()
    if not texture_file:
        raise ValueError("render_profile.surface_appearance.table.texture_file is required.")
    texture_path = (PROJECT_ROOT / texture_file).resolve()
    try:
        texture_repo_path = texture_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Surface texture must stay inside the project root: {texture_path}") from exc
    if not texture_path.is_file():
        raise FileNotFoundError(f"Configured table texture is missing: {texture_path}")
    texture_sha256 = hashlib.sha256(texture_path.read_bytes()).hexdigest()
    declared_sha256 = str(table_profile.get("sha256") or "").strip().lower()
    if declared_sha256 and not re.fullmatch(r"[0-9a-f]{64}", declared_sha256):
        raise ValueError("render_profile.surface_appearance.table.sha256 must be a lowercase SHA-256 digest.")
    if declared_sha256 and declared_sha256 != texture_sha256:
        raise ValueError(
            "Configured table texture checksum mismatch: "
            f"expected={declared_sha256}, actual={texture_sha256}, path={texture_path}"
        )
    return {
        "path": texture_path,
        "repo_path": str(texture_repo_path),
        "sha256": texture_sha256,
        "declared_sha256": declared_sha256 or None,
    }


def _render_source_fingerprint(bundle: dict[str, Any]) -> str:
    task = bundle.get("task") or {}
    task_trajectory = bundle.get("task_trajectory") or {}
    attempt = bundle.get("attempt") or {}
    payload = {
        "schema": "axis_render_source_v1",
        "task": {
            "id": task.get("id"),
            "name": task.get("name"),
            "mjcf_xml": task.get("mjcf_xml"),
            "scene_variants": task.get("scene_variants"),
        },
        "trajectory": {
            "trajectory_data": task_trajectory.get("trajectory_data"),
            "trajectory_data_source": task_trajectory.get("trajectory_data_source"),
        },
        "attempt_timing": {
            "simulation_time_seconds": attempt.get("simulation_time_seconds"),
        },
    }
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _render_fingerprint(
    camera_config: dict[str, Any],
    spec: RenderJobSpec,
    *,
    lerobot_repo_id: str,
    source_fingerprint: str | None = None,
) -> str:
    canonical_camera_config = {
        key: value
        for key, value in camera_config.items()
        if key != "config_path"
    }
    fixed_surface_texture = _fixed_surface_texture_asset(camera_config)
    payload = {
        "schema_version": RENDER_FINGERPRINT_SCHEMA_VERSION,
        "camera_config": canonical_camera_config,
        "render": {
            "width": int(spec.width),
            "height": int(spec.height),
            "frame_stride": int(spec.frame_stride),
            "max_frames": spec.max_frames,
            "seed": int(spec.seed),
            "variant_id": int(spec.variant_id),
            "randomization_enabled": bool(spec.randomization_enabled),
            "save_preview_video": bool(spec.save_preview_video),
            "save_contact_sheet": bool(spec.save_contact_sheet),
            "state_resampling_schema": STATE_RESAMPLING_SCHEMA,
            "state_resampling_policy": "joint_aware_interpolate_or_downsample_then_mj_forward",
            "physics_rollout": False,
        },
        "lerobot": {
            "repo_id": str(lerobot_repo_id),
            "schema": LEROBOT_POLICY_SCHEMA,
            "fps": int(spec.lerobot_fps),
            "robot_type": str(spec.lerobot_robot_type),
            "vcodec": spec.lerobot_vcodec,
        },
    }
    if source_fingerprint is not None:
        payload["source_fingerprint"] = str(source_fingerprint)
    profile_assets: dict[str, Any] = {
        "prepared_texture_manifest": {
            "repo_path": str(TABLE_TEXTURE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(TABLE_TEXTURE_MANIFEST_PATH.read_bytes()).hexdigest(),
        },
        "curated_tabletop_manifest": {
            "repo_path": str(CURATED_TABLE_TEXTURE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(CURATED_TABLE_TEXTURE_MANIFEST_PATH.read_bytes()).hexdigest(),
        }
    }
    if fixed_surface_texture is not None:
        profile_assets["fixed_surface_texture"] = {
            "repo_path": fixed_surface_texture["repo_path"],
            "sha256": fixed_surface_texture["sha256"],
        }
    payload["profile_assets"] = profile_assets
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _format_float_sequence(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _camera_xyaxes(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-8:
        raise ValueError("Camera position and target must be different.")
    forward /= norm
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-8:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
    right /= right_norm
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-8)
    return np.concatenate([right, up])


def _default_camera_pose(camera_index: int, placement: ScenePlacement) -> tuple[np.ndarray, np.ndarray]:
    center_x, center_y = placement.table_center_xy
    target_z = placement.table_top_z + 0.12
    camera_z = placement.table_top_z + 0.70
    poses = (
        (np.asarray([center_x + 1.20, center_y, camera_z], dtype=np.float64), np.asarray([center_x, center_y, target_z], dtype=np.float64)),
        (
            np.asarray([center_x + 1.20, center_y + 0.22, camera_z], dtype=np.float64),
            np.asarray([center_x, center_y, target_z], dtype=np.float64),
        ),
        (
            np.asarray([center_x + 1.20, center_y - 0.22, camera_z], dtype=np.float64),
            np.asarray([center_x, center_y, target_z], dtype=np.float64),
        ),
    )
    return poses[camera_index % len(poses)]


def _camera_fovy(camera_config: dict[str, Any]) -> float:
    intrinsics = camera_config.get("intrinsics") or {}
    return float(intrinsics.get("fovy", 58.0))


def _table_relative_camera_pose(camera_config: dict[str, Any], placement: ScenePlacement) -> tuple[np.ndarray, np.ndarray]:
    extrinsics = camera_config.get("extrinsics") or {}
    position_offset = _parse_vec(
        _format_float_sequence(extrinsics.get("position_offset", [1.2, 0.0, 0.7])),
        length=3,
    )
    target_offset = _parse_vec(
        _format_float_sequence(extrinsics.get("target_offset", [0.0, 0.0, 0.12])),
        length=3,
    )
    base = np.asarray([placement.table_center_xy[0], placement.table_center_xy[1], placement.table_top_z], dtype=np.float64)
    return base + position_offset, base + target_offset


def _normalize_quat_wxyz(quat: np.ndarray, *, label: str) -> np.ndarray:
    normalized = np.asarray(quat, dtype=np.float64)
    if normalized.shape != (4,):
        raise ValueError(f"{label} must contain four wxyz values.")
    norm = float(np.linalg.norm(normalized))
    if norm <= 1e-12:
        raise ValueError(f"{label} must be non-zero.")
    return normalized / norm


def _quat_multiply_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat, label="rotation quaternion")
    vector_quat = np.concatenate([np.zeros(1, dtype=np.float64), np.asarray(vector, dtype=np.float64)])
    conjugate = quat * np.asarray([1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    return _quat_multiply_wxyz(_quat_multiply_wxyz(quat, vector_quat), conjugate)[1:]


WRIST_CAMERA_RANDOMIZATION_MODE = "optical_pose_box_v1"
FRONT_CAMERA_RANDOMIZATION_MODE = "base_table_sector_lookat_v1"


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    normalized = _normalize_quat_wxyz(quat, label="matrix conversion quaternion")
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, normalized)
    return matrix.reshape(3, 3)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    resolved = np.asarray(matrix, dtype=np.float64)
    if resolved.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {resolved.shape}.")
    if not np.allclose(resolved.T @ resolved, np.eye(3), atol=1e-8) or not np.isclose(
        np.linalg.det(resolved), 1.0, atol=1e-8
    ):
        raise ValueError("Rotation matrix must be orthonormal with determinant +1.")
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, resolved.reshape(-1))
    quat = _normalize_quat_wxyz(quat, label="rotation matrix quaternion")
    return -quat if quat[0] < 0.0 else quat


def _rotation_x(angle_rad: float) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _strict_object_keys(
    value: Any,
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}.")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {unknown}.")
    return value


def _finite_float(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return resolved


def _configured_fovy_randomization(
    value: Any,
    *,
    label: str,
    configured_fovy: float,
) -> dict[str, float | str]:
    profile = _strict_object_keys(
        value,
        label=label,
        required={"center", "half_range", "distribution"},
    )
    if str(profile["distribution"]) != "uniform_symmetric":
        raise ValueError(f"{label}.distribution must be 'uniform_symmetric'.")
    center = _finite_float(profile["center"], label=f"{label}.center", minimum=1e-6, maximum=179.999)
    half_range = _finite_float(
        profile["half_range"],
        label=f"{label}.half_range",
        minimum=0.0,
        maximum=89.0,
    )
    if center - half_range <= 0.0 or center + half_range >= 180.0:
        raise ValueError(f"{label} must stay strictly inside (0, 180) degrees.")
    if not math.isclose(center, configured_fovy, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{label}.center={center} must equal the camera intrinsics fovy={configured_fovy}."
        )
    return {
        "center": center,
        "half_range": half_range,
        "distribution": "uniform_symmetric",
    }


def _configured_wrist_camera_randomization(
    camera: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    label = f"camera {camera.get('name')!r} randomization"
    _strict_object_keys(
        raw,
        label=label,
        required={
            "enabled",
            "mode",
            "position_half_ranges_m",
            "orientation_half_ranges_deg",
            "fovy_deg",
        },
    )
    if str((camera.get("mount") or {}).get("type") or "") != "body":
        raise ValueError(f"{label} mode {WRIST_CAMERA_RANDOMIZATION_MODE!r} requires a body-mounted camera.")
    position_raw = _strict_object_keys(
        raw["position_half_ranges_m"],
        label=f"{label}.position_half_ranges_m",
        required={"lateral", "vertical", "depth", "distribution"},
    )
    if str(position_raw["distribution"]) != "independent_uniform_symmetric":
        raise ValueError(
            f"{label}.position_half_ranges_m.distribution must be "
            "'independent_uniform_symmetric'."
        )
    orientation_raw = _strict_object_keys(
        raw["orientation_half_ranges_deg"],
        label=f"{label}.orientation_half_ranges_deg",
        required={"roll", "pitch", "yaw", "distribution", "composition"},
    )
    if str(orientation_raw["distribution"]) != "independent_uniform_symmetric":
        raise ValueError(
            f"{label}.orientation_half_ranges_deg.distribution must be "
            "'independent_uniform_symmetric'."
        )
    if str(orientation_raw["composition"]) != "camera_dr_basis_rz_yaw_ry_pitch_rx_roll_postmultiply":
        raise ValueError(
            f"{label}.orientation_half_ranges_deg.composition has an unsupported value."
        )
    configured_fovy = _camera_fovy(camera)
    return {
        "enabled": bool(raw["enabled"]),
        "mode": WRIST_CAMERA_RANDOMIZATION_MODE,
        "position_half_ranges_m": {
            axis: _finite_float(
                position_raw[axis],
                label=f"{label}.position_half_ranges_m.{axis}",
                minimum=0.0,
            )
            for axis in ("lateral", "vertical", "depth")
        }
        | {"distribution": "independent_uniform_symmetric"},
        "orientation_half_ranges_deg": {
            axis: _finite_float(
                orientation_raw[axis],
                label=f"{label}.orientation_half_ranges_deg.{axis}",
                minimum=0.0,
                maximum=45.0,
            )
            for axis in ("roll", "pitch", "yaw")
        }
        | {
            "distribution": "independent_uniform_symmetric",
            "composition": "camera_dr_basis_rz_yaw_ry_pitch_rx_roll_postmultiply",
        },
        "fovy_deg": _configured_fovy_randomization(
            raw["fovy_deg"],
            label=f"{label}.fovy_deg",
            configured_fovy=configured_fovy,
        ),
        "coordinate_convention": {
            "camera_dr_basis": ["depth_forward", "lateral_camera_left", "vertical_camera_up"],
            "mujoco_camera_basis": ["x_image_right", "y_image_up", "negative_z_forward"],
        },
    }


def _configured_front_camera_randomization(
    camera: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    label = f"camera {camera.get('name')!r} randomization"
    _strict_object_keys(
        raw,
        label=label,
        required={
            "enabled",
            "mode",
            "azimuth",
            "table_edge_clearance_m",
            "height_offset_in_reference_frame_m",
            "look_at",
            "fovy_deg",
            "visibility_gate",
        },
    )
    if str((camera.get("mount") or {}).get("type") or "") != "world":
        raise ValueError(f"{label} mode {FRONT_CAMERA_RANDOMIZATION_MODE!r} requires a world camera.")
    extrinsics = camera.get("extrinsics") or {}
    if str(extrinsics.get("type") or "") != "body_frame_relative_lookat":
        raise ValueError(
            f"{label} mode {FRONT_CAMERA_RANDOMIZATION_MODE!r} requires "
            "body_frame_relative_lookat extrinsics."
        )
    azimuth_raw = _strict_object_keys(
        raw["azimuth"],
        label=f"{label}.azimuth",
        required={"bounds", "distribution"},
    )
    if str(azimuth_raw["bounds"]) != "reference_base_to_table_positive_x_edge_corners":
        raise ValueError(f"{label}.azimuth.bounds has an unsupported value.")
    if str(azimuth_raw["distribution"]) != "uniform_angle":
        raise ValueError(f"{label}.azimuth.distribution must be 'uniform_angle'.")
    clearance_raw = _strict_object_keys(
        raw["table_edge_clearance_m"],
        label=f"{label}.table_edge_clearance_m",
        required={"minimum", "maximum", "distribution"},
    )
    if str(clearance_raw["distribution"]) != "uniform":
        raise ValueError(f"{label}.table_edge_clearance_m.distribution must be 'uniform'.")
    clearance_min = _finite_float(
        clearance_raw["minimum"], label=f"{label}.table_edge_clearance_m.minimum", minimum=0.0
    )
    clearance_max = _finite_float(
        clearance_raw["maximum"], label=f"{label}.table_edge_clearance_m.maximum", minimum=clearance_min
    )
    height_raw = _strict_object_keys(
        raw["height_offset_in_reference_frame_m"],
        label=f"{label}.height_offset_in_reference_frame_m",
        required={"center", "half_range", "distribution"},
    )
    if str(height_raw["distribution"]) != "uniform_symmetric":
        raise ValueError(
            f"{label}.height_offset_in_reference_frame_m.distribution must be 'uniform_symmetric'."
        )
    height_center = _finite_float(
        height_raw["center"], label=f"{label}.height_offset_in_reference_frame_m.center"
    )
    height_half_range = _finite_float(
        height_raw["half_range"],
        label=f"{label}.height_offset_in_reference_frame_m.half_range",
        minimum=0.0,
    )
    position_offset = np.asarray(extrinsics.get("position_offset") or [], dtype=np.float64)
    if position_offset.shape != (3,) or not math.isclose(
        height_center, float(position_offset[2]), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"{label} height center must equal extrinsics.position_offset.z={position_offset.tolist()}."
        )
    look_at_raw = _strict_object_keys(
        raw["look_at"],
        label=f"{label}.look_at",
        required={"center_offset_in_reference_frame_m", "disk_plane", "disk_diameter_m", "distribution"},
    )
    if str(look_at_raw["disk_plane"]) != "reference_xy":
        raise ValueError(f"{label}.look_at.disk_plane must be 'reference_xy'.")
    if str(look_at_raw["distribution"]) != "uniform_area":
        raise ValueError(f"{label}.look_at.distribution must be 'uniform_area'.")
    look_at_center = np.asarray(look_at_raw["center_offset_in_reference_frame_m"], dtype=np.float64)
    target_offset = np.asarray(extrinsics.get("target_offset") or [], dtype=np.float64)
    if look_at_center.shape != (3,) or not np.all(np.isfinite(look_at_center)):
        raise ValueError(f"{label}.look_at.center_offset_in_reference_frame_m must have three finite values.")
    if target_offset.shape != (3,) or not np.allclose(look_at_center, target_offset, atol=1e-9):
        raise ValueError(
            f"{label} look-at center must equal extrinsics.target_offset={target_offset.tolist()}."
        )
    disk_diameter = _finite_float(
        look_at_raw["disk_diameter_m"],
        label=f"{label}.look_at.disk_diameter_m",
        minimum=0.0,
    )
    visibility_raw = _strict_object_keys(
        raw["visibility_gate"],
        label=f"{label}.visibility_gate",
        required={
            "max_attempts",
            "workspace_center_offset_in_reference_frame_m",
            "horizontal_radius_m",
            "height_range_in_reference_frame_m",
            "azimuth_samples",
            "render_aspect_ratio",
        },
    )
    max_attempts = int(visibility_raw["max_attempts"])
    azimuth_samples = int(visibility_raw["azimuth_samples"])
    if max_attempts < 1:
        raise ValueError(f"{label}.visibility_gate.max_attempts must be >= 1.")
    if azimuth_samples < 4:
        raise ValueError(f"{label}.visibility_gate.azimuth_samples must be >= 4.")
    workspace_center = np.asarray(
        visibility_raw["workspace_center_offset_in_reference_frame_m"], dtype=np.float64
    )
    height_range = np.asarray(visibility_raw["height_range_in_reference_frame_m"], dtype=np.float64)
    aspect_ratio = np.asarray(visibility_raw["render_aspect_ratio"], dtype=np.float64)
    if workspace_center.shape != (3,) or not np.all(np.isfinite(workspace_center)):
        raise ValueError(f"{label}.visibility_gate workspace center must have three finite values.")
    if height_range.shape != (2,) or not np.all(np.isfinite(height_range)) or height_range[1] < height_range[0]:
        raise ValueError(f"{label}.visibility_gate height range must be [minimum, maximum].")
    if aspect_ratio.shape != (2,) or np.any(aspect_ratio <= 0.0):
        raise ValueError(f"{label}.visibility_gate.render_aspect_ratio must contain two positive values.")
    required_aspect = str((camera.get("intrinsics") or {}).get("required_aspect_ratio") or "").strip()
    required_match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*)", required_aspect)
    if required_match is None:
        raise ValueError(f"{label} camera intrinsics must declare required_aspect_ratio as W:H.")
    required_ratio = float(required_match.group(1)) / float(required_match.group(2))
    configured_gate_ratio = float(aspect_ratio[0]) / float(aspect_ratio[1])
    if not math.isclose(configured_gate_ratio, required_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{label}.visibility_gate.render_aspect_ratio must match intrinsics.required_aspect_ratio."
        )
    configured_fovy = _camera_fovy(camera)
    return {
        "enabled": bool(raw["enabled"]),
        "mode": FRONT_CAMERA_RANDOMIZATION_MODE,
        "azimuth": {
            "bounds": "reference_base_to_table_positive_x_edge_corners",
            "distribution": "uniform_angle",
        },
        "table_edge_clearance_m": {
            "minimum": clearance_min,
            "maximum": clearance_max,
            "distribution": "uniform",
        },
        "height_offset_in_reference_frame_m": {
            "center": height_center,
            "half_range": height_half_range,
            "distribution": "uniform_symmetric",
        },
        "look_at": {
            "center_offset_in_reference_frame_m": look_at_center.tolist(),
            "disk_plane": "reference_xy",
            "disk_diameter_m": disk_diameter,
            "distribution": "uniform_area",
        },
        "fovy_deg": _configured_fovy_randomization(
            raw["fovy_deg"],
            label=f"{label}.fovy_deg",
            configured_fovy=configured_fovy,
        ),
        "visibility_gate": {
            "max_attempts": max_attempts,
            "workspace_center_offset_in_reference_frame_m": workspace_center.tolist(),
            "horizontal_radius_m": _finite_float(
                visibility_raw["horizontal_radius_m"],
                label=f"{label}.visibility_gate.horizontal_radius_m",
                minimum=0.0,
            ),
            "height_range_in_reference_frame_m": height_range.tolist(),
            "azimuth_samples": azimuth_samples,
            "render_aspect_ratio": aspect_ratio.tolist(),
        },
    }


def _camera_randomization_by_name(camera_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for camera in camera_config["cameras"]:
        randomization = deepcopy(camera.get("randomization") or {})
        if not isinstance(randomization, dict):
            raise ValueError(f"camera {camera.get('name')!r} randomization must be an object.")
        mode = str(randomization.get("mode") or "").strip()
        if mode == WRIST_CAMERA_RANDOMIZATION_MODE:
            randomization = _configured_wrist_camera_randomization(camera, randomization)
        elif mode == FRONT_CAMERA_RANDOMIZATION_MODE:
            randomization = _configured_front_camera_randomization(camera, randomization)
        elif mode:
            raise ValueError(f"Unsupported camera randomization mode {mode!r}.")
        else:
            randomization.setdefault("enabled", False)
            randomization.setdefault("randomize_position", bool(randomization["enabled"]))
            randomization.setdefault("randomize_rotation", bool(randomization["enabled"]))
            randomization.setdefault("randomize_fovy", bool(randomization["enabled"]))
            randomization.setdefault("position_perturbation_size", 0.0)
            randomization.setdefault("rotation_perturbation_size", 0.0)
            randomization.setdefault("fovy_perturbation_size", 0.0)
        result[str(camera["name"])] = randomization
    return result


def _camera_field_uniform(
    camera_seed: int,
    field_name: str,
    minimum: float,
    maximum: float,
) -> tuple[float, int]:
    field_seed = stable_seed(camera_seed, field_name)
    random_state = np.random.RandomState(field_seed)
    return float(random_state.uniform(float(minimum), float(maximum))), int(field_seed)


def _sample_wrist_optical_pose_box(
    *,
    base_position: np.ndarray,
    base_quat_wxyz: np.ndarray,
    base_fovy_deg: float,
    profile: dict[str, Any],
    camera_seed: int,
) -> dict[str, Any]:
    if str(profile.get("mode") or "") != WRIST_CAMERA_RANDOMIZATION_MODE:
        raise ValueError(f"Expected wrist randomization mode {WRIST_CAMERA_RANDOMIZATION_MODE!r}.")
    base_position = np.asarray(base_position, dtype=np.float64)
    if base_position.shape != (3,) or not np.all(np.isfinite(base_position)):
        raise ValueError("Wrist base camera position must have three finite values.")
    base_quat = _normalize_quat_wxyz(base_quat_wxyz, label="wrist base camera quaternion")
    base_rotation = _quat_to_matrix_wxyz(base_quat)
    position_ranges = profile["position_half_ranges_m"]
    orientation_ranges = profile["orientation_half_ranges_deg"]
    fovy_profile = profile["fovy_deg"]
    if not math.isclose(float(base_fovy_deg), float(fovy_profile["center"]), abs_tol=1e-9):
        raise ValueError(
            f"Wrist base fovy={base_fovy_deg} does not match configured center={fovy_profile['center']}."
        )

    sampled: dict[str, float] = {}
    field_seeds: dict[str, int] = {}
    for axis in ("depth", "lateral", "vertical"):
        half_range = float(position_ranges[axis])
        sampled[f"{axis}_m"], field_seeds[f"position_{axis}"] = _camera_field_uniform(
            camera_seed,
            f"position_{axis}",
            -half_range,
            half_range,
        )
    for axis in ("roll", "pitch", "yaw"):
        half_range = float(orientation_ranges[axis])
        sampled[f"{axis}_deg"], field_seeds[f"orientation_{axis}"] = _camera_field_uniform(
            camera_seed,
            f"orientation_{axis}",
            -half_range,
            half_range,
        )
    fovy_delta, field_seeds["fovy"] = _camera_field_uniform(
        camera_seed,
        "fovy",
        -float(fovy_profile["half_range"]),
        float(fovy_profile["half_range"]),
    )
    resolved_fovy = float(fovy_profile["center"]) + fovy_delta

    # axis-training uses the right-handed convention (+X forward, +Z up), so
    # its +Y lateral axis points camera-left. MuJoCo/OpenGL camera coordinates
    # are (+X image-right, +Y image-up, -Z forward).
    dr_basis_to_mujoco = np.asarray(
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    if not math.isclose(float(np.linalg.det(dr_basis_to_mujoco)), 1.0, abs_tol=1e-12):
        raise ValueError("Camera DR basis conversion must be a proper right-handed rotation.")
    position_delta_dr = np.asarray(
        [sampled["depth_m"], sampled["lateral_m"], sampled["vertical_m"]],
        dtype=np.float64,
    )
    position_delta_camera = dr_basis_to_mujoco @ position_delta_dr
    position_delta_parent = base_rotation @ position_delta_camera
    resolved_position = base_position + position_delta_parent

    roll = math.radians(sampled["roll_deg"])
    pitch = math.radians(sampled["pitch_deg"])
    yaw = math.radians(sampled["yaw_deg"])
    local_rotation_dr = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    local_rotation_mujoco = dr_basis_to_mujoco @ local_rotation_dr @ dr_basis_to_mujoco.T
    resolved_rotation = base_rotation @ local_rotation_mujoco
    resolved_quat = _matrix_to_quat_wxyz(resolved_rotation)

    return {
        "mode": WRIST_CAMERA_RANDOMIZATION_MODE,
        "camera_seed": int(camera_seed),
        "field_seeds": field_seeds,
        "sampling_frequency": "once_per_trajectory_variant",
        "coordinate_frame": "camera_optical_local_then_panda_link8_local",
        "base": {
            "position": base_position.tolist(),
            "quat_wxyz": base_quat.tolist(),
            "fovy_deg": float(base_fovy_deg),
        },
        "sampled": sampled | {"fovy_delta_deg": float(fovy_delta)},
        "resolved": {
            "position": resolved_position.tolist(),
            "quat_wxyz": resolved_quat.tolist(),
            "fovy_deg": resolved_fovy,
            "position_delta_camera_mujoco_xyz": position_delta_camera.tolist(),
            "position_delta_parent": position_delta_parent.tolist(),
            "rotation_delta_camera_dr_basis": local_rotation_dr.tolist(),
            "rotation_delta_camera_mujoco": local_rotation_mujoco.tolist(),
            "rotation_delta_camera_mujoco_quat_wxyz": _matrix_to_quat_wxyz(
                local_rotation_mujoco
            ).tolist(),
        },
    }


def _front_table_sector_geometry(
    *,
    base_world_position: np.ndarray,
    base_world_quat_wxyz: np.ndarray,
    table_center_xy_world: tuple[float, float] | list[float],
    table_top_z_world: float,
    table_full_size: tuple[float, float, float] | list[float],
) -> dict[str, Any]:
    base_position = np.asarray(base_world_position, dtype=np.float64)
    base_quat = _normalize_quat_wxyz(base_world_quat_wxyz, label="front reference base quaternion")
    table_center = np.asarray(table_center_xy_world, dtype=np.float64)
    full_size = np.asarray(table_full_size, dtype=np.float64)
    if base_position.shape != (3,) or table_center.shape != (2,) or full_size.shape != (3,):
        raise ValueError("Front camera geometry requires base xyz, table center xy, and table full size xyz.")
    if not np.all(np.isfinite(base_position)) or not np.all(np.isfinite(table_center)):
        raise ValueError("Front camera geometry inputs must be finite.")
    if np.any(full_size <= 0.0) or not np.all(np.isfinite(full_size)):
        raise ValueError("Front camera table full size must contain three positive finite values.")
    base_up_world = _quat_rotate_wxyz(base_quat, np.asarray([0.0, 0.0, 1.0]))
    if not np.allclose(base_up_world, [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(
            "Front table-sector randomization requires an upright Franka base; yaw is supported, "
            f"but the resolved base +Z axis is {base_up_world.tolist()}."
        )
    inverse_base_quat = base_quat * np.asarray([1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    half_x = float(full_size[0]) * 0.5
    half_y = float(full_size[1]) * 0.5
    world_corners = np.asarray(
        [
            [table_center[0] - half_x, table_center[1] - half_y, table_top_z_world],
            [table_center[0] - half_x, table_center[1] + half_y, table_top_z_world],
            [table_center[0] + half_x, table_center[1] + half_y, table_top_z_world],
            [table_center[0] + half_x, table_center[1] - half_y, table_top_z_world],
        ],
        dtype=np.float64,
    )
    base_corners = np.stack(
        [_quat_rotate_wxyz(inverse_base_quat, corner - base_position) for corner in world_corners],
        axis=0,
    )
    edge_indexes = ((0, 1), (1, 2), (2, 3), (3, 0))
    selected_edge = max(
        edge_indexes,
        key=lambda edge: float(np.mean(base_corners[list(edge), 0])),
    )
    edge_start = base_corners[selected_edge[0], :2]
    edge_end = base_corners[selected_edge[1], :2]
    if float(np.mean([edge_start[0], edge_end[0]])) <= 0.0:
        raise ValueError("The table has no positive-X edge in the configured Franka base frame.")
    start_angle = math.atan2(float(edge_start[1]), float(edge_start[0]))
    end_angle = math.atan2(float(edge_end[1]), float(edge_end[0]))
    short_arc = math.atan2(math.sin(end_angle - start_angle), math.cos(end_angle - start_angle))
    if abs(short_arc) <= 1e-8 or abs(short_arc) >= math.pi:
        raise ValueError("Table far-edge corner rays do not define a valid short azimuth sector.")
    return {
        "base_world_position": base_position.tolist(),
        "base_world_quat_wxyz": base_quat.tolist(),
        "table_world_corners": world_corners.tolist(),
        "table_corners_in_reference_frame": base_corners.tolist(),
        "far_edge_corner_indexes": list(selected_edge),
        "far_edge_start_xy": edge_start.tolist(),
        "far_edge_end_xy": edge_end.tolist(),
        "azimuth_start_rad": start_angle,
        "azimuth_short_arc_rad": short_arc,
        "azimuth_end_unwrapped_rad": start_angle + short_arc,
        "azimuth_corner_degrees": [math.degrees(start_angle), math.degrees(start_angle + short_arc)],
    }


def _ray_segment_intersection_distance(
    direction_xy: np.ndarray,
    segment_start_xy: np.ndarray,
    segment_end_xy: np.ndarray,
) -> tuple[float, float]:
    direction = np.asarray(direction_xy, dtype=np.float64)
    start = np.asarray(segment_start_xy, dtype=np.float64)
    end = np.asarray(segment_end_xy, dtype=np.float64)
    matrix = np.column_stack([direction, -(end - start)])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= 1e-10:
        raise ValueError("Camera azimuth ray is parallel to the selected table edge.")
    distance, segment_fraction = np.linalg.solve(matrix, start)
    if distance <= 0.0 or segment_fraction < -1e-7 or segment_fraction > 1.0 + 1e-7:
        raise ValueError(
            "Camera azimuth ray did not intersect the selected table edge segment: "
            f"distance={distance}, segment_fraction={segment_fraction}."
        )
    return float(distance), float(np.clip(segment_fraction, 0.0, 1.0))


def _front_workspace_points_in_reference_frame(profile: dict[str, Any]) -> np.ndarray:
    visibility = profile["visibility_gate"]
    center = np.asarray(visibility["workspace_center_offset_in_reference_frame_m"], dtype=np.float64)
    radius = float(visibility["horizontal_radius_m"])
    height_min, height_max = (float(value) for value in visibility["height_range_in_reference_frame_m"])
    count = int(visibility["azimuth_samples"])
    points: list[np.ndarray] = []
    for height in (height_min, height_max):
        resolved_height = float(center[2]) + height
        points.append(np.asarray([center[0], center[1], resolved_height], dtype=np.float64))
        for angle in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False):
            points.append(
                np.asarray(
                    [
                        center[0] + radius * math.cos(angle),
                        center[1] + radius * math.sin(angle),
                        resolved_height,
                    ],
                    dtype=np.float64,
                )
            )
    return np.stack(points, axis=0)


def _camera_visible_point_count(
    *,
    camera_position: np.ndarray,
    look_at: np.ndarray,
    fovy_deg: float,
    aspect_ratio: float,
    points: np.ndarray,
) -> int:
    position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(look_at, dtype=np.float64)
    forward = target - position
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    xyaxes = _camera_xyaxes(position, target)
    right = xyaxes[:3]
    up = xyaxes[3:]
    vertical_limit = math.radians(float(fovy_deg)) * 0.5
    horizontal_limit = math.atan(math.tan(vertical_limit) * float(aspect_ratio))
    visible = 0
    for point in np.asarray(points, dtype=np.float64):
        relative = point - position
        depth = float(np.dot(relative, forward))
        if depth <= 1e-8:
            continue
        horizontal_angle = abs(math.atan2(float(np.dot(relative, right)), depth))
        vertical_angle = abs(math.atan2(float(np.dot(relative, up)), depth))
        if horizontal_angle <= horizontal_limit and vertical_angle <= vertical_limit:
            visible += 1
    return visible


def _lookat_quat_wxyz(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    xyaxes = _camera_xyaxes(position, target)
    right = xyaxes[:3]
    up = xyaxes[3:]
    backward = np.cross(right, up)
    return _matrix_to_quat_wxyz(np.column_stack([right, up, backward]))


def _sample_front_base_table_sector_lookat(
    *,
    profile: dict[str, Any],
    camera_seed: int,
    base_world_position: np.ndarray,
    base_world_quat_wxyz: np.ndarray,
    table_center_xy_world: tuple[float, float] | list[float],
    table_top_z_world: float,
    table_full_size: tuple[float, float, float] | list[float],
) -> dict[str, Any]:
    if str(profile.get("mode") or "") != FRONT_CAMERA_RANDOMIZATION_MODE:
        raise ValueError(f"Expected front randomization mode {FRONT_CAMERA_RANDOMIZATION_MODE!r}.")
    geometry = _front_table_sector_geometry(
        base_world_position=base_world_position,
        base_world_quat_wxyz=base_world_quat_wxyz,
        table_center_xy_world=table_center_xy_world,
        table_top_z_world=table_top_z_world,
        table_full_size=table_full_size,
    )
    base_position = np.asarray(geometry["base_world_position"], dtype=np.float64)
    base_quat = np.asarray(geometry["base_world_quat_wxyz"], dtype=np.float64)
    edge_start = np.asarray(geometry["far_edge_start_xy"], dtype=np.float64)
    edge_end = np.asarray(geometry["far_edge_end_xy"], dtype=np.float64)
    start_angle = float(geometry["azimuth_start_rad"])
    short_arc = float(geometry["azimuth_short_arc_rad"])
    clearance_profile = profile["table_edge_clearance_m"]
    height_profile = profile["height_offset_in_reference_frame_m"]
    look_at_profile = profile["look_at"]
    fovy_profile = profile["fovy_deg"]
    visibility_profile = profile["visibility_gate"]
    aspect_ratio = float(visibility_profile["render_aspect_ratio"][0]) / float(
        visibility_profile["render_aspect_ratio"][1]
    )
    workspace_points_reference = _front_workspace_points_in_reference_frame(profile)
    workspace_points_world = np.stack(
        [base_position + _quat_rotate_wxyz(base_quat, point) for point in workspace_points_reference],
        axis=0,
    )
    best_visible_count = -1
    total_points = int(workspace_points_world.shape[0])
    for attempt_index in range(int(visibility_profile["max_attempts"])):
        prefix = f"attempt_{attempt_index}"
        field_seeds: dict[str, int] = {}
        azimuth_fraction, field_seeds["azimuth"] = _camera_field_uniform(
            camera_seed, f"{prefix}_azimuth", 0.0, 1.0
        )
        azimuth_unwrapped = start_angle + short_arc * azimuth_fraction
        azimuth = math.atan2(math.sin(azimuth_unwrapped), math.cos(azimuth_unwrapped))
        direction = np.asarray([math.cos(azimuth), math.sin(azimuth)], dtype=np.float64)
        edge_distance, edge_segment_fraction = _ray_segment_intersection_distance(
            direction,
            edge_start,
            edge_end,
        )
        clearance, field_seeds["table_edge_clearance"] = _camera_field_uniform(
            camera_seed,
            f"{prefix}_table_edge_clearance",
            float(clearance_profile["minimum"]),
            float(clearance_profile["maximum"]),
        )
        height_delta, field_seeds["height"] = _camera_field_uniform(
            camera_seed,
            f"{prefix}_height",
            -float(height_profile["half_range"]),
            float(height_profile["half_range"]),
        )
        radius_unit, field_seeds["look_at_radius"] = _camera_field_uniform(
            camera_seed, f"{prefix}_look_at_radius", 0.0, 1.0
        )
        disk_angle, field_seeds["look_at_angle"] = _camera_field_uniform(
            camera_seed, f"{prefix}_look_at_angle", -math.pi, math.pi
        )
        fovy_delta, field_seeds["fovy"] = _camera_field_uniform(
            camera_seed,
            f"{prefix}_fovy",
            -float(fovy_profile["half_range"]),
            float(fovy_profile["half_range"]),
        )
        camera_radius = edge_distance + clearance
        camera_position_reference = np.asarray(
            [
                camera_radius * direction[0],
                camera_radius * direction[1],
                float(height_profile["center"]) + height_delta,
            ],
            dtype=np.float64,
        )
        disk_radius = float(look_at_profile["disk_diameter_m"]) * 0.5
        sampled_look_at_radius = disk_radius * math.sqrt(radius_unit)
        look_at_reference = np.asarray(
            look_at_profile["center_offset_in_reference_frame_m"], dtype=np.float64
        ).copy()
        look_at_reference[:2] += sampled_look_at_radius * np.asarray(
            [math.cos(disk_angle), math.sin(disk_angle)], dtype=np.float64
        )
        camera_position_world = base_position + _quat_rotate_wxyz(base_quat, camera_position_reference)
        look_at_world = base_position + _quat_rotate_wxyz(base_quat, look_at_reference)
        fovy = float(fovy_profile["center"]) + fovy_delta
        visible_count = _camera_visible_point_count(
            camera_position=camera_position_world,
            look_at=look_at_world,
            fovy_deg=fovy,
            aspect_ratio=aspect_ratio,
            points=workspace_points_world,
        )
        best_visible_count = max(best_visible_count, visible_count)
        if visible_count != total_points:
            continue
        camera_quat = _lookat_quat_wxyz(camera_position_world, look_at_world)
        camera_to_look_at_reference = look_at_reference - camera_position_reference
        return {
            "mode": FRONT_CAMERA_RANDOMIZATION_MODE,
            "camera_seed": int(camera_seed),
            "field_seeds": field_seeds,
            "sampling_frequency": "once_per_trajectory_variant",
            "candidate_distribution": "independent_configured_uniforms",
            "accepted_distribution": "candidate_distribution_conditioned_on_visibility_gate",
            "coordinate_frame": "franka_reference_base_then_world",
            "attempt_index": int(attempt_index),
            "visibility": {
                "required_point_count": total_points,
                "visible_point_count": visible_count,
                "workspace_points_in_reference_frame": workspace_points_reference.tolist(),
                "aspect_ratio": aspect_ratio,
            },
            "geometry": geometry,
            "sampled": {
                "azimuth_fraction": azimuth_fraction,
                "azimuth_rad": azimuth,
                "azimuth_deg": math.degrees(azimuth),
                "table_edge_intersection_distance_m": edge_distance,
                "table_edge_segment_fraction": edge_segment_fraction,
                "table_edge_clearance_m": clearance,
                "camera_horizontal_radius_from_base_m": camera_radius,
                "height_delta_m": height_delta,
                "look_at_disk_radius_m": sampled_look_at_radius,
                "look_at_disk_angle_rad": disk_angle,
                "fovy_delta_deg": fovy_delta,
                "camera_to_look_at_horizontal_distance_m": float(
                    np.linalg.norm(camera_to_look_at_reference[:2])
                ),
                "camera_to_look_at_distance_m": float(np.linalg.norm(camera_to_look_at_reference)),
            },
            "resolved": {
                "position_in_reference_frame": camera_position_reference.tolist(),
                "look_at_in_reference_frame": look_at_reference.tolist(),
                "world_position": camera_position_world.tolist(),
                "world_look_at": look_at_world.tolist(),
                "world_quat_wxyz": camera_quat.tolist(),
                "fovy_deg": fovy,
            },
        }
    raise ValueError(
        "Front camera visibility gate exhausted all deterministic candidates: "
        f"attempts={visibility_profile['max_attempts']}, "
        f"best_visible={best_visible_count}/{total_points}."
    )


def _absolutize_xml_file_attrs(root: ET.Element, *, xml_dir: Path) -> dict[str, Any]:
    resolved: list[dict[str, str]] = []
    resolved_count = 0
    for elem in root.iter():
        if elem.tag not in XML_FILE_ATTR_TAGS:
            continue
        raw_file = elem.get("file")
        if not raw_file or raw_file.startswith(("http://", "https://", "s3://")):
            continue
        raw_path = Path(raw_file)
        if raw_path.is_absolute():
            continue
        resolved_path = (xml_dir / raw_path).resolve()
        elem.set("file", str(resolved_path))
        resolved_count += 1
        if len(resolved) < 40:
            resolved.append({"tag": elem.tag, "from": raw_file, "to": str(resolved_path)})
    return {"resolved_file_attr_count": resolved_count, "resolved_file_attr_examples": resolved}


def _parse_vec(raw: str | None, *, length: int, default: tuple[float, ...] | None = None) -> np.ndarray:
    if raw is None or not str(raw).strip():
        if default is None:
            return np.zeros(length, dtype=np.float64)
        return np.asarray(default, dtype=np.float64)
    values = [float(part) for part in str(raw).split()]
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)} in {raw!r}.")
    return np.asarray(values, dtype=np.float64)


def _set_body_pos(body: ET.Element, pos: np.ndarray) -> None:
    body.set("pos", _format_float_sequence(np.asarray(pos, dtype=np.float64)))


def _body_has_direct_free_joint(body: ET.Element) -> bool:
    for child in list(body):
        if child.tag == "freejoint":
            return True
        if child.tag == "joint" and child.get("type") == "free":
            return True
    return False


def _name_contains(name: str, needles: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(needle in lower for needle in needles)


def _is_object_free_body(body: ET.Element) -> bool:
    name = body.get("name") or ""
    if not _body_has_direct_free_joint(body):
        return False
    if _name_contains(name, ARM_ROBOT_BODY_NEEDLES):
        return False
    return _name_contains(name, OBJECT_BODY_NAME_NEEDLES)


def _direct_worldbody_bodies(root: ET.Element) -> list[ET.Element]:
    bodies: list[ET.Element] = []
    for worldbody in root.findall("worldbody"):
        bodies.extend([child for child in list(worldbody) if child.tag == "body"])
    return bodies


def _direct_object_body_positions(root: ET.Element) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for body in _direct_worldbody_bodies(root):
        if _is_object_free_body(body):
            positions.append(_parse_vec(body.get("pos"), length=3, default=(0.0, 0.0, 0.0)))
    return positions


def _direct_arm_robot_body_positions(root: ET.Element) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for body in _direct_worldbody_bodies(root):
        name = body.get("name") or ""
        if _name_contains(name, ARM_ROBOT_BODY_NEEDLES):
            positions.append(_parse_vec(body.get("pos"), length=3, default=(0.0, 0.0, 0.0)))
    return positions


def _extract_original_table_metadata(root: ET.Element) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for body in _direct_worldbody_bodies(root):
        if not _is_original_scene_shell_body(body):
            continue
        body_pos = _parse_vec(body.get("pos"), length=3, default=(0.0, 0.0, 0.0))
        top_candidates: list[dict[str, Any]] = []
        for geom in body.findall("geom"):
            name = geom.get("name") or ""
            lower = name.lower()
            if "top" not in lower and "surface" not in lower:
                continue
            if geom.get("size") is None:
                continue
            geom_pos = _parse_vec(geom.get("pos"), length=3, default=(0.0, 0.0, 0.0))
            size = _parse_vec(geom.get("size"), length=3, default=(0.0, 0.0, 0.0))
            top_candidates.append(
                {
                    "geom_name": name,
                    "center": (body_pos + geom_pos).tolist(),
                    "top_z": float(body_pos[2] + geom_pos[2] + size[2]),
                    "full_size": [float(size[0] * 2.0), float(size[1] * 2.0), float(size[2] * 2.0)],
                }
            )
        if not top_candidates:
            continue
        top = max(top_candidates, key=lambda item: float(item["top_z"]))
        tables.append(
            {
                "body_name": body.get("name") or "",
                "body_pos": body_pos.tolist(),
                "top_geom": top,
            }
        )
    return tables


def _infer_table_center_xy(root: ET.Element, *, include_robot: bool) -> tuple[float, float]:
    samples = _direct_object_body_positions(root)
    if include_robot:
        samples.extend(_direct_arm_robot_body_positions(root))
    if not samples:
        return DEFAULT_TABLE_CENTER_XY
    stacked = np.stack(samples, axis=0)
    center = np.mean(stacked[:, :2], axis=0)
    return (float(center[0]), float(center[1]))


def _infer_table_full_size(root: ET.Element, *, min_size: tuple[float, float, float]) -> tuple[float, float, float]:
    positions = _direct_object_body_positions(root)
    if not positions:
        return min_size
    stacked = np.stack(positions, axis=0)
    span = np.max(stacked[:, :2], axis=0) - np.min(stacked[:, :2], axis=0)
    return (
        float(max(min_size[0], span[0] + TABLE_TOP_MARGIN * 2.0)),
        float(max(min_size[1], span[1] + TABLE_TOP_MARGIN * 2.0)),
        float(min_size[2]),
    )


def _direct_free_object_body_metadata(root: ET.Element) -> list[dict[str, Any]]:
    """Return every direct free body that is not an embodiment root.

    The table-size rule recognizes a fixed set of object-name tokens.
    Fixed-table containment is stricter: a knife, cutting board, or any other
    free object is required to start above the physical tabletop even when
    its name is not in that set.
    """

    objects: list[dict[str, Any]] = []
    for body in _direct_worldbody_bodies(root):
        name = body.get("name") or ""
        if not _body_has_direct_free_joint(body):
            continue
        if _name_contains(name, ARM_ROBOT_BODY_NEEDLES):
            continue
        objects.append(
            {
                "body_name": name,
                "center": _parse_vec(
                    body.get("pos"), length=3, default=(0.0, 0.0, 0.0)
                ).tolist(),
            }
        )
    return objects


def _fixed_body_frame_table_placement(
    root: ET.Element,
    placement_profile: dict[str, Any],
    *,
    dynamic_table_center_xy: tuple[float, float],
    dynamic_required_table_full_size: tuple[float, float, float],
) -> tuple[tuple[float, float], tuple[float, float, float], dict[str, Any]]:
    if not isinstance(placement_profile, dict):
        raise ValueError("render_profile.scene.table_placement must be a JSON object.")
    expected_fields = {
        "mode",
        "reference_body",
        "center_offset_xy_in_reference_frame_m",
        "full_size_m",
        "object_center_policy",
    }
    unknown_fields = sorted(set(placement_profile) - expected_fields)
    missing_fields = sorted(expected_fields - set(placement_profile))
    if unknown_fields or missing_fields:
        raise ValueError(
            "render_profile.scene.table_placement fields must match the fixed-table contract: "
            f"missing={missing_fields}, unknown={unknown_fields}."
        )

    mode = str(placement_profile["mode"] or "").strip()
    if mode != "fixed_body_frame":
        raise ValueError(
            "render_profile.scene.table_placement.mode must be 'fixed_body_frame', "
            f"got {mode!r}."
        )
    object_center_policy = str(placement_profile["object_center_policy"] or "").strip()
    if object_center_policy != "require_inside_fixed_footprint":
        raise ValueError(
            "render_profile.scene.table_placement.object_center_policy must be "
            f"'require_inside_fixed_footprint', got {object_center_policy!r}."
        )
    reference_body = str(placement_profile["reference_body"] or "").strip()
    if not reference_body:
        raise ValueError("render_profile.scene.table_placement.reference_body must be non-empty.")

    center_offset = np.asarray(
        placement_profile["center_offset_xy_in_reference_frame_m"], dtype=np.float64
    )
    if center_offset.shape != (2,) or not np.all(np.isfinite(center_offset)):
        raise ValueError(
            "render_profile.scene.table_placement.center_offset_xy_in_reference_frame_m "
            "must contain two finite values."
        )
    full_size = np.asarray(placement_profile["full_size_m"], dtype=np.float64)
    if full_size.shape != (3,) or not np.all(np.isfinite(full_size)) or np.any(full_size <= 0.0):
        raise ValueError(
            "render_profile.scene.table_placement.full_size_m must contain three positive finite values."
        )

    documents = [_MJCFDocument(root=root, path=None)]
    reference_match = _find_body_or_none(documents, reference_body)
    if reference_match is None:
        raise ValueError(
            f"Fixed-table reference body {reference_body!r} was not found in the entry MJCF."
        )
    reference_element, reference_document = reference_match
    if reference_document.path is not None:
        raise AssertionError("Internal error: the fixed-table reference must belong to the entry MJCF.")
    reference_world_pos, reference_world_quat = _body_static_world_pose(
        reference_element, _parent_map(documents)
    )
    center_offset_world = _quat_rotate_wxyz(
        reference_world_quat,
        np.asarray([center_offset[0], center_offset[1], 0.0], dtype=np.float64),
    )
    table_center_xy_array = reference_world_pos[:2] + center_offset_world[:2]
    half_size_xy = full_size[:2] / 2.0
    footprint_min_xy = table_center_xy_array - half_size_xy
    footprint_max_xy = table_center_xy_array + half_size_xy

    object_metadata = _direct_free_object_body_metadata(root)
    validated_objects: list[dict[str, Any]] = []
    minimum_axis_clearance: np.ndarray | None = None
    tolerance = 1e-9
    for item in object_metadata:
        center = np.asarray(item["center"], dtype=np.float64)
        axis_clearance = half_size_xy - np.abs(center[:2] - table_center_xy_array)
        if np.any(axis_clearance < -tolerance):
            raise ValueError(
                "Free object center lies outside the configured fixed Franka table footprint: "
                f"body={item['body_name']!r}, center_xy={center[:2].tolist()}, "
                f"footprint_min_xy={footprint_min_xy.tolist()}, "
                f"footprint_max_xy={footprint_max_xy.tolist()}."
            )
        minimum_axis_clearance = (
            axis_clearance.copy()
            if minimum_axis_clearance is None
            else np.minimum(minimum_axis_clearance, axis_clearance)
        )
        validated_objects.append(
            {
                **item,
                "edge_clearance_xy_m": axis_clearance.tolist(),
                "minimum_edge_clearance_m": float(np.min(axis_clearance)),
            }
        )

    dynamic_required = np.asarray(dynamic_required_table_full_size, dtype=np.float64)
    dynamic_overflow = np.maximum(dynamic_required - full_size, 0.0)
    return (
        (float(table_center_xy_array[0]), float(table_center_xy_array[1])),
        (float(full_size[0]), float(full_size[1]), float(full_size[2])),
        {
            "fixed_table_placement_applied": True,
            "fixed_table_placement": {
                "mode": mode,
                "reference_body": reference_body,
                "resolved_reference_body": reference_element.get("name") or reference_body,
                "reference_body_world_pos": reference_world_pos.tolist(),
                "reference_body_world_quat_wxyz": reference_world_quat.tolist(),
                "center_offset_xy_in_reference_frame_m": center_offset.tolist(),
                "resolved_table_center_xy_world_m": table_center_xy_array.tolist(),
                "full_size_m": full_size.tolist(),
                "footprint_min_xy_world_m": footprint_min_xy.tolist(),
                "footprint_max_xy_world_m": footprint_max_xy.tolist(),
                "object_center_policy": object_center_policy,
                "validated_object_count": len(validated_objects),
                "validated_objects": validated_objects,
                "minimum_edge_clearance_xy_m": (
                    minimum_axis_clearance.tolist()
                    if minimum_axis_clearance is not None
                    else None
                ),
                "minimum_edge_clearance_m": (
                    float(np.min(minimum_axis_clearance))
                    if minimum_axis_clearance is not None
                    else None
                ),
                "legacy_dynamic_table_center_xy_world_m": list(dynamic_table_center_xy),
                "legacy_dynamic_required_table_full_size_m": dynamic_required.tolist(),
                "legacy_dynamic_required_size_overflow_m": dynamic_overflow.tolist(),
                "legacy_aesthetic_margin_is_not_a_containment_requirement": True,
            },
        },
    )


def _infer_scene_placement(root: ET.Element, render_profile: dict[str, Any] | None = None) -> ScenePlacement:
    render_profile = render_profile or {}
    if not isinstance(render_profile, dict):
        raise ValueError("render_profile must be a JSON object.")
    scene_profile = render_profile.get("scene") or {}
    if not isinstance(scene_profile, dict):
        raise ValueError("render_profile.scene must be a JSON object.")
    original_tables = _extract_original_table_metadata(root)
    table_placement_metadata: dict[str, Any] = {"fixed_table_placement_applied": False}
    preserve_source_table = False
    policy = "arm_robot_and_objects_on_table"
    dynamic_table_center_xy = _infer_table_center_xy(root, include_robot=True)
    dynamic_required_table_full_size = _infer_table_full_size(
        root, min_size=DEFAULT_TABLE_FULL_SIZE
    )
    table_placement_profile = scene_profile.get("table_placement")
    if table_placement_profile is None:
        table_center_xy = dynamic_table_center_xy
        table_full_size = dynamic_required_table_full_size
        table_placement_metadata.update(
            {
                "legacy_dynamic_table_center_xy_world_m": list(dynamic_table_center_xy),
                "legacy_dynamic_required_table_full_size_m": list(
                    dynamic_required_table_full_size
                ),
            }
        )
    else:
        (
            table_center_xy,
            table_full_size,
            table_placement_metadata,
        ) = _fixed_body_frame_table_placement(
            root,
            table_placement_profile,
            dynamic_table_center_xy=dynamic_table_center_xy,
            dynamic_required_table_full_size=dynamic_required_table_full_size,
        )
    table_top_z = DEFAULT_TABLE_TOP_Z
    arm_scene_z_offset = DEFAULT_TABLE_TOP_Z
    free_joint_qpos_z_offset = DEFAULT_TABLE_TOP_Z
    table_surface_geom_names = (f"{ARENA_PREFIX}table_visual",)
    arm_root_body_names = tuple(
        str(name) for name in (scene_profile.get("arm_root_body_names") or ()) if str(name)
    )
    metadata = {
        "scene_placement_policy": policy,
        "table_center_xy": list(table_center_xy),
        "table_top_z": float(table_top_z),
        "table_full_size": list(table_full_size),
        "preserve_source_table": bool(preserve_source_table),
        "table_geometry_policy": scene_profile.get("table_geometry_policy"),
        "arm_scene_z_offset": float(arm_scene_z_offset),
        "arm_root_body_names": list(arm_root_body_names),
        "free_joint_qpos_z_offset": float(free_joint_qpos_z_offset),
        "table_surface_geom_names": list(table_surface_geom_names),
        "original_table_metadata": original_tables,
        "object_body_positions_for_table": [pos.tolist() for pos in _direct_object_body_positions(root)],
        "arm_robot_body_positions_for_table": [pos.tolist() for pos in _direct_arm_robot_body_positions(root)],
        **table_placement_metadata,
    }
    return ScenePlacement(
        policy=policy,
        table_center_xy=table_center_xy,
        table_top_z=table_top_z,
        table_full_size=table_full_size,
        preserve_source_table=preserve_source_table,
        arm_scene_z_offset=arm_scene_z_offset,
        free_joint_qpos_z_offset=free_joint_qpos_z_offset,
        table_surface_geom_names=table_surface_geom_names,
        metadata=metadata,
        arm_root_body_names=arm_root_body_names,
    )


def _apply_arm_scene_z_offset(root: ET.Element, placement: ScenePlacement) -> dict[str, Any]:
    if abs(float(placement.arm_scene_z_offset)) <= 1e-12:
        return {"scene_body_z_offset_applied": False, "scene_body_z_offset": 0.0, "z_shifted_body_names": []}
    shifted: list[dict[str, Any]] = []
    for body in _direct_worldbody_bodies(root):
        name = body.get("name") or ""
        if _is_original_scene_shell_body(body):
            continue
        declared_arm_root = name in placement.arm_root_body_names
        if not (_is_object_free_body(body)
                or _name_contains(name, ARM_ROBOT_BODY_NEEDLES)
                or declared_arm_root):
            continue
        old_pos = _parse_vec(body.get("pos"), length=3, default=(0.0, 0.0, 0.0))
        new_pos = old_pos.copy()
        new_pos[2] += float(placement.arm_scene_z_offset)
        _set_body_pos(body, new_pos)
        shifted.append({"body_name": name, "old_pos": old_pos.tolist(), "new_pos": new_pos.tolist()})
    return {
        "scene_body_z_offset_applied": True,
        "scene_body_z_offset": float(placement.arm_scene_z_offset),
        "z_shifted_body_names": [item["body_name"] for item in shifted],
        "z_shifted_body_examples": shifted[:40],
    }


def _direct_scene_element_name(elem: ET.Element, fallback_prefix: str, index: int) -> str:
    return elem.get("name") or f"{fallback_prefix}_{index}"


def _body_has_joint(body: ET.Element) -> bool:
    return any(descendant.tag == "joint" for descendant in body.iter())


def _body_subtree_names(body: ET.Element) -> list[str]:
    return [descendant.get("name") or "" for descendant in body.iter("body") if descendant.get("name")]


def _is_original_scene_shell_body(body: ET.Element) -> bool:
    name = (body.get("name") or "").lower()
    return "table" in name and not _body_has_joint(body)


def _remove_original_direct_scene_shell(
    root: ET.Element,
    *,
    preserve_table_bodies: bool = False,
) -> dict[str, Any]:
    removed: list[dict[str, str]] = []
    removed_body_names: list[str] = []
    preserved_table_body_names: list[str] = []
    for worldbody_index, worldbody in enumerate(root.findall("worldbody")):
        for child_index, child in enumerate(list(worldbody)):
            if child.tag == "body" and _is_original_scene_shell_body(child):
                body_names = _body_subtree_names(child)
                if preserve_table_bodies:
                    preserved_table_body_names.extend(body_names)
                    continue
                removed_body_names.extend(body_names)
                worldbody.remove(child)
                removed.append(
                    {
                        "worldbody_index": str(worldbody_index),
                        "tag": child.tag,
                        "name": _direct_scene_element_name(child, f"unnamed_{child.tag}", child_index),
                        "body_names_in_subtree": ",".join(body_names),
                    }
                )
                continue
            if child.tag not in {"geom", "camera", "light"}:
                continue
            worldbody.remove(child)
            removed.append(
                {
                    "worldbody_index": str(worldbody_index),
                    "tag": child.tag,
                    "name": _direct_scene_element_name(child, f"unnamed_{child.tag}", child_index),
                }
            )
    return {
        "removed_original_scene_shell": removed,
        "removed_original_scene_shell_body_names": sorted(dict.fromkeys(removed_body_names)),
        "preserved_source_table_body_names": sorted(dict.fromkeys(preserved_table_body_names)),
    }


def _configure_preserved_source_table(root: ET.Element, placement: ScenePlacement) -> dict[str, Any]:
    if not placement.preserve_source_table:
        return {"source_table_render_group_configured": False, "source_table_render_geoms": []}
    configured: list[dict[str, Any]] = []
    for body in _direct_worldbody_bodies(root):
        if not _is_original_scene_shell_body(body):
            continue
        for geom in body.iter("geom"):
            old_group = geom.get("group")
            geom.set("group", "1")
            configured.append(
                {
                    "body_name": body.get("name") or "",
                    "geom_name": geom.get("name") or "",
                    "old_group": old_group,
                    "new_group": 1,
                }
            )
    if not configured:
        raise ValueError("preserve_source_table is enabled but no source table geoms were found.")
    return {
        "source_table_render_group_configured": True,
        "source_table_render_geoms": configured,
    }


def _remove_arena_table_body(arena_root: ET.Element) -> dict[str, Any]:
    removed: list[str] = []
    worldbody = arena_root.find("worldbody")
    if worldbody is not None:
        for child in list(worldbody):
            if child.tag == "body" and (child.get("name") or "").lower() == "table":
                removed.append(child.get("name") or "")
                worldbody.remove(child)
    if len(removed) != 1:
        raise ValueError(f"Expected one Robosuite TableArena table body, removed {removed}.")
    return {"removed_arena_table_body_names": removed}


def _rename_arena_tree(arena_root: ET.Element, requested_camera_names: tuple[str, ...]) -> dict[str, Any]:
    requested = set(requested_camera_names)
    camera_names_kept: list[str] = []
    name_map: dict[str, str] = {}
    unnamed_counters: dict[str, int] = {}

    for elem in arena_root.iter():
        old_name = elem.get("name")
        if not old_name and elem.tag == "light":
            count = unnamed_counters.get("light", 0)
            old_name = f"arena_light_{count}"
            unnamed_counters["light"] = count + 1
            elem.set("name", old_name)
        if not old_name:
            continue
        if elem.tag == "camera" and old_name in requested:
            camera_names_kept.append(old_name)
            continue
        new_name = f"{ARENA_PREFIX}{old_name}"
        elem.set("name", new_name)
        name_map[old_name] = new_name

    reference_attrs = (
        "material",
        "texture",
        "mesh",
        "class",
        "childclass",
        "joint",
        "body",
        "site",
        "target",
        "objname",
    )
    for elem in arena_root.iter():
        for attr in reference_attrs:
            value = elem.get(attr)
            if value in name_map:
                elem.set(attr, name_map[value])

    return {
        "arena_prefix": ARENA_PREFIX,
        "arena_renamed_count": len(name_map),
        "arena_camera_names_kept": sorted(camera_names_kept),
        "arena_renamed_examples": [
            {"from": old_name, "to": new_name} for old_name, new_name in list(name_map.items())[:40]
        ],
    }


def _remove_arena_mocap_targets(arena_root: ET.Element) -> dict[str, Any]:
    removed: list[str] = []
    for parent in arena_root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("mocap") == "true":
                removed.append(child.get("name") or "")
                parent.remove(child)
    return {"removed_arena_mocap_bodies": removed}


def _append_children(dst_parent: ET.Element, src_parent: ET.Element | None) -> int:
    if src_parent is None:
        return 0
    count = 0
    for child in list(src_parent):
        dst_parent.append(deepcopy(child))
        count += 1
    return count


def _ensure_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.SubElement(root, tag)
    return child


def _merge_robosuite_table_arena(
    root: ET.Element,
    camera_names: tuple[str, ...],
    placement: ScenePlacement,
) -> dict[str, Any]:
    table_offset = (float(placement.table_center_xy[0]), float(placement.table_center_xy[1]), float(placement.table_top_z))
    arena = TableArena(table_full_size=placement.table_full_size, table_offset=table_offset, has_legs=True)
    arena_root = ET.fromstring(arena.get_xml())
    arena_table_metadata = (
        _remove_arena_table_body(arena_root)
        if placement.preserve_source_table
        else {"removed_arena_table_body_names": []}
    )
    rename_metadata = _rename_arena_tree(arena_root, camera_names)
    mocap_metadata = _remove_arena_mocap_targets(arena_root)
    original_removal_metadata = _remove_original_direct_scene_shell(
        root,
        preserve_table_bodies=placement.preserve_source_table,
    )

    asset = _ensure_child(root, "asset")
    worldbody = _ensure_child(root, "worldbody")
    arena_asset_count = _append_children(asset, arena_root.find("asset"))
    arena_worldbody_count = _append_children(worldbody, arena_root.find("worldbody"))

    return {
        "arena_backend": "robosuite.models.arenas.TableArena",
        "arena_table_full_size": list(placement.table_full_size),
        "arena_table_offset": list(table_offset),
        "arena_table_surface_z": float(placement.table_top_z),
        "arena_has_table": not placement.preserve_source_table,
        "arena_has_legs": not placement.preserve_source_table,
        "source_table_preserved": bool(placement.preserve_source_table),
        "arena_asset_child_count": arena_asset_count,
        "arena_worldbody_child_count": arena_worldbody_count,
        **arena_table_metadata,
        **rename_metadata,
        **mocap_metadata,
        **original_removal_metadata,
        **placement.metadata,
    }


def _profile_vector(
    value: Any,
    *,
    length: int,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must contain {length} finite values.")
    if minimum is not None and np.any(result < minimum):
        raise ValueError(f"{field_name} values must be >= {minimum}.")
    if maximum is not None and np.any(result > maximum):
        raise ValueError(f"{field_name} values must be <= {maximum}.")
    return result


def _apply_fixed_surface_appearance(
    root: ET.Element,
    camera_config: dict[str, Any],
    placement: ScenePlacement,
) -> dict[str, Any]:
    render_profile = camera_config.get("render_profile") or {}
    surface_profile = render_profile.get("surface_appearance") or {}
    if not surface_profile:
        return {"surface_appearance_applied": False}
    if not isinstance(surface_profile, dict):
        raise ValueError("render_profile.surface_appearance must be a JSON object.")
    mode = str(surface_profile.get("mode") or "").strip()
    if mode != "fixed_texture_material":
        raise ValueError(f"Unsupported surface appearance mode {mode!r}.")

    table_profile = surface_profile.get("table") or {}
    if not isinstance(table_profile, dict):
        raise ValueError("render_profile.surface_appearance.table must be a JSON object.")
    texture_asset = _fixed_surface_texture_asset(camera_config)
    if texture_asset is None:
        raise ValueError("fixed_texture_material requires a resolved table texture asset.")
    texture_path = Path(texture_asset["path"])
    texture_repo_path = str(texture_asset["repo_path"])

    rgba = _profile_vector(
        table_profile.get("rgba") or [],
        length=4,
        field_name="render_profile.surface_appearance.table.rgba",
        minimum=0.0,
        maximum=1.0,
    )
    texrepeat = _profile_vector(
        table_profile.get("texrepeat") or [],
        length=2,
        field_name="render_profile.surface_appearance.table.texrepeat",
        minimum=1e-8,
    )
    material_values: dict[str, float] = {}
    for field_name in ("reflectance", "shininess", "specular"):
        value = float(table_profile.get(field_name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"render_profile.surface_appearance.table.{field_name} must be in [0, 1].")
        material_values[field_name] = value

    asset = _ensure_child(root, "asset")
    profile_slug = re.sub(r"[^a-z0-9]+", "_", str(render_profile.get("profile_id") or "fixed_surface").lower()).strip("_")
    if not profile_slug:
        raise ValueError("render_profile.profile_id must contain an alphanumeric character.")
    texture_name = f"{ARENA_PREFIX}{profile_slug}_table_texture"
    material_name = f"{ARENA_PREFIX}{profile_slug}_table_material"
    existing_asset_names = {str(element.get("name") or "") for element in asset}
    conflicts = sorted({texture_name, material_name} & existing_asset_names)
    if conflicts:
        raise ValueError(f"Fixed surface appearance asset names already exist: {conflicts}")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": texture_name,
            "type": "2d",
            "file": str(texture_path),
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": material_name,
            "texture": texture_name,
            "texrepeat": _format_float_sequence(texrepeat),
            "texuniform": "true" if bool(table_profile.get("texuniform", False)) else "false",
            "rgba": _format_float_sequence(rgba),
            **{key: f"{value:.9g}" for key, value in material_values.items()},
        },
    )

    requested_geom_names = set(placement.table_surface_geom_names)
    matched_geom_names: list[str] = []
    for geom in root.iter("geom"):
        geom_name = str(geom.get("name") or "")
        if geom_name not in requested_geom_names:
            continue
        geom.set("material", material_name)
        geom.attrib.pop("rgba", None)
        matched_geom_names.append(geom_name)
    missing_geom_names = sorted(requested_geom_names - set(matched_geom_names))
    if missing_geom_names:
        raise ValueError(f"Fixed table appearance did not match table geoms: {missing_geom_names}")

    return {
        "surface_appearance_applied": True,
        "surface_appearance_mode": mode,
        "surface_appearance_source_style": surface_profile.get("source_style"),
        "surface_appearance_table": {
            "texture_file": texture_repo_path,
            "texture_sha256": texture_asset["sha256"],
            "texture_declared_sha256": texture_asset["declared_sha256"],
            "texture_name": texture_name,
            "material_name": material_name,
            "texrepeat": texrepeat.tolist(),
            "texuniform": bool(table_profile.get("texuniform", False)),
            "rgba": rgba.tolist(),
            **material_values,
            "matched_geom_names": sorted(matched_geom_names),
            "source_asset_id": table_profile.get("source_asset_id"),
            "source_url": table_profile.get("source_url"),
            "license": table_profile.get("license"),
        },
    }


def _configure_reference_lighting_rig(
    root: ET.Element,
    camera_config: dict[str, Any],
    placement: ScenePlacement,
) -> dict[str, Any]:
    render_profile = camera_config.get("render_profile") or {}
    lighting_profile = render_profile.get("lighting") or {}
    if not lighting_profile:
        return {"lighting_rig_configured": False}
    if not isinstance(lighting_profile, dict):
        raise ValueError("render_profile.lighting must be a JSON object.")
    mode = str(lighting_profile.get("mode") or "").strip()
    if mode != "reference_five_light_rig":
        return {"lighting_rig_configured": False, "lighting_rig_mode": mode}
    sources = lighting_profile.get("sources") or []
    if not isinstance(sources, list) or len(sources) != 5:
        raise ValueError("reference_five_light_rig requires exactly five light sources.")

    worldbody = _ensure_child(root, "worldbody")
    removed_light_names: list[str] = []
    for child in list(worldbody):
        if child.tag == "light" and str(child.get("name") or "").startswith(ARENA_PREFIX):
            removed_light_names.append(str(child.get("name") or ""))
            worldbody.remove(child)

    table_origin = np.asarray(
        [placement.table_center_xy[0], placement.table_center_xy[1], placement.table_top_z],
        dtype=np.float64,
    )
    source_metadata: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"lighting source {index} must be a JSON object.")
        source_name = str(source.get("name") or "").strip()
        if not source_name or source_name in used_names:
            raise ValueError(f"lighting source {index} requires a unique non-empty name.")
        used_names.add(source_name)
        light_name = f"{ARENA_PREFIX}{source_name}"
        if "position" in source:
            position = _profile_vector(
                source["position"], length=3, field_name=f"lighting.sources[{index}].position"
            )
            position_frame = "world"
        else:
            position_offset = _profile_vector(
                source.get("position_offset") or [],
                length=3,
                field_name=f"lighting.sources[{index}].position_offset",
            )
            position = table_origin + position_offset
            position_frame = "table_surface"
        if "direction" in source:
            direction = _profile_vector(
                source["direction"], length=3, field_name=f"lighting.sources[{index}].direction"
            )
        else:
            target_offset = _profile_vector(
                source.get("target_offset") or [0.0, 0.0, 0.0],
                length=3,
                field_name=f"lighting.sources[{index}].target_offset",
            )
            direction = table_origin + target_offset - position
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-8:
            raise ValueError(f"lighting source {source_name!r} direction must be non-zero.")
        direction /= direction_norm

        light_attrs = {
            "name": light_name,
            "mode": "fixed",
            "active": "true",
            "pos": _format_float_sequence(position),
            "dir": _format_float_sequence(direction),
            "directional": "true" if bool(source.get("directional", False)) else "false",
            "castshadow": "true" if bool(source.get("castshadow", False)) else "false",
        }
        resolved_values: dict[str, Any] = {}
        for field_name in ("ambient", "diffuse", "specular"):
            values = _profile_vector(
                source.get(field_name) or [],
                length=3,
                field_name=f"lighting.sources[{index}].{field_name}",
                minimum=0.0,
                maximum=1.0,
            )
            light_attrs[field_name] = _format_float_sequence(values)
            resolved_values[field_name] = values.tolist()
        attenuation = _profile_vector(
            source.get("attenuation") or [1.0, 0.0, 0.0],
            length=3,
            field_name=f"lighting.sources[{index}].attenuation",
            minimum=0.0,
        )
        cutoff = float(source.get("cutoff", 90.0))
        exponent = float(source.get("exponent", 1.0))
        if not math.isfinite(cutoff) or not 0.0 < cutoff <= 90.0:
            raise ValueError(f"lighting.sources[{index}].cutoff must be in (0, 90].")
        if not math.isfinite(exponent) or exponent < 0.0:
            raise ValueError(f"lighting.sources[{index}].exponent must be >= 0.")
        light_attrs.update(
            {
                "attenuation": _format_float_sequence(attenuation),
                "cutoff": f"{cutoff:.9g}",
                "exponent": f"{exponent:.9g}",
            }
        )
        ET.SubElement(worldbody, "light", light_attrs)
        source_metadata.append(
            {
                "name": light_name,
                "source_type": source.get("source_type"),
                "position": position.tolist(),
                "position_frame": position_frame,
                "direction": direction.tolist(),
                "directional": bool(source.get("directional", False)),
                "castshadow": bool(source.get("castshadow", False)),
                "attenuation": attenuation.tolist(),
                "cutoff": cutoff,
                "exponent": exponent,
                **resolved_values,
            }
        )

    quality_profile = lighting_profile.get("render_quality") or {}
    if not isinstance(quality_profile, dict):
        raise ValueError("render_profile.lighting.render_quality must be a JSON object.")
    shadowsize = int(quality_profile.get("shadowsize", 4096))
    offsamples = int(quality_profile.get("offsamples", 4))
    shadowclip = float(quality_profile.get("shadowclip", 2.8))
    shadowscale = float(quality_profile.get("shadowscale", 0.8))
    if shadowsize < 128 or offsamples < 1:
        raise ValueError("lighting render quality requires shadowsize >= 128 and offsamples >= 1.")
    if not math.isfinite(shadowclip) or shadowclip <= 0.0:
        raise ValueError("lighting render quality requires shadowclip > 0.")
    if not math.isfinite(shadowscale) or shadowscale <= 0.0:
        raise ValueError("lighting render quality requires shadowscale > 0.")
    visual = _ensure_child(root, "visual")
    quality = visual.find("quality")
    if quality is None:
        quality = ET.SubElement(visual, "quality")
    quality.set("shadowsize", str(shadowsize))
    quality.set("offsamples", str(offsamples))
    visual_map = visual.find("map")
    if visual_map is None:
        visual_map = ET.SubElement(visual, "map")
    visual_map.set("shadowclip", f"{shadowclip:.9g}")
    visual_map.set("shadowscale", f"{shadowscale:.9g}")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    disable_headlight = bool(quality_profile.get("disable_headlight", True))
    headlight_values: dict[str, list[float]] = {}
    if disable_headlight:
        for field_name in ("ambient", "diffuse", "specular"):
            headlight.set(field_name, "0 0 0")
            headlight_values[field_name] = [0.0, 0.0, 0.0]
    else:
        for field_name in ("ambient", "diffuse", "specular"):
            values = _profile_vector(
                quality_profile.get(f"headlight_{field_name}") or [],
                length=3,
                field_name=f"render_profile.lighting.render_quality.headlight_{field_name}",
                minimum=0.0,
                maximum=1.0,
            )
            headlight.set(field_name, _format_float_sequence(values))
            headlight_values[field_name] = values.tolist()

    return {
        "lighting_rig_configured": True,
        "lighting_rig_mode": mode,
        "lighting_rig_source_style": lighting_profile.get("source_style"),
        "lighting_rig_removed_arena_lights": removed_light_names,
        "lighting_rig_sources": source_metadata,
        "lighting_render_quality": {
            "shadowsize": shadowsize,
            "offsamples": offsamples,
            "shadowclip": shadowclip,
            "shadowscale": shadowscale,
            "headlight_active": not disable_headlight,
            "headlight": headlight_values,
        },
    }


def _load_mjcf_documents(root: ET.Element, *, entry_xml: Path) -> list[_MJCFDocument]:
    """Load the entry document plus local MJCF includes so mounted cameras can target included bodies."""

    entry_xml = Path(entry_xml).expanduser().resolve()
    documents = [_MJCFDocument(root=root, path=None)]
    seen_paths = {entry_xml}

    def visit(document_root: ET.Element, xml_dir: Path) -> None:
        for include in document_root.iter("include"):
            raw_file = str(include.get("file") or "").strip()
            if not raw_file or raw_file.startswith(("http://", "https://", "s3://")):
                continue
            include_path = Path(raw_file).expanduser()
            if not include_path.is_absolute():
                include_path = (xml_dir / include_path).resolve()
            else:
                include_path = include_path.resolve()
            if include_path in seen_paths:
                continue
            if not include_path.is_file():
                raise FileNotFoundError(
                    f"MJCF include required for mounted-camera resolution does not exist: {include_path}"
                )
            seen_paths.add(include_path)
            include_root = ET.parse(include_path).getroot()
            documents.append(_MJCFDocument(root=include_root, path=include_path))
            visit(include_root, include_path.parent)

    visit(root, entry_xml.parent)
    return documents


def _atomic_write_xml(root: ET.Element, path: Path) -> None:
    path = Path(path).expanduser().resolve()
    temp_path = path.with_name(f".{path.name}.tmp_{os.getpid()}_{time.time_ns()}")
    try:
        ET.ElementTree(root).write(temp_path, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _derive_mjcf_include_tree(root: ET.Element, *, entry_xml: Path) -> dict[str, Any]:
    entry_xml = Path(entry_xml).expanduser().resolve()
    token = f"{os.getpid()}_{time.time_ns()}"
    derived_by_source: dict[Path, Path] = {}

    def resolve_include(raw_file: str, parent_dir: Path) -> Path:
        include_path = Path(raw_file).expanduser()
        if not include_path.is_absolute():
            include_path = parent_dir / include_path
        include_path = include_path.resolve()
        if not include_path.is_file():
            raise FileNotFoundError(f"MJCF include does not exist: {include_path}")
        return include_path

    def derive(source_path: Path) -> Path:
        source_path = source_path.resolve()
        if source_path in derived_by_source:
            return derived_by_source[source_path]
        derived_path = source_path.with_name(
            f".{source_path.stem}.robosuite_derived_{token}{source_path.suffix}"
        )
        derived_by_source[source_path] = derived_path
        source_root = ET.parse(source_path).getroot()
        for include in source_root.iter("include"):
            raw_file = str(include.get("file") or "").strip()
            if not raw_file or raw_file.startswith(("http://", "https://", "s3://")):
                continue
            nested_source = resolve_include(raw_file, source_path.parent)
            include.set("file", str(derive(nested_source)))
        _atomic_write_xml(source_root, derived_path)
        return derived_path

    try:
        for include in root.iter("include"):
            raw_file = str(include.get("file") or "").strip()
            if not raw_file or raw_file.startswith(("http://", "https://", "s3://")):
                continue
            source_path = resolve_include(raw_file, entry_xml.parent)
            include.set("file", str(derive(source_path)))
    except Exception:
        for derived_path in derived_by_source.values():
            derived_path.unlink(missing_ok=True)
        raise

    return {
        "derived_mjcf_include_files": [str(path) for path in derived_by_source.values()],
        "derived_mjcf_include_sources": [
            {"source": str(source), "derived": str(derived)}
            for source, derived in derived_by_source.items()
        ],
        "derived_mjcf_include_policy": "unique_atomic_copy_per_render",
    }


def _cleanup_derived_mjcf_include_files(metadata: dict[str, Any]) -> dict[str, Any]:
    removed: list[str] = []
    for raw_path in metadata.get("derived_mjcf_include_files") or []:
        path = Path(str(raw_path)).expanduser().resolve()
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return {
        "derived_mjcf_include_cleanup_complete": True,
        "derived_mjcf_include_files_removed": removed,
    }


def _parent_map(documents: list[_MJCFDocument]) -> dict[ET.Element, ET.Element]:
    return {
        child: parent
        for document in documents
        for parent in document.root.iter()
        for child in list(parent)
    }


def _find_body_or_none(
    documents: list[_MJCFDocument],
    body_name: str,
) -> tuple[ET.Element, _MJCFDocument] | None:
    exact_matches: list[tuple[ET.Element, _MJCFDocument]] = []
    suffix_matches: list[tuple[ET.Element, _MJCFDocument]] = []
    for document in documents:
        for body in document.root.iter("body"):
            name = body.get("name") or ""
            if name == body_name:
                exact_matches.append((body, document))
            if name.endswith(f"/{body_name}"):
                suffix_matches.append((body, document))
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(f"Camera body {body_name!r} is duplicated in the final MJCF.")
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        match_names = [body.get("name") or "" for body, _ in suffix_matches]
        raise ValueError(f"Camera mount body {body_name!r} matched multiple prefixed bodies: {match_names}")
    return None


def _find_body(documents: list[_MJCFDocument], body_name: str) -> tuple[ET.Element, _MJCFDocument]:
    match = _find_body_or_none(documents, body_name)
    if match is not None:
        return match
    raise ValueError(f"Camera mount body {body_name!r} was not found in the final MJCF or its includes.")


def _body_name_matches(resolved_name: str, requested_name: str) -> bool:
    return resolved_name == requested_name or resolved_name.endswith(f"/{requested_name}")


def _body_static_world_pose(
    body: ET.Element,
    parents: dict[ET.Element, ET.Element],
) -> tuple[np.ndarray, np.ndarray]:
    chain: list[ET.Element] = []
    cursor: ET.Element | None = body
    while cursor is not None and cursor.tag == "body":
        chain.append(cursor)
        parent = parents.get(cursor)
        cursor = parent if parent is not None and parent.tag == "body" else None

    world_pos = np.zeros(3, dtype=np.float64)
    world_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for chain_body in reversed(chain):
        body_name = chain_body.get("name") or "<unnamed>"
        if chain_body.find("freejoint") is not None or chain_body.find("joint") is not None:
            raise ValueError(
                f"Camera reference body {body.get('name')!r} has dynamic ancestor {body_name!r}; "
                "a fixed base frame is required."
            )
        unsupported_orientation_attrs = [
            attr
            for attr in ("axisangle", "euler", "xyaxes", "zaxis")
            if chain_body.get(attr) is not None
        ]
        if unsupported_orientation_attrs:
            raise ValueError(
                f"Camera reference body {body_name!r} uses unsupported static orientation "
                f"attributes {unsupported_orientation_attrs}; use an explicit MJCF quat."
            )
        local_pos = _parse_vec(chain_body.get("pos"), length=3, default=(0.0, 0.0, 0.0))
        local_quat = _normalize_quat_wxyz(
            _parse_vec(chain_body.get("quat"), length=4, default=(1.0, 0.0, 0.0, 0.0)),
            label=f"body {body_name!r} quat",
        )
        world_pos = world_pos + _quat_rotate_wxyz(world_quat, local_pos)
        world_quat = _normalize_quat_wxyz(
            _quat_multiply_wxyz(world_quat, local_quat),
            label=f"body {body_name!r} world quat",
        )
    return world_pos, world_quat


def _body_frame_relative_camera_pose(
    camera_config: dict[str, Any],
    documents: list[_MJCFDocument],
    parents: dict[ET.Element, ET.Element],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    extrinsics = camera_config.get("extrinsics") or {}
    reference_body = str(extrinsics.get("reference_body") or "").strip()
    if not reference_body:
        raise ValueError(
            f"Body-frame-relative camera {camera_config.get('name')!r} must define "
            "extrinsics.reference_body."
        )
    for field_name in ("position_offset", "target_offset"):
        if field_name not in extrinsics:
            raise ValueError(
                f"Body-frame-relative camera {camera_config.get('name')!r} must define "
                f"extrinsics.{field_name}."
            )
    body, body_document = _find_body(documents, reference_body)
    resolved_reference_body = body.get("name") or reference_body
    if body_document.path is not None:
        raise ValueError(
            f"Camera reference body {resolved_reference_body!r} is defined in included MJCF "
            f"{body_document.path}; body_frame_relative_lookat currently requires a fixed body "
            "in the entry MJCF so no include-parent transform can be omitted."
        )
    base_pos, base_quat = _body_static_world_pose(body, parents)
    position_offset = _parse_vec(
        _format_float_sequence(extrinsics.get("position_offset") or []),
        length=3,
    )
    target_offset = _parse_vec(
        _format_float_sequence(extrinsics.get("target_offset") or []),
        length=3,
    )
    position = base_pos + _quat_rotate_wxyz(base_quat, position_offset)
    target = base_pos + _quat_rotate_wxyz(base_quat, target_offset)
    return position, target, {
        "extrinsics_type": "body_frame_relative_lookat",
        "reference_body": reference_body,
        "resolved_reference_body": resolved_reference_body,
        "reference_document": "entry_mjcf",
        "reference_body_world_pos": base_pos.tolist(),
        "reference_body_world_quat_wxyz": base_quat.tolist(),
        "position_offset_in_reference_body": position_offset.tolist(),
        "target_offset_in_reference_body": target_offset.tolist(),
    }


def _resolve_camera_mount_body(
    documents: list[_MJCFDocument],
    mount: dict[str, Any],
    parents: dict[ET.Element, ET.Element],
    element_documents: dict[ET.Element, _MJCFDocument],
) -> tuple[ET.Element, _MJCFDocument, dict[str, Any]]:
    mount_body = str(mount.get("body") or "").strip()
    if not mount_body:
        raise ValueError("Body-mounted camera must define mount.body.")
    existing = _find_body_or_none(documents, mount_body)
    if existing is not None:
        body, document = existing
        return body, document, {
            "type": "existing_body",
            "requested_body": mount_body,
            "resolved_body": body.get("name") or mount_body,
            "injected": False,
        }

    adapter = mount.get("missing_body_frame_adapter")
    if not isinstance(adapter, dict):
        raise ValueError(
            f"Camera mount body {mount_body!r} is missing and no explicit "
            "mount.missing_body_frame_adapter was configured."
        )
    adapter_type = str(adapter.get("type") or "").strip()
    if adapter_type != "validated_identity_child":
        raise ValueError(f"Unsupported missing body frame adapter {adapter_type!r}.")
    source_body_name = str(adapter.get("source_body") or "").strip()
    if not source_body_name:
        raise ValueError("validated_identity_child requires source_body.")
    source_body, source_document = _find_body(documents, source_body_name)
    resolved_source_body = source_body.get("name") or source_body_name
    source_parent = parents.get(source_body)
    expected_source_parent = str(adapter.get("expected_source_parent") or "").strip()
    if not expected_source_parent:
        raise ValueError("validated_identity_child requires expected_source_parent.")
    resolved_source_parent = (
        source_parent.get("name")
        if source_parent is not None and source_parent.tag == "body"
        else None
    )
    if resolved_source_parent is None or not _body_name_matches(
        resolved_source_parent, expected_source_parent
    ):
        raise ValueError(
            f"Camera frame adapter expected source body {resolved_source_body!r} under "
            f"{expected_source_parent!r}, got {resolved_source_parent!r}."
        )

    unsupported_orientation_attrs = [
        attr
        for attr in ("axisangle", "euler", "xyaxes", "zaxis")
        if source_body.get(attr) is not None
    ]
    if unsupported_orientation_attrs:
        raise ValueError(
            f"Camera frame adapter source body {resolved_source_body!r} uses unsupported "
            f"orientation attributes {unsupported_orientation_attrs}."
        )
    for field_name in ("expected_source_local_pos", "expected_source_local_quat_wxyz"):
        if field_name not in adapter:
            raise ValueError(f"validated_identity_child requires {field_name}.")
    actual_source_pos = _parse_vec(
        source_body.get("pos"), length=3, default=(0.0, 0.0, 0.0)
    )
    actual_source_quat = _normalize_quat_wxyz(
        _parse_vec(
            source_body.get("quat"),
            length=4,
            default=(1.0, 0.0, 0.0, 0.0),
        ),
        label=f"body {resolved_source_body!r} quat",
    )
    expected_source_pos = _parse_vec(
        _format_float_sequence(adapter["expected_source_local_pos"]),
        length=3,
    )
    expected_source_quat = _normalize_quat_wxyz(
        _parse_vec(
            _format_float_sequence(adapter["expected_source_local_quat_wxyz"]),
            length=4,
        ),
        label="expected source body quat",
    )
    tolerance = float(adapter.get("validation_tolerance", 1e-6))
    if tolerance <= 0.0:
        raise ValueError("Camera frame adapter validation_tolerance must be positive.")
    if not np.allclose(actual_source_pos, expected_source_pos, atol=tolerance, rtol=0.0):
        raise ValueError(
            f"Camera frame adapter source position mismatch for {resolved_source_body!r}: "
            f"expected={expected_source_pos.tolist()}, actual={actual_source_pos.tolist()}."
        )
    quat_matches = np.allclose(
        actual_source_quat, expected_source_quat, atol=tolerance, rtol=0.0
    ) or np.allclose(actual_source_quat, -expected_source_quat, atol=tolerance, rtol=0.0)
    if not quat_matches:
        raise ValueError(
            f"Camera frame adapter source quaternion mismatch for {resolved_source_body!r}: "
            f"expected={expected_source_quat.tolist()}, actual={actual_source_quat.tolist()}."
        )

    if "/" in mount_body:
        resolved_mount_body = mount_body
    elif "/" in resolved_source_body:
        resolved_mount_body = f"{resolved_source_body.rsplit('/', 1)[0]}/{mount_body}"
    else:
        resolved_mount_body = mount_body
    target_body = ET.SubElement(
        source_body,
        "body",
        {
            "name": resolved_mount_body,
            "pos": "0 0 0",
            "quat": "1 0 0 0",
        },
    )
    parents[target_body] = source_body
    element_documents[target_body] = source_document
    return target_body, source_document, {
        "type": adapter_type,
        "requested_body": mount_body,
        "resolved_body": resolved_mount_body,
        "source_body": source_body_name,
        "resolved_source_body": resolved_source_body,
        "expected_source_parent": expected_source_parent,
        "resolved_source_parent": resolved_source_parent,
        "source_local_pos": actual_source_pos.tolist(),
        "source_local_quat_wxyz": actual_source_quat.tolist(),
        "validation_tolerance": tolerance,
        "injected": True,
        "alias_local_pos": [0.0, 0.0, 0.0],
        "alias_local_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def _write_modified_include_documents(
    documents: list[_MJCFDocument],
    modified_include_paths: set[Path],
) -> list[str]:
    written_paths: list[str] = []
    for document in documents:
        if document.path is None or document.path not in modified_include_paths:
            continue
        _atomic_write_xml(document.root, document.path)
        written_paths.append(str(document.path))
    return written_paths


def _apply_embodiment_appearance(
    root: ET.Element,
    camera_config: dict[str, Any],
    *,
    entry_xml: Path,
) -> dict[str, Any]:
    render_profile = camera_config.get("render_profile") or {}
    if not isinstance(render_profile, dict):
        raise ValueError("render_profile must be a JSON object.")
    appearance = render_profile.get("appearance") or {}
    if not isinstance(appearance, dict):
        raise ValueError("render_profile.appearance must be a JSON object.")
    overrides = appearance.get("material_overrides") or []
    if not overrides:
        return {
            "appearance_profile_id": render_profile.get("profile_id"),
            "appearance_material_overrides": [],
            "appearance_modified_mjcf_include_files": [],
        }
    if not isinstance(overrides, list):
        raise ValueError("render_profile.appearance.material_overrides must be a list.")

    documents = _load_mjcf_documents(root, entry_xml=entry_xml)
    element_documents = {
        element: document
        for document in documents
        for element in document.root.iter()
    }
    modified_include_paths: set[Path] = set()
    applied: list[dict[str, Any]] = []
    used_material_names: set[str] = set()
    for index, override in enumerate(overrides):
        if not isinstance(override, dict):
            raise ValueError(f"appearance material override {index} must be a JSON object.")
        material_name = str(override.get("name") or "").strip()
        if not material_name:
            raise ValueError(f"appearance material override {index} requires name.")
        if material_name in used_material_names:
            raise ValueError(f"Duplicate appearance material override name {material_name!r}.")
        used_material_names.add(material_name)
        body_roots = [str(name).strip() for name in (override.get("body_roots") or []) if str(name).strip()]
        mesh_names = [str(name).strip() for name in (override.get("mesh_names") or []) if str(name).strip()]
        if len(mesh_names) != len(set(mesh_names)):
            raise ValueError(f"appearance material override {material_name!r} has duplicate mesh_names.")
        if not body_roots and not mesh_names:
            raise ValueError(
                f"appearance material override {material_name!r} requires body_roots or mesh_names."
            )
        rgba = np.asarray(override.get("rgba") or [], dtype=np.float64)
        if rgba.shape != (4,) or np.any(rgba < 0.0) or np.any(rgba > 1.0):
            raise ValueError(f"appearance material override {material_name!r} requires rgba in [0, 1].")

        include_descendants = bool(override.get("include_descendants", True))
        matched_body_names: list[str] = []
        matched_geom_names: list[str] = []
        matched_mesh_names: list[str] = []
        modified_geoms: set[int] = set()

        def apply_to_geom(geom: ET.Element) -> None:
            geom_identity = id(geom)
            if geom_identity in modified_geoms:
                return
            modified_geoms.add(geom_identity)
            geom.attrib.pop("material", None)
            geom.set("rgba", _format_float_sequence(rgba))
            matched_geom_names.append(geom.get("name") or "")
            if geom.get("mesh"):
                matched_mesh_names.append(str(geom.get("mesh")))
            document = element_documents.get(geom)
            if document is not None and document.path is not None:
                modified_include_paths.add(document.path)

        requested_mesh_names = set(mesh_names)
        for document in documents:
            for geom in document.root.iter("geom"):
                if str(geom.get("mesh") or "") in requested_mesh_names:
                    apply_to_geom(geom)
        missing_mesh_names = sorted(requested_mesh_names - set(matched_mesh_names))
        if missing_mesh_names:
            raise ValueError(
                f"appearance material override {material_name!r} did not match mesh_names {missing_mesh_names}."
            )

        for body_root in body_roots:
            target_body, target_document = _find_body(documents, body_root)
            matched_body_names.append(target_body.get("name") or body_root)
            geom_iter = target_body.iter("geom") if include_descendants else target_body.findall("geom")
            for geom in geom_iter:
                apply_to_geom(geom)
            if target_document.path is not None:
                modified_include_paths.add(target_document.path)
        if not modified_geoms:
            raise ValueError(f"appearance material override {material_name!r} matched no geoms.")
        applied.append(
            {
                "material_name": material_name,
                "body_roots": body_roots,
                "mesh_names": mesh_names,
                "include_descendants": include_descendants,
                "rgba": rgba.tolist(),
                "matched_body_names": sorted(dict.fromkeys(matched_body_names)),
                "matched_geom_count": len(modified_geoms),
                "matched_geom_names": sorted(dict.fromkeys(matched_geom_names)),
                "matched_mesh_names": sorted(dict.fromkeys(matched_mesh_names)),
            }
        )

    written = _write_modified_include_documents(documents, modified_include_paths)
    return {
        "appearance_profile_id": render_profile.get("profile_id"),
        "appearance_material_overrides": applied,
        "appearance_modified_mjcf_include_files": written,
    }


def _ensure_configured_cameras(
    root: ET.Element,
    camera_config: dict[str, Any],
    placement: ScenePlacement,
    *,
    entry_xml: Path,
) -> dict[str, Any]:
    requires_mjcf_documents = any(
        str((camera.get("mount") or {}).get("type") or "world") == "body"
        or str((camera.get("extrinsics") or {}).get("type") or "")
        == "body_frame_relative_lookat"
        for camera in camera_config["cameras"]
    )
    documents = (
        _load_mjcf_documents(root, entry_xml=entry_xml)
        if requires_mjcf_documents
        else [_MJCFDocument(root=root, path=None)]
    )
    element_documents = {
        element: document
        for document in documents
        for element in document.root.iter()
    }
    parents = _parent_map(documents)
    modified_include_paths: set[Path] = set()

    def mark_document_modified(document: _MJCFDocument | None) -> None:
        if document is not None and document.path is not None:
            modified_include_paths.add(document.path)

    resolved_mjcf_include_file_attr_count = 0
    for document in documents:
        if document.path is None:
            continue
        include_path_metadata = _absolutize_xml_file_attrs(document.root, xml_dir=document.path.parent)
        resolved_count = int(include_path_metadata["resolved_file_attr_count"])
        if resolved_count:
            resolved_mjcf_include_file_attr_count += resolved_count
            mark_document_modified(document)

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    existing_by_name = {
        elem.get("name"): elem
        for document in documents
        for elem in document.root.iter("camera")
        if elem.get("name")
    }
    injected: list[dict[str, Any]] = []
    reused: list[str] = []
    moved: list[dict[str, str]] = []
    injected_mount_frames: list[dict[str, Any]] = []
    configured: list[dict[str, Any]] = []
    for camera_entry in camera_config["cameras"]:
        camera_name = str(camera_entry["name"])
        role = str(camera_entry.get("role") or camera_name)
        mount = camera_entry.get("mount") or {"type": "world"}
        mount_type = str(mount.get("type") or "world")
        fovy = _camera_fovy(camera_entry)
        if mount_type == "world":
            extrinsics_type = str((camera_entry.get("extrinsics") or {}).get("type") or "")
            if extrinsics_type == "table_relative_lookat":
                position, target = _table_relative_camera_pose(camera_entry, placement)
                reference_metadata = {
                    "extrinsics_type": extrinsics_type,
                    "reference_table_center_xy": list(placement.table_center_xy),
                    "reference_table_top_z": float(placement.table_top_z),
                }
            elif extrinsics_type == "body_frame_relative_lookat":
                position, target, reference_metadata = _body_frame_relative_camera_pose(
                    camera_entry,
                    documents,
                    parents,
                )
            else:
                raise ValueError(
                    f"Unsupported world camera extrinsics type {extrinsics_type!r} "
                    f"for camera {camera_name!r}."
                )
            xyaxes = _camera_xyaxes(position, target)
            if camera_name in existing_by_name:
                reused.append(camera_name)
                camera_elem = existing_by_name[camera_name]
                mark_document_modified(element_documents.get(camera_elem))
                source = "reconfigured_existing_camera"
            else:
                camera_elem = ET.SubElement(worldbody, "camera", {"name": camera_name})
                existing_by_name[camera_name] = camera_elem
                element_documents[camera_elem] = documents[0]
                injected.append({"name": camera_name, "mode": "fixed", "mount": "world"})
                source = "injected_camera"
            camera_elem.set("mode", "fixed")
            camera_elem.set("fovy", f"{fovy:.9g}")
            camera_elem.set("pos", _format_float_sequence(position))
            camera_elem.set("xyaxes", _format_float_sequence(xyaxes))
            camera_elem.attrib.pop("quat", None)
            camera_elem.attrib.pop("euler", None)
            configured.append(
                {
                    "name": camera_name,
                    "role": role,
                    "mount": "world",
                    "mode": "fixed",
                    "pos": position.tolist(),
                    "target": target.tolist(),
                    "xyaxes": xyaxes.tolist(),
                    "fovy": fovy,
                    "intrinsics": deepcopy(camera_entry.get("intrinsics") or {}),
                    "extrinsics": deepcopy(camera_entry.get("extrinsics") or {}),
                    "resolved_reference_frame": reference_metadata,
                    "randomization": deepcopy(camera_entry.get("randomization") or {}),
                    "source": source,
                }
            )
            continue

        if mount_type == "body":
            mount_body = str(mount.get("body") or "").strip()
            if not mount_body:
                raise ValueError(f"Body-mounted camera {camera_name!r} must define mount.body.")
            target_body, target_document, mount_frame_metadata = _resolve_camera_mount_body(
                documents,
                mount,
                parents,
                element_documents,
            )
            resolved_mount_body = target_body.get("name") or mount_body
            if mount_frame_metadata["injected"]:
                injected_mount_frames.append(deepcopy(mount_frame_metadata))
            pos = _parse_vec(_format_float_sequence(mount.get("pos") or [0.0, 0.0, 0.0]), length=3)
            quat = _parse_vec(_format_float_sequence(mount.get("quat") or [1.0, 0.0, 0.0, 0.0]), length=4)
            if camera_name in existing_by_name:
                reused.append(camera_name)
                camera_elem = existing_by_name[camera_name]
                camera_document = element_documents.get(camera_elem)
                current_parent = parents.get(camera_elem)
                if current_parent is not None and current_parent is not target_body:
                    current_parent.remove(camera_elem)
                    target_body.append(camera_elem)
                    parents[camera_elem] = target_body
                    mark_document_modified(camera_document)
                    element_documents[camera_elem] = target_document
                    moved.append(
                        {
                            "name": camera_name,
                            "from": current_parent.get("name") or current_parent.tag,
                            "to": resolved_mount_body,
                        }
                    )
                mark_document_modified(element_documents.get(camera_elem))
                source = "reconfigured_existing_camera"
            else:
                camera_elem = ET.SubElement(target_body, "camera", {"name": camera_name})
                existing_by_name[camera_name] = camera_elem
                element_documents[camera_elem] = target_document
                injected.append({"name": camera_name, "mode": "fixed", "mount": f"body:{resolved_mount_body}"})
                source = "injected_camera"
            mark_document_modified(target_document)
            camera_elem.set("mode", "fixed")
            camera_elem.set("fovy", f"{fovy:.9g}")
            camera_elem.set("pos", _format_float_sequence(pos))
            camera_elem.set("quat", _format_float_sequence(quat))
            camera_elem.attrib.pop("xyaxes", None)
            camera_elem.attrib.pop("euler", None)
            configured.append(
                {
                    "name": camera_name,
                    "role": role,
                    "mount": "body",
                    "mount_body": mount_body,
                    "resolved_mount_body": resolved_mount_body,
                    "mode": "fixed",
                    "local_pos": pos.tolist(),
                    "local_quat": quat.tolist(),
                    "mount_frame_resolution": deepcopy(mount_frame_metadata),
                    "fovy": fovy,
                    "intrinsics": deepcopy(camera_entry.get("intrinsics") or {}),
                    "extrinsics": {"mount": deepcopy(mount)},
                    "randomization": deepcopy(camera_entry.get("randomization") or {}),
                    "source": source,
                }
            )
            continue

        raise ValueError(f"Unsupported camera mount type {mount_type!r} for camera {camera_name!r}.")
    modified_include_files = _write_modified_include_documents(documents, modified_include_paths)
    return {
        "camera_config": camera_config,
        "requested_cameras": _camera_names_from_config(camera_config),
        "reused_cameras": reused,
        "injected_cameras": injected,
        "moved_cameras": moved,
        "injected_camera_mount_frames": injected_mount_frames,
        "configured_cameras": configured,
        "camera_randomization": _camera_randomization_by_name(camera_config),
        "modified_mjcf_include_files": modified_include_files,
        "resolved_mjcf_include_file_attr_count": resolved_mjcf_include_file_attr_count,
    }


def _apply_mujoco_311_xml_compatibility(root: ET.Element) -> dict[str, Any]:
    removed_mesh_inertia: list[dict[str, str]] = []
    removed_count = 0
    for elem in root.iter("mesh"):
        inertia = elem.attrib.pop("inertia", None)
        if inertia is None:
            continue
        removed_count += 1
        if len(removed_mesh_inertia) < 40:
            removed_mesh_inertia.append(
                {
                    "mesh": elem.get("name") or "",
                    "removed_attr": "inertia",
                    "removed_value": inertia,
                }
            )
    return {
        "mujoco_311_compatibility": {
            "removed_mesh_inertia_attr_count": removed_count,
            "removed_mesh_inertia_attr_examples": removed_mesh_inertia,
        }
    }


def _strip_invisible_collision_mesh_geoms(root: ET.Element) -> dict[str, Any]:
    """Remove transparent collision meshes that are irrelevant to state replay rendering.

    AXIS replay writes qpos / qvel / ctrl directly and only consumes RGB plus
    kinematic state. Transparent collision meshes therefore cannot affect the
    rendered observations, but MuJoCo still builds their convex hulls while
    compiling the model. A small number of backend meshes make that compile
    consume hundreds of GiB, so omit only geoms that are explicitly fully
    transparent and assigned to the collision group.
    """

    removed_names: list[str] = []
    for parent in root.iter():
        for geom in list(parent):
            if geom.tag != "geom" or geom.get("type") != "mesh":
                continue
            if geom.get("group", "0") != "0":
                continue
            rgba = geom.get("rgba", "").split()
            if len(rgba) != 4:
                continue
            try:
                alpha = float(rgba[3])
            except ValueError:
                continue
            if abs(alpha) > 1e-12:
                continue
            removed_names.append(geom.get("name") or geom.get("mesh") or "")
            parent.remove(geom)
    return {
        "state_replay_collision_mesh_policy": {
            "policy": "omit_fully_transparent_group_0_mesh_geoms",
            "removed_geom_count": len(removed_names),
            "removed_geom_names": removed_names,
        }
    }


def _compile_model_from_xml_string(xml: str, *, cwd: Path) -> mujoco.MjModel:
    previous_cwd = Path.cwd()
    os.chdir(cwd)
    try:
        return mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(previous_cwd)


def _model_names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj_type, obj_id) or "" for obj_id in range(count)]


def _model_layout_signature(model: mujoco.MjModel) -> dict[str, Any]:
    joint_names = _model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    body_names = _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    actuator_names = _model_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "joint_names": joint_names,
        "joint_qposadr": [int(value) for value in model.jnt_qposadr[: model.njnt]],
        "joint_dofadr": [int(value) for value in model.jnt_dofadr[: model.njnt]],
        "joint_type": [int(value) for value in model.jnt_type[: model.njnt]],
        "body_names": body_names,
        "actuator_names": actuator_names,
    }


def _replay_layout_cache_key(
    original_xml: str,
    final_xml: str,
    *,
    allowed_missing_body_names: set[str],
) -> str:
    final_root = ET.fromstring(final_xml)
    structural_projection = [
        {
            "tag": element.tag,
            "name": element.get("name"),
            "type": element.get("type") if element.tag in {"joint", "motor", "position", "velocity"} else None,
        }
        for element in final_root.iter()
    ]
    payload = {
        "schema": "axis_replay_layout_cache_v1",
        "original_xml_sha256": hashlib.sha256(original_xml.encode("utf-8")).hexdigest(),
        "final_structure": structural_projection,
        "allowed_missing_body_names": sorted(allowed_missing_body_names),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _verify_replay_layout_preserved(
    original_xml: str,
    final_xml: str,
    *,
    cwd: Path,
    allowed_missing_body_names: set[str],
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    cache_key = _replay_layout_cache_key(
        original_xml,
        final_xml,
        allowed_missing_body_names=allowed_missing_body_names,
    )
    cache_path = Path(cache_dir) / f"{cache_key}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("schema") != "axis_replay_layout_cache_v1" or cached.get("cache_key") != cache_key:
            raise RuntimeError(f"Invalid replay-layout cache entry: {cache_path}")
        return {
            **deepcopy(cached["layout_metadata"]),
            "replay_layout_cache_hit": True,
            "replay_layout_cache_key": cache_key,
        }

    original_model = _compile_model_from_xml_string(original_xml, cwd=cwd)
    final_model = _compile_model_from_xml_string(final_xml, cwd=cwd)
    original = _model_layout_signature(original_model)
    final = _model_layout_signature(final_model)
    missing_bodies = sorted(set(original["body_names"]) - set(final["body_names"]))
    unexpected_missing_bodies = sorted(set(missing_bodies) - allowed_missing_body_names)
    checks = {
        "nq_preserved": original["nq"] == final["nq"],
        "nv_preserved": original["nv"] == final["nv"],
        "nu_preserved": original["nu"] == final["nu"],
        "joint_names_preserved": original["joint_names"] == final["joint_names"],
        "joint_qpos_order_preserved": original["joint_qposadr"] == final["joint_qposadr"],
        "joint_dof_order_preserved": original["joint_dofadr"] == final["joint_dofadr"],
        "joint_types_preserved": original["joint_type"] == final["joint_type"],
        "actuator_names_preserved": original["actuator_names"] == final["actuator_names"],
        "robot_object_body_names_preserved": not unexpected_missing_bodies,
    }
    preserved = all(checks.values())
    if not preserved:
        failed = [name for name, ok in checks.items() if not ok]
        raise ValueError(f"Robosuite arena merge changed replay layout: {failed}")
    result = {
        "replay_layout_preserved": True,
        "replay_layout_checks": checks,
        "original_signature": {
            "nq": original["nq"],
            "nv": original["nv"],
            "nu": original["nu"],
            "num_joints": len(original["joint_names"]),
            "num_bodies": len(original["body_names"]),
            "joint_names": original["joint_names"],
            "actuator_names": original["actuator_names"],
        },
        "final_signature": {
            "nq": final["nq"],
            "nv": final["nv"],
            "nu": final["nu"],
            "num_joints": len(final["joint_names"]),
            "num_bodies": len(final["body_names"]),
            "joint_names": final["joint_names"],
            "actuator_names": final["actuator_names"],
        },
        "allowed_missing_scene_shell_body_names": sorted(allowed_missing_body_names),
        "missing_original_body_names": missing_bodies,
        "unexpected_missing_original_body_names": unexpected_missing_bodies,
        "replay_layout_cache_hit": False,
        "replay_layout_cache_key": cache_key,
    }
    if cache_path is not None:
        lock_dir = cache_path.parent / ".locks" / cache_key
        with _exclusive_materialization_lock(lock_dir):
            if not cache_path.is_file():
                atomic_write_json(
                    cache_path,
                    {
                        "schema": "axis_replay_layout_cache_v1",
                        "cache_key": cache_key,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "layout_metadata": {
                            key: value
                            for key, value in result.items()
                            if key not in {"replay_layout_cache_hit", "replay_layout_cache_key"}
                        },
                    },
                )
    return result


def _prepare_robosuite_xml(
    entry_xml: Path,
    camera_config: dict[str, Any],
    *,
    layout_cache_dir: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    entry_xml = Path(entry_xml).expanduser().resolve()
    camera_names = _camera_names_from_config(camera_config)
    root = ET.parse(entry_xml).getroot()
    path_metadata = _absolutize_xml_file_attrs(root, xml_dir=entry_xml.parent)
    derived_include_metadata = _derive_mjcf_include_tree(root, entry_xml=entry_xml)
    try:
        compatibility_metadata = _apply_mujoco_311_xml_compatibility(root)
        placement = _infer_scene_placement(root, camera_config.get("render_profile"))
        body_transform_metadata = _apply_arm_scene_z_offset(root, placement)
        source_table_render_metadata = _configure_preserved_source_table(root, placement)
        collision_mesh_metadata = _strip_invisible_collision_mesh_geoms(root)
        original_replay_xml = ET.tostring(root, encoding="unicode")
        arena_metadata = _merge_robosuite_table_arena(root, camera_names, placement)
        visual_scene_metadata = apply_visual_scene(
            root,
            camera_config,
            table_center_xy=placement.table_center_xy,
            table_top_z=placement.table_top_z,
            table_full_size=placement.table_full_size,
            preserve_source_table=placement.preserve_source_table,
            arena_prefix=ARENA_PREFIX,
        )
        appearance_metadata = _apply_embodiment_appearance(
            root,
            camera_config,
            entry_xml=entry_xml,
        )
        surface_appearance_metadata = _apply_fixed_surface_appearance(root, camera_config, placement)
        lighting_rig_metadata = _configure_reference_lighting_rig(root, camera_config, placement)
        camera_metadata = _ensure_configured_cameras(root, camera_config, placement, entry_xml=entry_xml)
        final_xml = ET.tostring(root, encoding="unicode")
        layout_metadata = _verify_replay_layout_preserved(
            original_replay_xml,
            final_xml,
            cwd=entry_xml.parent,
            allowed_missing_body_names=set(arena_metadata["removed_original_scene_shell_body_names"]),
            cache_dir=layout_cache_dir,
        )
    except Exception:
        try:
            _cleanup_derived_mjcf_include_files(derived_include_metadata)
        except OSError:
            logger.exception("Failed to clean derived MJCF includes after XML preparation failure.")
        raise
    metadata = {
        "entry_xml": str(entry_xml),
        "xml_build_policy": "robosuite_table_arena_with_embodiment_aware_table_placement",
        **path_metadata,
        **derived_include_metadata,
        **collision_mesh_metadata,
        **source_table_render_metadata,
        **arena_metadata,
        **visual_scene_metadata,
        **appearance_metadata,
        **surface_appearance_metadata,
        **lighting_rig_metadata,
        **camera_metadata,
        **compatibility_metadata,
        **body_transform_metadata,
        **layout_metadata,
    }
    return final_xml, metadata


class AxisReplayXMLModel:
    """Minimal Robosuite model adapter around a materialized AXIS MJCF string."""

    def __init__(self, xml: str, metadata: dict[str, Any]):
        self._xml = xml
        self.metadata = metadata
        self.mujoco_objects: list[Any] = []

    def get_xml(self) -> str:
        return self._xml

    def generate_id_mappings(self, sim: Any) -> None:
        self.geom_names = list(getattr(sim.model, "geom_names", []))
        self.camera_names = list(getattr(sim.model, "camera_names", []))
        self.light_names = list(getattr(sim.model, "light_names", []))


def _apply_default_camera_clipping(sim: MjSim) -> dict[str, Any]:
    model = sim.model._model
    extent = float(model.stat.extent)
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError(f"MuJoCo model extent must be positive, got {extent}.")
    original_near_fraction = float(model.vis.map.znear)
    original_far_fraction = float(model.vis.map.zfar)
    near_m = float(DEFAULT_CAMERA_CLIPPING_NEAR_M)
    far_m = float(DEFAULT_CAMERA_CLIPPING_FAR_M)
    if not math.isfinite(near_m) or not math.isfinite(far_m):
        raise ValueError(
            f"Default camera clipping values must be finite, got near_m={near_m}, far_m={far_m}."
        )
    if near_m <= 0.0:
        raise ValueError(f"Default camera clipping near_m must be positive, got {near_m}.")
    if far_m <= near_m:
        raise ValueError(
            f"Default camera clipping far_m must be greater than near_m, got near_m={near_m}, far_m={far_m}."
        )
    model.vis.map.znear = near_m / extent
    model.vis.map.zfar = far_m / extent
    resolved_near_fraction = float(model.vis.map.znear)
    resolved_far_fraction = float(model.vis.map.zfar)
    return {
        "camera_clipping_profile": {
            "applied": True,
            "policy": "renderer_global_default",
            "model_extent_m": extent,
            "requested_near_m": near_m,
            "requested_far_m": far_m,
            "original_znear_fraction": original_near_fraction,
            "original_zfar_fraction": original_far_fraction,
            "original_effective_near_m": original_near_fraction * extent,
            "original_effective_far_m": original_far_fraction * extent,
            "resolved_znear_fraction": resolved_near_fraction,
            "resolved_zfar_fraction": resolved_far_fraction,
            "resolved_effective_near_m": resolved_near_fraction * extent,
            "resolved_effective_far_m": resolved_far_fraction * extent,
        }
    }


class AxisReplayRobosuiteEnv(MujocoEnv):
    """Robosuite environment that hosts the original AXIS robot/object MJCF for replay rendering."""

    def __init__(
        self,
        *,
        entry_xml: str | Path,
        camera_names: tuple[str, ...],
        camera_config: dict[str, Any],
        layout_cache_dir: str | Path | None = None,
        **kwargs: Any,
    ):
        self.entry_xml = Path(entry_xml).expanduser().resolve()
        self.requested_camera_names = tuple(camera_names)
        self.camera_config = deepcopy(camera_config)
        self.layout_cache_dir = (
            Path(layout_cache_dir).expanduser().resolve() if layout_cache_dir is not None else None
        )
        self.xml_metadata: dict[str, Any] = {}
        super().__init__(render_camera=list(camera_names), **kwargs)

    def _load_model(self) -> None:
        xml, metadata = _prepare_robosuite_xml(
            self.entry_xml,
            self.camera_config,
            layout_cache_dir=self.layout_cache_dir,
        )
        self.xml_metadata = metadata
        self.model = AxisReplayXMLModel(xml, metadata)

    def _initialize_sim(self, xml_string: str | None = None) -> None:
        xml = xml_string if xml_string else self.model.get_xml()
        for processor in self._xml_processors:
            xml = processor(xml)
        cwd = Path.cwd()
        os.chdir(self.entry_xml.parent)
        try:
            self.sim = MjSim.from_xml_string(xml)
        finally:
            os.chdir(cwd)
            self.xml_metadata.update(_cleanup_derived_mjcf_include_files(self.xml_metadata))
        self.xml_metadata.update(_apply_default_camera_clipping(self.sim))
        self.sim.forward()
        self.initialize_time(self.control_freq)

    def edit_model_xml(self, xml_str: str) -> str:
        return xml_str

    def reward(self, action: np.ndarray | None = None) -> float:
        return 0.0

    def _check_success(self) -> bool:
        return False

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        nu = int(self.sim.model.nu) if getattr(self, "sim", None) is not None else 0
        return np.zeros(nu, dtype=np.float32), np.zeros(nu, dtype=np.float32)

    def visualize(self, vis_settings: dict[str, bool]) -> None:
        return None


def _render_rgb_cameras_shared_scene(
    sim: MjSim,
    camera_names: tuple[str, ...],
    *,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    """Render one simulation state from many cameras with one full scene update."""
    if not camera_names:
        raise ValueError("camera_names must contain at least one camera.")
    if len(set(camera_names)) != len(camera_names):
        raise ValueError(f"camera_names must be unique; got {camera_names!r}.")
    if width <= 0 or height <= 0:
        raise ValueError(f"Render dimensions must be positive; got width={width}, height={height}.")

    context = sim._render_context_offscreen
    if context is None:
        raise RuntimeError("Shared-scene rendering requires an initialized offscreen render context.")
    headlight = sim.model.vis.headlight
    headlight_values = np.concatenate(
        [
            np.asarray(headlight.ambient, dtype=np.float64),
            np.asarray(headlight.diffuse, dtype=np.float64),
            np.asarray(headlight.specular, dtype=np.float64),
        ]
    )
    if np.any(headlight_values != 0.0):
        raise RuntimeError(
            "Shared-scene multi-camera rendering requires the camera-relative MuJoCo "
            "headlight to be disabled; otherwise mjv_updateCamera would leave the first "
            "camera's headlight in the shared scene."
        )

    frames: dict[str, np.ndarray] = {}
    atlas_columns = min(
        len(camera_names),
        max(1, int(math.ceil(math.sqrt(len(camera_names) * float(width) / float(height))))),
    )
    atlas_rows = int(math.ceil(len(camera_names) / atlas_columns))
    atlas_width = atlas_columns * width
    atlas_height = atlas_rows * height
    with robosuite_binding_utils._MjSim_render_lock:
        context.gl_ctx.make_current()
        if atlas_width > context.con.offWidth or atlas_height > context.con.offHeight:
            context.update_offscreen_size(
                max(atlas_width, sim.model.vis.global_.offwidth),
                max(atlas_height, sim.model.vis.global_.offheight),
            )

        # The render context is mutable and can outlive an environment reset. Enforce
        # the RGB contract at the scene-update boundary so a stale visualization
        # option cannot expose robosuite's coplanar yellow collision tabletop and
        # make it z-fight with the textured visual tabletop.
        if len(context.vopt.geomgroup) < 2:
            raise RuntimeError("RGB rendering requires MuJoCo geom groups 0 and 1.")
        # The mujoco 3.1.1 Python binding
        # exposes vopt.geomgroup as a numpy view with strides=(0,), meaning every
        # index aliases the same underlying C byte. Writing geomgroup[0]=0 then
        # geomgroup[1]=1 leaves byte 0 as 1 (last write wins) and never touches
        # bytes 1..5, so group 0 (collision) stays enabled. Bypass numpy and write
        # only bytes 0 and 1 directly via ctypes so we hide the collision group
        # without disturbing the rest (robot / object visuals may live in group 2
        # etc. and must remain enabled per mjv_defaultOption which sets bytes 2..5
        # to their defaults on init).
        import ctypes as _ctypes
        _geomgroup_ptr, _ = context.vopt.geomgroup.__array_interface__["data"]
        _ctypes.memmove(_geomgroup_ptr, bytes([0]), 1)
        _ctypes.memmove(_geomgroup_ptr + 1, bytes([1]), 1)
        for camera_index, camera_name in enumerate(camera_names):
            column = camera_index % atlas_columns
            row = camera_index // atlas_columns
            viewport = mujoco.MjrRect(column * width, row * height, width, height)
            context.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            context.cam.fixedcamid = sim.model.camera_name2id(camera_name)
            if camera_index == 0:
                mujoco.mjv_updateScene(
                    context.model._model,
                    context.data._data,
                    context.vopt,
                    context.pert,
                    context.cam,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    context.scn,
                )
            else:
                mujoco.mjv_updateCamera(
                    context.model._model,
                    context.data._data,
                    context.cam,
                    context.scn,
                )
            mujoco.mjr_render(viewport=viewport, scn=context.scn, con=context.con)
        atlas = np.asarray(
            context.read_pixels(atlas_width, atlas_height, depth=False, segmentation=False),
            dtype=np.uint8,
        )
        for camera_index, camera_name in enumerate(camera_names):
            column = camera_index % atlas_columns
            row = camera_index // atlas_columns
            tile = atlas[row * height : (row + 1) * height, column * width : (column + 1) * width]
            frames[camera_name] = np.flipud(tile).copy()
    return frames


def _scene_texture_geom_names(env: AxisReplayRobosuiteEnv) -> dict[str, list[str]]:
    names = [name for name in env.sim.model.geom_names if name]
    generated_scene_prefix = f"{ARENA_PREFIX}visual_scene_"
    requested = set((env.xml_metadata.get("table_surface_geom_names") or []))
    table_names = [name for name in names if name in requested]
    if not table_names:
        table_names = [
            name
            for name in names
            if name.startswith(ARENA_PREFIX)
            and not name.startswith(generated_scene_prefix)
            and "table" in name.lower()
            and "visual" in name.lower()
        ]
    if not table_names:
        raise ValueError("No Robosuite table surface geom found for table texture randomization.")
    floor_names = [
        name
        for name in names
        if name == f"{ARENA_PREFIX}floor"
        or (
            name.startswith(ARENA_PREFIX)
            and not name.startswith(generated_scene_prefix)
            and "floor" in name.lower()
        )
    ]
    if not floor_names:
        raise ValueError("No Robosuite floor geom found for floor texture randomization.")
    wall_names = [
        name
        for name in names
        if name.startswith(ARENA_PREFIX)
        and not name.startswith(generated_scene_prefix)
        and "wall" in name.lower()
        and "visual" in name.lower()
    ]
    if not wall_names:
        raise ValueError("No Robosuite wall visual geoms found for wall texture randomization.")
    return {
        "table": sorted(dict.fromkeys(table_names)),
        "floor": sorted(dict.fromkeys(floor_names)),
        "wall": sorted(dict.fromkeys(wall_names)),
    }


def _camera_randomization_metadata(env: AxisReplayRobosuiteEnv, camera_names: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    raw_model = env.sim.model._model
    for camera_name in camera_names:
        camera_id = int(env.sim.model.camera_name2id(camera_name))
        mount_body_id = int(env.sim.model.cam_bodyid[camera_id])
        mount_body_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_BODY, mount_body_id)
        result[camera_name] = {
            "camera_id": camera_id,
            "camera_type": "robosuite_fixed",
            "mount_body_id": mount_body_id,
            "mount_body_name": mount_body_name,
            "pose_frame": "world" if mount_body_id == 0 else f"body_local:{mount_body_name}",
            "camera_pos": np.asarray(env.sim.model.cam_pos[camera_id], dtype=np.float64).tolist(),
            "camera_quat": np.asarray(env.sim.model.cam_quat[camera_id], dtype=np.float64).tolist(),
            "camera_fovy": float(env.sim.model.cam_fovy[camera_id]),
        }
    return result


def _lighting_randomization_metadata(env: AxisReplayRobosuiteEnv, light_names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for light_name in light_names:
        light_id = int(env.sim.model.light_name2id(light_name))
        result[light_name] = {
            "light_id": light_id,
            "light_pos": np.asarray(env.sim.model.light_pos[light_id], dtype=np.float64).tolist(),
            "light_dir": np.asarray(env.sim.model.light_dir[light_id], dtype=np.float64).tolist(),
            "ambient": np.asarray(env.sim.model.light_ambient[light_id], dtype=np.float64).tolist(),
            "diffuse": np.asarray(env.sim.model.light_diffuse[light_id], dtype=np.float64).tolist(),
            "specular": np.asarray(env.sim.model.light_specular[light_id], dtype=np.float64).tolist(),
            "active": int(env.sim.model.light_active[light_id]),
        }
    return result


def _apply_fixed_lighting_profile(
    env: AxisReplayRobosuiteEnv,
    light_names: list[str],
    lighting_profile: dict[str, Any],
) -> dict[str, Any]:
    if not lighting_profile:
        return {"applied": False, "mode": "arena_default"}
    if not isinstance(lighting_profile, dict):
        raise ValueError("render_profile.lighting must be a JSON object.")
    mode = str(lighting_profile.get("mode") or "").strip()
    if mode == "reference_five_light_rig":
        expected_names = {
            f"{ARENA_PREFIX}{str(source.get('name') or '').strip()}"
            for source in (lighting_profile.get("sources") or [])
        }
        missing_names = sorted(expected_names - set(light_names))
        if missing_names:
            raise ValueError(f"Reference lighting rig is missing compiled lights: {missing_names}")
        return {
            "applied": True,
            "mode": mode,
            "definition": "mjcf_xml_reference_rig",
            "light_names": sorted(expected_names),
        }
    if mode != "fixed_soft_neutral":
        raise ValueError(f"Unsupported render_profile lighting mode {mode!r}.")
    values_by_array: dict[str, np.ndarray] = {}
    for profile_key, array_name in (
        ("ambient", "light_ambient"),
        ("diffuse", "light_diffuse"),
        ("specular", "light_specular"),
    ):
        values = np.asarray(lighting_profile.get(profile_key) or [], dtype=np.float64)
        if values.shape != (3,) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"render_profile.lighting.{profile_key} must contain three values in [0, 1].")
        values_by_array[array_name] = values
    for light_name in light_names:
        light_id = int(env.sim.model.light_name2id(light_name))
        for array_name, values in values_by_array.items():
            getattr(env.sim.model, array_name)[light_id, :] = values
        env.sim.model.light_active[light_id] = 1
    env.sim.forward()
    return {
        "applied": True,
        "mode": mode,
        "ambient": values_by_array["light_ambient"].tolist(),
        "diffuse": values_by_array["light_diffuse"].tolist(),
        "specular": values_by_array["light_specular"].tolist(),
    }


def _randomize_neutral_light_intensity(
    env: AxisReplayRobosuiteEnv,
    light_names: list[str],
    *,
    random_state: np.random.RandomState,
    ambient_perturbation_size: float,
    diffuse_perturbation_size: float,
    specular_perturbation_size: float,
) -> None:
    perturbation_by_array = {
        "light_ambient": ambient_perturbation_size,
        "light_diffuse": diffuse_perturbation_size,
        "light_specular": specular_perturbation_size,
    }
    for light_name in light_names:
        light_id = int(env.sim.model.light_name2id(light_name))
        for array_name, perturbation_size in perturbation_by_array.items():
            values = getattr(env.sim.model, array_name)
            base_intensity = float(np.mean(values[light_id]))
            intensity = max(0.0, base_intensity + float(random_state.uniform(-perturbation_size, perturbation_size)))
            values[light_id, :] = intensity


def _jitter_rgb(random_state: np.random.RandomState, rgb: tuple[float, float, float], amount: float = 10.0) -> np.ndarray:
    return np.clip(np.asarray(rgb, dtype=np.float64) + random_state.uniform(-amount, amount, size=3), 0, 255).astype(
        np.uint8
    )


def _jitter_material(
    random_state: np.random.RandomState,
    material: tuple[float, float, float],
    amount: float = 0.08,
) -> np.ndarray:
    return np.clip(np.asarray(material, dtype=np.float64) + random_state.uniform(-amount, amount, size=3), 0.0, 1.0)


def _configured_surface_constraints(render_profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_constraints = render_profile.get("surface_constraints") or {}
    if not isinstance(raw_constraints, dict):
        raise ValueError("render_profile.surface_constraints must be an object.")
    unknown_surfaces = sorted(set(str(name) for name in raw_constraints) - set(SCENE_TEXTURE_SURFACES))
    if unknown_surfaces:
        raise ValueError(f"Unknown constrained texture surfaces: {unknown_surfaces}.")

    resolved: dict[str, dict[str, Any]] = {}
    for surface_name, raw_surface in raw_constraints.items():
        if not isinstance(raw_surface, dict):
            raise ValueError(f"Surface constraints for {surface_name!r} must be an object.")
        unknown_fields = sorted(set(raw_surface) - {"image_luminance_guard", "material_caps"})
        if unknown_fields:
            raise ValueError(
                f"Unknown surface constraint fields for {surface_name!r}: {unknown_fields}."
            )
        surface_constraints: dict[str, Any] = {}

        raw_guard = raw_surface.get("image_luminance_guard")
        if raw_guard is not None:
            if not isinstance(raw_guard, dict):
                raise ValueError(
                    f"image_luminance_guard for {surface_name!r} must be an object."
                )
            missing_guard_fields = sorted(
                {"metric", "mode", "percentile", "max_value"} - set(raw_guard)
            )
            if missing_guard_fields:
                raise ValueError(
                    f"image_luminance_guard for {surface_name!r} is missing "
                    f"{missing_guard_fields}."
                )
            metric = str(raw_guard.get("metric") or "").strip()
            mode = str(raw_guard.get("mode") or "").strip()
            percentile = float(raw_guard.get("percentile"))
            maximum = float(raw_guard.get("max_value"))
            if metric != "bt709_srgb_uint8_percentile":
                raise ValueError(f"Unsupported image luminance metric {metric!r}.")
            if mode != "downscale_only":
                raise ValueError(f"Unsupported image luminance guard mode {mode!r}.")
            if not math.isfinite(percentile) or not 0.0 < percentile <= 100.0:
                raise ValueError("image_luminance_guard.percentile must be in (0, 100].")
            if not math.isfinite(maximum) or not 0.0 < maximum <= 255.0:
                raise ValueError("image_luminance_guard.max_value must be in (0, 255].")
            surface_constraints["image_luminance_guard"] = {
                "metric": metric,
                "mode": mode,
                "percentile": percentile,
                "max_value": maximum,
            }

        raw_caps = raw_surface.get("material_caps")
        if raw_caps is not None:
            if not isinstance(raw_caps, dict) or not raw_caps:
                raise ValueError(f"material_caps for {surface_name!r} must be a non-empty object.")
            allowed_caps = {"max_reflectance", "max_shininess", "max_specular"}
            unknown_caps = sorted(set(raw_caps) - allowed_caps)
            if unknown_caps:
                raise ValueError(f"Unknown material caps for {surface_name!r}: {unknown_caps}.")
            material_caps: dict[str, float] = {}
            for cap_name, raw_value in raw_caps.items():
                value = float(raw_value)
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{surface_name}.{cap_name} must be in [0, 1].")
                material_caps[str(cap_name)] = value
            surface_constraints["material_caps"] = material_caps

        if not surface_constraints:
            raise ValueError(f"Surface constraints for {surface_name!r} cannot be empty.")
        resolved[str(surface_name)] = surface_constraints
    return resolved


def _apply_image_luminance_guard(
    bitmap: np.ndarray,
    guard: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if bitmap.ndim != 3 or bitmap.shape[-1] not in (3, 4):
        raise ValueError(f"Image luminance guard requires RGB or RGBA bitmap, got {bitmap.shape}.")
    if bitmap.dtype != np.uint8:
        raise ValueError(f"Image luminance guard requires uint8 bitmap, got {bitmap.dtype}.")
    percentile = float(guard["percentile"])
    maximum = float(guard["max_value"])
    rgb = np.asarray(bitmap[..., :3], dtype=np.float64)
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    source_luminance = rgb @ weights
    source_value = float(np.percentile(source_luminance, percentile))
    gain = min(1.0, maximum / source_value) if source_value > 0.0 else 1.0
    resolved = np.asarray(bitmap).copy()
    if gain < 1.0:
        resolved[..., :3] = np.clip(np.floor(rgb * gain), 0.0, 255.0).astype(np.uint8)
    resolved_luminance = np.asarray(resolved[..., :3], dtype=np.float64) @ weights
    resolved_value = float(np.percentile(resolved_luminance, percentile))
    if resolved_value > maximum + 1e-9:
        raise ValueError(
            "Downscale-only image luminance guard failed its configured limit: "
            f"resolved={resolved_value}, maximum={maximum}."
        )
    return resolved, {
        "metric": guard["metric"],
        "mode": guard["mode"],
        "percentile": percentile,
        "max_value": maximum,
        "source_percentile_value": source_value,
        "resolved_percentile_value": resolved_value,
        "gain": float(gain),
        "applied": bool(gain < 1.0),
    }


def _load_scene_image_texture_bitmap(
    modder: TextureModder,
    geom_name: str,
    family: dict[str, Any],
    *,
    luminance_guard: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    texture_file = str(family["texture_file"])
    texture_asset_dir = Path(str(family.get("_texture_asset_dir") or ""))
    if not texture_asset_dir.is_absolute():
        raise ValueError(f"Scene texture asset directory must be absolute: {texture_asset_dir}")
    texture_path = texture_asset_dir / texture_file
    if not texture_path.is_file():
        raise FileNotFoundError(f"Scene texture image is missing: {texture_path}")

    from PIL import Image

    target_bitmap = modder.get_texture(geom_name).bitmap
    target_h, target_w = target_bitmap.shape[:2]
    with Image.open(texture_path) as image:
        image = image.convert("RGB")
        if image.size != (target_w, target_h):
            resampling = getattr(Image, "Resampling", Image).BILINEAR
            image = image.resize((target_w, target_h), resampling)
        bitmap = np.asarray(image, dtype=target_bitmap.dtype)

    if bitmap.shape[-1] != target_bitmap.shape[-1]:
        if target_bitmap.shape[-1] == 4 and bitmap.shape[-1] == 3:
            alpha = np.full((*bitmap.shape[:2], 1), 255, dtype=bitmap.dtype)
            bitmap = np.concatenate([bitmap, alpha], axis=-1)
        else:
            raise ValueError(
                f"Scene texture channel mismatch for {texture_file}: "
                f"image={bitmap.shape}, target={target_bitmap.shape}"
            )
    guard_metadata = None
    if luminance_guard is not None:
        bitmap, guard_metadata = _apply_image_luminance_guard(bitmap, luminance_guard)
    return bitmap, guard_metadata


def _sample_and_apply_surface_material_family(
    env: AxisReplayRobosuiteEnv,
    geom_names: list[str],
    *,
    surface_name: str,
    candidates: tuple[tuple[int, dict[str, Any]], ...],
    random_state: np.random.RandomState,
    surface_constraints: dict[str, Any],
) -> dict[str, Any]:
    candidate_index = int(random_state.randint(len(candidates)))
    family_index, family = candidates[candidate_index]
    variation = str(family["texture_variation"])
    rgb1 = _jitter_rgb(random_state, family["rgb1"]) if variation != "image" else None
    rgb2 = _jitter_rgb(random_state, family["rgb2"]) if variation != "image" else None
    material = _jitter_material(random_state, family["material"])
    vertical = bool(random_state.uniform() > 0.5)
    fraction = float(np.clip(float(family.get("fraction", 0.45)) + random_state.uniform(-0.08, 0.08), 0.08, 0.92))
    modder = TextureModder(
        env.sim,
        random_state=random_state,
        geom_names=geom_names,
        randomize_local=False,
        randomize_material=False,
        randomize_skybox=False,
    )
    modder.whiten_materials()
    luminance_guard = surface_constraints.get("image_luminance_guard")
    geom_updates: dict[str, Any] = {}
    for geom_name in geom_names:
        texture_bitmap_shape = None
        luminance_guard_metadata = None
        if variation == "image":
            bitmap, luminance_guard_metadata = _load_scene_image_texture_bitmap(
                modder,
                geom_name,
                family,
                luminance_guard=luminance_guard,
            )
            modder.set_texture(geom_name, bitmap, perturb=False)
            texture_bitmap_shape = list(bitmap.shape)
        elif variation == "checker":
            assert rgb1 is not None and rgb2 is not None
            modder.set_checker(geom_name, rgb1, rgb2, perturb=False)
        elif variation == "noise":
            assert rgb1 is not None and rgb2 is not None
            modder.set_noise(geom_name, rgb1, rgb2, fraction=fraction, perturb=False)
        elif variation == "gradient":
            assert rgb1 is not None and rgb2 is not None
            modder.set_gradient(geom_name, rgb1, rgb2, vertical=vertical, perturb=False)
        else:
            raise ValueError(f"Unsupported scene texture variation: {variation!r}")
        render_context = getattr(env.sim, "_render_context_offscreen", None)
        if render_context is not None:
            render_context.upload_texture(modder.get_texture(geom_name).id)
        modder.set_material(geom_name, material, perturb=False)
        geom_id = int(env.sim.model.geom_name2id(geom_name))
        mat_id = int(env.sim.model.geom_matid[geom_id])
        geom_updates[geom_name] = {
            "geom_id": geom_id,
            "material_id": mat_id,
            "geom_rgba": np.asarray(env.sim.model.geom_rgba[geom_id], dtype=np.float64).tolist(),
            "mat_rgba": np.asarray(env.sim.model.mat_rgba[mat_id], dtype=np.float64).tolist(),
            "reflectance": float(env.sim.model.mat_reflectance[mat_id]),
            "shininess": float(env.sim.model.mat_shininess[mat_id]),
            "specular": float(env.sim.model.mat_specular[mat_id]),
            "texture_bitmap_shape": texture_bitmap_shape,
            "image_luminance_guard": luminance_guard_metadata,
        }
    return {
        "surface_name": surface_name,
        "material_family_index": family_index,
        "surface_candidate_index": candidate_index,
        "surface_candidate_count": len(candidates),
        "material_variant_id": family.get("material_variant_id"),
        "material_family": family["material_family"],
        "texture_variation": variation,
        "rgb1": rgb1.tolist() if rgb1 is not None else None,
        "rgb2": rgb2.tolist() if rgb2 is not None else None,
        "noise_fraction": fraction if variation == "noise" else None,
        "gradient_vertical": vertical if variation == "gradient" else None,
        "texture_file": family.get("texture_file"),
        "texture_bank_id": family.get("texture_bank_id"),
        "sha256": family.get("sha256"),
        "bytes": family.get("bytes"),
        "source_asset_id": family.get("source_asset_id"),
        "source_asset_type": family.get("source_asset_type"),
        "source_url": family.get("source_url"),
        "direct_download_url": family.get("direct_download_url"),
        "license": family.get("license"),
        "source_map_type": family.get("source_map_type"),
        "source_resolution": family.get("source_resolution"),
        "source_format": family.get("source_format"),
        "source_md5": family.get("source_md5"),
        "source_bytes": family.get("source_bytes"),
        "resize": family.get("resize"),
        "material": material.tolist(),
        "surface_constraints": deepcopy(surface_constraints),
        "geom_updates": geom_updates,
    }


def _apply_surface_material_caps(
    env: AxisReplayRobosuiteEnv,
    geom_names_by_surface: dict[str, list[str]],
    surface_constraints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    model = env.sim.model
    results: dict[str, Any] = {}
    material_owner: dict[int, str] = {}
    field_specs = (
        ("max_reflectance", "mat_reflectance"),
        ("max_shininess", "mat_shininess"),
        ("max_specular", "mat_specular"),
    )
    for surface_name, constraints in surface_constraints.items():
        caps = constraints.get("material_caps")
        if not caps:
            continue
        material_updates: dict[int, dict[str, Any]] = {}
        geom_updates: dict[str, Any] = {}
        for geom_name in geom_names_by_surface[surface_name]:
            geom_id = int(model.geom_name2id(geom_name))
            mat_id = int(model.geom_matid[geom_id])
            if mat_id < 0:
                raise ValueError(
                    f"Configured material caps require geom {geom_name!r} to reference a material."
                )
            previous_owner = material_owner.get(mat_id)
            if previous_owner is not None and previous_owner != surface_name:
                raise ValueError(
                    f"Material {mat_id} is shared by constrained surfaces "
                    f"{previous_owner!r} and {surface_name!r}."
                )
            material_owner[mat_id] = surface_name
            if mat_id not in material_updates:
                before = {
                    "reflectance": float(model.mat_reflectance[mat_id]),
                    "shininess": float(model.mat_shininess[mat_id]),
                    "specular": float(model.mat_specular[mat_id]),
                }
                for cap_name, model_field in field_specs:
                    if cap_name in caps:
                        values = getattr(model, model_field)
                        values[mat_id] = min(float(values[mat_id]), float(caps[cap_name]))
                after = {
                    "reflectance": float(model.mat_reflectance[mat_id]),
                    "shininess": float(model.mat_shininess[mat_id]),
                    "specular": float(model.mat_specular[mat_id]),
                }
                material_updates[mat_id] = {
                    "material_id": mat_id,
                    "before": before,
                    "after": after,
                    "changed": before != after,
                }
            geom_updates[geom_name] = {
                "geom_id": geom_id,
                "material_id": mat_id,
            }
        results[surface_name] = {
            "material_caps": deepcopy(caps),
            "material_updates": list(material_updates.values()),
            "geom_updates": geom_updates,
        }
    return results


def _sample_and_apply_local_material_jitter(
    env: AxisReplayRobosuiteEnv,
    geom_names: list[str],
    *,
    surface_name: str,
    random_state: np.random.RandomState,
    rgb_jitter: float,
    material_jitter: float,
) -> dict[str, Any]:
    if not 0.0 <= rgb_jitter <= 1.0:
        raise ValueError(f"rgb_jitter must be in [0, 1], got {rgb_jitter}.")
    if not 0.0 <= material_jitter <= 1.0:
        raise ValueError(f"material_jitter must be in [0, 1], got {material_jitter}.")
    model = env.sim.model
    geom_updates: dict[str, Any] = {}
    updated_material_ids: set[int] = set()
    for geom_name in geom_names:
        geom_id = int(model.geom_name2id(geom_name))
        mat_id = int(model.geom_matid[geom_id])
        if mat_id >= 0 and mat_id not in updated_material_ids:
            original_rgba = np.asarray(model.mat_rgba[mat_id], dtype=np.float64).copy()
            original_material = np.asarray(
                [model.mat_reflectance[mat_id], model.mat_shininess[mat_id], model.mat_specular[mat_id]],
                dtype=np.float64,
            )
            resolved_rgba = original_rgba.copy()
            resolved_rgba[:3] = np.clip(
                original_rgba[:3] + random_state.uniform(-rgb_jitter, rgb_jitter, size=3),
                0.0,
                1.0,
            )
            resolved_material = np.clip(
                original_material + random_state.uniform(-material_jitter, material_jitter, size=3),
                0.0,
                1.0,
            )
            model.mat_rgba[mat_id, :] = resolved_rgba
            model.mat_reflectance[mat_id] = resolved_material[0]
            model.mat_shininess[mat_id] = resolved_material[1]
            model.mat_specular[mat_id] = resolved_material[2]
            updated_material_ids.add(mat_id)
            geom_updates[geom_name] = {
                "geom_id": geom_id,
                "material_id": mat_id,
                "original_rgba": original_rgba.tolist(),
                "resolved_rgba": resolved_rgba.tolist(),
                "original_material": original_material.tolist(),
                "resolved_material": resolved_material.tolist(),
            }
        elif mat_id < 0:
            original_rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float64).copy()
            resolved_rgba = original_rgba.copy()
            resolved_rgba[:3] = np.clip(
                original_rgba[:3] + random_state.uniform(-rgb_jitter, rgb_jitter, size=3),
                0.0,
                1.0,
            )
            model.geom_rgba[geom_id, :] = resolved_rgba
            geom_updates[geom_name] = {
                "geom_id": geom_id,
                "material_id": -1,
                "original_rgba": original_rgba.tolist(),
                "resolved_rgba": resolved_rgba.tolist(),
                "original_material": None,
                "resolved_material": None,
            }
    env.sim.forward()
    return {
        "surface_name": surface_name,
        "mode": "local_material_jitter",
        "rgb_jitter": float(rgb_jitter),
        "material_jitter": float(material_jitter),
        "geom_updates": geom_updates,
    }


def _require_robosuite_texture_modder_runtime() -> None:
    if mujoco.__version__ != "3.1.1":
        raise RuntimeError(
            "Robosuite TextureModder requires mujoco==3.1.1 in this renderer. "
            f"Current mujoco version is {mujoco.__version__!r}."
        )


def _sample_and_apply_randomization(
    env: AxisReplayRobosuiteEnv,
    *,
    task_id: int,
    attempt_id: int,
    task_name: str,
    variant_id: int,
    global_seed: int,
    camera_names: tuple[str, ...],
    camera_config: dict[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    render_profile = camera_config.get("render_profile") or {}
    if not isinstance(render_profile, dict):
        raise ValueError("render_profile must be a JSON object.")
    randomization_profile = render_profile.get("randomization") or {}
    if not isinstance(randomization_profile, dict):
        raise ValueError("render_profile.randomization must be a JSON object.")
    texture_profile = randomization_profile.get("texture") or {}
    camera_profile = randomization_profile.get("camera") or {}
    lighting_randomization_profile = randomization_profile.get("lighting") or {}
    for profile_name, profile in (
        ("texture", texture_profile),
        ("camera", camera_profile),
        ("lighting", lighting_randomization_profile),
    ):
        if not isinstance(profile, dict):
            raise ValueError(f"render_profile.randomization.{profile_name} must be a JSON object.")

    texture_profile_enabled = bool(texture_profile.get("enabled", True))
    camera_profile_enabled = bool(camera_profile.get("enabled", True))
    lighting_profile_enabled = bool(lighting_randomization_profile.get("enabled", True))
    texture_mode = str(texture_profile.get("mode") or "full_material_replacement")
    texture_surfaces = tuple(str(name) for name in (texture_profile.get("surfaces") or SCENE_TEXTURE_SURFACES))
    unknown_surfaces = sorted(set(texture_surfaces) - set(SCENE_TEXTURE_SURFACES))
    if unknown_surfaces:
        raise ValueError(f"Unknown render_profile texture surfaces: {unknown_surfaces}.")
    if not texture_surfaces:
        raise ValueError("render_profile texture surfaces cannot be empty when texture randomization is configured.")
    surface_material_candidates, surface_candidate_metadata = _configured_surface_material_candidates(
        texture_profile
    )
    surface_constraints = _configured_surface_constraints(render_profile)

    variant_seed = stable_seed(global_seed, task_id, attempt_id, task_name, variant_id)
    seeds = {
        "texture_seed": stable_seed(variant_seed, "texture"),
        "table_material_seed": stable_seed(variant_seed, "table_material"),
        "floor_material_seed": stable_seed(variant_seed, "floor_material"),
        "wall_material_seed": stable_seed(variant_seed, "wall_material"),
        "camera_seed": stable_seed(variant_seed, "camera"),
        "lighting_seed": stable_seed(variant_seed, "lighting"),
        "lighting_intensity_seed": stable_seed(variant_seed, "lighting_intensity"),
    }
    texture_geom_names_by_surface = _scene_texture_geom_names(env)
    texture_geom_names = sorted(
        dict.fromkeys(
            geom_name
            for surface_name in SCENE_TEXTURE_SURFACES
            for geom_name in texture_geom_names_by_surface[surface_name]
        )
    )
    light_names = [name for name in env.sim.model.light_names if name and name.startswith(ARENA_PREFIX)]
    fixed_lighting = _apply_fixed_lighting_profile(
        env,
        light_names,
        render_profile.get("lighting") or {},
    )
    camera_randomization = _camera_randomization_by_name(camera_config)
    camera_entries_by_name = {
        str(camera["name"]): camera
        for camera in camera_config.get("cameras") or []
    }
    randomizable_camera_names = [
        camera_name
        for camera_name in camera_names
        if camera_profile_enabled and bool((camera_randomization.get(camera_name) or {}).get("enabled"))
    ]
    texture_applied = bool(enabled and texture_profile_enabled)
    camera_applied = bool(enabled and camera_profile_enabled and randomizable_camera_names)
    lighting_applied = bool(enabled and lighting_profile_enabled)
    randomization: dict[str, Any] = {
        "enabled": bool(enabled),
        "render_profile_id": render_profile.get("profile_id"),
        "environment_backend": "robosuite.environments.base.MujocoEnv",
        "randomizer_backend": "axis_mode_dispatch_with_robosuite_mjmod",
        "randomizer_classes": ["TextureModder", "CameraModder", "LightingModder"],
        "custom_camera_sampler_modes": [
            WRIST_CAMERA_RANDOMIZATION_MODE,
            FRONT_CAMERA_RANDOMIZATION_MODE,
        ],
        "variant_seed": int(variant_seed),
        **{key: int(value) for key, value in seeds.items()},
        "texture": {
            "modder_class": "robosuite.utils.mjmod.TextureModder",
            "profile_enabled": texture_profile_enabled,
            "applied": texture_applied,
            "mode": texture_mode,
            "scope": "configured_scene_surfaces_only",
            "configured_surfaces": list(texture_surfaces),
            "applied_surfaces": list(texture_surfaces) if texture_applied else [],
            "geom_names": texture_geom_names,
            "surface_geom_names": texture_geom_names_by_surface,
            "randomize_skybox": False,
            "randomize_local": texture_mode == "local_material_jitter",
            "randomize_material": True,
            "texture_variations": sorted(
                {
                    str(family["texture_variation"])
                    for candidates in surface_material_candidates.values()
                    for _, family in candidates
                }
            ),
            "material_family_sampling": "seeded_random_choice_from_scene_curated_bank_per_surface",
            "external_texture_asset_dirs": {
                "prepared_256": str(TABLE_TEXTURE_ASSET_DIR),
                "curated_tabletop_1k": str(CURATED_TABLE_TEXTURE_ASSET_DIR),
            },
            "external_texture_manifests": {
                "prepared_256": str(TABLE_TEXTURE_MANIFEST_PATH),
                "curated_tabletop_1k": str(CURATED_TABLE_TEXTURE_MANIFEST_PATH),
            },
            "table_material_bank_size": len(CURATED_TABLE_MATERIAL_FAMILIES),
            "surface_material_bank_sizes": {
                surface_name: len(candidates)
                for surface_name, candidates in surface_material_candidates.items()
            },
            "surface_material_filters": surface_candidate_metadata,
            "table_material_bank_min_required": MIN_CURATED_TABLE_TEXTURE_BANK_SIZE,
        },
        "surface_constraints": {
            "configured": deepcopy(surface_constraints),
            "image_luminance_guard_scope": "uploaded_randomized_image_textures",
            "image_luminance_guard_active": bool(
                texture_applied and texture_mode == "full_material_replacement"
                and any(
                    surface_name in texture_surfaces
                    and "image_luminance_guard" in constraints
                    for surface_name, constraints in surface_constraints.items()
                )
            ),
            "material_caps_scope": "fixed_render_profile_all_runs",
        },
        "cameras": {
            "backend": "axis_semantic_sampler_with_legacy_camera_modder_dispatch",
            "legacy_modder_class": "robosuite.utils.mjmod.CameraModder",
            "profile_enabled": camera_profile_enabled,
            "applied": camera_applied,
            "camera_names": list(camera_names),
            "randomizable_camera_names": randomizable_camera_names,
            "applied_camera_names": randomizable_camera_names if camera_applied else [],
            "per_camera": camera_randomization,
            "pair_seeds": {},
            "samples": {},
        },
        "lighting": {
            "modder_class": "robosuite.utils.mjmod.LightingModder",
            "profile_randomization_enabled": lighting_profile_enabled,
            "randomization_applied": lighting_applied,
            "fixed_profile": fixed_lighting,
            "light_names": light_names,
            "randomize_active": False,
            "randomize_position": lighting_profile_enabled,
            "randomize_direction": lighting_profile_enabled,
            "randomize_rgb_color_channels": False,
            "randomize_neutral_intensity": lighting_profile_enabled,
            "color_policy": "neutral_white_equal_rgb_channels",
        },
    }
    if texture_applied:
        surface_material_samples: dict[str, Any] = {}
        seed_name_by_surface = {
            "table": "table_material_seed",
            "floor": "floor_material_seed",
            "wall": "wall_material_seed",
        }
        if texture_mode == "full_material_replacement":
            _require_robosuite_texture_modder_runtime()
        elif texture_mode != "local_material_jitter":
            raise ValueError(f"Unsupported render_profile texture mode {texture_mode!r}.")
        for surface_name in texture_surfaces:
            random_state = np.random.RandomState(seeds[seed_name_by_surface[surface_name]])
            if texture_mode == "full_material_replacement":
                sample = _sample_and_apply_surface_material_family(
                    env,
                    texture_geom_names_by_surface[surface_name],
                    surface_name=surface_name,
                    candidates=surface_material_candidates[surface_name],
                    random_state=random_state,
                    surface_constraints=surface_constraints.get(surface_name) or {},
                )
            else:
                sample = _sample_and_apply_local_material_jitter(
                    env,
                    texture_geom_names_by_surface[surface_name],
                    surface_name=surface_name,
                    random_state=random_state,
                    rgb_jitter=float(texture_profile.get("rgb_jitter", 0.04)),
                    material_jitter=float(texture_profile.get("material_jitter", 0.03)),
                )
            surface_material_samples[surface_name] = sample
        randomization["texture"]["surface_material_samples"] = surface_material_samples
        for surface_name, sample in surface_material_samples.items():
            randomization["texture"][f"{surface_name}_material_sample"] = sample
    randomization["surface_constraints"]["material_cap_results"] = _apply_surface_material_caps(
        env,
        texture_geom_names_by_surface,
        surface_constraints,
    )
    if camera_applied:
        for camera_name in randomizable_camera_names:
            camera_randomizer = camera_randomization[camera_name]
            capture_replica = deepcopy(
                (camera_entries_by_name.get(camera_name) or {}).get("capture_replica") or {}
            )
            camera_seed, pair_seed, logical_camera_name = _camera_randomization_seed(
                seeds["camera_seed"],
                camera_name,
                camera_entries_by_name.get(camera_name) or {},
            )
            if pair_seed is not None:
                replica_id = int(capture_replica["camera_replica_id"])
                randomization["cameras"]["pair_seeds"].setdefault(
                    str(replica_id),
                    int(pair_seed),
                )
            camera_id = int(env.sim.model.camera_name2id(camera_name))
            mode = str(camera_randomizer.get("mode") or "")
            if mode == WRIST_CAMERA_RANDOMIZATION_MODE:
                sample = _sample_wrist_optical_pose_box(
                    base_position=np.asarray(env.sim.model.cam_pos[camera_id], dtype=np.float64),
                    base_quat_wxyz=np.asarray(env.sim.model.cam_quat[camera_id], dtype=np.float64),
                    base_fovy_deg=float(env.sim.model.cam_fovy[camera_id]),
                    profile=camera_randomizer,
                    camera_seed=camera_seed,
                )
                env.sim.model.cam_pos[camera_id] = np.asarray(
                    sample["resolved"]["position"], dtype=np.float64
                )
                env.sim.model.cam_quat[camera_id] = np.asarray(
                    sample["resolved"]["quat_wxyz"], dtype=np.float64
                )
                env.sim.model.cam_fovy[camera_id] = float(sample["resolved"]["fovy_deg"])
            elif mode == FRONT_CAMERA_RANDOMIZATION_MODE:
                configured_camera = next(
                    (
                        entry
                        for entry in env.xml_metadata.get("configured_cameras") or []
                        if str(entry.get("name") or "") == camera_name
                    ),
                    None,
                )
                if configured_camera is None:
                    raise ValueError(f"No prepared XML metadata found for camera {camera_name!r}.")
                reference = configured_camera.get("resolved_reference_frame") or {}
                sample = _sample_front_base_table_sector_lookat(
                    profile=camera_randomizer,
                    camera_seed=camera_seed,
                    base_world_position=np.asarray(reference.get("reference_body_world_pos") or [], dtype=np.float64),
                    base_world_quat_wxyz=np.asarray(
                        reference.get("reference_body_world_quat_wxyz") or [], dtype=np.float64
                    ),
                    table_center_xy_world=env.xml_metadata.get("table_center_xy") or [],
                    table_top_z_world=float(env.xml_metadata.get("table_top_z")),
                    table_full_size=env.xml_metadata.get("table_full_size") or [],
                )
                env.sim.model.cam_pos[camera_id] = np.asarray(
                    sample["resolved"]["world_position"], dtype=np.float64
                )
                env.sim.model.cam_quat[camera_id] = np.asarray(
                    sample["resolved"]["world_quat_wxyz"], dtype=np.float64
                )
                env.sim.model.cam_fovy[camera_id] = float(sample["resolved"]["fovy_deg"])
            else:
                CameraModder(
                    env.sim,
                    random_state=np.random.RandomState(camera_seed),
                    camera_names=[camera_name],
                    randomize_position=bool(camera_randomizer.get("randomize_position")),
                    randomize_rotation=bool(camera_randomizer.get("randomize_rotation")),
                    randomize_fovy=bool(camera_randomizer.get("randomize_fovy")),
                    position_perturbation_size=float(camera_randomizer.get("position_perturbation_size") or 0.0),
                    rotation_perturbation_size=float(camera_randomizer.get("rotation_perturbation_size") or 0.0),
                    fovy_perturbation_size=float(camera_randomizer.get("fovy_perturbation_size") or 0.0),
                ).randomize()
                sample = {
                    "mode": "legacy_robosuite_camera_modder",
                    "camera_seed": int(camera_seed),
                }
            if pair_seed is not None:
                sample["capture_seed_contract"] = (
                    "stable_seed(camera_seed,randomized_pair,replica_id)_then_logical_name"
                )
                sample["pair_seed"] = int(pair_seed)
                sample["camera_seed"] = int(camera_seed)
                sample["logical_camera_name"] = logical_camera_name
                sample["camera_group"] = "randomized"
                sample["camera_replica_id"] = int(capture_replica["camera_replica_id"])
            randomization["cameras"]["samples"][camera_name] = sample
    if lighting_applied and light_names:
        LightingModder(
            env.sim,
            random_state=np.random.RandomState(seeds["lighting_seed"]),
            light_names=light_names,
            randomize_position=True,
            randomize_direction=True,
            randomize_specular=False,
            randomize_ambient=False,
            randomize_diffuse=False,
            randomize_active=False,
            position_perturbation_size=0.12,
            direction_perturbation_size=0.22,
        ).randomize()
        _randomize_neutral_light_intensity(
            env,
            light_names,
            random_state=np.random.RandomState(seeds["lighting_intensity_seed"]),
            ambient_perturbation_size=0.06,
            diffuse_perturbation_size=0.08,
            specular_perturbation_size=0.08,
        )
    env.sim.forward()
    randomization["texture"]["geom_rgba"] = {
        name: np.asarray(env.sim.model.geom_rgba[int(env.sim.model.geom_name2id(name))], dtype=np.float64).tolist()
        for name in texture_geom_names
    }
    randomization["cameras"]["resolved"] = _camera_randomization_metadata(env, camera_names)
    randomization["lighting"]["resolved"] = _lighting_randomization_metadata(env, light_names)
    return randomization


def _build_scene_state_transform(model: mujoco.MjModel, xml_metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    offset = float(xml_metadata.get("free_joint_qpos_z_offset") or 0.0)
    qpos_offset = np.zeros(int(model.nq), dtype=np.float64)
    entries: list[dict[str, Any]] = []
    if abs(offset) > 1e-12:
        free_joint_type = int(mujoco.mjtJoint.mjJNT_FREE)
        for joint_id in range(model.njnt):
            if int(model.jnt_type[joint_id]) != free_joint_type:
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            body_id = int(model.jnt_bodyid[joint_id])
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            qpos_offset[qpos_adr + 2] += offset
            entries.append(
                {
                    "joint_name": joint_name,
                    "body_name": body_name,
                    "qpos_address": qpos_adr,
                    "z_qpos_index": qpos_adr + 2,
                    "z_offset": offset,
                }
            )
    metadata = {
        "policy": xml_metadata.get("scene_placement_policy"),
        "qpos_order_preserved": True,
        "qpos_z_offsets_applied": bool(entries),
        "qpos_z_offset_entries": entries,
        "qpos_z_offset_nonzero_count": len(entries),
        "states_qpos_are_render_transformed": bool(entries),
    }
    return qpos_offset, metadata


def _apply_qpos_offset(decoded: PackedStateTrajectory, qpos_offset: np.ndarray) -> PackedStateTrajectory:
    offset = np.asarray(qpos_offset, dtype=np.float64)
    if offset.shape != (decoded.nq,):
        raise ValueError(f"qpos offset shape {offset.shape} does not match decoded nq={decoded.nq}.")
    if not np.any(np.abs(offset) > 1e-12):
        return decoded
    initial_snapshot = dict(decoded.initial_snapshot)
    initial_snapshot["qpos"] = (np.asarray(initial_snapshot["qpos"], dtype=np.float64) + offset).tolist()
    return PackedStateTrajectory(
        task_id=decoded.task_id,
        timestep=decoded.timestep,
        steps_per_ctrl=decoded.steps_per_ctrl,
        step_counts=decoded.step_counts,
        n_steps=decoded.n_steps,
        nq=decoded.nq,
        nv=decoded.nv,
        nu=decoded.nu,
        initial_snapshot=initial_snapshot,
        qpos=decoded.qpos + offset.reshape(1, -1),
        qvel=decoded.qvel,
        ctrl=decoded.ctrl,
    )


def _validate_normalized_render_fps(fps: int) -> None:
    if int(fps) != NORMALIZED_RENDER_FPS:
        raise ValueError(
            "Robosuite render data uses one normalized 15 Hz timeline; "
            f"got lerobot_fps={fps!r}."
        )


def _validate_rgb_stack(camera_name: str, frames: np.ndarray) -> dict[str, float]:
    if frames.size == 0:
        raise ValueError(f"Camera {camera_name} rendered an empty RGB stack.")
    mean = float(np.mean(frames))
    std = float(np.std(frames))
    min_value = float(np.min(frames))
    max_value = float(np.max(frames))
    if std < 1.0 or max_value - min_value < 4.0:
        raise ValueError(
            f"Camera {camera_name} rendered a near-blank RGB stack "
            f"(mean={mean:.2f}, std={std:.2f}, min={min_value:.1f}, max={max_value:.1f})."
        )
    return {"mean": mean, "std": std, "min": min_value, "max": max_value}


@dataclass
class _StreamingRgbStats:
    count: int = 0
    value_sum: float = 0.0
    value_square_sum: float = 0.0
    min_value: int = 255
    max_value: int = 0

    def update(self, frame: np.ndarray) -> None:
        values = np.asarray(frame, dtype=np.uint8)
        if values.ndim != 3 or values.shape[2] != 3:
            raise ValueError(f"Streaming RGB frame must have shape [H,W,3], got {values.shape}.")
        self.count += int(values.size)
        self.value_sum += float(np.sum(values, dtype=np.float64))
        self.value_square_sum += float(
            np.sum(np.square(values, dtype=np.float64), dtype=np.float64)
        )
        self.min_value = min(self.min_value, int(values.min()))
        self.max_value = max(self.max_value, int(values.max()))

    def validate(self, camera_name: str) -> dict[str, float]:
        if self.count <= 0:
            raise ValueError(f"Camera {camera_name} rendered an empty RGB stream.")
        mean = self.value_sum / float(self.count)
        variance = max(self.value_square_sum / float(self.count) - mean * mean, 0.0)
        std = math.sqrt(variance)
        if std < 1.0 or self.max_value - self.min_value < 4:
            raise ValueError(
                f"Camera {camera_name} rendered a near-blank RGB stream "
                f"(mean={mean:.2f}, std={std:.2f}, min={self.min_value:.1f}, "
                f"max={self.max_value:.1f})."
            )
        return {
            "mean": float(mean),
            "std": float(std),
            "min": float(self.min_value),
            "max": float(self.max_value),
        }


def _safe_slug(text_value: str, *, max_len: int = 80) -> str:
    chars = []
    for char in str(text_value).lower():
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "_"}:
            chars.append("_")
    slug = "_".join("".join(chars).split("_"))
    return slug[:max_len].strip("_") or "task"


def _write_contact_sheet(frames: np.ndarray, path: Path) -> None:
    from PIL import Image

    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError("contact sheet requires a non-empty [T,H,W,3] RGB stack.")
    count = min(12, frames.shape[0])
    indexes = np.linspace(0, frames.shape[0] - 1, count).astype(int)
    thumbs = [Image.fromarray(frames[index]).resize((160, 160)) for index in indexes]
    columns = 4
    rows = int(math.ceil(count / columns))
    sheet = Image.new("RGB", (columns * 160, rows * 160), (255, 255, 255))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 160, (index // columns) * 160))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


@contextmanager
def _exclusive_materialization_lock(work_dir: Path) -> Iterator[None]:
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / ".robosuite_materialization.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@contextmanager
def _exclusive_camera_capture_batch_lock(metadata_root: Path) -> Iterator[None]:
    """Serialize one attempt/variant manifest from compatibility check through completion."""

    metadata_root = Path(metadata_root).expanduser().resolve()
    lock_dir = metadata_root.parent / ".axis_camera_capture_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(str(metadata_root).encode("utf-8")).hexdigest()[:24] + ".lock"
    with (lock_dir / lock_name).open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _materialized_asset_tree_fingerprint(
    root: Path,
    relative_paths: list[str],
) -> dict[str, Any]:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    normalized_paths = sorted(dict.fromkeys(str(path) for path in relative_paths))
    for relative in normalized_paths:
        path = (root / relative).resolve()
        try:
            normalized_relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Materialized asset escapes its root: {relative!r}.") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Declared materialized asset is missing: {path}.")
        file_digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                file_digest.update(chunk)
                size += len(chunk)
        digest.update(normalized_relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.digest())
        digest.update(b"\0")
        file_count += 1
        total_bytes += size
    return {
        "content_fingerprint_schema": "axis_materialized_asset_tree_v1",
        "content_sha256": digest.hexdigest(),
        "content_file_count": file_count,
        "content_total_bytes": total_bytes,
    }


def _materialize_bundle(
    bundle: dict[str, Any],
    *,
    work_dir: Path,
    asset_dir: str | None,
    asset_base_url: str,
    prepared_cache_dir: Path,
) -> dict[str, Any]:
    task_payload = bundle.get("task") or {}
    work_dir = Path(work_dir).expanduser().resolve()
    prepared_cache_dir = Path(prepared_cache_dir).expanduser().resolve()
    cache_identity = {
        "schema": "axis_materialized_asset_cache_v1",
        "mjcf_xml": str(task_payload["mjcf_xml"]),
        "asset_source": (
            {"type": "local", "path": str(Path(asset_dir).expanduser().resolve())}
            if asset_dir
            else {"type": "http", "base_url": str(asset_base_url)}
        ),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_entry = prepared_cache_dir / cache_key
    cache_manifest_path = cache_entry / "manifest.json"
    cache_hit = cache_manifest_path.is_file()

    def populate_cache() -> None:
        if cache_manifest_path.is_file():
            return
        temporary = prepared_cache_dir / f".{cache_key}.tmp.{os.getpid()}.{time.time_ns()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        asset_root = temporary / "assets"
        store = LocalAssetStore(asset_dir) if asset_dir else HttpAssetStore(asset_base_url)
        try:
            resolution = materialize_scene(task_payload["mjcf_xml"], str(asset_root), store)
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()
        if resolution.missing_files:
            shutil.rmtree(temporary, ignore_errors=True)
            raise FileNotFoundError(f"Missing assets: {resolution.missing_files}")
        declared_content_paths = [
            *resolution.xml_files.keys(),
            *resolution.binary_files.keys(),
        ]
        content_fingerprint = _materialized_asset_tree_fingerprint(
            asset_root,
            declared_content_paths,
        )
        entry_xml_path = Path(resolution.entry_xml_path)
        if not entry_xml_path.is_absolute():
            entry_xml_path = asset_root / entry_xml_path
        entry_xml_relative = str(entry_xml_path.resolve().relative_to(asset_root.resolve()))
        atomic_write_json(
            temporary / "manifest.json",
            {
                **cache_identity,
                "cache_key": cache_key,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "entry_xml_relative": entry_xml_relative,
                "missing_files": [],
                "xml_files": sorted(resolution.xml_files.keys()),
                "binary_file_count": len(resolution.binary_files),
                **content_fingerprint,
            },
        )
        prepared_cache_dir.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, cache_entry)
        _fsync_directory(prepared_cache_dir)

    lock_dir = prepared_cache_dir / ".locks" / cache_key
    with _exclusive_materialization_lock(lock_dir):
        populate_cache()
    manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != cache_identity["schema"] or manifest.get("cache_key") != cache_key:
        raise RuntimeError(f"Invalid materialized-asset cache entry: {cache_manifest_path}")

    refreshed_xml_files = [str(path) for path in sorted(work_dir.rglob("*.xml"))] if work_dir.exists() else []
    if work_dir.exists():
        shutil.rmtree(work_dir)

    def clone_cached_file(source: str, destination: str) -> str:
        if Path(source).suffix.lower() == ".xml":
            return shutil.copy2(source, destination)
        try:
            os.link(source, destination)
            return destination
        except OSError:
            return shutil.copy2(source, destination)

    shutil.copytree(cache_entry / "assets", work_dir, copy_function=clone_cached_file)
    return {
        "entry_xml": str(work_dir / manifest["entry_xml_relative"]),
        "missing_files": [],
        "xml_files": list(manifest["xml_files"]),
        "binary_file_count": int(manifest["binary_file_count"]),
        "refreshed_xml_file_count": len(refreshed_xml_files),
        "refreshed_xml_files": refreshed_xml_files,
        "prepared_asset_cache_schema": cache_identity["schema"],
        "prepared_asset_cache_key": cache_key,
        "prepared_asset_cache_path": str(cache_entry),
        "prepared_asset_cache_hit": cache_hit,
        "content_fingerprint_schema": manifest["content_fingerprint_schema"],
        "content_sha256": manifest["content_sha256"],
        "content_file_count": int(manifest["content_file_count"]),
        "content_total_bytes": int(manifest["content_total_bytes"]),
    }


def _task_lerobot_dataset_slug(task_id: int, task_name: str, embodiment_id: str) -> str:
    return f"task_{int(task_id)}_{_safe_slug(embodiment_id, max_len=32)}_{_safe_slug(task_name, max_len=80)}"


def _task_lerobot_repo_id(base_repo_id: str | None, task_id: int, task_name: str, embodiment_id: str) -> str:
    base = (base_repo_id or "local/axis_robosuite_render").strip()
    if not base:
        base = "local/axis_robosuite_render"
    if "/" not in base:
        base = f"local/{base}"
    namespace, name = base.split("/", 1)
    suffix = _task_lerobot_dataset_slug(task_id, task_name, embodiment_id)
    return f"{namespace}/{_safe_slug(name, max_len=80)}_{suffix}"


def _capture_lerobot_repo_id(
    base_repo_id: str | None,
    capture: CameraCapturePair,
    task_id: int,
    task_name: str,
    embodiment_id: str,
) -> str:
    if capture.camera_group is None:
        return _task_lerobot_repo_id(base_repo_id, task_id, task_name, embodiment_id)
    base = (base_repo_id or "local/axis_robosuite_render").strip() or "local/axis_robosuite_render"
    if "/" not in base:
        base = f"local/{base}"
    namespace, name = base.split("/", 1)
    grouped_base = f"{namespace}/{_safe_slug(name, max_len=68)}_camera_{capture.camera_group}"
    return _task_lerobot_repo_id(grouped_base, task_id, task_name, embodiment_id)


def _capture_lerobot_base_root(base_root: Path, capture: CameraCapturePair) -> Path:
    if capture.camera_group is None:
        return Path(base_root)
    return Path(base_root) / f"camera_{capture.camera_group}"


def _lerobot_task_dataset_root(base_root: Path, task_id: int, task_name: str, embodiment_id: str) -> Path:
    return base_root / _task_lerobot_dataset_slug(task_id, task_name, embodiment_id)


def _lerobot_camera_key(camera_name: str, camera_entry: dict[str, Any]) -> str:
    role = str(camera_entry.get("role") or "").strip().lower().replace("-", "_")
    if role in {"front", "third_person", "third_view", "third"}:
        key = "third_person"
    elif role in {"wrist", "hand", "first_person"}:
        key = "wrist"
    else:
        key = _safe_slug(camera_name, max_len=80)
    if key[0].isdigit():
        key = f"camera_{key}"
    return key


def _lerobot_camera_keys(camera_config: dict[str, Any]) -> dict[str, str]:
    keys: dict[str, str] = {}
    used: set[str] = set()
    for camera_entry in camera_config["cameras"]:
        camera_name = str(camera_entry["name"])
        base = _lerobot_camera_key(camera_name, camera_entry)
        key = base
        suffix = 2
        while key in used:
            key = f"{base}_{suffix}"
            suffix += 1
        used.add(key)
        keys[camera_name] = key
    return keys


def _build_task_lerobot_features(
    *,
    camera_keys: dict[str, str],
    width: int,
    height: int,
    state_dim: int,
    action_dim: int,
    root_state_dim: int | None,
    ee_state_dim: int | None,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (int(state_dim),),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (int(action_dim),),
            "names": ["action"],
        },
    }
    if root_state_dim is not None:
        features["observation.robot_root_state"] = {
            "dtype": "float32",
            "shape": (int(root_state_dim),),
            "names": ["root_state"],
        }
    if ee_state_dim is not None:
        features["observation.ee_state"] = {
            "dtype": "float32",
            "shape": (int(ee_state_dim),),
            "names": ["ee_state"],
        }
    for camera_key in camera_keys.values():
        features[f"observation.images.{camera_key}"] = {
            "dtype": "video",
            "shape": (int(height), int(width), 3),
            "names": ["height", "width", "channel"],
        }
        features[f"observation.cameras.{camera_key}.intrinsics"] = {
            "dtype": "float32",
            "shape": (9,),
            "names": ["row_major"],
        }
        features[f"observation.cameras.{camera_key}.camera_to_world"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": ["row_major"],
        }
    return features


def _name_for_mujoco_id(model: mujoco.MjModel, obj_type: int, obj_id: int) -> str:
    if obj_id < 0:
        return ""
    return mujoco.mj_id2name(model, obj_type, int(obj_id)) or ""


def _policy_joint_qpos_metadata(model: mujoco.MjModel) -> dict[str, Any]:
    joint_trn_type = int(mujoco.mjtTrn.mjTRN_JOINT)
    scalar_joint_types = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    qpos_indices: list[int] = []
    actuator_entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for actuator_id in range(int(model.nu)):
        actuator_name = _name_for_mujoco_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        trn_type = int(model.actuator_trntype[actuator_id])
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if trn_type != joint_trn_type or joint_id < 0:
            skipped.append(
                {
                    "actuator_id": actuator_id,
                    "actuator_name": actuator_name,
                    "reason": "non_joint_transmission",
                    "transmission_type": trn_type,
                    "target_id": joint_id,
                }
            )
            continue
        joint_type = int(model.jnt_type[joint_id])
        joint_name = _name_for_mujoco_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_type not in scalar_joint_types:
            skipped.append(
                {
                    "actuator_id": actuator_id,
                    "actuator_name": actuator_name,
                    "joint_id": joint_id,
                    "joint_name": joint_name,
                    "reason": "non_scalar_joint",
                    "joint_type": joint_type,
                }
            )
            continue
        qpos_index = int(model.jnt_qposadr[joint_id])
        qpos_indices.append(qpos_index)
        actuator_entries.append(
            {
                "actuator_id": actuator_id,
                "actuator_name": actuator_name,
                "joint_id": joint_id,
                "joint_name": joint_name,
                "joint_type": joint_type,
                "qpos_index": qpos_index,
            }
        )
    if not qpos_indices:
        raise ValueError("Could not derive policy-level joint state: no scalar joint actuators were found.")
    return {
        "state_semantics": "actuated_joint_qpos_in_actuator_order",
        "qpos_indices": qpos_indices,
        "actuator_joint_entries": actuator_entries,
        "skipped_actuators": skipped,
        "state_dim": len(qpos_indices),
        "action_dim": int(model.nu),
    }


def _policy_joint_state(decoded: PackedStateTrajectory, policy_metadata: dict[str, Any]) -> np.ndarray:
    qpos_indices = np.asarray(policy_metadata["qpos_indices"], dtype=np.int64)
    return np.asarray(decoded.qpos[:, qpos_indices], dtype=np.float32)


def _camera_intrinsics_matrix(env: AxisReplayRobosuiteEnv, camera_name: str, *, width: int, height: int) -> np.ndarray:
    camera_id = int(env.sim.model.camera_name2id(camera_name))
    fovy = float(env.sim.model.cam_fovy[camera_id])
    focal = 0.5 * float(height) / math.tan(math.radians(fovy) * 0.5)
    return np.asarray(
        [
            [focal, 0.0, (float(width) - 1.0) * 0.5],
            [0.0, focal, (float(height) - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _camera_to_world_matrix(env: AxisReplayRobosuiteEnv, camera_name: str) -> np.ndarray:
    camera_id = int(env.sim.model.camera_name2id(camera_name))
    data = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(data.cam_xmat[camera_id], dtype=np.float32).reshape(3, 3)
    transform[:3, 3] = np.asarray(data.cam_xpos[camera_id], dtype=np.float32)
    return transform


def _body_pose7(env: AxisReplayRobosuiteEnv, body_name: str) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    data = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    return np.concatenate(
        [
            np.asarray(data.xpos[body_id], dtype=np.float32),
            np.asarray(data.xquat[body_id], dtype=np.float32),
        ]
    ).astype(np.float32)


def _lerobot_ee_body_name(xml_metadata: dict[str, Any]) -> str | None:
    wrist_bodies = {
        str(camera["resolved_mount_body"])
        for camera in xml_metadata.get("configured_cameras") or []
        if str(camera.get("role") or "").lower() == "wrist"
        and camera.get("resolved_mount_body")
    }
    if len(wrist_bodies) > 1:
        raise ValueError(f"Configured wrist cameras disagree on end-effector body: {wrist_bodies!r}.")
    return next(iter(wrist_bodies), None)


def _build_lerobot_episode_schema(
    *,
    camera_config: dict[str, Any],
    state_dim: int,
    action_dim: int,
    ee_state_dim: int | None,
    width: int,
    height: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    camera_keys = _lerobot_camera_keys(camera_config)
    features = _build_task_lerobot_features(
        camera_keys=camera_keys,
        width=int(width),
        height=int(height),
        state_dim=int(state_dim),
        action_dim=int(action_dim),
        root_state_dim=None,
        ee_state_dim=ee_state_dim,
    )
    return features, camera_keys


def _build_lerobot_stream_frame(
    *,
    camera_keys: dict[str, str],
    rendered_frames: dict[str, np.ndarray],
    camera_matrices: dict[str, dict[str, np.ndarray]],
    state: np.ndarray,
    action: np.ndarray,
    ee_state: np.ndarray | None,
) -> dict[str, Any]:
    frame_payload: dict[str, Any] = {
        "observation.state": np.asarray(state, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
    }
    if ee_state is not None:
        frame_payload["observation.ee_state"] = np.asarray(ee_state, dtype=np.float32)
    for camera_name, camera_key in camera_keys.items():
        frame_payload[f"observation.images.{camera_key}"] = np.asarray(
            rendered_frames[camera_name], dtype=np.uint8
        )
        intrinsics = np.asarray(camera_matrices[camera_name]["intrinsics"], dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Camera {camera_name} intrinsics must have shape (3, 3), got {intrinsics.shape}."
            )
        frame_payload[f"observation.cameras.{camera_key}.intrinsics"] = intrinsics.reshape(-1)
        camera_to_world = np.asarray(
            camera_matrices[camera_name]["camera_to_world"], dtype=np.float32
        )
        if camera_to_world.shape != (4, 4):
            raise ValueError(
                f"Camera {camera_name} camera_to_world must have shape (4, 4), "
                f"got {camera_to_world.shape}."
            )
        frame_payload[f"observation.cameras.{camera_key}.camera_to_world"] = (
            camera_to_world.reshape(-1)
        )
    return frame_payload


def _build_lerobot_episode_payload(
    *,
    frames_by_camera: dict[str, np.ndarray],
    camera_config: dict[str, Any],
    camera_matrices: dict[str, dict[str, np.ndarray]],
    decoded: PackedStateTrajectory,
    policy_state: np.ndarray,
    ee_state: np.ndarray | None,
    frame_indexes: list[int],
    width: int,
    height: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], int, int, int | None]:
    expected_count = len(frame_indexes)
    if expected_count == 0:
        raise ValueError("LeRobot episode payload requires at least one resampled state.")
    if frame_indexes != list(range(decoded.n_steps)):
        raise ValueError(
            "LeRobot episode payload expects sequential indices over the resampled state trajectory."
        )
    if policy_state.shape[0] != decoded.n_steps:
        raise ValueError(
            f"policy_state has {policy_state.shape[0]} rows, expected {decoded.n_steps}."
        )
    for camera_name in _camera_names_from_config(camera_config):
        if camera_name not in frames_by_camera or frames_by_camera[camera_name].shape[0] != expected_count:
            raise ValueError(f"Camera {camera_name} RGB rows do not align with resampled states.")
        matrices = camera_matrices.get(camera_name) or {}
        for matrix_name in ("intrinsics", "camera_to_world"):
            matrix_values = matrices.get(matrix_name)
            if matrix_values is None or matrix_values.shape[0] != expected_count:
                raise ValueError(
                    f"Camera {camera_name} {matrix_name} rows do not align with resampled states."
                )
    if ee_state is not None and ee_state.shape[0] != expected_count:
        raise ValueError("EE state rows do not align with resampled states.")

    camera_keys = _lerobot_camera_keys(camera_config)
    action = np.asarray(decoded.ctrl[frame_indexes], dtype=np.float32)
    state = np.asarray(policy_state[frame_indexes], dtype=np.float32)
    ee_state_selected = np.asarray(ee_state, dtype=np.float32) if ee_state is not None else None
    features = _build_task_lerobot_features(
        camera_keys=camera_keys,
        width=int(width),
        height=int(height),
        state_dim=int(state.shape[1]),
        action_dim=int(action.shape[1]),
        root_state_dim=None,
        ee_state_dim=int(ee_state_selected.shape[1]) if ee_state_selected is not None else None,
    )
    frames: list[dict[str, Any]] = []
    for local_index, _source_index in enumerate(frame_indexes):
        frame_payload: dict[str, Any] = {
            "observation.state": state[local_index],
            "action": action[local_index],
        }
        if ee_state_selected is not None:
            frame_payload["observation.ee_state"] = ee_state_selected[local_index]
        for camera_name, camera_key in camera_keys.items():
            frame_payload[f"observation.images.{camera_key}"] = frames_by_camera[camera_name][local_index]
            intrinsics = np.asarray(camera_matrices[camera_name]["intrinsics"][local_index], dtype=np.float32)
            if intrinsics.shape != (3, 3):
                raise ValueError(f"Camera {camera_name} intrinsics must have shape (3, 3), got {intrinsics.shape}.")
            frame_payload[f"observation.cameras.{camera_key}.intrinsics"] = intrinsics.reshape(-1)
            camera_to_world = np.asarray(
                camera_matrices[camera_name]["camera_to_world"][local_index], dtype=np.float32
            )
            if camera_to_world.shape != (4, 4):
                raise ValueError(
                    f"Camera {camera_name} camera_to_world must have shape (4, 4), got {camera_to_world.shape}."
                )
            frame_payload[f"observation.cameras.{camera_key}.camera_to_world"] = camera_to_world.reshape(-1)
        frames.append(frame_payload)

    return (
        features,
        frames,
        camera_keys,
        int(state.shape[1]),
        int(action.shape[1]),
        int(ee_state_selected.shape[1]) if ee_state_selected is not None else None,
    )


def _render_attempt_job_camera_multiplex(
    spec: RenderJobSpec,
    *,
    start_time: float,
    bundle: dict[str, Any],
    task_id: int,
    task_name: str,
    embodiment_id: str,
    logical_camera_config: dict[str, Any],
    output_dir: Path,
    cache_dir: Path,
    asset_base_url: str,
) -> RenderRecord:
    """Render all multiplexed cameras from one state replay and commit paired episodes."""

    runtime_camera_config, captures = _expand_camera_capture_plan(
        logical_camera_config,
        randomization_enabled=bool(spec.randomization_enabled),
    )
    if spec.save_preview_video:
        raise ValueError(
            "Multiplexed streaming render already writes canonical H.264 camera videos; "
            "--save-preview-video is not supported because it would duplicate every stream."
        )
    _validate_camera_render_aspect_ratio(
        runtime_camera_config,
        width=int(spec.width),
        height=int(spec.height),
    )
    camera_names = _camera_names_from_config(runtime_camera_config)
    trajectory_data = (bundle.get("task_trajectory") or {}).get("trajectory_data") or {}
    if trajectory_data.get("format") != "packed_state_v1":
        raise ValueError(f"Expected packed_state_v1 trajectory, got {trajectory_data.get('format')!r}.")
    source_fingerprint = _render_source_fingerprint(bundle)

    lerobot_base_root = (
        Path(spec.lerobot_root).expanduser().resolve()
        if spec.lerobot_root
        else output_dir / "lerobot_tasks"
    )
    task_dataset_slug = _task_lerobot_dataset_slug(task_id, task_name, embodiment_id)
    variant_slug = (
        f"task_{task_id}_attempt_{spec.attempt_id}_{embodiment_id}_"
        f"variant_{spec.variant_id:03d}"
    )
    metadata_root = output_dir / "episode_metadata" / task_dataset_slug / variant_slug
    capture_manifest_path = metadata_root / "capture_manifest.json"
    targets: list[dict[str, Any]] = []
    for capture in captures:
        pair_camera_config = _camera_config_for_capture_pair(runtime_camera_config, capture)
        capture_base_root = _capture_lerobot_base_root(lerobot_base_root, capture)
        dataset_root = _lerobot_task_dataset_root(
            capture_base_root,
            task_id,
            task_name,
            embodiment_id,
        )
        repo_id = _capture_lerobot_repo_id(
            spec.lerobot_repo_id,
            capture,
            task_id,
            task_name,
            embodiment_id,
        )
        render_fingerprint = _render_fingerprint(
            pair_camera_config,
            spec,
            lerobot_repo_id=repo_id,
            source_fingerprint=source_fingerprint,
        )
        if capture.camera_group is None:
            capture_metadata_root = metadata_root
        else:
            capture_metadata_root = (
                metadata_root
                / f"camera_{capture.camera_group}"
                / f"replica_{int(capture.camera_replica_id):03d}"
            )
        targets.append(
            {
                "capture": capture,
                "camera_config": pair_camera_config,
                "dataset_root": dataset_root,
                "repo_id": repo_id,
                "render_fingerprint": render_fingerprint,
                "metadata_root": capture_metadata_root,
                "metadata_path": capture_metadata_root / "metadata.json",
            }
        )

    group_destinations: dict[str, tuple[Path, str]] = {}
    for target in targets:
        capture = target["capture"]
        if capture.camera_group is None:
            continue
        group_destinations.setdefault(
            capture.camera_group,
            (target["dataset_root"], target["repo_id"]),
        )
    if len({str(value[0]) for value in group_destinations.values()}) != len(group_destinations):
        raise ValueError(f"Camera capture groups must use distinct dataset roots: {group_destinations!r}.")
    if len({value[1] for value in group_destinations.values()}) != len(group_destinations):
        raise ValueError(f"Camera capture groups must use distinct repo ids: {group_destinations!r}.")

    canonical_runtime_config = {
        key: value
        for key, value in runtime_camera_config.items()
        if key != "config_path"
    }
    batch_fingerprint_payload = {
        "schema": CAMERA_CAPTURE_MANIFEST_SCHEMA,
        "source_fingerprint": source_fingerprint,
        "runtime_camera_config": canonical_runtime_config,
        "captures": [
            {
                "capture": asdict(target["capture"]),
                "repo_id": target["repo_id"],
                "render_fingerprint": target["render_fingerprint"],
            }
            for target in targets
        ],
    }
    capture_batch_fingerprint = hashlib.sha256(
        json.dumps(
            batch_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    source_trajectory_group_id = (
        f"task_{task_id}_attempt_{int(spec.attempt_id)}_variant_{int(spec.variant_id):03d}"
    )
    existing_manifest: dict[str, Any] | None = None
    existing_fingerprint_matches = False
    if capture_manifest_path.is_file():
        existing_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing_manifest, dict):
            raise ValueError(f"Capture manifest must contain a JSON object: {capture_manifest_path}.")
    current_roots_exist = any(Path(target["dataset_root"]).exists() for target in targets)
    prior_roots_exist = False
    if existing_manifest is not None:
        prior_entries_key = (
            "captures" if bool(existing_manifest.get("complete")) else "expected_captures"
        )
        prior_entries = existing_manifest.get(prior_entries_key) or []
        prior_by_id = {
            str(entry.get("capture_id") or ""): entry
            for entry in prior_entries
        }
        prior_roots_exist = any(
            bool(entry.get("dataset_path"))
            and Path(str(entry["dataset_path"])).expanduser().exists()
            for entry in prior_entries
        )
        for target in targets:
            previous = prior_by_id.get(target["capture"].capture_id)
            if previous is None or not previous.get("dataset_path"):
                continue
            previous_root = Path(str(previous["dataset_path"])).expanduser().resolve()
            current_root = Path(target["dataset_root"]).expanduser().resolve()
            if previous_root != current_root and previous_root.exists():
                raise ValueError(
                    "Changing --lerobot-root while a prior capture dataset still exists would "
                    f"orphan manifest provenance for {target['capture'].capture_id}: "
                    f"stored={previous_root}, current={current_root}. Use a new output directory."
                )
    target_roots_exist = bool(current_roots_exist or prior_roots_exist)
    if not spec.resume:
        if target_roots_exist:
            raise ValueError(
                "--no-resume cannot reuse an existing multiplexed camera dataset. "
                "Use --resume, choose a new output/LeRobot root, or pass --lerobot-overwrite."
            )
        existing_manifest = None
    if existing_manifest is not None:
        existing_schema = existing_manifest.get("schema")
        existing_fingerprint = existing_manifest.get("capture_batch_fingerprint")
        if existing_schema != CAMERA_CAPTURE_MANIFEST_SCHEMA:
            if target_roots_exist:
                raise ValueError(
                    f"Existing capture manifest has unsupported schema {existing_schema!r}: "
                    f"{capture_manifest_path}. Use a new output root or overwrite both datasets."
                )
            existing_manifest = None
        else:
            existing_fingerprint_matches = (
                existing_fingerprint == capture_batch_fingerprint
            )
            if not existing_fingerprint_matches and not target_roots_exist:
                existing_manifest = None

    if existing_manifest is not None and target_roots_exist:
        current_capture_ids = {target["capture"].capture_id for target in targets}
        _validate_append_only_capture_manifest(existing_manifest, current_capture_ids)

    if (
        existing_manifest is not None
        and bool(existing_manifest.get("complete"))
        and not existing_fingerprint_matches
    ):
        expected_identity = {
            "task_id": int(task_id),
            "task_name": task_name,
            "attempt_id": int(spec.attempt_id),
            "embodiment_id": embodiment_id,
            "variant_id": int(spec.variant_id),
            "source_trajectory_group_id": source_trajectory_group_id,
            "source_fingerprint": source_fingerprint,
        }
        observed_identity = {key: existing_manifest.get(key) for key in expected_identity}
        if observed_identity != expected_identity:
            raise ValueError(
                "Completed capture manifest identity mismatch: "
                f"expected={expected_identity!r}, observed={observed_identity!r}."
            )
        manifest_model_id = str(existing_manifest.get("model_id") or "")
        if not manifest_model_id:
            raise ValueError(f"Completed capture manifest is missing model_id: {capture_manifest_path}.")
        old_capture_ids = {
            str(capture.get("capture_id") or "")
            for capture in existing_manifest.get("captures") or []
        }
        for target in targets:
            capture = target["capture"]
            if capture.capture_id not in old_capture_ids:
                continue
            preflight_identity = EpisodeIdentity(
                task_id=task_id,
                task_name=task_name,
                attempt_id=int(spec.attempt_id),
                embodiment_id=embodiment_id,
                model_id=manifest_model_id,
                variant_id=int(spec.variant_id),
                render_fingerprint=target["render_fingerprint"],
                camera_group=capture.camera_group,
                camera_replica_id=capture.camera_replica_id,
            )
            load_committed_attempt(target["dataset_root"], preflight_identity)
        # The existing per-episode fingerprints are compatible. A changed batch
        # fingerprint therefore represents an append-only replica extension (or
        # other manifest-only change), and may proceed through normal recovery.
        existing_manifest = None

    if existing_manifest is not None and bool(existing_manifest.get("complete")):
        expected_manifest_identity = {
            "task_id": int(task_id),
            "task_name": task_name,
            "attempt_id": int(spec.attempt_id),
            "embodiment_id": embodiment_id,
            "variant_id": int(spec.variant_id),
            "source_trajectory_group_id": source_trajectory_group_id,
            "source_fingerprint": source_fingerprint,
        }
        observed_manifest_identity = {
            key: existing_manifest.get(key) for key in expected_manifest_identity
        }
        if observed_manifest_identity != expected_manifest_identity:
            raise ValueError(
                "Completed capture manifest identity mismatch: "
                f"expected={expected_manifest_identity!r}, observed={observed_manifest_identity!r}."
            )
        manifest_model_id = str(existing_manifest.get("model_id") or "")
        if not manifest_model_id:
            raise ValueError(f"Completed capture manifest is missing model_id: {capture_manifest_path}.")
        fast_committed: list[dict[str, Any]] = []
        for target in targets:
            capture = target["capture"]
            identity = EpisodeIdentity(
                task_id=task_id,
                task_name=task_name,
                attempt_id=int(spec.attempt_id),
                embodiment_id=embodiment_id,
                model_id=manifest_model_id,
                variant_id=int(spec.variant_id),
                render_fingerprint=target["render_fingerprint"],
                camera_group=capture.camera_group,
                camera_replica_id=capture.camera_replica_id,
            )
            committed = load_committed_attempt(target["dataset_root"], identity)
            if committed is None:
                fast_committed = []
                break
            target["identity"] = identity
            target["committed"] = committed
            fast_committed.append(committed)
        if len(fast_committed) == len(targets):
            fast_episodes: list[dict[str, Any]] = []
            fast_outputs: list[dict[str, Any]] = []
            frame_counts: set[int] = set()
            for target, committed in zip(targets, fast_committed):
                capture = target["capture"]
                episode = committed["episode"]
                metadata = deepcopy(committed["metadata"])
                metadata["lerobot_episode"] = episode
                atomic_write_json(target["metadata_path"], metadata)
                frame_counts.add(int(episode["frame_count"]))
                fast_episodes.append(
                    {
                        **deepcopy(episode),
                        "capture_id": capture.capture_id,
                        "camera_group": capture.camera_group,
                        "camera_replica_id": capture.camera_replica_id,
                        "source_trajectory_group_id": source_trajectory_group_id,
                        "dataset_path": str(target["dataset_root"]),
                        "metadata_path": str(target["metadata_path"]),
                        "logical_camera_names": list(capture.logical_camera_names),
                        "physical_camera_names": list(capture.physical_camera_names),
                    }
                )
                fast_outputs.append(
                    {
                        "capture_id": capture.capture_id,
                        "camera_group": capture.camera_group,
                        "camera_replica_id": capture.camera_replica_id,
                        "source_trajectory_group_id": source_trajectory_group_id,
                        "dataset_path": str(target["dataset_root"]),
                        "repo_id": target["repo_id"],
                        "metadata_path": str(target["metadata_path"]),
                        "render_fingerprint": target["render_fingerprint"],
                        "logical_camera_names": list(capture.logical_camera_names),
                        "physical_camera_names": list(capture.physical_camera_names),
                        "episode_index": int(episode["episode_index"]),
                        "frame_count": int(episode["frame_count"]),
                        "commit_status": "already_committed",
                    }
                )
            if len(frame_counts) != 1:
                raise ValueError(f"Completed capture episodes disagree on frame count: {frame_counts!r}.")
            frame_count = next(iter(frame_counts))
            return RenderRecord(
                status="skipped",
                task_id=task_id,
                task_name=task_name,
                attempt_id=int(spec.attempt_id),
                embodiment_id=embodiment_id,
                model_id=manifest_model_id,
                variant_id=int(spec.variant_id),
                seed=int(spec.seed),
                num_frames=frame_count,
                camera_names=list(camera_names),
                render_time_s=time.perf_counter() - start_time,
                output_path=str(targets[0]["dataset_root"]),
                metadata_path=str(capture_manifest_path),
                lerobot_episode=fast_episodes[0],
                lerobot_episodes=fast_episodes,
                camera_dataset_outputs=fast_outputs,
                capture_manifest_path=str(capture_manifest_path),
                num_lerobot_episodes=len(fast_episodes),
                num_state_frames_replayed=0,
                num_camera_frames_rendered=0,
                num_lerobot_frames_written=0,
                preview_videos=deepcopy(existing_manifest.get("preview_videos") or {}),
                contact_sheet=existing_manifest.get("contact_sheet"),
                randomization_config=deepcopy(existing_manifest.get("randomization_config")),
            )

    atomic_write_json(
        capture_manifest_path,
        {
            "schema": CAMERA_CAPTURE_MANIFEST_SCHEMA,
            "complete": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "capture_batch_fingerprint": capture_batch_fingerprint,
            "source_trajectory_group_id": source_trajectory_group_id,
            "source_fingerprint": source_fingerprint,
            "reason": "render_or_resume_in_progress",
            "expected_captures": [
                {
                    "capture_id": target["capture"].capture_id,
                    "camera_group": target["capture"].camera_group,
                    "camera_replica_id": target["capture"].camera_replica_id,
                    "dataset_path": str(target["dataset_root"]),
                    "repo_id": target["repo_id"],
                }
                for target in targets
            ],
        },
    )

    env: AxisReplayRobosuiteEnv | None = None
    try:
        work_dir = cache_dir / "tasks" / f"task_{task_id}_attempt_{spec.attempt_id}"
        with _exclusive_materialization_lock(work_dir):
            asset_resolution = _materialize_bundle(
                bundle,
                work_dir=work_dir / "assets",
                asset_dir=spec.asset_dir,
                asset_base_url=asset_base_url,
                prepared_cache_dir=cache_dir / "prepared_assets",
            )
            entry_xml_path = Path(asset_resolution["entry_xml"])
            if not entry_xml_path.is_absolute():
                entry_xml_path = (work_dir / "assets" / entry_xml_path).resolve()
            env = AxisReplayRobosuiteEnv(
                entry_xml=entry_xml_path,
                camera_names=camera_names,
                camera_config=runtime_camera_config,
                layout_cache_dir=cache_dir / "prepared_layouts",
                has_renderer=False,
                has_offscreen_renderer=True,
                render_collision_mesh=False,
                render_visual_mesh=True,
                render_gpu_device_id=int(spec.gpu_id) if spec.gpu_id is not None else -1,
                control_freq=20,
                lite_physics=True,
                horizon=1000,
                ignore_done=True,
                hard_reset=False,
                renderer="mujoco",
                seed=int(spec.seed),
            )

        model = env.sim.model._model
        data = env.sim.data._data
        source_decoded = decode_packed_state_v1(trajectory_data)
        _validate_dimensions(model, source_decoded)
        qpos_offset, scene_state_transform = _build_scene_state_transform(model, env.xml_metadata)
        decoded = _apply_qpos_offset(source_decoded, qpos_offset)
        expected_duration_s = (bundle.get("attempt") or {}).get("simulation_time_seconds")
        state_times, time_summary = _build_state_times(
            decoded,
            expected_duration_s=expected_duration_s,
        )
        resampled_timeline = resample_packed_state_to_fps(
            model,
            decoded,
            state_times,
            time_summary,
            fps=int(spec.lerobot_fps),
            expected_duration_s=expected_duration_s,
            frame_stride=spec.frame_stride,
            max_frames=spec.max_frames,
        )
        decoded = resampled_timeline.trajectory
        state_times = resampled_timeline.state_times.tolist()
        frame_schedule = resampled_timeline.metadata
        time_summary = frame_schedule["source_timing"]
        frame_indexes = list(range(decoded.n_steps))
        model_id = f"{embodiment_id}:task_{task_id}:nq{model.nq}:nv{model.nv}:nu{model.nu}"
        for target in targets:
            capture = target["capture"]
            target["identity"] = EpisodeIdentity(
                task_id=task_id,
                task_name=task_name,
                attempt_id=int(spec.attempt_id),
                embodiment_id=embodiment_id,
                model_id=model_id,
                variant_id=int(spec.variant_id),
                render_fingerprint=target["render_fingerprint"],
                camera_group=capture.camera_group,
                camera_replica_id=capture.camera_replica_id,
            )
            target["committed"] = load_committed_attempt(
                target["dataset_root"],
                target["identity"],
            )
        current_asset_fingerprint = str(asset_resolution["content_sha256"])
        for target in targets:
            committed = target["committed"]
            if committed is None:
                continue
            committed_source = committed["metadata"].get("source") or {}
            stored_source_fingerprint = committed_source.get("source_fingerprint")
            stored_asset_fingerprint = (
                committed_source.get("materialized_assets") or {}
            ).get("content_sha256")
            if stored_source_fingerprint != source_fingerprint:
                raise ValueError(
                    f"Committed source fingerprint mismatch for {target['capture'].capture_id}: "
                    f"stored={stored_source_fingerprint!r}, current={source_fingerprint!r}."
                )
            if stored_asset_fingerprint != current_asset_fingerprint:
                raise ValueError(
                    f"Committed materialized asset fingerprint mismatch for "
                    f"{target['capture'].capture_id}: stored={stored_asset_fingerprint!r}, "
                    f"current={current_asset_fingerprint!r}."
                )

        def decorated_episode(target: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
            capture: CameraCapturePair = target["capture"]
            return {
                **deepcopy(episode),
                "capture_id": capture.capture_id,
                "camera_group": capture.camera_group,
                "camera_replica_id": capture.camera_replica_id,
                "source_trajectory_group_id": source_trajectory_group_id,
                "dataset_path": str(target["dataset_root"]),
                "metadata_path": str(target["metadata_path"]),
                "logical_camera_names": list(capture.logical_camera_names),
                "physical_camera_names": list(capture.physical_camera_names),
            }

        def capture_output(
            target: dict[str, Any],
            episode: dict[str, Any],
            *,
            commit_status: str,
        ) -> dict[str, Any]:
            capture: CameraCapturePair = target["capture"]
            return {
                "capture_id": capture.capture_id,
                "camera_group": capture.camera_group,
                "camera_replica_id": capture.camera_replica_id,
                "source_trajectory_group_id": source_trajectory_group_id,
                "dataset_path": str(target["dataset_root"]),
                "repo_id": target["repo_id"],
                "metadata_path": str(target["metadata_path"]),
                "render_fingerprint": target["render_fingerprint"],
                "logical_camera_names": list(capture.logical_camera_names),
                "physical_camera_names": list(capture.physical_camera_names),
                "episode_index": int(episode["episode_index"]),
                "frame_count": int(episode["frame_count"]),
                "commit_status": commit_status,
            }

        def write_capture_manifest(
            outputs: list[dict[str, Any]],
            *,
            randomization: dict[str, Any] | None,
            preview_videos: dict[str, str],
            contact_sheet: str | None,
        ) -> None:
            if len(outputs) != len(targets):
                raise RuntimeError(
                    f"Refusing to mark an incomplete capture batch complete: "
                    f"expected={len(targets)}, observed={len(outputs)}."
                )
            atomic_write_json(
                capture_manifest_path,
                {
                    "schema": CAMERA_CAPTURE_MANIFEST_SCHEMA,
                    "complete": True,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "capture_batch_fingerprint": capture_batch_fingerprint,
                    "source_trajectory_group_id": source_trajectory_group_id,
                    "source_fingerprint": source_fingerprint,
                    "materialized_asset_fingerprint": asset_resolution["content_sha256"],
                    "task_id": task_id,
                    "task_name": task_name,
                    "attempt_id": int(spec.attempt_id),
                    "embodiment_id": embodiment_id,
                    "model_id": model_id,
                    "variant_id": int(spec.variant_id),
                    "logical_camera_names": list(
                        runtime_camera_config.get("logical_camera_names") or []
                    ),
                    "physical_camera_names": list(camera_names),
                    "capture_count": len(targets),
                    "fixed_episode_count": sum(
                        output["camera_group"] == "fixed" for output in outputs
                    ),
                    "randomized_episode_count": sum(
                        output["camera_group"] == "randomized" for output in outputs
                    ),
                    "state_frame_count_per_episode": len(frame_indexes),
                    "camera_stream_count": len(camera_names),
                    "dataset_semantics": (
                        "camera_fixed fixes only camera pose; scene, table material, lighting, "
                        "trajectory, and actions are paired with camera_randomized"
                        if any(output["camera_group"] == "randomized" for output in outputs)
                        else "fixed-camera-only capture; no randomized episodes were requested"
                    ),
                    "atomicity": (
                        "each episode is exactly-once; the capture batch is recoverable "
                        "rather than cross-dataset all-or-nothing"
                    ),
                    "split_policy": (
                        "keep every episode with the same source_trajectory_group_id in one "
                        "train/validation/test split"
                    ),
                    "runtime_camera_config": runtime_camera_config,
                    "randomization_config": randomization,
                    "preview_videos": preview_videos,
                    "contact_sheet": contact_sheet,
                    "captures": outputs,
                },
            )

        committed_targets = [target for target in targets if target["committed"] is not None]
        pending_targets = [target for target in targets if target["committed"] is None]
        if not pending_targets:
            lerobot_episodes: list[dict[str, Any]] = []
            outputs: list[dict[str, Any]] = []
            preview_videos: dict[str, str] = {}
            contact_sheet: str | None = None
            randomization: dict[str, Any] | None = None
            for target in targets:
                committed = target["committed"]
                episode = committed["episode"]
                if int(episode["frame_count"]) != len(frame_indexes):
                    raise ValueError(
                        f"Committed capture frame count mismatch for {target['capture'].capture_id}: "
                        f"stored={episode['frame_count']}, current={len(frame_indexes)}."
                    )
                metadata = deepcopy(committed["metadata"])
                metadata["lerobot_episode"] = episode
                atomic_write_json(target["metadata_path"], metadata)
                preview_videos.update(metadata.get("preview_videos") or {})
                contact_sheet = contact_sheet or metadata.get("contact_sheet")
                randomization = randomization or metadata.get("randomization_config")
                lerobot_episodes.append(decorated_episode(target, episode))
                outputs.append(capture_output(target, episode, commit_status="already_committed"))
            write_capture_manifest(
                outputs,
                randomization=randomization,
                preview_videos=preview_videos,
                contact_sheet=contact_sheet,
            )
            primary = lerobot_episodes[0]
            return RenderRecord(
                status="skipped",
                task_id=task_id,
                task_name=task_name,
                attempt_id=int(spec.attempt_id),
                embodiment_id=embodiment_id,
                model_id=model_id,
                variant_id=int(spec.variant_id),
                seed=int(spec.seed),
                num_frames=len(frame_indexes),
                camera_names=list(camera_names),
                output_path=str(targets[0]["dataset_root"]),
                metadata_path=str(capture_manifest_path),
                lerobot_episode=primary,
                lerobot_episodes=lerobot_episodes,
                camera_dataset_outputs=outputs,
                capture_manifest_path=str(capture_manifest_path),
                num_lerobot_episodes=len(lerobot_episodes),
                num_state_frames_replayed=0,
                num_camera_frames_rendered=0,
                num_lerobot_frames_written=0,
                preview_videos=preview_videos,
                contact_sheet=contact_sheet,
                randomization_config=randomization,
            )

        policy_state_metadata = _policy_joint_qpos_metadata(model)
        policy_state = _policy_joint_state(decoded, policy_state_metadata)
        ee_body_name = _lerobot_ee_body_name(env.xml_metadata)
        randomization = _sample_and_apply_randomization(
            env,
            task_id=task_id,
            attempt_id=int(spec.attempt_id),
            task_name=task_name,
            variant_id=int(spec.variant_id),
            global_seed=int(spec.seed),
            camera_names=camera_names,
            camera_config=runtime_camera_config,
            enabled=spec.randomization_enabled,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(spec.width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(spec.height))
        pending_camera_names = tuple(
            dict.fromkeys(
                camera_name
                for target in pending_targets
                for camera_name in target["capture"].physical_camera_names
            )
        )
        for stale_spool in metadata_root.glob(".camera_frame_spool.*.partial"):
            parts = stale_spool.name.split(".")
            try:
                owner_pid = int(parts[2])
                os.kill(owner_pid, 0)
            except ProcessLookupError:
                shutil.rmtree(stale_spool, ignore_errors=True)
            except (IndexError, ValueError, PermissionError):
                logger.warning("Leaving unrecognized or active camera spool untouched: %s", stale_spool)
        image_stats = {
            camera_name: _StreamingRgbStats() for camera_name in pending_camera_names
        }
        contact_sheet_indexes = (
            set(np.linspace(0, len(frame_indexes) - 1, min(12, len(frame_indexes))).astype(int))
            if spec.save_contact_sheet
            else set()
        )
        contact_sheet_frames: list[np.ndarray] = []
        ee_state_dim = 7 if ee_body_name else None
        camera_intrinsics = {
            camera_name: _camera_intrinsics_matrix(
                env,
                camera_name,
                width=int(spec.width),
                height=int(spec.height),
            )
            for camera_name in pending_camera_names
        }
        with ExitStack() as shard_stack:
            for target in pending_targets:
                lerobot_features, camera_keys = _build_lerobot_episode_schema(
                    camera_config=target["camera_config"],
                    state_dim=int(policy_state.shape[1]),
                    action_dim=int(decoded.nu),
                    ee_state_dim=ee_state_dim,
                    width=int(spec.width),
                    height=int(spec.height),
                )
                target["lerobot_features"] = lerobot_features
                target["camera_keys"] = camera_keys
                target["policy_state_dim"] = int(policy_state.shape[1])
                target["action_dim"] = int(decoded.nu)
                target["ee_state_dim"] = ee_state_dim
                target["shard_partial_root"] = (
                    target["metadata_root"] / "lerobot_episode_shard.partial"
                )
                target["shard_ready_root"] = (
                    target["metadata_root"] / "lerobot_episode_shard.ready"
                )
                target["stream_writer"] = shard_stack.enter_context(
                    StreamingEpisodeShardWriter(
                        partial_root=target["shard_partial_root"],
                        ready_root=target["shard_ready_root"],
                        dataset_root=target["dataset_root"],
                        repo_id=target["repo_id"],
                        features=lerobot_features,
                        task_name=task_name,
                        identity=target["identity"],
                        fps=int(spec.lerobot_fps),
                        robot_type=str(spec.lerobot_robot_type),
                        vcodec=spec.lerobot_vcodec,
                        encoder_queue_maxsize=int(spec.lerobot_encoder_queue_maxsize),
                        encoder_threads=spec.lerobot_encoder_threads,
                    )
                )

            for local_index, frame_index in enumerate(frame_indexes):
                _set_row(data, decoded, int(frame_index), state_times[int(frame_index)])
                env.sim.forward()
                ee_state_frame = _body_pose7(env, ee_body_name) if ee_body_name else None
                camera_matrices = {
                    camera_name: {
                        "intrinsics": camera_intrinsics[camera_name],
                        "camera_to_world": _camera_to_world_matrix(env, camera_name),
                    }
                    for camera_name in pending_camera_names
                }
                rendered_frames = _render_rgb_cameras_shared_scene(
                    env.sim,
                    pending_camera_names,
                    width=int(spec.width),
                    height=int(spec.height),
                )
                for camera_name, frame in rendered_frames.items():
                    image_stats[camera_name].update(frame)
                if local_index in contact_sheet_indexes:
                    contact_sheet_frames.append(rendered_frames[pending_camera_names[0]].copy())
                for target in pending_targets:
                    writer = target["stream_writer"]
                    if not writer.needs_frames:
                        continue
                    writer.add_frame(
                        _build_lerobot_stream_frame(
                            camera_keys=target["camera_keys"],
                            rendered_frames=rendered_frames,
                            camera_matrices=camera_matrices,
                            state=policy_state[local_index],
                            action=decoded.ctrl[local_index],
                            ee_state=ee_state_frame,
                        )
                    )

            image_quality = {
                camera_name: stats.validate(camera_name)
                for camera_name, stats in image_stats.items()
            }
            for target in pending_targets:
                target["stream_writer"].finish()

        preview_videos: dict[str, str] = {}
        contact_sheet_path: Path | None = None
        if contact_sheet_frames:
            contact_sheet_path = metadata_root / "contact_sheet.jpg"
            _write_contact_sheet(np.stack(contact_sheet_frames), contact_sheet_path)
        recovered_contact_sheet: str | None = None
        for target in committed_targets:
            committed_metadata = target["committed"]["metadata"]
            preview_videos = {
                **(committed_metadata.get("preview_videos") or {}),
                **preview_videos,
            }
            recovered_contact_sheet = (
                recovered_contact_sheet or committed_metadata.get("contact_sheet")
            )
        manifest_contact_sheet = (
            str(contact_sheet_path) if contact_sheet_path else recovered_contact_sheet
        )

        shared_metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "renderer_backend": "robosuite_table_arena_env_state_resampling",
            "environment_backend": "robosuite.environments.base.MujocoEnv",
            "randomizer_backend": "robosuite.utils.mjmod",
            "robosuite_version": _robosuite_version(),
            "mujoco_version": mujoco.__version__,
            "task_id": task_id,
            "task_name": task_name,
            "attempt_id": int(spec.attempt_id),
            "embodiment_id": embodiment_id,
            "model_id": model_id,
            "source": {
                "source_fingerprint": source_fingerprint,
                "trajectory_data_source": (bundle.get("task_trajectory") or {}).get(
                    "trajectory_data_source"
                ),
                "materialized_assets": asset_resolution,
                "work_dir": str(work_dir),
                "robosuite_env_xml": env.xml_metadata,
            },
            "trajectory": {
                "state_format": "packed_state_v1",
                "source_trajectory_group_id": source_trajectory_group_id,
                "num_source_frames": int(source_decoded.n_steps),
                "num_frames": len(frame_indexes),
                "output_state_indices": frame_indexes,
                "nq": int(decoded.nq),
                "nv": int(decoded.nv),
                "nu": int(decoded.nu),
                "timestep": float(decoded.timestep),
                "timing": time_summary,
                "frame_schedule": frame_schedule,
                "scene_state_transform": scene_state_transform,
            },
            "render": {
                "render_fingerprint_schema_version": RENDER_FINGERPRINT_SCHEMA_VERSION,
                "capture_batch_fingerprint": capture_batch_fingerprint,
                "width": int(spec.width),
                "height": int(spec.height),
                "runtime_physical_camera_names": list(camera_names),
                "logical_camera_names": list(
                    runtime_camera_config.get("logical_camera_names") or []
                ),
                "runtime_camera_config": runtime_camera_config,
                "frame_stride": int(spec.frame_stride),
                "max_frames": spec.max_frames,
                "gpu_id": spec.gpu_id,
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "pixel_origin_correction": "flipud_each_tile_after_single_atlas_readback",
                "vertical_flip_applied": True,
                "frame_storage": "direct_bounded_queue_to_streaming_h264",
                "streaming_encoder_queue_maxsize": int(spec.lerobot_encoder_queue_maxsize),
                "streaming_encoder_threads": spec.lerobot_encoder_threads,
                "state_forward_calls": len(frame_indexes),
                "configured_camera_count": len(camera_names),
                "configured_camera_names": list(camera_names),
                "executed_renders_per_state": len(pending_camera_names),
                "executed_camera_names": list(pending_camera_names),
                "scene_update_strategy": (
                    "one_scene_update_then_camera_updates_with_single_tiled_atlas_readback"
                ),
                "full_scene_updates_per_state": 1,
                "camera_only_updates_per_state": max(len(pending_camera_names) - 1, 0),
                "gpu_readbacks_per_state": 1,
            },
            "capture_batch": {
                "schema": CAMERA_CAPTURE_MANIFEST_SCHEMA,
                "source_trajectory_group_id": source_trajectory_group_id,
                "capture_count": len(targets),
                "physical_camera_count": len(camera_names),
                "trajectory_replay_count": 1,
                "pairing": "same_replica_index_across_logical_cameras",
            },
            "runtime": runtime_metadata_from_bundle(bundle),
            "randomization_config": randomization,
            "image_quality": image_quality,
            "preview_videos": preview_videos,
            "contact_sheet": str(contact_sheet_path) if contact_sheet_path else None,
        }

        lerobot_episodes = []
        outputs = []
        commit_requests: list[dict[str, Any]] = []
        pending_commit_targets: list[dict[str, Any]] = []
        for target in targets:
            capture: CameraCapturePair = target["capture"]
            if target["committed"] is not None:
                committed = target["committed"]
                episode = committed["episode"]
                recovered_metadata = deepcopy(committed["metadata"])
                target["result_episode"] = episode
                target["result_metadata"] = recovered_metadata
                target["result_commit_status"] = "already_committed"
            else:
                pair_camera_config = target["camera_config"]
                metadata = deepcopy(shared_metadata)
                metadata["render"].update(
                    {
                        "render_fingerprint": target["render_fingerprint"],
                        "camera_names": list(capture.physical_camera_names),
                        "camera_config": pair_camera_config,
                    }
                )
                metadata["capture_episode"] = {
                    "schema": CAMERA_CAPTURE_MULTIPLEX_SCHEMA,
                    "capture_id": capture.capture_id,
                    "camera_group": capture.camera_group,
                    "camera_replica_id": capture.camera_replica_id,
                    "source_trajectory_group_id": source_trajectory_group_id,
                    "logical_to_physical_camera_names": {
                        logical_name: physical_name
                        for logical_name, physical_name in capture.camera_name_map
                    },
                    "camera_pose_sampling_frequency": "once_per_trajectory",
                    "shared_scene_sampling_frequency": "once_per_source_trajectory",
                }
                metadata["image_quality"] = {
                    name: image_quality[name] for name in capture.physical_camera_names
                }
                metadata["preview_videos"] = {
                    name: preview_videos[name]
                    for name in capture.physical_camera_names
                    if name in preview_videos
                }
                metadata["lerobot"] = {
                    "layout": "one_lerobot_dataset_per_task_and_camera_group",
                    "schema": LEROBOT_POLICY_SCHEMA,
                    "task_dataset_root": str(target["dataset_root"]),
                    "repo_id": target["repo_id"],
                    "camera_group": capture.camera_group,
                    "state_feature": "observation.state",
                    "state_semantics": "actuated_joint_qpos_in_actuator_order",
                    "state_dim": int(policy_state.shape[1]),
                    "policy_state_metadata": policy_state_metadata,
                    "action_feature": "action",
                    "action_semantics": "ctrl",
                    "state_resampling_schema": STATE_RESAMPLING_SCHEMA,
                    "state_resampling_mode": frame_schedule["timeline_mode"],
                    "state_resampling_physics_rollout": False,
                    "state_resampling_mj_step_calls": 0,
                    "action_resampling": frame_schedule["ctrl_method"],
                    "action_dim": int(decoded.nu),
                    "robot_root_state_feature": None,
                    "robot_root_state_policy": "omitted_no_stable_source",
                    "ee_state_feature": "observation.ee_state" if ee_body_name else None,
                    "ee_state_source_body": ee_body_name,
                    "image_cache_policy": "direct_streaming_h264_no_rgb_spool_or_png_round_trip",
                    "commit_schema": ATOMIC_COMMIT_SCHEMA,
                    "video_encoding_scope": "concurrent_with_render_outside_per_task_commit_lock",
                    "camera_features": {
                        key: {
                            "source_camera_name": camera_name,
                            "logical_camera_name": next(
                                logical
                                for logical, physical in capture.camera_name_map
                                if physical == camera_name
                            ),
                            "image_feature": f"observation.images.{key}",
                            "intrinsics_feature": f"observation.cameras.{key}.intrinsics",
                            "camera_to_world_feature": (
                                f"observation.cameras.{key}.camera_to_world"
                            ),
                            "intrinsics_storage": "row_major_flat",
                            "intrinsics_source_shape": [3, 3],
                            "camera_to_world_storage": "row_major_flat",
                            "camera_to_world_source_shape": [4, 4],
                            "camera_frame_convention": (
                                "MuJoCo/OpenGL +X image-right,+Y up,-Z forward to world"
                            ),
                        }
                        for camera_name, key in _lerobot_camera_keys(
                            pair_camera_config
                        ).items()
                    },
                }
                target["result_metadata"] = metadata
                commit_requests.append(
                    {
                        "shard_root": target["shard_ready_root"],
                        "dataset_root": target["dataset_root"],
                        "repo_id": target["repo_id"],
                        "expected_features": target["lerobot_features"],
                        "identity": target["identity"],
                        "metadata": metadata,
                        "fps": int(spec.lerobot_fps),
                        "robot_type": str(spec.lerobot_robot_type),
                        "vcodec": spec.lerobot_vcodec,
                        "camera_keys": target["camera_keys"],
                        "policy_state_dim": target["policy_state_dim"],
                        "action_dim": target["action_dim"],
                        "ee_state_dim": target["ee_state_dim"],
                        "policy_state_metadata": policy_state_metadata,
                        "nq": int(decoded.nq),
                        "nv": int(decoded.nv),
                        "nu": int(decoded.nu),
                    }
                )
                pending_commit_targets.append(target)

        committed_batch = commit_episode_shards(commit_requests)
        for target, episode in zip(pending_commit_targets, committed_batch, strict=True):
            target["result_episode"] = episode
            target["result_commit_status"] = "committed"

        for target in targets:
            episode = target["result_episode"]
            metadata = target["result_metadata"]
            metadata["lerobot_episode"] = episode
            atomic_write_json(target["metadata_path"], metadata)
            lerobot_episodes.append(decorated_episode(target, episode))
            outputs.append(
                capture_output(
                    target,
                    episode,
                    commit_status=target["result_commit_status"],
                )
            )

        for target in targets:
            durable = load_committed_attempt(target["dataset_root"], target["identity"])
            if durable is None:
                raise RuntimeError(
                    f"Capture commit marker missing after write: {target['capture'].capture_id}."
                )
        write_capture_manifest(
            outputs,
            randomization=randomization,
            preview_videos=preview_videos,
            contact_sheet=manifest_contact_sheet,
        )
        elapsed = time.perf_counter() - start_time
        return RenderRecord(
            status="success",
            task_id=task_id,
            task_name=task_name,
            attempt_id=int(spec.attempt_id),
            embodiment_id=embodiment_id,
            model_id=model_id,
            variant_id=int(spec.variant_id),
            seed=int(spec.seed),
            num_frames=len(frame_indexes),
            camera_names=list(camera_names),
            render_time_s=elapsed,
            output_path=str(targets[0]["dataset_root"]),
            metadata_path=str(capture_manifest_path),
            lerobot_episode=lerobot_episodes[0],
            lerobot_episodes=lerobot_episodes,
            camera_dataset_outputs=outputs,
            capture_manifest_path=str(capture_manifest_path),
            num_lerobot_episodes=len(lerobot_episodes),
            num_state_frames_replayed=len(frame_indexes),
            num_camera_frames_rendered=len(pending_camera_names) * len(frame_indexes),
            num_lerobot_frames_written=len(pending_targets) * len(frame_indexes),
            preview_videos=preview_videos,
            contact_sheet=manifest_contact_sheet,
            randomization_config=randomization,
        )
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            logger.exception("Failed to close the multiplexed Robosuite environment.")


def _runtime_scene_variant_count(task_payload: dict[str, Any]) -> int:
    scene_variants = task_payload.get("scene_variants")
    if isinstance(scene_variants, list):
        return len(scene_variants)
    checker_config = task_payload.get("checker_config")
    if isinstance(checker_config, dict):
        layout_variants = checker_config.get("layout_variants")
        if isinstance(layout_variants, list):
            return len(layout_variants)
    return 0


def render_attempt_job(spec: RenderJobSpec) -> RenderRecord:
    start = time.perf_counter()
    _validate_normalized_render_fps(spec.lerobot_fps)
    disable_proxy_env()
    if spec.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_id)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(spec.gpu_id)
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = Path(spec.output_dir).expanduser().resolve()
    cache_dir = Path(spec.cache_dir).expanduser().resolve()
    if not spec.local_bundle_path:
        raise ValueError("render_attempt_job needs spec.local_bundle_path: the JSON bundle to render.")
    # The bundle carries its own scene and trajectory, and --asset-dir carries its assets.
    asset_base_url = spec.asset_base_url or ""
    record = RenderRecord(status="failed", attempt_id=int(spec.attempt_id), variant_id=int(spec.variant_id))
    env: AxisReplayRobosuiteEnv | None = None
    try:
        bundle = json.loads(Path(spec.local_bundle_path).read_text(encoding="utf-8"))
        task_payload = bundle.get("task") or {}
        task_id = int(task_payload.get("id") or (bundle.get("attempt") or {}).get("task_id"))
        task_name = str(task_payload.get("name") or f"task_{task_id}")
        embodiment_id = spec.embodiment_id_override or infer_embodiment_id(
            task_name, task_payload.get("mjcf_xml"))
        camera_config = _load_embodiment_camera_config(
            embodiment_id,
            spec.cameras,
            config_id=_camera_config_id(embodiment_id, task_payload.get("mjcf_xml")),
        )
        camera_config = resolve_visual_scene_profile(
            camera_config,
            project_root=PROJECT_ROOT,
            global_seed=int(spec.seed),
            task_id=task_id,
            attempt_id=int(spec.attempt_id),
            task_name=task_name,
            variant_id=int(spec.variant_id),
        )
        if bool((camera_config.get("capture_multiplexing") or {}).get("enabled", False)):
            capture_metadata_root = (
                output_dir
                / "episode_metadata"
                / _task_lerobot_dataset_slug(task_id, task_name, embodiment_id)
                / (
                    f"task_{task_id}_attempt_{spec.attempt_id}_{embodiment_id}_"
                    f"variant_{spec.variant_id:03d}"
                )
            )
            with _exclusive_camera_capture_batch_lock(capture_metadata_root):
                return _render_attempt_job_camera_multiplex(
                    spec,
                    start_time=start,
                    bundle=bundle,
                    task_id=task_id,
                    task_name=task_name,
                    embodiment_id=embodiment_id,
                    logical_camera_config=camera_config,
                    output_dir=output_dir,
                    cache_dir=cache_dir,
                    asset_base_url=asset_base_url,
                )
        _validate_camera_render_aspect_ratio(
            camera_config,
            width=int(spec.width),
            height=int(spec.height),
        )
        camera_names = _camera_names_from_config(camera_config)
        trajectory_data = (bundle.get("task_trajectory") or {}).get("trajectory_data") or {}
        if trajectory_data.get("format") != "packed_state_v1":
            raise ValueError(f"Expected packed_state_v1 trajectory, got {trajectory_data.get('format')!r}.")

        variant_slug = f"task_{task_id}_attempt_{spec.attempt_id}_{embodiment_id}_variant_{spec.variant_id:03d}"
        lerobot_base_root = (
            Path(spec.lerobot_root).expanduser().resolve()
            if spec.lerobot_root
            else output_dir / "lerobot_tasks"
        )
        lerobot_dataset_root = _lerobot_task_dataset_root(
            lerobot_base_root,
            task_id,
            task_name,
            embodiment_id,
        )
        lerobot_repo_id = _task_lerobot_repo_id(spec.lerobot_repo_id, task_id, task_name, embodiment_id)
        source_fingerprint = _render_source_fingerprint(bundle)
        render_fingerprint = _render_fingerprint(
            camera_config,
            spec,
            lerobot_repo_id=lerobot_repo_id,
            source_fingerprint=source_fingerprint,
        )
        out_root = output_dir / "episode_metadata" / lerobot_dataset_root.name / variant_slug
        metadata_path = out_root / "metadata.json"
        output_path = lerobot_dataset_root
        if spec.resume and metadata_path.is_file() and output_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored_camera_names = tuple((metadata.get("render") or {}).get("camera_names") or ())
            stored_render_fingerprint = (metadata.get("render") or {}).get("render_fingerprint")
            stored_lerobot_schema = (metadata.get("lerobot") or {}).get("schema")
            if (
                stored_camera_names == camera_names
                and stored_render_fingerprint == render_fingerprint
                and stored_lerobot_schema == LEROBOT_POLICY_SCHEMA
            ):
                trajectory_meta = metadata.get("trajectory") or {}
                return RenderRecord(
                    status="skipped",
                    task_id=task_id,
                    task_name=task_name,
                    attempt_id=int(spec.attempt_id),
                    embodiment_id=embodiment_id,
                    model_id=metadata.get("model_id"),
                    variant_id=int(spec.variant_id),
                    seed=int(spec.seed),
                    num_frames=int(trajectory_meta.get("num_frames") or metadata.get("num_frames") or 0),
                    camera_names=list(camera_names),
                    output_path=str(output_path),
                    metadata_path=str(metadata_path),
                    lerobot_episode=metadata.get("lerobot_episode"),
                    lerobot_episodes=(
                        [metadata["lerobot_episode"]]
                        if metadata.get("lerobot_episode") is not None
                        else None
                    ),
                    num_lerobot_episodes=int(metadata.get("lerobot_episode") is not None),
                    num_state_frames_replayed=0,
                    num_camera_frames_rendered=0,
                    num_lerobot_frames_written=0,
                    randomization_config=metadata.get("randomization_config"),
                )
            raise ValueError(
                "Resume rejected because the existing task output does not match the current render "
                f"configuration for task {task_id}, attempt {spec.attempt_id}, variant {spec.variant_id}: "
                f"stored_fingerprint={stored_render_fingerprint!r}, "
                f"current_fingerprint={render_fingerprint!r}, "
                f"stored_cameras={list(stored_camera_names)!r}, current_cameras={list(camera_names)!r}, "
                f"stored_lerobot_schema={stored_lerobot_schema!r}. Use a new --output-dir (and a new "
                "--lerobot-root if one was set explicitly), or pass --lerobot-overwrite to rebuild the "
                "task dataset; the renderer will not append a mixed or duplicate episode during resume."
            )

        work_dir = cache_dir / "tasks" / f"task_{task_id}_attempt_{spec.attempt_id}"
        with _exclusive_materialization_lock(work_dir):
            asset_resolution = _materialize_bundle(
                bundle,
                work_dir=work_dir / "assets",
                asset_dir=spec.asset_dir,
                asset_base_url=asset_base_url,
                prepared_cache_dir=cache_dir / "prepared_assets",
            )
            entry_xml_path = Path(asset_resolution["entry_xml"])
            if not entry_xml_path.is_absolute():
                entry_xml_path = (work_dir / "assets" / entry_xml_path).resolve()
            env = AxisReplayRobosuiteEnv(
                entry_xml=entry_xml_path,
                camera_names=camera_names,
                camera_config=camera_config,
                layout_cache_dir=cache_dir / "prepared_layouts",
                has_renderer=False,
                has_offscreen_renderer=True,
                render_collision_mesh=False,
                render_visual_mesh=True,
                render_gpu_device_id=int(spec.gpu_id) if spec.gpu_id is not None else -1,
                control_freq=20,
                lite_physics=True,
                horizon=1000,
                ignore_done=True,
                hard_reset=False,
                renderer="mujoco",
                seed=int(spec.seed),
            )
        model = env.sim.model._model
        data = env.sim.data._data
        source_decoded = decode_packed_state_v1(trajectory_data)
        _validate_dimensions(model, source_decoded)
        qpos_offset, scene_state_transform = _build_scene_state_transform(model, env.xml_metadata)
        decoded = _apply_qpos_offset(source_decoded, qpos_offset)
        expected_duration_s = (bundle.get("attempt") or {}).get("simulation_time_seconds")
        state_times, time_summary = _build_state_times(
            decoded,
            expected_duration_s=expected_duration_s,
        )
        resampled_timeline = resample_packed_state_to_fps(
            model,
            decoded,
            state_times,
            time_summary,
            fps=int(spec.lerobot_fps),
            expected_duration_s=expected_duration_s,
            frame_stride=spec.frame_stride,
            max_frames=spec.max_frames,
        )
        decoded = resampled_timeline.trajectory
        state_times = resampled_timeline.state_times.tolist()
        frame_schedule = resampled_timeline.metadata
        time_summary = frame_schedule["source_timing"]
        frame_indexes = list(range(decoded.n_steps))
        model_id = f"{embodiment_id}:task_{task_id}:nq{model.nq}:nv{model.nv}:nu{model.nu}"
        episode_identity = EpisodeIdentity(
            task_id=task_id,
            task_name=task_name,
            attempt_id=int(spec.attempt_id),
            embodiment_id=embodiment_id,
            model_id=model_id,
            variant_id=int(spec.variant_id),
            render_fingerprint=render_fingerprint,
        )
        if spec.resume:
            committed_attempt = load_committed_attempt(lerobot_dataset_root, episode_identity)
            if committed_attempt is not None:
                committed_episode = committed_attempt["episode"]
                committed_metadata = dict(committed_attempt["metadata"])
                committed_metadata["lerobot_episode"] = committed_episode
                atomic_write_json(metadata_path, committed_metadata)
                for shard_path in (
                    out_root / "lerobot_episode_shard.partial",
                    out_root / "lerobot_episode_shard.ready",
                ):
                    if shard_path.exists():
                        shutil.rmtree(shard_path)
                return RenderRecord(
                    status="skipped",
                    task_id=task_id,
                    task_name=task_name,
                    attempt_id=int(spec.attempt_id),
                    embodiment_id=embodiment_id,
                    model_id=model_id,
                    variant_id=int(spec.variant_id),
                    seed=int(spec.seed),
                    num_frames=int(committed_episode.get("frame_count") or 0),
                    camera_names=list(camera_names),
                    output_path=str(output_path),
                    metadata_path=str(metadata_path),
                    lerobot_episode=committed_episode,
                    lerobot_episodes=[committed_episode],
                    num_lerobot_episodes=1,
                    num_state_frames_replayed=0,
                    num_camera_frames_rendered=0,
                    num_lerobot_frames_written=0,
                    preview_videos=committed_metadata.get("preview_videos") or {},
                    contact_sheet=committed_metadata.get("contact_sheet"),
                    randomization_config=committed_metadata.get("randomization_config"),
                )
        policy_state_metadata = _policy_joint_qpos_metadata(model)
        policy_state = _policy_joint_state(decoded, policy_state_metadata)
        ee_body_name = _lerobot_ee_body_name(env.xml_metadata)
        randomization = _sample_and_apply_randomization(
            env,
            task_id=task_id,
            attempt_id=int(spec.attempt_id),
            task_name=task_name,
            variant_id=int(spec.variant_id),
            global_seed=int(spec.seed),
            camera_names=camera_names,
            camera_config=camera_config,
            enabled=spec.randomization_enabled,
        )
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(spec.width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(spec.height))
        if spec.save_preview_video:
            raise ValueError(
                "Streaming state replay writes canonical H.264 camera videos directly; "
                "--save-preview-video is not supported because it would duplicate every stream."
            )
        ee_state_dim = 7 if ee_body_name else None
        lerobot_features, camera_keys = _build_lerobot_episode_schema(
            camera_config=camera_config,
            state_dim=int(policy_state.shape[1]),
            action_dim=int(decoded.nu),
            ee_state_dim=ee_state_dim,
            width=int(spec.width),
            height=int(spec.height),
        )
        policy_state_dim = int(policy_state.shape[1])
        action_dim = int(decoded.nu)
        shard_partial_root = out_root / "lerobot_episode_shard.partial"
        shard_ready_root = out_root / "lerobot_episode_shard.ready"
        image_stats = {camera_name: _StreamingRgbStats() for camera_name in camera_names}
        contact_sheet_indexes = (
            set(np.linspace(0, len(frame_indexes) - 1, min(12, len(frame_indexes))).astype(int))
            if spec.save_contact_sheet
            else set()
        )
        contact_sheet_frames: list[np.ndarray] = []
        camera_intrinsics = {
            camera_name: _camera_intrinsics_matrix(
                env,
                camera_name,
                width=int(spec.width),
                height=int(spec.height),
            )
            for camera_name in camera_names
        }
        with StreamingEpisodeShardWriter(
            partial_root=shard_partial_root,
            ready_root=shard_ready_root,
            dataset_root=lerobot_dataset_root,
            repo_id=lerobot_repo_id,
            features=lerobot_features,
            task_name=task_name,
            identity=episode_identity,
            fps=int(spec.lerobot_fps),
            robot_type=str(spec.lerobot_robot_type),
            vcodec=spec.lerobot_vcodec,
            encoder_queue_maxsize=int(spec.lerobot_encoder_queue_maxsize),
            encoder_threads=spec.lerobot_encoder_threads,
        ) as stream_writer:
            for local_index, frame_index in enumerate(frame_indexes):
                _set_row(data, decoded, int(frame_index), state_times[int(frame_index)])
                env.sim.forward()
                ee_state_frame = _body_pose7(env, ee_body_name) if ee_body_name else None
                camera_matrices = {
                    camera_name: {
                        "intrinsics": camera_intrinsics[camera_name],
                        "camera_to_world": _camera_to_world_matrix(env, camera_name),
                    }
                    for camera_name in camera_names
                }
                rendered_frames = _render_rgb_cameras_shared_scene(
                    env.sim,
                    camera_names,
                    width=int(spec.width),
                    height=int(spec.height),
                )
                for camera_name, frame in rendered_frames.items():
                    image_stats[camera_name].update(frame)
                if local_index in contact_sheet_indexes:
                    contact_sheet_frames.append(rendered_frames[camera_names[0]].copy())
                if stream_writer.needs_frames:
                    stream_writer.add_frame(
                        _build_lerobot_stream_frame(
                            camera_keys=camera_keys,
                            rendered_frames=rendered_frames,
                            camera_matrices=camera_matrices,
                            state=policy_state[local_index],
                            action=decoded.ctrl[local_index],
                            ee_state=ee_state_frame,
                        )
                    )
            image_quality = {
                camera_name: stats.validate(camera_name)
                for camera_name, stats in image_stats.items()
            }
            stream_writer.finish()
        preview_videos: dict[str, str] = {}
        contact_sheet_path: Path | None = None
        if contact_sheet_frames:
            contact_sheet_path = out_root / "contact_sheet.jpg"
            _write_contact_sheet(np.stack(contact_sheet_frames), contact_sheet_path)

        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "renderer_backend": "robosuite_table_arena_env_state_resampling",
            "environment_backend": "robosuite.environments.base.MujocoEnv",
            "randomizer_backend": "robosuite.utils.mjmod",
            "robosuite_version": _robosuite_version(),
            "mujoco_version": mujoco.__version__,
            "task_id": task_id,
            "task_name": task_name,
            "attempt_id": int(spec.attempt_id),
            "embodiment_id": embodiment_id,
            "model_id": model_id,
            "source": {
                "source_fingerprint": source_fingerprint,
                "trajectory_data_source": (bundle.get("task_trajectory") or {}).get("trajectory_data_source"),
                "materialized_assets": asset_resolution,
                "work_dir": str(work_dir),
                "robosuite_env_xml": env.xml_metadata,
            },
            "trajectory": {
                "state_format": "packed_state_v1",
                "num_source_frames": int(source_decoded.n_steps),
                "num_frames": len(frame_indexes),
                "output_state_indices": frame_indexes,
                "nq": int(decoded.nq),
                "nv": int(decoded.nv),
                "nu": int(decoded.nu),
                "timestep": float(decoded.timestep),
                "timing": time_summary,
                "frame_schedule": frame_schedule,
                "scene_state_transform": scene_state_transform,
            },
            "render": {
                "render_fingerprint_schema_version": RENDER_FINGERPRINT_SCHEMA_VERSION,
                "render_fingerprint": render_fingerprint,
                "width": int(spec.width),
                "height": int(spec.height),
                "camera_names": list(camera_names),
                "camera_config": camera_config,
                "frame_stride": int(spec.frame_stride),
                "max_frames": spec.max_frames,
                "gpu_id": spec.gpu_id,
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "pixel_origin_correction": "flipud_each_tile_after_single_atlas_readback",
                "vertical_flip_applied": True,
                "state_forward_calls": len(frame_indexes),
                "scene_update_strategy": (
                    "one_scene_update_then_camera_updates_with_single_tiled_atlas_readback"
                ),
                "full_scene_updates_per_state": 1,
                "camera_only_updates_per_state": max(len(camera_names) - 1, 0),
                "gpu_readbacks_per_state": 1,
            },
            "runtime": runtime_metadata_from_bundle(bundle),
            "randomization_config": randomization,
            "image_quality": image_quality,
            "preview_videos": preview_videos,
            "contact_sheet": str(contact_sheet_path) if contact_sheet_path else None,
        }
        metadata["lerobot"] = {
            "layout": "one_lerobot_dataset_per_task",
            "schema": LEROBOT_POLICY_SCHEMA,
            "task_dataset_root": str(lerobot_dataset_root),
            "repo_id": lerobot_repo_id,
            "state_feature": "observation.state",
            "state_semantics": "actuated_joint_qpos_in_actuator_order",
            "state_dim": int(policy_state.shape[1]),
            "policy_state_metadata": policy_state_metadata,
            "action_feature": "action",
            "action_semantics": "ctrl",
            "state_resampling_schema": STATE_RESAMPLING_SCHEMA,
            "state_resampling_mode": frame_schedule["timeline_mode"],
            "state_resampling_physics_rollout": False,
            "state_resampling_mj_step_calls": 0,
            "action_resampling": frame_schedule["ctrl_method"],
            "action_dim": int(decoded.nu),
            "robot_root_state_feature": None,
            "robot_root_state_policy": "omitted_no_stable_source",
            "ee_state_feature": "observation.ee_state" if ee_state_dim is not None else None,
            "ee_state_source_body": ee_body_name,
            "image_cache_policy": "direct_streaming_h264_no_rgb_spool_or_png_round_trip",
            "commit_schema": ATOMIC_COMMIT_SCHEMA,
            "video_encoding_scope": "during_state_replay_outside_per_task_commit_lock",
            "streaming_encoder_queue_maxsize": int(spec.lerobot_encoder_queue_maxsize),
            "streaming_encoder_threads": spec.lerobot_encoder_threads,
            "camera_features": {
                key: {
                    "source_camera_name": camera_name,
                    "image_feature": f"observation.images.{key}",
                    "intrinsics_feature": f"observation.cameras.{key}.intrinsics",
                    "camera_to_world_feature": f"observation.cameras.{key}.camera_to_world",
                    "intrinsics_storage": "row_major_flat",
                    "intrinsics_source_shape": [3, 3],
                    "camera_to_world_storage": "row_major_flat",
                    "camera_to_world_source_shape": [4, 4],
                }
                for camera_name, key in _lerobot_camera_keys(camera_config).items()
            },
        }
        lerobot_episode = commit_episode_shard(
            shard_root=shard_ready_root,
            dataset_root=lerobot_dataset_root,
            repo_id=lerobot_repo_id,
            expected_features=lerobot_features,
            identity=episode_identity,
            metadata=metadata,
            fps=int(spec.lerobot_fps),
            robot_type=str(spec.lerobot_robot_type),
            vcodec=spec.lerobot_vcodec,
            camera_keys=camera_keys,
            policy_state_dim=policy_state_dim,
            action_dim=action_dim,
            ee_state_dim=ee_state_dim,
            policy_state_metadata=policy_state_metadata,
            nq=int(decoded.nq),
            nv=int(decoded.nv),
            nu=int(decoded.nu),
        )
        metadata["lerobot_episode"] = lerobot_episode
        atomic_write_json(metadata_path, metadata)
        elapsed = time.perf_counter() - start
        record = RenderRecord(
            status="success",
            task_id=task_id,
            task_name=task_name,
            attempt_id=int(spec.attempt_id),
            embodiment_id=embodiment_id,
            model_id=model_id,
            variant_id=int(spec.variant_id),
            seed=int(spec.seed),
            num_frames=len(frame_indexes),
            camera_names=list(camera_names),
            render_time_s=elapsed,
            output_path=str(output_path),
            metadata_path=str(metadata_path),
            lerobot_episode=lerobot_episode,
            lerobot_episodes=[lerobot_episode],
            num_lerobot_episodes=1,
            num_state_frames_replayed=len(frame_indexes),
            num_camera_frames_rendered=len(camera_names) * len(frame_indexes),
            num_lerobot_frames_written=len(frame_indexes),
            preview_videos=preview_videos,
            contact_sheet=str(contact_sheet_path) if contact_sheet_path else None,
            randomization_config=randomization,
        )
    except Exception as exc:
        record.error_message = str(exc)
        record.traceback = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
    return record


def _robosuite_version() -> str:
    try:
        import robosuite
    except Exception:
        return "unavailable"
    return str(getattr(robosuite, "__version__", "unknown"))


def write_job_summary(
    output_dir: Path,
    *,
    job_name: str,
    started_at: datetime,
    ended_at: datetime,
    records: list[RenderRecord],
    candidates: list[RenderCandidate],
    num_workers: int,
    gpu_ids: list[int | None],
) -> Path:
    finished = [record for record in records if record.status in {"success", "skipped"}]
    failed = [record for record in records if record.status == "failed"]
    frames = sum(int(record.num_frames or 0) for record in finished)
    camera_frames_rendered = sum(
        int(record.num_camera_frames_rendered or 0) for record in finished
    )
    elapsed = max(1e-9, (ended_at - started_at).total_seconds())
    lerobot_task_datasets: dict[str, dict[str, Any]] = {}
    for record in finished:
        episodes = list(record.lerobot_episodes or [])
        if not episodes and record.lerobot_episode:
            episodes = [
                {
                    **record.lerobot_episode,
                    "dataset_path": record.output_path,
                }
            ]
        for episode in episodes:
            key = str(episode.get("dataset_path") or episode.get("lerobot_path") or "")
            if not key:
                continue
            item = lerobot_task_datasets.setdefault(
                key,
                {
                    "task_id": record.task_id,
                    "task_name": record.task_name,
                    "embodiment_id": record.embodiment_id,
                    "camera_group": episode.get("camera_group"),
                    "dataset_path": key,
                    "repo_id": episode.get("repo_id"),
                    "episodes_in_job": 0,
                    "frames_in_job": 0,
                    "episode_indices": [],
                    "total_episodes_after_write": episode.get("total_episodes_after_write"),
                    "total_frames_after_write": episode.get("total_frames_after_write"),
                },
            )
            if item.get("camera_group") != episode.get("camera_group"):
                raise ValueError(
                    f"Dataset {key} mixes camera groups: "
                    f"{item.get('camera_group')!r} versus {episode.get('camera_group')!r}."
                )
            item["episodes_in_job"] += 1
            item["frames_in_job"] += int(episode.get("frame_count") or 0)
            if episode.get("episode_index") is not None:
                item["episode_indices"].append(int(episode["episode_index"]))
            if episode.get("total_episodes_after_write") is not None:
                item["total_episodes_after_write"] = max(
                    int(item.get("total_episodes_after_write") or 0),
                    int(episode["total_episodes_after_write"]),
                )
            if episode.get("total_frames_after_write") is not None:
                item["total_frames_after_write"] = max(
                    int(item.get("total_frames_after_write") or 0),
                    int(episode["total_frames_after_write"]),
                )
    lerobot_task_dataset_list = sorted(
        lerobot_task_datasets.values(),
        key=lambda item: (
            int(item["task_id"] or -1),
            str(item.get("camera_group") or ""),
            str(item["dataset_path"]),
        ),
    )
    collection_manifest_path: Path | None = None
    if lerobot_task_dataset_list:
        collection_manifest_path = output_dir / "lerobot_task_collection.json"
    summary = {
        "job_id": f"{job_name}_{started_at.strftime('%Y%m%d_%H%M%S')}",
        "job_name": job_name,
        "start_time": started_at.isoformat(),
        "end_time": ended_at.isoformat(),
        "output_dataset_path": str(output_dir.resolve()),
        "dataset_layout": "lerobot_task_datasets",
        "num_trajectories_total": len(records),
        "num_trajectories_finished": len(finished),
        "num_trajectories_failed": len(failed),
        "num_frames_rendered": frames,
        "num_state_frames_replayed": sum(
            int(record.num_state_frames_replayed or 0) for record in finished
        ),
        "num_camera_frames_rendered": camera_frames_rendered,
        "num_lerobot_frames_written": sum(
            int(record.num_lerobot_frames_written or 0) for record in finished
        ),
        "num_lerobot_episodes": sum(
            len(record.lerobot_episodes or [])
            if record.lerobot_episodes is not None
            else int(record.lerobot_episode is not None)
            for record in finished
        ),
        "num_lerobot_observation_frames": sum(
            int(item["frames_in_job"]) for item in lerobot_task_dataset_list
        ),
        "num_visual_variants": len({record.variant_id for record in records}),
        "num_workers": int(num_workers),
        "gpu_ids": gpu_ids,
        "average_fps": frames / elapsed,
        "lerobot_layout": (
            "one_lerobot_dataset_per_task_and_camera_group"
            if any(item.get("camera_group") for item in lerobot_task_dataset_list)
            else "one_lerobot_dataset_per_task"
            if lerobot_task_datasets
            else None
        ),
        "lerobot_collection_manifest": str(collection_manifest_path.resolve()) if collection_manifest_path else None,
        "lerobot_task_datasets": lerobot_task_dataset_list,
        "candidates": [asdict(candidate) for candidate in candidates],
        "records": [asdict(record) for record in records],
        "failed_trajectory_list": [asdict(record) for record in failed],
    }
    if collection_manifest_path is not None:
        collection_payload = {
            "layout": summary["lerobot_layout"],
            "created_at": ended_at.isoformat(),
            "job_name": job_name,
            "job_id": summary["job_id"],
            "root": str(output_dir.resolve()),
            "num_task_datasets": len(lerobot_task_dataset_list),
            "num_episodes_in_job": sum(int(item["episodes_in_job"]) for item in lerobot_task_dataset_list),
            "num_frames_in_job": sum(int(item["frames_in_job"]) for item in lerobot_task_dataset_list),
            "task_datasets": lerobot_task_dataset_list,
        }
        collection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        collection_manifest_path.write_text(
            json.dumps(collection_payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
    path = output_dir / "job_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path
