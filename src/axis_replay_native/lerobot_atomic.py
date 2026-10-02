"""Two-phase LeRobot episode commits for concurrent render workers.

Each worker first writes a complete, single-episode LeRobot shard without
holding the per-task lock.  The short commit phase then remaps global indices,
moves the already encoded videos, and updates the canonical task dataset while
holding one exclusive lock.  A durable transaction marker makes an interrupted
commit fail closed instead of silently appending the same attempt twice.

This module intentionally targets the LeRobot v3 API shipped by LeRobot 0.4.4.
Private methods are checked explicitly before use so an incompatible upgrade
fails before mutating a dataset.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import os
import queue
import re
import shutil
import time
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Sequence

import numpy as np

ATOMIC_COMMIT_SCHEMA = "axis_lerobot_preencoded_commit_v1"
SHARD_SCHEMA = "axis_lerobot_single_episode_shard_v1"
SUPPORTED_LEROBOT_VERSION = "0.4.4"
LEROBOT_AUTO_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
_LEROBOT_THREADS_CONFIGURED = False
_HF_PARQUET_IN_MEMORY_CONFIGURED = False
_STREAMING_ENCODER_THREAD_PATCHED = False


@dataclass(frozen=True)
class EpisodeIdentity:
    task_id: int
    task_name: str
    attempt_id: int
    embodiment_id: str
    model_id: str
    variant_id: int
    render_fingerprint: str
    camera_group: str | None = None
    camera_replica_id: int | None = None

    def __post_init__(self) -> None:
        if (self.camera_group is None) != (self.camera_replica_id is None):
            raise ValueError("camera_group and camera_replica_id must be set together.")
        if self.camera_group is None:
            return
        if re.fullmatch(r"[a-z][a-z0-9_]*", str(self.camera_group)) is None:
            raise ValueError(
                "camera_group must match [a-z][a-z0-9_]* when camera multiplexing is enabled."
            )
        if int(self.camera_replica_id) < 0:
            raise ValueError("camera_replica_id must be non-negative.")

    @property
    def key(self) -> str:
        base = (
            f"task_{int(self.task_id)}_attempt_{int(self.attempt_id)}_"
            f"variant_{int(self.variant_id):03d}"
        )
        if self.camera_group is None:
            return base
        return (
            f"{base}_camera_{self.camera_group}_"
            f"replica_{int(self.camera_replica_id):03d}"
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path: Path, payload: Any) -> None:
    """Durably replace one JSON file without exposing a partial write."""

    path = Path(path)
    created_directories: list[Path] = []
    cursor = path.parent
    while not cursor.exists():
        created_directories.append(cursor)
        cursor = cursor.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default).encode("utf-8")
    with temp_path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    directories_to_sync = [path.parent, *(directory.parent for directory in created_directories)]
    seen: set[Path] = set()
    for directory in directories_to_sync:
        if directory in seen:
            continue
        seen.add(directory)
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


@contextmanager
def exclusive_lerobot_dataset_lock(dataset_root: Path) -> Iterator[None]:
    """Serialize only the canonical append phase for one task dataset."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    lock_dir = dataset_root.parent / ".axis_lerobot_write_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(str(dataset_root).encode("utf-8")).hexdigest()[:24] + ".lock"
    lock_path = lock_dir / lock_name
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_lerobot_attempt_lock(dataset_root: Path, identity: EpisodeIdentity) -> Iterator[None]:
    """Serialize retries of one attempt without blocking other attempts of the task."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    lock_dir = dataset_root.parent / ".axis_lerobot_attempt_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_key = f"{dataset_root}\0{identity.key}"
    lock_name = hashlib.sha256(lock_key.encode("utf-8")).hexdigest()[:24] + ".lock"
    with (lock_dir / lock_name).open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _configure_hf_parquet_in_memory() -> None:
    """Avoid persistent Arrow cache files for one-shot local Parquet reads."""
    global _HF_PARQUET_IN_MEMORY_CONFIGURED
    if _HF_PARQUET_IN_MEMORY_CONFIGURED:
        return

    import datasets

    original_from_parquet = datasets.Dataset.from_parquet

    def from_parquet_in_memory(path_or_paths: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            kwargs["keep_in_memory"] = True
            return original_from_parquet(path_or_paths, *args, **kwargs)

        features = kwargs.pop("features", None)
        split = kwargs.pop("split", None)
        columns = kwargs.pop("columns", None)
        filters = kwargs.pop("filters", None)
        kwargs.pop("keep_in_memory", None)
        if kwargs:
            kwargs["features"] = features
            kwargs["split"] = split
            kwargs["columns"] = columns
            kwargs["filters"] = filters
            kwargs["keep_in_memory"] = True
            return original_from_parquet(path_or_paths, **kwargs)

        import pandas as pd

        paths = path_or_paths if isinstance(path_or_paths, (list, tuple)) else [path_or_paths]
        frames = [pd.read_parquet(path, columns=columns, filters=filters) for path in paths]
        frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
        return datasets.Dataset.from_pandas(
            frame,
            features=features,
            split=split,
            preserve_index=False,
        )

    datasets.Dataset.from_parquet = staticmethod(from_parquet_in_memory)
    _HF_PARQUET_IN_MEMORY_CONFIGURED = True


def _load_lerobot_api() -> tuple[type, dict[str, Any]]:
    global _LEROBOT_THREADS_CONFIGURED
    raw_limit = os.environ.get("AXIS_RENDER_COMPUTE_THREADS")
    if raw_limit is not None and not _LEROBOT_THREADS_CONFIGURED:
        thread_limit = int(raw_limit)
        if thread_limit <= 0:
            raise ValueError("AXIS_RENDER_COMPUTE_THREADS must be positive when provided.")
        import cv2
        import torch

        torch.set_num_threads(thread_limit)
        torch.set_num_interop_threads(thread_limit)
        cv2.setNumThreads(thread_limit)
        _LEROBOT_THREADS_CONFIGURED = True
    try:
        import lerobot.datasets.lerobot_dataset as module
    except ImportError as exc:
        raise ImportError("Install LeRobot 0.4.4 before writing AXIS render datasets.") from exc
    _configure_hf_parquet_in_memory()

    try:
        installed_version = package_version("lerobot")
    except PackageNotFoundError as exc:
        raise ImportError("LeRobot package metadata is unavailable.") from exc
    if installed_version != SUPPORTED_LEROBOT_VERSION:
        raise RuntimeError(
            "Unsupported LeRobot version for atomic AXIS commits: "
            f"expected={SUPPORTED_LEROBOT_VERSION}, installed={installed_version}."
        )
    if str(getattr(module, "CODEBASE_VERSION", "")) != "v3.0":
        raise RuntimeError(
            "Unsupported LeRobot dataset codebase version; expected v3.0, "
            f"got {getattr(module, 'CODEBASE_VERSION', None)!r}."
        )

    dataset_class = module.LeRobotDataset
    save_params = inspect.signature(dataset_class.save_episode).parameters
    if list(save_params) != ["self", "episode_data", "parallel_encoding"]:
        raise RuntimeError(
            f"Unsupported LeRobot save_episode API: {list(save_params)}."
        )
    data_params = inspect.signature(dataset_class._save_episode_data).parameters
    if list(data_params) != ["self", "episode_buffer"]:
        raise RuntimeError(f"Unsupported LeRobot _save_episode_data API: {list(data_params)}.")
    create_params = inspect.signature(dataset_class.create).parameters
    required_streaming_params = {
        "streaming_encoding",
        "encoder_queue_maxsize",
        "encoder_threads",
    }
    missing_streaming_params = required_streaming_params - set(create_params)
    if missing_streaming_params:
        raise RuntimeError(
            "Pinned LeRobot writer is missing streaming H.264 controls: "
            f"{sorted(missing_streaming_params)}."
        )
    required_module_symbols = ("aggregate_stats",)
    for symbol in required_module_symbols:
        if not hasattr(module, symbol):
            raise RuntimeError(f"Unsupported LeRobot API; missing {symbol}.")
    return dataset_class, {"aggregate_stats": module.aggregate_stats}


def _feature_shape(feature: Any) -> tuple[int, ...]:
    shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", ())
    if shape is None:
        return ()
    return tuple(int(item) for item in shape)


def _feature_dtype(feature: Any) -> str:
    if isinstance(feature, dict):
        return str(feature.get("dtype"))
    return str(getattr(feature, "dtype", None))


def _feature_names(feature: Any) -> Any:
    names = feature.get("names") if isinstance(feature, dict) else getattr(feature, "names", None)
    if names is None:
        return None
    return list(names)


def assert_features_match(dataset: Any, expected_features: dict[str, Any]) -> None:
    actual_features = getattr(dataset, "features", None)
    if actual_features is None:
        actual_features = getattr(getattr(dataset, "meta", None), "features", None)
    if actual_features is None:
        raise RuntimeError("LeRobot dataset does not expose features for schema validation.")
    actual_keys = {str(key) for key in actual_features}
    expected_keys = {str(key) for key in expected_features} | LEROBOT_AUTO_FEATURE_KEYS
    if actual_keys != expected_keys:
        raise ValueError(
            "LeRobot feature key mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, unexpected={sorted(actual_keys - expected_keys)}."
        )
    for key, expected in expected_features.items():
        actual = actual_features.get(key)
        if actual is None:
            raise ValueError(f"LeRobot dataset is missing feature {key!r}.")
        if _feature_dtype(actual) != _feature_dtype(expected):
            raise ValueError(
                f"LeRobot feature {key!r} dtype mismatch: "
                f"expected {_feature_dtype(expected)!r}, got {_feature_dtype(actual)!r}."
            )
        if _feature_shape(actual) != _feature_shape(expected):
            raise ValueError(
                f"LeRobot feature {key!r} shape mismatch: "
                f"expected {_feature_shape(expected)}, got {_feature_shape(actual)}."
            )
        if _feature_names(actual) != _feature_names(expected):
            raise ValueError(
                f"LeRobot feature {key!r} names mismatch: "
                f"expected {_feature_names(expected)!r}, got {_feature_names(actual)!r}."
            )


def _assert_dataset_config(
    dataset: Any,
    *,
    repo_id: str,
    fps: int,
    robot_type: str,
) -> None:
    info = dataset.meta.info
    observed = {
        "repo_id": str(dataset.repo_id),
        "fps": int(info.get("fps", -1)),
        "robot_type": str(info.get("robot_type")),
        "codebase_version": str(info.get("codebase_version")),
    }
    expected = {
        "repo_id": str(repo_id),
        "fps": int(fps),
        "robot_type": str(robot_type),
        "codebase_version": "v3.0",
    }
    if observed != expected:
        raise ValueError(f"LeRobot dataset configuration mismatch: expected={expected}, observed={observed}.")


def _assert_video_info_match(source: Any, target: Any) -> None:
    fields = (
        "video.height",
        "video.width",
        "video.codec",
        "video.pix_fmt",
        "video.is_depth_map",
        "video.fps",
        "video.channels",
        "has_audio",
    )
    for video_key in source.meta.video_keys:
        source_info = ((source.meta.info.get("features") or {}).get(video_key) or {}).get("info") or {}
        target_info = ((target.meta.info.get("features") or {}).get(video_key) or {}).get("info") or {}
        source_values = {field: source_info.get(field) for field in fields}
        target_values = {field: target_info.get(field) for field in fields}
        if source_values != target_values:
            raise ValueError(
                f"LeRobot video metadata mismatch for {video_key!r}: "
                f"source={source_values}, target={target_values}."
            )


def _open_dataset(dataset_class: type, *, root: Path, repo_id: str, vcodec: str | None) -> Any:
    os.environ["HF_LEROBOT_HOME"] = str(Path(root).resolve().parent)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "root": root,
        "download_videos": False,
    }
    if vcodec:
        kwargs["vcodec"] = vcodec
    return dataset_class(**kwargs)


def _load_stats_json(root: Path) -> dict[str, dict[str, np.ndarray]]:
    payload = _load_json(Path(root) / "meta" / "stats.json")
    return {
        str(feature): {str(name): np.asarray(value) for name, value in stats.items()}
        for feature, stats in payload.items()
    }


def _assert_info_features_match(info: dict[str, Any], expected_features: dict[str, Any]) -> None:
    actual_features = info.get("features") or {}
    actual_keys = {str(key) for key in actual_features}
    expected_keys = {str(key) for key in expected_features} | LEROBOT_AUTO_FEATURE_KEYS
    if actual_keys != expected_keys:
        raise ValueError(
            "LeRobot feature key mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, unexpected={sorted(actual_keys - expected_keys)}."
        )
    for key, expected in expected_features.items():
        actual = actual_features.get(key)
        if actual is None:
            raise ValueError(f"LeRobot dataset is missing feature {key!r}.")
        if _feature_dtype(actual) != _feature_dtype(expected):
            raise ValueError(
                f"LeRobot feature {key!r} dtype mismatch: "
                f"expected {_feature_dtype(expected)!r}, got {_feature_dtype(actual)!r}."
            )
        if _feature_shape(actual) != _feature_shape(expected):
            raise ValueError(
                f"LeRobot feature {key!r} shape mismatch: "
                f"expected {_feature_shape(expected)}, got {_feature_shape(actual)}."
            )
        if _feature_names(actual) != _feature_names(expected):
            raise ValueError(
                f"LeRobot feature {key!r} names mismatch: "
                f"expected {_feature_names(expected)!r}, got {_feature_names(actual)!r}."
            )


def _assert_info_config(
    info: dict[str, Any],
    *,
    fps: int,
    robot_type: str,
) -> None:
    observed = {
        "fps": int(info.get("fps", -1)),
        "robot_type": str(info.get("robot_type")),
        "codebase_version": str(info.get("codebase_version")),
    }
    expected = {
        "fps": int(fps),
        "robot_type": str(robot_type),
        "codebase_version": "v3.0",
    }
    if observed != expected:
        raise ValueError(f"LeRobot dataset configuration mismatch: expected={expected}, observed={observed}.")


def _video_keys_from_info(info: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(key)
        for key, feature in (info.get("features") or {}).items()
        if str((feature or {}).get("dtype")) == "video"
    )


def _assert_video_info_dict_match(source_info: dict[str, Any], target_info: dict[str, Any]) -> None:
    fields = (
        "video.height",
        "video.width",
        "video.codec",
        "video.pix_fmt",
        "video.is_depth_map",
        "video.fps",
        "video.channels",
        "has_audio",
    )
    source_keys = _video_keys_from_info(source_info)
    target_keys = _video_keys_from_info(target_info)
    if source_keys != target_keys:
        raise ValueError(f"LeRobot video feature mismatch: source={source_keys}, target={target_keys}.")
    for video_key in source_keys:
        source_values = {
            field: (((source_info.get("features") or {}).get(video_key) or {}).get("info") or {}).get(field)
            for field in fields
        }
        target_values = {
            field: (((target_info.get("features") or {}).get(video_key) or {}).get("info") or {}).get(field)
            for field in fields
        }
        if source_values != target_values:
            raise ValueError(
                f"LeRobot video metadata mismatch for {video_key!r}: "
                f"source={source_values}, target={target_values}."
            )


def _load_single_episode_shard(
    *,
    shard_root: Path,
    manifest: dict[str, Any],
    expected_features: dict[str, Any],
    fps: int,
    robot_type: str,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    shard_root = Path(shard_root).resolve()
    info = _load_json(shard_root / "meta" / "info.json")
    _assert_info_features_match(info, expected_features)
    _assert_info_config(info, fps=fps, robot_type=robot_type)
    frame_count = int(manifest["frame_count"])
    if (int(info.get("total_episodes", -1)), int(info.get("total_frames", -1))) != (1, frame_count):
        raise RuntimeError(f"Prepared shard changed after validation: {shard_root}")

    episode_paths = sorted((shard_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if len(episode_paths) != 1:
        raise RuntimeError(f"Prepared shard must contain one episode parquet: {shard_root}")
    episode_table = pq.read_table(episode_paths[0])
    if episode_table.num_rows != 1:
        raise RuntimeError(f"Prepared shard episode parquet must contain one row: {episode_paths[0]}")
    episode = episode_table.to_pylist()[0]
    data_path = shard_root / str(info["data_path"]).format(
        chunk_index=int(episode["data/chunk_index"]),
        file_index=int(episode["data/file_index"]),
    )
    data_table = pq.read_table(data_path)
    if data_table.num_rows != frame_count:
        raise RuntimeError(f"Prepared shard data row count mismatch: {data_path}")

    video_durations: dict[str, float] = {}
    for video_key in _video_keys_from_info(info):
        video_path = shard_root / str(info["video_path"]).format(
            video_key=video_key,
            chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
            file_index=int(episode[f"videos/{video_key}/file_index"]),
        )
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError(f"Prepared video disappeared before commit: {video_path}")
        declared_path = (manifest.get("video_paths") or {}).get(video_key)
        if declared_path != str(video_path.relative_to(shard_root)):
            raise ValueError(f"Prepared video path disagrees with shard manifest for {video_key!r}.")
        video_durations[video_key] = float(
            episode[f"videos/{video_key}/to_timestamp"]
            - episode[f"videos/{video_key}/from_timestamp"]
        )
    return {
        "info": info,
        "stats": _load_stats_json(shard_root),
        "episode": episode,
        "episode_schema": episode_table.schema,
        "data_table": data_table,
        "video_durations": video_durations,
    }


def _shard_manifest_path(root: Path) -> Path:
    return Path(root) / "meta" / "axis_episode_shard.json"


def _attempt_commit_path(root: Path, identity: EpisodeIdentity) -> Path:
    return Path(root) / "meta" / "axis_attempt_commits" / f"{identity.key}.json"


def _dataset_token(root: Path) -> str:
    resolved = str(Path(root).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


def _transaction_base(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    return root.parent / ".axis_lerobot_transactions" / f"{root.name}-{_dataset_token(root)}"


def _transaction_dir(root: Path, identity: EpisodeIdentity) -> Path:
    return _transaction_base(root) / identity.key


def _transaction_path(root: Path, identity: EpisodeIdentity) -> Path:
    return _transaction_dir(root, identity) / "transaction.json"


def _commit_state_path(root: Path) -> Path:
    return Path(root) / "meta" / "axis_commit_state.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _validate_identity_payload(payload: dict[str, Any], identity: EpisodeIdentity, *, source: Path) -> None:
    expected = _identity_payload(identity)
    actual = payload.get("identity")
    if actual != expected:
        raise ValueError(f"Episode identity mismatch in {source}: expected={expected!r}, actual={actual!r}")


def _identity_payload(identity: EpisodeIdentity) -> dict[str, Any]:
    payload = {
        "task_id": int(identity.task_id),
        "task_name": identity.task_name,
        "attempt_id": int(identity.attempt_id),
        "embodiment_id": identity.embodiment_id,
        "model_id": identity.model_id,
        "variant_id": int(identity.variant_id),
        "render_fingerprint": identity.render_fingerprint,
    }
    if identity.camera_group is not None:
        payload["camera_group"] = identity.camera_group
        payload["camera_replica_id"] = int(identity.camera_replica_id)
    return payload


def _axis_episode_path(root: Path, identity: EpisodeIdentity, episode_index: int) -> Path:
    return Path(root) / "meta" / "axis_episodes" / f"episode_{int(episode_index):06d}_{identity.key}.json"


def _load_committed_attempt_unlocked(
    dataset_root: Path,
    identity: EpisodeIdentity,
) -> dict[str, Any] | None:
    dataset_root = Path(dataset_root)
    marker_path = _attempt_commit_path(dataset_root, identity)
    if marker_path.is_file():
        payload = _load_json(marker_path)
        _validate_identity_payload(payload, identity, source=marker_path)
        episode = payload.get("episode")
        if not isinstance(episode, dict):
            raise ValueError(f"Attempt commit is missing an episode record: {marker_path}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            sidecar_path = _axis_episode_path(dataset_root, identity, int(episode["episode_index"]))
            sidecar = _load_json(sidecar_path)
            metadata = sidecar.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Attempt commit cannot recover worker metadata: {marker_path}")
        return {"episode": episode, "metadata": metadata}

    legacy_dir = dataset_root / "meta" / "axis_episodes"
    pattern = f"episode_*_{identity.key}.json"
    matches = sorted(legacy_dir.glob(pattern)) if legacy_dir.is_dir() else []
    if len(matches) > 1:
        raise RuntimeError(f"Multiple legacy AXIS episode sidecars match {identity.key}: {matches}")
    if not matches:
        return None
    payload = _load_json(matches[0])
    metadata = payload.get("metadata") or {}
    stored_fingerprint = (metadata.get("render") or {}).get("render_fingerprint")
    if stored_fingerprint != identity.render_fingerprint:
        raise ValueError(
            f"Legacy commit fingerprint mismatch for {identity.key}: "
            f"stored={stored_fingerprint!r}, current={identity.render_fingerprint!r}."
        )
    episode = payload.get("episode")
    if not isinstance(episode, dict) or not isinstance(metadata, dict):
        raise ValueError(f"Legacy AXIS sidecar is incomplete: {matches[0]}")
    return {"episode": episode, "metadata": metadata}


def load_committed_attempt(
    dataset_root: Path,
    identity: EpisodeIdentity,
) -> dict[str, Any] | None:
    """Recover pending WAL state, then return a durable attempt commit if present."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    with exclusive_lerobot_dataset_lock(dataset_root):
        _recover_pending_transactions_unlocked(dataset_root)
        return _load_committed_attempt_unlocked(dataset_root, identity)


def load_committed_episode(dataset_root: Path, identity: EpisodeIdentity) -> dict[str, Any] | None:
    """Compatibility wrapper returning only the episode portion of a durable commit."""

    committed = load_committed_attempt(dataset_root, identity)
    return None if committed is None else committed["episode"]


def _lossless_streaming_feed_frame(encoder: Any, video_key: str, image: np.ndarray) -> None:
    """Feed one frame without LeRobot's real-time frame-dropping policy.

    Formal replay is an offline data-production path, so temporary encoder
    backpressure must slow the renderer instead of silently shortening a video.
    Keep checking encoder health while waiting so a crashed encoder still fails
    promptly rather than leaving the worker blocked forever.
    """

    if not bool(getattr(encoder, "_episode_active", False)):
        raise RuntimeError("No active episode. Call start_episode() first.")

    frame = np.ascontiguousarray(image)
    while True:
        thread = encoder._threads[video_key]
        if not thread.is_alive():
            try:
                status, message = encoder._result_queues[video_key].get_nowait()
            except queue.Empty:
                status, message = None, None
            if status == "error":
                raise RuntimeError(f"Encoder thread for {video_key} crashed: {message}")
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")
        try:
            encoder._frame_queues[video_key].put(frame, timeout=0.5)
            return
        except queue.Full:
            continue


def _patch_streaming_encoder_thread() -> None:
    """Use PyAV's ndarray path and avoid the NumPy -> PIL -> PyAV copy."""

    global _STREAMING_ENCODER_THREAD_PATCHED
    if _STREAMING_ENCODER_THREAD_PATCHED:
        return

    import av
    import contextlib
    import logging
    from fractions import Fraction

    import lerobot.datasets.video_utils as video_utils
    from lerobot.datasets.compute_stats import RunningQuantileStats, auto_downsample_height_width

    original_thread_class = video_utils._CameraEncoderThread

    class AxisCameraEncoderThread(original_thread_class):
        def run(self) -> None:
            container = None
            output_stream = None
            stats_tracker = RunningQuantileStats()
            frame_count = 0
            try:
                logging.getLogger("libav").setLevel(av.logging.WARNING)
                while True:
                    try:
                        frame_data = self.frame_queue.get(timeout=1)
                    except queue.Empty:
                        if self.stop_event.is_set():
                            break
                        continue
                    if frame_data is None:
                        break
                    if isinstance(frame_data, np.ndarray):
                        if frame_data.ndim == 3 and frame_data.shape[0] == 3:
                            frame_data = frame_data.transpose(1, 2, 0)
                        if frame_data.dtype != np.uint8:
                            frame_data = (frame_data * 255).astype(np.uint8)
                    frame_data = np.ascontiguousarray(frame_data)

                    if container is None:
                        height, width = frame_data.shape[:2]
                        video_options = video_utils._get_codec_options(
                            self.vcodec,
                            self.g,
                            self.crf,
                            self.preset,
                        )
                        if self.encoder_threads is not None:
                            if self.vcodec == "libsvtav1":
                                lp_param = f"lp={self.encoder_threads}"
                                if "svtav1-params" in video_options:
                                    video_options["svtav1-params"] += f":{lp_param}"
                                else:
                                    video_options["svtav1-params"] = lp_param
                            else:
                                video_options["threads"] = str(self.encoder_threads)
                        Path(self.video_path).parent.mkdir(parents=True, exist_ok=True)
                        container = av.open(str(self.video_path), "w")
                        output_stream = container.add_stream(
                            self.vcodec,
                            self.fps,
                            options=video_options,
                        )
                        output_stream.pix_fmt = self.pix_fmt
                        output_stream.width = width
                        output_stream.height = height
                        output_stream.time_base = Fraction(1, self.fps)

                    video_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                    video_frame.pts = frame_count
                    video_frame.time_base = Fraction(1, self.fps)
                    packet = output_stream.encode(video_frame)
                    if packet:
                        container.mux(packet)

                    img_chw = frame_data.transpose(2, 0, 1)
                    img_downsampled = auto_downsample_height_width(img_chw)
                    channels = img_downsampled.shape[0]
                    stats_tracker.update(img_downsampled.transpose(1, 2, 0).reshape(-1, channels))
                    frame_count += 1

                if output_stream is not None:
                    packet = output_stream.encode()
                    if packet:
                        container.mux(packet)
                if container is not None:
                    container.close()
                av.logging.restore_default_callback()
                stats = stats_tracker.get_statistics() if frame_count >= 2 else None
                self.result_queue.put(("ok", stats))
            except Exception as exc:
                logging.error("Encoder thread error: %s", exc)
                if container is not None:
                    with contextlib.suppress(Exception):
                        container.close()
                self.result_queue.put(("error", str(exc)))

    AxisCameraEncoderThread.__name__ = "AxisCameraEncoderThread"
    video_utils._CameraEncoderThread = AxisCameraEncoderThread
    _STREAMING_ENCODER_THREAD_PATCHED = True


def _configure_lossless_streaming_encoder(dataset: Any) -> None:
    _patch_streaming_encoder_thread()
    encoder = getattr(dataset, "_streaming_encoder", None)
    if encoder is None:
        raise RuntimeError("LeRobot streaming encoder was not initialized.")
    encoder.feed_frame = MethodType(_lossless_streaming_feed_frame, encoder)


def _validate_shard_manifest(
    manifest: dict[str, Any],
    *,
    path: Path,
    identity: EpisodeIdentity,
    repo_id: str,
    fps: int,
    robot_type: str,
    vcodec: str | None,
) -> None:
    if manifest.get("schema") != SHARD_SCHEMA:
        raise ValueError(f"Unsupported episode shard schema: {path}")
    _validate_identity_payload(manifest, identity, source=path)
    expected = {
        "repo_id": str(repo_id),
        "fps": int(fps),
        "robot_type": str(robot_type),
        "vcodec": vcodec,
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Episode shard configuration mismatch: expected={expected}, actual={actual}.")


class StreamingEpisodeShardWriter:
    """Build one crash-safe LeRobot shard while frames are being rendered.

    LeRobot's pinned v3 writer owns one H.264 encoder thread per video feature.
    Frames are fed directly to those encoders; no PNG sequence or full-episode
    RGB array is materialized. The shard remains ``.partial`` until every
    encoder and metadata writer has finished and the result has been reopened
    and validated.
    """

    def __init__(
        self,
        *,
        partial_root: Path,
        ready_root: Path,
        dataset_root: Path,
        repo_id: str,
        features: dict[str, Any],
        task_name: str,
        identity: EpisodeIdentity,
        fps: int,
        robot_type: str,
        vcodec: str | None,
        encoder_queue_maxsize: int,
        encoder_threads: int | None,
    ) -> None:
        if int(encoder_queue_maxsize) <= 0:
            raise ValueError("encoder_queue_maxsize must be positive.")
        if encoder_threads is not None and int(encoder_threads) <= 0:
            raise ValueError("encoder_threads must be positive when provided.")
        self.partial_root = Path(partial_root).resolve()
        self.ready_root = Path(ready_root).resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.repo_id = str(repo_id)
        self.features = features
        self.task_name = str(task_name)
        self.identity = identity
        self.fps = int(fps)
        self.robot_type = str(robot_type)
        self.vcodec = vcodec
        self.encoder_queue_maxsize = int(encoder_queue_maxsize)
        self.encoder_threads = None if encoder_threads is None else int(encoder_threads)
        self._attempt_lock: Any = None
        self._dataset: Any = None
        self._existing_result: dict[str, Any] | None = None
        self._ready_manifest: dict[str, Any] | None = None
        self._frame_count = 0
        self._encode_started = 0.0
        self._finished = False

    @property
    def needs_frames(self) -> bool:
        return self._dataset is not None

    @property
    def frame_count(self) -> int:
        return int(self._frame_count)

    def __enter__(self) -> "StreamingEpisodeShardWriter":
        dataset_class, _ = _load_lerobot_api()
        self._attempt_lock = exclusive_lerobot_attempt_lock(self.dataset_root, self.identity)
        self._attempt_lock.__enter__()
        try:
            with exclusive_lerobot_dataset_lock(self.dataset_root):
                _recover_pending_transactions_unlocked(self.dataset_root)
                committed = _load_committed_attempt_unlocked(self.dataset_root, self.identity)
            if committed is not None:
                self._existing_result = {
                    "schema": SHARD_SCHEMA,
                    "identity": _identity_payload(self.identity),
                    "already_committed": True,
                    "episode": committed["episode"],
                }
                return self
            if self.ready_root.exists():
                manifest_path = _shard_manifest_path(self.ready_root)
                manifest = _load_json(manifest_path)
                _validate_shard_manifest(
                    manifest,
                    path=manifest_path,
                    identity=self.identity,
                    repo_id=self.repo_id,
                    fps=self.fps,
                    robot_type=self.robot_type,
                    vcodec=self.vcodec,
                )
                self._ready_manifest = manifest
                return self
            if self.partial_root.exists():
                shutil.rmtree(self.partial_root)
            self.partial_root.parent.mkdir(parents=True, exist_ok=True)
            os.environ["HF_LEROBOT_HOME"] = str(self.partial_root.parent)
            kwargs: dict[str, Any] = {
                "repo_id": self.repo_id,
                "root": self.partial_root,
                "robot_type": self.robot_type,
                "fps": self.fps,
                "features": self.features,
                "use_videos": True,
                "image_writer_threads": 0,
                "image_writer_processes": 0,
                "streaming_encoding": True,
                "encoder_queue_maxsize": self.encoder_queue_maxsize,
                "encoder_threads": self.encoder_threads,
            }
            if self.vcodec:
                kwargs["vcodec"] = self.vcodec
            self._dataset = dataset_class.create(**kwargs)
            _configure_lossless_streaming_encoder(self._dataset)
            self._encode_started = time.perf_counter()
            return self
        except BaseException:
            self.abort()
            raise

    def add_frame(self, frame: dict[str, Any]) -> None:
        if self._dataset is None:
            raise RuntimeError("This shard is already prepared or committed and does not accept frames.")
        payload = dict(frame)
        payload["task"] = self.task_name
        self._dataset.add_frame(payload)
        self._frame_count += 1

    def finish(self) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("Streaming episode shard was already finished.")
        if self._existing_result is not None:
            self._finished = True
            self._release_lock()
            return self._existing_result
        if self._ready_manifest is not None:
            self._finished = True
            self._release_lock()
            return self._ready_manifest
        if self._dataset is None or self._frame_count <= 0:
            raise RuntimeError("Streaming episode shard requires at least one frame.")

        dataset = self._dataset
        dataset.save_episode(parallel_encoding=False)
        dropped_frames = dict(getattr(dataset._streaming_encoder, "_dropped_frames", {}) or {})
        if any(int(count) > 0 for count in dropped_frames.values()):
            raise RuntimeError(
                "Streaming H.264 encoder dropped frames; refusing to publish shard: "
                f"{dropped_frames}"
            )
        dataset.finalize()
        encode_seconds = time.perf_counter() - self._encode_started
        self._dataset = None
        manifest = _validate_and_publish_episode_shard(
            partial_root=self.partial_root,
            ready_root=self.ready_root,
            repo_id=self.repo_id,
            features=self.features,
            identity=self.identity,
            fps=self.fps,
            robot_type=self.robot_type,
            vcodec=self.vcodec,
            expected_frame_count=self._frame_count,
            encode_seconds=encode_seconds,
            encoding_mode="streaming_h264",
        )
        self._finished = True
        self._release_lock()
        return manifest

    def abort(self) -> None:
        dataset = self._dataset
        self._dataset = None
        if dataset is not None:
            encoder = getattr(dataset, "_streaming_encoder", None)
            if encoder is not None:
                encoder.cancel_episode()
            dataset.finalize()
        if self.partial_root.exists():
            shutil.rmtree(self.partial_root)
        self._release_lock()

    def _release_lock(self) -> None:
        lock = self._attempt_lock
        self._attempt_lock = None
        if lock is not None:
            lock.__exit__(None, None, None)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is not None or not self._finished:
            self.abort()
        else:
            self._release_lock()
        return False


def _validate_and_publish_episode_shard(
    *,
    partial_root: Path,
    ready_root: Path,
    repo_id: str,
    features: dict[str, Any],
    identity: EpisodeIdentity,
    fps: int,
    robot_type: str,
    vcodec: str | None,
    expected_frame_count: int,
    encode_seconds: float,
    encoding_mode: str,
) -> dict[str, Any]:
    dataset_class, _ = _load_lerobot_api()
    reopened = _open_dataset(dataset_class, root=partial_root, repo_id=repo_id, vcodec=vcodec)
    assert_features_match(reopened, features)
    _assert_dataset_config(reopened, repo_id=repo_id, fps=fps, robot_type=robot_type)
    if int(reopened.meta.total_episodes) != 1:
        raise RuntimeError(f"Prepared shard must contain one episode, got {reopened.meta.total_episodes}.")
    if int(reopened.meta.total_frames) != int(expected_frame_count):
        raise RuntimeError(
            "Prepared shard frame count mismatch: "
            f"expected={expected_frame_count}, got={reopened.meta.total_frames}."
        )
    episode_meta = reopened.meta.episodes[0]
    video_paths: dict[str, str] = {}
    expected_duration = float(expected_frame_count) / float(fps)
    for video_key in reopened.meta.video_keys:
        video_path = partial_root / reopened.meta.video_path.format(
            video_key=video_key,
            chunk_index=int(episode_meta[f"videos/{video_key}/chunk_index"]),
            file_index=int(episode_meta[f"videos/{video_key}/file_index"]),
        )
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise RuntimeError(f"Prepared shard video is missing or empty: {video_path}")
        duration = float(
            episode_meta[f"videos/{video_key}/to_timestamp"]
            - episode_meta[f"videos/{video_key}/from_timestamp"]
        )
        if abs(duration - expected_duration) > 0.5 / float(fps):
            raise RuntimeError(
                f"Prepared video duration mismatch for {video_key!r}: "
                f"expected={expected_duration}, observed={duration}."
            )
        video_paths[video_key] = str(video_path.relative_to(partial_root))

    manifest = {
        "schema": SHARD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": _identity_payload(identity),
        "repo_id": str(repo_id),
        "fps": int(fps),
        "robot_type": str(robot_type),
        "vcodec": vcodec,
        "frame_count": int(expected_frame_count),
        "video_paths": video_paths,
        "encode_seconds": float(encode_seconds),
        "encoding_mode": str(encoding_mode),
    }
    atomic_write_json(_shard_manifest_path(partial_root), manifest)
    os.replace(partial_root, ready_root)
    _fsync_directory(ready_root.parent)
    return manifest


def prepare_episode_shard(
    *,
    partial_root: Path,
    ready_root: Path,
    dataset_root: Path,
    repo_id: str,
    features: dict[str, Any],
    frames: Sequence[dict[str, Any]],
    task_name: str,
    identity: EpisodeIdentity,
    fps: int,
    robot_type: str,
    vcodec: str | None,
    image_writer_threads: int,
    image_writer_processes: int,
    encoder_queue_maxsize: int = 128,
    encoder_threads: int | None = 1,
) -> dict[str, Any]:
    """Stream an in-memory frame sequence into one validated atomic shard.

    The image-writer arguments remain in the public signature for controller
    compatibility, but streaming mode intentionally never creates PNG workers.
    """

    del image_writer_threads, image_writer_processes
    with StreamingEpisodeShardWriter(
        partial_root=partial_root,
        ready_root=ready_root,
        dataset_root=dataset_root,
        repo_id=repo_id,
        features=features,
        task_name=task_name,
        identity=identity,
        fps=fps,
        robot_type=robot_type,
        vcodec=vcodec,
        encoder_queue_maxsize=encoder_queue_maxsize,
        encoder_threads=encoder_threads,
    ) as writer:
        if writer.needs_frames:
            for frame in frames:
                writer.add_frame(frame)
        return writer.finish()


def _shift_stats(
    stats: dict[str, dict[str, np.ndarray]],
    *,
    episode_index: int,
    frame_start: int,
    task_index: int,
) -> dict[str, dict[str, np.ndarray]]:
    shifted = deepcopy(stats)
    location_stats = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
    for stat_name in location_stats:
        if stat_name in shifted.get("episode_index", {}):
            shifted["episode_index"][stat_name] = (
                np.asarray(shifted["episode_index"][stat_name]) + int(episode_index)
            )
        if stat_name in shifted.get("index", {}):
            shifted["index"][stat_name] = np.asarray(shifted["index"][stat_name]) + int(frame_start)
        if stat_name in shifted.get("task_index", {}):
            source = np.asarray(shifted["task_index"][stat_name])
            shifted["task_index"][stat_name] = np.full_like(source, int(task_index))
    if "std" in shifted.get("task_index", {}):
        shifted["task_index"]["std"] = np.zeros_like(shifted["task_index"]["std"])
    return shifted


def _axis_episode_record(
    *,
    dataset_root: Path,
    repo_id: str,
    identity: EpisodeIdentity,
    episode_index: int,
    frame_count: int,
    fps: int,
    robot_type: str,
    camera_keys: dict[str, str],
    policy_state_dim: int,
    action_dim: int,
    ee_state_dim: int | None,
    policy_state_metadata: dict[str, Any],
    nq: int,
    nv: int,
    nu: int,
    total_episodes: int,
    total_frames: int,
    encode_seconds: float,
    stage_seconds: float,
    lock_wait_seconds: float,
    lock_work_seconds: float,
    commit_seconds: float,
) -> dict[str, Any]:
    record = {
        "lerobot_path": str(dataset_root),
        "repo_id": repo_id,
        "episode_index": int(episode_index),
        "frame_count": int(frame_count),
        "task_id": int(identity.task_id),
        "task_name": identity.task_name,
        "attempt_id": int(identity.attempt_id),
        "embodiment_id": identity.embodiment_id,
        "model_id": identity.model_id,
        "variant_id": int(identity.variant_id),
        "fps": int(fps),
        "robot_type": robot_type,
        "camera_keys": camera_keys,
        "schema": "reference_policy_fields_v2",
        "image_cache_policy": "encoded_in_per_attempt_shard",
        "image_cache_removed": True,
        "policy_state_dim": int(policy_state_dim),
        "action_dim": int(action_dim),
        "ee_state_dim": int(ee_state_dim) if ee_state_dim is not None else None,
        "policy_state_metadata": policy_state_metadata,
        "nq": int(nq),
        "nv": int(nv),
        "nu": int(nu),
        "total_episodes_after_write": int(total_episodes),
        "total_frames_after_write": int(total_frames),
        "commit": {
            "schema": ATOMIC_COMMIT_SCHEMA,
            "video_encoding_outside_task_lock": True,
            "encode_seconds": float(encode_seconds),
            "stage_seconds": float(stage_seconds),
            "lock_wait_seconds": float(lock_wait_seconds),
            "lock_work_seconds": float(lock_work_seconds),
            "commit_seconds": float(commit_seconds),
        },
    }
    if identity.camera_group is not None:
        record["camera_group"] = identity.camera_group
        record["camera_replica_id"] = int(identity.camera_replica_id)
    return record


def _write_axis_sidecars(
    *,
    dataset_root: Path,
    identity: EpisodeIdentity,
    metadata: dict[str, Any],
    episode_record: dict[str, Any],
) -> None:
    episode_path = _axis_episode_path(dataset_root, identity, int(episode_record["episode_index"]))
    atomic_write_json(episode_path, {"episode": episode_record, "metadata": metadata})
    camera_features = deepcopy((metadata.get("lerobot") or {}).get("camera_features", {}))
    if identity.camera_group == "randomized":
        for feature in camera_features.values():
            physical_name = feature.pop("source_camera_name", None)
            feature["logical_camera_name"] = (
                feature.get("logical_camera_name") or physical_name
            )
            feature["source_camera_name_policy"] = (
                "physical camera name is per episode in meta/axis_episodes"
            )
    task_payload = {
        "task_id": int(identity.task_id),
        "task_name": identity.task_name,
        "embodiment_id": identity.embodiment_id,
        "model_id": identity.model_id,
        "repo_id": episode_record.get("repo_id"),
        "dataset_path": episode_record.get("lerobot_path"),
        "renderer_backend": metadata["renderer_backend"],
        "state_feature": "observation.state",
        "state_semantics": "actuated_joint_qpos_in_actuator_order",
        "action_feature": "action",
        "action_semantics": "ctrl",
        "camera_features": camera_features,
        "camera_feature_provenance": "dataset logical schema plus per-episode physical camera sidecars",
        "episode_sidecar_dir": "meta/axis_episodes",
        "attempt_commit_dir": "meta/axis_attempt_commits",
    }
    if identity.camera_group is not None:
        task_payload["camera_group"] = identity.camera_group
    atomic_write_json(dataset_root / "meta" / "axis_task_dataset.json", task_payload)


def _commit_state_payload(
    *,
    total_episodes: int,
    total_frames: int,
    repo_id: str,
    fps: int,
    robot_type: str,
    vcodec: str | None,
    identity: EpisodeIdentity | None = None,
    migration: str | None = None,
    last_indices: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": ATOMIC_COMMIT_SCHEMA,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "repo_id": str(repo_id),
        "fps": int(fps),
        "robot_type": str(robot_type),
        "vcodec": vcodec,
    }
    if identity is not None:
        payload["last_identity"] = _identity_payload(identity)
    if migration is not None:
        payload["migration"] = migration
    if last_indices is not None:
        payload["last_indices"] = deepcopy(last_indices)
    return payload


def _validate_or_bootstrap_commit_state(
    dataset_root: Path,
    dataset: Any,
    *,
    repo_id: str,
    fps: int,
    robot_type: str,
    vcodec: str | None,
) -> None:
    state_path = _commit_state_path(dataset_root)
    total_episodes = int(dataset.meta.total_episodes)
    total_frames = int(dataset.meta.total_frames)
    if state_path.is_file():
        state = _load_json(state_path)
        if state.get("schema") != ATOMIC_COMMIT_SCHEMA:
            raise ValueError(f"Unsupported AXIS commit-state schema: {state_path}")
        observed = (int(state.get("total_episodes", -1)), int(state.get("total_frames", -1)))
        expected = (total_episodes, total_frames)
        if observed != expected:
            raise RuntimeError(
                f"AXIS commit state disagrees with LeRobot info for {dataset_root}: "
                f"state={observed}, info={expected}."
            )
        expected_config = {
            "repo_id": str(repo_id),
            "fps": int(fps),
            "robot_type": str(robot_type),
            "vcodec": vcodec,
        }
        observed_config = {key: state.get(key) for key in expected_config}
        if observed_config != expected_config:
            raise ValueError(
                f"AXIS commit-state configuration mismatch for {dataset_root}: "
                f"expected={expected_config}, observed={observed_config}."
            )
        return

    sidecar_dir = dataset_root / "meta" / "axis_episodes"
    sidecars = sorted(sidecar_dir.glob("episode_*.json")) if sidecar_dir.is_dir() else []
    if len(sidecars) != total_episodes:
        raise RuntimeError(
            "Cannot bootstrap AXIS commit state: episode sidecar count does not match LeRobot info "
            f"for {dataset_root}: sidecars={len(sidecars)}, total_episodes={total_episodes}."
        )
    episode_indices: list[int] = []
    frame_total = 0
    for sidecar in sidecars:
        episode = (_load_json(sidecar).get("episode") or {})
        episode_indices.append(int(episode.get("episode_index", -1)))
        frame_total += int(episode.get("frame_count", 0))
    if episode_indices != list(range(total_episodes)) or frame_total != total_frames:
        raise RuntimeError(
            "Cannot bootstrap AXIS commit state: legacy sidecars are not contiguous or frame counts differ "
            f"for {dataset_root}."
        )
    atomic_write_json(
        state_path,
        _commit_state_payload(
            total_episodes=total_episodes,
            total_frames=total_frames,
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            vcodec=vcodec,
            migration="validated_legacy_axis_episode_sidecars",
        ),
    )


def _last_indices_from_episode(episode: dict[str, Any], video_keys: Sequence[str]) -> dict[str, Any]:
    return {
        "data": [int(episode["data/chunk_index"]), int(episode["data/file_index"])],
        "episode": [
            int(episode["meta/episodes/chunk_index"]),
            int(episode["meta/episodes/file_index"]),
        ],
        "videos": {
            str(video_key): [
                int(episode[f"videos/{video_key}/chunk_index"]),
                int(episode[f"videos/{video_key}/file_index"]),
            ]
            for video_key in video_keys
        },
    }


def _read_latest_episode_metadata(dataset_root: Path, info: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes <= 0:
        raise RuntimeError(f"Existing LeRobot dataset has no episodes: {dataset_root}")
    chunks_size = int(info["chunks_size"])
    expected_chunk = (total_episodes - 1) // chunks_size
    expected_file = (total_episodes - 1) % chunks_size
    expected_path = (
        Path(dataset_root)
        / "meta"
        / "episodes"
        / f"chunk-{expected_chunk:03d}"
        / f"file-{expected_file:03d}.parquet"
    )
    candidates = [expected_path] if expected_path.is_file() else []
    if not candidates:
        candidates = sorted(
            (Path(dataset_root) / "meta" / "episodes").glob("chunk-*/*.parquet"),
            key=lambda path: (
                int(path.parent.name.rsplit("-", 1)[-1]),
                int(path.stem.rsplit("-", 1)[-1]),
            ),
            reverse=True,
        )
    for path in candidates:
        rows = pq.read_table(path).to_pylist()
        for row in reversed(rows):
            if int(row.get("episode_index", -1)) == total_episodes - 1:
                return row
    raise RuntimeError(
        f"Could not locate latest episode metadata index {total_episodes - 1} under {dataset_root}."
    )


def _load_lightweight_target_state(
    dataset_root: Path,
    *,
    repo_id: str,
    expected_features: dict[str, Any],
    fps: int,
    robot_type: str,
    vcodec: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_root = Path(dataset_root)
    info = _load_json(dataset_root / "meta" / "info.json")
    _assert_info_features_match(info, expected_features)
    _assert_info_config(info, fps=fps, robot_type=robot_type)
    state_path = _commit_state_path(dataset_root)
    if not state_path.is_file():
        total_episodes = int(info["total_episodes"])
        sidecar_dir = dataset_root / "meta" / "axis_episodes"
        sidecars = sorted(sidecar_dir.glob("episode_*.json")) if sidecar_dir.is_dir() else []
        if len(sidecars) != total_episodes:
            raise RuntimeError(
                "Cannot bootstrap AXIS commit state: episode sidecar count does not match LeRobot info "
                f"for {dataset_root}: sidecars={len(sidecars)}, total_episodes={total_episodes}."
            )
        frame_total = sum(
            int((_load_json(path).get("episode") or {}).get("frame_count", 0))
            for path in sidecars
        )
        if frame_total != int(info["total_frames"]):
            raise RuntimeError(
                f"Cannot bootstrap AXIS commit state: frame totals differ for {dataset_root}."
            )
        latest = _read_latest_episode_metadata(dataset_root, info)
        state = _commit_state_payload(
            total_episodes=total_episodes,
            total_frames=int(info["total_frames"]),
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            vcodec=vcodec,
            migration="validated_legacy_axis_episode_sidecars_lightweight",
            last_indices=_last_indices_from_episode(latest, _video_keys_from_info(info)),
        )
        atomic_write_json(state_path, state)
    else:
        state = _load_json(state_path)

    if state.get("schema") != ATOMIC_COMMIT_SCHEMA:
        raise ValueError(f"Unsupported AXIS commit-state schema: {state_path}")
    state_totals = (int(state.get("total_episodes", -1)), int(state.get("total_frames", -1)))
    info_totals = (int(info.get("total_episodes", -1)), int(info.get("total_frames", -1)))
    if state_totals != info_totals:
        raise RuntimeError(
            f"AXIS commit state disagrees with LeRobot info for {dataset_root}: "
            f"state={state_totals}, info={info_totals}."
        )
    expected_config = {
        "repo_id": str(repo_id),
        "fps": int(fps),
        "robot_type": str(robot_type),
        "vcodec": vcodec,
    }
    observed_config = {key: state.get(key) for key in expected_config}
    if observed_config != expected_config:
        raise ValueError(
            f"AXIS commit-state configuration mismatch for {dataset_root}: "
            f"expected={expected_config}, observed={observed_config}."
        )
    if not isinstance(state.get("last_indices"), dict):
        latest = _read_latest_episode_metadata(dataset_root, info)
        state = deepcopy(state)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["last_indices"] = _last_indices_from_episode(
            latest,
            _video_keys_from_info(info),
        )
        state["migration"] = "added_lightweight_append_indices"
        atomic_write_json(state_path, state)
    return info, state


def _plan_lightweight_append_paths(
    dataset_root: Path,
    *,
    info: dict[str, Any],
    state: dict[str, Any],
    identity: EpisodeIdentity,
) -> dict[str, Any]:
    chunks_size = int(info["chunks_size"])
    last_indices = state["last_indices"]
    data_chunk, data_file = _next_chunk_file(*last_indices["data"], chunks_size)
    episode_chunk, episode_file = _next_chunk_file(*last_indices["episode"], chunks_size)
    video_paths: dict[str, str] = {}
    video_indices: dict[str, tuple[int, int]] = {}
    for video_key in _video_keys_from_info(info):
        if video_key not in (last_indices.get("videos") or {}):
            raise RuntimeError(f"Commit state is missing video indices for {video_key!r}.")
        chunk_index, file_index = _next_chunk_file(
            *last_indices["videos"][video_key],
            chunks_size,
        )
        video_indices[video_key] = (chunk_index, file_index)
        video_paths[video_key] = str(info["video_path"]).format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
    data_path = str(info["data_path"]).format(chunk_index=data_chunk, file_index=data_file)
    episode_path = f"meta/episodes/chunk-{episode_chunk:03d}/file-{episode_file:03d}.parquet"
    episode_index = int(info["total_episodes"])
    axis_episode_path = str(
        _axis_episode_path(dataset_root, identity, episode_index).relative_to(dataset_root)
    )
    marker_path = str(_attempt_commit_path(dataset_root, identity).relative_to(dataset_root))
    created_files = [data_path, episode_path, *video_paths.values(), axis_episode_path, marker_path]
    for relative_path in created_files:
        if (Path(dataset_root) / relative_path).exists():
            raise FileExistsError(f"Atomic append target already exists: {Path(dataset_root) / relative_path}")
    return {
        "data_path": data_path,
        "data_indices": (data_chunk, data_file),
        "episode_path": episode_path,
        "episode_indices": (episode_chunk, episode_file),
        "video_paths": video_paths,
        "video_indices": video_indices,
        "axis_episode_path": axis_episode_path,
        "marker_path": marker_path,
        "created_files": created_files,
    }


def _task_index_for_name(dataset_root: Path, task_name: str) -> int:
    import pyarrow.parquet as pq

    rows = pq.read_table(Path(dataset_root) / "meta" / "tasks.parquet").to_pylist()
    matches = [
        int(row["task_index"])
        for row in rows
        if str(row.get("__index_level_0__", row.get("task", ""))) == str(task_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one task row for {task_name!r} in {dataset_root}, got {len(matches)}."
        )
    return matches[0]


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    path = Path(path)
    if not path.is_dir():
        return
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fsync_tree(root: Path) -> None:
    root = Path(root)
    for path in root.rglob("*"):
        if path.is_file():
            _fsync_file(path)
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _stage_shard_on_canonical_filesystem(
    shard_root: Path,
    dataset_root: Path,
    identity: EpisodeIdentity,
) -> Path:
    staging_base = dataset_root.parent / ".axis_lerobot_commit_staging" / _dataset_token(dataset_root)
    # Keep each worker in its own owner directory.  The old shared staging
    # level let one worker's empty-parent cleanup race another worker between
    # mkdir and copytree on JuiceFS, producing intermittent FileNotFoundError.
    staging_owner = staging_base / f"worker-{os.getpid()}-{time.time_ns()}"
    staging_owner.mkdir(parents=True, exist_ok=True)
    stage_root = staging_owner / identity.key
    shutil.copytree(shard_root, stage_root, copy_function=_link_or_copy)
    _fsync_tree(stage_root)
    _fsync_directory(staging_base)
    _fsync_directory(staging_base.parent)
    _fsync_directory(dataset_root.parent)
    return stage_root


def _remove_tree_and_empty_parents(path: Path, *, stop: Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
        _fsync_directory(path.parent)
    parent = path.parent
    stop = Path(stop)
    while parent != stop and parent.is_dir():
        try:
            parent.rmdir()
        except OSError:
            break
        _fsync_directory(parent.parent)
        parent = parent.parent
    if stop.is_dir() and not any(stop.iterdir()):
        try:
            stop.rmdir()
            _fsync_directory(stop.parent)
        except OSError:
            pass


def _next_chunk_file(chunk_index: int, file_index: int, chunks_size: int) -> tuple[int, int]:
    file_index = int(file_index) + 1
    chunk_index = int(chunk_index)
    if file_index >= int(chunks_size):
        return chunk_index + 1, 0
    return chunk_index, file_index


def _plan_append_paths(dataset_root: Path, target: Any, identity: EpisodeIdentity) -> dict[str, Any]:
    latest = target.meta.episodes[-1]
    chunks_size = int(target.meta.chunks_size)
    data_chunk, data_file = _next_chunk_file(
        int(latest["data/chunk_index"]),
        int(latest["data/file_index"]),
        chunks_size,
    )
    episode_chunk, episode_file = _next_chunk_file(
        int(latest["meta/episodes/chunk_index"]),
        int(latest["meta/episodes/file_index"]),
        chunks_size,
    )
    video_paths: dict[str, str] = {}
    video_indices: dict[str, tuple[int, int]] = {}
    for video_key in target.meta.video_keys:
        chunk_index, file_index = _next_chunk_file(
            int(latest[f"videos/{video_key}/chunk_index"]),
            int(latest[f"videos/{video_key}/file_index"]),
            chunks_size,
        )
        video_indices[video_key] = (chunk_index, file_index)
        video_paths[video_key] = target.meta.video_path.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
    data_path = target.meta.data_path.format(chunk_index=data_chunk, file_index=data_file)
    episode_path = f"meta/episodes/chunk-{episode_chunk:03d}/file-{episode_file:03d}.parquet"
    episode_index = int(target.meta.total_episodes)
    axis_episode_path = str(_axis_episode_path(dataset_root, identity, episode_index).relative_to(dataset_root))
    marker_path = str(_attempt_commit_path(dataset_root, identity).relative_to(dataset_root))
    created_files = [data_path, episode_path, *video_paths.values(), axis_episode_path, marker_path]
    for relative_path in created_files:
        if (dataset_root / relative_path).exists():
            raise FileExistsError(f"Atomic append target already exists: {dataset_root / relative_path}")
    return {
        "data_path": data_path,
        "data_indices": (data_chunk, data_file),
        "episode_path": episode_path,
        "episode_indices": (episode_chunk, episode_file),
        "video_paths": video_paths,
        "video_indices": video_indices,
        "axis_episode_path": axis_episode_path,
        "marker_path": marker_path,
        "created_files": created_files,
    }


_TRANSACTION_MUTABLE_FILES = (
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/axis_commit_state.json",
    "meta/axis_task_dataset.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_transaction(
    *,
    dataset_root: Path,
    identity: EpisodeIdentity,
    plan: dict[str, Any],
    repo_id: str,
    fps: int,
    robot_type: str,
    vcodec: str | None,
    old_total_episodes: int,
    old_total_frames: int,
    frame_count: int,
    stage_root: Path,
    shard_root: Path,
) -> dict[str, Any]:
    transaction_dir = _transaction_dir(dataset_root, identity)
    if transaction_dir.exists():
        raise RuntimeError(f"Transaction directory was not recovered: {transaction_dir}")
    backup_root = transaction_dir / "backup"
    backup_root.mkdir(parents=True, exist_ok=False)
    backups: list[dict[str, Any]] = []
    for relative_path in _TRANSACTION_MUTABLE_FILES:
        source = dataset_root / relative_path
        entry: dict[str, Any] = {"path": relative_path, "existed": source.is_file()}
        if source.is_file():
            destination = backup_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            _fsync_file(destination)
            entry.update({"size": source.stat().st_size, "sha256": _sha256_file(source)})
        backups.append(entry)
    _fsync_tree(backup_root)
    transaction_id = f"{identity.key}-{time.time_ns()}"
    payload = {
        "schema": ATOMIC_COMMIT_SCHEMA,
        "state": "prepared",
        "transaction_id": transaction_id,
        "transaction_dir": str(transaction_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "identity": _identity_payload(identity),
        "repo_id": str(repo_id),
        "fps": int(fps),
        "robot_type": str(robot_type),
        "vcodec": vcodec,
        "old_total_episodes": int(old_total_episodes),
        "old_total_frames": int(old_total_frames),
        "expected_total_episodes": int(old_total_episodes) + 1,
        "expected_total_frames": int(old_total_frames) + int(frame_count),
        "frame_count": int(frame_count),
        "created_files": list(plan["created_files"]),
        "axis_episode_path": plan["axis_episode_path"],
        "marker_path": plan["marker_path"],
        "backups": backups,
        "stage_root": str(stage_root),
        "shard_root": str(shard_root),
    }
    atomic_write_json(_transaction_path(dataset_root, identity), payload)
    _fsync_directory(transaction_dir)
    _fsync_directory(transaction_dir.parent)
    _fsync_directory(transaction_dir.parent.parent)
    _fsync_directory(dataset_root.parent)
    return payload


def _atomic_restore_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.restore.{os.getpid()}")
    shutil.copy2(source, temporary)
    _fsync_file(temporary)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _cleanup_transaction_directory(dataset_root: Path, transaction_dir: Path) -> None:
    base = _transaction_base(dataset_root)
    if transaction_dir.exists():
        shutil.rmtree(transaction_dir)
        _fsync_directory(base)
    if base.is_dir() and not any(base.iterdir()):
        try:
            base.rmdir()
            _fsync_directory(base.parent)
        except OSError:
            return
        parent = base.parent
        if parent.is_dir() and not any(parent.iterdir()):
            try:
                parent.rmdir()
                _fsync_directory(dataset_root.parent)
            except OSError:
                pass


def _rollback_transaction_unlocked(dataset_root: Path, transaction: dict[str, Any]) -> None:
    transaction_dir = Path(str(transaction.get("transaction_dir"))).resolve()
    expected_base = _transaction_base(dataset_root).resolve()
    if transaction_dir.parent != expected_base or not (transaction_dir / "transaction.json").is_file():
        raise RuntimeError(f"Cannot locate transaction backup for {transaction.get('transaction_id')!r}.")

    for relative_path in transaction.get("created_files") or []:
        path = dataset_root / str(relative_path)
        if path.is_file() or path.is_symlink():
            path.unlink()
            _fsync_directory(path.parent)
    backup_root = transaction_dir / "backup"
    for entry in transaction.get("backups") or []:
        relative_path = str(entry["path"])
        destination = dataset_root / relative_path
        if entry.get("existed"):
            backup = backup_root / relative_path
            if not backup.is_file():
                raise RuntimeError(f"Transaction backup is missing: {backup}")
            if _sha256_file(backup) != entry.get("sha256"):
                raise RuntimeError(f"Transaction backup checksum mismatch: {backup}")
            _atomic_restore_file(backup, destination)
            if destination.stat().st_size != int(entry["size"]) or _sha256_file(destination) != entry["sha256"]:
                raise RuntimeError(f"Restored transaction file failed verification: {destination}")
        elif destination.exists():
            destination.unlink()
            _fsync_directory(destination.parent)

    stage_raw = transaction.get("stage_root")
    stage_root = Path(str(stage_raw)) if stage_raw else None
    if stage_root is not None:
        _remove_tree_and_empty_parents(
            stage_root,
            stop=dataset_root.parent / ".axis_lerobot_commit_staging",
        )
    info = _load_json(dataset_root / "meta" / "info.json")
    state = _load_json(_commit_state_path(dataset_root))
    observed = (int(info["total_episodes"]), int(info["total_frames"]))
    observed_state = (int(state["total_episodes"]), int(state["total_frames"]))
    expected = (int(transaction["old_total_episodes"]), int(transaction["old_total_frames"]))
    if observed != expected or observed_state != expected:
        raise RuntimeError(
            f"Transaction rollback totals mismatch: expected={expected}, "
            f"info={observed}, state={observed_state}."
        )
    _cleanup_transaction_directory(dataset_root, transaction_dir)


def _transaction_is_complete(dataset_root: Path, transaction: dict[str, Any]) -> bool:
    marker_path = dataset_root / str(transaction["marker_path"])
    if not marker_path.is_file():
        return False
    marker = _load_json(marker_path)
    if marker.get("transaction_id") != transaction.get("transaction_id"):
        raise RuntimeError(f"Attempt marker transaction id mismatch: {marker_path}")
    if marker.get("identity") != transaction.get("identity"):
        raise RuntimeError(f"Attempt marker identity mismatch: {marker_path}")
    episode = marker.get("episode") or {}
    if (
        int(episode.get("episode_index", -1)) != int(transaction["old_total_episodes"])
        or int(episode.get("frame_count", -1)) != int(transaction["frame_count"])
    ):
        raise RuntimeError(f"Attempt marker episode record mismatch: {marker_path}")
    state = _load_json(_commit_state_path(dataset_root))
    expected = (int(transaction["expected_total_episodes"]), int(transaction["expected_total_frames"]))
    observed_state = (int(state.get("total_episodes", -1)), int(state.get("total_frames", -1)))
    if observed_state[0] < expected[0] or observed_state[1] < expected[1]:
        return False
    for relative_path in transaction.get("created_files") or []:
        path = dataset_root / str(relative_path)
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    info = _load_json(dataset_root / "meta" / "info.json")
    observed_info = (int(info.get("total_episodes", -1)), int(info.get("total_frames", -1)))
    return observed_info == observed_state


def _transaction_is_obsolete_after_later_commits(
    dataset_root: Path,
    transaction: dict[str, Any],
) -> bool:
    """Detect a deleted pre-marker WAL that reappeared after later durable commits."""

    marker_path = dataset_root / str(transaction["marker_path"])
    if marker_path.exists():
        return False
    state_path = _commit_state_path(dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    if not state_path.is_file() or not info_path.is_file():
        return False
    state = _load_json(state_path)
    info = _load_json(info_path)
    state_totals = (int(state.get("total_episodes", -1)), int(state.get("total_frames", -1)))
    info_totals = (int(info.get("total_episodes", -1)), int(info.get("total_frames", -1)))
    old_totals = (int(transaction["old_total_episodes"]), int(transaction["old_total_frames"]))
    if state_totals != info_totals or state_totals[0] <= old_totals[0]:
        return False
    planned_sidecar = dataset_root / str(transaction["axis_episode_path"])
    if planned_sidecar.exists():
        return False
    sidecar_dir = dataset_root / "meta" / "axis_episodes"
    sidecars = sorted(sidecar_dir.glob("episode_*.json")) if sidecar_dir.is_dir() else []
    if len(sidecars) != state_totals[0]:
        return False
    episode_indices = [int((_load_json(path).get("episode") or {}).get("episode_index", -1)) for path in sidecars]
    return episode_indices == list(range(state_totals[0]))


def _recover_pending_transactions_unlocked(dataset_root: Path) -> None:
    base = _transaction_base(dataset_root)
    if not base.is_dir():
        return
    for transaction_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        path = transaction_dir / "transaction.json"
        if not path.is_file():
            shutil.rmtree(transaction_dir)
            continue
        transaction = _load_json(path)
        if transaction.get("schema") != ATOMIC_COMMIT_SCHEMA:
            raise RuntimeError(f"Unsupported AXIS transaction schema: {path}")
        if Path(str(transaction.get("dataset_root"))).resolve() != dataset_root.resolve():
            raise RuntimeError(f"AXIS transaction targets another dataset: {path}")
        if _transaction_is_complete(dataset_root, transaction):
            stage_raw = transaction.get("stage_root")
            shard_raw = transaction.get("shard_root")
            stage_root = Path(str(stage_raw)) if stage_raw else None
            shard_root = Path(str(shard_raw)) if shard_raw else None
            if stage_root is not None:
                _remove_tree_and_empty_parents(
                    stage_root,
                    stop=dataset_root.parent / ".axis_lerobot_commit_staging",
                )
            if shard_root is not None and shard_root.exists():
                shutil.rmtree(shard_root)
            _cleanup_transaction_directory(dataset_root, transaction_dir)
        elif _transaction_is_obsolete_after_later_commits(dataset_root, transaction):
            stage_raw = transaction.get("stage_root")
            stage_root = Path(str(stage_raw)) if stage_raw else None
            if stage_root is not None:
                _remove_tree_and_empty_parents(
                    stage_root,
                    stop=dataset_root.parent / ".axis_lerobot_commit_staging",
                )
            _cleanup_transaction_directory(dataset_root, transaction_dir)
        else:
            _rollback_transaction_unlocked(dataset_root, transaction)


def _maybe_crash(test_crash_phase: str | None, phase: str) -> None:
    if test_crash_phase == phase:
        os._exit(97)


def _validate_written_append(
    *,
    dataset_root: Path,
    plan: dict[str, Any],
    episode_index: int,
    frame_start: int,
    frame_count: int,
    task_index: int,
    expected_total_episodes: int,
    expected_total_frames: int,
) -> None:
    import pyarrow.parquet as pq

    data_table = pq.read_table(dataset_root / plan["data_path"])
    if data_table.num_rows != frame_count:
        raise RuntimeError(f"Committed data parquet row count mismatch: {data_table.num_rows}.")
    data = data_table.to_pydict()
    if data["index"] != list(range(frame_start, frame_start + frame_count)):
        raise RuntimeError("Committed global frame indices are not contiguous.")
    if set(data["episode_index"]) != {episode_index} or set(data["task_index"]) != {task_index}:
        raise RuntimeError("Committed episode_index/task_index columns are inconsistent.")
    episode_table = pq.read_table(dataset_root / plan["episode_path"])
    if episode_table.num_rows != 1:
        raise RuntimeError(f"Committed episode metadata row count mismatch: {episode_table.num_rows}.")
    episode = episode_table.to_pylist()[0]
    if int(episode["episode_index"]) != episode_index:
        raise RuntimeError("Committed episode metadata index mismatch.")
    if (int(episode["dataset_from_index"]), int(episode["dataset_to_index"])) != (
        frame_start,
        frame_start + frame_count,
    ):
        raise RuntimeError("Committed episode frame range mismatch.")
    info = _load_json(dataset_root / "meta" / "info.json")
    observed = (int(info["total_episodes"]), int(info["total_frames"]))
    expected = (expected_total_episodes, expected_total_frames)
    if observed != expected:
        raise RuntimeError(f"Committed info totals mismatch: expected={expected}, observed={observed}.")
    for relative_path in plan["video_paths"].values():
        path = dataset_root / relative_path
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Committed video is missing or empty: {path}")


def _commit_episode_shard_v1_reference(
    *,
    shard_root: Path,
    dataset_root: Path,
    repo_id: str,
    expected_features: dict[str, Any],
    identity: EpisodeIdentity,
    metadata: dict[str, Any],
    fps: int,
    robot_type: str,
    vcodec: str | None,
    camera_keys: dict[str, str],
    policy_state_dim: int,
    action_dim: int,
    ee_state_dim: int | None,
    policy_state_metadata: dict[str, Any],
    nq: int,
    nv: int,
    nu: int,
    test_crash_phase: str | None = None,
) -> dict[str, Any]:
    """Publish one encoded shard with a crash-recoverable, exactly-once append."""

    dataset_class, _ = _load_lerobot_api()
    shard_root = Path(shard_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    with exclusive_lerobot_attempt_lock(dataset_root, identity):
        with exclusive_lerobot_dataset_lock(dataset_root):
            _recover_pending_transactions_unlocked(dataset_root)
            existing = _load_committed_attempt_unlocked(dataset_root, identity)
        if existing is not None:
            if shard_root.exists():
                shutil.rmtree(shard_root)
            return existing["episode"]

        manifest_path = _shard_manifest_path(shard_root)
        manifest = _load_json(manifest_path)
        _validate_shard_manifest(
            manifest,
            path=manifest_path,
            identity=identity,
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            vcodec=vcodec,
        )
        frame_count = int(manifest.get("frame_count", -1))
        encode_seconds = float(manifest.get("encode_seconds", 0.0))
        if frame_count <= 0:
            raise ValueError(f"Prepared episode shard has invalid frame_count={frame_count}: {manifest_path}")

        source = _open_dataset(dataset_class, root=shard_root, repo_id=repo_id, vcodec=vcodec)
        assert_features_match(source, expected_features)
        _assert_dataset_config(source, repo_id=repo_id, fps=fps, robot_type=robot_type)
        if int(source.meta.total_episodes) != 1 or int(source.meta.total_frames) != frame_count:
            raise RuntimeError(f"Prepared shard changed after validation: {shard_root}")
        source_frames = source.hf_dataset.with_format("numpy")[:]
        source_stats = deepcopy(source.meta.stats)
        source_episode = source.meta.episodes[0]
        source_video_durations: dict[str, float] = {}
        for video_key in source.meta.video_keys:
            video_path = shard_root / source.meta.video_path.format(
                video_key=video_key,
                chunk_index=int(source_episode[f"videos/{video_key}/chunk_index"]),
                file_index=int(source_episode[f"videos/{video_key}/file_index"]),
            )
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                raise RuntimeError(f"Prepared video disappeared before commit: {video_path}")
            declared_path = (manifest.get("video_paths") or {}).get(video_key)
            if declared_path != str(video_path.relative_to(shard_root)):
                raise ValueError(f"Prepared video path disagrees with shard manifest for {video_key!r}.")
            source_video_durations[video_key] = float(
                source_episode[f"videos/{video_key}/to_timestamp"]
                - source_episode[f"videos/{video_key}/from_timestamp"]
            )

        stage_started = time.perf_counter()
        stage_root = _stage_shard_on_canonical_filesystem(shard_root, dataset_root, identity)
        stage_seconds = time.perf_counter() - stage_started
        result: dict[str, Any] | None = None
        committed = False
        try:
            lock_started = time.perf_counter()
            with exclusive_lerobot_dataset_lock(dataset_root):
                lock_acquired = time.perf_counter()
                _recover_pending_transactions_unlocked(dataset_root)
                existing = _load_committed_attempt_unlocked(dataset_root, identity)
                if existing is not None:
                    result = existing["episode"]
                    committed = True
                elif not dataset_root.exists():
                    commit_started = time.perf_counter()
                    result = _axis_episode_record(
                        dataset_root=dataset_root,
                        repo_id=repo_id,
                        identity=identity,
                        episode_index=0,
                        frame_count=frame_count,
                        fps=fps,
                        robot_type=robot_type,
                        camera_keys=camera_keys,
                        policy_state_dim=policy_state_dim,
                        action_dim=action_dim,
                        ee_state_dim=ee_state_dim,
                        policy_state_metadata=policy_state_metadata,
                        nq=nq,
                        nv=nv,
                        nu=nu,
                        total_episodes=1,
                        total_frames=frame_count,
                        encode_seconds=encode_seconds,
                        stage_seconds=stage_seconds,
                        lock_wait_seconds=lock_acquired - lock_started,
                        lock_work_seconds=time.perf_counter() - lock_acquired,
                        commit_seconds=0.0,
                    )
                    _write_axis_sidecars(
                        dataset_root=stage_root,
                        identity=identity,
                        metadata=metadata,
                        episode_record=result,
                    )
                    atomic_write_json(
                        _commit_state_path(stage_root),
                        _commit_state_payload(
                            total_episodes=1,
                            total_frames=frame_count,
                            repo_id=repo_id,
                            fps=fps,
                            robot_type=robot_type,
                            vcodec=vcodec,
                            identity=identity,
                        ),
                    )
                    result["commit"]["commit_seconds"] = time.perf_counter() - commit_started
                    result["commit"]["lock_work_seconds"] = time.perf_counter() - lock_acquired
                    _write_axis_sidecars(
                        dataset_root=stage_root,
                        identity=identity,
                        metadata=metadata,
                        episode_record=result,
                    )
                    atomic_write_json(
                        _attempt_commit_path(stage_root, identity),
                        {
                            "schema": ATOMIC_COMMIT_SCHEMA,
                            "committed_at": datetime.now(timezone.utc).isoformat(),
                            "transaction_id": "initial-directory-publish",
                            "identity": _identity_payload(identity),
                            "episode": result,
                            "metadata": metadata,
                        },
                    )
                    stage_manifest = _shard_manifest_path(stage_root)
                    if stage_manifest.exists():
                        stage_manifest.unlink()
                        _fsync_directory(stage_manifest.parent)
                    os.replace(stage_root, dataset_root)
                    _fsync_directory(dataset_root.parent)
                    committed = True
                else:
                    target = _open_dataset(dataset_class, root=dataset_root, repo_id=repo_id, vcodec=vcodec)
                    assert_features_match(target, expected_features)
                    _assert_dataset_config(target, repo_id=repo_id, fps=fps, robot_type=robot_type)
                    _assert_video_info_match(source, target)
                    _validate_or_bootstrap_commit_state(
                        dataset_root,
                        target,
                        repo_id=repo_id,
                        fps=fps,
                        robot_type=robot_type,
                        vcodec=vcodec,
                    )
                    old_total_episodes = int(target.meta.total_episodes)
                    old_total_frames = int(target.meta.total_frames)
                    plan = _plan_append_paths(dataset_root, target, identity)
                    transaction = _prepare_transaction(
                        dataset_root=dataset_root,
                        identity=identity,
                        plan=plan,
                        repo_id=repo_id,
                        fps=fps,
                        robot_type=robot_type,
                        vcodec=vcodec,
                        old_total_episodes=old_total_episodes,
                        old_total_frames=old_total_frames,
                        frame_count=frame_count,
                        stage_root=stage_root,
                        shard_root=shard_root,
                    )
                    _maybe_crash(test_crash_phase, "after_transaction")
                    commit_started = time.perf_counter()
                    marker_written = False
                    try:
                        target.meta.save_episode_tasks([identity.task_name])
                        task_index = int(target.meta.get_task_index(identity.task_name))
                        episode_data = {key: np.asarray(source_frames[key]) for key in target.hf_features}
                        episode_data["index"] = np.arange(
                            old_total_frames,
                            old_total_frames + frame_count,
                            dtype=np.int64,
                        )
                        episode_data["episode_index"] = np.full(
                            (frame_count,), old_total_episodes, dtype=np.int64
                        )
                        episode_data["task_index"] = np.full((frame_count,), task_index, dtype=np.int64)
                        episode_metadata = target._save_episode_data(episode_data)
                        if (
                            int(episode_metadata["data/chunk_index"]),
                            int(episode_metadata["data/file_index"]),
                        ) != tuple(plan["data_indices"]):
                            raise RuntimeError("LeRobot data writer did not use the WAL-planned file.")
                        _maybe_crash(test_crash_phase, "after_data")
                        for video_key in target.meta.video_keys:
                            chunk_index, file_index = plan["video_indices"][video_key]
                            staged_video = stage_root / str(manifest["video_paths"][video_key])
                            destination = dataset_root / plan["video_paths"][video_key]
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(staged_video, destination)
                            episode_metadata.update(
                                {
                                    "episode_index": old_total_episodes,
                                    f"videos/{video_key}/chunk_index": chunk_index,
                                    f"videos/{video_key}/file_index": file_index,
                                    f"videos/{video_key}/from_timestamp": 0.0,
                                    f"videos/{video_key}/to_timestamp": source_video_durations[video_key],
                                }
                            )
                        _maybe_crash(test_crash_phase, "after_videos")
                        shifted_stats = _shift_stats(
                            source_stats,
                            episode_index=old_total_episodes,
                            frame_start=old_total_frames,
                            task_index=task_index,
                        )
                        target.meta.save_episode(
                            old_total_episodes,
                            frame_count,
                            [identity.task_name],
                            shifted_stats,
                            episode_metadata,
                        )
                        _maybe_crash(test_crash_phase, "after_save_episode")
                        target.finalize()
                        new_total_episodes = int(target.meta.total_episodes)
                        new_total_frames = int(target.meta.total_frames)
                        expected_totals = (old_total_episodes + 1, old_total_frames + frame_count)
                        if (new_total_episodes, new_total_frames) != expected_totals:
                            raise RuntimeError(
                                f"LeRobot totals changed unexpectedly while committing {identity.key}: "
                                f"expected={expected_totals}, "
                                f"actual={(new_total_episodes, new_total_frames)}"
                            )
                        _validate_written_append(
                            dataset_root=dataset_root,
                            plan=plan,
                            episode_index=old_total_episodes,
                            frame_start=old_total_frames,
                            frame_count=frame_count,
                            task_index=task_index,
                            expected_total_episodes=new_total_episodes,
                            expected_total_frames=new_total_frames,
                        )
                        _maybe_crash(test_crash_phase, "after_meta")
                        commit_seconds = time.perf_counter() - commit_started
                        result = _axis_episode_record(
                            dataset_root=dataset_root,
                            repo_id=repo_id,
                            identity=identity,
                            episode_index=old_total_episodes,
                            frame_count=frame_count,
                            fps=fps,
                            robot_type=robot_type,
                            camera_keys=camera_keys,
                            policy_state_dim=policy_state_dim,
                            action_dim=action_dim,
                            ee_state_dim=ee_state_dim,
                            policy_state_metadata=policy_state_metadata,
                            nq=nq,
                            nv=nv,
                            nu=nu,
                            total_episodes=new_total_episodes,
                            total_frames=new_total_frames,
                            encode_seconds=encode_seconds,
                            stage_seconds=stage_seconds,
                            lock_wait_seconds=lock_acquired - lock_started,
                            lock_work_seconds=time.perf_counter() - lock_acquired,
                            commit_seconds=commit_seconds,
                        )
                        _write_axis_sidecars(
                            dataset_root=dataset_root,
                            identity=identity,
                            metadata=metadata,
                            episode_record=result,
                        )
                        atomic_write_json(
                            _commit_state_path(dataset_root),
                            _commit_state_payload(
                                total_episodes=new_total_episodes,
                                total_frames=new_total_frames,
                                repo_id=repo_id,
                                fps=fps,
                                robot_type=robot_type,
                                vcodec=vcodec,
                                identity=identity,
                            ),
                        )
                        durable_paths = [
                            *(dataset_root / path for path in plan["created_files"] if path != plan["marker_path"]),
                            *(dataset_root / path for path in _TRANSACTION_MUTABLE_FILES),
                        ]
                        for path in durable_paths:
                            if path.is_file():
                                _fsync_file(path)
                                _fsync_directory(path.parent)
                        _maybe_crash(test_crash_phase, "after_sidecars")
                        atomic_write_json(
                            _attempt_commit_path(dataset_root, identity),
                            {
                                "schema": ATOMIC_COMMIT_SCHEMA,
                                "committed_at": datetime.now(timezone.utc).isoformat(),
                                "transaction_id": transaction["transaction_id"],
                                "identity": _identity_payload(identity),
                                "episode": result,
                                "metadata": metadata,
                            },
                        )
                        marker_written = True
                        _maybe_crash(test_crash_phase, "after_marker")
                        _cleanup_transaction_directory(
                            dataset_root,
                            _transaction_path(dataset_root, identity).parent,
                        )
                        committed = True
                    except Exception:
                        if marker_written:
                            raise
                        try:
                            target.finalize()
                        except Exception:
                            pass
                        _rollback_transaction_unlocked(dataset_root, transaction)
                        raise
            if result is None:
                raise RuntimeError(f"Atomic commit produced no episode record for {identity.key}.")
            return result
        finally:
            staging_base = dataset_root.parent / ".axis_lerobot_commit_staging"
            _remove_tree_and_empty_parents(stage_root, stop=staging_base)
            if committed and shard_root.exists():
                shutil.rmtree(shard_root)


def commit_episode_shards(commits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Commit prepared shards through the lightweight batched append path."""

    from .lerobot_fast_commit import commit_episode_shards as commit_batch

    return commit_batch(commits)


def commit_episode_shard(
    *,
    shard_root: Path,
    dataset_root: Path,
    repo_id: str,
    expected_features: dict[str, Any],
    identity: EpisodeIdentity,
    metadata: dict[str, Any],
    fps: int,
    robot_type: str,
    vcodec: str | None,
    camera_keys: dict[str, str],
    policy_state_dim: int,
    action_dim: int,
    ee_state_dim: int | None,
    policy_state_metadata: dict[str, Any],
    nq: int,
    nv: int,
    nu: int,
    test_crash_phase: str | None = None,
) -> dict[str, Any]:
    """Commit one prepared shard through the lightweight append path."""

    from .lerobot_fast_commit import commit_episode_shard as commit_one

    return commit_one(
        shard_root=shard_root,
        dataset_root=dataset_root,
        repo_id=repo_id,
        expected_features=expected_features,
        identity=identity,
        metadata=metadata,
        fps=fps,
        robot_type=robot_type,
        vcodec=vcodec,
        camera_keys=camera_keys,
        policy_state_dim=policy_state_dim,
        action_dim=action_dim,
        ee_state_dim=ee_state_dim,
        policy_state_metadata=policy_state_metadata,
        nq=nq,
        nv=nv,
        nu=nu,
        test_crash_phase=test_crash_phase,
    )
