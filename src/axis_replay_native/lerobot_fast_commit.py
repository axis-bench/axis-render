"""Lightweight, batched publication of prepared LeRobot episode shards."""

from __future__ import annotations

import os
import shutil
import time
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import lerobot_atomic as atomic


def _request_signature(request: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(request["repo_id"]),
        int(request["fps"]),
        str(request["robot_type"]),
        request.get("vcodec"),
        repr(request["expected_features"]),
    )


def _normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(request)
    normalized["shard_root"] = Path(request["shard_root"]).resolve()
    normalized["dataset_root"] = Path(request["dataset_root"]).resolve()
    if not isinstance(normalized.get("identity"), atomic.EpisodeIdentity):
        raise TypeError("identity must be an EpisodeIdentity.")
    return normalized


def _remove_tree_if_present(path: Path) -> None:
    """Remove an ephemeral tree while tolerating an idempotent cleanup race."""

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _prepare_request(request: dict[str, Any]) -> dict[str, Any]:
    shard_root = request["shard_root"]
    manifest_path = atomic._shard_manifest_path(shard_root)
    manifest = atomic._load_json(manifest_path)
    atomic._validate_shard_manifest(
        manifest,
        path=manifest_path,
        identity=request["identity"],
        repo_id=request["repo_id"],
        fps=int(request["fps"]),
        robot_type=request["robot_type"],
        vcodec=request.get("vcodec"),
    )
    frame_count = int(manifest.get("frame_count", -1))
    if frame_count <= 0:
        raise ValueError(
            f"Prepared episode shard has invalid frame_count={frame_count}: {manifest_path}"
        )
    source = atomic._load_single_episode_shard(
        shard_root=shard_root,
        manifest=manifest,
        expected_features=request["expected_features"],
        fps=int(request["fps"]),
        robot_type=request["robot_type"],
    )
    stage_started = time.perf_counter()
    stage_root = atomic._stage_shard_on_canonical_filesystem(
        shard_root,
        request["dataset_root"],
        request["identity"],
    )
    return {
        "request": request,
        "manifest": manifest,
        "source": source,
        "frame_count": frame_count,
        "encode_seconds": float(manifest.get("encode_seconds", 0.0)),
        "stage_root": stage_root,
        "stage_seconds": time.perf_counter() - stage_started,
        "committed": False,
    }


def _episode_record(
    item: dict[str, Any],
    *,
    episode_index: int,
    total_episodes: int,
    total_frames: int,
    lock_wait_seconds: float,
    lock_work_seconds: float,
    commit_seconds: float,
) -> dict[str, Any]:
    request = item["request"]
    return atomic._axis_episode_record(
        dataset_root=request["dataset_root"],
        repo_id=request["repo_id"],
        identity=request["identity"],
        episode_index=episode_index,
        frame_count=item["frame_count"],
        fps=int(request["fps"]),
        robot_type=request["robot_type"],
        camera_keys=request["camera_keys"],
        policy_state_dim=int(request["policy_state_dim"]),
        action_dim=int(request["action_dim"]),
        ee_state_dim=request.get("ee_state_dim"),
        policy_state_metadata=request["policy_state_metadata"],
        nq=int(request["nq"]),
        nv=int(request["nv"]),
        nu=int(request["nu"]),
        total_episodes=total_episodes,
        total_frames=total_frames,
        encode_seconds=item["encode_seconds"],
        stage_seconds=item["stage_seconds"],
        lock_wait_seconds=lock_wait_seconds,
        lock_work_seconds=lock_work_seconds,
        commit_seconds=commit_seconds,
    )


def _publish_initial_unlocked(
    item: dict[str, Any],
    *,
    lock_wait_seconds: float,
    lock_work_started: float,
) -> dict[str, Any]:
    request = item["request"]
    stage_root = item["stage_root"]
    identity = request["identity"]
    commit_started = time.perf_counter()
    result = _episode_record(
        item,
        episode_index=0,
        total_episodes=1,
        total_frames=item["frame_count"],
        lock_wait_seconds=lock_wait_seconds,
        lock_work_seconds=time.perf_counter() - lock_work_started,
        commit_seconds=0.0,
    )
    atomic._write_axis_sidecars(
        dataset_root=stage_root,
        identity=identity,
        metadata=request["metadata"],
        episode_record=result,
    )
    atomic.atomic_write_json(
        atomic._commit_state_path(stage_root),
        atomic._commit_state_payload(
            total_episodes=1,
            total_frames=item["frame_count"],
            repo_id=request["repo_id"],
            fps=int(request["fps"]),
            robot_type=request["robot_type"],
            vcodec=request.get("vcodec"),
            identity=identity,
            last_indices=atomic._last_indices_from_episode(
                item["source"]["episode"],
                atomic._video_keys_from_info(item["source"]["info"]),
            ),
        ),
    )
    result["commit"]["commit_seconds"] = time.perf_counter() - commit_started
    result["commit"]["lock_work_seconds"] = time.perf_counter() - lock_work_started
    atomic._write_axis_sidecars(
        dataset_root=stage_root,
        identity=identity,
        metadata=request["metadata"],
        episode_record=result,
    )
    atomic.atomic_write_json(
        atomic._attempt_commit_path(stage_root, identity),
        {
            "schema": atomic.ATOMIC_COMMIT_SCHEMA,
            "committed_at": datetime.now(timezone.utc).isoformat(),
            "transaction_id": "initial-directory-publish",
            "identity": atomic._identity_payload(identity),
            "episode": result,
            "metadata": request["metadata"],
        },
    )
    stage_manifest = atomic._shard_manifest_path(stage_root)
    if stage_manifest.exists():
        stage_manifest.unlink()
        atomic._fsync_directory(stage_manifest.parent)
    os.replace(stage_root, request["dataset_root"])
    atomic._fsync_directory(request["dataset_root"].parent)
    item["committed"] = True
    return result


def _replace_int_column(table: Any, name: str, values: np.ndarray) -> Any:
    import pyarrow as pa

    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Prepared data parquet is missing {name!r}.")
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def _write_remapped_data(
    *,
    source_table: Any,
    destination: Path,
    episode_index: int,
    frame_start: int,
    frame_count: int,
    task_index: int,
) -> None:
    import pyarrow.parquet as pq

    table = source_table
    table = _replace_int_column(
        table,
        "index",
        np.arange(frame_start, frame_start + frame_count, dtype=np.int64),
    )
    table = _replace_int_column(
        table,
        "episode_index",
        np.full(frame_count, episode_index, dtype=np.int64),
    )
    table = _replace_int_column(
        table,
        "task_index",
        np.full(frame_count, task_index, dtype=np.int64),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="snappy", use_dictionary=True)


def _write_episode_metadata(
    *,
    item: dict[str, Any],
    plan: dict[str, Any],
    shifted_stats: dict[str, dict[str, np.ndarray]],
    episode_index: int,
    frame_start: int,
    task_index: int,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = item["source"]
    request = item["request"]
    row = deepcopy(source["episode"])
    row.update(
        {
            "episode_index": episode_index,
            "tasks": [request["identity"].task_name],
            "length": item["frame_count"],
            "data/chunk_index": int(plan["data_indices"][0]),
            "data/file_index": int(plan["data_indices"][1]),
            "dataset_from_index": frame_start,
            "dataset_to_index": frame_start + item["frame_count"],
            "meta/episodes/chunk_index": int(plan["episode_indices"][0]),
            "meta/episodes/file_index": int(plan["episode_indices"][1]),
        }
    )
    for video_key, (chunk_index, file_index) in plan["video_indices"].items():
        row.update(
            {
                f"videos/{video_key}/chunk_index": int(chunk_index),
                f"videos/{video_key}/file_index": int(file_index),
                f"videos/{video_key}/from_timestamp": 0.0,
                f"videos/{video_key}/to_timestamp": float(source["video_durations"][video_key]),
            }
        )
    for feature, stats in shifted_stats.items():
        for name, value in stats.items():
            row[f"stats/{feature}/{name}"] = np.asarray(value).tolist()
    row["stats/task_index/min"] = [task_index]
    row["stats/task_index/max"] = [task_index]

    destination = request["dataset_root"] / plan["episode_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([row], schema=source["episode_schema"])
    pq.write_table(table, destination, compression="snappy", use_dictionary=True)
    return row


def _append_unlocked(
    item: dict[str, Any],
    *,
    aggregate_stats: Any,
    lock_wait_seconds: float,
    lock_work_started: float,
) -> dict[str, Any]:
    request = item["request"]
    dataset_root = request["dataset_root"]
    identity = request["identity"]
    info, state = atomic._load_lightweight_target_state(
        dataset_root,
        repo_id=request["repo_id"],
        expected_features=request["expected_features"],
        fps=int(request["fps"]),
        robot_type=request["robot_type"],
        vcodec=request.get("vcodec"),
    )
    atomic._assert_video_info_dict_match(item["source"]["info"], info)
    old_total_episodes = int(info["total_episodes"])
    old_total_frames = int(info["total_frames"])
    plan = atomic._plan_lightweight_append_paths(
        dataset_root,
        info=info,
        state=state,
        identity=identity,
    )
    transaction = atomic._prepare_transaction(
        dataset_root=dataset_root,
        identity=identity,
        plan=plan,
        repo_id=request["repo_id"],
        fps=int(request["fps"]),
        robot_type=request["robot_type"],
        vcodec=request.get("vcodec"),
        old_total_episodes=old_total_episodes,
        old_total_frames=old_total_frames,
        frame_count=item["frame_count"],
        stage_root=item["stage_root"],
        shard_root=request["shard_root"],
    )
    atomic._maybe_crash(request.get("test_crash_phase"), "after_transaction")
    commit_started = time.perf_counter()
    marker_written = False
    try:
        task_index = atomic._task_index_for_name(dataset_root, identity.task_name)
        _write_remapped_data(
            source_table=item["source"]["data_table"],
            destination=dataset_root / plan["data_path"],
            episode_index=old_total_episodes,
            frame_start=old_total_frames,
            frame_count=item["frame_count"],
            task_index=task_index,
        )
        atomic._maybe_crash(request.get("test_crash_phase"), "after_data")

        for video_key, destination_relative in plan["video_paths"].items():
            staged_video = item["stage_root"] / str(item["manifest"]["video_paths"][video_key])
            destination = dataset_root / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_video, destination)
        atomic._maybe_crash(request.get("test_crash_phase"), "after_videos")

        shifted_stats = atomic._shift_stats(
            item["source"]["stats"],
            episode_index=old_total_episodes,
            frame_start=old_total_frames,
            task_index=task_index,
        )
        episode_row = _write_episode_metadata(
            item=item,
            plan=plan,
            shifted_stats=shifted_stats,
            episode_index=old_total_episodes,
            frame_start=old_total_frames,
            task_index=task_index,
        )
        target_stats = atomic._load_stats_json(dataset_root)
        combined_stats = aggregate_stats([target_stats, shifted_stats])
        new_total_episodes = old_total_episodes + 1
        new_total_frames = old_total_frames + item["frame_count"]
        new_info = deepcopy(info)
        new_info["total_episodes"] = new_total_episodes
        new_info["total_frames"] = new_total_frames
        new_info["splits"] = {"train": f"0:{new_total_episodes}"}
        atomic.atomic_write_json(dataset_root / "meta" / "info.json", new_info)
        atomic.atomic_write_json(dataset_root / "meta" / "stats.json", combined_stats)
        atomic._maybe_crash(request.get("test_crash_phase"), "after_save_episode")

        atomic._validate_written_append(
            dataset_root=dataset_root,
            plan=plan,
            episode_index=old_total_episodes,
            frame_start=old_total_frames,
            frame_count=item["frame_count"],
            task_index=task_index,
            expected_total_episodes=new_total_episodes,
            expected_total_frames=new_total_frames,
        )
        atomic._maybe_crash(request.get("test_crash_phase"), "after_meta")
        commit_seconds = time.perf_counter() - commit_started
        result = _episode_record(
            item,
            episode_index=old_total_episodes,
            total_episodes=new_total_episodes,
            total_frames=new_total_frames,
            lock_wait_seconds=lock_wait_seconds,
            lock_work_seconds=time.perf_counter() - lock_work_started,
            commit_seconds=commit_seconds,
        )
        atomic._write_axis_sidecars(
            dataset_root=dataset_root,
            identity=identity,
            metadata=request["metadata"],
            episode_record=result,
        )
        atomic.atomic_write_json(
            atomic._commit_state_path(dataset_root),
            atomic._commit_state_payload(
                total_episodes=new_total_episodes,
                total_frames=new_total_frames,
                repo_id=request["repo_id"],
                fps=int(request["fps"]),
                robot_type=request["robot_type"],
                vcodec=request.get("vcodec"),
                identity=identity,
                last_indices=atomic._last_indices_from_episode(
                    episode_row,
                    atomic._video_keys_from_info(new_info),
                ),
            ),
        )
        durable_paths = [
            *(dataset_root / path for path in plan["created_files"] if path != plan["marker_path"]),
            *(dataset_root / path for path in atomic._TRANSACTION_MUTABLE_FILES),
        ]
        for path in durable_paths:
            if path.is_file():
                atomic._fsync_file(path)
                atomic._fsync_directory(path.parent)
        atomic._maybe_crash(request.get("test_crash_phase"), "after_sidecars")
        atomic.atomic_write_json(
            atomic._attempt_commit_path(dataset_root, identity),
            {
                "schema": atomic.ATOMIC_COMMIT_SCHEMA,
                "committed_at": datetime.now(timezone.utc).isoformat(),
                "transaction_id": transaction["transaction_id"],
                "identity": atomic._identity_payload(identity),
                "episode": result,
                "metadata": request["metadata"],
            },
        )
        marker_written = True
        atomic._maybe_crash(request.get("test_crash_phase"), "after_marker")
        atomic._cleanup_transaction_directory(
            dataset_root,
            atomic._transaction_path(dataset_root, identity).parent,
        )
        item["committed"] = True
        return result
    except Exception:
        if marker_written:
            raise
        atomic._rollback_transaction_unlocked(dataset_root, transaction)
        raise


def _commit_one_dataset(requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not requests:
        return []
    signatures = {_request_signature(request) for request in requests}
    if len(signatures) != 1:
        raise ValueError("Batched commits to one dataset must share repo/config/features.")
    identity_keys = [request["identity"].key for request in requests]
    if len(identity_keys) != len(set(identity_keys)):
        raise ValueError("Batched commits cannot contain duplicate episode identities.")

    _, api = atomic._load_lerobot_api()
    results: list[dict[str, Any] | None] = [None] * len(requests)
    prepared: list[tuple[int, dict[str, Any]]] = []
    dataset_root = requests[0]["dataset_root"]
    with ExitStack() as attempt_locks:
        for request in sorted(requests, key=lambda item: item["identity"].key):
            attempt_locks.enter_context(
                atomic.exclusive_lerobot_attempt_lock(dataset_root, request["identity"])
            )
        try:
            with atomic.exclusive_lerobot_dataset_lock(dataset_root):
                atomic._recover_pending_transactions_unlocked(dataset_root)
                for index, request in enumerate(requests):
                    existing = atomic._load_committed_attempt_unlocked(
                        dataset_root,
                        request["identity"],
                    )
                    if existing is not None:
                        results[index] = existing["episode"]

            for index, request in enumerate(requests):
                if results[index] is None:
                    prepared.append((index, _prepare_request(request)))

            if prepared:
                lock_started = time.perf_counter()
                with atomic.exclusive_lerobot_dataset_lock(dataset_root):
                    lock_acquired = time.perf_counter()
                    atomic._recover_pending_transactions_unlocked(dataset_root)
                    for index, item in prepared:
                        request = item["request"]
                        existing = atomic._load_committed_attempt_unlocked(
                            dataset_root,
                            request["identity"],
                        )
                        if existing is not None:
                            results[index] = existing["episode"]
                            item["committed"] = True
                            continue
                        item_started = time.perf_counter()
                        if not dataset_root.exists():
                            results[index] = _publish_initial_unlocked(
                                item,
                                lock_wait_seconds=lock_acquired - lock_started,
                                lock_work_started=item_started,
                            )
                        else:
                            results[index] = _append_unlocked(
                                item,
                                aggregate_stats=api["aggregate_stats"],
                                lock_wait_seconds=lock_acquired - lock_started,
                                lock_work_started=item_started,
                            )
        finally:
            for request, result in zip(requests, results, strict=True):
                if result is not None:
                    _remove_tree_if_present(request["shard_root"])
            for _, item in prepared:
                stage_root = item["stage_root"]
                staging_base = dataset_root.parent / ".axis_lerobot_commit_staging"
                atomic._remove_tree_and_empty_parents(stage_root, stop=staging_base)
    if any(result is None for result in results):
        raise RuntimeError("Batched atomic commit did not produce every episode record.")
    return [result for result in results if result is not None]


def commit_episode_shards(commits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Commit prepared shards, acquiring each dataset lock once per batch."""

    normalized = [_normalize_request(request) for request in commits]
    grouped: dict[Path, list[tuple[int, dict[str, Any]]]] = {}
    for index, request in enumerate(normalized):
        grouped.setdefault(request["dataset_root"], []).append((index, request))
    results: list[dict[str, Any] | None] = [None] * len(normalized)
    for group in grouped.values():
        group_results = _commit_one_dataset([request for _, request in group])
        for (index, _), result in zip(group, group_results, strict=True):
            results[index] = result
    return [result for result in results if result is not None]


def commit_episode_shard(
    *,
    shard_root: Path,
    dataset_root: Path,
    repo_id: str,
    expected_features: dict[str, Any],
    identity: atomic.EpisodeIdentity,
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
    return commit_episode_shards(
        [
            {
                "shard_root": shard_root,
                "dataset_root": dataset_root,
                "repo_id": repo_id,
                "expected_features": expected_features,
                "identity": identity,
                "metadata": metadata,
                "fps": fps,
                "robot_type": robot_type,
                "vcodec": vcodec,
                "camera_keys": camera_keys,
                "policy_state_dim": policy_state_dim,
                "action_dim": action_dim,
                "ee_state_dim": ee_state_dim,
                "policy_state_metadata": policy_state_metadata,
                "nq": nq,
                "nv": nv,
                "nu": nu,
                "test_crash_phase": test_crash_phase,
            }
        ]
    )[0]
