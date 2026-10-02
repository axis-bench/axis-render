"""Resolve task runtime fields from trajectory metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .backend_replay import ResolvedSceneVariant, resolve_scene_variant_fields


SCENE_VARIANT_METADATA_KEYS = (
    "scene_variant_index",
    "sceneVariantIndex",
    "layout_variant_index",
    "layoutVariantIndex",
)


def extract_trajectory_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    task_trajectory = bundle.get("task_trajectory") or {}
    for key in ("metadata", "trajectory_metadata"):
        metadata = task_trajectory.get(key)
        if isinstance(metadata, dict):
            return metadata

    trajectory_data = task_trajectory.get("trajectory_data")
    if isinstance(trajectory_data, dict):
        metadata = trajectory_data.get("metadata")
        if isinstance(metadata, dict):
            return metadata

    return {}


def extract_submit_nonce(bundle: dict[str, Any]) -> Any | None:
    metadata = extract_trajectory_metadata(bundle)
    for key in ("submit_nonce", "submitNonce"):
        if key in metadata and metadata.get(key) is not None:
            return metadata.get(key)
    return None


def _task_id_from_bundle(bundle: dict[str, Any]) -> int:
    task_payload = bundle.get("task") or {}
    task_id = task_payload.get("id") or (bundle.get("attempt") or {}).get("task_id")
    if task_id is None:
        raise ValueError("Bundle is missing task id; cannot resolve runtime task fields.")
    return int(task_id)


def _normalize_cache_revision(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    else:
        return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def cache_revision_from_task_payload(task_payload: dict[str, Any]) -> str:
    return _normalize_cache_revision(
        task_payload.get("_cache_revision")
        or task_payload.get("cache_revision")
        or task_payload.get("updated_at")
    )


def _frontend_scene_variants_from_task_payload(task_payload: dict[str, Any]) -> list[Any]:
    scene_variants = task_payload.get("scene_variants")
    if isinstance(scene_variants, list):
        return scene_variants

    checker_config = task_payload.get("checker_config")
    if isinstance(checker_config, dict):
        layout_variants = checker_config.get("layout_variants")
        if isinstance(layout_variants, list):
            return layout_variants
    return []


def _metadata_scene_variant_selection(
    task_payload: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[bool, int | None, dict[str, Any] | None]:
    scene_variants = _frontend_scene_variants_from_task_payload(task_payload)
    for key in SCENE_VARIANT_METADATA_KEYS:
        if key not in metadata or metadata.get(key) is None:
            continue
        try:
            scene_variant_index = int(metadata[key])
        except (TypeError, ValueError):
            return True, None, None
        if not (0 <= scene_variant_index < len(scene_variants)):
            return True, None, None
        variant = scene_variants[scene_variant_index]
        if not isinstance(variant, dict):
            return True, scene_variant_index, None
        return True, scene_variant_index, variant
    return False, None, None


def resolve_bundle_runtime_fields(bundle: dict[str, Any]) -> ResolvedSceneVariant:
    task_payload = bundle.get("task") or {}
    metadata = extract_trajectory_metadata(bundle)
    submit_nonce = extract_submit_nonce(bundle)
    heavy = {
        "mjcf_xml": task_payload.get("mjcf_xml"),
        "checker_config": task_payload.get("checker_config"),
        "initial_state": task_payload.get("initial_state"),
        "scene_variants": task_payload.get("scene_variants"),
    }
    if "domain_randomization" in task_payload:
        heavy["domain_randomization"] = task_payload.get("domain_randomization")
    explicit_variant, scene_variant_index, selected_variant = _metadata_scene_variant_selection(
        task_payload,
        metadata,
    )
    if explicit_variant:
        heavy["_selected_scene_variant"] = selected_variant or {}
        heavy["_selected_scene_variant_index"] = scene_variant_index
    return resolve_scene_variant_fields(
        heavy=heavy,
        task_id=_task_id_from_bundle(bundle),
        submit_nonce=submit_nonce,
    )


def apply_backend_runtime_task_fields(bundle: dict[str, Any]) -> dict[str, Any]:
    runtime_fields = resolve_bundle_runtime_fields(bundle)
    task_payload = dict(bundle.get("task") or {})
    task_payload["mjcf_xml"] = runtime_fields.mjcf_xml
    task_payload["initial_state"] = runtime_fields.initial_state
    task_payload["_scene_variant_index"] = runtime_fields.scene_variant_index
    task_payload["_scene_variant_name"] = runtime_fields.scene_variant_name
    task_payload["_scene_variant_source"] = "backend_resolver"
    task_payload["_submit_nonce"] = extract_submit_nonce(bundle)
    task_payload["_has_domain_randomization"] = bool(runtime_fields.has_domain_randomization)
    task_payload["_domain_randomization"] = runtime_fields.domain_randomization
    task_payload["_cache_revision"] = cache_revision_from_task_payload(task_payload)
    if runtime_fields.has_domain_randomization:
        task_payload["domain_randomization"] = runtime_fields.domain_randomization
    bundle["task"] = task_payload
    return bundle


def runtime_variant_key(bundle: dict[str, Any]) -> tuple[int | None, str | None]:
    task_payload = bundle.get("task") or {}
    index = task_payload.get("_scene_variant_index")
    return (int(index) if index is not None else None, task_payload.get("_scene_variant_name"))


def runtime_metadata_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    task_payload = bundle.get("task") or {}
    index = task_payload.get("_scene_variant_index")
    return {
        "submit_nonce": task_payload.get("_submit_nonce", extract_submit_nonce(bundle)),
        "scene_variant_index": int(index) if index is not None else None,
        "scene_variant_name": task_payload.get("_scene_variant_name"),
        "scene_variant_source": task_payload.get("_scene_variant_source"),
        "has_domain_randomization": bool(task_payload.get("_has_domain_randomization")),
        "domain_randomization": task_payload.get("_domain_randomization"),
        "cache_revision": cache_revision_from_task_payload(task_payload),
    }
