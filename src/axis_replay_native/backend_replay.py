"""Resolve AXIS scene variants with the bundled `backend_compat` resolver."""

from __future__ import annotations

from .backend_compat.scene_variant_resolver import ResolvedSceneVariant, resolve_scene_variant_fields

__all__ = ["ResolvedSceneVariant", "resolve_scene_variant_fields"]
