from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .initial_state_resolver import resolve_initial_state
from .submit_nonce import derive_scene_variant_index


@dataclass(frozen=True)
class ResolvedSceneVariant:
    mjcf_xml: str | None
    initial_state: dict[str, Any] | None
    domain_randomization: Any | None = None
    has_domain_randomization: bool = False
    scene_variant_index: int | None = None
    scene_variant_name: str | None = None


def _frontend_scene_variants_from_heavy(heavy: dict[str, Any]) -> list[dict[str, Any]] | None:
    scene_variants = heavy.get("scene_variants")
    if isinstance(scene_variants, list):
        return scene_variants

    checker_config = heavy.get("checker_config")
    if isinstance(checker_config, dict):
        layout_variants = checker_config.get("layout_variants")
        if isinstance(layout_variants, list):
            return layout_variants
    return None


def resolve_scene_variant_fields(
    *,
    task_id: int,
    heavy: dict[str, Any],
    submit_nonce: int | str | None,
) -> ResolvedSceneVariant:
    mjcf_xml = heavy.get("mjcf_xml")
    initial_state = heavy.get("initial_state")
    domain_randomization = heavy.get("domain_randomization")
    has_domain_randomization = "domain_randomization" in heavy

    scene_variant_index: int | None = None
    scene_variant_name: str | None = None
    selected_variant = heavy.get("_selected_scene_variant")
    selected_variant_index = heavy.get("_selected_scene_variant_index")
    if isinstance(selected_variant, dict):
        try:
            scene_variant_index = int(selected_variant_index) if selected_variant_index is not None else None
        except (TypeError, ValueError):
            scene_variant_index = None
        scene_variant_name = selected_variant.get("name") if isinstance(selected_variant.get("name"), str) else None
        if selected_variant.get("mjcf_xml"):
            mjcf_xml = selected_variant.get("mjcf_xml")
        if selected_variant.get("initial_state"):
            initial_state = selected_variant.get("initial_state")
        if "domain_randomization" in selected_variant:
            domain_randomization = selected_variant.get("domain_randomization")
            has_domain_randomization = True
        resolved_initial_state = resolve_initial_state(initial_state, task_id=task_id, submit_nonce=submit_nonce)
        return ResolvedSceneVariant(
            mjcf_xml=mjcf_xml,
            initial_state=resolved_initial_state.initial_state,
            domain_randomization=domain_randomization,
            has_domain_randomization=has_domain_randomization,
            scene_variant_index=scene_variant_index,
            scene_variant_name=scene_variant_name,
        )

    scene_variants = _frontend_scene_variants_from_heavy(heavy)
    if scene_variants and submit_nonce is not None:
        scene_variant_index = derive_scene_variant_index(task_id, submit_nonce, len(scene_variants))
        if scene_variant_index is not None:
            variant = scene_variants[scene_variant_index]
            if isinstance(variant, dict):
                scene_variant_name = variant.get("name") if isinstance(variant.get("name"), str) else None
                if variant.get("mjcf_xml"):
                    mjcf_xml = variant.get("mjcf_xml")
                if variant.get("initial_state"):
                    initial_state = variant.get("initial_state")
                if "domain_randomization" in variant:
                    domain_randomization = variant.get("domain_randomization")
                    has_domain_randomization = True

    resolved_initial_state = resolve_initial_state(initial_state, task_id=task_id, submit_nonce=submit_nonce)
    return ResolvedSceneVariant(
        mjcf_xml=mjcf_xml,
        initial_state=resolved_initial_state.initial_state,
        domain_randomization=domain_randomization,
        has_domain_randomization=has_domain_randomization,
        scene_variant_index=scene_variant_index,
        scene_variant_name=scene_variant_name,
    )
