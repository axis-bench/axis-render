from __future__ import annotations

from typing import Any

_MASK32 = (1 << 32) - 1


def normalize_submit_nonce_text(submit_nonce: int | str) -> str:
    text = str(submit_nonce).strip()
    if not text or not text.isdigit():
        raise ValueError("submit_nonce must be an 8-digit numeric string")
    value = int(text)
    if value < 0 or value >= 100_000_000:
        raise ValueError("submit_nonce must be between 00000000 and 99999999")
    return f"{value:08d}"


def _fnv1a32_bytes(text: str) -> int:
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & _MASK32
    return h


def derive_scene_variant_index(task_id: int, submit_nonce: int | str, count: int) -> int | None:
    if count <= 0:
        return None
    nonce_text = normalize_submit_nonce_text(submit_nonce)
    text = f"{int(task_id)}:{nonce_text}:scene_variant:v1"
    return _fnv1a32_bytes(text) % count
