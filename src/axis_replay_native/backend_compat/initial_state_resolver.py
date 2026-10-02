from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MASK32 = 0xFFFFFFFF
INITIAL_STATE_SET_SALT = "initial_state:v1"


@dataclass(frozen=True)
class ResolvedInitialState:
    initial_state: dict[str, Any] | None
    index: int | None = None
    name: str | None = None
    count: int = 0


def _fnv1a32(text: str) -> int:
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = (h * 0x01000193) & _MASK32
    return h


def _normalize_submit_nonce_text(submit_nonce: int | str) -> str:
    text = str(submit_nonce).strip()
    if not text.isdigit():
        raise ValueError("submit_nonce must be an 8-digit numeric string")
    value = int(text)
    if value < 0 or value > 99_999_999:
        raise ValueError("submit_nonce must be between 00000000 and 99999999")
    return f"{value:08d}"


def derive_initial_state_index(task_id: int, submit_nonce: int | str, count: int) -> int | None:
    if count <= 0:
        return None
    nonce_text = _normalize_submit_nonce_text(submit_nonce)
    return _fnv1a32(f"{int(task_id)}:{nonce_text}:{INITIAL_STATE_SET_SALT}") % int(count)


def is_initial_state_set(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("items"), list)


def _initial_state_from_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    selected = item.get("initial_state")
    if isinstance(selected, dict):
        return selected
    if isinstance(item.get("robots"), dict) or isinstance(item.get("objects"), dict):
        return item
    return None


def resolve_initial_state(
    initial_state: Any,
    *,
    task_id: int,
    submit_nonce: int | str | None,
) -> ResolvedInitialState:
    if not is_initial_state_set(initial_state):
        return ResolvedInitialState(initial_state if isinstance(initial_state, dict) else None)

    items = initial_state.get("items") or []
    count = len(items)
    if count <= 0:
        return ResolvedInitialState(None, count=0)

    if submit_nonce is None:
        index = 0
    else:
        index = derive_initial_state_index(task_id, submit_nonce, count)
        if index is None:
            index = 0

    item = items[int(index)]
    selected = _initial_state_from_item(item)
    return ResolvedInitialState(
        selected,
        index=int(index),
        name=item.get("name") if isinstance(item, dict) and isinstance(item.get("name"), str) else None,
        count=count,
    )
