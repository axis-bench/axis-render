"""KeyPressSetChecker: checks if all required keys were pressed."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseChecker
from .runtime_types import CheckerRuntimePayload
from .utils import cfg

logger = logging.getLogger("app.checkers")

DEFAULT_REQUIRED_KEYS = [
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "KeyE", "KeyD", "KeyQ", "KeyW", "KeyA", "KeyS",
    "KeyZ", "KeyX", "Space", "KeyR",
]


class KeyPressSetChecker(BaseChecker):
    """Check if all required keyboard keys have been pressed."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.required_keys: list[str] = cfg(
            config, "requiredKeys", "required_keys", default=DEFAULT_REQUIRED_KEYS,
        )

    def check(self, payload: CheckerRuntimePayload) -> bool:
        if payload.pressed_keys is None:
            logger.warning("KeyPressSetChecker: pressed_keys not provided")
            return False

        missing = [key for key in self.required_keys if key not in payload.pressed_keys]
        if missing:
            logger.info(
                "KeyPressSetChecker FAIL: missing keys: %s  (pressed: %s)",
                ", ".join(missing), ", ".join(sorted(payload.pressed_keys)),
            )
            return False
        return True

    def check_detail(self, payload: CheckerRuntimePayload) -> tuple[bool, dict[str, Any]]:
        detail: dict[str, Any] = {"checker_type": "KeyPressSetChecker", "passed": False}
        if payload.pressed_keys is None:
            detail.update(required_keys=self.required_keys, pressed_keys=[], missing_keys=self.required_keys)
            return False, detail

        pressed = sorted(payload.pressed_keys)
        missing = [key for key in self.required_keys if key not in payload.pressed_keys]
        passed = len(missing) == 0
        detail.update(
            passed=passed,
            required_keys=self.required_keys,
            pressed_keys=pressed,
            missing_keys=missing,
        )
        return passed, detail
