from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CheckerRuntimePayload(BaseModel):
    """Standardised runtime input accepted by every backend checker."""

    # Body centre-of-mass positions  {"body_name": [x, y, z]}
    object_positions: dict[str, list[float]] = Field(default_factory=dict)

    # Body orientations as quaternions [x, y, z, w]
    object_orientations: dict[str, list[float]] | None = None

    # Joint positions (qpos)  {"joint_name": float}
    joint_positions: dict[str, float] | None = None

    # Site positions  {"site_name": [x, y, z]}
    site_positions: dict[str, list[float]] | None = None

    # Site rotation matrices (3x3 flattened, length 9)
    site_rot_mats: dict[str, list[float]] | None = None

    # Body rotation matrices (3x3 flattened, length 9)
    body_rot_mats: dict[str, list[float]] | None = None

    # Keyboard keys pressed during the trajectory
    pressed_keys: set[str] | None = None

    # Initial positions (for delta checkers)
    initial_object_positions: dict[str, list[float]] | None = None

    # Initial orientations (for rotation checkers)
    initial_object_orientations: dict[str, list[float]] | None = None

    # Reserved for future use
    extras: dict[str, Any] | None = None
