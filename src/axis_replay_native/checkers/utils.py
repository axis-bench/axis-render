"""Shared helper functions for backend checkers.

Includes:
- Fuzzy name resolution (mirrors frontend body-name matching)
- Quaternion math utilities
- 3x3 rotation matrix utilities
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger("app.checkers")


# ---------------------------------------------------------------------------
# Name resolution helpers
# ---------------------------------------------------------------------------

def _resolve_key(name: str, mapping: dict) -> str | None:
    """Resolve *name* against the keys of *mapping*.

    Match order (mirrors frontend):
      1. Exact match
      2. Key ends with ``"/" + name``
      3. Key ends with ``name``
    """
    if name in mapping:
        return name
    suffix_slash = "/" + name
    for key in mapping:
        if key.endswith(suffix_slash):
            return key
    for key in mapping:
        if key.endswith(name):
            return key
    return None


def resolve_position(name: str, positions: dict[str, list[float]]) -> list[float] | None:
    key = _resolve_key(name, positions)
    return positions[key] if key is not None else None


def resolve_orientation(name: str, orientations: dict[str, list[float]]) -> list[float] | None:
    key = _resolve_key(name, orientations)
    return orientations[key] if key is not None else None


def resolve_joint(name: str, joints: dict[str, float]) -> float | None:
    key = _resolve_key(name, joints)
    return joints[key] if key is not None else None


def resolve_site_position(name: str, site_positions: dict[str, list[float]]) -> list[float] | None:
    key = _resolve_key(name, site_positions)
    return site_positions[key] if key is not None else None


def resolve_rot_mat(name: str, rot_mats: dict[str, list[float]]) -> list[float] | None:
    key = _resolve_key(name, rot_mats)
    return rot_mats[key] if key is not None else None


# ---------------------------------------------------------------------------
# Config key helpers  (accept both camelCase and snake_case)
# ---------------------------------------------------------------------------

def cfg(config: dict, *keys, default=None):
    """Return the first existing key's value from *config*, else *default*."""
    for k in keys:
        if k in config:
            return config[k]
    return default


# ---------------------------------------------------------------------------
# Quaternion math  (quaternions stored as [x, y, z, w])
# ---------------------------------------------------------------------------

def quat_inverse(q: list[float]) -> list[float]:
    """Return the inverse (conjugate) of a unit quaternion [x,y,z,w]."""
    return [-q[0], -q[1], -q[2], q[3]]


def quat_multiply(a: list[float], b: list[float]) -> list[float]:
    """Hamilton product of two quaternions [x,y,z,w]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def quat_normalize(q: list[float]) -> list[float]:
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [c / n for c in q]


def quat_angle_deg(q1: list[float], q2: list[float]) -> float:
    """Rotation angle in degrees between two unit quaternions [x,y,z,w]."""
    q_rel = quat_multiply(q2, quat_inverse(q1))
    w = abs(q_rel[3])
    w = min(1.0, max(-1.0, w))
    return math.degrees(2.0 * math.acos(w))


def quat_rotate_vec(q: list[float], v: list[float]) -> list[float]:
    """Rotate vector *v* by quaternion *q* [x,y,z,w]."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # result = v + qw * t + cross(q.xyz, t)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


# ---------------------------------------------------------------------------
# 3x3 rotation matrix utilities  (row-major flat list of 9 floats)
# ---------------------------------------------------------------------------

def mat3_mul_vec(m: list[float], v: list[float]) -> list[float]:
    """R * v  (row-major 3x3 matrix times 3-vector)."""
    return [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]


def mat3_transpose_mul_vec(m: list[float], v: list[float]) -> list[float]:
    """R^T * v  (transpose of row-major 3x3 matrix times 3-vector)."""
    return [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]


def mat3_mul_mat(a: list[float], b: list[float]) -> list[float]:
    """Matrix product of two row-major 3x3 matrices."""
    result = [0.0] * 9
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += a[i * 3 + k] * b[k * 3 + j]
            result[i * 3 + j] = s
    return result


def quat_to_mat3(q: list[float]) -> list[float]:
    """Convert quaternion [w, x, y, z] (wxyz order) to row-major 3x3 matrix.

    Note: this function expects **wxyz** order because DrawerBBoxChecker config
    stores ``relativeQuat`` as ``[w, x, y, z]``.
    """
    w, x, y, z = q
    return [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]


def vec3_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vec3_dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_length(v: list[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_add(a: list[float], b: list[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec3_scale(v: list[float], s: float) -> list[float]:
    return [v[0] * s, v[1] * s, v[2] * s]
