"""Small geometry helpers shared by the pipeline.

Conventions
-----------
* Camera space follows OpenCV: x right, y down, z forward, in metres, in the
  sensor's native (landscape) image orientation.
* Head space follows ARKit's face anchor: origin roughly at the centre of the
  head, +y up, +z out of the face (towards the camera when looking at it),
  +x towards the face's own left.
* A pose ``T_hc`` maps camera points to head space: p_head = R @ p_cam + t.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .capture import Intrinsics


def backproject(depth: np.ndarray, k: Intrinsics) -> np.ndarray:
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    x = (u - k.cx) / k.fx * depth
    y = (v - k.cy) / k.fy * depth
    return np.stack([x, y, depth], axis=-1)


def clean_depth(depth: np.ndarray, min_d: float, max_d: float, jump_rel: float = 0.015) -> np.ndarray:
    """Return a copy of ``depth`` with out-of-range and 'flying' pixels zeroed.

    Flying pixels sit between foreground and background at depth edges; they
    must go, or they smear silhouettes and carve real surface away.
    """
    d = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
    valid = (d > min_d) & (d < max_d)
    thr = jump_rel * d + 0.002
    edge = np.zeros_like(valid)
    for axis in (0, 1):
        for shift in (1, -1):
            nb = np.roll(d, shift, axis=axis)
            nb_valid = np.roll(valid, shift, axis=axis)
            edge |= (np.abs(d - nb) > thr) | ~nb_valid
    valid &= ~edge
    valid = ndimage.binary_erosion(valid, iterations=1, border_value=0)
    return np.where(valid, d, 0).astype(np.float32)


def vertex_normals(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel normals of an organised point map, oriented towards the camera."""
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1)
    ok = valid.copy()
    ok[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    ok[1:-1] &= valid[2:] & valid[:-2]
    ok[[0, -1], :] = False
    ok[:, [0, -1]] = False
    ok &= norm > 1e-12
    n = n / np.maximum(norm, 1e-12)[..., None]
    flip = (n * points).sum(-1) > 0
    n[flip] *= -1
    return n.astype(np.float32), ok


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Best rigid (R, t) with dst ~ R @ src + t.

    Returns (R, t, reflection_sign, max_residual). A reflection_sign of -1
    means the best orthogonal fit was a mirror image.
    """
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    sign = float(np.sign(np.linalg.det(Vt.T @ U.T))) or 1.0
    D = np.diag([1.0, 1.0, sign])
    R = Vt.T @ D @ U.T
    t = cd - R @ cs
    resid = float(np.linalg.norm(src @ R.T + t - dst, axis=1).max())
    return R, t, sign, resid


def pose_from_head_points(
    uvd: np.ndarray, k_color: Intrinsics, color_width: int, axis_length: float
) -> tuple[np.ndarray, bool] | None:
    """Head pose from ARKit's projected head-axis points.

    The phone sends the head origin and three axis tips projected into the
    colour image, plus their depths. Re-lifting them with the colour
    intrinsics gives the pose in the same camera frame as the depth data,
    without depending on ARKit's own camera-axis conventions. If the
    projection turns out to be mirrored (front camera), the mirror is undone.

    Returns (T_hc, was_mirrored) or None if the points are inconsistent.
    """
    a = axis_length
    local = np.array([[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a]], dtype=np.float64)
    d = uvd[:, 2]
    if np.any(d <= 0.05) or np.any(d > 3):
        return None
    for mirrored in (False, True):
        u = (color_width - 1) - uvd[:, 0] if mirrored else uvd[:, 0]
        v = uvd[:, 1]
        cam = np.stack([(u - k_color.cx) / k_color.fx * d, (v - k_color.cy) / k_color.fy * d, d], axis=1)
        R, t, sign, resid = kabsch(local, cam)
        if sign > 0 and resid < 0.15 * a:
            return invert(make_T(R, t)), mirrored
    return None


def rotation_from_vector(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def rotation_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def voxel_average(points: np.ndarray, normals: np.ndarray, voxel: float, min_count: int = 1):
    """Average points and normals per voxel. Returns (points, normals, counts)."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = counts.size
    p = np.zeros((n, 3))
    nn = np.zeros((n, 3))
    for c in range(3):
        p[:, c] = np.bincount(inv, weights=points[:, c], minlength=n)
        nn[:, c] = np.bincount(inv, weights=normals[:, c], minlength=n)
    p /= counts[:, None]
    norm = np.linalg.norm(nn, axis=1)
    keep = (counts >= min_count) & (norm > 0.3 * counts)  # normals must roughly agree
    nn = nn[keep] / norm[keep, None]
    return p[keep], nn, counts[keep]
