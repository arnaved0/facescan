"""Aligning depth frames into one head-centred coordinate system.

ARKit's per-frame head pose is a good starting point but is only accurate to a
few millimetres. The poses are refined by repeatedly registering every frame to
a model built from all frames (point-to-plane ICP with a robust loss), with the
match distance shrinking each round. Without ARKit poses, frames are first
chained with frame-to-frame ICP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from .capture import Capture
from .geometry import (
    apply,
    backproject,
    clean_depth,
    make_T,
    pose_from_head_points,
    rotation_from_vector,
    vertex_normals,
    voxel_average,
)

# Region kept around the head, in head space (metres). Generous on purpose:
# it only has to exclude shoulders and the room.
HEAD_BOX = np.array([[-0.12, -0.17, -0.14], [0.12, 0.16, 0.16]])

HEAD_DEPTH_RANGE = (0.12, 0.9)
MODEL_VOXEL = 0.0012
SAMPLES_PER_FRAME = 30000
ICP_POINTS = 6000
MIN_FITNESS = 0.45
MAX_RMSE = 0.0018

Progress = Callable[[float], None]


@dataclass
class FrameSamples:
    points: np.ndarray  # (N, 3) camera space
    normals: np.ndarray  # (N, 3) camera space


@dataclass
class IcpStats:
    fitness: float  # fraction of source points with a valid match
    rmse: float  # point-to-plane RMS of matches, metres
    matches: int


@dataclass
class Registration:
    poses: np.ndarray  # (F, 4, 4) T_hc
    active: np.ndarray  # (F,) bool, frames used for fusion
    fitness: np.ndarray
    rmse: np.ndarray
    pose_source: str  # "arkit" or "depth-only"
    mirrored_projection: bool
    model_points: np.ndarray
    model_normals: np.ndarray


def in_box(p: np.ndarray, box: np.ndarray = HEAD_BOX) -> np.ndarray:
    return np.all((p >= box[0]) & (p <= box[1]), axis=1)


def head_depth(capture: Capture, frame) -> np.ndarray:
    return clean_depth(capture.load_depth(frame), *HEAD_DEPTH_RANGE)


def sample_frame(depth: np.ndarray, k, rng: np.random.Generator) -> FrameSamples:
    pts = backproject(depth, k)
    normals, ok = vertex_normals(pts, depth > 0)
    idx = np.flatnonzero(ok.ravel())
    if idx.size > SAMPLES_PER_FRAME:
        idx = rng.choice(idx, SAMPLES_PER_FRAME, replace=False)
    return FrameSamples(
        pts.reshape(-1, 3)[idx].astype(np.float64),
        normals.reshape(-1, 3)[idx].astype(np.float64),
    )


def icp(
    src_p: np.ndarray,
    src_n: np.ndarray,
    tree: cKDTree,
    tgt_p: np.ndarray,
    tgt_n: np.ndarray,
    T: np.ndarray,
    max_dist: float,
    rng: np.random.Generator,
    iters: int = 25,
) -> tuple[np.ndarray, IcpStats]:
    """Robust point-to-plane ICP. Returns the refined T (source -> target)."""
    if len(src_p) < 100:
        return T, IcpStats(0.0, np.inf, 0)
    if len(src_p) > ICP_POINTS:
        sel = rng.choice(len(src_p), ICP_POINTS, replace=False)
        src_p, src_n = src_p[sel], src_n[sel]
    T = T.copy()
    scale = max_dist / 3
    for _ in range(iters):
        p = apply(T, src_p)
        ns = src_n @ T[:3, :3].T
        dist, idx = tree.query(p, distance_upper_bound=max_dist, workers=-1)
        m = np.isfinite(dist)
        idx_m = idx[m]
        compat = (ns[m] * tgt_n[idx_m]).sum(1) > 0.6
        pm, q, nq = p[m][compat], tgt_p[idx_m[compat]], tgt_n[idx_m[compat]]
        if len(pm) < 100:
            return T, IcpStats(0.0, np.inf, len(pm))
        r = ((pm - q) * nq).sum(1)
        w = 1.0 / (1.0 + (r / scale) ** 2)
        A = np.concatenate([np.cross(pm, nq), nq], axis=1)
        Aw = A * w[:, None]
        x = np.linalg.solve(Aw.T @ A + 1e-9 * np.eye(6), -(Aw.T @ r))
        T = make_T(rotation_from_vector(x[:3]), x[3:]) @ T
        if np.linalg.norm(x[:3]) < 2e-6 and np.linalg.norm(x[3:]) < 2e-7:
            break

    p = apply(T, src_p)
    ns = src_n @ T[:3, :3].T
    dist, idx = tree.query(p, distance_upper_bound=max_dist, workers=-1)
    m = np.isfinite(dist)
    idx_m = idx[m]
    compat = (ns[m] * tgt_n[idx_m]).sum(1) > 0.6
    r = ((p[m][compat] - tgt_p[idx_m[compat]]) * tgt_n[idx_m[compat]]).sum(1)
    n = int(compat.sum())
    rmse = float(np.sqrt(np.mean(r**2))) if n else np.inf
    return T, IcpStats(n / len(src_p), rmse, n)


def build_model(samples, poses, active, voxel=MODEL_VOXEL):
    P, N = [], []
    for s, T, a in zip(samples, poses, active):
        if not a:
            continue
        p = apply(T, s.points)
        keep = in_box(p)
        P.append(p[keep])
        N.append(s.normals[keep] @ T[:3, :3].T)
    if not P:
        raise RuntimeError("no frames to build a model from")
    min_count = 2 if int(np.sum(active)) >= 10 else 1
    pts, nrm, _ = voxel_average(np.concatenate(P), np.concatenate(N), voxel, min_count)
    return pts, nrm


def _arkit_poses(capture: Capture):
    poses: list[np.ndarray | None] = []
    mirrored_votes = 0
    for f in capture.frames:
        res = None
        if f.head_uvd is not None:
            res = pose_from_head_points(f.head_uvd, f.color_k, capture.color_size[0], f.head_axis_length)
        if res is None:
            poses.append(None)
        else:
            poses.append(res[0])
            mirrored_votes += int(res[1])
    return poses, mirrored_votes


def _fill_missing(poses: list[np.ndarray | None]) -> list[np.ndarray]:
    known = [i for i, p in enumerate(poses) if p is not None]
    out = []
    for i, p in enumerate(poses):
        if p is None:
            j = min(known, key=lambda k: abs(k - i))
            p = poses[j]
        out.append(np.array(p, dtype=np.float64))
    return out


def _guess_head_frame(samples: FrameSamples) -> np.ndarray:
    """Head frame from one roughly frontal frame, for scans without ARKit poses."""
    first = samples.points
    z = first[:, 2]
    head = first[z < np.percentile(z, 2) + 0.25]
    origin = head.mean(0) + np.array([0, 0, 0.06])
    R = np.diag([1.0, -1.0, -1.0])  # camera axes -> head axes (y up, z to camera)
    return make_T(R, -R @ origin)


def _chain(samples: list[FrameSamples], arkit: list[np.ndarray] | None, rng, progress: Progress) -> list[np.ndarray]:
    """Frame-to-frame ICP. ARKit, when present, predicts each frame's motion
    and fixes the head frame; ICP supplies the accuracy."""
    poses = [arkit[0].copy() if arkit else _guess_head_frame(samples[0])]
    for i in range(1, len(samples)):
        prev = samples[i - 1]
        tp = apply(poses[i - 1], prev.points)
        keep = in_box(tp)
        tp, tn = tp[keep], prev.normals[keep] @ poses[i - 1][:3, :3].T
        if arkit:
            guess = poses[i - 1] @ np.linalg.inv(arkit[i - 1]) @ arkit[i]
        else:
            guess = poses[i - 1]
        T, st = guess, IcpStats(0.0, np.inf, 0)
        if len(tp) >= 100:
            tree = cKDTree(tp)
            for md in (0.02, 0.008, 0.004):
                T, st = icp(samples[i].points, samples[i].normals, tree, tp, tn, T, md, rng)
        if st.fitness < 0.3:
            T = guess  # ICP lost track; trust the prediction for this frame
        poses.append(T)
        progress(i / len(samples))
    return poses


def register(capture: Capture, progress: Progress = lambda f: None, seed: int = 0) -> Registration:
    rng = np.random.default_rng(seed)
    n = len(capture.frames)
    samples = []
    for i, f in enumerate(capture.frames):
        samples.append(sample_frame(head_depth(capture, f), f.depth_k, rng))
        progress(0.15 * (i + 1) / n)

    arkit, mirrored_votes = _arkit_poses(capture)
    n_arkit = sum(p is not None for p in arkit)
    attempts = []
    if n_arkit >= max(3, n // 2):
        attempts.append(("arkit", _fill_missing(arkit)))
    attempts.append(("depth-only", None))

    # Final passes against a model of all frames remove chaining drift.
    schedule = (0.004, 0.002)
    result = None
    for source, arkit_poses in attempts:
        poses = _chain(samples, arkit_poses, rng, lambda f: progress(0.15 + 0.45 * f))
        active = np.ones(n, dtype=bool)
        fitness = np.zeros(n)
        rmse = np.full(n, np.inf)
        for r, max_dist in enumerate(schedule):
            mp, mn = build_model(samples, poses, active)
            tree = cKDTree(mp)
            for i, s in enumerate(samples):
                ph = apply(poses[i], s.points)
                keep = in_box(ph)
                poses[i], st = icp(s.points[keep], s.normals[keep], tree, mp, mn, poses[i], max_dist, rng)
                fitness[i], rmse[i] = st.fitness, st.rmse
                progress(0.6 + 0.35 * (r + (i + 1) / n) / len(schedule))
            if r < len(schedule) - 1:
                # Drop frames that clearly do not fit before tightening.
                active = fitness > 0.25
                if active.sum() < 3:
                    break
        active = (fitness >= MIN_FITNESS) & (rmse <= MAX_RMSE)
        result = (source, poses, active, fitness, rmse)
        # ARKit poses that cannot be refined suggest a convention mismatch;
        # fall back to depth-only registration in that case.
        if active.sum() >= max(3, 0.5 * n):
            break

    source, poses, active, fitness, rmse = result
    if active.sum() < 3:
        raise RuntimeError(
            "could not align the depth frames; rescan holding the phone steady and turning your head slowly"
        )
    mp, mn = build_model(samples, poses, active)
    progress(1.0)
    return Registration(
        poses=np.stack(poses),
        active=active,
        fitness=fitness,
        rmse=rmse,
        pose_source=source,
        mirrored_projection=mirrored_votes > n_arkit // 2 if n_arkit else False,
        model_points=mp,
        model_normals=mn,
    )
