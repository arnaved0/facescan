"""Mesh clean-up, colouring and print preparation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .capture import Capture
from .registration import HEAD_DEPTH_RANGE, Registration
from .geometry import clean_depth

UNSEEN_COLOR = np.array([190, 190, 190], dtype=np.float64)
COLOR_FRAMES = 40

# Head space (y up, z out of the face) -> print space (z up, face towards -y).
HEAD_TO_PRINT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, 0
    best = max(parts, key=lambda m: len(m.faces))
    return best, len(parts) - 1


def smooth(mesh: trimesh.Trimesh, iterations: int) -> None:
    if iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=iterations)


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce triangle count without ever breaking printability.

    Plain quadric decimation pinches thin areas into non-manifold edges, so
    this uses Manifold's topology-preserving simplification, raising the
    allowed deviation step by step until the target is met. A result is only
    accepted if it is still watertight once vertices are merged by position,
    which is how STL readers see it.
    """
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        import manifold3d as m3

        man = m3.Manifold(
            m3.Mesh(vert_properties=mesh.vertices.astype(np.float32), tri_verts=mesh.faces.astype(np.uint32))
        )
        if man.status() != m3.Error.NoError:
            return mesh
    except Exception:  # noqa: BLE001 - optional optimisation only
        return mesh
    best = mesh
    for tol_mm in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2):
        mm = man.simplify(tol_mm / 1000.0).to_mesh()
        cand = trimesh.Trimesh(mm.vert_properties[:, :3].astype(np.float64), mm.tri_verts, process=True)
        if not (cand.is_watertight and cand.is_winding_consistent):
            break
        best = cand
        if len(cand.faces) <= target_faces:
            break
    if best is not mesh and best.volume < 0:
        best.invert()
    return best


def vertex_colors(
    mesh: trimesh.Trimesh, capture: Capture, reg: Registration, progress: Callable[[float], None]
) -> np.ndarray | None:
    """Blend colour from the frames that see each vertex most head-on."""
    idx = np.flatnonzero(reg.active)
    idx = [i for i in idx if capture.frames[i].color_path is not None]
    if not idx:
        return None
    if len(idx) > COLOR_FRAMES:
        idx = [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, COLOR_FRAMES)]

    V = mesh.vertices
    N = mesh.vertex_normals
    acc = np.zeros((len(V), 3))
    wsum = np.zeros(len(V))
    dw, dh = capture.depth_size
    cw, ch = capture.color_size
    for n, i in enumerate(idx):
        f = capture.frames[i]
        color = capture.load_color(f)
        depth = clean_depth(capture.load_depth(f), *HEAD_DEPTH_RANGE)
        T = reg.poses[i]
        R, t = T[:3, :3], T[:3, 3]
        pc = (V - t) @ R
        nc = N @ R
        z = pc[:, 2]
        ok = z > 0.05
        zs = np.where(ok, z, 1.0)
        facing = (nc * (-pc / np.linalg.norm(pc, axis=1, keepdims=True))).sum(1)
        ok &= facing > 0.3

        k = f.depth_k
        ud = np.rint(k.fx * pc[:, 0] / zs + k.cx).astype(np.int64)
        vd = np.rint(k.fy * pc[:, 1] / zs + k.cy).astype(np.int64)
        ok &= (ud >= 0) & (ud < dw) & (vd >= 0) & (vd < dh)
        dd = np.zeros(len(V))
        dd[ok] = depth[vd[ok], ud[ok]]
        ok &= (dd > 0) & (np.abs(dd - z) < 0.004)  # visible, not occluded

        k = f.color_k
        uc = np.rint(k.fx * pc[:, 0] / zs + k.cx).astype(np.int64)
        vc = np.rint(k.fy * pc[:, 1] / zs + k.cy).astype(np.int64)
        ok &= (uc >= 0) & (uc < cw) & (vc >= 0) & (vc < ch)

        w = np.where(ok, np.clip(facing, 0, 1) ** 4, 0.0)
        sel = np.flatnonzero(ok)
        acc[sel] += color[vc[sel], uc[sel]] * w[sel, None]
        wsum[sel] += w[sel]
        progress((n + 1) / len(idx))

    out = np.tile(UNSEEN_COLOR, (len(V), 1))
    seen = wsum > 0
    out[seen] = acc[seen] / wsum[seen, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def to_print_space(
    mesh: trimesh.Trimesh, scale_percent: float, target_height_mm: float | None
) -> tuple[trimesh.Trimesh, np.ndarray, float]:
    """Metres/head space -> millimetres, z up, sitting on z = 0, centred."""
    m = mesh.copy()
    R = np.eye(4)
    R[:3, :3] = HEAD_TO_PRINT
    m.apply_transform(R)
    height = float(np.ptp(m.vertices[:, 2]))
    if target_height_mm:
        s = target_height_mm / height
    else:
        s = 1000.0 * scale_percent / 100.0
    S = np.diag([s, s, s, 1.0])
    m.apply_transform(S)
    lo, hi = m.bounds
    Tr = np.eye(4)
    Tr[:3, 3] = [-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]]
    m.apply_transform(Tr)
    return m, Tr @ S @ R, s / 10.0  # s/10 = percent of life size


def preview_mesh(mesh: trimesh.Trimesh, colors: np.ndarray | None, faces: int = 200_000) -> trimesh.Trimesh:
    small = mesh
    if len(mesh.faces) > faces:
        try:
            small = mesh.simplify_quadric_decimation(face_count=faces)
        except Exception:  # noqa: BLE001
            small = mesh
        small = trimesh.Trimesh(small.vertices, small.faces, process=False)
    if colors is not None:
        if small is mesh:
            c = colors
        else:
            _, nn = cKDTree(mesh.vertices).query(small.vertices, workers=-1)
            c = colors[nn]
        small.visual.vertex_colors = np.concatenate([c, np.full((len(c), 1), 255, np.uint8)], 1)
    return small
