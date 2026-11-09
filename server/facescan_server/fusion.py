"""Fusing aligned depth frames into a closed (watertight) surface.

Two volumes share one bounding box:

* A fine truncated signed distance field (TSDF) in a narrow band around the
  observed surface. Averaging many frames here is what gives sub-millimetre
  detail.
* A coarse space-carving grid. A voxel is "free" when some frame sees through
  it (it lies in front of a measured depth, or the person-segmentation mask
  says background there). Everything never seen through counts as solid.
  This closes the parts the camera never saw: the inside of the head, hair
  that returned no depth, and so on.

The fine TSDF wins wherever it has data; the carved hull fills in the rest.
Flat cut planes (bottom of the neck, back of the head) are applied to the
field itself, then marching cubes on a field that is "outside" on every border
gives a closed, manifold mesh by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

from .capture import Capture
from .geometry import backproject, clean_depth, vertex_normals
from .registration import HEAD_BOX, HEAD_DEPTH_RANGE, Registration, in_box

MAX_FINE_VOXELS = 90_000_000
CARVE_MIN_VOTES = 2
FULL_WEIGHT = 3.0

Progress = Callable[[float], None]


@dataclass
class Grid:
    origin: np.ndarray  # corner of voxel (0, 0, 0), metres
    voxel: float
    factor: int  # fine voxels per coarse voxel along each axis
    coarse_shape: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(c * self.factor) for c in self.coarse_shape)

    @property
    def coarse_voxel(self) -> float:
        return self.voxel * self.factor

    def axis_centers(self, axis: int, coarse: bool = False) -> np.ndarray:
        v = self.coarse_voxel if coarse else self.voxel
        n = (self.coarse_shape if coarse else self.shape)[axis]
        return self.origin[axis] + (np.arange(n) + 0.5) * v

    def centers(self, flat_index: np.ndarray, coarse: bool = False) -> np.ndarray:
        shape = self.coarse_shape if coarse else self.shape
        v = self.coarse_voxel if coarse else self.voxel
        ijk = np.stack(np.unravel_index(flat_index, shape), axis=1)
        return (self.origin + (ijk + 0.5) * v).astype(np.float32)


@dataclass
class FusionParams:
    voxel: float  # metres
    trunc: float  # metres
    bottom_cut: float  # head-space y of the flat base, metres
    back_cut: float  # head-space z of the flat back, metres


def make_grid(points: np.ndarray, params: FusionParams, warnings: list[str]) -> Grid:
    # The box must hold the whole head, including parts with no depth (hair)
    # whose shape comes from carving, so it is not sized from measured points.
    pts = points[in_box(points)]
    lo = np.minimum(HEAD_BOX[0], np.percentile(pts, 0.2, axis=0) - 0.012)
    hi = np.maximum(HEAD_BOX[1], np.percentile(pts, 99.8, axis=0) + 0.012)
    lo, hi = np.maximum(lo, HEAD_BOX[0] - 0.03), np.minimum(hi, HEAD_BOX[1] + 0.03)
    lo[1] = max(lo[1], params.bottom_cut - 0.004)
    lo[2] = max(lo[2], params.back_cut - 0.004)
    if np.any(hi - lo < 0.02):
        raise ValueError("the cut planes remove the whole model; lower the bottom cut or move the back cut back")

    voxel = params.voxel
    while True:
        factor = max(1, int(round(0.002 / voxel)))
        vc = voxel * factor
        coarse = np.ceil((hi - lo) / vc).astype(int) + 2
        if int(np.prod(coarse)) * factor**3 <= MAX_FINE_VOXELS:
            break
        voxel *= 1.15
    if voxel > params.voxel * 1.01:
        warnings.append(
            f"Voxel size raised from {params.voxel * 1000:.2f} mm to {voxel * 1000:.2f} mm to fit in memory."
        )
        params.voxel = voxel
    return Grid(origin=lo - vc, voxel=voxel, factor=factor, coarse_shape=tuple(int(c) for c in coarse))


def _project(pc: np.ndarray, k, w: int, h: int):
    z = pc[:, 2]
    ok = z > 0.05
    zs = np.where(ok, z, 1.0)
    u = np.rint(k.fx * pc[:, 0] / zs + k.cx).astype(np.int64)
    v = np.rint(k.fy * pc[:, 1] / zs + k.cy).astype(np.int64)
    ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, ok


def fuse(capture: Capture, reg: Registration, params: FusionParams, warnings: list[str], progress: Progress):
    grid = make_grid(reg.model_points, params, warnings)
    trunc = max(params.trunc, 2.5 * grid.voxel)
    vc = grid.coarse_voxel
    margin = max(0.0015, 0.5 * trunc)
    frames = [(f, reg.poses[i]) for i, f in enumerate(capture.frames) if reg.active[i]]
    dw, dh = capture.depth_size

    # ---------------------------------------------------------- narrow band
    shape = grid.shape
    band = np.zeros(shape, dtype=bool)
    ijk = np.floor((reg.model_points - grid.origin) / grid.voxel).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
    band[tuple(ijk[ok].T)] = True
    radius = trunc + margin + 2.5 * vc
    band = ndimage.binary_dilation(band, iterations=int(np.ceil(radius / grid.voxel)))
    band_idx = np.flatnonzero(band)
    del band
    centers = grid.centers(band_idx)
    D = np.zeros(len(band_idx), dtype=np.float32)
    W = np.zeros(len(band_idx), dtype=np.float32)

    for n, (f, T) in enumerate(frames):
        depth = clean_depth(capture.load_depth(f), *HEAD_DEPTH_RANGE)
        pts = backproject(depth, f.depth_k)
        valid = depth > 0
        # Keep only the head: mask pixels that land outside the head box.
        head = pts.reshape(-1, 3) @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
        valid &= in_box(head).reshape(valid.shape)
        depth = np.where(valid, depth, 0)
        normals, nok = vertex_normals(pts, valid)

        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)
        pc = (centers - t) @ R
        u, v, m = _project(pc, f.depth_k, dw, dh)
        d = np.zeros(len(pc), dtype=np.float32)
        d[m] = depth[v[m], u[m]]
        sdf = d - pc[:, 2]
        m &= (d > 0) & (sdf > -trunc)
        if not m.any():
            continue
        sel = np.flatnonzero(m)
        nrm = normals[v[sel], u[sel]]
        view = -pc[sel] / np.linalg.norm(pc[sel], axis=1, keepdims=True)
        wgt = np.clip((nrm * view).sum(1), 0.15, 1.0) * nok[v[sel], u[sel]]
        tsdf = np.minimum(1.0, sdf[sel] / trunc)
        wsum = W[sel] + wgt
        D[sel] = np.where(wsum > 0, (D[sel] * W[sel] + tsdf * wgt) / np.maximum(wsum, 1e-6), D[sel])
        W[sel] = wsum
        progress(0.5 * (n + 1) / len(frames))

    # ------------------------------------------------------------ carving
    cshape = grid.coarse_shape
    cidx = np.arange(int(np.prod(cshape)))
    ccenters = grid.centers(cidx, coarse=True)
    votes = np.zeros(len(cidx), dtype=np.uint16)
    for n, (f, T) in enumerate(frames):
        # Far background is noisy; only real foreground/background edges matter here.
        depth = clean_depth(capture.load_depth(f), 0.1, 6.0, jump_rel=0.06)
        R = T[:3, :3].astype(np.float32)
        t = T[:3, 3].astype(np.float32)
        pc = (ccenters - t) @ R
        u, v, m = _project(pc, f.depth_k, dw, dh)
        d = np.zeros(len(pc), dtype=np.float32)
        d[m] = depth[v[m], u[m]]
        free = m & (d > 0) & (pc[:, 2] < d - margin)
        seg = capture.load_segmentation(f)
        if seg is not None:
            sk = capture.segmentation_k(f)
            sw, sh = capture.segmentation_size
            person = ndimage.binary_dilation(seg >= 128, iterations=2)
            us, vs, ms = _project(pc, sk, sw, sh)
            bg = np.zeros(len(pc), dtype=bool)
            bg[ms] = ~person[vs[ms], us[ms]]
            free |= bg
        votes += free
        progress(0.5 + 0.3 * (n + 1) / len(frames))

    solid = (votes < CARVE_MIN_VOTES).reshape(cshape).astype(np.float32)
    hull = (0.5 - ndimage.gaussian_filter(solid, sigma=0.75)) * 2.0  # > 0 outside
    F = ndimage.zoom(hull, grid.factor, order=1, grid_mode=True, mode="nearest").astype(np.float32)
    if F.shape != shape:  # zoom rounding guard
        F = np.ascontiguousarray(F[: shape[0], : shape[1], : shape[2]])
    flat = F.reshape(-1)  # a view: edits below write into F
    alpha = np.clip(W / FULL_WEIGHT, 0.0, 1.0)
    flat[band_idx] = alpha * D + (1 - alpha) * flat[band_idx]

    # ------------------------------------------------------------- cuts
    y = grid.axis_centers(1)
    z = grid.axis_centers(2)
    F = np.maximum(F, np.clip(-(y - params.bottom_cut) / trunc, -1, 1).astype(np.float32)[None, :, None])
    F = np.maximum(F, np.clip(-(z - params.back_cut) / trunc, -1, 1).astype(np.float32)[None, None, :])
    for axis in range(3):
        sl = [slice(None)] * 3
        sl[axis] = 0
        F[tuple(sl)] = 1.0
        sl[axis] = -1
        F[tuple(sl)] = 1.0
    progress(0.85)

    if F.min() >= 0:
        raise RuntimeError("fusion produced an empty volume")
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(grid.voxel,) * 3, allow_degenerate=False)
    verts = verts + grid.origin + 0.5 * grid.voxel
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    progress(1.0)
    observed = float((W > 0).mean())
    return mesh, grid, observed
