"""Measure a processed synthetic scan against the analytic head it came from.

    python tools/evaluate_synthetic.py out_dir [--render views.png]

Reports surface error (distance from output vertices to the true surface) on
the measured part of the face, after a small rigid alignment, and whether the
mesh is printable. Optionally renders shaded views for a visual check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_head import head_sdf, sdf_gradient  # noqa: E402


def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def render(mesh: trimesh.Trimesh, path: Path, size: int = 520) -> None:
    """Tiny z-buffer renderer (vertex splats with Lambert shading)."""
    from PIL import Image

    v = mesh.vertices - mesh.bounds.mean(0)
    n = mesh.vertex_normals
    col = mesh.visual.vertex_colors[:, :3].astype(float) if mesh.visual.kind == "vertex" else None
    views = []
    for yaw in (0, 45, 90, 180):
        a = np.radians(yaw)
        # print space: z up, face towards -y. Camera looks from -y, rotated about z.
        R = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
        p = v @ R.T
        nn = n @ R.T
        scale = size * 0.45 / np.abs(v).max()
        x = np.clip((p[:, 0] * scale + size / 2).astype(int), 0, size - 1)
        y = np.clip((-p[:, 2] * scale + size / 2).astype(int), 0, size - 1)
        depth = p[:, 1]
        order = np.argsort(-depth)  # far first, near overwrite
        img = np.full((size, size, 3), 30.0)
        light = np.clip(-nn[:, 1] * 0.8 + nn[:, 2] * 0.3, 0.08, 1)
        base = col if col is not None else np.full((len(v), 3), 220.0)
        shade = base * light[:, None]
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(y[order] + dy, 0, size - 1), np.clip(x[order] + dx, 0, size - 1)] = shade[order]
        views.append(img)
    Image.fromarray(np.concatenate(views, 1).astype(np.uint8)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--render", type=Path)
    args = ap.parse_args()

    report = json.loads((args.out_dir / "report.json").read_text())
    mesh = trimesh.load(args.out_dir / "model.ply", process=False)
    M = np.linalg.inv(np.array(report["head_to_print"]))
    V = mesh.vertices @ M[:3, :3].T + M[:3, 3]

    p = report["params"]
    # Align output to the truth first (the pipeline's head frame is only an
    # estimate, especially without ARKit), using the best-fitting 70 % of
    # vertices so cut planes and filled-in areas do not drag the fit.
    sub = V[:: max(1, len(V) // 60000)]
    best = None
    # Start from the pipeline's head frame, and from a centred guess.
    for shift in (np.zeros(3), np.array([0, -0.01, 0.02]) - sub.mean(0)):
        Rt, tt = np.eye(3), shift.copy()
        for _ in range(40):
            cur = sub @ Rt.T + tt
            d = head_sdf(cur)
            keep = np.abs(d) <= np.percentile(np.abs(d), 70)
            target = cur[keep] - d[keep, None] * sdf_gradient(cur[keep])
            R, t = kabsch(cur[keep], target)
            Rt, tt = R @ Rt, R @ tt + t
        d = np.abs(head_sdf(sub @ Rt.T + tt))
        score = np.mean(np.sort(d)[: int(0.7 * len(d))])
        if best is None or score < best[0]:
            best = (score, Rt, tt)
    _, Rt, tt = best
    V = V @ Rt.T + tt
    # Measured face region in the true head frame, away from cut planes and hair.
    region = (V[:, 2] > 0.0) & (V[:, 1] > p["bottom_cut_mm"] / 1000 + 0.01) & (V[:, 1] < 0.05)
    region &= V[:, 1] > -0.1
    err = np.abs(head_sdf(V[region])) * 1000
    out = {
        "face_vertices": int(region.sum()),
        "error_mean_mm": round(float(err.mean()), 3),
        "error_median_mm": round(float(np.median(err)), 3),
        "error_p95_mm": round(float(np.percentile(err, 95)), 3),
        "error_max_mm": round(float(err.max()), 3),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "volume_cm3": round(float(mesh.volume) / 1000, 1),
    }
    print(json.dumps(out, indent=2))
    if args.render:
        render(mesh, args.render)
        print(f"rendered {args.render}")


if __name__ == "__main__":
    main()
