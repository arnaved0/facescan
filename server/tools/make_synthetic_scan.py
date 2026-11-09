"""Render a synthetic FaceScan capture bundle from an analytic head.

Lets you try the server and pipeline without an iPhone, and is how the
pipeline is tested. The output uses the same format as the iOS app
(docs/capture-format.md), including TrueDepth-like noise, missing depth on
the hair, a person-segmentation mask, and noisy ARKit head poses.

    python tools/make_synthetic_scan.py out/synthetic.zip --frames 120
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthetic_head import head_sdf  # noqa: E402

from facescan_server.capture import write_bundle_zip  # noqa: E402

DEPTH_W, DEPTH_H = 640, 480
COLOR_W, COLOR_H = 640, 360
SEG_W, SEG_H = 256, 144
F = 450.0
DEPTH_K = dict(fx=F, fy=F, cx=319.5, cy=239.5)
COLOR_K = dict(fx=F, fy=F, cx=319.5, cy=179.5)
UNITS = 10000
WALL = 1.6
PIVOT = np.array([0.0, -0.10, -0.02])  # heads turn about the neck, not their centre


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def cam_from_head(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    Rrot = rot_y(np.radians(yaw_deg)) @ rot_x(np.radians(-pitch_deg))
    R_front = np.diag([1.0, -1.0, -1.0])
    t0 = np.array([0.0, -0.01, 0.33])
    T = np.eye(4)
    T[:3, :3] = R_front @ Rrot
    T[:3, 3] = R_front @ (PIVOT - Rrot @ PIVOT) + t0
    return T


def trajectory(n: int) -> list[tuple[float, float]]:
    keys = np.array([(0, 0), (70, 0), (-70, 0), (0, 0), (0, 25), (0, -25), (0, 0)], dtype=float)
    seg = np.linalg.norm(np.diff(keys, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for x in np.linspace(0, s[-1], n):
        i = min(np.searchsorted(s, x, side="right") - 1, len(seg) - 1)
        a = (x - s[i]) / seg[i]
        out.append(tuple(keys[i] * (1 - a) + keys[i + 1] * a))
    return out


def trace(T_ch: np.ndarray, k: dict, w: int, h: int):
    """Sphere-trace the head. Returns (camera z or WALL, hit mask, head-space hits)."""
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dirs = np.stack([(u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], np.ones_like(u, float)], -1).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    T_hc = np.linalg.inv(T_ch)
    o = T_hc[:3, 3]
    dh = dirs @ T_hc[:3, :3].T
    t = np.full(len(dh), 0.05)
    active = np.ones(len(dh), dtype=bool)
    hit = np.zeros(len(dh), dtype=bool)
    for _ in range(160):
        idx = np.flatnonzero(active)
        if not idx.size:
            break
        d = head_sdf(o + dh[idx] * t[idx, None])
        t[idx] += 0.8 * d
        done = d < 2e-5
        hit[idx[done]] = True
        active[idx[done]] = False
        active[idx[t[idx] > 1.0]] = False
    z = np.where(hit, t * dirs[:, 2], WALL)
    pts = o + dh * t[:, None]
    return z.reshape(h, w), hit.reshape(h, w), pts.reshape(h, w, 3)


def render_frame(args):
    i, yaw, pitch, seed, out_dir, with_pose, mirror = args
    rng = np.random.default_rng(seed + i)
    T = cam_from_head(yaw, pitch)

    z, hit, ph = trace(T, DEPTH_K, DEPTH_W, DEPTH_H)
    sigma = 0.0004 * (z / 0.3) ** 2
    z = z + rng.normal(0, 1, z.shape) * sigma
    hair = hit & (ph[..., 1] > 0.06) & (ph[..., 2] < 0.07) & (rng.random(z.shape) < 0.85)
    dropout = rng.random(z.shape) < 0.01
    z[hair | dropout] = 0
    depth = np.clip(np.rint(z * UNITS), 0, 65535).astype("<u2")
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    (out_dir / f"frames/{i:06d}.depth").write_bytes(c.compress(depth.tobytes()) + c.flush())

    _, chit, cph = trace(T, COLOR_K, COLOR_W, COLOR_H)
    skin = np.array([205, 160, 140], float)
    stripes = 0.85 + 0.15 * np.sin(cph[..., 1] * 400)[..., None]
    img = np.where(chit[..., None], skin * stripes, np.array([90, 100, 110], float))
    nose = chit & (cph[..., 2] > 0.095)
    img[nose] = [200, 60, 60]
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=90)
    (out_dir / f"frames/{i:06d}.jpg").write_bytes(buf.getvalue())

    seg = np.asarray(Image.fromarray((chit * 255).astype(np.uint8)).resize((SEG_W, SEG_H), Image.NEAREST))
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    (out_dir / f"frames/{i:06d}.seg").write_bytes(c.compress(seg.tobytes()) + c.flush())

    frame = {
        "index": i,
        "timestamp": i / 15.0,
        "depth_file": f"frames/{i:06d}.depth",
        "color_file": f"frames/{i:06d}.jpg",
        "segmentation_file": f"frames/{i:06d}.seg",
        "depth_intrinsics": DEPTH_K,
        "color_intrinsics": COLOR_K,
        "truth_cam_from_head": T.tolist(),
    }
    if with_pose:
        # ARKit-like pose error: ~0.7 degrees and ~2 mm.
        noise = np.eye(4)
        w = rng.normal(0, np.radians(0.7), 3)
        th = np.linalg.norm(w)
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / max(th, 1e-12)
        noise[:3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        noise[:3, 3] = rng.normal(0, 0.002, 3)
        Tn = T @ noise
        a = 0.05
        pts = []
        for loc in ([0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a]):
            pc = Tn[:3, :3] @ np.array(loc, float) + Tn[:3, 3]
            u = COLOR_K["fx"] * pc[0] / pc[2] + COLOR_K["cx"]
            v = COLOR_K["fy"] * pc[1] / pc[2] + COLOR_K["cy"]
            if mirror:
                u = (COLOR_W - 1) - u
            pts.append([float(u), float(v), float(pc[2])])
        frame["head_pose"] = {"points": pts, "axis_length": a}
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", type=Path, help="output .zip")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-arkit", action="store_true", help="omit ARKit head poses")
    ap.add_argument("--mirror", action="store_true", help="simulate mirrored ARKit projections")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    root = tmp / "synthetic-capture"
    (root / "frames").mkdir(parents=True)
    jobs = [
        (i, yaw, pitch, args.seed, root, not args.no_arkit, args.mirror)
        for i, (yaw, pitch) in enumerate(trajectory(args.frames))
    ]
    with Pool() as pool:
        frames = pool.map(render_frame, jobs)
    manifest = {
        "format": "facescan-capture",
        "version": 1,
        "app_version": "synthetic",
        "device_model": "synthetic",
        "created_at": "2026-01-01T00:00:00Z",
        "depth": {"encoding": "uint16-deflate", "units_per_meter": UNITS, "width": DEPTH_W, "height": DEPTH_H},
        "color": {"encoding": "jpeg", "width": COLOR_W, "height": COLOR_H},
        "segmentation": {"encoding": "uint8-deflate", "width": SEG_W, "height": SEG_H},
        "frames": frames,
    }
    (root / "capture.json").write_text(json.dumps(manifest, indent=1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_bundle_zip(root, args.output)
    shutil.rmtree(tmp)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
