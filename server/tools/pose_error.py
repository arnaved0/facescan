"""Per-frame registration error of a processed synthetic scan vs. ground truth.

    python tools/pose_error.py extracted_bundle_dir out_dir

Poses are only defined up to one common rigid transform (the head frame is an
estimate), so this reports how much each frame deviates from the consensus.
"""

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    bundle, out = Path(sys.argv[1]), Path(sys.argv[2])
    frames = json.loads((bundle / "capture.json").read_text())["frames"]
    z = np.load(out / "work" / "registration.npz")
    G = [z["poses"][i] @ np.array(f["truth_cam_from_head"]) for i, f in enumerate(frames)]
    G = np.stack(G)
    act = z["active"]
    # Consensus: chordal mean rotation, mean translation.
    U, _, Vt = np.linalg.svd(G[act, :3, :3].sum(0))
    Rm = U @ Vt
    tm = G[act, :3, 3].mean(0)
    rot = [np.degrees(np.arccos(np.clip((np.trace(Rm.T @ g[:3, :3]) - 1) / 2, -1, 1))) for g in G]
    # Translation error measured at the face surface (~10 cm from the origin).
    trans = [np.linalg.norm(g[:3, 3] - tm) * 1000 for g in G]
    rot, trans = np.array(rot), np.array(trans)
    face_mm = trans + np.radians(rot) * 100
    print(f"frames {act.sum()}/{len(act)} active")
    print(f"rotation dev deg: median {np.median(rot[act]):.3f} max {rot[act].max():.3f}")
    print(f"translation dev mm: median {np.median(trans[act]):.3f} max {trans[act].max():.3f}")
    print(f"approx surface error mm: median {np.median(face_mm[act]):.3f} p95 {np.percentile(face_mm[act], 95):.3f}")


if __name__ == "__main__":
    main()
