"""End-to-end processing: capture bundle -> printable mesh files."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh

from .capture import Capture, load_capture
from .fusion import FusionParams, fuse
from .postprocess import (
    decimate,
    largest_component,
    preview_mesh,
    smooth,
    to_print_space,
    vertex_colors,
)
from .registration import Registration, register

REGISTRATION_CACHE = "registration.npz"
REGISTRATION_VERSION = 2

OUTPUT_FILES = {
    "model.stl": "STL (for slicers)",
    "model.3mf": "3MF (for slicers)",
    "model.obj": "OBJ with vertex colour",
    "model.ply": "PLY with vertex colour",
}
PREVIEW_FILE = "preview.ply"


@dataclass
class ProcessParams:
    voxel_size_mm: float = 0.7
    back_cut_mm: float = -40.0
    bottom_cut_mm: float = -110.0
    smoothing_iterations: int = 4
    target_faces: int = 600_000
    scale_percent: float = 100.0
    target_height_mm: float | None = None

    LIMITS = {
        "voxel_size_mm": (0.4, 3.0),
        "back_cut_mm": (-120.0, 40.0),
        "bottom_cut_mm": (-200.0, 0.0),
        "smoothing_iterations": (0, 50),
        "target_faces": (0, 5_000_000),
        "scale_percent": (1.0, 500.0),
        "target_height_mm": (5.0, 1000.0),
    }

    @classmethod
    def from_dict(cls, raw: dict | None) -> "ProcessParams":
        raw = dict(raw or {})
        out = cls()
        for f in fields(cls):
            if f.name not in raw or raw[f.name] is None or raw[f.name] == "":
                continue
            val = raw[f.name]
            try:
                val = int(val) if f.name in ("smoothing_iterations", "target_faces") else float(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{f.name}: not a number") from exc
            lo, hi = cls.LIMITS[f.name]
            if not (lo <= val <= hi) or val != val:
                raise ValueError(f"{f.name} must be between {lo} and {hi}")
            setattr(out, f.name, val)
        return out

    def to_dict(self) -> dict:
        return asdict(self)


Progress = Callable[[str, float], None]

# Share of total processing time per stage, for the progress bar.
STAGES = [("Aligning frames", 0.45), ("Fusing depth", 0.30), ("Building mesh", 0.10), ("Colouring", 0.10), ("Exporting", 0.05)]


class _Tracker:
    def __init__(self, cb: Progress):
        self.cb = cb

    def stage(self, name: str) -> Callable[[float], None]:
        share = dict(STAGES)[name]
        base = sum(s for n, s in STAGES[: [n for n, _ in STAGES].index(name)])
        self.cb(name, base)
        return lambda f: self.cb(name, base + share * min(max(f, 0.0), 1.0))


def _load_or_register(capture: Capture, work: Path, progress) -> Registration:
    cache = work / REGISTRATION_CACHE
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if int(z["version"]) == REGISTRATION_VERSION and len(z["poses"]) == len(capture.frames):
                progress(1.0)
                return Registration(
                    poses=z["poses"],
                    active=z["active"],
                    fitness=z["fitness"],
                    rmse=z["rmse"],
                    pose_source=str(z["pose_source"]),
                    mirrored_projection=bool(z["mirrored"]),
                    model_points=z["model_points"],
                    model_normals=z["model_normals"],
                )
        except (OSError, KeyError, ValueError):
            pass
    reg = register(capture, progress)
    tmp = work / (REGISTRATION_CACHE + ".tmp.npz")
    np.savez_compressed(
        tmp,
        version=REGISTRATION_VERSION,
        poses=reg.poses,
        active=reg.active,
        fitness=reg.fitness,
        rmse=reg.rmse,
        pose_source=reg.pose_source,
        mirrored=reg.mirrored_projection,
        model_points=reg.model_points,
        model_normals=reg.model_normals,
    )
    tmp.replace(cache)
    return reg


def process_capture(bundle_dir: Path, out_dir: Path, params: ProcessParams, progress: Progress = lambda s, f: None) -> dict:
    """Process an extracted capture bundle. Returns the report dict."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    tracker = _Tracker(progress)
    warnings: list[str] = []
    timings = {}

    capture = load_capture(bundle_dir)
    if len(capture.frames) < 20:
        warnings.append(f"Only {len(capture.frames)} frames were captured; 80 or more gives better detail.")

    p = tracker.stage("Aligning frames")
    reg = _load_or_register(capture, work, p)
    timings["align_s"] = round(time.time() - t0, 1)
    used = int(reg.active.sum())
    if reg.pose_source != "arkit":
        warnings.append("ARKit head poses were missing or unusable, so frames were aligned from depth alone.")
    if used < 0.7 * len(capture.frames):
        warnings.append(
            f"{len(capture.frames) - used} of {len(capture.frames)} frames did not align and were skipped "
            "(usually caused by moving the phone, fast head turns or changing expression)."
        )

    t1 = time.time()
    p = tracker.stage("Fusing depth")
    fparams = FusionParams(
        voxel=params.voxel_size_mm / 1000.0,
        trunc=max(0.003, 3.5 * params.voxel_size_mm / 1000.0),
        bottom_cut=params.bottom_cut_mm / 1000.0,
        back_cut=params.back_cut_mm / 1000.0,
    )
    mesh, grid, observed = fuse(capture, reg, fparams, warnings, p)
    timings["fuse_s"] = round(time.time() - t1, 1)

    t2 = time.time()
    p = tracker.stage("Building mesh")
    mesh, dropped = largest_component(mesh)
    p(0.3)
    smooth(mesh, params.smoothing_iterations)
    p(0.6)
    mesh = decimate(mesh, params.target_faces)
    p(1.0)
    timings["mesh_s"] = round(time.time() - t2, 1)

    t3 = time.time()
    p = tracker.stage("Colouring")
    colors = vertex_colors(mesh, capture, reg, p)
    timings["color_s"] = round(time.time() - t3, 1)

    t4 = time.time()
    p = tracker.stage("Exporting")
    printed, head_to_print, life_percent = to_print_space(mesh, params.scale_percent, params.target_height_mm)
    if colors is not None:
        printed.visual.vertex_colors = np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], 1)

    tmp_suffix = ".tmp"
    written = {}
    plain = trimesh.Trimesh(printed.vertices, printed.faces, process=False)  # STL/3MF: geometry only
    for name, mesh_for_file in (
        ("model.stl", plain),
        ("model.3mf", plain),
        ("model.obj", printed),
        ("model.ply", printed),
    ):
        try:
            data = mesh_for_file.export(file_type=name.rsplit(".", 1)[1])
        except Exception as exc:  # noqa: BLE001 - one format failing must not lose the others
            warnings.append(f"Could not write {name}: {exc}")
            continue
        path = out_dir / (name + tmp_suffix)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        path.replace(out_dir / name)
        written[name] = (out_dir / name).stat().st_size
        p(len(written) / 5)
    prev = preview_mesh(printed, colors)
    (out_dir / (PREVIEW_FILE + tmp_suffix)).write_bytes(prev.export(file_type="ply"))
    (out_dir / (PREVIEW_FILE + tmp_suffix)).replace(out_dir / PREVIEW_FILE)
    p(1.0)
    timings["export_s"] = round(time.time() - t4, 1)
    timings["total_s"] = round(time.time() - t0, 1)

    watertight = bool(printed.is_watertight and printed.is_winding_consistent)
    if not watertight:
        warnings.append("The mesh is not watertight; run it through your slicer's repair tool before printing.")
    if dropped:
        warnings.append(f"Removed {dropped} small disconnected piece(s).")
    if observed < 0.3:
        warnings.append("Large parts of the model were filled in rather than measured; scan with more head turns.")

    ok = np.isfinite(reg.rmse) & reg.active
    report = {
        "frames_total": len(capture.frames),
        "frames_used": used,
        "pose_source": reg.pose_source,
        "registration_rmse_mm": round(float(np.median(reg.rmse[ok])) * 1000, 3) if ok.any() else None,
        "voxel_size_mm": round(fparams.voxel * 1000, 3),
        "mesh": {
            "watertight": watertight,
            "vertices": int(len(printed.vertices)),
            "faces": int(len(printed.faces)),
            "volume_cm3": round(float(printed.volume) / 1000.0, 1),
            "dimensions_mm": [round(float(x), 1) for x in np.ptp(printed.vertices, axis=0)],
            "scale_percent_of_life_size": round(life_percent, 2),
        },
        "files": written,
        "head_to_print": head_to_print.tolist(),
        "device_model": capture.meta.get("device_model"),
        "warnings": warnings,
        "timings": timings,
        "params": params.to_dict(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
