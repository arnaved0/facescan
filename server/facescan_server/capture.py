"""Loading and validating FaceScan capture bundles.

A capture bundle is a zip produced by the iOS app. Its layout is documented in
docs/capture-format.md. Everything in it is untrusted input, so extraction and
parsing enforce size limits and reject paths that escape the bundle directory.
"""

from __future__ import annotations

import json
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

FORMAT_NAME = "facescan-capture"
SUPPORTED_VERSIONS = {1}
MANIFEST_NAME = "capture.json"

MAX_FRAMES = 600
MAX_ZIP_MEMBERS = 4 * MAX_FRAMES + 16
MAX_UNCOMPRESSED_BYTES = 3 * 1024**3
MAX_IMAGE_SIDE = 4096


class CaptureError(ValueError):
    """The capture bundle is malformed or unsupported."""


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def parse(cls, raw: object, what: str) -> "Intrinsics":
        if not isinstance(raw, dict):
            raise CaptureError(f"{what}: intrinsics missing")
        try:
            vals = {k: float(raw[k]) for k in ("fx", "fy", "cx", "cy")}
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptureError(f"{what}: invalid intrinsics") from exc
        if not all(np.isfinite(v) for v in vals.values()) or vals["fx"] <= 0 or vals["fy"] <= 0:
            raise CaptureError(f"{what}: invalid intrinsics")
        return cls(**vals)

    def scaled(self, sx: float, sy: float) -> "Intrinsics":
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy)


@dataclass
class Frame:
    index: int
    timestamp: float
    depth_path: Path
    color_path: Path | None
    segmentation_path: Path | None
    depth_k: Intrinsics
    color_k: Intrinsics
    # ARKit head pose as 4 projected points (origin, +x, +y, +z axis tips):
    # rows of (u, v, depth) in colour-image pixels and metres. None if absent.
    head_uvd: np.ndarray | None
    head_axis_length: float


@dataclass
class Capture:
    root: Path
    meta: dict
    depth_size: tuple[int, int]  # (width, height)
    depth_units_per_meter: float
    color_size: tuple[int, int]
    segmentation_size: tuple[int, int] | None
    frames: list[Frame]

    # ----------------------------------------------------------------- loaders
    def load_depth(self, frame: Frame) -> np.ndarray:
        """Depth in metres as float32 (H, W). Invalid pixels are 0."""
        w, h = self.depth_size
        raw = _inflate(frame.depth_path.read_bytes(), w * h * 2, frame.depth_path.name)
        depth = np.frombuffer(raw, dtype="<u2").reshape(h, w).astype(np.float32)
        return depth / np.float32(self.depth_units_per_meter)

    def load_color(self, frame: Frame) -> np.ndarray | None:
        if frame.color_path is None:
            return None
        with Image.open(frame.color_path) as img:
            img = img.convert("RGB")
            if img.size != self.color_size:
                img = img.resize(self.color_size, Image.BILINEAR)
            return np.asarray(img, dtype=np.uint8)

    def load_segmentation(self, frame: Frame) -> np.ndarray | None:
        """Person mask (H, W) uint8, 255 = person. Same field of view as colour."""
        if frame.segmentation_path is None or self.segmentation_size is None:
            return None
        w, h = self.segmentation_size
        raw = _inflate(frame.segmentation_path.read_bytes(), w * h, frame.segmentation_path.name)
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w)

    def segmentation_k(self, frame: Frame) -> Intrinsics | None:
        if self.segmentation_size is None:
            return None
        sw, sh = self.segmentation_size
        cw, ch = self.color_size
        return frame.color_k.scaled(sw / cw, sh / ch)


# --------------------------------------------------------------------------- zip


def extract_bundle(zip_path: Path, dest: Path) -> Path:
    """Safely extract a capture zip into ``dest`` and return the bundle root.

    The zip may contain the bundle at its top level or inside one folder
    (iOS wraps a zipped directory in a folder of the same name).
    """
    dest.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise CaptureError("upload is not a valid zip file") from exc

    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        infos = [i for i in infos if "__MACOSX" not in PurePosixPath(i.filename).parts]
        if len(infos) > MAX_ZIP_MEMBERS:
            raise CaptureError("too many files in capture")
        manifests = [i for i in infos if PurePosixPath(i.filename).name == MANIFEST_NAME]
        if not manifests:
            raise CaptureError(f"{MANIFEST_NAME} not found in capture")
        manifest = min(manifests, key=lambda i: len(PurePosixPath(i.filename).parts))
        prefix = PurePosixPath(manifest.filename).parent
        if len(prefix.parts) > 1:
            raise CaptureError(f"{MANIFEST_NAME} is nested too deeply")

        budget = MAX_UNCOMPRESSED_BYTES
        dest_resolved = dest.resolve()
        for info in infos:
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise CaptureError(f"unsafe path in capture: {info.filename!r}")
            try:
                rel = name.relative_to(prefix)
            except ValueError:
                continue  # outside the bundle folder; ignore
            target = (dest / Path(*rel.parts)).resolve()
            if dest_resolved not in target.parents:
                raise CaptureError(f"unsafe path in capture: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    budget -= len(chunk)
                    if budget < 0:
                        raise CaptureError("capture is too large when uncompressed")
                    dst.write(chunk)
    return dest


# ---------------------------------------------------------------------- manifest


def load_capture(root: Path) -> Capture:
    manifest_path = root / MANIFEST_NAME
    try:
        meta = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("format") != FORMAT_NAME:
        raise CaptureError("not a FaceScan capture")
    if meta.get("version") not in SUPPORTED_VERSIONS:
        raise CaptureError(f"unsupported capture version {meta.get('version')!r}")

    depth = _section(meta, "depth")
    if depth.get("encoding") != "uint16-deflate":
        raise CaptureError("unsupported depth encoding")
    depth_size = _size(depth, "depth")
    units = float(depth.get("units_per_meter", 0))
    if not (100 <= units <= 100000):
        raise CaptureError("invalid depth units_per_meter")

    color = _section(meta, "color")
    color_size = _size(color, "color")

    seg_size = None
    seg = meta.get("segmentation")
    if isinstance(seg, dict):
        if seg.get("encoding") != "uint8-deflate":
            raise CaptureError("unsupported segmentation encoding")
        seg_size = _size(seg, "segmentation")

    raw_frames = meta.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise CaptureError("capture has no frames")
    if len(raw_frames) > MAX_FRAMES:
        raise CaptureError(f"capture has more than {MAX_FRAMES} frames")

    frames = []
    for i, rf in enumerate(raw_frames):
        if not isinstance(rf, dict):
            raise CaptureError(f"frame {i}: invalid entry")
        what = f"frame {i}"
        head_uvd = None
        axis_len = 0.05
        hp = rf.get("head_pose")
        if isinstance(hp, dict) and isinstance(hp.get("points"), list):
            try:
                pts = np.asarray(hp["points"], dtype=np.float64)
                axis_len = float(hp.get("axis_length", 0.05))
            except (TypeError, ValueError):
                pts = None
            if pts is not None and pts.shape == (4, 3) and np.all(np.isfinite(pts)) and axis_len > 0:
                head_uvd = pts
        frames.append(
            Frame(
                index=i,
                timestamp=float(rf.get("timestamp", i)),
                depth_path=_member(root, rf.get("depth_file"), what, required=True),
                color_path=_member(root, rf.get("color_file"), what),
                segmentation_path=_member(root, rf.get("segmentation_file"), what) if seg_size else None,
                depth_k=Intrinsics.parse(rf.get("depth_intrinsics"), what),
                color_k=Intrinsics.parse(rf.get("color_intrinsics"), what),
                head_uvd=head_uvd,
                head_axis_length=axis_len,
            )
        )

    return Capture(
        root=root,
        meta=meta,
        depth_size=depth_size,
        depth_units_per_meter=units,
        color_size=color_size,
        segmentation_size=seg_size,
        frames=frames,
    )


def _section(meta: dict, key: str) -> dict:
    sec = meta.get(key)
    if not isinstance(sec, dict):
        raise CaptureError(f"missing '{key}' section")
    return sec


def _size(sec: dict, what: str) -> tuple[int, int]:
    try:
        w, h = int(sec["width"]), int(sec["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureError(f"{what}: missing width/height") from exc
    if not (8 <= w <= MAX_IMAGE_SIDE and 8 <= h <= MAX_IMAGE_SIDE):
        raise CaptureError(f"{what}: unsupported size {w}x{h}")
    return w, h


def _member(root: Path, rel: object, what: str, required: bool = False) -> Path | None:
    if rel is None and not required:
        return None
    if not isinstance(rel, str) or not rel:
        raise CaptureError(f"{what}: missing file reference")
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts:
        raise CaptureError(f"{what}: unsafe file reference {rel!r}")
    path = root / Path(*p.parts)
    if not path.is_file():
        if required:
            raise CaptureError(f"{what}: file {rel!r} not found")
        return None
    return path


def _inflate(data: bytes, expected: int, name: str) -> bytes:
    """Decompress raw DEFLATE (Apple's COMPRESSION_ZLIB) or zlib-wrapped data."""
    for wbits in (-15, 15):
        try:
            d = zlib.decompressobj(wbits)
            out = d.decompress(data, expected + 1)
        except zlib.error:
            continue
        if len(out) == expected:
            return out
        raise CaptureError(f"{name}: unexpected size {len(out)} (expected {expected})")
    raise CaptureError(f"{name}: cannot decompress")


def write_bundle_zip(root: Path, zip_path: Path) -> None:
    """Zip a bundle directory (used by tools and tests)."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, Path(root.name) / path.relative_to(root))
