"""Command-line processing, without the web app.

    python -m facescan_server.cli capture.zip out_dir [--voxel-size-mm 0.7 ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path

from .capture import extract_bundle
from .pipeline import ProcessParams, process_capture


def main() -> None:
    ap = argparse.ArgumentParser(description="Process a FaceScan capture into printable mesh files.")
    ap.add_argument("capture", type=Path, help="capture .zip from the iPhone app (or an extracted folder)")
    ap.add_argument("out_dir", type=Path)
    for f in fields(ProcessParams):
        ap.add_argument("--" + f.name.replace("_", "-"), dest=f.name, default=None)
    args = ap.parse_args()
    params = ProcessParams.from_dict({f.name: getattr(args, f.name) for f in fields(ProcessParams)})

    tmp = None
    if args.capture.is_dir():
        bundle = args.capture
    else:
        tmp = Path(tempfile.mkdtemp())
        bundle = extract_bundle(args.capture, tmp / "bundle")
    last = [""]

    def progress(stage: str, frac: float) -> None:
        if stage != last[0]:
            print(f"\n{stage}", end="", flush=True)
            last[0] = stage
        print(f"\r{stage}: {frac * 100:5.1f}%", end="", flush=True)

    try:
        report = process_capture(bundle, args.out_dir, params, progress)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    print()
    print(json.dumps({k: v for k, v in report.items() if k != "head_to_print"}, indent=2))


if __name__ == "__main__":
    main()
