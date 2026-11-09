"""On-disk scan records.

Each scan lives in DATA_DIR/scans/<id>/ with:
  scan.json   state, progress, params, report
  upload.zip  the capture as uploaded
  bundle/     the extracted capture
  output/     generated files (model.stl, preview.ply, report.json, ...)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

SCAN_ID_RE = re.compile(r"^[a-z0-9]{12}$")
UPLOAD_TOKEN_TTL_S = 60 * 60

# waiting_upload -> queued -> processing -> done | failed
STATES = {"waiting_upload", "queued", "processing", "done", "failed"}


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class Store:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "scans"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------- paths
    def scan_dir(self, scan_id: str) -> Path:
        if not SCAN_ID_RE.match(scan_id):
            raise KeyError(scan_id)
        return self.root / scan_id

    def output_dir(self, scan_id: str) -> Path:
        return self.scan_dir(scan_id) / "output"

    def bundle_dir(self, scan_id: str) -> Path:
        return self.scan_dir(scan_id) / "bundle"

    def upload_path(self, scan_id: str) -> Path:
        return self.scan_dir(scan_id) / "upload.zip"

    # ----------------------------------------------------------- records
    def create(self, name: str | None = None, with_token: bool = True) -> tuple[dict, str | None]:
        with self._lock:
            while True:
                scan_id = "".join(secrets.choice("abcdefghijkmnpqrstuvwxyz23456789") for _ in range(12))
                d = self.root / scan_id
                if not d.exists():
                    break
            d.mkdir()
            token = secrets.token_urlsafe(24) if with_token else None
            rec = {
                "id": scan_id,
                "name": name or time.strftime("Scan %Y-%m-%d %H:%M"),
                "created_at": time.time(),
                "state": "waiting_upload",
                "stage": None,
                "progress": 0.0,
                "error": None,
                "params": None,
                "report": None,
                "upload_token_hash": _hash(token) if token else None,
                "upload_token_expires": time.time() + UPLOAD_TOKEN_TTL_S if token else None,
            }
            self._write(rec)
            return rec, token

    def get(self, scan_id: str) -> dict:
        path = self.scan_dir(scan_id) / "scan.json"
        with self._lock:
            try:
                return json.loads(path.read_text())
            except FileNotFoundError:
                raise KeyError(scan_id) from None

    def update(self, scan_id: str, **changes) -> dict:
        with self._lock:
            rec = self.get(scan_id)
            rec.update(changes)
            self._write(rec)
            return rec

    def list(self) -> list[dict]:
        out = []
        with self._lock:
            for d in self.root.iterdir():
                if SCAN_ID_RE.match(d.name) and (d / "scan.json").exists():
                    try:
                        out.append(self.get(d.name))
                    except (KeyError, json.JSONDecodeError):
                        continue
        return sorted(out, key=lambda r: r["created_at"], reverse=True)

    def find_by_token(self, token: str) -> dict | None:
        h = _hash(token)
        for rec in self.list():
            stored = rec.get("upload_token_hash")
            if stored and secrets.compare_digest(stored, h):
                if rec.get("upload_token_expires", 0) < time.time():
                    return None
                return rec
        return None

    def delete(self, scan_id: str) -> None:
        with self._lock:
            d = self.scan_dir(scan_id)
            if not d.exists():
                raise KeyError(scan_id)
            shutil.rmtree(d)

    def _write(self, rec: dict) -> None:
        d = self.root / rec["id"]
        tmp = d / "scan.json.tmp"
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, d / "scan.json")


def public(rec: dict) -> dict:
    """Record as sent to the browser (no token material)."""
    return {k: v for k, v in rec.items() if not k.startswith("upload_token")}
