"""Background processing queue (one scan at a time; processing is CPU and memory heavy)."""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import traceback

from .capture import CaptureError, extract_bundle
from .pipeline import ProcessParams, process_capture
from .storage import Store

log = logging.getLogger("facescan.jobs")


class Worker:
    def __init__(self, store: Store):
        self.store = store
        self.q: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="facescan-worker", daemon=True)
        self.thread.start()

    def submit(self, scan_id: str, params: ProcessParams) -> None:
        self.store.update(scan_id, state="queued", stage="Waiting", progress=0.0, error=None, params=params.to_dict())
        self.q.put((scan_id, params.to_dict()))

    def recover(self) -> None:
        """Scans interrupted by a restart are marked failed so they can be reprocessed."""
        for rec in self.store.list():
            if rec["state"] in ("queued", "processing"):
                self.store.update(rec["id"], state="failed", error="Interrupted by a server restart. Click Reprocess.")

    def _run(self) -> None:
        while True:
            scan_id, raw = self.q.get()
            try:
                self._process(scan_id, ProcessParams.from_dict(raw))
            except KeyError:
                pass  # deleted while queued
            finally:
                self.q.task_done()

    def _process(self, scan_id: str, params: ProcessParams) -> None:
        store = self.store
        rec = store.update(scan_id, state="processing", stage="Unpacking", progress=0.0)
        last = [0.0]

        def progress(stage: str, frac: float) -> None:
            if frac - last[0] >= 0.01 or frac >= 1.0 or frac < last[0]:
                last[0] = frac
                store.update(scan_id, stage=stage, progress=round(frac, 3))

        try:
            bundle = store.bundle_dir(scan_id)
            if not (bundle / "capture.json").exists():
                shutil.rmtree(bundle, ignore_errors=True)
                extract_bundle(store.upload_path(scan_id), bundle)
            report = process_capture(bundle, store.output_dir(scan_id), params, progress)
            store.update(scan_id, state="done", stage=None, progress=1.0, report=report)
            log.info("scan %s done in %.0fs", scan_id, report["timings"]["total_s"])
        except (CaptureError, ValueError, RuntimeError) as exc:
            store.update(scan_id, state="failed", stage=None, error=str(exc))
            log.warning("scan %s failed: %s", rec["id"], exc)
        except MemoryError:
            store.update(scan_id, state="failed", stage=None, error="Out of memory. Try a larger voxel size.")
        except Exception as exc:  # noqa: BLE001 - report anything to the user
            log.error("scan %s crashed:\n%s", scan_id, traceback.format_exc())
            store.update(scan_id, state="failed", stage=None, error=f"Internal error: {exc}")
