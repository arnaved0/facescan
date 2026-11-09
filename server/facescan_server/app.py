"""FaceScan web app: scan management API, phone upload endpoint and the browser UI."""

from __future__ import annotations

import html
import io
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import qrcode
import qrcode.image.svg
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .capture import CaptureError, extract_bundle, load_capture
from .jobs import Worker
from .pipeline import OUTPUT_FILES, PREVIEW_FILE, ProcessParams
from .storage import Store, public

log = logging.getLogger("facescan")

STATIC_DIR = Path(__file__).parent / "static"
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
DOWNLOADABLE = {**OUTPUT_FILES, PREVIEW_FILE: "preview", "report.json": "report"}
URL_SCHEME = "facescan"


class Settings:
    def __init__(self) -> None:
        self.data_dir = Path(os.environ.get("FACESCAN_DATA_DIR", "data")).resolve()
        self.public_url = os.environ.get("FACESCAN_PUBLIC_URL", "").rstrip("/") or None
        self.max_upload = int(os.environ.get("FACESCAN_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
        self.team_id = os.environ.get("FACESCAN_APPLE_TEAM_ID") or None
        self.bundle_id = os.environ.get("FACESCAN_BUNDLE_ID") or None


def lan_ip() -> str | None:
    """Best guess at this machine's LAN address (no packets are sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = Store(settings.data_dir)
    worker = Worker(store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.recover()
        yield

    app = FastAPI(title="FaceScan", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.worker = worker
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # ------------------------------------------------------------ helpers
    def base_url(request: Request) -> str:
        if settings.public_url:
            return settings.public_url
        parts = urlsplit(str(request.base_url))
        host = parts.hostname or "localhost"
        is_local = host == "localhost"
        try:
            is_local = is_local or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if is_local:
            ip = lan_ip()
            if ip:
                netloc = f"{ip}:{parts.port}" if parts.port else ip
                parts = parts._replace(netloc=netloc)
        return urlunsplit(parts._replace(path="", query="", fragment="")).rstrip("/")

    def get_scan(scan_id: str) -> dict:
        try:
            return store.get(scan_id)
        except KeyError:
            raise HTTPException(404, "scan not found") from None

    def scan_view(rec: dict) -> dict:
        out = public(rec)
        out_dir = store.output_dir(rec["id"])
        out["files"] = [
            {"name": n, "label": label, "size": (out_dir / n).stat().st_size}
            for n, label in OUTPUT_FILES.items()
            if rec["state"] == "done" and (out_dir / n).exists()
        ]
        out["has_preview"] = rec["state"] == "done" and (out_dir / PREVIEW_FILE).exists()
        out["has_upload"] = store.upload_path(rec["id"]).exists()
        return out

    async def receive_zip(request: Request, dest: Path) -> None:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_upload:
            raise HTTPException(413, "upload too large")
        size = 0
        tmp = dest.with_suffix(".part")
        try:
            with open(tmp, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > settings.max_upload:
                        raise HTTPException(413, "upload too large")
                    f.write(chunk)
            if size == 0:
                raise HTTPException(400, "empty upload")
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)

    def validate_upload(scan_id: str) -> int:
        bundle = store.bundle_dir(scan_id)
        shutil.rmtree(bundle, ignore_errors=True)
        extract_bundle(store.upload_path(scan_id), bundle)
        return len(load_capture(bundle).frames)

    async def accept_upload(request: Request, scan_id: str) -> dict:
        await receive_zip(request, store.upload_path(scan_id))
        try:
            frames = await run_in_threadpool(validate_upload, scan_id)
        except CaptureError as exc:
            store.upload_path(scan_id).unlink(missing_ok=True)
            shutil.rmtree(store.bundle_dir(scan_id), ignore_errors=True)
            raise HTTPException(400, f"invalid capture: {exc}") from None
        store.update(scan_id, upload_token_hash=None, upload_token_expires=None, frames=frames)
        worker.submit(scan_id, ProcessParams())
        return {"ok": True, "scan_id": scan_id, "frames": frames}

    # ---------------------------------------------------------------- UI
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/c/{token}", response_class=HTMLResponse, include_in_schema=False)
    def phone_landing(token: str, request: Request):
        """Opened on the iPhone from the QR code when the App Clip is not set up."""
        if not TOKEN_RE.match(token) or store.find_by_token(token) is None:
            return HTMLResponse(_page("This scan link has expired or was already used. Create a new scan on your computer."), 404)
        upload = f"{base_url(request)}/api/upload/{token}"
        app_link = f"{URL_SCHEME}://scan?upload={quote(upload, safe='')}"
        banner = ""
        if settings.bundle_id:
            banner = f'<meta name="apple-itunes-app" content="app-clip-bundle-id={html.escape(settings.bundle_id)}.Clip, app-clip-display=card">'
        body = (
            "<h1>FaceScan</h1><p>Open the FaceScan app to record your scan. "
            "It uploads straight to the computer that showed this code.</p>"
            f'<p><a class="btn" href="{html.escape(app_link)}">Open in FaceScan</a></p>'
            "<p class='small'>Nothing happens? Install the FaceScan app on this iPhone first.</p>"
        )
        return HTMLResponse(_page(body, head=banner))

    @app.get("/.well-known/apple-app-site-association", include_in_schema=False)
    def aasa():
        if not (settings.team_id and settings.bundle_id):
            raise HTTPException(404)
        app_id = f"{settings.team_id}.{settings.bundle_id}"
        return JSONResponse(
            {
                "applinks": {"details": [{"appIDs": [app_id], "components": [{"/": "/c/*"}]}]},
                "appclips": {"apps": [f"{app_id}.Clip"]},
            }
        )

    # --------------------------------------------------------------- API
    @app.get("/api/scans")
    def list_scans():
        return [scan_view(r) for r in store.list()]

    @app.post("/api/scans")
    async def new_scan(request: Request):
        """Start a scan: returns the QR code the phone uses to upload."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        name = str(body.get("name") or "").strip()[:80] if isinstance(body, dict) else ""
        rec, token = store.create(name or None)
        phone_url = f"{base_url(request)}/c/{token}"
        img = qrcode.make(phone_url, image_factory=qrcode.image.svg.SvgPathImage, box_size=12, border=2)
        buf = io.BytesIO()
        img.save(buf)
        return {
            "scan": scan_view(rec),
            "phone_url": phone_url,
            "upload_url": f"{base_url(request)}/api/upload/{token}",
            "qr_svg": buf.getvalue().decode(),
            "is_lan_url": settings.public_url is None,
        }

    @app.post("/api/scans/import")
    async def import_scan(request: Request):
        """Upload a capture zip from the browser (e.g. after AirDropping it from the phone)."""
        rec, _ = store.create(with_token=False)
        try:
            return await accept_upload(request, rec["id"])
        except HTTPException:
            store.delete(rec["id"])
            raise

    @app.get("/api/scans/{scan_id}")
    def scan_detail(scan_id: str):
        return scan_view(get_scan(scan_id))

    @app.patch("/api/scans/{scan_id}")
    async def rename_scan(scan_id: str, request: Request):
        get_scan(scan_id)
        body = await request.json()
        name = str(body.get("name", "")).strip()[:80] if isinstance(body, dict) else ""
        if not name:
            raise HTTPException(400, "name required")
        return scan_view(store.update(scan_id, name=name))

    @app.post("/api/scans/{scan_id}/process")
    async def reprocess(scan_id: str, request: Request):
        rec = get_scan(scan_id)
        if rec["state"] in ("queued", "processing"):
            raise HTTPException(409, "scan is already being processed")
        if not store.upload_path(scan_id).exists():
            raise HTTPException(409, "no capture uploaded yet")
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        try:
            params = ProcessParams.from_dict(body if isinstance(body, dict) else {})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        worker.submit(scan_id, params)
        return scan_view(store.get(scan_id))

    @app.delete("/api/scans/{scan_id}")
    def delete_scan(scan_id: str):
        rec = get_scan(scan_id)
        if rec["state"] == "processing":
            raise HTTPException(409, "wait for processing to finish")
        store.delete(scan_id)
        return {"ok": True}

    @app.get("/api/scans/{scan_id}/files/{name}")
    def download(scan_id: str, name: str):
        rec = get_scan(scan_id)
        if name == "capture.zip":
            path = store.upload_path(scan_id)
        elif name in DOWNLOADABLE:
            path = store.output_dir(scan_id) / name
        else:
            raise HTTPException(404)
        if not path.exists():
            raise HTTPException(404)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", rec["name"]).strip("-") or rec["id"]
        stem, ext = name.rsplit(".", 1)
        filename = f"{safe}-{stem}.{ext}" if name != PREVIEW_FILE else name
        return FileResponse(path, filename=filename, headers={"Cache-Control": "no-cache"})

    @app.post("/api/upload/{token}")
    async def phone_upload(token: str, request: Request):
        """Capture upload from the iPhone app. The token comes from the QR code."""
        if not TOKEN_RE.match(token):
            raise HTTPException(404, "unknown upload link")
        rec = store.find_by_token(token)
        if rec is None:
            raise HTTPException(404, "this upload link has expired or was already used")
        return await accept_upload(request, rec["id"])

    return app


def _page(body: str, head: str = "") -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">{head}
<title>FaceScan</title><style>
body{{font:17px -apple-system,system-ui,sans-serif;max-width:32em;margin:3em auto;padding:0 1.2em;color:#1d1d1f}}
.btn{{display:inline-block;background:#0a66ff;color:#fff;padding:.8em 1.4em;border-radius:12px;text-decoration:none;font-weight:600}}
.small{{color:#666;font-size:14px}}</style></head><body>{body}</body></html>"""


def get_app() -> FastAPI:  # uvicorn factory entry point
    return create_app()
