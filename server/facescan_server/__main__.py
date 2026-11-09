"""Run the FaceScan web app:  python -m facescan_server [--host 0.0.0.0] [--port 8000]"""

import argparse
import logging

import uvicorn

from .app import lan_ip


def main() -> None:
    ap = argparse.ArgumentParser(description="FaceScan web app")
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets the iPhone reach it over Wi-Fi")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ip = lan_ip()
    print(f"\nFaceScan: open http://localhost:{args.port} on this computer")
    if ip and args.host in ("0.0.0.0", ip):
        print(f"          the iPhone will upload to http://{ip}:{args.port}\n")
    uvicorn.run("facescan_server.app:get_app", factory=True, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
