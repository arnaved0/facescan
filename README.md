# FaceScan

Self-scan your face with an iPhone's Face ID (TrueDepth) camera and get a watertight,
millimetre-scaled model ready for 3D printing.

There are two parts:

* **iPhone app** (`ios/`, Swift/SwiftUI, with an optional App Clip). It records depth,
  colour and head pose while spoken prompts guide you through turning your head.
* **Web app** (`server/`, Python). It runs on your computer, receives the scan over
  Wi-Fi, fuses the depth frames into a mesh, shows a 3D preview, and exports STL, 3MF,
  OBJ and PLY.

A browser can't read TrueDepth depth data, so the capture step has to be native. Everything
after capture happens in the web app.

## How it works

1. The web app shows a QR code. Scanning it with the iPhone Camera app opens FaceScan,
   linked to that scan.
2. You hold the phone at arm's length (about 30 cm) and follow the prompts: face the
   phone, turn your head to each side, then tilt your chin up and down. A coverage grid
   fills in as you go, and the scan stops by itself when coverage is complete
   (about 30 seconds, 100 to 240 frames).
3. The zip uploads to the web app, which processes it in 1 to 3 minutes:
   * **Align**: ARKit's head pose predicts each frame's motion, and point-to-plane ICP
     against the previous frame and then against a model of all frames makes it precise.
   * **Fuse**: a 0.7 mm TSDF averages the noisy depth frames. Space carving with ARKit's
     person mask fills in parts with no depth (hair, the inside of the head).
   * **Mesh**: flat cuts at the neck and behind the ears, then marching cubes. The mesh
     is closed by construction and is simplified only in ways that keep it watertight.
   * **Colour**: each vertex takes its colour from the frames that see it most head-on.
4. You preview the bust in the browser, adjust the cuts, detail or scale, reprocess if
   needed, and download.

Exports are in millimetres, with Z up, the flat base at Z = 0, and the face towards −Y.

## Requirements

* An iPhone or iPad with Face ID (iPhone X or later), running iOS 16 or later.
* A Mac with Xcode 15 or later and [XcodeGen](https://github.com/yonaskolb/XcodeGen)
  (`brew install xcodegen`), to build the app.
* Python 3.10 or later for the web app, on macOS, Linux or Windows. It needs about 2 GB
  of free memory while processing.
* The phone and the computer on the same Wi-Fi network.

## Run the web app

```bash
cd server
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip ...
.venv/bin/python -m facescan_server            # http://localhost:8000
```

At startup it prints the LAN address the phone will upload to. Allow incoming
connections if your firewall asks. Scans are stored in `server/data/`; set
`FACESCAN_DATA_DIR` to change that.

To try it without an iPhone, generate a synthetic scan and use **Import capture**:

```bash
.venv/bin/python tools/make_synthetic_scan.py synthetic.zip
```

There is also a command-line processor:
`python -m facescan_server.cli capture.zip out_dir [--voxel-size-mm 0.6 ...]`.

## Build the iPhone app

```bash
cd ios
# Edit project.yml: DEVELOPMENT_TEAM, BUNDLE_ID (and APP_DOMAIN if you use the App Clip)
xcodegen generate
open FaceScan.xcodeproj
```

Select the **FaceScan** scheme and your iPhone, then Run. A free Apple ID works for
running on your own phone. In that case, delete the `entitlements` sections and the
`FaceScanClip` dependency from `project.yml` before `xcodegen generate`, because
associated domains and App Clips need a paid account.

## Optional: App Clip (scan without installing the app)

With an App Clip, scanning the QR code opens a lightweight version of the app straight
from the Camera, with no install. It needs:

1. A paid Apple Developer account.
2. The web app served over **HTTPS** at a real domain, for example behind a reverse proxy.
3. `APP_DOMAIN` in `project.yml` set to that domain, and the server started with:
   ```bash
   FACESCAN_PUBLIC_URL=https://scan.yourdomain.com \
   FACESCAN_APPLE_TEAM_ID=ABCDE12345 \
   FACESCAN_BUNDLE_ID=com.yourname.facescan \
   python -m facescan_server
   ```
   The server then serves `/.well-known/apple-app-site-association` for App Clip and
   universal-link verification.
4. For testing, register a **Local Experience** on the phone (Settings → Developer →
   App Clips Testing) for `https://scan.yourdomain.com/c/`. For release, set up an App
   Clip experience in App Store Connect.

Without an App Clip, the QR code opens a small web page with an **Open in FaceScan**
button (`facescan://` link) that hands the upload link to the installed app.

## Scanning tips

* Use soft, even light. Avoid direct sunlight; its infrared swamps the depth sensor.
* Tie back hair that covers your forehead or ears. Hair returns little depth, so its
  shape is filled in from the silhouette and comes out smooth.
* Keep the phone still and move only your head, slowly. Keep a neutral expression.
  Frames where your mouth is open or you are smiling are skipped.
* Glasses: take them off. Lenses distort the depth.

## Accuracy

TrueDepth depth noise at 30 cm is roughly 0.5 to 1 mm per frame; fusing 100+ frames
averages most of it away. On the bundled synthetic test (simulated TrueDepth noise,
missing hair depth, ±2 mm / 0.7° ARKit pose error), the reconstructed face surface is
within 0.18 mm on average and 0.36 mm at the 95th percentile of the true shape. Real
scans will be somewhat worse: facial movement, lens distortion (not corrected) and skin
translucency all add error. Test with your own phone before relying on the model for a
fitted part. You can calibrate by scanning an object of known size.

## Privacy and security

A face scan is biometric data. The web app keeps everything on your computer and has
**no login**: anyone who can reach its port on your network can see and download scans.
Run it only on a network you trust, or put it behind a reverse proxy with
authentication. Upload links are random, single use, and expire after an hour. Uploads
are size-limited (`FACESCAN_MAX_UPLOAD_MB`, default 1024) and validated before
processing.

## Layout

```
ios/                 iPhone app + App Clip (XcodeGen project.yml, Swift sources)
server/
  facescan_server/   web app, API and processing pipeline
    capture.py       bundle extraction and parsing (untrusted input)
    registration.py  frame alignment (ARKit-seeded ICP)
    fusion.py        TSDF + space carving -> watertight mesh
    postprocess.py   clean-up, simplification, colour, print orientation
    pipeline.py      stages, parameters, report
    app.py           FastAPI routes; static/ is the browser UI
  tools/             synthetic scan generator and accuracy checks
docs/capture-format.md
```
