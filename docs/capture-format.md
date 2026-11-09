# FaceScan capture format (version 1)

The iPhone app produces a zip containing one folder. The folder holds `capture.json`
and a `frames/` directory. The server also accepts the files at the zip's top level.

```
facescan-2026-11-09T16-42-00Z/
  capture.json
  frames/000000.depth   depth map
  frames/000000.jpg     colour image
  frames/000000.seg     person mask (optional)
  ...
```

## capture.json

```jsonc
{
  "format": "facescan-capture",
  "version": 1,
  "app_version": "1.0",
  "device_model": "iPhone16,1",
  "created_at": "2026-11-09T16:42:00Z",
  "depth": { "encoding": "uint16-deflate", "width": 640, "height": 480, "units_per_meter": 10000 },
  "color": { "encoding": "jpeg", "width": 1280, "height": 720 },
  "segmentation": { "encoding": "uint8-deflate", "width": 256, "height": 144 },  // or absent
  "frames": [
    {
      "index": 0,
      "timestamp": 1234.567,                       // seconds, ARFrame.timestamp
      "depth_file": "frames/000000.depth",
      "color_file": "frames/000000.jpg",
      "segmentation_file": "frames/000000.seg",
      "depth_intrinsics": { "fx": 0, "fy": 0, "cx": 0, "cy": 0 },   // pixels, for the depth map
      "color_intrinsics": { "fx": 0, "fy": 0, "cx": 0, "cy": 0 },   // pixels, for the colour image
      "intrinsics_source": "calibration",          // or "derived" (see below)
      "head_pose": { "points": [[u, v, d], [u, v, d], [u, v, d], [u, v, d]], "axis_length": 0.05 },
      "yaw_deg": 0, "pitch_deg": 0,                // guidance only
      "camera_transform": [16 floats], "face_transform": [16 floats]  // raw ARKit, debugging only
    }
  ]
}
```

## Images

All images are stored in the sensor's native landscape orientation, unrotated and
unmirrored, so the intrinsics apply to them directly.

* **Depth**: little-endian `uint16`, row-major, `width × height`, compressed with raw
  DEFLATE (Apple `COMPRESSION_ZLIB`, no zlib header). Value / `units_per_meter` gives
  metres along the optical axis. 0 means no measurement. The server also accepts
  zlib-wrapped data.
* **Colour**: JPEG, sRGB.
* **Segmentation**: `uint8` (255 = person, 0 = background), raw DEFLATE. It has the same
  field of view as the colour image, so its intrinsics are the colour intrinsics scaled
  by `segmentation.width / color.width` and `segmentation.height / color.height`.

The depth map and colour image come from the same TrueDepth colour camera, with
different crops and resolutions. Depth intrinsics come from
`AVDepthData.cameraCalibrationData`, scaled from its reference size. When that is
missing (`intrinsics_source: "derived"`), they are derived from the colour intrinsics,
assuming the depth map spans the full sensor width at a 4:3 crop.

## Head pose

`head_pose.points` holds the ARKit face anchor's origin and the tips of its +x, +y and
+z axes (each `axis_length` metres long). Each point is projected into the colour image
with `ARCamera.projectPoint(_:orientation: .landscapeRight, viewportSize: imageResolution)`,
and its depth along the camera axis is attached. The server lifts the points back to
3D with the colour intrinsics and fits a rigid transform. This gives the head pose in
the same camera frame as the depth data, without relying on ARKit's camera-axis
conventions. If the fit is a mirror image, the server flips `u` and tries again.

Head space follows ARKit's face anchor: +y up, +z out of the face, origin near the
centre of the head. Cut planes in the web app are measured in this space.
