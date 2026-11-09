import ARKit
import Combine
import Foundation
import simd

/// Runs the TrueDepth face-tracking session, guides the user, and picks keyframes.
///
/// ARKit delegate callbacks arrive on `arQueue`; anything published to SwiftUI
/// is sent to the main queue.
final class ScanController: NSObject, ObservableObject, ARSessionDelegate {
    enum Phase: Equatable {
        case ready, recording, finishing, finished(CaptureResult), failed(String)
    }

    // Distance from phone to head centre that gives the best depth.
    static let distanceRange: ClosedRange<Float> = 0.24...0.45
    static let maxFrames = 240
    static let minKeyframeAngle: Float = 2.0  // degrees between keyframes
    static let repeatInterval: TimeInterval = 0.6  // also take a frame this often at the same pose

    @Published private(set) var phase: Phase = .ready
    @Published private(set) var instruction: Instruction = .findFace
    @Published private(set) var coverage = CoverageGrid()
    @Published private(set) var frameCount = 0
    @Published private(set) var faceVisible = false
    @Published private(set) var distance: Float?
    @Published var voiceEnabled = true

    let session = ARSession()
    private let arQueue = DispatchQueue(label: "facescan.ar", qos: .userInteractive)
    private let coach = SpeechCoach()

    // Touched only on arQueue.
    private var writer: CaptureWriter?
    private var recording = false
    private var grid = CoverageGrid()
    private var lastKeyYaw: Float = .nan, lastKeyPitch: Float = .nan
    private var lastKeyTime: TimeInterval = 0
    private var lastYaw: Float = 0, lastTime: TimeInterval = 0
    private var yawSpeed: Double = 0
    private var lastPublish: TimeInterval = 0
    private var finishRequested = false

    static var isSupported: Bool { ARFaceTrackingConfiguration.isSupported }

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = arQueue
    }

    func start() {
        let config = ARFaceTrackingConfiguration()
        config.maximumNumberOfTrackedFaces = 1
        config.isLightEstimationEnabled = false
        if ARFaceTrackingConfiguration.supportsFrameSemantics(.personSegmentation) {
            config.frameSemantics.insert(.personSegmentation)
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        coach.prepare()
    }

    func stop() {
        session.pause()
        coach.stop()
    }

    func beginRecording() {
        arQueue.async {
            do {
                self.writer = try CaptureWriter()
            } catch {
                self.publish { $0.phase = .failed("Could not create scan files: \(error.localizedDescription)") }
                return
            }
            self.grid = CoverageGrid()
            self.lastKeyYaw = .nan
            self.lastKeyPitch = .nan
            self.finishRequested = false
            self.recording = true
            self.publish {
                $0.coverage = CoverageGrid()
                $0.frameCount = 0
                $0.phase = .recording
            }
        }
    }

    func finishRecording() {
        arQueue.async { self.finishOnQueue() }
    }

    func cancelRecording() {
        arQueue.async {
            self.recording = false
            self.writer?.discard()
            self.writer = nil
            self.publish { $0.phase = .ready }
        }
    }

    /// Discards a finished capture and goes back to the start.
    func reset() {
        arQueue.async {
            self.writer = nil
            self.publish {
                $0.phase = .ready
                $0.coverage = CoverageGrid()
                $0.frameCount = 0
            }
        }
    }

    private func finishOnQueue() {
        guard recording, let writer else { return }
        recording = false
        publish { $0.phase = .finishing }
        let model = ScanController.deviceModel()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let result = try writer.finish(deviceModel: model)
                self.publish { $0.phase = .finished(result) }
            } catch {
                self.publish { $0.phase = .failed(error.localizedDescription) }
            }
        }
        coach.say("Scan complete")
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let face = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first, face.isTracked else {
            publishThrottled(frame.timestamp) {
                $0.faceVisible = false
                $0.distance = nil
                $0.instruction = .findFace
            }
            return
        }

        let camT = frame.camera.transform
        let faceT = face.transform
        let facePos = SIMD3<Float>(faceT.columns.3.x, faceT.columns.3.y, faceT.columns.3.z)
        let camPos = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
        let dist = simd_length(camPos - facePos)
        let (yaw, pitch) = ScanController.headAngles(face: faceT, cameraPosition: camPos)

        let dt = frame.timestamp - lastTime
        if dt > 0 && dt < 0.5 {
            let speed = Double(abs(yaw - lastYaw)) / dt
            yawSpeed = 0.8 * yawSpeed + 0.2 * speed
        }
        lastYaw = yaw
        lastTime = frame.timestamp

        let jaw = face.blendShapes[.jawOpen]?.floatValue ?? 0
        let smile = max(face.blendShapes[.mouthSmileLeft]?.floatValue ?? 0, face.blendShapes[.mouthSmileRight]?.floatValue ?? 0)
        let neutral = jaw < 0.25 && smile < 0.5

        var next: Instruction
        if dist > ScanController.distanceRange.upperBound {
            next = .moveCloser
        } else if dist < ScanController.distanceRange.lowerBound {
            next = .moveBack
        } else if !neutral {
            next = .relaxFace
        } else {
            next = grid.instruction(yaw: Double(yaw), pitch: Double(pitch), yawSpeed: yawSpeed)
        }

        if recording, let writer, neutral, ScanController.distanceRange.contains(dist),
           let depthData = frame.capturedDepthData, !writer.isBusy {
            let moved = lastKeyYaw.isNaN || hypot(yaw - lastKeyYaw, pitch - lastKeyPitch) >= ScanController.minKeyframeAngle
            let stale = frame.timestamp - lastKeyTime >= ScanController.repeatInterval
            if (moved || stale) && yawSpeed < 90 {
                let input = FrameInput(
                    timestamp: frame.timestamp,
                    depthData: depthData,
                    colorBuffer: frame.capturedImage,
                    segmentationBuffer: frame.segmentationBuffer,
                    colorIntrinsics: ScanController.intrinsics(frame.camera.intrinsics),
                    colorSize: frame.camera.imageResolution,
                    headPose: ScanController.headPose(frame: frame, face: faceT),
                    yawDeg: yaw,
                    pitchDeg: pitch,
                    cameraTransform: ScanController.flatten(camT),
                    faceTransform: ScanController.flatten(faceT)
                )
                if writer.add(input) {
                    lastKeyYaw = yaw
                    lastKeyPitch = pitch
                    lastKeyTime = frame.timestamp
                    grid.add(yaw: Double(yaw), pitch: Double(pitch))
                }
            }
            if grid.isComplete || writer.frameCount >= ScanController.maxFrames {
                next = .done
                finishOnQueue()
            }
        }

        if recording || next == .done {
            coach.update(next, enabled: voiceEnabledSnapshot)
        }
        let gridSnapshot = grid
        let count = writer?.frameCount ?? 0
        publishThrottled(frame.timestamp) {
            $0.faceVisible = true
            $0.distance = dist
            $0.instruction = next
            $0.coverage = gridSnapshot
            $0.frameCount = count
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        publish { $0.phase = .failed("Camera error: \(error.localizedDescription)") }
    }

    // MARK: - Helpers

    private var voiceEnabledSnapshot: Bool {
        // Read on the AR queue; a stale value for one frame is harmless.
        voiceEnabled
    }

    private func publish(_ change: @escaping (ScanController) -> Void) {
        DispatchQueue.main.async { change(self) }
    }

    private func publishThrottled(_ t: TimeInterval, _ change: @escaping (ScanController) -> Void) {
        guard t - lastPublish >= 1.0 / 15.0 else { return }
        lastPublish = t
        publish(change)
    }

    /// Head yaw and pitch relative to the direction of the phone, in degrees,
    /// measured in ARKit's gravity-aligned world (y up), so it does not depend
    /// on how the phone is held.
    static func headAngles(face: simd_float4x4, cameraPosition: SIMD3<Float>) -> (Float, Float) {
        let facePos = SIMD3<Float>(face.columns.3.x, face.columns.3.y, face.columns.3.z)
        let forward = simd_normalize(SIMD3<Float>(face.columns.2.x, face.columns.2.y, face.columns.2.z))
        let toCam = simd_normalize(cameraPosition - facePos)
        let up = SIMD3<Float>(0, 1, 0)
        func horizontal(_ v: SIMD3<Float>) -> SIMD3<Float> {
            let h = SIMD3<Float>(v.x, 0, v.z)
            return simd_length(h) > 1e-5 ? simd_normalize(h) : SIMD3<Float>(0, 0, 1)
        }
        let fh = horizontal(forward), ch = horizontal(toCam)
        let yaw = atan2(simd_dot(simd_cross(ch, fh), up), simd_dot(ch, fh))
        let pitch = asin(max(-1, min(1, forward.y))) - asin(max(-1, min(1, toCam.y)))
        return (yaw * 180 / .pi, pitch * 180 / .pi)
    }

    /// The head origin and axis tips projected into the captured colour image,
    /// with their depth. The server rebuilds the pose from these in the same
    /// pixel space as the depth data.
    static func headPose(frame: ARFrame, face: simd_float4x4) -> CaptureManifest.HeadPose {
        let a: Float = 0.05
        let viewInverse = frame.camera.transform.inverse
        let size = frame.camera.imageResolution
        let locals: [SIMD4<Float>] = [SIMD4(0, 0, 0, 1), SIMD4(a, 0, 0, 1), SIMD4(0, a, 0, 1), SIMD4(0, 0, a, 1)]
        let points: [[Float]] = locals.map { local in
            let world = face * local
            let p = frame.camera.projectPoint(SIMD3<Float>(world.x, world.y, world.z), orientation: .landscapeRight, viewportSize: size)
            let inCamera = viewInverse * world
            return [Float(p.x), Float(p.y), abs(inCamera.z)]
        }
        return CaptureManifest.HeadPose(points: points, axisLength: a)
    }

    static func intrinsics(_ k: simd_float3x3) -> CaptureManifest.Intrinsics {
        .init(fx: k.columns.0.x, fy: k.columns.1.y, cx: k.columns.2.x, cy: k.columns.2.y)
    }

    static func flatten(_ m: simd_float4x4) -> [Float] {
        [m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
    }

    static func deviceModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }
}
