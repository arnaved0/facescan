import ARKit
import AVFoundation
import CoreImage
import Foundation
import ImageIO

/// One keyframe handed from the AR session to the writer. Holds the pixel
/// buffers (not the ARFrame, which ARKit needs back quickly).
struct FrameInput {
    let timestamp: TimeInterval
    let depthData: AVDepthData
    let colorBuffer: CVPixelBuffer
    let segmentationBuffer: CVPixelBuffer?
    let colorIntrinsics: CaptureManifest.Intrinsics
    let colorSize: CGSize
    let headPose: CaptureManifest.HeadPose?
    let yawDeg: Float
    let pitchDeg: Float
    let cameraTransform: [Float]
    let faceTransform: [Float]
}

enum CaptureWriterError: LocalizedError {
    case noFrames, zipFailed(String)
    var errorDescription: String? {
        switch self {
        case .noFrames: return "No frames were captured."
        case .zipFailed(let why): return "Could not package the scan: \(why)"
        }
    }
}

/// Writes keyframes to disk in the FaceScan capture format on a background queue.
final class CaptureWriter {
    static let depthUnitsPerMeter: Float = 10_000  // 0.1 mm steps, max 6.5 m

    let directory: URL
    private let queue = DispatchQueue(label: "facescan.writer", qos: .userInitiated)
    private let lock = NSLock()
    private var pending = 0
    private var frames: [CaptureManifest.FrameRecord] = []
    private var depthInfo: CaptureManifest.ImageInfo?
    private var colorInfo: CaptureManifest.ImageInfo?
    private var segInfo: CaptureManifest.ImageInfo?
    private var nextIndex = 0
    private let ciContext = CIContext()
    private(set) var lastError: Error?

    init() throws {
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("facescan-\(stamp)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("frames"), withIntermediateDirectories: true)
    }

    /// True while a frame is still being encoded; the caller skips frames then.
    var isBusy: Bool {
        lock.lock(); defer { lock.unlock() }
        return pending >= 2
    }

    var frameCount: Int {
        lock.lock(); defer { lock.unlock() }
        return nextIndex
    }

    @discardableResult
    func add(_ input: FrameInput) -> Bool {
        lock.lock()
        if pending >= 2 { lock.unlock(); return false }
        pending += 1
        let index = nextIndex
        nextIndex += 1
        lock.unlock()

        queue.async { [weak self] in
            guard let self else { return }
            do {
                try self.write(input, index: index)
            } catch {
                self.lastError = error
            }
            self.lock.lock(); self.pending -= 1; self.lock.unlock()
        }
        return true
    }

    private func write(_ input: FrameInput, index: Int) throws {
        let name = String(format: "%06d", index)

        // Depth -> uint16 (0.1 mm) -> raw DEFLATE.
        let depth = input.depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? input.depthData
            : input.depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = depth.depthDataMap
        let dw = CVPixelBufferGetWidth(map), dh = CVPixelBufferGetHeight(map)
        var depth16 = [UInt16](repeating: 0, count: dw * dh)
        CVPixelBufferLockBaseAddress(map, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(map) {
            let rowBytes = CVPixelBufferGetBytesPerRow(map)
            for y in 0..<dh {
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
                for x in 0..<dw {
                    let d = row[x]
                    if d.isFinite && d > 0 && d < 6.5 {
                        depth16[y * dw + x] = UInt16((d * CaptureWriter.depthUnitsPerMeter).rounded())
                    }
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(map, .readOnly)
        let depthRaw = depth16.withUnsafeBufferPointer { Data(buffer: $0) }  // little-endian on iOS
        let depthFile = "frames/\(name).depth"
        try (depthRaw as NSData).compressed(using: .zlib).write(to: directory.appendingPathComponent(depthFile))

        // Depth intrinsics, scaled from the calibration's reference size.
        var source = "calibration"
        let depthK: CaptureManifest.Intrinsics
        if let cal = depth.cameraCalibrationData {
            let K = cal.intrinsicMatrix
            let ref = cal.intrinsicMatrixReferenceDimensions
            let sx = Float(dw) / Float(ref.width), sy = Float(dh) / Float(ref.height)
            depthK = .init(fx: K.columns.0.x * sx, fy: K.columns.1.y * sy, cx: K.columns.2.x * sx, cy: K.columns.2.y * sy)
        } else {
            // Depth shares the colour camera's optics; it covers the full
            // sensor width with a taller (4:3) crop than the 16:9 video.
            source = "derived"
            let s = Float(dw) / Float(input.colorSize.width)
            let ck = input.colorIntrinsics
            let yOffset = (Float(dh) - Float(input.colorSize.height) * s) / 2
            depthK = .init(fx: ck.fx * s, fy: ck.fy * s, cx: ck.cx * s, cy: ck.cy * s + yOffset)
        }

        // Colour JPEG in the sensor's native orientation (matches the intrinsics).
        let colorFile = "frames/\(name).jpg"
        let image = CIImage(cvPixelBuffer: input.colorBuffer)
        let options = [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.85]
        if let srgb = CGColorSpace(name: CGColorSpace.sRGB),
           let jpeg = ciContext.jpegRepresentation(of: image, colorSpace: srgb, options: options) {
            try jpeg.write(to: directory.appendingPathComponent(colorFile))
        }

        // Person segmentation mask (8-bit), when ARKit provides one.
        var segFile: String?
        var segSize: (Int, Int)?
        if let seg = input.segmentationBuffer {
            let sw = CVPixelBufferGetWidth(seg), sh = CVPixelBufferGetHeight(seg)
            var bytes = [UInt8](repeating: 0, count: sw * sh)
            CVPixelBufferLockBaseAddress(seg, .readOnly)
            if let base = CVPixelBufferGetBaseAddress(seg) {
                let rowBytes = CVPixelBufferGetBytesPerRow(seg)
                for y in 0..<sh {
                    let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
                    for x in 0..<sw { bytes[y * sw + x] = row[x] }
                }
            }
            CVPixelBufferUnlockBaseAddress(seg, .readOnly)
            let file = "frames/\(name).seg"
            let raw = bytes.withUnsafeBufferPointer { Data(buffer: $0) }
            try (raw as NSData).compressed(using: .zlib).write(to: directory.appendingPathComponent(file))
            segFile = file
            segSize = (sw, sh)
        }

        let record = CaptureManifest.FrameRecord(
            index: index,
            timestamp: input.timestamp,
            depthFile: depthFile,
            colorFile: colorFile,
            segmentationFile: segFile,
            depthIntrinsics: depthK,
            colorIntrinsics: input.colorIntrinsics,
            intrinsicsSource: source,
            headPose: input.headPose,
            yawDeg: input.yawDeg,
            pitchDeg: input.pitchDeg,
            cameraTransform: input.cameraTransform,
            faceTransform: input.faceTransform
        )
        lock.lock()
        frames.append(record)
        if depthInfo == nil {
            depthInfo = .init(encoding: "uint16-deflate", width: dw, height: dh, unitsPerMeter: Int(CaptureWriter.depthUnitsPerMeter))
            colorInfo = .init(encoding: "jpeg", width: Int(input.colorSize.width), height: Int(input.colorSize.height))
        }
        if let (sw, sh) = segSize, segInfo == nil {
            segInfo = .init(encoding: "uint8-deflate", width: sw, height: sh)
        }
        lock.unlock()
    }

    /// Waits for pending frames, writes capture.json and zips the bundle.
    func finish(deviceModel: String) throws -> CaptureResult {
        queue.sync {}
        lock.lock()
        let records = frames.sorted { $0.index < $1.index }
        let depth = depthInfo, color = colorInfo
        var seg = segInfo
        lock.unlock()
        guard let depth, let color, !records.isEmpty else { throw CaptureWriterError.noFrames }
        // A mask must exist for every frame or for none.
        var finalRecords = records
        if seg != nil && records.contains(where: { $0.segmentationFile == nil }) {
            seg = nil
            finalRecords = records.map { var r = $0; r.segmentationFile = nil; return r }
        }
        let manifest = CaptureManifest(
            appVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0",
            deviceModel: deviceModel,
            createdAt: ISO8601DateFormatter().string(from: Date()),
            depth: depth,
            color: color,
            segmentation: seg,
            frames: finalRecords
        )
        try manifest.encoded().write(to: directory.appendingPathComponent("capture.json"))
        let zip = try CaptureWriter.zip(directory)
        return CaptureResult(directory: directory, zipURL: zip, frameCount: finalRecords.count)
    }

    func discard() {
        queue.sync {}
        try? FileManager.default.removeItem(at: directory)
    }

    /// Zips a directory using the system's built-in "for uploading" archive.
    static func zip(_ dir: URL) throws -> URL {
        let dest = dir.deletingLastPathComponent().appendingPathComponent(dir.lastPathComponent + ".zip")
        try? FileManager.default.removeItem(at: dest)
        var coordError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: dir, options: [.forUploading], error: &coordError) { tmpZip in
            do { try FileManager.default.copyItem(at: tmpZip, to: dest) } catch { copyError = error }
        }
        if let e = coordError ?? (copyError as NSError?) { throw CaptureWriterError.zipFailed(e.localizedDescription) }
        return dest
    }
}

struct CaptureResult: Equatable {
    let directory: URL
    let zipURL: URL
    let frameCount: Int

    var sizeDescription: String {
        let bytes = (try? zipURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
