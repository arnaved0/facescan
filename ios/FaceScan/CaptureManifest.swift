import Foundation

/// capture.json, encoded with snake_case keys. Format spec: docs/capture-format.md.
struct CaptureManifest: Codable {
    var format = "facescan-capture"
    var version = 1
    var appVersion: String
    var deviceModel: String
    var createdAt: String
    var depth: ImageInfo
    var color: ImageInfo
    var segmentation: ImageInfo?
    var frames: [FrameRecord]

    struct ImageInfo: Codable, Equatable {
        var encoding: String
        var width: Int
        var height: Int
        var unitsPerMeter: Int?
    }

    struct Intrinsics: Codable, Equatable {
        var fx: Float
        var fy: Float
        var cx: Float
        var cy: Float
    }

    struct HeadPose: Codable {
        /// Head origin and the tips of its +x, +y, +z axes, projected into the
        /// colour image: [u, v, depth in metres].
        var points: [[Float]]
        var axisLength: Float
    }

    struct FrameRecord: Codable {
        var index: Int
        var timestamp: Double
        var depthFile: String
        var colorFile: String?
        var segmentationFile: String?
        var depthIntrinsics: Intrinsics
        var colorIntrinsics: Intrinsics
        var intrinsicsSource: String
        var headPose: HeadPose?
        var yawDeg: Float
        var pitchDeg: Float
        /// Raw ARKit transforms (column-major), kept for debugging.
        var cameraTransform: [Float]
        var faceTransform: [Float]
    }

    func encoded() throws -> Data {
        let enc = JSONEncoder()
        enc.keyEncodingStrategy = .convertToSnakeCase
        enc.outputFormatting = [.sortedKeys]
        return try enc.encode(self)
    }
}
