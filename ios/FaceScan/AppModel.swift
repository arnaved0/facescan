import Foundation

/// App-wide state: where to upload, and what happened to the last capture.
final class AppModel: ObservableObject {
    enum UploadState: Equatable {
        case idle, uploading, uploaded, failed(String)
    }

    @Published var target: UploadTarget?
    @Published var linkError: String?
    @Published private(set) var uploadState: UploadState = .idle
    let uploader = Uploader()

    func handle(link: URL) {
        if let t = UploadTarget(link: link) {
            target = t
            linkError = nil
            uploadState = .idle
        } else {
            linkError = "That link is not a FaceScan scan link. Create a new scan in the web app and scan its QR code."
        }
    }

    func upload(_ result: CaptureResult) {
        guard let target else { return }
        uploadState = .uploading
        uploader.upload(result.zipURL, to: target) { [weak self] outcome in
            switch outcome {
            case .success:
                self?.uploadState = .uploaded
                self?.target = nil  // links are single use
            case .failure(let error):
                self?.uploadState = .failed(error.localizedDescription)
            }
        }
    }

    func resetUpload() {
        uploadState = .idle
    }
}
