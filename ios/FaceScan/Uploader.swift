import Foundation

/// Uploads a capture zip to the FaceScan web app.
final class Uploader: NSObject, ObservableObject {
    @Published private(set) var progress: Double = 0
    @Published private(set) var isUploading = false

    private var observation: NSKeyValueObservation?
    private var task: URLSessionUploadTask?

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 30 * 60
        config.waitsForConnectivity = false
        return URLSession(configuration: config)
    }()

    func upload(_ zip: URL, to target: UploadTarget, completion: @escaping (Result<Void, Error>) -> Void) {
        var request = URLRequest(url: target.uploadURL)
        request.httpMethod = "POST"
        request.setValue("application/zip", forHTTPHeaderField: "Content-Type")
        progress = 0
        isUploading = true

        let task = session.uploadTask(with: request, fromFile: zip) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.isUploading = false
                self?.observation = nil
                if let error {
                    completion(.failure(UploadError.network(error.localizedDescription)))
                    return
                }
                guard let http = response as? HTTPURLResponse else {
                    completion(.failure(UploadError.server("No response from the computer.")))
                    return
                }
                if (200..<300).contains(http.statusCode) {
                    self?.progress = 1
                    completion(.success(()))
                } else {
                    completion(.failure(UploadError.server(Uploader.message(from: data) ?? "Upload failed (HTTP \(http.statusCode)).")))
                }
            }
        }
        observation = task.progress.observe(\.fractionCompleted) { [weak self] p, _ in
            DispatchQueue.main.async { self?.progress = p.fractionCompleted }
        }
        self.task = task
        task.resume()
    }

    func cancel() {
        task?.cancel()
    }

    private static func message(from data: Data?) -> String? {
        guard let data, let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj["detail"] as? String
    }
}

enum UploadError: LocalizedError {
    case network(String), server(String)
    var errorDescription: String? {
        switch self {
        case .network(let m):
            return "Could not reach the computer: \(m). Check that both are on the same Wi-Fi and the FaceScan web app is running."
        case .server(let m):
            return m
        }
    }
}
