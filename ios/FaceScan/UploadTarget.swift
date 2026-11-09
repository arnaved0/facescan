import Foundation

/// Where a finished capture is uploaded. Comes from the QR code shown by the
/// FaceScan web app, either as an App Clip / universal link
/// (`https://host/c/<token>`) or the custom scheme
/// (`facescan://scan?upload=<url-encoded upload URL>`).
struct UploadTarget: Equatable {
    let uploadURL: URL

    var hostDescription: String {
        if let host = uploadURL.host {
            if let port = uploadURL.port { return "\(host):\(port)" }
            return host
        }
        return uploadURL.absoluteString
    }

    init?(uploadURL: URL) {
        guard let scheme = uploadURL.scheme?.lowercased(), scheme == "http" || scheme == "https",
              uploadURL.host != nil,
              uploadURL.path.hasPrefix("/api/upload/") else { return nil }
        let token = String(uploadURL.path.dropFirst("/api/upload/".count))
        guard UploadTarget.isValidToken(token) else { return nil }
        self.uploadURL = uploadURL
    }

    /// Parses any link the app can be opened with.
    init?(link: URL) {
        let scheme = link.scheme?.lowercased()
        if scheme == "facescan" {
            guard let comps = URLComponents(url: link, resolvingAgainstBaseURL: false),
                  let value = comps.queryItems?.first(where: { $0.name == "upload" })?.value,
                  let url = URL(string: value) else { return nil }
            self.init(uploadURL: url)
            return
        }
        guard scheme == "http" || scheme == "https", let host = link.host else { return nil }
        let parts = link.path.split(separator: "/")
        guard parts.count == 2, parts[0] == "c", UploadTarget.isValidToken(String(parts[1])) else { return nil }
        var comps = URLComponents()
        comps.scheme = scheme
        comps.host = host
        comps.port = link.port
        comps.path = "/api/upload/\(parts[1])"
        guard let url = comps.url else { return nil }
        self.init(uploadURL: url)
    }

    static func isValidToken(_ token: String) -> Bool {
        guard (20...64).contains(token.count) else { return false }
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        return token.unicodeScalars.allSatisfy { allowed.contains($0) }
    }
}
