import SwiftUI

@main
struct FaceScanApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(model)
                // facescan://scan?upload=... from the web app's phone page
                .onOpenURL { model.handle(link: $0) }
                // https://host/c/<token> via App Clip invocation or universal link
                .onContinueUserActivity(NSUserActivityTypeBrowsingWeb) { activity in
                    if let url = activity.webpageURL { model.handle(link: url) }
                }
        }
    }
}
