import SwiftUI

struct RootView: View {
    @EnvironmentObject private var model: AppModel
    @State private var scanning = false

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 20) {
                Text("Make a 3D-printable model of your face with the Face ID camera.")
                    .font(.title3)

                connectionCard

                VStack(alignment: .leading, spacing: 8) {
                    Label("Good, even light on your face; no strong backlight", systemImage: "sun.max")
                    Label("Tie back hair that covers your forehead or ears", systemImage: "person.crop.circle")
                    Label("Hold the phone still at arm's length and turn only your head", systemImage: "iphone")
                    Label("Keep a relaxed, neutral expression", systemImage: "face.smiling")
                }
                .font(.subheadline)
                .foregroundStyle(.secondary)

                Spacer()

                if ScanController.isSupported {
                    Button {
                        model.resetUpload()
                        scanning = true
                    } label: {
                        Text("Start scan").frame(maxWidth: .infinity).padding(.vertical, 6)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                } else {
                    Text("This device has no TrueDepth (Face ID) camera, which FaceScan needs.")
                        .foregroundStyle(.red)
                }
            }
            .padding()
            .navigationTitle("FaceScan")
        }
        .fullScreenCover(isPresented: $scanning) {
            ScanView(isPresented: $scanning)
                .environmentObject(model)
        }
    }

    @ViewBuilder private var connectionCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let target = model.target {
                Label("Linked to \(target.hostDescription)", systemImage: "desktopcomputer")
                    .font(.headline)
                Text("The scan will upload to this computer when you finish.")
                    .font(.subheadline).foregroundStyle(.secondary)
            } else {
                Label("No computer linked", systemImage: "desktopcomputer.trianglebadge.exclamationmark")
                    .font(.headline)
                Text("Open the FaceScan web app on your computer, click New scan and scan its QR code with the Camera app. You can also scan now and share the file later.")
                    .font(.subheadline).foregroundStyle(.secondary)
            }
            if let err = model.linkError {
                Text(err).font(.footnote).foregroundStyle(.red)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}
